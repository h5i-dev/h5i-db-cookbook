# %% [markdown]
# # Streaming appends and `tail()`: a live feed on a versioned store
#
# An h5i-db table with an append-only history doubles as a message log:
# every `append` is one commit, commits are strictly ordered, and a reader
# that remembers the last version it processed can fetch exactly the rows
# added since - no timestamps-as-cursors guesswork, no missed or double
# rows. That turns "intraday ticks land in the research database with a
# one-day lag" into "the research database *is* the feed consumer". In this
# recipe we:
#
# 1. simulate a live tick feed as a writer appending in chunks,
# 2. consume it with SQL `tail('trades', after_version, poll_ms)` - and see
#    why the `LIMIT` is not optional,
# 3. show the simpler high-water-mark polling pattern used in production,
# 4. maintain 1-minute bars **incrementally**, recomputing only the buckets
#    the newest chunk touched,
# 5. break the pure-append chain on purpose and read the error.

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_streaming"), create=True)

# %% [markdown]
# ## 1. One trading day, chopped into feed chunks
#
# `tail` works on a **pure-append version chain**: any `write`, delete,
# replace or restore in the version range breaks the incremental diff (we
# demonstrate this in section 5). So streaming tables should be append-only
# by construction - raw feeds usually are.
#
# We interleave writer and reader sequentially in one process. That is not a
# limitation of the database (handles from different processes coordinate
# through optimistic concurrency - see the multi-writer recipe); it keeps
# this notebook deterministic.

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=15_000, seed=7)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

N_CHUNKS = 6
n = len(trades)
step = n // N_CHUNKS
chunks = [trades.slice(i * step, step if i < N_CHUNKS - 1 else n - (N_CHUNKS - 1) * step)
          for i in range(N_CHUNKS)]
print(f"{n:,} trades -> {N_CHUNKS} feed chunks of ~{step:,} rows")

# %% [markdown]
# ## 2. Writer appends, reader tails
#
# The reader's cursor is a **version number**, not a timestamp.
# `tail('trades', after_version, poll_ms)` streams every row committed after
# `after_version`, polling for new commits every `poll_ms` - which makes it
# an *unbounded* source: without a `LIMIT` the query never finishes, it
# just waits for the next commit. So the reader first peeks at
# `versions()` to learn how many rows are actually available, then tails
# with exactly that `LIMIT`.
#
# We also maintain 1-minute bars incrementally as rows arrive - but only
# for the buckets the new batch touched. The first affected bucket is the
# minute containing the batch's earliest row; everything before that is
# closed and never recomputed. `read(time_start=...)` (raw microseconds)
# prunes the re-read to that tail of the table.

# %%
def consume_new(last_version: int, rows_seen: int) -> pd.DataFrame | None:
    """Fetch exactly the rows committed after `last_version`, or None."""
    head = db.versions("trades")[-1]
    available = head["rows"] - rows_seen
    if available == 0:
        return None
    return db.sql(
        f"SELECT * FROM tail('trades', {last_version}, 25) LIMIT {available}"
    ).to_pandas()


def recompute_bars(bars: dict, batch: pd.DataFrame) -> int:
    """Recompute 1m bars from the first bucket `batch` touches, in place."""
    first_bucket = batch["ts"].min().floor("1min")
    part = db.read("trades", time_start=int(first_bucket.value // 1000)).to_pandas()
    grp = part.groupby([part["ts"].dt.floor("1min").rename("bar"), "symbol"])
    fresh = grp.apply(
        lambda g: pd.Series({
            "high": g["price"].max(),
            "low": g["price"].min(),
            "volume": g["size"].sum(),
            "vwap": (g["price"] * g["size"]).sum() / g["size"].sum(),
        }),
        include_groups=False,
    )
    bars.update(fresh.to_dict("index"))
    return len(fresh)


bars: dict = {}
last_version, rows_seen = db.versions("trades")[-1]["sequence"], 0

for i, chunk in enumerate(chunks):
    db.append("trades", chunk)                      # -- the writer

    batch = consume_new(last_version, rows_seen)    # -- the reader
    last_version = db.versions("trades")[-1]["sequence"]
    rows_seen += len(batch)
    touched = recompute_bars(bars, batch)
    print(f"chunk {i}: consumed {len(batch):>5,} rows "
          f"({batch['ts'].min():%H:%M:%S} -> {batch['ts'].max():%H:%M:%S}), "
          f"recomputed {touched:>3} of {len(bars)} bars")

assert rows_seen == n
print(f"\nreader consumed all {rows_seen:,} rows; cursor at version {last_version}")

# %% [markdown]
# The incremental bars must equal a from-scratch rollup - one SQL
# `time_bucket` over the full table is the ground truth:

# %%
full = db.sql(
    """
    SELECT time_bucket('1m', ts) AS bar, symbol,
           max(price) AS high, min(price) AS low,
           sum(size) AS volume, vwap(price, size) AS vwap
    FROM trades GROUP BY bar, symbol ORDER BY bar, symbol
    """
).to_pandas()

inc = (pd.DataFrame.from_dict(bars, orient="index")
       .rename_axis(["bar", "symbol"]).reset_index().sort_values(["bar", "symbol"])
       .astype({"volume": "int64"}))  # the pd.Series aggregation upcast volume to float
pd.testing.assert_frame_equal(
    full.reset_index(drop=True), inc.reset_index(drop=True), check_like=True
)
print(f"{len(full)} incremental bars match the full recompute exactly")

# %% [markdown]
# ## 3. `tail` is unbounded - respect the LIMIT
#
# Ask for more rows than have been committed and `tail` does exactly what a
# feed consumer should: it *waits* for the next commit. In a notebook that
# means hanging forever, so always pair `tail` with a `LIMIT` sized to what
# you know is available (as above) and a `timeout=` backstop. Here we
# request one row too many, with a 1.5 s deadline:

# %%
try:
    db.sql(
        f"SELECT * FROM tail('trades', 0, 25) LIMIT {n + 1}",
        timeout=1.5,
    )
except h5i_db.TimeoutError as e:
    print(f"{type(e).__name__}: code={e.code!r} retryable={e.retryable}")
    print("tail kept polling for a commit that never came - the timeout, not")
    print("the LIMIT, ended the query.")

# %% [markdown]
# One more streaming caveat: `tail` produces an unbounded stream, so
# pipeline-breaking operators (aggregations, sorts) cannot run directly on
# top of it. Consume rows first, aggregate client-side - exactly what the
# incremental-bar loop above does.
#
# ## 4. The high-water-mark pattern (no `tail` required)
#
# Most production pollers do not need a blocking stream: on a schedule,
# compare the head version to your cursor and range-read only the new rows.
# `read(time_start=...)` prunes on the time column, so re-reading "everything
# after my high-water mark" costs proportionally to the new data, not the
# table size. This is the pattern to reach for first - `tail` adds value
# when you want push-style blocking delivery.

# %%
cursor_version, hwm_us = 2, None  # pretend we last processed version 2

head = db.versions("trades")[-1]["sequence"]
if head > cursor_version:
    hwm_us = int(pa.compute.max(db.read("trades", version=cursor_version)["ts"]).value)
    new_rows = db.read("trades", time_start=hwm_us + 1)
    print(f"cursor v{cursor_version} -> head v{head}: {len(new_rows):,} new rows "
          f"(matches tail-based count: {len(new_rows) == n - 2 * step})")

# %% [markdown]
# ## 5. Breaking the pure-append chain
#
# Apply any non-append commit - here a one-minute `delete_range` - and
# `tail` across that version can no longer compute an incremental diff. The
# error is loud and specific, which is the point: silent gaps are what kill
# feed consumers.

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
db.plan_delete_range("trades", lo, lo + 60_000_000, note="ops correction").apply()

try:
    db.sql("SELECT * FROM tail('trades', 0, 25) LIMIT 10", timeout=5)
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: code={e.code!r}")
    print(str(e))

# %% [markdown]
# The message names the offending version and offers the fallback: full
# scan of the post-mutation state, or tail a version range that excludes
# it. If a table must stay tail-able, protect it with
# `set_policy(direct_delete=False, direct_write=False, ...)` (recipe 06) and
# route corrections to a downstream *cleaned* table instead.
#
# ## Batch-size guidance
#
# Every commit writes a manifest and at least one Parquet segment, so
# commits-per-second - not rows-per-commit - is what you should budget:
#
# - **Batch ticks into commits** (per second, per venue message-block, per
#   1k-10k rows). One row per commit is pathological: thousands of tiny
#   segments slow every scan.
# - Readers see whole commits atomically, so your batch size is also your
#   delivery latency floor. A 1 s writer batch + `poll_ms=25` reader gives
#   ~1 s end-to-end.
# - After a day of small streaming appends, `db.compact("trades")` merges
#   the small segments (recipe 08). Compaction is a version like any other -
#   but note it breaks pure-append `tail` chains across it, so compact at
#   session boundaries, not mid-stream.
#
# ## Takeaways
#
# - Append-only tables double as message logs: the reader's cursor is a
#   **version number**, and `tail('t', after_version, poll_ms)` delivers
#   exactly-once, in-order rows after it.
# - `tail` is unbounded by design - always size the `LIMIT` from
#   `versions()` and set a `timeout=` backstop; aggregate after consuming,
#   not inside the stream.
# - The high-water-mark pattern (`versions()` delta + `read(time_start=hwm)`)
#   is the simpler production default; time-column pruning makes it cheap.
# - Incremental bar maintenance falls out of the same idea: only recompute
#   buckets the new batch touched - verified equal to the full rollup.
# - `tail` requires a pure-append chain and fails loudly (not silently)
#   when a mutation lands in the range: keep streaming tables append-only
#   by policy.

# %%
db.close()
