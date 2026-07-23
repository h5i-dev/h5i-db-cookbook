# %% [markdown]
# # Designing market data schemas
#
# In h5i-db a table is an Arrow schema plus a time column, persisted as
# immutable, time-sorted Parquet segments under versioned manifests. That
# makes schema design the one decision you cannot cheaply revisit: the types
# you pick determine storage size, which SQL operators apply cleanly, and -
# through `time_column` and `sort_key` - how well scans prune and how fast
# ASOF joins and bar rollups run. This recipe designs the three tables every
# equity desk starts with (ticks, quotes, daily bars), explains each type
# choice, and then probes the *strict append contract*: what h5i-db rejects
# at the door, and why that strictness is a feature for a shared research
# database.

# %%
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_schemas"), create=True)

# %% [markdown]
# ## 1. Type choices for tick-level trades
#
# The decisions that matter, column by column:
#
# - **`ts` - `timestamp[us, tz=UTC]`, non-nullable.** Microseconds cover
#   consolidated-feed granularity; tz-aware UTC removes a whole class of
#   DST bugs (convert to exchange time at query time with `time_bucket`'s
#   timezone argument, not at storage time). Make it non-nullable: a print
#   without a timestamp is not data. Every raw-unit argument elsewhere in
#   the API (plan ranges, gapfill steps, ASOF tolerances) is then in
#   microseconds, consistently.
# - **`price` - `float64`.** The classic tradeoff: `decimal128` is exact and
#   the right call in a settlement ledger, but float64 round-trips cent-grid
#   equity prices exactly up to ~$10^13$, is what every analytics function
#   (`vwap`, `ewma`, `stddev`, `corr`) expects, and is twice as compact.
#   At the research layer, float64 is the convention; keep decimals for
#   books and records systems.
# - **`size` - `int64`.** Never float: share counts are integers, and int64
#   headroom means one type for everything from odd lots to index rebalance
#   crosses.
# - **`symbol`, `exchange`, `side` - `utf8`.** Low-cardinality strings are
#   dictionary-encoded inside the Parquet segments automatically, so plain
#   `utf8` at the schema layer costs almost nothing on disk and keeps the
#   schema simple.

# %%
trades_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)

quotes_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
    ]
)

bars_1d_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)

# %% [markdown]
# ## 2. `time_column` and `sort_key`
#
# Declaring `time_column="ts"` is not metadata decoration - it is the
# contract h5i-db builds everything on:
#
# - segments are stored **sorted by `ts`**, and each segment's time range is
#   recorded in the manifest, so a `WHERE ts BETWEEN ...` predicate skips
#   whole segments without opening them (manifest pruning);
# - **ASOF joins** and **`time_bucket` rollups** stream over that sort order
#   instead of re-sorting your ticks on every query;
# - `append` can enforce feed semantics (more on that below) because "later
#   than everything stored" is well defined.
#
# `sort_key=["ts", "symbol"]` adds a secondary order *within* each timestamp
# tie - and, more importantly, groups rows so per-symbol scans touch
# contiguous runs of data. The rule: the sort key must start with the time
# column; put the column you filter by most (almost always `symbol`) next.

# %%
db.create_table("trades", trades_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("quotes", quotes_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("bars_1d", bars_1d_schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %%
trades = cu.make_trades(days=2, trades_per_day=5_000)
quotes = cu.make_quotes(days=2, quotes_per_day=8_000)
daily = cu.make_daily_prices(symbols=["AAPL", "MSFT", "NVDA"], days=250)

for name, data in [("trades", trades), ("quotes", quotes), ("bars_1d", daily)]:
    commit = db.append(name, data, note="initial load")
    print(f"{name:8s} v{commit['sequence']}: {commit['rows_total']:>7,} rows, "
          f"{commit['segments_total']} segment(s)")

db.schema("trades")

# %% [markdown]
# A quick query to confirm the schemas earn their keep - per-symbol quoted
# spread in basis points straight off the quotes table. Plain `utf8` symbol
# columns group and filter exactly as you'd hope; no special "category" type
# needed at this layer.

# %%
db.sql(
    """
    SELECT symbol,
           count(*)                                    AS quotes,
           round(avg((ask - bid) / ((ask + bid) / 2)) * 1e4, 2) AS avg_spread_bps,
           round(min(bid), 2)                          AS min_bid,
           round(max(ask), 2)                          AS max_ask
    FROM quotes
    GROUP BY symbol
    ORDER BY symbol
    """
).to_pandas()

# %% [markdown]
# ## 3. What strict append rejects
#
# `append` has feed semantics: the batch must match the declared schema, be
# sorted by the time column, and start at or after the table's current
# maximum timestamp. Everything else is rejected *before* anything is
# written - the table head never moves on a failed append. In a database
# shared by a team (or written by an agent), this strictness is the point:
# malformed vendor files and out-of-order backfills fail loudly at ingest
# instead of silently corrupting downstream research.
#
# Every h5i-db exception carries a machine-readable `.code`, a `.retryable`
# flag, and often a `.hint` with the recommended fix.
#
# **Rejection 1 - schema mismatch.** A vendor "helpfully" ships prices as
# float32 and forgets the `side` column:

# %%
bad_schema_batch = pa.table(
    {
        "ts": trades["ts"][:100],
        "symbol": trades["symbol"][:100],
        "price": trades["price"][:100].cast(pa.float32()),   # wrong width
        "size": trades["size"][:100],
        "exchange": trades["exchange"][:100],
        # 'side' column missing entirely
    }
)
try:
    db.append("trades", bad_schema_batch)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"message   {e}")
    print(f"hint      {e.hint or '(none - the message says it all)'}")

# %% [markdown]
# **Rejection 2 - unsorted data.** Same rows, shuffled - a classic outcome
# of concatenating per-exchange files without a final sort:

# %%
import numpy as np

rng = np.random.default_rng(0)
shuffled = trades.take(rng.permutation(len(trades)))
try:
    db.append("trades", shuffled)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %% [markdown]
# **Rejection 3 - overlapping time range.** Re-delivering data the table
# already has (the batch's min `ts` is before the stored max) is refused for
# the same reason: `append` means *extend the feed*, never *interleave into
# history*. Deliberate restatements go through `write` or the previewable
# `plan_replace_range` flow (recipes 05 and 06).

# %%
try:
    db.append("trades", trades)  # the exact batch already ingested
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %%
# The head never moved: still one data commit per table.
[{k: v[k] for k in ("sequence", "op", "rows") if k in v} for v in db.versions("trades")]

# %% [markdown]
# ## 4. Schema evolution: know the limits up front
#
# Because segments are immutable, schemas are deliberately hard to change.
# Plan for the evolutions that are cheap and avoid the ones that are not:
#
# - **Cheap:** appending *trailing nullable* columns (old segments read as
#   NULL) and *widening* numerics (int32 → int64, float32 → float64).
# - **Expensive:** renames, type narrowing, reordering, or dropping columns -
#   these mean writing a new table and backfilling.
#
# Practical consequence: start wide enough (`int64`, `float64`,
# `timestamp[us]`), and if a column *might* exist one day (e.g. a
# `condition` code on trades), it costs little to include it as nullable
# from day one.

# %% [markdown]
# ## Takeaways
#
# - A table = Arrow schema + `time_column`; segments are stored time-sorted
#   Parquet, so declaring the time column buys pruning, streaming
#   `time_bucket` rollups and sort-free ASOF joins.
# - Defaults that serve quant work well: `timestamp[us, tz=UTC]`
#   non-nullable for time, `float64` for prices at the research layer,
#   `int64` for sizes, plain `utf8` for symbols (Parquet
#   dictionary-encodes them anyway).
# - `sort_key=["ts", "symbol"]` must start with the time column; the
#   secondary key makes per-symbol scans touch contiguous data.
# - Strict append rejects schema mismatches, unsorted batches and
#   overlapping time ranges *before* writing - errors carry `.code` and
#   `.hint`, and the table head never moves on failure.
# - Evolve schemas by trailing nullable adds and numeric widening; anything
#   else is a rebuild, so start wide.

# %%
db.close()
