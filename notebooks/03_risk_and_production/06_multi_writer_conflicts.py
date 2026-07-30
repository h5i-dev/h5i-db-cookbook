# %% [markdown]
# # Multi-writer coordination: optimistic locking, conflicts, and retries
#
# An h5i-db database is a directory, and nothing stops two processes from
# opening it at the same time: a feed handler and a corrections job, or two
# teammates' notebooks.
#
# The answer to concurrent writers is *optimistic concurrency*. Every commit can
# carry an `expected_version`, and if the table head has moved since you read
# it, the commit is rejected with an explicit `ConflictError` instead of
# silently interleaving.
#
# That is the right trade for teams. A lost update in a positions or marks table
# is a silent P&L error. A `ConflictError` is a retry.
#
# In this recipe we:
#
# 1. simulate two writers with two `Database` handles on the same path,
# 2. scale up to three threads racing to ingest a chunked feed,
# 3. hit the same conflict machinery through the plan/apply mutation flow.

# %% [markdown]
# ## Terms used here
#
# | term                   | meaning |
# | ---------------------- | --- |
# | optimistic concurrency | writers do not lock; a commit is rejected if the head moved under it |
# | `expected_version`     | the version a commit believes it is extending |
# | `ConflictError`        | the rejection raised when it was not; a retry, not a lost update |
# | lost update            | two writers interleaving so one silently overwrites the other's rows |
# | retry                  | re-reading the new head and reapplying the write against it |
# | plan / apply           | stage a delete or replace, review it, then commit it; conflicts apply here too |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import threading

import pyarrow as pa

import h5i_db
import cookbook_utils as cu

path = cu.fresh_db("prod_writers")
writer_a = h5i_db.Database(path, create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.make_trades` gives one session of ticks for three names, our "feed". One
# row per print.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | trade timestamp, ascending |
# | `symbol` | `string` | ticker |
# | `price` | `float64` | trade price |
# | `size` | `int64` | shares traded |
# | `exchange` | `string` | reporting venue |
# | `side` | `string` | `B` buyer-initiated, `S` seller-initiated |
#
# Sections below carve it into time-contiguous chunks. The table is sorted by
# `ts`, so slices are windows.

# %%
feed = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, trades_per_day=8_000, seed=7)
print(f"feed: {feed.num_rows:,} rows x {feed.num_columns} columns")
feed.to_pandas().head()

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
writer_a.create_table("trades", schema, time_column="ts", sort_key=["ts", "symbol"])

# %% [markdown]
# ## 2. Two handles, one database
#
# `writer_b` opens the *same* directory. Think of it as a second process.
#
# Commits made through one handle are immediately visible through the other.
# Both handles read the same manifest on disk, and there is no per-handle cache
# of the table head to go stale.

# %%
writer_b = h5i_db.Database(path)  # no create: the db already exists

writer_a.append("trades", feed.slice(0, 3_000), note="A: chunk 0")
head_a = writer_a.versions("trades")[-1]["sequence"]
head_b = writer_b.versions("trades")[-1]["sequence"]
print(f"head seen by A: v{head_a}, by B: v{head_b}")

# %% [markdown]
# ## 3. A stale `expected_version` fails loudly
#
# B reads the head at v1 and plans to append. Before it does, A commits another
# chunk.
#
# B's `append(..., expected_version=1)` is now a compare-and-swap against a head
# that no longer exists, and h5i-db rejects it.
#
# The error is machine-readable: `.code` for dispatch, `.retryable` telling you
# this is safe to retry, `.hint` for the operator.

# %%
head_b_saw = writer_b.versions("trades")[-1]["sequence"]  # B reads: v1

writer_a.append("trades", feed.slice(3_000, 1_500), note="A: chunk 1")  # head -> v2

try:
    writer_b.append("trades", feed.slice(4_500, 1_500), expected_version=head_b_saw)
except h5i_db.ConflictError as e:
    print(f"code      = {e.code}")
    print(f"retryable = {e.retryable}")
    print(f"hint      = {e.hint}")

# %% [markdown]
# Nothing was written. B's rows are not in the table, and the version chain is
# untouched.
#
# Contrast this with "last writer wins" storage such as plain Parquet
# directories or CSV drops. There, B's write would have clobbered or interleaved
# with A's, and nobody would know until reconciliation.

# %%
print(f"rows in table: {len(writer_b.read('trades')):,}  (B's 1,500 rows were NOT committed)")

# %% [markdown]
# ## 4. The retry pattern
#
# Since `retryable=True`, the fix is mechanical. Re-read the head, re-issue the
# append against it, and give up after a few attempts.
#
# This is the same CAS-retry loop you would write against any
# optimistic-concurrency store.

# %%
def append_with_retry(handle, table, data, note=None, max_attempts=5):
    """Append with optimistic locking; retry on conflict."""
    for attempt in range(1, max_attempts + 1):
        head = handle.versions(table)[-1]["sequence"]
        try:
            commit = handle.append(table, data, expected_version=head, note=note)
            return commit, attempt
        except h5i_db.ConflictError:
            continue  # head moved between read and commit - re-read and retry
    raise RuntimeError(f"gave up after {max_attempts} attempts")


commit, attempts = append_with_retry(writer_b, "trades", feed.slice(4_500, 1_500), note="B: chunk 2 (retried)")
print(f"B committed v{commit['sequence']} on attempt {attempts}; rows_total={commit['rows_total']:,}")

# %% [markdown]
# ## 5. Three threads racing on one feed
#
# Now the stress test. The rest of the feed is split into 9 time-contiguous
# chunks, and three writer threads, each with its **own** `Database` handle,
# race to ingest them.
#
# Coordination happens *entirely* through the CAS. Each thread derives which
# chunk is next from the committed head sequence, and appends it with
# `expected_version=head`.
#
# When two threads pick the same chunk, exactly one commit lands. The loser gets
# `ConflictError`, re-reads the head, and moves on to the next chunk. No locks
# and no queue: the version chain is the queue.

# %%
N_CHUNKS = 9
base_rows = 6_000  # rows already committed in sections 1-3
late_rows = 1_500  # held back for section 5
chunk_rows = (len(feed) - base_rows - late_rows) // N_CHUNKS
chunks = [feed.slice(base_rows + i * chunk_rows, chunk_rows) for i in range(N_CHUNKS)]
thread_rows = sum(len(c) for c in chunks)
print(f"{N_CHUNKS} chunks x {chunk_rows:,} rows for the race")

base_seq = writer_a.versions("trades")[-1]["sequence"]
rows_before = len(writer_a.read("trades"))
stats = {}


def feed_worker(name: str) -> None:
    handle = h5i_db.Database(path)  # per-thread handle, like a separate process
    wins = conflicts = 0
    try:
        while True:
            head = handle.versions("trades")[-1]["sequence"]
            next_chunk = head - base_seq  # chunk index is derived from the committed head
            if next_chunk >= N_CHUNKS:
                break
            try:
                handle.append("trades", chunks[next_chunk], expected_version=head,
                              note=f"{name}: chunk {next_chunk}")
                wins += 1
            except h5i_db.ConflictError:
                conflicts += 1  # another writer landed this chunk first - retry
    finally:
        handle.close()
    stats[name] = {"commits": wins, "conflicts": conflicts}


threads = [threading.Thread(target=feed_worker, args=(f"writer-{i}",)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

stats

# %% [markdown]
# Which thread lands which chunk varies run to run. The *outcome* is
# deterministic: every chunk committed exactly once, in order, with a linear
# version chain. Below we verify no rows were lost or duplicated.

# %%
rows_after = len(writer_a.read("trades"))
expected = rows_before + thread_rows
assert rows_after == expected, f"lost updates! {rows_after} != {expected}"

seqs = [v["sequence"] for v in writer_a.versions("trades")]
assert seqs == list(range(len(seqs))), "version chain is not linear"

total_commits = sum(s["commits"] for s in stats.values())
total_conflicts = sum(s["conflicts"] for s in stats.values())
print(f"rows: {rows_before:,} -> {rows_after:,} (all {N_CHUNKS} chunks landed, none lost)")
print(f"commits: {total_commits}, conflicts absorbed by retries: {total_conflicts}")
print(f"version chain: v0..v{seqs[-1]}, strictly linear")

# %%
[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in writer_a.versions("trades")[-4:]
]

# %% [markdown]
# ## 6. Plan/apply hits the same wall
#
# The previewable-mutation flow, `plan_delete_range` then inspect then `apply`,
# is CAS-guarded too. A plan is built against a specific base version, and
# `apply()` refuses if the head has moved since.
#
# Here A plans to delete a window of suspect prints, but B commits fresh data
# before A applies. That is exactly the race you want caught when a feed and an
# ops job share a table.
#
# Range arguments are raw **microseconds**, the `ts` column's unit, and the end
# bound is exclusive.

# %%
ts0 = feed["ts"][0].value  # raw us since epoch
bad_lo, bad_hi = ts0, ts0 + 60_000_000  # first minute of the day

plan = writer_a.plan_delete_range("trades", bad_lo, bad_hi, note="drop suspect open prints")
print("planned:", plan.summary["rows_affected"], "rows to delete",
      f"({plan.summary['rows_before']:,} -> {plan.summary['rows_after']:,})")

# B lands one more chunk while A's plan sits unapplied:
writer_b.append("trades", feed.slice(base_rows + thread_rows, late_rows), note="B: late chunk")

try:
    plan.apply()
except h5i_db.ConflictError as e:
    print(f"\napply failed - code={e.code}, retryable={e.retryable}")
    print(f"hint = {e.hint}")

# %% [markdown]
# The stale plan is dead, but re-planning is cheap. The second plan is built
# against the *new* head, so its preview reflects B's late chunk too.
#
# That is the point of the CAS. You re-decide with current facts, instead of
# blindly mutating a table that changed under you.

# %%
plan.discard()  # drop the stale plan
plan2 = writer_a.plan_delete_range("trades", bad_lo, bad_hi, note="drop suspect open prints (re-planned)")
commit = plan2.apply()
print(f"re-planned and applied as v{commit['sequence']} ({commit['op']}), "
      f"rows_total={commit['rows_total']:,}")

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in writer_a.versions("trades")[-3:]
]

# %% [markdown]
# ## Takeaways
#
# - Multiple `Database` handles on one path are first-class. Commits through one
#   handle are immediately visible to the others, and the version chain stays
#   linear no matter who writes.
# - `expected_version` turns `append` into a compare-and-swap. A stale write
#   raises `ConflictError` carrying `.code`, `.retryable` and `.hint`: an
#   explicit, retryable failure instead of a silent lost update.
# - The retry loop is five lines: re-read head, re-append, bounded attempts.
#   Three racing threads, coordinated through nothing but the CAS, ingested a
#   chunked feed with zero lost rows.
# - `plan.apply()` is guarded by the same mechanism. Plans bind to a base
#   version, and a moved head forces a re-plan, so your preview can never be
#   stale at apply time.
# - Explicit conflict beats last-writer-wins for shared research and production
#   tables. The failure mode is a retry, not a reconciliation break.

# %%
writer_a.close()
writer_b.close()
