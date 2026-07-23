# %% [markdown]
# # Multi-writer coordination: optimistic locking, conflicts, and retries
#
# An h5i-db database is a directory, and nothing stops two processes — a feed
# handler and a corrections job, or two teammates' notebooks — from opening it
# at the same time. h5i-db's answer to concurrent writers is *optimistic
# concurrency*: every commit can carry an `expected_version`, and if the table
# head has moved since you read it, the commit is rejected with an explicit
# `ConflictError` instead of silently interleaving. That is the right trade
# for teams: a lost update in a positions or marks table is a silent P&L
# error; a `ConflictError` is a retry.
#
# In this recipe we simulate two writers with two `Database` handles on the
# same path, then scale to three threads racing to ingest a chunked feed, and
# finally hit the same conflict machinery through the plan/apply mutation flow.

# %%
import threading

import pyarrow as pa

import h5i_db
import cookbook_utils as cu

path = cu.fresh_db("prod_writers")
writer_a = h5i_db.Database(path, create=True)

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

# One day of ticks — our "feed". Sections below carve it into
# time-contiguous chunks (the table is sorted by ts, so slices are windows).
feed = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, trades_per_day=8_000, seed=7)
print(f"feed: {len(feed):,} rows")

# %% [markdown]
# ## 1. Two handles, one database
#
# `writer_b` opens the *same* directory — think of it as a second process.
# Commits made through one handle are immediately visible through the other:
# both handles read the same manifest on disk, there is no per-handle cache of
# the table head to go stale.

# %%
writer_b = h5i_db.Database(path)  # no create: the db already exists

writer_a.append("trades", feed.slice(0, 3_000), note="A: chunk 0")
head_a = writer_a.versions("trades")[-1]["sequence"]
head_b = writer_b.versions("trades")[-1]["sequence"]
print(f"head seen by A: v{head_a}, by B: v{head_b}")

# %% [markdown]
# ## 2. A stale `expected_version` fails loudly
#
# B reads the head (v1), plans to append — but before it does, A commits
# another chunk. B's `append(..., expected_version=1)` is now a compare-and-
# swap against a head that no longer exists, and h5i-db rejects it. Note the
# error is machine-readable: `.code` for dispatch, `.retryable` telling you
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
# Nothing was written — B's rows are not in the table, and the version chain
# is untouched. Contrast this with "last writer wins" storage (plain Parquet
# directories, CSV drops), where B's write would have clobbered or interleaved
# with A's and nobody would know until reconciliation.

# %%
print(f"rows in table: {len(writer_b.read('trades')):,}  (B's 1,500 rows were NOT committed)")

# %% [markdown]
# ## 3. The retry pattern
#
# Since `retryable=True`, the fix is mechanical: re-read the head, re-issue
# the append against it, give up after a few attempts. This is the same
# CAS-retry loop you would write against any optimistic-concurrency store.

# %%
def append_with_retry(handle, table, data, note=None, max_attempts=5):
    """Append with optimistic locking; retry on conflict."""
    for attempt in range(1, max_attempts + 1):
        head = handle.versions(table)[-1]["sequence"]
        try:
            commit = handle.append(table, data, expected_version=head, note=note)
            return commit, attempt
        except h5i_db.ConflictError:
            continue  # head moved between read and commit — re-read and retry
    raise RuntimeError(f"gave up after {max_attempts} attempts")


commit, attempts = append_with_retry(writer_b, "trades", feed.slice(4_500, 1_500), note="B: chunk 2 (retried)")
print(f"B committed v{commit['sequence']} on attempt {attempts}; rows_total={commit['rows_total']:,}")

# %% [markdown]
# ## 4. Three threads racing on one feed
#
# Now the stress test: the rest of the feed is split into 9 time-contiguous
# chunks, and three writer threads — each with its **own** `Database` handle —
# race to ingest them. Coordination is done *entirely* through the CAS: each
# thread derives "which chunk is next" from the committed head sequence and
# appends it with `expected_version=head`. When two threads pick the same
# chunk, exactly one commit lands; the loser gets `ConflictError`, re-reads
# the head, and moves on to the next chunk. No locks, no queue — the version
# chain is the queue.

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
                conflicts += 1  # another writer landed this chunk first — retry
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
# Which thread lands which chunk varies run to run — but the *outcome* is
# deterministic: every chunk committed exactly once, in order, and the version
# chain is linear. Verify no rows were lost or duplicated:

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
# ## 5. Plan/apply hits the same wall
#
# The previewable-mutation flow (`plan_delete_range` → inspect → `apply`) is
# CAS-guarded too: a plan is built against a specific base version, and
# `apply()` refuses if the head has moved since. Here A plans to delete a
# window of suspect prints, but B commits fresh data before A applies —
# exactly the race you want caught when a feed and an ops job share a table.
#
# Range arguments are raw **microseconds** (the `ts` column's unit), and the
# end bound is exclusive.

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
    print(f"\napply failed — code={e.code}, retryable={e.retryable}")
    print(f"hint = {e.hint}")

# %% [markdown]
# The stale plan is dead — but re-planning is cheap, and the second plan is
# built against the *new* head, so its preview reflects B's late chunk too.
# That is the point of the CAS: you re-decide with current facts, instead of
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
# - Multiple `Database` handles on one path are first-class — commits through
#   one handle are immediately visible to the others, and the version chain
#   stays linear no matter who writes.
# - `expected_version` turns `append` into a compare-and-swap. A stale write
#   raises `ConflictError` with `.code` / `.retryable` / `.hint` — an explicit,
#   retryable failure instead of a silent lost update.
# - The retry loop is five lines: re-read head, re-append, bounded attempts.
#   Three racing threads coordinated through nothing but the CAS ingested a
#   chunked feed with zero lost rows.
# - `plan.apply()` is guarded by the same mechanism: plans bind to a base
#   version, and a moved head forces a re-plan — your preview can never be
#   stale at apply time.
# - Explicit conflict beats last-writer-wins for shared research/production
#   tables: the failure mode is a retry, not a reconciliation break.

# %%
writer_a.close()
writer_b.close()
