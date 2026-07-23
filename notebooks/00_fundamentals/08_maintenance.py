# %% [markdown]
# # Maintenance: verify, compact, vacuum - keeping a versioned store healthy
#
# A versioned database makes an unusual bargain: it never overwrites data, so
# it accumulates - manifests, segments, history. That is the feature. But it
# means three maintenance questions need honest answers:
#
# 1. **Is the data intact?** `verify()` walks the manifest checksum chain;
#    `verify(deep=True)` re-checksums every stored byte.
# 2. **Why did queries get slower?** Streaming ingestion leaves many small
#    Parquet segments; `compact()` merges them into one - as a new version.
# 3. **Where did my disk go, and what can I get back?** `vacuum()` reclaims
#    *unreferenced* objects only. We test - rather than assume - what that
#    means for old versions.
#
# The dataset is deliberately mistreated: a week of ticks appended in 150
# tiny commits, the way an unbatched feed writer would.

# %%
import time
from pathlib import Path

import pandas as pd

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_maintenance"), create=True)


def du_mb(path: str) -> float:
    """Directory size in MiB - the du -s of this recipe."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20


def bench(fn, repeat: int = 3) -> float:
    """Min-of-N wall time: the standard cheap timing harness."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# %% [markdown]
# ## 1. Simulate an unbatched writer: 150 tiny commits
#
# Every commit writes a manifest and at least one segment. 150 appends of
# ~2,000 rows leave the table with 150 small Parquet files - each one a file
# open, a footer parse, and a merge step for every query that follows.

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

N = 150
n = len(trades)
step = n // N
for i in range(N):
    db.append("trades", trades.slice(i * step, step if i < N - 1 else n - (N - 1) * step))

head = db.versions("trades")[-1]
print(f"{head['rows']:,} rows across {head['segments']} segments, "
      f"{head['bytes'] / 2**20:.1f} MiB of live data, db dir = {du_mb(db.path):.1f} MiB")

# %% [markdown]
# ## 2. `verify()`: the checksum chain
#
# Shallow `verify` checks the table's manifest chain - each manifest commits
# to its predecessor and to every segment it references, so a truncated
# file, a lost commit, or a tampered manifest breaks the chain. It reads
# metadata only (`bytes_checked: 0`), so it is cheap enough to run on every
# open. `deep=True` additionally re-reads and re-checksums every segment
# byte - the periodic bit-rot audit, priced accordingly.

# %%
print("shallow:", db.verify("trades"))
print("deep   :", db.verify("trades", deep=True))

# %% [markdown]
# An empty `problems` list from a deep verify is an attestation you can
# hand to anyone: the bytes on disk are exactly the bytes every commit
# signed for.
#
# ## 3. `compact()`: undo the segment sprawl
#
# Time a representative aggregation against the 150-segment table, compact,
# and time it again. Compaction rewrites the *same rows* into one
# well-sized segment and commits the result as a new version - history is
# untouched, so this is safe to run whenever ingestion leaves debris.

# %%
QUERY = """
    SELECT time_bucket('5m', ts) AS bar, symbol,
           vwap(price, size) AS vwap, sum(size) AS volume
    FROM trades GROUP BY bar, symbol ORDER BY bar, symbol
"""

t_before = bench(lambda: db.sql(QUERY).to_arrow())

commit = db.compact("trades", note="merge 150 streaming appends")
print({k: commit[k] for k in ("sequence", "op", "segments_total", "segments_added")})

t_after = bench(lambda: db.sql(QUERY).to_arrow())
print(f"\n5m-bar rollup: {t_before * 1e3:.1f} ms on 150 segments "
      f"-> {t_after * 1e3:.1f} ms on 1 segment  ({t_before / t_after:.1f}x)")

# %% [markdown]
# Two honest observations. First, the speedup here is real but modest -
# per-segment overhead scales with segment count, so a day of per-tick
# commits (tens of thousands of segments) hurts far more than our 150.
# Second, the directory *grew*: the 150 old segments are still referenced
# by versions 1-150, and the compacted table carries one more copy of every
# row. Version history is a storage bargain, and compaction does not break
# it:

# %%
print(f"db dir after compact: {du_mb(db.path):.1f} MiB")
v75 = db.read("trades", version=75)
print(f"version 75 still readable: {len(v75):,} rows (head has {head['rows']:,})")

# %% [markdown]
# ## 4. `vacuum()`: what it does - and does not - reclaim
#
# The natural fear is that vacuum trades history for disk. Test it instead
# of assuming: a dry run (`apply=False`) with `grace_seconds=0` - the most
# aggressive setting - right after compacting.

# %%
db.vacuum("trades", grace_seconds=0, apply=False)

# %% [markdown]
# **Zero candidates.** Every one of those 151 segments is referenced by some
# version's manifest, and vacuum only ever collects *unreferenced* objects:
# staging files from discarded mutation plans, debris from interrupted
# ingests. In this build, vacuum never prunes version history - every
# version stays readable no matter how aggressively you vacuum. Retention
# is a policy decision it refuses to make implicitly, which is the correct
# default for anything auditors may ask about.
#
# So let's manufacture the garbage vacuum *does* exist for: stage a
# replace plan (which writes its repaired segment to storage immediately)
# and then discard it. The staged segment is now referenced by nothing.

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
hi = int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").value // 1000)
window = db.read("trades", time_start=lo, time_end=hi)

plan = db.plan_replace_range("trades", lo, hi, data=window, note="staged then abandoned")
plan.discard()

print("dry run, default grace (1h):", db.vacuum("trades", apply=False))
print("dry run, grace_seconds=0   :", db.vacuum("trades", grace_seconds=0, apply=False))

# %% [markdown]
# The default one-hour grace period hides the orphan: vacuum will not touch
# young files, because a file that looks unreferenced *right now* might be
# part of a commit another process is seconds away from publishing. Only
# `grace_seconds=0` - safe here, since nothing else is writing - exposes
# the discarded plan's segment as a candidate. Reclaim it for real:

# %%
size_before = du_mb(db.path)
result = db.vacuum("trades", grace_seconds=0, apply=True)
print(result)
print(f"db dir: {size_before:.2f} MiB -> {du_mb(db.path):.2f} MiB")

# %%
# The load-bearing claim, checked: vacuum deleted the orphan and nothing else.
versions = db.versions("trades")
for v in versions:  # every version's manifest still resolves and reads
    db.read("trades", version=v["sequence"], limit=5)
assert db.vacuum("trades", grace_seconds=0, apply=False)["candidates"] == []
print(f"all {len(versions)} versions still readable after vacuum; "
      f"deep verify clean: {db.verify('trades', deep=True)['problems'] == []}")

# %% [markdown]
# ## 5. Snapshots: named, checksummed pins
#
# Since vacuum never eats history here, snapshots are not a defensive
# necessity - their maintenance role is different. `snapshot(name)` records
# the head version of chosen tables *plus the manifest checksum* under a
# durable name: a read point humans can cite ("the EOD cut") and an
# integrity anchor you can later re-verify against. Pending mutation plans
# get the same courtesy: their staged segments are protected from vacuum
# until applied, discarded, or expired (7-day TTL).

# %%
snap = db.snapshot("eod-2026-06-05", tables=["trades"], note="post-compact EOD cut")
entry = next(iter(snap["entries"].values()))
print(f"snapshot {snap['name']!r} pins {entry['table_name']} @ v{entry['sequence']}")
print(f"manifest checksum: {entry['manifest_checksum'][:16]}…")

db.sql("SELECT count(*) AS rows FROM h5i('trades', 'eod-2026-06-05')").to_pandas()

# %% [markdown]
# ## A maintenance cadence that works
#
# - **Every open / hourly**: shallow `verify` - metadata-only, effectively free.
# - **After each ingestion session**: `compact` tables that streamed in as
#   many small commits (recipe 07's batch-size guidance reduces the need).
# - **Daily, off-hours**: `vacuum(apply=False)`, review the candidate list,
#   then `apply=True` with the default grace period. Never pass
#   `grace_seconds=0` while writers may be active.
# - **Weekly / before attestations**: `verify(deep=True)` plus a named
#   snapshot of the tables you would have to defend.
#
# ## Takeaways
#
# - `verify()` proves the manifest checksum chain cheaply; `deep=True` turns
#   it into a full bit-rot audit with an empty `problems` list as the receipt.
# - Small commits are a *query* tax, not just a storage tax: `compact()`
#   merged 150 segments into 1 as an ordinary new version, with measurable
#   speedup and zero history lost.
# - `vacuum()` collects unreferenced objects only - discarded plan staging,
#   interrupted-write debris. We verified that even `grace_seconds=0,
#   apply=True` leaves every historical version readable: history pruning is
#   simply not something vacuum does in this build.
# - The grace period is crash-safety for concurrent writers; zero it only in
#   single-writer maintenance windows.
# - Snapshots are named, checksummed read points - the citable artifact for
#   EOD cuts and attestations, not a vacuum workaround.

# %%
db.close()
