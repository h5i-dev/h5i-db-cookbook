# %% [markdown]
# # Time travel and versioning: which version did my backtest see?
#
# Every write to an h5i-db table is an atomic commit that produces a new
# immutable version. That covers `append`, `write`, `delete` and `restore`. Old
# versions are never rewritten, and reading one is an O(1) manifest lookup
# rather than a log replay.
#
# For quant work this is the difference between two sentences. "The backtest ran
# on *some* state of the price file." Versus "the backtest ran on version 7,
# committed at 14:02:31 UTC, and anyone can re-read exactly that."
#
# In this recipe we:
#
# 1. read the anatomy of `versions()`,
# 2. time-travel three ways: by version number, by commit wall-clock time
#    (`as_of`), and by named snapshot, in both Python and SQL,
# 3. take a vendor restatement as a `write()` with an audit note, and diff the
#    two versions in one query,
# 4. use `restore()` as a rollback that *adds* history instead of erasing it.

# %% [markdown]
# ## Terms used here
#
# | term        | meaning |
# | ----------- | --- |
# | commit      | one atomic write, which produces a new version of the table |
# | version     | the state of a table after a commit; versions are immutable and stay readable |
# | manifest    | the small file listing which segments make up a version |
# | time travel | reading a table as it was at an earlier version, time, or named snapshot |
# | as-of read  | time travel addressed by wall-clock commit time rather than version number |
# | snapshot    | a named, checksummed pin of table versions; O(1) because it copies nothing |
# | restatement | a vendor correcting historical data after the fact |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_timetravel"), create=True)

# %% [markdown]
# ## 1. The data
#
# A daily OHLCV panel from `cu.make_daily_prices`: 10 names over 120 sessions,
# one row per symbol per session.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | session close, 20:00 UTC |
# | `symbol` | `string` | ticker, `STK000` … `STK009` |
# | `open`, `high`, `low`, `close` | `float64` | session prices |
# | `volume` | `int64` | shares traded |

# %%
prices = cu.make_daily_prices(symbols=[f"STK{i:03d}" for i in range(10)], days=120)
print(f"{prices.num_rows:,} rows x {prices.num_columns} columns")
prices.to_pandas().head()

# %% [markdown]
# We load it in three 40-day tranches, the way a real price file grows: one
# delivery at a time. `note=` attaches a human-readable label to the commit. It
# shows up in `versions()` and is the cheapest audit trail you will ever build.

# %%
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])

n = len(prices)
db.append("prices", prices.slice(0, n // 3), note="vendor delivery: days 1-40")
db.append("prices", prices.slice(n // 3, n // 3), note="vendor delivery: days 41-80")
db.append("prices", prices.slice(2 * (n // 3)), note="vendor delivery: days 81-120")

# %% [markdown]
# ## 2. Anatomy of `versions()`
#
# Each entry is one commit:
#
# - `sequence` is the version number you pass to `read` and `h5i()`;
# - `op` is what happened: `create`, `append`, `write`, `delete_range`,
#   `replace_range`, `restore` or `compact`;
# - `committed_at_ns` is the wall-clock commit time;
# - `rows`, `bytes` and `segments` describe the table *as of that version*, not
#   the delta.

# %%
hist = pd.DataFrame(db.versions("prices"))
hist["committed_at"] = pd.to_datetime(hist["committed_at_ns"], utc=True)
hist[["sequence", "op", "committed_at", "rows", "segments", "note"]]

# %% [markdown]
# ## 3. Three ways to time-travel
#
# - **By version number.** Exact, and the one to persist in run metadata.
# - **By commit time** (`as_of`, RFC 3339). This answers "what did we know at
#   9am?" by resolving to the last version committed at or before that instant.
# - **In SQL** via the `h5i()` table function, which accepts either form as well
#   as snapshot names (section 6). Old and new states can therefore meet in one
#   query.

# %%
v1_rows = len(db.read("prices", version=1))

# A wall-clock instant just after tranche 2 was committed:
t_after_2 = pd.Timestamp(
    db.versions("prices")[2]["committed_at_ns"] + 1, unit="ns", tz="UTC"
).isoformat()
asof_rows = len(db.read("prices", as_of=t_after_2))

print(f"version 1          : {v1_rows} rows")
print(f"as_of {t_after_2}: {asof_rows} rows")

db.sql(
    f"""
    SELECT 'version 1' AS read_point, count(*) AS rows, max(ts) AS last_day FROM h5i('prices', 1)
    UNION ALL
    SELECT 'as-of tranche 2',          count(*),        max(ts)             FROM h5i('prices', '{t_after_2}')
    UNION ALL
    SELECT 'latest',                    count(*),        max(ts)             FROM prices
    """
).to_pandas()

# %% [markdown]
# ## 4. A restatement, and the version it did not destroy
#
# Start with a toy "backtest": the annualized mean daily return per symbol,
# computed straight off the live table. We record the head version alongside the
# result. That habit is what this whole recipe is arguing for.

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")


def backtest_mean_return(version=None) -> pd.DataFrame:
    """The same study, parameterized by read point rather than by SQL text."""
    return (
        db.table("prices", version=version)
        .with_columns(ret=col("close") / PREV_CLOSE - 1)
        .filter(col("ret").is_not_null())
        .group_by("symbol")
        .agg(ann_mean_ret=col("ret").mean() * 252)
        .sort("symbol")
        .to_pandas()
    )


v_pre = db.versions("prices")[-1]["sequence"]
result_original = backtest_mean_return()
print(f"backtest ran against version {v_pre}")
result_original.head(3)

# %% [markdown]
# Now the vendor restates day-2 closes: a bad closing auction print, corrected
# by +0.25%. The idiomatic move is a `write()` of the full corrected panel with
# a note. `write` replaces the table contents *as a new version*, so the
# pre-restatement table is still there, one integer away.

# %%
df = prices.to_pandas()
day2 = df["ts"].unique()[1]
df.loc[df["ts"] == day2, "close"] *= 1.0025
corrected = pa.Table.from_pandas(df, schema=prices.schema, preserve_index=False)

commit = db.write("prices", corrected, note="vendor restatement: day-2 closes +25bp")
v_post = commit["sequence"]
print(f"restatement committed as version {v_post}")

# %% [markdown]
# What exactly changed? Join the two versions in one statement. There is no
# export step and no second database: two pinned `db.table(...)` reads put both
# states in the same query, and the join aliases them `l` and `r`.

# %%
new, old = db.table("prices", version=v_post), db.table("prices", version=v_pre)

(
    new.join(old, on=["ts", "symbol"])
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        close_old=col("close", relation="r"),
        close_new=col("close", relation="l"),
        delta=col("close", relation="l") - col("close", relation="r"),
    )
    .filter(col("close_new") != col("close_old"))
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# The restatement moves the backtest slightly and silently. Had you overwritten
# a CSV, it would also have moved it irreversibly. Here, re-running against the
# *pinned* version reproduces the original numbers exactly.

# %%
result_after = backtest_mean_return()
result_pinned = backtest_mean_return(version=v_pre)

drifted = (result_after["ann_mean_ret"] - result_original["ann_mean_ret"]).abs().max()
pinned = (result_pinned["ann_mean_ret"] - result_original["ann_mean_ret"]).abs().max()
print(f"max drift, re-run on live head : {drifted:.2e}")
print(f"max drift, re-run on version {v_pre} : {pinned:.2e}  (bit-for-bit)")
assert pinned == 0.0

# %% [markdown]
# ## 5. `restore()`: rollback that adds history
#
# Suppose the restatement turns out to have been applied to the wrong day.
# `restore(version)` makes an old version the new head, as a *new commit*.
#
# Nothing is erased. The bad write stays in the log, attributable and diffable,
# which is exactly what you want when someone asks who changed the price file
# last Tuesday.

# %%
db.restore("prices", v_pre)

hist = pd.DataFrame(db.versions("prices"))
hist[["sequence", "op", "rows", "note"]]

# %%
# The head is byte-identical to the pre-restatement version:
assert db.read("prices").equals(db.read("prices", version=v_pre))
print("head content == version", v_pre)

# %% [markdown]
# ## 6. Named snapshots: read points with a name
#
# Version numbers are precise but anonymous. `snapshot(name)` pins the current
# version of the tables you choose under a name. That name can go in a run
# registry, an email or a compliance report, and it queries directly from SQL.

# %%
db.snapshot("model-run-2026-07-21", tables=["prices"], note="momentum study, run 42")

(
    db.table("prices", snapshot="model-run-2026-07-21")
    .select(
        rows=count_star(),
        first_day=col("ts").min(),
        last_day=col("ts").max(),
    )
    .to_pandas()
)

# %% [markdown]
# ## Takeaways
#
# - `versions()` is the audit trail. Every commit has a sequence number, an op,
#   a wall-clock time and, if you are disciplined about `note=`, a reason. That
#   discipline costs one keyword argument.
# - Three read points, one mental model: `version=` for exactness, `as_of=` for
#   "what did we know at time T", snapshot names for things humans refer to. All
#   three work in `db.read` and in SQL via `h5i()`.
# - Restatements are `write()` plus a note. The correction and the original
#   coexist, and one join shows precisely what changed.
# - `restore()` rolls the head back without destroying the record of the thing
#   being rolled back.
# - The habit that pays for all of it: **record the version number next to every
#   research result.** Re-running pinned is then bit-for-bit reproducible, a
#   pattern recipe `03_risk_and_production/02` develops in full.

# %%
db.close()
