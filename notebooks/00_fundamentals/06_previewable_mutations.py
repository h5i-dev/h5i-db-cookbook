# %% [markdown]
# # Previewable mutations: fix bad ticks without fearing the delete key
#
# Deleting or rewriting rows in a shared tick store is the scariest routine
# operation on a quant desk - a fat-fingered `DELETE WHERE` has ruined more
# research datasets than any hardware failure. h5i-db's answer is the
# **plan/apply flow**: a mutation is first *staged* as a plan with a machine
# summary and before/after row samples, reviewed, and only then applied - as
# a new version, with the old one intact. In this recipe we:
#
# 1. inject two classic tick-data pathologies (a feed glitch and a 10x
#    scaling bug) and find them with SQL,
# 2. stage a delete, catch ourselves over-deleting in the preview, discard,
#    re-plan narrower, apply,
# 3. stage a `replace_range` that repairs prices in place,
# 4. lock the database down with a mutation policy so direct destructive
#    writes raise `PolicyError` - the safety story for shared and
#    agent-operated databases.

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_mutations"), create=True)

# %% [markdown]
# ## 1. Two days of ticks, two injected pathologies
#
# We corrupt the synthetic feed the way real feeds break:
#
# - **Feed glitch**: for ten seconds on day 1 (15:00:00-15:00:10 UTC) every
#   print, all symbols, comes in at 10x - a decimal-shift burst.
# - **Scaling bug**: for ten minutes on day 2 (14:00-14:10 UTC) one symbol's
#   (MSFT) prices are all 10x - a per-symbol multiplier bug in a handler.
#
# The glitch rows are garbage (delete them); the scaling-bug rows are real
# trades with a recoverable price (repair them).

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=2, trades_per_day=20_000, seed=7)
df = trades.to_pandas()

glitch = (df["ts"] >= "2026-06-01 15:00:00+00:00") & (df["ts"] < "2026-06-01 15:00:10+00:00")
scaling = (
    (df["ts"] >= "2026-06-02 14:00:00+00:00")
    & (df["ts"] < "2026-06-02 14:10:00+00:00")
    & (df["symbol"] == "MSFT")
)
df.loc[glitch | scaling, "price"] *= 10.0
corrupted = pa.Table.from_pandas(df, schema=trades.schema, preserve_index=False)

db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", corrupted, note="raw feed capture, days 1-2")
print(f"{glitch.sum()} glitch prints, {scaling.sum()} scaled prints, {len(df):,} rows total")

# %% [markdown]
# ## 2. Find the damage with SQL
#
# A robust screen: compare every print to its symbol's daily median
# (`approx_percentile_cont`). The corrupt rows sit at ~10x their median; the
# median itself barely moves because the corrupt fraction is small. This is
# why medians, not means, anchor outlier screens.

# %%
flagged = db.sql(
    """
    WITH med AS (
        SELECT symbol, time_bucket('1d', ts) AS day,
               approx_percentile_cont(price, 0.5) AS med_px
        FROM trades GROUP BY symbol, day
    )
    SELECT t.ts, t.symbol, t.price, t.price / m.med_px AS x_median
    FROM trades t
    JOIN med m ON t.symbol = m.symbol AND time_bucket('1d', t.ts) = m.day
    WHERE t.price > 3 * m.med_px
    ORDER BY t.ts
    """
).to_pandas()

flagged.groupby([flagged["ts"].dt.floor("1D").rename("day"), "symbol"]).size()

# %% [markdown]
# Two clusters, exactly as injected: an all-symbol burst on day 1 and an
# MSFT-only stretch on day 2. Derive the time bounds of each from the
# flagged rows. **Mutation-plan ranges are raw int64 microseconds** (the
# unit of the `ts` column) and half-open `[start, end)` - so add 1µs past
# the last bad print.

# %%
burst = flagged[flagged["ts"] < "2026-06-02"]
bug = flagged[(flagged["ts"] >= "2026-06-02") & (flagged["symbol"] == "MSFT")]

burst_lo = int(burst["ts"].min().value // 1000)
burst_hi = int(burst["ts"].max().value // 1000) + 1
bug_lo = int(bug["ts"].min().value // 1000)
bug_hi = int(bug["ts"].max().value // 1000) + 1
print("burst window:", burst["ts"].min(), "->", burst["ts"].max())
print("bug   window:", bug["ts"].min(), "->", bug["ts"].max())

# %% [markdown]
# ## 3. Stage a delete - and catch the over-delete before it happens
#
# First instinct: "nuke the whole five minutes around the glitch, to be
# safe." Stage it and look at the plan before believing it. A plan is not a
# commit: nothing has changed in the table yet.
#
# One crucial property of range mutations: **the range is on time only** -
# it does not filter by symbol. A "safe, wide" window silently takes healthy
# AAPL and NVDA prints with it. The plan summary makes that visible *before*
# any damage.

# %%
sloppy = db.plan_delete_range(
    "trades",
    int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000),
    int(pd.Timestamp("2026-06-01 15:05:00", tz="UTC").value // 1000),
    note="delete glitch burst (wide window)",
)
sloppy.summary

# %% [markdown]
# `rows_affected` is ~25x the number of bad prints - the wide window would
# destroy hundreds of good trades. `before_sample` shows the doomed rows:
# healthy \\$200-ish prints mixed in with the 10x garbage. Discard and
# re-plan on the tight bounds the detector gave us.

# %%
sloppy.before_sample.to_pandas().head(6)

# %%
sloppy.discard()

tight = db.plan_delete_range("trades", burst_lo, burst_hi, note="delete 10s decimal-shift burst")
print("rows_affected:", tight.summary["rows_affected"], " (bad prints:", len(burst), ")")
commit = tight.apply()
print("applied as version", commit["sequence"], "op:", commit["op"])

# %% [markdown]
# `apply()` re-checks the table head first: if anyone committed after the
# plan was staged, it raises `ConflictError` instead of publishing a
# mutation computed against a stale base - the same optimistic concurrency
# as `append(expected_version=...)`.
#
# ## 4. Repair in place with `plan_replace_range`
#
# The MSFT scaling bug is different: the trades are real, the prices are
# recoverable (`/10`). `plan_replace_range(start, end, data)` swaps
# *everything* in the window for the data you provide - so the replacement
# must contain the window's AAPL and NVDA rows unchanged, plus the repaired
# MSFT rows.

# %%
win = df[(df["ts"] >= pd.Timestamp(bug_lo, unit="us", tz="UTC"))
         & (df["ts"] < pd.Timestamp(bug_hi, unit="us", tz="UTC"))].copy()
win.loc[win["symbol"] == "MSFT", "price"] /= 10.0
repaired = pa.Table.from_pandas(win, schema=trades.schema, preserve_index=False)

fix = db.plan_replace_range("trades", bug_lo, bug_hi, data=repaired,
                            note="MSFT 10x scaling bug: prices /10")
fix.summary

# %%
# after_sample previews the post-mutation rows - MSFT back at sane levels:
fix.after_sample.to_pandas().query("symbol == 'MSFT'").head(4)

# %%
fix.apply()
assert len(db.sql(
    """
    WITH med AS (SELECT symbol, time_bucket('1d', ts) AS day,
                        approx_percentile_cont(price, 0.5) AS med_px
                 FROM trades GROUP BY symbol, day)
    SELECT t.ts FROM trades t
    JOIN med m ON t.symbol = m.symbol AND time_bucket('1d', t.ts) = m.day
    WHERE t.price > 3 * m.med_px
    """
)) == 0
print("detector re-run: 0 outliers remaining")

# %% [markdown]
# Every correction is a version. The uncorrected capture is one integer
# away - so "did the cleaning change my backtest?" is a SQL join, not an
# archaeology project (recipe 05).

# %%
pd.DataFrame(db.versions("trades"))[["sequence", "op", "rows", "note"]]

# %%
import matplotlib.pyplot as plt

before = db.sql(
    """
    SELECT ts, price FROM h5i('trades', 1)
    WHERE symbol = 'MSFT' AND ts >= '2026-06-02T13:50:00Z' AND ts < '2026-06-02T14:20:00Z'
    ORDER BY ts
    """
).to_pandas()
after = db.sql(
    """
    SELECT ts, price FROM trades
    WHERE symbol = 'MSFT' AND ts >= '2026-06-02T13:50:00Z' AND ts < '2026-06-02T14:20:00Z'
    ORDER BY ts
    """
).to_pandas()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(before["ts"], before["price"], lw=0.8, color="tab:red", label="version 1 (raw capture)")
ax.plot(after["ts"], after["price"], lw=0.8, color="tab:blue", label="head (repaired)")
ax.set_yscale("log")
ax.set_title("MSFT prints around the scaling bug: raw capture vs repaired head")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price (log scale)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 5. Mutation policy: make the safe path the only path
#
# On a shared database - or one an automated agent writes to - you want
# direct destructive operations off by default. `set_policy` flips boolean
# gates per operation class; a gated direct call raises `PolicyError`
# **before touching anything**. The plan/apply flow stays available: that
# *is* the sanctioned route, because it forces a previewable, reviewable
# artifact between intent and mutation.

# %%
print("default policy:", db.policy())
db.set_policy(direct_write=False, direct_delete=False, direct_restore=False)

try:
    db.write("trades", corrupted)
except h5i_db.PolicyError as e:
    print(f"\nwrite blocked -> {type(e).__name__}")
    print("  code:", e.code)
    print("  hint:", e.hint)

# %% [markdown]
# Planning is still allowed under the restrictive policy. Pending plans are
# first-class objects: `list_plans` shows what is staged and by which
# summary; plans expire after a 7-day TTL, and their staged segments are
# protected from `vacuum` until they are applied, discarded, or expire.

# %%
pending = db.plan_delete_range(
    "trades",
    int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").value // 1000),
    int(pd.Timestamp("2026-06-01 16:00:30", tz="UTC").value // 1000),
    note="staged under restrictive policy",
)

for p in db.list_plans("trades"):
    ttl_days = (p.raw["expires_at_ns"] - p.raw["created_at_ns"]) / 86_400e9
    print(f"plan {p.plan_id[:8]}…  note={p.raw['note']!r}  "
          f"rows_affected={p.summary['rows_affected']}  ttl={ttl_days:.0f}d")

pending.discard()
db.set_policy(direct_write=True, direct_delete=True, direct_restore=True)

# %% [markdown]
# ## Takeaways
#
# - **Plan, look, then apply.** `plan_delete_range` / `plan_replace_range`
#   stage a mutation with a row-count summary and before/after samples; the
#   over-wide delete announced itself in `rows_affected` before it could do
#   any harm.
# - Mutation ranges are **raw int64 microseconds, half-open, time-only** -
#   they never filter by symbol, so a replacement must carry the innocent
#   rows of the window through unchanged.
# - `apply()` is conflict-checked against the table head; every applied plan
#   is a new version with a note, and the uncorrected data stays one
#   `h5i('trades', v)` away.
# - `set_policy(direct_write=False, ...)` turns "please be careful" into
#   `PolicyError` - the right default for shared stores and for anything an
#   LLM agent is allowed to touch.

# %%
db.close()
