# %% [markdown]
# # Point-in-time fundamentals: killing lookahead bias with ASOF joins
#
# Fundamentals have two timestamps. `period_end` is the fiscal date the numbers
# describe. The report date, typically 25 to 55 days later, is when the market
# actually learned them.
#
# Index your database by the wrong one and every backtest silently trades on
# numbers that did not exist yet. It is the most common and the most flattering
# bug in equity research.
#
# In this recipe we:
#
# 1. store fundamentals keyed by *report time*,
# 2. attach "latest reported EPS" to a daily price panel with `asof_join`,
# 3. quantify how much a `period_end`-keyed join inflates a simple earnings
#    signal,
# 4. use version pinning to keep a study reproducible after a restatement lands.

# %% [markdown]
# ## Terms used here
#
# | term           | meaning |
# | -------------- | --- |
# | fundamentals   | accounting figures such as earnings and book value, reported quarterly |
# | `period_end`   | the fiscal date the numbers describe |
# | report date    | the date the market actually learned them, typically 25 to 55 days later |
# | reporting lag  | the gap between those two dates |
# | point-in-time  | data stored and queried as it was known at each moment |
# | lookahead bias | using information the strategy could not have had; the most flattering bug there is |
# | EPS            | earnings per share |
# | ASOF join      | join each left row to the most recent right row at or before its timestamp |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_pit"), create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.make_fundamentals` generates quarterly EPS with realistic reporting lags.
# The two timestamps are the point of the whole table.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | **report** time: when the numbers went public |
# | `period_end` | `timestamp[us, tz=UTC]` | fiscal quarter end the numbers describe |
# | `symbol` | `string` | ticker |
# | `eps` | `float64` | reported earnings per share |
# | `revenue_m`, `book_value_m` | `float64` | revenue and book value, USD millions |

# %%
SYMS = [f"STK{i:03d}" for i in range(30)]

funda = cu.make_fundamentals(symbols=SYMS, quarters=10).to_pandas()
print(f"{len(funda):,} reports x {funda.shape[1]} columns, "
      f"{funda['ts'].min():%Y-%m-%d} .. {funda['ts'].max():%Y-%m-%d}")
funda.head()

# %% [markdown]
# The price side is a daily OHLCV panel from `cu.make_daily_prices`: `ts`,
# `symbol`, `open`, `high`, `low`, `close`, `volume`, one row per symbol per
# session.

# %%
daily = cu.make_daily_prices(symbols=SYMS, days=700).to_pandas()
print(f"{len(daily):,} daily rows, {daily['ts'].min():%Y-%m-%d} .. {daily['ts'].max():%Y-%m-%d}")
daily.head()

# %% [markdown]
# The two generators are independent, so out of the box no EPS signal predicts
# anything. To make lookahead bias *measurable* we graft in the one mechanism
# that matters: on the first session after each report, the stock jumps 4%
# permanently in the direction of the EPS surprise.
#
# That is the textbook announcement reaction, and the honest way to demo the
# bias. Information is priced in *when it is released*, not when the quarter
# ended.

# %%
funda = funda.sort_values(["symbol", "ts"]).reset_index(drop=True)
funda["eps_growth"] = funda.groupby("symbol")["eps"].pct_change().round(4)

JUMP = 0.04
for r in funda.dropna(subset=["eps_growth"]).itertuples():
    direction = float(np.sign(r.eps_growth))
    if direction == 0.0:
        continue
    mask = (daily["symbol"] == r.symbol) & (daily["ts"] >= r.ts)
    daily.loc[mask, ["open", "high", "low", "close"]] *= 1 + JUMP * direction

print(f"{len(funda)} reports for {len(SYMS)} symbols, "
      f"{funda['ts'].min():%Y-%m-%d} .. {funda['ts'].max():%Y-%m-%d}")
print(f"{len(daily)} daily rows, {daily['ts'].min():%Y-%m-%d} .. {daily['ts'].max():%Y-%m-%d}")

# %% [markdown]
# The gap between the two timestamps is the whole problem. For one symbol, each
# quarter's numbers sit in limbo for the length of the grey bar. A
# `period_end`-keyed join hands your backtest the numbers at the *left* end of
# the bar. The market only saw them at the *right* end.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
f0 = funda[funda["symbol"] == "STK000"].reset_index(drop=True)
for i, r in f0.iterrows():
    axes[0].hlines(i, r["period_end"], r["ts"], color="grey", lw=4, alpha=0.6)
    axes[0].plot(r["period_end"], i, "o", color="tab:blue")
    axes[0].plot(r["ts"], i, "o", color="tab:red")
axes[0].set_title("STK000: period_end (blue) vs report date (red)")
axes[0].set_xlabel("date")
axes[0].set_ylabel("fiscal quarter #")

lags = (funda["ts"] - funda["period_end"]).dt.days
axes[1].hist(lags, bins=15, color="grey", edgecolor="white")
axes[1].set_title(f"Reporting lag, all reports (mean {lags.mean():.0f} days)")
axes[1].set_xlabel("days from period_end to report")
axes[1].set_ylabel("count")
fig.tight_layout()

# %% [markdown]
# ## 2. Store both tables, keyed by the right clock
#
# The fundamentals table's `time_column` is the **report** timestamp, the moment
# each row became public knowledge. `period_end` rides along as an ordinary
# column.
#
# This single schema decision is what makes every downstream ASOF query
# point-in-time by default.

# %%
PRICE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
db.create_table("prices", PRICE_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    pa.Table.from_pandas(
        daily[["ts", "symbol", "close", "volume"]].sort_values(["ts", "symbol"]), preserve_index=False
    ).cast(PRICE_SCHEMA),
    note="daily closes with announcement reactions",
)

FUNDA_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),  # report time
        pa.field("period_end", pa.timestamp("us", tz="UTC")),
        pa.field("symbol", pa.string()),
        pa.field("eps", pa.float64()),
        pa.field("eps_growth", pa.float64()),
    ]
)
db.create_table("fundamentals", FUNDA_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "fundamentals",
    pa.Table.from_pandas(
        funda[["ts", "period_end", "symbol", "eps", "eps_growth"]].sort_values(["ts", "symbol"]),
        preserve_index=False,
    ).cast(FUNDA_SCHEMA),
    note="as-reported quarterly EPS, 10 quarters",
)

# %% [markdown]
# Our study rebalances monthly, so the research panel is the month-end cross
# section: one `time_bucket('1mo', ...)` rollup, materialized as its own table.
#
# Deriving compact, purpose-built tables from the canonical daily store is the
# usual shape of this workflow. The join then runs on about 990 month-ends
# against 300 reports instead of the full daily panel.
#
# Whatever the sizes, assert one output row per left row after any ASOF join, as
# below. It turns silent join mistakes into loud ones.

# %%
monthly = (
    db.table("prices")
    .group_by(time_bucket("1mo", col("ts")).alias("month"), "symbol")
    .agg(month_end=col("ts").max(), close=col("close").last("ts"))
    .select(col("month_end").alias("ts"), "symbol", "close")
    .sort(["ts", "symbol"])
    .to_arrow()
)

MONTHLY_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("prices_m", MONTHLY_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices_m", monthly.cast(MONTHLY_SCHEMA), note="month-end closes derived from prices")
N_MONTHLY = monthly.num_rows
print(f"{N_MONTHLY} month-end observations")

# %% [markdown]
# ## 3. The point-in-time join
#
# `asof_join('prices_m', 'fundamentals', 'ts', 'ts', 'symbol')` attaches to each
# month-end bar the latest fundamentals row whose *report time* is at or before
# that bar. Per symbol, streaming on sorted storage, with no window gymnastics.
#
# Colliding right-side columns get a `_right` suffix, so the report timestamp
# survives as `ts_right`. Every row carries its own "known since" provenance.

# %%
pit = (
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), on="ts", by="symbol")
    .select("ts", "symbol", "close", "eps", "eps_growth", "period_end",
            known_since=col("ts_right"))
    .sort(["ts", "symbol"])
    .to_pandas()
)
assert len(pit) == N_MONTHLY, f"ASOF returned {len(pit)} rows for {N_MONTHLY} left rows"

# spot-check the ASOF result against an independent point query
probe = pit[pit["symbol"] == "STK005"].iloc[12]
point = (
    db.table("fundamentals")
    .filter(col("symbol") == "STK005", col("ts") <= probe["ts"].isoformat())
    .sort("ts", descending=True)
    .limit(1)
    .select("eps")
    .to_pandas()["eps"][0]
)
assert point == probe["eps"]

pit[pit["symbol"] == "STK000"].iloc[3:9]

# %% [markdown]
# A `tolerance`, in raw microseconds, caps staleness. With a 120-day limit,
# month-ends whose latest report is older than roughly a quarter and a half get
# NULLs instead of a zombie EPS. That is useful for flagging delisted names and
# late filers.

# %%
TOL_US = 120 * 86_400 * 1_000_000

(
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), on="ts", by="symbol", tolerance=TOL_US)
    .select(
        month_ends=count_star(),
        no_fresh_report=when(col("eps").is_null()).then(lit(1)).otherwise(lit(0)).sum(),
    )
    .to_pandas()
)

# %% [markdown]
# ## 4. The wrong join, and what it costs
#
# The classic mistake is joining on `period_end`, as if the numbers were known
# the night the quarter closed. `asof_join` takes any right-side time column, so
# the buggy version is one argument away. Our reporting lags keep `period_end`
# monotone in report order, which the join requires.
#
# First, the mechanical size of the leak.

# %%
ahead = (
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), left_on="ts", right_on="period_end", by="symbol")
    .select("ts", "symbol", "close", eps_ahead=col("eps"), growth_ahead=col("eps_growth"))
    .sort(["ts", "symbol"])
    .to_pandas()
)
assert len(ahead) == N_MONTHLY

panel = pit.merge(ahead[["ts", "symbol", "eps_ahead", "growth_ahead"]], on=["ts", "symbol"])
both = panel.dropna(subset=["eps", "eps_ahead"])
leak = (both["eps"] != both["eps_ahead"]).mean()
print(f"{leak:.1%} of month-end observations see a *different* EPS under the period_end join")
print(f"on those, the join is early by up to the reporting lag (mean {lags.mean():.0f} days)")

# %% [markdown]
# Now the damage in signal terms.
#
# Take the simplest earnings signal, the sign of QoQ EPS growth, and relate it
# to the *forward* 21-session return. We measure that return relative to the
# cross-sectional mean, so the common factor and its handful of effective
# observations do not drown the comparison.
#
# The point-in-time signal knows nothing, because the announcement pop is
# already in the price by the time the signal exists. The lookahead signal
# "predicts" the pop it peeked at.

# %%
fut = daily.sort_values(["symbol", "ts"]).copy()
fut["fwd21"] = fut.groupby("symbol")["close"].transform(lambda s: s.shift(-21) / s - 1)
panel = panel.merge(fut[["ts", "symbol", "fwd21"]], on=["ts", "symbol"], how="left")
panel["fwd21_rel"] = panel["fwd21"] - panel.groupby("ts")["fwd21"].transform("mean")

rows = []
for label, field in [("point-in-time", "eps_growth"), ("lookahead", "growth_ahead")]:
    d = panel.dropna(subset=[field, "fwd21_rel"])
    d = d[d[field] != 0]
    sig = np.sign(d[field])
    up, dn = d.loc[sig > 0, "fwd21_rel"].mean(), d.loc[sig < 0, "fwd21_rel"].mean()
    rows.append(
        {
            "join": label,
            "corr(sign, fwd ret)": round(float(np.corrcoef(sig, d["fwd21_rel"])[0, 1]), 3),
            "fwd_ret_eps_up_pct": round(100 * up, 2),
            "fwd_ret_eps_down_pct": round(100 * dn, 2),
            "spread_pct": round(100 * (up - dn), 2),
        }
    )
bias = pd.DataFrame(rows).set_index("join")
bias

# %%
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
ax.bar(x - 0.18, bias["fwd_ret_eps_up_pct"], 0.36, label="EPS growth > 0", color="tab:green")
ax.bar(x + 0.18, bias["fwd_ret_eps_down_pct"], 0.36, label="EPS growth < 0", color="tab:red")
ax.set_xticks(x, bias.index)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Market-relative forward 21-session return - the lookahead mirage")
ax.set_xlabel("how fundamentals were joined")
ax.set_ylabel("mean forward return (%)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The entire "alpha" of the lookahead variant is the announcement reaction it
# saw before the market did. Same data, same signal, one wrong timestamp, and a
# spread appears out of nowhere. On real data this is exactly how
# too-good-to-be-true fundamental backtests are born.
#
# ## 5. Restatements: reproducing the original study
#
# Fundamentals get restated. In an append-keyed PIT table a restatement is a
# *new row at its own report time*, so history is never edited and the old view
# remains queryable.
#
# Suppose STK003's latest EPS is revised up 35% after an audit adjustment.

# %%
v_study = db.versions("fundamentals")[-1]["sequence"]  # pin: the version our study used

last = funda[funda["symbol"] == "STK003"].iloc[-1]
restated = pa.table(
    {
        "ts": pa.array([funda["ts"].max() + pd.Timedelta(days=3)], pa.timestamp("us", tz="UTC")),
        "period_end": pa.array([last["period_end"]], pa.timestamp("us", tz="UTC")),
        "symbol": ["STK003"],
        "eps": [round(last["eps"] * 1.35, 2)],
        "eps_growth": [None],
    }
).cast(FUNDA_SCHEMA)
db.append("fundamentals", restated, note="STK003 EPS restated +35% (audit adjustment)")

def latest_report(label: str, version=None):
    return (
        db.table("fundamentals", version=version)
        .filter(col("symbol") == "STK003")
        .sort("ts", descending=True)
        .limit(1)
        .select(lit(label).alias("view"), "eps", known_since=col("ts"))
        .to_pandas()
    )


latest_report("head (post-restatement)")

# %%
latest_report(f"pinned v{v_study} (as studied)", version=v_study)

# %% [markdown]
# Recording `v_study`, or a named snapshot, alongside a research run means the
# study re-executes against byte-identical inputs forever. The restatement lands
# for live use without rewriting your paper trail.
#
# One caveat: `asof_join` operates on table *heads*. To rebuild a full pinned
# panel, read the pinned version with `db.read("fundamentals", version=v_study)`
# or `h5i(...)` in SQL, and join against that.

# %% [markdown]
# ## Takeaways
#
# - Key fundamentals by **report time** and keep `period_end` as payload. That
#   one schema decision makes every `asof_join` point-in-time by default.
# - `asof_join(prices, fundamentals, 'ts', 'ts', 'symbol')` is the whole PIT
#   machinery: per-symbol latest-known-value, with `ts_right` as built-in
#   "known since" provenance and `tolerance` to refuse stale numbers.
# - Joining on `period_end` handed a do-nothing signal a fat announcement-pop
#   spread. That is lookahead bias measured, not merely asserted.
# - Restatements are appends, not edits. Pinning a version with `h5i('t', v)` or
#   `read(version=)` reproduces the original study exactly while the head serves
#   the corrected view.

# %%
db.close()
