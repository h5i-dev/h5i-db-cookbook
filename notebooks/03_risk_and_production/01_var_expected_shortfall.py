# %% [markdown]
# # VaR and Expected Shortfall with an auditable risk table
#
# A risk number nobody can reproduce is a liability.
#
# In this recipe we:
#
# 1. compute historical and parametric VaR and Expected Shortfall for a fixed
#    equity book on real daily data,
# 2. backtest the VaR with Kupiec's POF test,
# 3. write the daily risk metrics into a versioned table, one commit per
#    production day.
#
# That third step is the part most notebooks skip. It turns "what VaR did we
# report on date X, and from what inputs?" into a query rather than an email
# thread.
#
# Three h5i-db features do the work: SQL window functions for the return
# pipeline, append-only commits as the audit trail, where `versions()` shows
# every EOD run with its note and wall-clock commit time, and `compact()` to
# keep the table healthy after many small daily appends.

# %% [markdown]
# ## Terms used here
#
# | term               | meaning |
# | ------------------ | --- |
# | VaR                | the loss a book will not exceed on a given day at a chosen confidence |
# | Expected Shortfall | the average loss on the days that do breach VaR, which is what VaR ducks |
# | historical VaR     | read straight off the empirical distribution of past returns |
# | parametric VaR     | assume a distribution, usually normal, and read the quantile from it |
# | confidence level   | the 95% or 99% in the VaR statement |
# | exception          | a day whose loss exceeded the forecast; 99% VaR should produce them 1% of the time |
# | Kupiec POF test    | a test of whether the observed exception count matches the confidence claimed |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_var"), create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.fetch_daily` gives eight-plus years of daily data for 30 liquid US names,
# cached from Yahoo Finance. One row per symbol per session.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | session date |
# | `symbol` | `string` | ticker |
# | `open`, `high`, `low`, `close` | `float64` | session prices |
# | `adj_close` | `float64` | close adjusted for splits and dividends |
# | `volume` | `int64` | shares traded |

# %%
real = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
print(f"{real.num_rows:,} rows x {real.num_columns} columns, "
      f"{len(set(real['symbol'].to_pylist()))} symbols")
real.to_pandas().head()

# %% [markdown]
# We store `adj_close`, a roughly total-return series, which is what P&L risk
# should be computed on. One bulk `append` is one atomic commit.
#
# One strictness is worth knowing. With `sort_key=["ts", "symbol"]`, `append`
# requires the input sorted by the *full* key. Daily panels have 30 symbols
# sharing each timestamp, so a ts-only sort is not enough. Sort by
# (ts, symbol) before appending.

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    real.select(["ts", "symbol", "adj_close"]).sort_by([("ts", "ascending"), ("symbol", "ascending")]),
    note="vendor backfill 2018-2026",
)

db.table("prices").select(rows=count_star(), names=col("symbol").n_unique()).to_pandas()

# %% [markdown]
# ## 2. Portfolio P&L series in one SQL statement
#
# The book is a fixed 12-name portfolio: weights sum to 1, notional is \$10M.
#
# Daily simple returns come from a `lag()` window per symbol. The portfolio
# return is a weighted sum, via a join against an inline `VALUES` weights
# relation.
#
# Everything streams on time-sorted storage, with no pandas reshaping until we
# actually need rolling quantiles.

# %%
NOTIONAL = 10_000_000.0

WEIGHTS_SQL = """
    (VALUES ('AAPL', 0.10), ('MSFT', 0.10), ('NVDA', 0.08), ('AMZN', 0.08),
            ('GOOGL', 0.08), ('JPM', 0.10), ('XOM', 0.08), ('UNH', 0.08),
            ('PG', 0.08), ('KO', 0.06), ('CAT', 0.08), ('GS', 0.08)
    ) AS w(symbol, weight)
"""

# An inline VALUES relation has no builder verb, so this one stays SQL.
port = db.sql(
    f"""
    WITH rets AS (
        SELECT ts, symbol,
               adj_close / lag(adj_close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
        FROM prices
    )
    SELECT r.ts, sum(r.ret * w.weight) AS port_ret, count(*) AS names
    FROM rets r
    JOIN {WEIGHTS_SQL} ON r.symbol = w.symbol
    WHERE r.ret IS NOT NULL
    GROUP BY r.ts
    ORDER BY r.ts
    """
).to_pandas()

# Keep only days where all 12 names have a return (guards against partial days).
port = port[port["names"] == 12].set_index("ts")
port["pnl"] = NOTIONAL * port["port_ret"]
print(f"{len(port):,} trading days, "
      f"ann. vol {port['port_ret'].std() * np.sqrt(252):.1%}, "
      f"worst day {port['pnl'].min():,.0f} USD")

# %% [markdown]
# ## 3. Rolling historical VaR / ES, plus parametric flavors
#
# All measures use a trailing 252-day window **shifted by one day**, so the VaR
# reported for day *t* only uses information through *t−1*.
#
# That shift is the difference between a risk model and a hindsight model, and
# it is exactly the kind of detail an auditor will ask about.
#
# - **Historical.** The empirical quantile of the window. ES is the mean of the
#   tail beyond that quantile.
# - **Parametric normal.** Rolling mean and vol, plus a normal quantile.
# - **Parametric Student-t.** The same, with a t quantile rescaled to unit
#   variance, where the degrees of freedom are fitted once on the full
#   standardized sample. A cheap fat-tail fix.

# %%
from scipy import stats

r = port["port_ret"]
W = 252

def hist_var(x, q):
    return -np.quantile(x, q)

def hist_es(x, q):
    cut = np.quantile(x, q)
    return -x[x <= cut].mean()

roll = r.rolling(W)
port["var95"] = roll.apply(hist_var, args=(0.05,), raw=True).shift(1)
port["var99"] = roll.apply(hist_var, args=(0.01,), raw=True).shift(1)
port["es95"] = roll.apply(hist_es, args=(0.05,), raw=True).shift(1)
port["es99"] = roll.apply(hist_es, args=(0.01,), raw=True).shift(1)

mu, sd = roll.mean().shift(1), roll.std().shift(1)
port["var95_norm"] = -(mu + sd * stats.norm.ppf(0.05))

# Fit t df on the full standardized history, rescale to unit variance.
z = ((r - r.mean()) / r.std()).to_numpy()
nu = stats.t.fit(z, floc=0)[0]
t_scale = np.sqrt((nu - 2) / nu)
port["var95_t"] = -(mu + sd * stats.t.ppf(0.05, nu) * t_scale)

port[["var95", "es95", "var95_norm", "var95_t"]].dropna().tail(3).mul(NOTIONAL).round(0)

# %% [markdown]
# ## 4. Kupiec POF backtest: is 95% really 95%?
#
# Count the days where the realized loss exceeded the previous day's VaR, then
# run Kupiec's proportion-of-failures likelihood-ratio test, which is
# asymptotically chi-squared with 1 degree of freedom.
#
# A well-calibrated 95% VaR should be breached on about 5% of days. Rejecting
# the null means the model's coverage is wrong.

# %%
def kupiec_pof(returns: pd.Series, var: pd.Series, p: float) -> dict:
    mask = var.notna()
    ret, v = returns[mask], var[mask]
    n = len(ret)
    x = int((ret < -v).sum())
    pi = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = ((n - x) * np.log(1 - pi) + x * np.log(pi)) if 0 < x < n else 0.0
    lr = -2 * (ll_null - ll_alt)
    return {"n_days": n, "breaches": x, "expected": round(n * p, 1),
            "breach_rate": round(pi, 4), "LR_pof": round(lr, 2),
            "p_value": round(float(stats.chi2.sf(lr, 1)), 4)}

backtest = pd.DataFrame(
    {
        "hist 95%": kupiec_pof(r, port["var95"], 0.05),
        "hist 99%": kupiec_pof(r, port["var99"], 0.01),
        "normal 95%": kupiec_pof(r, port["var95_norm"], 0.05),
        "t 95%": kupiec_pof(r, port["var95_t"], 0.05),
    }
).T
backtest

# %% [markdown]
# Read the table honestly.
#
# The historical and normal models hold coverage at 95%, but the 99% historical
# VaR under-covers. Breaches cluster in vol-regime shifts, which a trailing
# empirical quantile is slow to catch. Kupiec tests coverage only, not
# independence; Christoffersen's test would flag the clustering.
#
# And note the trap in the "fat-tail fix". A Student-t rescaled to unit variance
# has *less* extreme quantiles than the normal at the 5% level, because the fat
# tails only cross over around 1%. A t-model naively swapped in at 95% is
# anti-conservative, and the test catches it.

# %% [markdown]
# ## 5. The audit trail: a versioned `risk_metrics` table
#
# The production pattern is to **backfill history in one commit, then commit
# once per EOD run**.
#
# Each daily append is atomic and carries a note. `versions()` then answers
# "which risk numbers were published on which day, and when exactly", with
# wall-clock commit timestamps and no extra logging infrastructure.
#
# Below we simulate the last 60 production days as individual commits.

# %%
metrics = port.dropna(subset=["var95", "var99", "es95", "es99"])[
    ["pnl", "var95", "es95", "var99", "es99"]
].copy()
for c in ["var95", "es95", "var99", "es99"]:
    metrics[c] *= NOTIONAL

mschema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pnl", pa.float64()),
        pa.field("var95", pa.float64()),
        pa.field("es95", pa.float64()),
        pa.field("var99", pa.float64()),
        pa.field("es99", pa.float64()),
    ]
)
db.create_table("risk_metrics", mschema, time_column="ts")

hist_part = metrics.iloc[:-60].reset_index()
db.append("risk_metrics", pa.Table.from_pandas(hist_part, schema=mschema, preserve_index=False),
          note="backfill through model go-live")

for day, row in metrics.iloc[-60:].iterrows():
    one = pa.Table.from_pandas(
        pd.DataFrame([row]).rename_axis("ts").reset_index(), schema=mschema, preserve_index=False
    )
    db.append("risk_metrics", one, note=f"eod risk {day.date()}")

[
    {k: v.get(k) for k in ("sequence", "op", "rows", "note")}
    for v in db.versions("risk_metrics")[-4:]
]

# %% [markdown]
# Sixty tiny commits also means sixty tiny Parquet segments. `compact()` merges
# them into efficient ones, as its own audited commit, so even maintenance shows
# up in the history.

# %%
before = db.versions("risk_metrics")[-1]
compacted = db.compact("risk_metrics", note="post-EOD-loop compaction")
print(f"segments: {before['segments']} -> {compacted['segments_total']}, "
      f"op={compacted['op']}, sequence={compacted['sequence']}")

# %% [markdown]
# The risk table is now queryable like any other. Here is every breach day of
# the last two years, straight from SQL.

# %%
(
    db.table("risk_metrics")
    .filter(col("pnl") < -col("var95"), col("ts") >= "2024-07-01T00:00:00Z")
    .select(
        "ts",
        pnl=col("pnl").round(),
        var95_floor=(-col("var95")).round(),
        var99_floor=(-col("var99")).round(),
    )
    .sort("ts", descending=True)
    .limit(8)
    .to_pandas()
)

# %% [markdown]
# ## 6. P&L against the VaR band, breaches marked

# %%
import matplotlib.pyplot as plt

plot_df = metrics.loc["2024-07-01":]
breach95 = plot_df[plot_df["pnl"] < -plot_df["var95"]]
breach99 = plot_df[plot_df["pnl"] < -plot_df["var99"]]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(plot_df.index, plot_df["pnl"], lw=0.6, color="0.4", label="daily P&L")
ax.plot(plot_df.index, -plot_df["var95"], lw=1.2, color="tab:orange", label="-VaR 95%")
ax.plot(plot_df.index, -plot_df["var99"], lw=1.2, color="tab:red", label="-VaR 99%")
ax.scatter(breach95.index, breach95["pnl"], s=28, color="tab:orange", zorder=3,
           label=f"95% breach ({len(breach95)})")
ax.scatter(breach99.index, breach99["pnl"], s=36, color="tab:red", marker="x", zorder=4,
           label=f"99% breach ({len(breach99)})")
ax.set_title("Daily P&L vs rolling 252d historical VaR ($10M book)")
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.legend(loc="lower left", ncols=3, fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - The whole return pipeline, per-symbol `lag()` returns plus weighted
#   portfolio aggregation, is one SQL statement on time-sorted storage. pandas
#   only enters for rolling quantiles.
# - Shift your VaR window by one day. Kupiec's POF test then tells you honestly
#   whether coverage holds, and the historical model's failure at 99% shows up
#   in the numbers rather than in folklore.
# - `risk_metrics` is an *audit trail for free*. One append per EOD run with a
#   note, and `versions()` gives the exact publication time of every risk number
#   ever reported. No extra logging system.
# - Many small daily commits are fine. Run `compact()` periodically, and it is
#   itself a versioned, noted commit.

# %%
db.close()
