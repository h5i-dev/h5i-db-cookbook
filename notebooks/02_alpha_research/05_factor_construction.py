# %% [markdown]
# # Building a point-in-time factor library
#
# Equity factors die by lookahead: a B/P ratio computed with a book value the
# market had not seen yet will backtest beautifully and trade terribly. This
# recipe builds three classic factors - value (B/P), momentum (12-1), and a
# quality proxy (revenue-growth stability) - with the fundamental data joined
# **as of its report date** via h5i-db's `asof_join`, evaluates them with
# monthly ICs and quintile spreads, and persists the result as a versioned,
# snapshotted factor panel: the "factor library" pattern, where every rebuild
# is a commit you can diff and pin.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_factors"), create=True)

# %% [markdown]
# ## 1. Prices and fundamentals as separate feeds
#
# 50 synthetic names: ~3 years of daily prices, and 12 quarters of
# fundamentals where `ts` is the **report** (public availability) timestamp -
# 25–55 days after `period_end`, like real filings. Keeping the two feeds in
# separate tables with their own time semantics is what makes point-in-time
# joins possible at all.

# %%
prices = cu.make_daily_prices()   # 50 symbols, 750 sessions
funda = cu.make_fundamentals()    # 50 symbols, 12 quarters, report-lagged ts

px_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)]
    + [prices.schema.field(i) for i in range(1, len(prices.schema))]
)
db.create_table("prices", px_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices.sort_by([("ts", "ascending"), ("symbol", "ascending")]))

f_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)]
    + [funda.schema.field(i) for i in range(1, len(funda.schema))]
)
db.create_table("fundamentals", f_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fundamentals", funda.sort_by([("ts", "ascending"), ("symbol", "ascending")]))
print(f"prices: {prices.num_rows:,} rows   fundamentals: {funda.num_rows:,} rows")

# %% [markdown]
# ## 2. Derived fundamental signals, per report date
#
# Quality here = *stability* of quarterly revenue growth: the (negated) stddev
# of the last four quarter-over-quarter growth rates. Both the growth and its
# trailing stddev are window functions over the report-time series, so this is
# one SQL statement whose output we store as its own table, `fund_signals` -
# derived data is data, and it should be versioned like everything else.

# %%
fund_sig = db.sql(
    """
    WITH g AS (
        SELECT ts, symbol, book_value_m,
               revenue_m / lag(revenue_m) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS rev_g
        FROM fundamentals
    )
    SELECT ts, symbol, book_value_m, rev_g,
           stddev(rev_g) OVER (PARTITION BY symbol ORDER BY ts
                               ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS rev_g_std,
           count(rev_g)  OVER (PARTITION BY symbol ORDER BY ts
                               ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS n_g
    FROM g
    ORDER BY ts, symbol
    """
).to_arrow()

fs_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("book_value_m", pa.float64()),
        pa.field("rev_g", pa.float64()),
        pa.field("rev_g_std", pa.float64()),
        pa.field("n_g", pa.int64()),
    ]
)
db.create_table("fund_signals", fs_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fund_signals", fund_sig.cast(fs_schema), note="rev growth + 4q stability")

# %% [markdown]
# ## 3. Month-end price panel with momentum
#
# Same pattern as the momentum recipe: `lag(21)/lag(252)` per symbol for 12-1
# momentum, sampled on the last trading day of each month via
# `time_bucket('1mo', ...)`. This panel also gets stored - `asof_join`
# operates on tables, and month-end observation dates are exactly the left
# side we want fundamentals attached to.

# %%
panel_tbl = db.sql(
    """
    WITH sig AS (
        SELECT ts, symbol, close,
               lag(close, 21)  OVER w AS px_1m_ago,
               lag(close, 252) OVER w AS px_12m_ago
        FROM prices
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ),
    month_end AS (
        SELECT time_bucket('1mo', ts) AS month, symbol, max(ts) AS ts
        FROM prices GROUP BY month, symbol
    )
    SELECT s.ts, s.symbol, s.close,
           s.px_1m_ago / s.px_12m_ago - 1 AS momentum
    FROM sig s JOIN month_end USING (ts, symbol)
    ORDER BY s.ts, s.symbol
    """
).to_arrow()

panel_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("momentum", pa.float64()),
    ]
)
db.create_table("panel_me", panel_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("panel_me", panel_tbl.cast(panel_schema), note="month-end closes + 12-1 momentum")
db.sql("SELECT count(*) AS rows, count(DISTINCT ts) AS months FROM panel_me").to_pandas()

# %% [markdown]
# ## 4. The point-in-time join
#
# `asof_join('panel_me', 'fund_signals', 'ts', 'ts', 'symbol')`: for every
# month-end row, the latest fundamental row **reported at or before** that
# date, per symbol. No calendar arithmetic, no "shift the quarter by 45 days"
# heuristics - the report timestamp is the join key, so restatement lags are
# handled by construction. `ts_right` shows exactly which filing each month
# used; months before a symbol's first report get NULLs, as they should.

# %%
pit = db.sql(
    """
    SELECT ts, symbol, close, momentum,
           book_value_m, rev_g_std, n_g,
           ts_right AS report_ts
    FROM asof_join('panel_me', 'fund_signals', 'ts', 'ts', 'symbol')
    ORDER BY ts, symbol
    """
).to_pandas()
pit["staleness_days"] = (pit["ts"] - pit["report_ts"]).dt.days
pit[["ts", "symbol", "close", "momentum", "book_value_m", "report_ts", "staleness_days"]].tail(4)

# %% [markdown]
# ## 5. Assemble the three factors
#
# - **value**: B/P = book value / market cap, with a fixed synthetic share
#   count per symbol (the generator has no shares outstanding; a static,
#   seeded share count keeps the cross-section meaningful and deterministic).
# - **momentum**: 12-1, straight from the panel.
# - **quality**: −stddev of the last 4 revenue growth rates, required
#   complete (`n_g = 4`).
#
# Each factor is cross-sectionally z-scored per month (clipped at ±3), which
# makes them comparable and combinable.

# %%
rng = np.random.default_rng(123)
symbols = sorted(pit["symbol"].unique())
shares_m = dict(zip(symbols, rng.uniform(100, 2000, len(symbols)).round(1)))

pit["value"] = pit["book_value_m"] / (pit["close"] * pit["symbol"].map(shares_m))
pit["quality"] = np.where(pit["n_g"] == 4, -pit["rev_g_std"], np.nan)

FACTORS = ["value", "momentum", "quality"]


def zscore_by_month(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("ts")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std")).clip(-3, 3)


for f in FACTORS:
    pit[f"z_{f}"] = zscore_by_month(pit, f)
pit["z_combo"] = pit[[f"z_{f}" for f in FACTORS]].mean(axis=1)

coverage = pit.groupby("ts")[[f"z_{f}" for f in FACTORS]].count()
coverage.tail(3)

# %% [markdown]
# ## 6. Evaluation: information coefficients and quintile spreads
#
# The IC is the monthly Spearman rank correlation between the factor value at
# month-end *t* and the return over month *t+1* - the signal is fully
# observable before the return begins. A |t-stat| above ~2 on the mean IC is
# the usual bar for "this factor predicts something".

# %%
close_piv = pit.pivot(index="ts", columns="symbol", values="close")
fwd_ret = close_piv.pct_change().shift(-1)
pit = pit.merge(
    fwd_ret.stack().rename("fwd_ret").reset_index().rename(columns={"level_1": "symbol"}),
    on=["ts", "symbol"],
    how="left",
)


def monthly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[col, "fwd_ret"]).groupby("ts"):
        if len(g) >= 20:
            out[ts] = stats.spearmanr(g[col], g["fwd_ret"])[0]
    return pd.Series(out, name=col)


ics = pd.concat([monthly_ic(pit, f"z_{f}") for f in FACTORS + ["combo"]], axis=1)
ic_summary = pd.DataFrame(
    {
        "mean_IC": ics.mean(),
        "IC_vol": ics.std(),
        "t_stat": ics.mean() / ics.std() * np.sqrt(ics.notna().sum()),
        "months": ics.notna().sum(),
    }
)
ic_summary.round(3)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
for ax, f in zip(axes, FACTORS):
    s = ics[f"z_{f}"].dropna()
    ax.bar(s.index, s.values, width=18, color=np.where(s.values >= 0, "tab:blue", "tab:red"))
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(s.mean(), color="tab:green", lw=1.0, ls="--", label=f"mean {s.mean():+.3f}")
    ax.set_ylabel(f"{f} IC")
    ax.legend(loc="upper left")
axes[0].set_title("Monthly rank IC by factor (Spearman, forward 1-month return)")
axes[-1].set_xlabel("month end")
fig.tight_layout()

# %% [markdown]
# Quintile spreads give the same information in return space: sort into five
# 10-name buckets each month, hold equal-weight for one month, and look at
# Q5 − Q1.

# %%
def quintile_spread(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[col, "fwd_ret"]).groupby("ts"):
        if len(g) >= 25:
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            out[ts] = g["fwd_ret"][q == 4].mean() - g["fwd_ret"][q == 0].mean()
    return pd.Series(out)


spreads = pd.DataFrame({f: quintile_spread(pit, f"z_{f}") for f in FACTORS + ["combo"]})
pd.DataFrame(
    {
        "ann_spread": spreads.mean() * 12,
        "t_stat": spreads.mean() / spreads.std() * np.sqrt(spreads.notna().sum()),
    }
).round(3)

# %% [markdown]
# Read the numbers for what they are - this is where synthetic data teaches
# the most:
#
# - **Momentum** is genuinely positive (IC t ≈ 2.7, double-digit annualized
#   Q5−Q1 spread): the generator gives each name a persistent drift, so past
#   winners keep winning. The pipeline detects a real effect.
# - **Value** is a null, correctly: the cross-sectional B/P ranking is
#   dominated by each name's static book/shares draw, which carries no
#   information about future returns.
# - **Quality** is the trap. Its mean IC has |t| ≈ 3 - yet the factor is
#   built from *pure noise* (every name has identical revenue-growth vol by
#   construction). The trailing 4-quarter window means this month's factor is
#   nearly identical to last month's, so the ~18 monthly ICs are one or two
#   effective observations wearing eighteen hats, and the i.i.d. t-stat is
#   fiction. Slow-moving signals need overlap-robust inference (Newey-West,
#   non-overlapping windows) before you believe any t-stat.
#
# The combo inherits the noise it averages in. On ~20-30 months, none of this
# would clear a research bar on real data - the pipeline, not the alphas, is
# the deliverable.
#
# ## 7. Persist the panel - the factor library pattern
#
# The finished panel (raw factors + z-scores) becomes a `factor_panel` table,
# and the snapshot names the state of *every* table that produced it. Rebuild
# the library next month and you get a new version to diff; pin a paper or a
# production model to the snapshot it was trained on.

# %%
out_cols = ["ts", "symbol", "value", "momentum", "quality", "z_value", "z_momentum", "z_quality", "z_combo"]
panel_out = pit[out_cols].sort_values(["ts", "symbol"])

fp_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("symbol", pa.string())]
    + [pa.field(c, pa.float64()) for c in out_cols[2:]]
)
db.create_table("factor_panel", fp_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "factor_panel",
    pa.Table.from_pandas(panel_out, preserve_index=False).cast(fp_schema),
    note="build 001: value/momentum/quality + combo",
)
db.snapshot(
    "factor-build-001",
    tables=["prices", "fundamentals", "fund_signals", "panel_me", "factor_panel"],
    note="monthly factor library build",
)

db.sql(
    """
    SELECT ts, count(*) AS names, avg(z_combo) AS combo_mean, stddev(z_momentum) AS mom_z_std
    FROM h5i('factor_panel', 'factor-build-001')
    GROUP BY ts ORDER BY ts DESC LIMIT 3
    """
).to_pandas().set_index("ts").round(3)

# %% [markdown]
# ## Takeaways
#
# - `asof_join` on the **report timestamp** is the whole point-in-time
#   machinery: every month-end row picks up the latest filing the market had
#   actually seen, and `ts_right` documents which one.
# - Derived data is data: growth/stability signals, the month-end panel and
#   the final factor library all live as versioned h5i tables, built by SQL
#   windows (`lag`, `stddev ... OVER ROWS`, `time_bucket('1mo')`).
# - The evaluation is honest on synthetic data: momentum works (drift is
#   persistent by construction), value is a null, and quality's "significant"
#   t-stat is an overlapping-window artifact built from pure noise - the
#   cheapest lesson in factor-research inference you will ever get.
# - One `db.snapshot(...)` pins the *entire* input lineage of a factor build.
#   "Which fundamentals produced the panel my model trained on?" becomes a
#   query, not an investigation.

# %%
db.close()
