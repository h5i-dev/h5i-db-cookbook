# %% [markdown]
# # Pairs trading with a version-pinned data spine
#
# A cointegration pair strategy on real prices: scan candidate pairs with an
# Engle-Granger test, build a rolling hedge ratio, compute the spread z-score
# *in SQL* with h5i-db's window functions, and backtest with lagged signals
# and costs. The h5i-db twist: prices are loaded in two commits, so at the end
# we re-run the whole pipeline against `h5i('prices', v_early)` — the exact
# table an earlier study would have seen — and show that results are pinned to
# a data version, not to whatever the vendor file happens to contain today.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_pairs"), create=True)

# %% [markdown]
# ## 1. Load prices in two commits
#
# Same 30-name daily cache as the momentum recipe. We deliberately split the
# load at 2025-01-01: the first `append` is the table a hypothetical 2024
# study ran on, the second brings it to the present. Each `append` is one
# atomic commit; `versions()` keeps both forever.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])

prices = (
    daily.select(["ts", "symbol", "adj_close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
    .cast(schema)
)
cutoff = pd.Timestamp("2025-01-01", tz="UTC")
mask = pa.compute.less(prices["ts"], pa.scalar(cutoff, type=pa.timestamp("us", tz="UTC")))

c1 = db.append("prices", prices.filter(mask), note="history through 2024")
v_early = c1["sequence"]
c2 = db.append("prices", prices.filter(pa.compute.invert(mask)), note="2025 onwards")
print(f"v{v_early}: {c1['rows_total']:,} rows   v{c2['sequence']} (head): {c2['rows_total']:,} rows")

# %% [markdown]
# ## 2. Cointegration scan
#
# Three economically sensible candidates from the universe: KO/PEP, GS/JPM,
# CVX/XOM. We pull log adjusted closes with a plain SQL scan and run the
# Engle-Granger test (`statsmodels.coint`) on each. Correlated is not
# cointegrated — the p-value tests whether the *spread* is stationary, which
# is what mean reversion actually needs.

# %%
from statsmodels.tsa.stattools import coint

CANDIDATES = [("KO", "PEP"), ("GS", "JPM"), ("CVX", "XOM")]

logpx = (
    db.sql(
        """
        SELECT ts, symbol, ln(adj_close) AS log_px
        FROM prices
        WHERE symbol IN ('KO','PEP','GS','JPM','CVX','XOM')
        ORDER BY ts, symbol
        """
    )
    .to_pandas()
    .pivot(index="ts", columns="symbol", values="log_px")
)

scan = pd.DataFrame(
    [
        {
            "pair": f"{a}/{b}",
            "corr": logpx[a].diff().corr(logpx[b].diff()),
            "coint_p": coint(logpx[a], logpx[b])[1],
        }
        for a, b in CANDIDATES
    ]
).set_index("pair")
scan.round(4)

# %% [markdown]
# Only CVX/XOM clears a 5% threshold on this sample — KO/PEP, the textbook
# pair, does not (their return correlation is high but the spread trends).
# That is a typical scan outcome and worth internalizing: most "obvious"
# pairs fail the stationarity test. We trade CVX/XOM.
#
# ## 3. Rolling hedge ratio and the spread
#
# The hedge ratio is a rolling 252-day OLS beta of log XOM on log CVX
# (rolling cov/var — same thing, vectorized). The spread
# `log(XOM) - beta*log(CVX)` and its beta go into their own h5i table so the
# signal step can run in SQL, and so the spread series itself is versioned
# alongside the prices that produced it.

# %%
lx, ly = logpx["CVX"], logpx["XOM"]
beta = ly.rolling(252).cov(lx) / lx.rolling(252).var()
spread = ly - beta * lx

spread_df = (
    pd.DataFrame({"ts": spread.index, "spread": spread.values, "beta": beta.values})
    .dropna()
    .reset_index(drop=True)
)
spread_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("spread", pa.float64()),
        pa.field("beta", pa.float64()),
    ]
)
db.create_table("spread", spread_schema, time_column="ts", sort_key=["ts"])
db.append(
    "spread",
    pa.Table.from_pandas(spread_df, preserve_index=False).cast(spread_schema),
    note="XOM vs CVX, rolling 252d OLS hedge",
)
len(spread_df)

# %% [markdown]
# ## 4. Z-score in SQL
#
# The trading signal is the spread's z-score against a 60-day window. h5i-db
# gives us `rolling_avg(x, ts, n)` sugar for the mean and a standard
# `stddev(...) OVER (... ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)` for the
# dispersion — one query, streaming over sorted storage, no pandas
# `rolling()` needed until the stateful backtest itself.

# %%
zs = db.sql(
    """
    SELECT ts, spread, beta,
           (spread - rolling_avg(spread, ts, 60))
             / stddev(spread) OVER (ORDER BY ts ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
             AS z
    FROM spread
    ORDER BY ts
    """
).to_pandas()
zs = zs.set_index("ts")
zs.tail(3).round(4)

# %% [markdown]
# ## 5. Backtest: enter at |z| > 2, exit at zero-crossing
#
# The entry/exit rule is stateful, so we run a small explicit loop (2k rows —
# instant). Hygiene:
#
# - positions are decided on **yesterday's z** (`shift(1)`) — no lookahead;
# - daily P&L uses the **lagged** hedge ratio:
#   `pos[t-1] * (dlog XOM - beta[t-1] * dlog CVX)` — the return of the book
#   you actually held, not of a spread whose beta silently rebalanced itself;
# - costs: 10 bps per leg on notional traded, so a round turn on the pair
#   costs ~`2 * (1 + |beta|) * 10` bps.

# %%
z_lag = zs["z"].shift(1)
pos = np.zeros(len(zs))
p = 0.0
for i, z in enumerate(z_lag.fillna(0.0).to_numpy()):
    if p == 0.0:
        if z > 2:
            p = -1.0  # spread rich: short XOM, long beta*CVX
        elif z < -2:
            p = 1.0
    elif (p < 0 and z <= 0) or (p > 0 and z >= 0):
        p = 0.0
    pos[i] = p
pos = pd.Series(pos, index=zs.index, name="pos")

dlx = lx.diff().reindex(zs.index)
dly = ly.diff().reindex(zs.index)
beta_lag = zs["beta"].shift(1)
pair_ret = dly - beta_lag * dlx           # per unit of XOM leg
pnl_gross = pos.shift(1) * pair_ret
traded = pos.diff().abs().fillna(0.0) * (1 + beta_lag.abs())
pnl_net = (pnl_gross - 10 / 1e4 * traded).dropna()

equity = pnl_net.cumsum()
sharpe = pnl_net.mean() / pnl_net.std() * np.sqrt(252)
max_dd = (equity - equity.cummax()).min()
n_trades = int((pos.diff().abs() > 0).sum())
print(
    f"round turns: {n_trades // 2}   time in market: {(pos != 0).mean():.0%}\n"
    f"Sharpe (net): {sharpe:.2f}   total P&L: {equity.iloc[-1]:+.1%} (log, unit notional)\n"
    f"max drawdown: {max_dd:.1%}"
)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(zs.index, zs["z"], lw=0.7, color="tab:blue")
axes[0].axhline(2, color="tab:red", lw=0.7, ls="--")
axes[0].axhline(-2, color="tab:red", lw=0.7, ls="--")
axes[0].axhline(0, color="gray", lw=0.6)
entries = pos[(pos.diff() != 0) & (pos != 0)].index
axes[0].plot(entries, zs.loc[entries, "z"], "v", color="tab:red", ms=5, label="entry")
axes[0].set_ylabel("spread z-score")
axes[0].set_title("CVX/XOM spread z-score (60d) and net P&L")
axes[0].legend(loc="upper left")
axes[1].plot(equity.index, equity, lw=1.2, color="tab:green")
axes[1].set_ylabel("cum. P&L (log points)")
axes[1].set_xlabel("date")
fig.tight_layout()

# %% [markdown]
# A weakly positive Sharpe with deep drawdowns: holding a diverging spread all
# the way back to its mean is exactly how pairs books get hurt. With one pair,
# a borderline p-value and no stop-loss, this is a methodology demo, not a
# strategy — a real book would diversify across many pairs and cap divergence
# risk.
#
# ## 6. Store the signal and pin the run
#
# Positions and z-scores go into a `signals` table, and one snapshot pins
# `prices`, `spread` and `signals` together under a run id.

# %%
sig_df = zs.reset_index()[["ts", "z"]].assign(pos=pos.values).dropna()
sig_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("z", pa.float64()),
        pa.field("pos", pa.float64()),
    ]
)
db.create_table("signals", sig_schema, time_column="ts", sort_key=["ts"])
db.append("signals", pa.Table.from_pandas(sig_df, preserve_index=False).cast(sig_schema))
db.snapshot("pairs-run-001", tables=["prices", "spread", "signals"], note="CVX/XOM |z|>2")
db.sql("SELECT count(*) AS rows, min(ts) AS first, max(ts) AS last FROM h5i('signals','pairs-run-001')").to_pandas()

# %% [markdown]
# ## 7. Re-run the pipeline on an earlier data version
#
# The whole study, parameterized by *which version of `prices` it reads*. SQL
# time travel makes this a one-string change: `FROM prices` becomes
# `FROM h5i('prices', v_early)`. Running on the pre-2025 version reproduces
# what a 2024 study would have found; running it twice on the same version is
# bit-identical — the property that matters when a vendor restates history
# under your feet.

# %%
def run_pipeline(prices_rel: str) -> dict:
    """Coint p-value + net Sharpe for CVX/XOM from any prices relation."""
    px = (
        db.sql(
            f"""
            SELECT ts, symbol, ln(adj_close) AS log_px
            FROM {prices_rel}
            WHERE symbol IN ('CVX','XOM') ORDER BY ts, symbol
            """
        )
        .to_pandas()
        .pivot(index="ts", columns="symbol", values="log_px")
    )
    lx_, ly_ = px["CVX"], px["XOM"]
    b = ly_.rolling(252).cov(lx_) / lx_.rolling(252).var()
    s = ly_ - b * lx_
    z = ((s - s.rolling(60).mean()) / s.rolling(60).std()).shift(1)
    q = np.zeros(len(z))
    p_ = 0.0
    for i, v in enumerate(z.fillna(0.0).to_numpy()):
        if p_ == 0.0:
            if v > 2:
                p_ = -1.0
            elif v < -2:
                p_ = 1.0
        elif (p_ < 0 and v <= 0) or (p_ > 0 and v >= 0):
            p_ = 0.0
        q[i] = p_
    q = pd.Series(q, index=z.index)
    ret = q.shift(1) * (ly_.diff() - b.shift(1) * lx_.diff())
    net_ = (ret - 10 / 1e4 * q.diff().abs().fillna(0) * (1 + b.shift(1).abs())).dropna()
    return {
        "rows": len(px),
        "last_day": px.index[-1].date().isoformat(),
        "coint_p": round(coint(lx_, ly_)[1], 4),
        "sharpe_net": round(net_.mean() / net_.std() * np.sqrt(252), 3),
    }


runs = pd.DataFrame(
    {
        "head (today)": run_pipeline("prices"),
        f"h5i('prices', {v_early})": run_pipeline(f"h5i('prices', {v_early})"),
        f"h5i('prices', {v_early}) again": run_pipeline(f"h5i('prices', {v_early})"),
    }
).T
assert runs.iloc[1].equals(runs.iloc[2]), "same version must reproduce exactly"
runs

# %% [markdown]
# The pre-2025 run sees a different sample (different p-value, different
# Sharpe) — and re-running on that pinned version is exactly reproducible.
# "Which data did this number come from?" has a precise answer: a version
# integer.
#
# ## Takeaways
#
# - Engle-Granger on real data is humbling: of three sensible candidates only
#   CVX/XOM was cointegrated at 5%. Correlation is not cointegration.
# - The z-score ran entirely in SQL — `rolling_avg` sugar plus a standard
#   `stddev ... OVER ROWS` window on the stored `spread` table.
# - Backtest honesty: lagged z, lagged hedge ratio in the P&L, per-leg costs.
#   The result (modest Sharpe, ugly drawdowns) is what single-pair books look
#   like.
# - Loading data in meaningful commits pays off later: `h5i('prices', v)`
#   re-runs any study against the exact bytes it originally saw, and
#   `db.snapshot(...)` names that state across all three tables at once.

# %%
db.close()
