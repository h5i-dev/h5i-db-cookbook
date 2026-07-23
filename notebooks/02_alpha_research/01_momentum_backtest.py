# %% [markdown]
# # Cross-sectional momentum: an honest monthly backtest
#
# The classic 12-1 momentum factor, end to end on real prices: signal
# computation as a SQL window query, monthly rebalance dates from
# `time_bucket('1mo', ...)`, and — the part most backtest stacks get wrong —
# a versioned, snapshotted signal table so that six months from now you can
# reproduce *exactly* what this run saw. The strategy itself is deliberately
# vanilla; the h5i-db-native workflow around it is the point.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_momentum"), create=True)

# %% [markdown]
# ## 1. Load real prices into a versioned table
#
# 30 large caps, daily, 2018–2026 (cached Yahoo Finance data). We store only
# what the study needs — `ts, symbol, adj_close` — with `sort_key`
# `["ts", "symbol"]` so per-symbol window scans stream in order. `append` is
# strict about that sort key: the input must be ordered by `ts`, then
# `symbol` within each timestamp.

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
commit = db.append("prices", prices, note="yfinance 30 names 2018-2026")
print(f"version {commit['sequence']}: {commit['rows_total']:,} rows")

# %% [markdown]
# ## 2. The signal, in one SQL statement
#
# 12-1 momentum = return from 12 months ago to 1 month ago, skipping the most
# recent month (short-term reversal). On a trading-day series that is
# `lag(px, 21) / lag(px, 252) - 1` per symbol. The rebalance grid comes from
# `time_bucket('1mo', ts)`: the last trading day of each calendar month.
# Joining the two gives the month-end signal panel — no pandas resampling, no
# calendar edge cases.

# %%
panel = db.sql(
    """
    WITH sig AS (
        SELECT ts, symbol, adj_close,
               lag(adj_close, 21)  OVER w AS px_1m_ago,
               lag(adj_close, 252) OVER w AS px_12m_ago
        FROM prices
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ),
    month_end AS (
        SELECT time_bucket('1mo', ts) AS month, symbol, max(ts) AS ts
        FROM prices
        GROUP BY month, symbol
    )
    SELECT s.ts, s.symbol,
           s.adj_close             AS close,
           s.px_1m_ago / s.px_12m_ago - 1 AS momentum
    FROM sig s
    JOIN month_end USING (ts, symbol)
    ORDER BY s.ts, s.symbol
    """
).to_pandas()
panel.tail(4)

# %% [markdown]
# ## 3. Portfolio construction and an honest P&L
#
# With only 30 names a "decile" is 3 stocks, so we long the top 10 and short
# the bottom 10, equal weight — closer to terciles, and less noisy. Discipline
# checklist:
#
# - **No lookahead**: the signal observed at month-end *t* earns the return
#   from *t* to *t+1*. The last month has no forward return and is dropped.
# - **Costs**: 10 bps of one-way turnover, charged on every weight change
#   (including the initial build).
# - **Eligibility**: a name needs 252 days of history before it can be ranked;
#   months with fewer than 25 rankable names are skipped.

# %%
mom = panel.pivot(index="ts", columns="symbol", values="momentum")
close = panel.pivot(index="ts", columns="symbol", values="close")
fwd_ret = close.pct_change().shift(-1)  # month-end t -> t+1, aligned at t

N_SIDE, COST_BPS = 10, 10
ranks = mom.rank(axis=1)  # NaNs are left unranked
n_valid = mom.notna().sum(axis=1)

weights = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
weights[ranks.ge(ranks.max(axis=1).to_numpy()[:, None] - N_SIDE + 1)] = 1 / N_SIDE
weights[ranks.le(N_SIDE)] = -1 / N_SIDE
weights.loc[n_valid < 25] = 0.0

turnover = weights.diff().abs().sum(axis=1)
turnover.iloc[0] = weights.iloc[0].abs().sum()

gross = (weights * fwd_ret).sum(axis=1)
net = (gross - COST_BPS / 1e4 * turnover).iloc[:-1]  # drop month with no fwd ret
bench = fwd_ret.mean(axis=1).iloc[:-1]  # equal-weight long-only, no costs


def perf_stats(r: pd.Series, periods: int = 12) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "ann_ret": r.mean() * periods,
        "ann_vol": r.std() * np.sqrt(periods),
        "sharpe": r.mean() / r.std() * np.sqrt(periods),
        "max_dd": dd.min(),
    }


stats = pd.DataFrame(
    {"momentum L/S (net)": perf_stats(net), "equal-weight bench": perf_stats(bench)}
).T
print(f"avg monthly one-way turnover: {turnover.mean():.0%}")
stats.round(3)

# %% [markdown]
# ## 4. Equity curves
#
# The long/short book is roughly market-neutral, so comparing it to a
# long-only benchmark is about *shape*, not level: the benchmark rides the
# 2018–2026 bull market while the momentum book mostly treads water.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot((1 + net).cumprod(), label=f"12-1 momentum L/S, net {COST_BPS}bps", lw=1.4)
ax.plot((1 + bench).cumprod(), label="equal-weight 30 names (long-only)", lw=1.4)
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.set_title("Monthly 12-1 momentum on 30 large caps, 2019-2026")
ax.set_xlabel("rebalance date")
ax.set_ylabel("growth of $1")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The momentum book lands near a **zero Sharpe** here. That is not a bug: 30
# mega caps are a terrible momentum universe (the premium historically lives
# in broader, smaller cross-sections), and 2020–2023 contained brutal momentum
# crashes. Resist the urge to tweak lookbacks until the curve looks good —
# that is how backtests die. Report it as it is.

# %% [markdown]
# ## 5. Version the signal — the reproducibility layer
#
# The panel that produced these numbers goes back into h5i-db as a `signals`
# table, and a **named snapshot** pins both `prices` and `signals` at this
# exact state. Anyone can later query `h5i('signals', 'mom-run-001')` — or
# re-read the prices the run saw — even after both tables move on. That is the
# audit trail a research platform actually needs.

# %%
sig_out = (
    pd.concat(
        [
            mom.stack().rename("momentum"),
            weights.stack().rename("weight"),
        ],
        axis=1,
    )
    .dropna(subset=["momentum"])
    .reset_index()
    .rename(columns={"level_0": "ts", "level_1": "symbol"})
    .sort_values(["ts", "symbol"])
)

sig_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("momentum", pa.float64()),
        pa.field("weight", pa.float64()),
    ]
)
db.create_table("signals", sig_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "signals",
    pa.Table.from_pandas(sig_out, preserve_index=False).cast(sig_schema),
    note="12-1 momentum, top/bottom 10, 10bps",
)
db.snapshot("mom-run-001", tables=["prices", "signals"], note="momentum backtest run 001")

db.sql(
    """
    SELECT ts, count(*) AS names,
           sum(CASE WHEN weight > 0 THEN 1 ELSE 0 END) AS longs,
           sum(CASE WHEN weight < 0 THEN 1 ELSE 0 END) AS shorts
    FROM h5i('signals', 'mom-run-001')
    GROUP BY ts ORDER BY ts DESC LIMIT 3
    """
).to_pandas()

# %%
[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("signals")
]

# %% [markdown]
# ## Takeaways
#
# - `lag(...) OVER (PARTITION BY symbol ORDER BY ts)` plus
#   `time_bucket('1mo', ts)` turns "compute 12-1 momentum on month-end dates"
#   into one SQL statement that streams on sorted storage.
# - Backtest hygiene is cheap: lag the signal one period, charge turnover,
#   drop the unknowable last month. Do it every time.
# - The result — near-zero Sharpe for large-cap momentum, ~70% monthly
#   turnover eating 10 bps a side — is the honest one. A universe of 30 mega
#   caps is a demo, not an alpha source.
# - `db.snapshot("mom-run-001", ...)` pins prices *and* signals in one named,
#   queryable state: `h5i('signals', 'mom-run-001')` reproduces this run
#   forever, no CSV archaeology required.

# %%
db.close()
