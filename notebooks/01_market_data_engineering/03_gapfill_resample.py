# %% [markdown]
# # Regular grids for irregular markets: gapfill and resample
#
# Illiquid names don't trade every minute, but almost everything downstream -
# covariance matrices, joins against liquid benchmarks, risk marks - wants a
# regular time grid. h5i-db's `gapfill()` (alias `resample()`) turns a stored
# bar table into a regular grid in one SQL call, with three fill policies:
# `'null'` (holes stay visible), `'locf'` (last observation carried forward)
# and `'interpolate'`. The policy you pick is a modelling decision with real
# P&L consequences, and this recipe ends on the classic trap: phantom
# (zero) returns from stale locf prices.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, count_star, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_gapfill"), create=True)

# %% [markdown]
# ## 1. An illiquid name in a liquid tape
#
# Five sessions of ticks for three symbols, then we thin NVDA to 2% of its
# prints (~400 trades/day) to stand in for a small-cap that trades a few
# times a minute at best.

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7
)
tp = trades.to_pandas()
rng = np.random.default_rng(5)
keep = (tp["symbol"] != "NVDA") | (rng.random(len(tp)) < 0.02)
tp = tp[keep]

ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64()), pa.field("exchange", pa.string()),
     pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(tp, preserve_index=False).cast(trade_schema))

db.table("trades").group_by("symbol").agg(n_trades=count_star()).sort("symbol").to_pandas()

# %% [markdown]
# ## 2. Bars with holes
#
# 1-minute closes for NVDA. One h5i-db-specific design point: `gapfill()`
# operates on a **stored table name** and treats the whole table as a single
# series - there is no partition-by argument. So the pattern is: build the
# per-instrument bar series, store it as its own table, then gapfill that.
# (For a multi-symbol universe, write one bar table per series, or gapfill
# filtered copies.)

# %%
bar_schema = pa.schema(
    [ts_field, pa.field("close", pa.float64()), pa.field("volume", pa.int64())]
)
nvda_bars = (
    db.table("trades")
    .filter(col("symbol") == "NVDA")
    .group_by(time_bucket("1m", col("ts")).alias("bar"))
    .agg(close=col("price").last("ts"), volume=col("size").sum())
    .select(col("bar").alias("ts"), "close", "volume")
    .sort("ts")
    .to_arrow()
)
db.create_table("bars_nvda_1m", bar_schema, time_column="ts", sort_key=["ts"])
db.append("bars_nvda_1m", nvda_bars.cast(bar_schema))

n_bars = len(nvda_bars)
session_minutes = 5 * 390  # 13:30-20:00 UTC, five sessions
print(f"{n_bars:,} one-minute bars observed out of {session_minutes:,} session minutes "
      f"({n_bars / session_minutes:.0%} coverage)")

# %% [markdown]
# ## 3. `gapfill` onto a regular grid
#
# The step is in **raw microseconds** (the unit of the `ts` column):
# 1 minute = 60_000_000. The grid spans the table's min to max timestamp -
# overnight included - so filter to session hours downstream when that is
# what your analytics need.

# %%
locf = db.sql(
    "SELECT * FROM gapfill('bars_nvda_1m', 'ts', 60000000, 'locf')"
).to_pandas()
print(f"grid rows: {len(locf):,} (observed bars: {n_bars:,})")
locf.head(8)

# %% [markdown]
# Note the fine print visible above: locf carries **every** column forward,
# including `volume` - a phantom-volume artifact. For carry-forward marking,
# store price-only series (as here, use `close`), or run `'null'` mode and
# `COALESCE(volume, 0)` yourself.
#
# ## 4. The three fill policies, side by side
#
# Same call, three modes, one gappy morning window. `gapfill` composes like
# any table - the outer `WHERE` just filters its output.

# %%
window = {}
for mode in ("null", "locf", "interpolate"):
    window[mode] = db.sql(
        f"""
        SELECT ts, close FROM gapfill('bars_nvda_1m', 'ts', 60000000, '{mode}')
        WHERE ts >= '2026-06-02T14:00:00Z' AND ts < '2026-06-02T16:00:00Z'
        ORDER BY ts
        """
    ).to_pandas()

obs = window["null"].dropna(subset=["close"])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(window["locf"]["ts"], window["locf"]["close"],
        drawstyle="steps-post", lw=1.2, label="locf (carry forward)")
ax.plot(window["interpolate"]["ts"], window["interpolate"]["close"],
        lw=1.2, ls="--", label="interpolate")
ax.scatter(obs["ts"], obs["close"], s=22, color="black", zorder=3,
           label="observed 1m closes")
ax.set_title("NVDA (thinned) 1m closes, 2026-06-02 14:00-16:00: fill policies")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# Two of these are safe in different ways, one is dangerous: `'null'` keeps
# holes honest, `'locf'` is causal (uses only past data) but goes stale,
# while `'interpolate'` **looks ahead** - each filled point uses the *next*
# observation, so an interpolated series must never feed a backtest signal.
#
# ## 5. `resample` alias, and the row-count guard
#
# `resample(...)` is an exact alias of `gapfill(...)`. Both refuse to
# generate more than 1M rows - a fat-finger step (100ms over three days,
# below) raises a `LimitError` instead of silently materializing millions of
# rows. A dense continuous series shows the honest use: 72h of EURUSD ticks
# onto a 1-second grid.

# %%
fx = cu.make_fx_ticks(pairs=["EURUSD"], hours=72).to_pandas()
fx["mid"] = (fx["bid"] + fx["ask"]) / 2
fx_schema = pa.schema([ts_field, pa.field("mid", pa.float64())])
db.create_table("fx_mid", fx_schema, time_column="ts", sort_key=["ts"])
db.append("fx_mid", pa.Table.from_pandas(fx[["ts", "mid"]], preserve_index=False).cast(fx_schema))

grid_1s = db.sql(
    "SELECT count(*) AS n, count(mid) AS filled FROM resample('fx_mid', 'ts', 1000000, 'locf')"
).to_pandas()
print(f"1s grid: {grid_1s['n'][0]:,} rows from {len(fx):,} irregular ticks")

try:
    db.sql("SELECT count(*) FROM resample('fx_mid', 'ts', 100000, 'locf')")
except h5i_db.LimitError as e:
    print(f"100ms grid refused -> {type(e).__name__} [{e.code}]: {e}")

# %% [markdown]
# ## 6. Phantom returns: why the fill mode is a modelling decision
#
# Mark the illiquid book off the locf grid and your minute returns are
# mostly zero - stale prices masquerading as "no move". Compute the same
# return series both ways (intraday pairs only, overnight excluded):

# %%
def session_returns(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    mins = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d = d[(mins >= 13 * 60 + 30) & (mins < 20 * 60)]
    ret = np.log(d["close"]).diff()
    ret[d["ts"].dt.date != d["ts"].shift().dt.date] = np.nan  # drop overnight pair
    return ret.dropna()

null_grid = db.sql(
    "SELECT ts, close FROM gapfill('bars_nvda_1m', 'ts', 60000000, 'null') ORDER BY ts"
).to_pandas()

ann = np.sqrt(390 * 252)  # per-minute -> annualized
r_locf = session_returns(locf)
r_null = session_returns(null_grid)  # defined only where adjacent minutes both traded

for name, r in (("locf grid", r_locf), ("null grid", r_null)):
    print(f"{name}: {len(r):>5,} minute returns | {(r == 0).mean():>4.0%} exactly zero | "
          f"ann. vol {r.std() * ann:.1%} | excess kurtosis {r.kurtosis():.1f}")

# %% [markdown]
# Note what locf does and does not distort. Total variance is roughly
# conserved - a stale run collapses the true move into one delayed jump, so
# headline realized vol barely shifts. The *distribution* is mangled: nearly
# half the minutes report exactly zero move and excess kurtosis explodes
# (~5.5 vs ~0.5), i.e. the locf series is "nothing happened" punctuated by
# fictitious jumps timed at the next print, not at the move. Anything
# consuming minute returns - short-horizon risk, jump detection,
# cross-correlations against liquid names (the Epps effect) - inherits that
# artifact. A sane production pattern: store observed bars, publish the
# `'null'` grid, and make carry-forward an explicit, documented step at the
# point of use.
#
# ## Takeaways
#
# - `gapfill('table', 'ts', step_us, mode)` / `resample(...)` turn a stored
#   bar table into a regular grid in one call - the step is raw
#   microseconds (1m = 60_000_000) and generation is capped at 1M rows
#   (`LimitError`, not an OOM).
# - It treats the table as one series: store one bar table per instrument
#   (a natural fit, since bar tables are cheap versioned tables in h5i-db).
# - `'locf'` is causal but goes stale - and carries *all* columns, volume
#   included; `'interpolate'` looks ahead and must never feed a backtest;
#   `'null'` keeps the holes honest.
# - Phantom returns are measurable: on the locf grid ~half the minutes show
#   a zero return and kurtosis is ~10x the observed-pair series - headline
#   vol survives, but everything sensitive to the return *path* does not.

# %%
db.close()
