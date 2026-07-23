# %% [markdown]
# # Realized volatility from ticks: signature plots, jumps, overnight risk
#
# Realized variance — the sum of squared intraday returns — is the standard
# nonparametric vol estimate, and computing it well is mostly a *sampling*
# problem: sample too fast and bid-ask bounce inflates the estimate, too slow
# and you throw information away. With ticks stored time-sorted in h5i-db,
# re-bucketing the same data at any frequency is one `time_bucket` query, so
# the classic diagnostics (signature plot, bipower variation, overnight
# decomposition) each take a few lines.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_rv"), create=True)

# %% [markdown]
# ## 1. Tick data with a known jump
#
# Five sessions of dense synthetic trades (100k/day) for one symbol. The
# generator quotes trades on either side of the mid — genuine bid-ask bounce,
# which is exactly the microstructure noise the signature plot will expose.
# We also inject a **one-off +1.5% jump** mid-session on day 4 (a level shift
# in all subsequent prices) so the jump-detection section has something real
# to find.

# %%
trades = cu.make_trades(symbols=["SPX"], days=5, trades_per_day=100_000, seed=42)

jump_ts = pd.Timestamp("2026-06-04 17:00:00", tz="UTC")
tdf = trades.to_pandas()
tdf.loc[tdf["ts"] >= jump_ts, "price"] *= 1.015  # permanent level shift

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
    ]
)
db.create_table("trades", schema, time_column="ts", sort_key=["ts"])
db.append(
    "trades",
    pa.Table.from_pandas(tdf[["ts", "symbol", "price", "size"]], preserve_index=False).cast(schema),
    note="5 sessions, +1.5% jump injected 06-04 17:00",
)
print(f"{len(tdf):,} trades over {tdf.ts.dt.date.nunique()} sessions")

# %% [markdown]
# ## 2. Daily RV from 5-minute bars
#
# The convention: bucket to 5 minutes, take last trade price per bucket,
# square the within-session log returns and sum. `time_bucket('5m', ts)` +
# `last_value(price ORDER BY ts)` does the bucketing in one streaming pass;
# the per-day diff/sum is a two-liner in pandas. Annualized vol is
# `sqrt(RV_day * 252)`.

# %%
bars5 = db.sql(
    """
    SELECT time_bucket('5m', ts) AS bar,
           last_value(price ORDER BY ts) AS close,
           count(*) AS n_trades
    FROM trades
    GROUP BY bar
    ORDER BY bar
    """
).to_pandas()
bars5["day"] = bars5["bar"].dt.date
bars5["r"] = np.log(bars5["close"]).groupby(bars5["day"]).diff()  # within-session only

rv_daily = bars5.groupby("day")["r"].apply(lambda r: (r**2).sum()).rename("rv")
pd.DataFrame({"rv": rv_daily, "ann_vol_%": np.sqrt(rv_daily * 252) * 100}).round(5)

# %% [markdown]
# ## 3. The signature plot
#
# Recompute RV at sampling intervals from 1 second to 30 minutes — each one is
# the *same* SQL query with a different bucket width. In a frictionless world
# RV would be flat across frequencies; with bid-ask bounce, observed returns
# are `true return + iid noise`, and the noise variance is paid *per
# observation* — so RV blows up as the interval shrinks. The elbow where the
# curve flattens is the highest frequency you can trust; for liquid names it
# sits near a few minutes, which is why "5-minute RV" is the industry default.

# %%
INTERVALS_S = [1, 5, 15, 30, 60, 120, 300, 600, 900, 1800]

sig_rows = []
for sec in INTERVALS_S:
    b = db.sql(
        f"""
        SELECT time_bucket('{sec}sec', ts) AS bar,
               last_value(price ORDER BY ts) AS close
        FROM trades
        GROUP BY bar
        ORDER BY bar
        """
    ).to_pandas()
    b["day"] = b["bar"].dt.date
    r = np.log(b["close"]).groupby(b["day"]).diff()
    rv = (r**2).groupby(b["day"]).sum()
    sig_rows.append({"interval_s": sec, "ann_vol": float(np.sqrt(rv.mean() * 252))})
sig = pd.DataFrame(sig_rows)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(sig["interval_s"], sig["ann_vol"] * 100, "o-", lw=1.4)
ax.set_xscale("log")
ax.set_xticks(INTERVALS_S)
ax.set_xticklabels(["1s", "5s", "15s", "30s", "1m", "2m", "5m", "10m", "15m", "30m"])
ax.set_title("Volatility signature plot: RV vs sampling interval (5-day mean)")
ax.set_xlabel("sampling interval")
ax.set_ylabel("annualized RV vol (%)")
fig.tight_layout()

# %% [markdown]
# Textbook shape: the 1-second estimate is roughly double the stable value —
# at that frequency a large share of each observed "return" is the spread,
# not volatility. Here the curve flattens by about a minute; on real tick
# data with more noise sources the elbow typically sits a bit further out,
# hence the 5-minute convention.
#
# ## 4. Jumps: bipower variation vs RV
#
# RV converges to *total* quadratic variation — continuous variance **plus**
# squared jumps. Barndorff-Nielsen & Shephard's bipower variation
# $BV = \frac{\pi}{2}\sum |r_i||r_{i-1}|$ is robust to jumps (a single huge
# return is multiplied by its finite neighbors instead of being squared), so
# `max(RV - BV, 0)` isolates the jump contribution. Day 4 — where we injected
# the 1.5% jump — should light up; the other days should show RV ≈ BV.

# %%
def bipower(r: pd.Series) -> float:
    a = r.abs().to_numpy()
    return float(np.pi / 2 * np.nansum(a[1:] * a[:-1]))


bv_daily = bars5.groupby("day")["r"].apply(bipower).rename("bv")
jumps = pd.DataFrame({"rv": rv_daily, "bv": bv_daily})
jumps["jump_var"] = (jumps["rv"] - jumps["bv"]).clip(lower=0)
jumps["jump_share_%"] = 100 * jumps["jump_var"] / jumps["rv"]
jumps.round(5)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(jumps))
ax.bar(x - 0.2, jumps["rv"] * 1e4, width=0.4, label="RV")
ax.bar(x + 0.2, jumps["bv"] * 1e4, width=0.4, label="bipower variation")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in jumps.index], rotation=20)
ax.set_title("RV vs bipower variation by day (jump injected on 06-04)")
ax.set_xlabel("session")
ax.set_ylabel("daily variance (bps²  ×10⁴)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The injected jump day shows RV well above BV — the gap is close to the
# injected `ln(1.015)² ≈ 2.2e-4` — while ordinary days agree to within noise.
#
# ## 5. Overnight vs intraday on real SPY data
#
# Close-to-close variance splits into an overnight gap (`open/prev close`)
# and an intraday part (`close/open`). Using the cached 30-minute SPY bars,
# one SQL query gets session opens and closes — with
# `time_bucket('1d', ts, 'America/New_York')` so sessions are bucketed on
# exchange days, not UTC days.

# %%
spy = cu.fetch_intraday(["SPY"], period="30d", interval="30m")
sschema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("spy_30m", sschema, time_column="ts", sort_key=["ts"])
db.append("spy_30m", spy.select(["ts", "symbol", "open", "close"]).cast(sschema))

days = db.sql(
    """
    SELECT time_bucket('1d', ts, 'America/New_York') AS day,
           first_value(open  ORDER BY ts) AS day_open,
           last_value(close ORDER BY ts)  AS day_close
    FROM spy_30m
    GROUP BY day
    ORDER BY day
    """
).to_pandas()

r_on = np.log(days["day_open"] / days["day_close"].shift(1)).dropna()
r_id = np.log(days["day_close"] / days["day_open"]).iloc[1:]
var_on, var_id = r_on.var(), r_id.var()
print(
    f"{len(days)} sessions of SPY\n"
    f"overnight var share: {var_on / (var_on + var_id):.0%}   "
    f"intraday var share: {var_id / (var_on + var_id):.0%}\n"
    f"ann vol — overnight: {np.sqrt(var_on * 252):.1%}, intraday: {np.sqrt(var_id * 252):.1%}"
)

# %% [markdown]
# A meaningful slice of SPY's total variance accrues while the market is
# closed — risk that intraday RV never sees, which matters for anything
# marked close-to-close (VaR, options). With only ~30 sessions this split is
# an illustration, not an estimate; in production you would run it over years
# of bars, which is the same query on a bigger table.
#
# ## Takeaways
#
# - One tick table + `time_bucket` at ten different widths = a complete
#   signature plot. Re-bucketing is a query, not a data pipeline. (Note:
#   sub-minute widths spell the unit out — `'30sec'`, not `'30s'`.)
# - Microstructure noise is visible even in synthetic data: bid-ask bounce
#   roughly doubled 1-second RV relative to its stable value; 5-minute
#   sampling is the conventional safe elbow.
# - RV − bipower variation cleanly flagged the one injected jump day and
#   recovered its magnitude; BV is the jump-robust denominator you want in
#   continuous-vol forecasts.
# - Overnight variance is real risk (a visible share of SPY's total) and
#   intraday RV excludes it by construction — decompose before you compare RV
#   to close-to-close vols.

# %%
db.close()
