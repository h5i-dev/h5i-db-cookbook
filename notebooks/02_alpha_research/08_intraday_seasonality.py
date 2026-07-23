# %% [markdown]
# # Intraday seasonality: volume U-shape, volatility smile, spread decay
#
# Almost every execution and alpha model conditions on time of day: volume
# concentrates at the open and close, volatility peaks in the first hour,
# spreads narrow through the morning. Measuring these curves correctly is a
# timezone problem before it is a statistics problem — "09:30 New York"
# drifts against UTC at every DST transition, and a naive `EXTRACT(hour)`
# on UTC timestamps smears buckets by an hour twice a year. h5i-db's
# `time_bucket` takes an IANA timezone as its third argument, so buckets
# align to *wall-clock* time in that zone, DST included.
#
# We measure the curves twice: on synthetic tick data (where we know which
# effects the generator does and does not contain — honesty checkpoint), and
# on real SPY/QQQ hourly bars.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_seasonality"), create=True)

# %% [markdown]
# ## 1. Five days of ticks, sixty days of real bars

# %%
trades, quotes = cu.make_trades_and_quotes(days=5)
for name, tbl in (("trades", trades), ("quotes", quotes)):
    db.create_table(name, tbl.schema, time_column="ts", sort_key=["ts", "symbol"])
    db.append(name, tbl, note="5-day synthetic feed")

# append enforces the declared sort key, so order by (ts, symbol) explicitly
bars = cu.fetch_intraday(["SPY", "QQQ"], period="60d", interval="1h").sort_by(
    [("ts", "ascending"), ("symbol", "ascending")]
)
db.create_table("bars_1h", bars.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("bars_1h", bars, note="real SPY/QQQ hourly bars")
{t: len(db.read(t)) for t in db.tables()}

# %% [markdown]
# ## 2. Volume by 30 minutes of the New York day
#
# `time_bucket('30m', ts, 'America/New_York')` anchors bucket boundaries to
# NY wall-clock time. Within one June week that is indistinguishable from
# UTC bucketing — but across the March/November DST transitions the NY
# buckets stay pinned to 09:30, 10:00, ... while UTC buckets would shift by
# an hour mid-sample and split every time-of-day average into two blurred
# populations. Same query, correct all year. The wall-clock label comes from
# converting the bucket start to NY time in pandas.

# %%
vol30 = db.sql(
    """
    SELECT time_bucket('30m', ts, 'America/New_York') AS bucket,
           symbol,
           sum(size)  AS volume,
           count(*)   AS n_trades
    FROM trades
    GROUP BY bucket, symbol
    ORDER BY bucket, symbol
    """
).to_pandas()
vol30["tod"] = vol30["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")

profile = (
    vol30.groupby(["tod", "symbol"])["volume"].mean().unstack()  # avg across the 5 days
)
profile_share = profile / profile.sum()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
for sym in profile_share.columns:
    ax.plot(profile_share.index, profile_share[sym] * 100, marker="o", ms=3, label=sym)
ax.set_title("Synthetic trades: share of daily volume by 30-min NY bucket")
ax.set_xlabel("bucket start (America/New_York)")
ax.set_ylabel("% of daily volume")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()

# %% [markdown]
# The U-shape is pronounced — by construction: the generator concentrates
# 40% of arrivals near the open and 20% near the close, mimicking the real
# pattern. The useful summary numbers for an execution scheduler:

# %%
first_last = pd.DataFrame(
    {
        "first 30m share": profile_share.loc["09:30"],
        "last 30m share": profile_share.loc["15:30"],
        "midday hour share": profile_share.loc["12:00"] + profile_share.loc["12:30"],
    }
).T
(first_last * 100).round(1)

# %% [markdown]
# ## 3. Volatility and spreads by time of day — an honest null
#
# The same bucketing applied to 30-minute bar returns (volatility smile) and
# quoted spreads. Here the synthetic data *cannot* show the real-world
# effect, and it is worth being precise about why: the generator's price is
# a homogeneous diffusion in calendar time (variance ∝ elapsed time,
# regardless of trade intensity) and its spread is drawn once per
# symbol-day. So the honest expectation is a *flat* volatility curve and a
# *flat* spread curve — which is what we get, and a useful placebo: any
# curvature the pipeline showed here would be a bug in the pipeline.

# %%
smile = db.sql(
    """
    WITH bars30 AS (
        SELECT time_bucket('30m', ts, 'America/New_York') AS bucket, symbol,
               first_value(price ORDER BY ts) AS open,
               last_value(price ORDER BY ts)  AS close
        FROM trades
        GROUP BY bucket, symbol
    )
    SELECT bucket, symbol, close / open - 1 AS bar_ret
    FROM bars30 ORDER BY bucket
    """
).to_pandas()
smile["tod"] = smile["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
vol_by_tod = smile.groupby("tod")["bar_ret"].std() * 1e4

# each bucket std rests on only 15 bar returns (5 days x 3 symbols), so test
# flatness properly: standardize per symbol, then Levene's test across buckets
from scipy import stats

smile["z"] = smile.groupby("symbol")["bar_ret"].transform(lambda s: s / s.std())
levene = stats.levene(*[g["z"].values for _, g in smile.groupby("tod")])
print(f"Levene test for equal variance across buckets: p = {levene.pvalue:.2f}")

spread = db.sql(
    """
    SELECT time_bucket('30m', ts, 'America/New_York') AS bucket, symbol,
           avg((ask - bid) / ((ask + bid) / 2)) * 10000 AS spread_bps
    FROM quotes
    GROUP BY bucket, symbol
    ORDER BY bucket
    """
).to_pandas()
spread["tod"] = spread["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
spread_by_tod = spread.groupby("tod")["spread_bps"].mean()

pd.DataFrame(
    {
        "bar-return std (bps)": vol_by_tod,
        "quoted spread (bps)": spread_by_tod,
    }
).loc[["09:30", "10:30", "12:30", "14:30", "15:30"]].round(2)

# %% [markdown]
# Flat, as predicted: the raw bucket stds wiggle (each rests on 15 bar
# returns), but Levene's test finds no evidence of unequal variance across
# buckets, and the spread column is constant to the basis point. The method
# is validated; the effect is absent from this generator. Real markets show
# ~2-3x first-hour volatility and spreads that start wide and tighten within
# minutes — the volatility half of that is next, from real data.
#
# One more timezone note while we are here: with `'1d'` widths the timezone
# argument controls where the *day* boundary falls. For a 24h asset class
# that decides which trades belong to "Monday":

# %%
db.sql(
    """
    SELECT time_bucket('1d', ts)                     AS day_utc,
           time_bucket('1d', ts, 'America/New_York') AS day_ny,
           sum(size)                                 AS volume
    FROM trades
    GROUP BY day_utc, day_ny
    ORDER BY day_utc
    LIMIT 3
    """
).to_pandas()

# %% [markdown]
# For US equities the session sits inside one UTC day, so both groupings
# agree and only the labels differ (NY midnight = 04:00 UTC). For FX or
# crypto flowing through 00:00 UTC, the choice changes every daily number —
# see the 24/7 markets recipe.
#
# ## 4. The real thing: SPY and QQQ, 60 days of hourly bars
#
# Returns per bar via `lag()` in SQL, then the same wall-clock grouping.
# One caveat kept honest: this cached sample (late April to July) does not
# straddle a DST transition, so UTC bucketing would happen to work here too
# — the NY-anchored version is the one that keeps working in March.

# %%
real = db.sql(
    """
    SELECT ts, symbol, volume,
           close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
    FROM bars_1h
    ORDER BY ts
    """
).to_pandas()
real["tod"] = real["ts"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
# first bar of each day carries an overnight return -> exclude from the smile
real.loc[real["tod"] == "09:30", "ret"] = np.nan

by_tod = real.groupby(["tod", "symbol"]).agg(
    avg_volume=("volume", "mean"), ret_std=("ret", "std")
)
share = by_tod["avg_volume"].unstack()
share = share / share.sum()
vol_curve = by_tod["ret_std"].unstack() * 1e4

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for sym in share.columns:
    axes[0].plot(share.index, share[sym] * 100, marker="o", label=sym)
    axes[1].plot(vol_curve.index, vol_curve[sym], marker="o", label=sym)
axes[0].set_title("Real hourly bars: volume share by NY hour")
axes[0].set_ylabel("% of daily volume")
axes[1].set_title("Real hourly bars: return std by NY hour")
axes[1].set_ylabel("hourly return std (bps)")
for ax in axes:
    ax.set_xlabel("bar start (America/New_York)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
fig.tight_layout()

print("SPY first-hour volume share: {:.1%}   last-hour: {:.1%}".format(
    share["SPY"].iloc[0], share["SPY"].iloc[-1]))

# %% [markdown]
# The real curves deliver what the synthetic ticks could not: the U-shaped
# volume profile *and* an elevated post-open volatility that decays into
# midday (the 09:30 bar is excluded from the smile since its `lag()` return
# spans the overnight gap — a classic subtlety worth encoding in the
# pipeline, not in a footnote).
#
# ## Takeaways
#
# - `time_bucket(width, ts, 'America/New_York')` is the DST-safe way to
#   build time-of-day statistics: buckets pin to wall-clock sessions, so
#   March and November do not smear your seasonal curves. With `'1d'` widths
#   the same argument decides where the trading day starts.
# - The whole pipeline is two GROUP BY queries per curve — volume, bar
#   returns, and spreads each collapse from hundreds of thousands of ticks
#   to a few dozen buckets inside the database.
# - Synthetic data is a placebo, not a substitute: it reproduced the volume
#   U-shape (built into the generator) and correctly showed *flat*
#   volatility and spread curves (absent from the generator). The real
#   SPY/QQQ bars supplied the volatility smile.
# - Excluding the overnight return from the 09:30 bucket is the kind of
#   one-line correctness detail that separates a usable seasonality curve
#   from a subtly wrong one.

# %%
db.close()
