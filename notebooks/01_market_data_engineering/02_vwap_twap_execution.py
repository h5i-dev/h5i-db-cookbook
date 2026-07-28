# %% [markdown]
# # VWAP, TWAP and execution benchmarks
#
# Execution desks live and die by benchmark arithmetic: interval VWAP, TWAP,
# arrival price, slippage in basis points.
#
# All of it is bucketed, weighted aggregation over a trade tape, plus one
# point-in-time lookup against the quote stream. That is exactly the shape of
# query `time_bucket`, `vwap()` and `asof_join` are built for. `vwap()` is both
# an aggregate and a window function, which covers interval and running
# benchmarks with one operator.
#
# In this recipe we:
#
# 1. derive a microstructure-consistent tape from a quote stream,
# 2. compute full-day and 30-minute interval VWAPs,
# 3. contrast them with TWAP, where volume skew makes the two diverge,
# 4. benchmark a simulated parent order, 50k shares of AAPL worked over an
#    hour, against interval VWAP and arrival mid.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, time_bucket, vwap, wavg
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_vwap"), create=True)

# %% [markdown]
# ## 1. The data
#
# We start from quotes. `cu.make_quotes` gives top-of-book snapshots: one row
# every time the best bid or offer changes.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | quote timestamp, ascending |
# | `symbol` | `string` | ticker |
# | `bid`, `ask` | `float64` | best bid and offer |
# | `bid_size`, `ask_size` | `int64` | displayed depth at each side |

# %%
quotes = cu.make_quotes(symbols=["AAPL", "MSFT", "NVDA"], days=3, seed=11)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# Benchmarking fills against quotes only means something if trades and quotes
# describe the *same* market. So we derive the printed tape from the quote
# stream rather than generating it independently. Each trade lifts the offer or
# hits the bid of the prevailing quote, 10% execute at the mid, and every print
# carries a small exchange latency.
#
# The cookbook's stock trade and quote generators share only an opening price.
# That is fine for bar recipes and useless for quote-relative benchmarks.
#
# The derived tape has one row per print: `ts`, `symbol`, `price`, `size` and
# `side` (`B` for a buyer who paid the offer, `S` for a seller who hit the bid).

# %%
qp = quotes.to_pandas()
rng = np.random.default_rng(17)
is_trade = rng.random(len(qp)) < 0.35
tr = qp.loc[is_trade, ["ts", "symbol", "bid", "ask"]].reset_index(drop=True)
n = len(tr)
buy = rng.random(n) < 0.5
at_mid = rng.random(n) < 0.10
px = np.where(buy, tr["ask"], tr["bid"])
px = np.where(at_mid, (tr["bid"] + tr["ask"]) / 2, px)
tr["price"] = np.round(px, 4)
tr["size"] = np.maximum(1, rng.lognormal(4.0, 1.2, n) // 100 * 100).astype("int64")
tr["side"] = np.where(buy, "B", "S")
tr["ts"] = (tr["ts"] + pd.to_timedelta(rng.uniform(0.2, 5.0, n), unit="ms")).dt.floor("us")
tr = tr.sort_values(["ts", "symbol"])[["ts", "symbol", "price", "size", "side"]]

print(f"{len(tr):,} trades derived from {len(qp):,} quotes")
tr.head()

# %%
ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64()), pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(tr, preserve_index=False).cast(trade_schema))

quote_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("bid", pa.float64()),
     pa.field("ask", pa.float64()), pa.field("bid_size", pa.int64()),
     pa.field("ask_size", pa.int64())]
)
db.create_table("quotes", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes.cast(quote_schema))

db.tables()

# %% [markdown]
# ## 2. Full-day and interval VWAP
#
# `vwap(price, size)` is a native aggregate, so day VWAP is a one-liner. The
# `time_bucket('1d', ts, 'America/New_York')` argument cuts sessions at New York
# midnight, which keeps the numbers DST-safe.
#
# Change the width to `'30m'` and the same statement gives the interval VWAPs an
# execution scheduler would target.

# %%
SESSION = time_bucket("1d", col("ts"), timezone="America/New_York")

day_vwap = (
    db.table("trades")
    .group_by(SESSION.alias("session"), "symbol")
    .agg(day_vwap=vwap(col("price"), col("size")), volume=col("size").sum())
    .sort(["session", "symbol"])
    .to_pandas()
)
day_vwap

# %%
ivwap = (
    db.table("trades")
    .group_by(time_bucket("30m", col("ts")).alias("interval_start"), "symbol")
    .agg(interval_vwap=vwap(col("price"), col("size")), volume=col("size").sum())
    .sort(["interval_start", "symbol"])
    .to_pandas()
)
ivwap.head(6)

# %% [markdown]
# ## 3. TWAP vs VWAP
#
# TWAP weights every minute equally. VWAP weights minutes by volume. With a
# U-shaped volume profile, VWAP concentrates its weight at the open and close,
# so the two benchmarks part ways whenever the price near the extremes differs
# from the middle of the day.
#
# We build TWAP from 1-minute closes and measure the divergence per session in
# basis points. Aggregating a frame that is already an aggregate just nests it,
# so the CTE writes itself.

# %%
bars_1m = (
    db.table("trades")
    .group_by(time_bucket("1m", col("ts")).alias("bar"), "symbol")
    .agg(close=col("price").last("ts"))
)

twap = (
    bars_1m.group_by(
        time_bucket("1d", col("bar"), timezone="America/New_York").alias("session"),
        "symbol",
    )
    .agg(twap=col("close").mean())
    .sort(["session", "symbol"])
    .to_pandas()
)

bench = day_vwap.merge(twap, on=["session", "symbol"])
bench["vwap_minus_twap_bps"] = (bench["day_vwap"] / bench["twap"] - 1) * 1e4
bench[["session", "symbol", "day_vwap", "twap", "vwap_minus_twap_bps"]]

# %% [markdown]
# A few bps either way, driven entirely by whether the heavy open and close
# volume printed above or below the day's average price.
#
# The running view makes the mechanics visible. `vwap()` also works as a
# *window* function, so cumulative VWAP needs no manual `sum(pv)/sum(v)`
# bookkeeping.

# %%
def aapl_window(t_lo: str, t_hi: str):
    return db.table("trades").filter(
        col("symbol") == "AAPL", col("ts") >= t_lo, col("ts") < t_hi
    )


session = aapl_window("2026-06-02T00:00:00Z", "2026-06-03T00:00:00Z")

run = session.select(
    "ts", "price", running_vwap=vwap(col("price"), col("size")).over(order_by="ts")
).sort("ts").to_pandas()

closes_1m = (
    session.group_by(time_bucket("1m", col("ts")).alias("bar"))
    .agg(close=col("price").last("ts"))
    .sort("bar")
    .to_pandas()
)
closes_1m["running_twap"] = closes_1m["close"].expanding().mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(run["ts"], run["price"], lw=0.3, color="0.7", label="trades")
ax.plot(run["ts"], run["running_vwap"], lw=1.4, label="running VWAP")
ax.plot(closes_1m["bar"], closes_1m["running_twap"], lw=1.4, ls="--", label="running TWAP")
ax.set_title("AAPL 2026-06-02: running VWAP vs running TWAP")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 4. Benchmarking a parent order
#
# The desk buys 50,000 AAPL between 14:00 and 15:00 UTC (10:00–11:00 New York)
# on 2026-06-02. The order is worked POV-style, at roughly constant
# participation in the aggressive offer-lifting flow.
#
# We simulate the fills from the tape and store them as a `fills` table. Arrival
# price gets marked the standard way: the prevailing quote mid at order receipt,
# fetched with a one-row `asof_join`.

# %%
t0, t1 = pd.Timestamp("2026-06-02 14:00", tz="UTC"), pd.Timestamp("2026-06-02 15:00", tz="UTC")
win = tr[(tr["symbol"] == "AAPL") & (tr["ts"] >= t0) & (tr["ts"] < t1)]
liftable = win[win["side"] == "B"]  # prints where an aggressive buyer paid the offer

target = 50_000
pov = target / liftable["size"].sum()
fills = liftable[["ts", "symbol", "price"]].copy()
fills["size"] = np.maximum(1, (liftable["size"] * pov).round()).astype("int64")
fills.iloc[-1, fills.columns.get_loc("size")] += target - fills["size"].sum()

fill_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64())]
)
db.create_table("fills", fill_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fills", pa.Table.from_pandas(fills, preserve_index=False).cast(fill_schema))
print(f"{len(fills):,} fills, {fills['size'].sum():,} shares, "
      f"~{pov:.0%} of aggressive buy volume")

# %% [markdown]
# `asof_join` takes stored table names, so we first snapshot the quote window
# around order receipt into its own small table. Desks persist exactly such
# windows for later TCA review. Then the one-row parent order joins against it.
#
# As a cross-check, we confirm the asof lookup agrees with an explicit "latest
# quote at or before 14:00" point query on the full quote table.

# %%
qwin = (
    db.table("quotes")
    .filter(
        col("symbol") == "AAPL",
        col("ts") >= "2026-06-02T13:50:00Z",
        col("ts") < "2026-06-02T14:10:00Z",
    )
    .sort("ts")
    .to_arrow()
)
db.create_table("quotes_arrival", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_arrival", qwin.cast(quote_schema))

parent = pa.table(
    {"ts": pa.array([t0], type=pa.timestamp("us", tz="UTC")), "symbol": ["AAPL"]}
).cast(pa.schema([ts_field, pa.field("symbol", pa.string())]))
db.create_table("parent", parent.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("parent", parent)

MID = (col("bid") + col("ask")) / 2

arrival = (
    db.table("parent")
    .join_asof(db.table("quotes_arrival"), on="ts", by="symbol")
    .select("ts", "symbol", quote_ts=col("ts_right"), arrival_mid=MID)
    .to_pandas()
)

direct = (
    db.table("quotes")
    .filter(col("symbol") == "AAPL", col("ts") <= "2026-06-02T14:00:00Z")
    .sort("ts", descending=True)
    .limit(1)
    .select(mid=MID)
    .to_pandas()["mid"][0]
)
assert np.isclose(arrival["arrival_mid"][0], direct)
arrival

# %% [markdown]
# `asof_join` picked the last quote at or before 14:00:00. `quote_ts` shows how
# stale that quote was, and colliding right-side columns get a `_right` suffix.
#
# Now the scorecard: fill VWAP against interval VWAP, and against arrival.

# %%
VWAP = vwap(col("price"), col("size"))
hour = aapl_window("2026-06-02T14:00:00Z", "2026-06-02T15:00:00Z")

fill_vwap = db.table("fills").select(v=VWAP).to_pandas()["v"][0]
mkt_vwap = hour.select(v=VWAP).to_pandas()["v"][0]
arrival_mid = arrival["arrival_mid"][0]

print(f"arrival mid          : {arrival_mid:.4f}")
print(f"interval VWAP (14-15): {mkt_vwap:.4f}")
print(f"fill VWAP            : {fill_vwap:.4f}")
print(f"slippage vs interval VWAP: {(fill_vwap / mkt_vwap - 1) * 1e4:+.2f} bps")
print(f"slippage vs arrival mid  : {(fill_vwap / arrival_mid - 1) * 1e4:+.2f} bps")

# %%
fp = fills
mkt = hour.select("ts", "price", running_vwap=VWAP.over(order_by="ts")).sort("ts").to_pandas()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mkt["ts"], mkt["price"], lw=0.3, color="0.7", label="market trades")
ax.plot(mkt["ts"], mkt["running_vwap"], lw=1.4, label="running interval VWAP")
ax.scatter(fp["ts"], fp["price"], s=6, color="tab:red", label="our fills", zorder=3)
ax.axhline(arrival_mid, ls=":", color="tab:green", label="arrival mid")
ax.axhline(fill_vwap, ls="--", color="tab:red", label="fill VWAP")
ax.set_title("50k AAPL buy, 14:00-15:00 UTC: fills vs benchmarks")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

# %% [markdown]
# Paying the offer costs roughly the half-spread against interval VWAP, which is
# exactly what a POV schedule of marketable orders should show. Slippage against
# arrival additionally carries the price drift over the hour, the quantity known
# as implementation shortfall.
#
# ## 5. `vwap` and kdb-style `wavg` are the same statistic
#
# Migrating from q? `w wavg x` is weight-first. h5i-db ships both spellings.

# %%
eq = db.table("fills").select(
    vwap=vwap(col("price"), col("size")),
    wavg=wavg(col("size"), col("price")),
).to_pandas()
assert np.isclose(eq["vwap"][0], eq["wavg"][0])
eq

# %% [markdown]
# ## Takeaways
#
# - `vwap(price, size)` works as an aggregate, giving day and interval VWAP with
#   `time_bucket`, and as a window function, giving running VWAP. No manual
#   `sum(pv)/sum(v)` scaffolding either way. `wavg(w, x)` is the kdb-style
#   spelling of the same statistic.
# - TWAP from 1-minute closes is a CTE away, and the VWAP-TWAP divergence falls
#   out in bps per session, driven by U-shaped volume.
# - Arrival price is a one-row `asof_join` against a stored quote window. It is
#   cheap to cross-check against an explicit point query, and that is a habit
#   worth keeping for any benchmark that moves money.
# - Benchmarks are only meaningful on a consistent tape. Derive your trades from
#   the quote stream, or verify them against it, before trusting any
#   quote-relative number.

# %%
db.close()
