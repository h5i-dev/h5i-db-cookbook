# %% [markdown]
# # OHLCV bars from tick data
#
# Rolling ticks into bars is where most tick pipelines pick up their first bug:
# a bucket boundary off by one, a close taken in file order instead of event
# order, a session cut in half at UTC midnight.
#
# In h5i-db the rollup is a single aggregation. `time_bucket` defines the grid,
# ordered `first_value`/`last_value` give open and close, and `vwap` is a native
# aggregate. Because segments are stored time-sorted, the query streams instead
# of sorting.
#
# The result is an ordinary table, so you can persist it, audit it and
# time-travel it like any other.
#
# In this recipe we:
#
# 1. roll a week of ticks into 1m/5m/1h bars,
# 2. build session-aligned daily bars that survive timezones and DST,
# 3. persist the 5m bars as a versioned `bars_5m` table,
# 4. validate the SQL bars against a pandas reference, field by field.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, count_star, lit, time_bucket, vwap
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_ohlcv"), create=True)

# %% [markdown]
# ## 1. The data
#
# A week of ticks from `cu.make_trades`: 3 symbols across 5 sessions, one row
# per print. Arrival times are U-shaped within each session and prices bounce
# between bid and ask.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | trade timestamp, ascending |
# | `symbol` | `string` | ticker |
# | `price` | `float64` | trade price |
# | `size` | `int64` | shares traded |
# | `exchange` | `string` | reporting venue |
# | `side` | `string` | `B` buyer-initiated, `S` seller-initiated |

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=30_000, seed=7
)
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# The table is declared with `time_column="ts"` and a secondary sort on
# `symbol`. That physical ordering is what lets the bar queries below stream
# rather than sort.

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades, note="week of ticks")["rows_total"]

# %% [markdown]
# ## 2. Bars at any width
#
# The bar rollup is one function of the width, so we build it once and call it.
# That is the payoff of a query that is a Python value rather than a string.
#
# Three idioms carry the whole query:
#
# - `time_bucket('<width>', ts)` floors each tick to its bucket. Widths look
#   like `'1m'`, `'5m'`, `'1h'`, and also `'1d'`, `'1mo'` and up;
# - `.first("ts")` and `.last("ts")` are `first_value(price ORDER BY ts)` and
#   its mirror. Open and close come from *event time*, not from an accident of
#   row order, and no self-join is needed;
# - `vwap(price, size)` is a native aggregate.
#
# One naming rule is worth internalizing: never alias a computed group key to
# the name of a column that already exists. `GROUP BY "ts"` would bind to the
# raw `ts` column rather than the bucket, giving one group per tick, silently.
# Bucket to `bar`, then rename in a following `select`, which lands a level down
# where the name is free.

# %%
def bars(width: str):
    return (
        db.table("trades")
        .group_by(time_bucket(width, col("ts")).alias("bar"), "symbol")
        .agg(
            open=col("price").first("ts"),
            high=col("price").max(),
            low=col("price").min(),
            close=col("price").last("ts"),
            volume=col("size").sum(),
            vwap=vwap(col("price"), col("size")),
            n_trades=count_star(),
        )
        .select(
            col("bar").alias("ts"),
            "symbol", "open", "high", "low", "close", "volume", "vwap", "n_trades",
        )
    )


bars_5m = bars("5m").sort(["ts", "symbol"]).to_pandas()
bars_5m.head(6)

# %% [markdown]
# Counting the bars at each width just extends the same frame. The aggregation
# closes a level, so a `count(*)` on top of it nests as a subquery without any
# string surgery:

# %%
for width in ("1m", "5m", "1h"):
    n = bars(width).select(count_star().alias("n")).to_pandas()["n"][0]
    print(f"{width:>3} bars: {n:,}")

# %% [markdown]
# ## 3. Session-aligned daily bars
#
# `time_bucket` takes an optional third argument: an IANA timezone, or an origin
# timestamp. `time_bucket('1d', ts, 'America/New_York')` cuts days at New York
# midnight, which is 04:00 or 05:00 UTC depending on DST, instead of UTC
# midnight. The boundaries below differ by exactly the EDT offset.
#
# For a 13:30–20:00 UTC cash session both cuts happen to *group* ticks the same
# way. The New York variant stays correct across DST transitions, where a fixed
# UTC offset would drift by an hour.

# %%
(
    db.table("trades")
    .group_by(
        time_bucket("1d", col("ts")).alias("day_utc"),
        time_bucket("1d", col("ts"), timezone="America/New_York").alias("day_new_york"),
    )
    .agg(n_trades=count_star())
    .sort("day_utc")
    .to_pandas()
)

# %% [markdown]
# Where the bucket choice changes the *numbers*, not just the labels, is any
# session that crosses UTC midnight: overnight futures, FX, crypto.
#
# A Globex-style session opens at 18:00 New York, which is 22:00 UTC. A naive
# UTC day slices every session in two, and so does a New-York-midnight day. The
# fix is the origin form. `time_bucket('1d', ts, '<session open>')` aligns
# buckets to the trading session itself.
#
# We synthesize three overnight sessions to make the difference concrete. Each
# holds 4,000 trades with `ts`, `price` and `size` columns.

# %%
rng = np.random.default_rng(42)
frames = []
for day in ("2026-06-01", "2026-06-02", "2026-06-03"):
    open_ts = pd.Timestamp(f"{day} 22:00", tz="UTC")  # 18:00 New York open
    n = 4_000
    secs = np.sort(rng.uniform(0, 22 * 3600, n))  # ~22h overnight session
    px = 5_000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n)))
    frames.append(
        pd.DataFrame(
            {
                "ts": open_ts + pd.to_timedelta(secs, unit="s"),
                "price": np.round(px * 4) / 4,  # quarter-point ticks
                "size": rng.integers(1, 10, n),
            }
        )
    )
fut_pd = pd.concat(frames).sort_values("ts").reset_index(drop=True)
fut_pd["ts"] = fut_pd["ts"].dt.floor("us")  # pandas is ns-resolution; h5i time is us
fut = pa.Table.from_pandas(fut_pd, preserve_index=False)

print(f"{len(fut_pd):,} overnight trades across 3 sessions")
fut_pd.head()

# %%
fut_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
    ]
)
db.create_table("fut_trades", fut_schema, time_column="ts", sort_key=["ts"])
db.append("fut_trades", fut.cast(fut_schema))

schemes = {
    "UTC day": time_bucket("1d", col("ts")),
    "New York day": time_bucket("1d", col("ts"), timezone="America/New_York"),
    "session day": time_bucket("1d", col("ts"), origin="2026-06-01T22:00:00Z"),
}
pd.concat(
    db.table("fut_trades")
    .group_by(bucket.alias("bar_start"))
    .agg(n_trades=count_star())
    .select(scheme=lit(label), bar_start=col("bar_start"), n_trades=col("n_trades"))
    .sort("bar_start")
    .to_pandas()
    for label, bucket in schemes.items()
).reset_index(drop=True)

# %% [markdown]
# Three sessions of 4,000 trades each, and only the session-origin buckets
# recover them as three clean daily bars. The calendar-day schemes split every
# session across two buckets.
#
# ## 4. Persist bars as a versioned table
#
# Bars are a derived dataset you will rebuild, which is exactly what `db.write`
# is for. It *replaces* the table contents in one atomic commit while keeping
# every previous build in the version history.
#
# Rebuild the bars after a tick correction and the old bars remain queryable via
# `h5i('bars_5m', <version>)`.

# %%
bar_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("n_trades", pa.int64()),
    ]
)
db.create_table("bars_5m", bar_schema, time_column="ts", sort_key=["ts", "symbol"])
commit = db.write(
    "bars_5m", bars("5m").sort(["ts", "symbol"]).to_arrow().cast(bar_schema),
    note="5m bars built from trades v1",
)
{k: commit[k] for k in ("table", "sequence", "op", "rows_total")}

# %% [markdown]
# ## 5. Candlestick view
#
# One symbol, one session, straight from the stored bar table: close and VWAP
# with the high-low range as a band.

# %%
one_day = (
    db.table("bars_5m")
    .filter(
        col("symbol") == "AAPL",
        time_bucket("1d", col("ts")) == "2026-06-02T00:00:00Z",
    )
    .select("ts", "high", "low", "close", "vwap")
    .sort("ts")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(one_day["ts"], one_day["low"], one_day["high"],
                alpha=0.25, label="high-low range")
ax.plot(one_day["ts"], one_day["close"], lw=1.2, label="close")
ax.plot(one_day["ts"], one_day["vwap"], lw=1.0, ls="--", label="bar VWAP")
ax.set_title("AAPL 5-minute bars, 2026-06-02")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. Validate against pandas
#
# Trust, but verify. We rebuild the 5m bars with `pandas.Grouper` from the same
# ticks and compare every field. Both constructions floor timestamps to the
# bucket start and both take open and close by event order, so they should agree
# to float precision.

# %%
tp = trades.to_pandas()
tp["pv"] = tp["price"] * tp["size"]
ref = (
    tp.groupby(["symbol", pd.Grouper(key="ts", freq="5min")])
    .agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        pv=("pv", "sum"),
        n_trades=("price", "count"),
    )
    .reset_index()
)
ref["vwap"] = ref["pv"] / ref["volume"]

stored = db.table("bars_5m").sort(["symbol", "ts"]).to_pandas()
merged = stored.merge(ref, on=["symbol", "ts"], suffixes=("", "_ref"))
assert len(merged) == len(stored) == len(ref)
for field in ("open", "high", "low", "close", "volume", "vwap", "n_trades"):
    assert np.allclose(merged[field], merged[f"{field}_ref"]), field
print(f"all {len(merged):,} bars match pandas across OHLC, volume, VWAP, count")

# %% [markdown]
# ## Takeaways
#
# - Ticks to OHLCV+VWAP bars is one aggregation: `time_bucket` plus
#   `.first("ts")`/`.last("ts")` plus `vwap`, streaming over time-sorted
#   storage. It matches a pandas reference exactly.
# - Writing it as a `bars(width)` function rather than a SQL template means
#   every width, and the row counts on top of them, come from one definition.
# - `time_bucket`'s third argument does session alignment. Use `timezone=` for
#   DST-safe calendar days and `origin=` for overnight sessions that straddle
#   midnight. Naive UTC days silently split Globex-style sessions in two.
# - Derived bars belong in the database. `db.write` on `bars_5m` gives atomic
#   rebuilds with the full build history retained, so every downstream consumer
#   can pin the bar version it was computed from.

# %%
db.close()
