# %% [markdown]
# # OHLCV bars from tick data
#
# Bar construction is the first transformation every tick dataset goes
# through, and it is where subtle bugs are born: wrong bucket boundaries,
# `last` picked by file order instead of event time, sessions split across
# UTC midnights. h5i-db makes the whole pipeline one SQL statement —
# `time_bucket` + ordered `first_value`/`last_value` + `vwap` stream over
# time-sorted Parquet segments without a sort step — and the resulting bar
# table is itself a versioned table you can persist, audit and time-travel.
#
# In this recipe we:
#
# 1. roll a week of ticks into 1m/5m/1h bars,
# 2. build session-aligned daily bars (timezone- and DST-aware),
# 3. persist the 5m bars as a versioned `bars_5m` table,
# 4. validate the SQL bars bit-for-bit against a pandas reference.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_ohlcv"), create=True)

# %% [markdown]
# ## 1. Load a week of ticks
#
# ~450k synthetic trades: 3 symbols, 5 sessions, U-shaped intraday activity
# and bid-ask bounce. The table is declared with `time_column="ts"` and a
# secondary sort on `symbol` — that physical ordering is what lets the bar
# queries below stream instead of sort.

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=30_000, seed=7
)

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
# One statement per bar width. The idioms that matter:
#
# - `time_bucket('<width>', ts)` floors each tick to its bucket — widths like
#   `'1m'`, `'5m'`, `'1h'` (also `'1d'`, `'1mo'`, ...);
# - `first_value(price ORDER BY ts)` / `last_value(price ORDER BY ts)` inside
#   `GROUP BY` give open/close by *event time*, not by accident of row order —
#   no self-joins;
# - `vwap(price, size)` is a native aggregate.

# %%
def bar_query(width: str) -> str:
    return f"""
        SELECT time_bucket('{width}', ts) AS ts,
               symbol,
               first_value(price ORDER BY ts) AS open,
               max(price)                     AS high,
               min(price)                     AS low,
               last_value(price ORDER BY ts)  AS close,
               sum(size)                      AS volume,
               vwap(price, size)              AS vwap,
               count(*)                       AS n_trades
        FROM trades
        GROUP BY 1, 2
        ORDER BY 1, 2
    """


bars_5m = db.sql(bar_query("5m")).to_pandas()
bars_5m.head(6)

# %%
for width in ("1m", "5m", "1h"):
    n = db.sql(f"SELECT count(*) AS n FROM ({bar_query(width)})").to_pandas()["n"][0]
    print(f"{width:>3} bars: {n:,}")

# %% [markdown]
# ## 3. Session-aligned daily bars
#
# `time_bucket` takes an optional third argument: an IANA timezone or an
# origin timestamp. `time_bucket('1d', ts, 'America/New_York')` cuts days at
# New York midnight (04:00/05:00 UTC depending on DST) instead of UTC
# midnight — the boundaries below differ by exactly the EDT offset. For a
# 13:30–20:00 UTC cash session both cuts happen to *group* ticks the same
# way, but the NY variant stays correct across DST transitions where a fixed
# UTC offset would drift by an hour.

# %%
db.sql(
    """
    SELECT time_bucket('1d', ts)                     AS day_utc,
           time_bucket('1d', ts, 'America/New_York') AS day_new_york,
           count(*)                                  AS n_trades
    FROM trades
    GROUP BY 1, 2
    ORDER BY 1
    """
).to_pandas()

# %% [markdown]
# Where the bucket choice changes the *numbers*, not just the labels, is any
# session that crosses UTC midnight — overnight futures, FX, crypto. A
# Globex-style session opens 18:00 New York (22:00 UTC), so a naive UTC day
# slices every session in two, and so does a New-York-midnight day. The fix
# is the origin form: `time_bucket('1d', ts, '<session open>')` aligns
# buckets to the trading session itself. We synthesize three overnight
# sessions to make the difference concrete.

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
    "UTC day": "time_bucket('1d', ts)",
    "New York day": "time_bucket('1d', ts, 'America/New_York')",
    "session day": "time_bucket('1d', ts, '2026-06-01T22:00:00Z')",
}
pd.concat(
    db.sql(
        f"SELECT '{label}' AS scheme, {expr} AS bar_start, count(*) AS n_trades "
        "FROM fut_trades GROUP BY 1, 2 ORDER BY 2"
    ).to_pandas()
    for label, expr in schemes.items()
).reset_index(drop=True)

# %% [markdown]
# Three sessions of 4,000 trades each: only the session-origin buckets
# recover them as three clean daily bars — the calendar-day schemes split
# every session across two buckets.
#
# ## 4. Persist bars as a versioned table
#
# Bars are a derived dataset you will rebuild — which is exactly what
# `db.write` is for: it *replaces* the table contents in one atomic commit
# while keeping every previous build in the version history. Rebuild the
# bars after a tick correction and the old bars remain queryable via
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
    "bars_5m", db.sql(bar_query("5m")).to_arrow().cast(bar_schema),
    note="5m bars built from trades v1",
)
{k: commit[k] for k in ("table", "sequence", "op", "rows_total")}

# %% [markdown]
# ## 5. Candlestick view
#
# One symbol, one session, straight from the stored bar table: close and
# VWAP with the high–low range as a band.

# %%
one_day = db.sql(
    """
    SELECT ts, high, low, close, vwap
    FROM bars_5m
    WHERE symbol = 'AAPL' AND time_bucket('1d', ts) = '2026-06-02T00:00:00Z'
    ORDER BY ts
    """
).to_pandas()

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
# Trust, but verify: rebuild the 5m bars with `pandas.Grouper` from the same
# ticks and compare every field. Both floor timestamps to the bucket start
# and both take open/close by event order, so the two constructions should
# agree to float precision.

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

stored = db.sql("SELECT * FROM bars_5m ORDER BY symbol, ts").to_pandas()
merged = stored.merge(ref, on=["symbol", "ts"], suffixes=("", "_ref"))
assert len(merged) == len(stored) == len(ref)
for col in ("open", "high", "low", "close", "volume", "vwap", "n_trades"):
    assert np.allclose(merged[col], merged[f"{col}_ref"]), col
print(f"all {len(merged):,} bars match pandas across OHLC, volume, VWAP, count")

# %% [markdown]
# ## Takeaways
#
# - Ticks to OHLCV+VWAP bars is one SQL statement: `time_bucket` +
#   `first_value/last_value(... ORDER BY ts)` + `vwap`, streaming over
#   time-sorted storage — and it matches a pandas reference exactly.
# - `time_bucket`'s third argument does session alignment: an IANA timezone
#   for DST-safe calendar days, an origin timestamp for overnight sessions
#   that straddle midnight. Naive UTC days silently split Globex-style
#   sessions in two.
# - Derived bars belong in the database: `db.write` on `bars_5m` gives you
#   atomic rebuilds with the full build history retained — every downstream
#   consumer can pin the bar version it was computed from.

# %%
db.close()
