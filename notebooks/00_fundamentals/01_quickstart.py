# %% [markdown]
# # Quickstart: your first h5i-db market database
#
# h5i-db is an embedded, versioned time-series database built for quant
# workloads: every write is an atomic commit producing an immutable version,
# and the SQL layer (Apache DataFusion) ships native time-series operators —
# `time_bucket`, `vwap`, `ewma`, ASOF joins, gapfill. There is no server to
# run: the database is a directory, like SQLite or DuckDB.
#
# In five minutes we will:
#
# 1. create a database and a `trades` table,
# 2. ingest a few days of tick data,
# 3. compute minute bars with VWAP in one SQL statement,
# 4. travel back in time to a previous version of the table.

# %%
import h5i_db
import pyarrow as pa

import cookbook_utils as cu

print("h5i-db version:", h5i_db.__version__)

# %% [markdown]
# ## 1. Create a database and a table
#
# A `Database` is a directory on disk. Tables are declared with an Arrow
# schema plus a `time_column` — h5i-db stores segments sorted by that column
# and uses it for pruning, ASOF joins and bar rollups. The `sort_key` adds a
# secondary sort (symbol within each timestamp); it must start with the time
# column.

# %%
db = h5i_db.Database(cu.fresh_db("00_quickstart"), create=True)

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
db.tables()

# %% [markdown]
# ## 2. Ingest tick data
#
# `append` takes any pyarrow Table / RecordBatch. It is *strict*: data must be
# time-sorted and start at or after the table's current max timestamp — feed
# semantics, not upsert semantics. Each call is one atomic commit that
# produces a new immutable version.

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
commit = db.append("trades", trades)
commit

# %% [markdown]
# ## 3. Query it with SQL
#
# Full SQL via DataFusion, plus finance-native operators. One statement turns
# 100k+ ticks into minute bars with VWAP — and because segments are stored
# time-sorted, this streams instead of sorting.

# %%
bars = db.sql(
    """
    SELECT time_bucket('1m', ts) AS bar,
           symbol,
           first_value(price ORDER BY ts) AS open,
           max(price)                     AS high,
           min(price)                     AS low,
           last_value(price ORDER BY ts)  AS close,
           sum(size)                      AS volume,
           vwap(price, size)              AS vwap
    FROM trades
    GROUP BY bar, symbol
    ORDER BY bar, symbol
    """
).to_pandas()
bars.head(8)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for sym, g in bars.groupby("symbol"):
    ax.plot(g["bar"], g["vwap"] / g["vwap"].iloc[0], label=sym, lw=0.8)
ax.set_title("1-minute VWAP, normalized")
ax.set_xlabel("time")
ax.set_ylabel("VWAP (first bar = 1)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 4. Versions and time travel
#
# Every commit is listed in `versions()`. Append another day of data, then
# read the table *as it was before* — an O(1) operation, not a replay.

# %%
day4 = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, start="2026-06-04", seed=8)
db.append("trades", day4, note="day 4 feed")

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("trades")
]

# %%
v1_rows = len(db.read("trades", version=1))
v2_rows = len(db.read("trades", version=2))
latest = db.sql("SELECT count(*) AS n FROM trades").to_pandas()["n"][0]
print(f"version 1: {v1_rows:,} rows\nversion 2: {v2_rows:,} rows\nlatest:    {latest:,} rows")

# %% [markdown]
# Time travel works inside SQL too, via the `h5i()` table function — query an
# old version and the live table in the same statement:

# %%
db.sql(
    """
    SELECT now.symbol,
           now.n  - was.n  AS trades_added,
           now.mx - was.mx AS ts_advanced_by
    FROM      (SELECT symbol, count(*) AS n, max(ts) AS mx FROM trades GROUP BY symbol) now
    JOIN      (SELECT symbol, count(*) AS n, max(ts) AS mx FROM h5i('trades', 1) GROUP BY symbol) was
    USING (symbol)
    ORDER BY symbol
    """
).to_pandas()

# %% [markdown]
# ## Takeaways
#
# - A database is a directory; a table is an Arrow schema + a time column.
#   No server, no daemon — `pip install`, `Database(path, create=True)`, done.
# - `append` is an atomic commit with feed semantics (strictly ordered in
#   time). Bad ingest? Every previous version is still there.
# - One SQL statement gets you OHLCV + VWAP bars, streaming on sorted storage.
# - Time travel is first-class: `db.read(v)` in Python, `h5i('table', v)` in
#   SQL, both O(1). This becomes the backbone of reproducible research —
#   see recipe 05.

# %%
db.close()
