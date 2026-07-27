# %% [markdown]
# # Quickstart: your first h5i-db market database
#
# h5i-db is an embedded, versioned time-series database built for quant
# workloads: every write is an atomic commit producing an immutable version,
# and the SQL layer (Apache DataFusion) ships native time-series operators -
# `time_bucket`, `vwap`, `ewma`, ASOF joins, gapfill. There is no server to
# run: the database is a directory, like SQLite or DuckDB.
#
# In five minutes we will:
#
# 1. create a database and a `trades` table,
# 2. ingest a few days of tick data,
# 3. compute minute bars with VWAP in one query,
# 4. travel back in time to a previous version of the table.

# %%
import h5i_db
import pyarrow as pa
from h5i_db import col, count_star, time_bucket, vwap

import cookbook_utils as cu

print("h5i-db version:", h5i_db.__version__)

# %% [markdown]
# ## 1. Create a database and a table
#
# A `Database` is a directory on disk. Tables are declared with an Arrow
# schema plus a `time_column` - h5i-db stores segments sorted by that column
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
# time-sorted and start at or after the table's current max timestamp - feed
# semantics, not upsert semantics. Each call is one atomic commit that
# produces a new immutable version.

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
commit = db.append("trades", trades)
commit

# %% [markdown]
# ## 3. Query it
#
# There are two surfaces on the same engine. `db.table(...)` starts a **lazy
# query** you build with method calls - nothing runs until `.to_pandas()` -
# and `db.sql(...)` takes a string. Both go through DataFusion with the same
# finance-native operators: `time_bucket` for the bar grid,
# `first_value/last_value(... ORDER BY ts)` for open and close by event time,
# and `vwap` as a native aggregate. Because segments are stored time-sorted,
# this streams instead of sorting.

# %%
bar_query = (
    db.table("trades")
    .group_by(time_bucket("1m", col("ts")).alias("bar"), "symbol")
    .agg(
        col("price").first("ts").alias("open"),
        col("price").max().alias("high"),
        col("price").min().alias("low"),
        col("price").last("ts").alias("close"),
        col("size").sum().alias("volume"),
        vwap(col("price"), col("size")).alias("vwap"),
    )
    .sort(["bar", "symbol"])
)
bars = bar_query.to_pandas()
bars.head(8)

# %% [markdown]
# The builder is a compiler, not a second engine - `.sql()` shows exactly
# what it handed to DataFusion, and that string works verbatim in `db.sql()`.
# Recipe 09 covers the builder properly; the rest of this cookbook uses it by
# default and drops to SQL where a string reads better.

# %%
print(bar_query.sql())

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
# read the table *as it was before* - an O(1) operation, not a replay.

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
latest = db.table("trades").select(count_star().alias("n")).to_pandas()["n"][0]
print(f"version 1: {v1_rows:,} rows\nversion 2: {v2_rows:,} rows\nlatest:    {latest:,} rows")

# %% [markdown]
# Time travel works inside a query too, so an old version and the live table
# can meet in one statement. Pass a read point to `db.table(...)` - it lowers
# to the `h5i()` table function - and write the query once as a function of
# the version:

# %%
def per_symbol(version=None):
    return (
        db.table("trades", version=version)
        .group_by("symbol")
        .agg(count_star().alias("n"), col("ts").max().alias("mx"))
    )


was, now = per_symbol(1), per_symbol()
now.join(was, on="symbol").select(
    symbol=col("symbol", relation="l"),
    trades_added=col("n", relation="l") - col("n", relation="r"),
    ts_advanced_by=col("mx", relation="l") - col("mx", relation="r"),
).sort("symbol").to_pandas()

# %% [markdown]
# ## Takeaways
#
# - A database is a directory; a table is an Arrow schema + a time column.
#   No server, no daemon - `pip install`, `Database(path, create=True)`, done.
# - `append` is an atomic commit with feed semantics (strictly ordered in
#   time). Bad ingest? Every previous version is still there.
# - One query gets you OHLCV + VWAP bars, streaming on sorted storage -
#   built with `db.table(...)` verbs, or written as SQL; `.sql()` is the door
#   between them (recipe 09).
# - Time travel is first-class: `db.read(v)` in Python, `db.table(v)` in the
#   builder, `h5i('table', v)` in SQL, all O(1). This becomes the backbone of
#   reproducible research - see recipe 05.

# %%
db.close()
