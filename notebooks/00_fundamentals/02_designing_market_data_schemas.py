# %% [markdown]
# # Designing market data schemas
#
# A table in h5i-db is an Arrow schema plus a time column, persisted as
# immutable, time-sorted Parquet segments under versioned manifests. Immutable
# segments are what makes schema design the one decision you cannot cheaply
# revisit.
#
# The types you pick decide storage size and which SQL operators apply cleanly.
# The `time_column` and `sort_key` decide how well scans prune, and how fast
# ASOF joins and bar rollups run.
#
# In this recipe we:
#
# 1. look at the three feeds every equity desk starts with,
# 2. justify each type choice, column by column,
# 3. declare `time_column` and `sort_key`, and say what they buy,
# 4. probe the strict append contract: what h5i-db rejects at the door,
# 5. map out which schema changes are cheap later, and which are a rebuild.

# %% [markdown]
# ## Terms used here
#
# | term        | meaning |
# | ----------- | --- |
# | trade       | an executed transaction: price, size, timestamp |
# | quote       | a venue's current willingness to trade, published as a bid and an ask |
# | bid / ask   | highest price a buyer will pay, lowest price a seller will accept |
# | OHLCV bar   | ticks aggregated into a fixed interval: open, high, low, close, volume |
# | Arrow       | the in-memory columnar format h5i-db speaks; pandas and polars convert cheaply |
# | schema      | the table's column names and types, fixed at creation |
# | time column | the column h5i-db sorts and prunes on; every table declares one |
# | sort key    | physical row order inside a segment; must start with the time column |
# | segment     | one immutable Parquet file holding a time range of rows |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_schemas"), create=True)

# %% [markdown]
# ## 1. The data
#
# Three feeds, three tables. Trades are the tick tape from `cu.make_trades`:
# one row per print.
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
trades = cu.make_trades(days=2, trades_per_day=5_000)
print(f"trades: {trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# Quotes are top-of-book snapshots: one row every time the best bid or offer
# changes, carrying `bid` / `ask` prices and `bid_size` / `ask_size` depth.

# %%
quotes = cu.make_quotes(days=2, quotes_per_day=8_000)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# Daily bars are the same three names rolled up to one row per symbol per
# session, with `ts` stamped at the 20:00 UTC close.

# %%
daily = cu.make_daily_prices(symbols=["AAPL", "MSFT", "NVDA"], days=250)
print(f"daily:  {daily.num_rows:,} rows x {daily.num_columns} columns")
daily.to_pandas().head()

# %% [markdown]
# ## 2. Type choices for tick-level trades
#
# The decisions that matter, column by column:
#
# - **`ts`: `timestamp[us, tz=UTC]`, non-nullable.** Microseconds cover
#   consolidated-feed granularity. Timezone-aware UTC removes a whole class of
#   DST bugs: convert to exchange time at query time with `time_bucket`'s
#   timezone argument, never at storage time. Make it non-nullable, because a
#   print without a timestamp is not data. Every raw-unit argument elsewhere in
#   the API (plan ranges, gapfill steps, ASOF tolerances) is then in
#   microseconds too.
# - **`price`: `float64`.** `decimal128` is exact and the right call in a
#   settlement ledger. At the research layer float64 wins: it round-trips
#   cent-grid equity prices exactly up to ~$10^{13}$, it is what every analytics
#   function (`vwap`, `ewma`, `stddev`, `corr`) expects, and it is half the
#   width. Keep decimals for books and records systems.
# - **`size`: `int64`.** Never float. Share counts are integers, and int64
#   headroom covers everything from odd lots to index rebalance crosses with one
#   type.
# - **`symbol`, `exchange`, `side`: `utf8`.** Parquet dictionary-encodes
#   low-cardinality strings inside the segment automatically. Plain `utf8` at the
#   schema layer therefore costs almost nothing on disk and keeps the schema
#   simple.

# %%
trades_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)

quotes_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
    ]
)

bars_1d_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)

# %% [markdown]
# ## 3. `time_column` and `sort_key`
#
# Declaring `time_column="ts"` is not metadata decoration. It is the contract
# h5i-db builds everything else on:
#
# - segments are stored **sorted by `ts`**, and each segment's time range goes
#   into the manifest, so a `WHERE ts BETWEEN ...` predicate skips whole
#   segments without opening them;
# - **ASOF joins** and **`time_bucket` rollups** stream over that sort order
#   instead of re-sorting your ticks on every query;
# - `append` can enforce feed semantics, because "later than everything stored"
#   is now well defined.
#
# `sort_key=["ts", "symbol"]` adds a secondary order within each timestamp tie.
# More importantly, it groups rows so per-symbol scans touch contiguous runs of
# data. The rule: the sort key must start with the time column, and the column
# you filter by most often (almost always `symbol`) goes next.

# %%
db.create_table("trades", trades_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("quotes", quotes_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("bars_1d", bars_1d_schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %%
for name, data in [("trades", trades), ("quotes", quotes), ("bars_1d", daily)]:
    commit = db.append(name, data, note="initial load")
    print(f"{name:8s} v{commit['sequence']}: {commit['rows_total']:>7,} rows, "
          f"{commit['segments_total']} segment(s)")

db.schema("trades")

# %% [markdown]
# A quick query to confirm the schemas earn their keep: per-symbol quoted spread
# in basis points, straight off the quotes table. Plain `utf8` symbol columns
# group and filter exactly as you would hope, with no special "category" type at
# this layer.

# %%
mid = (col("ask") + col("bid")) / 2

(
    db.table("quotes")
    .group_by("symbol")
    .agg(
        quotes=count_star(),
        avg_spread_bps=(((col("ask") - col("bid")) / mid).mean() * 1e4).round(2),
        min_bid=col("bid").min().round(2),
        max_ask=col("ask").max().round(2),
    )
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# ## 4. What strict append rejects
#
# `append` has feed semantics. The batch must match the declared schema, be
# sorted by the time column, and start at or after the table's current maximum
# timestamp. Everything else is rejected *before* anything is written, so the
# table head never moves on a failed append.
#
# In a database shared by a team, or written by an agent, that strictness is the
# point. Malformed vendor files and out-of-order backfills fail loudly at ingest
# instead of quietly corrupting downstream research.
#
# Every h5i-db exception carries a machine-readable `.code`, a `.retryable` flag
# and often a `.hint` with the recommended fix.
#
# **Rejection 1: schema mismatch.** A vendor "helpfully" ships prices as
# float32 and forgets the `side` column:

# %%
bad_schema_batch = pa.table(
    {
        "ts": trades["ts"][:100],
        "symbol": trades["symbol"][:100],
        "price": trades["price"][:100].cast(pa.float32()),   # wrong width
        "size": trades["size"][:100],
        "exchange": trades["exchange"][:100],
        # 'side' column missing entirely
    }
)
try:
    db.append("trades", bad_schema_batch)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"message   {e}")
    print(f"hint      {e.hint or '(none - the message says it all)'}")

# %% [markdown]
# **Rejection 2: unsorted data.** The same rows, shuffled. This is the classic
# outcome of concatenating per-exchange files without a final sort:

# %%
import numpy as np

rng = np.random.default_rng(0)
shuffled = trades.take(rng.permutation(len(trades)))
try:
    db.append("trades", shuffled)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %% [markdown]
# **Rejection 3: overlapping time range.** Re-delivering data the table already
# holds, so that the batch's minimum `ts` falls before the stored maximum, is
# refused for the same reason. `append` means *extend the feed*, never
# *interleave into history*. Deliberate restatements go through `write` or the
# previewable `plan_replace_range` flow, which recipes 05 and 06 cover.

# %%
try:
    db.append("trades", trades)  # the exact batch already ingested
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %%
# The head never moved: still one data commit per table.
[{k: v[k] for k in ("sequence", "op", "rows") if k in v} for v in db.versions("trades")]

# %% [markdown]
# ## 5. Schema evolution: know the limits up front
#
# Segments are immutable, so schemas are deliberately hard to change. Plan for
# the evolutions that are cheap and avoid the ones that are not.
#
# - **Cheap:** appending *trailing nullable* columns, which old segments read
#   back as NULL, and *widening* numerics (int32 → int64, float32 → float64).
# - **Expensive:** renames, type narrowing, reordering and dropping columns.
#   Each of those means writing a new table and backfilling it.
#
# The practical consequence is to start wide: `int64`, `float64`,
# `timestamp[us]`. And if a column *might* exist one day, such as a `condition`
# code on trades, it costs little to include it as nullable from day one.

# %% [markdown]
# ## Takeaways
#
# - A table is an Arrow schema plus a `time_column`. Segments are stored as
#   time-sorted Parquet, so declaring the time column buys pruning, streaming
#   `time_bucket` rollups and sort-free ASOF joins.
# - Defaults that serve quant work well: `timestamp[us, tz=UTC]` non-nullable
#   for time, `float64` for prices at the research layer, `int64` for sizes,
#   plain `utf8` for symbols.
# - `sort_key=["ts", "symbol"]` must start with the time column. The secondary
#   key is what makes per-symbol scans touch contiguous data.
# - Strict append rejects schema mismatches, unsorted batches and overlapping
#   time ranges before writing anything. Errors carry `.code` and `.hint`, and
#   the table head never moves on failure.
# - Evolve schemas by trailing nullable adds and numeric widening. Anything else
#   is a rebuild, so start wide.

# %%
db.close()
