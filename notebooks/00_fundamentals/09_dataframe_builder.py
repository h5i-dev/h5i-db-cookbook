# %% [markdown]
# # The DataFrame builder: queries as Python objects
#
# `db.table(...)` starts a **lazy** query you assemble with method calls
# instead of a SQL string. Nothing runs until a terminal call like
# `.collect()`. It is a compiler, not a second engine: every verb lowers to
# SQL that goes through `db.sql()`, so a built query sees the same session,
# the same table functions and the same version pins as the string you would
# have written - and `.sql()` shows you exactly what it produced.
#
# The payoff for a research desk is generated queries. A factor library that
# sweeps windows and columns in a loop builds SQL with f-strings today, which
# is where quoting bugs and `'` injection live. Here the identifiers are
# quoted at one site and a partially-built pipeline is an ordinary Python
# value you can pass around, extend and reuse.

# %%
import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, vwap, when

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_dataframe_builder"), create=True)

trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades)

prices = cu.make_daily_prices(days=500)  # 50 names x 500 sessions
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices)

print(f"trades: {len(trades):,} rows   prices: {len(prices):,} rows")

# %% [markdown]
# ## 1. A frame is a query that has not run
#
# `db.table("trades")` is the whole table as a starting point. Verbs return
# **new** frames, so nothing is mutated and a partial pipeline is safe to
# reuse. `.sql()` renders it; `.collect()` runs it.

# %%
liquid = db.table("trades").filter(col("symbol").is_in(["AAPL", "NVDA"]))
print(liquid.sql())

# %% [markdown]
# The idiomatic OHLCV rollup, built. `group_by(...).agg(...)` projects the
# keys alongside the aggregates, and `.first("ts")` / `.last("ts")` are the
# `first_value(x ORDER BY ts)` idiom that gives you bar opens and closes
# without a self-join.

# %%
bars = (
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
print(bars.sql())

# %%
bars.to_pandas().head(6)

# %% [markdown]
# ## 2. Expressions
#
# `col(name)` is a column, `lit(value)` a constant, and arithmetic and
# comparisons build up from there. Two traps worth meeting early:
#
# - Python cannot overload `and` / `or` / `not`, so boolean logic uses
#   `&`, `|`, `~`. They bind *tighter* than comparisons, so each comparison
#   needs its own parentheses.
# - Expressions keep **SQL** semantics, not Python's. `/` between two integer
#   columns is integer division. Cast when you mean true division.

# %%
signed = (
    db.table("trades")
    .filter((col("price") > 0) & (col("size") >= 100))
    .select(
        "ts",
        "symbol",
        "price",
        notional=col("price") * col("size"),
        lots=col("size").cast("DOUBLE") / 100,
        direction=when(col("side") == "B").then(lit(1)).otherwise(lit(-1)),
    )
)
print(signed.sql())

# %%
signed.to_pandas().head(4)

# %% [markdown]
# Identifiers are always quoted, so case survives (`col("Symbol")` finds a
# field named `Symbol`, which bare SQL would fold to lowercase), and a string
# literal is always a string - never syntax:

# %%
print(db.table("trades").filter(col("symbol") == "'; DROP TABLE trades; --").sql())

# %% [markdown]
# ## 3. How a pipeline becomes SQL
#
# Most pipelines compile to one flat `SELECT`: independent `with_columns`
# calls coalesce, and filtering a *base* column stays in the same `WHERE`.
# A stage that reads a column an earlier stage **computed** gets its own
# level, because SQL resolves `WHERE` against the `FROM`, not against sibling
# entries in the select list. Aggregation, `LIMIT` and `DISTINCT` also close a
# level, since whatever follows acts on their output.

# %%
movers = (
    db.table("prices")
    .with_columns(ret=col("close") / col("open") - 1)
    .filter(col("ret") > 0.01)  # reads a computed column -> subquery
    .sort("ret", descending=True)
    .limit(5)
)
print(movers.sql())

# %%
movers.to_pandas()

# %% [markdown]
# Knowing *where* levels close is the one thing worth internalizing, because
# it decides what the next stage can see. While the pipeline stays flat, a
# verb still reaches the base table - `select("ts", "symbol").sort("close")`
# resolves fine, because SQL's `ORDER BY` reads the `FROM`, not the select
# list. Once an aggregate closes the level, that column is genuinely gone and
# the engine says so:

# %%
try:
    db.table("prices").group_by("symbol").agg(count_star().alias("n")).sort("close").collect()
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: {str(e)[:180]}")

# %% [markdown]
# ## 4. The payoff: queries you generate
#
# This is where the builder earns its place. A sweep over several lookbacks
# is a Python loop over frames, not string surgery - and each frame is a
# value you can hold, name and reuse. Here: the gap between price and its own
# trailing mean, at three windows.
#
# Rolling methods take a `window` and an `order_by`, and optionally a
# `partition_by`. Unlike the `rolling_avg` SQL sugar, they carry a real
# `PARTITION BY`, so they do **not** mix symbols on a multi-symbol table.

# %%
WINDOWS = (5, 20, 60)

base = db.table("prices").filter(col("symbol").is_in(["STK000", "STK001", "STK002"]))

ma_gap = base.with_columns(
    **{
        f"gap_{n}d": col("close") / col("close").rolling_mean(n, order_by="ts", partition_by="symbol") - 1
        for n in WINDOWS
    }
)
print(ma_gap.sql())

# %%
ma_gap.select("ts", "symbol", *[f"gap_{n}d" for n in WINDOWS]).sort(["ts", "symbol"]).to_pandas().tail(6)

# %% [markdown]
# Cross-sectional operators rank a value against its peers *at the same
# instant*, so they take the bucket to compare within. Combining a few
# z-scored signals into one composite is the whole shape of a factor build:

# %%
combo = (
    db.table("prices")
    .with_columns(
        z_ret=(col("close") / col("open") - 1).cs_zscore(partition_by="ts"),
        z_vol=col("volume").cast("DOUBLE").cs_zscore(partition_by="ts"),
    )
    .with_columns(score=(col("z_ret") - col("z_vol")) / 2)
    .select("ts", "symbol", "z_ret", "z_vol", "score")
    .sort(["ts", "score"], descending=[False, True])
)
combo.to_pandas().head(5)

# %% [markdown]
# ## 5. Version pins and joins
#
# A read point passes straight to `db.table()` and lowers to `h5i()`, so a
# pinned builder query is bound at the source exactly like hand-written SQL.
# `.join()` renders both sides as subqueries aliased `l` and `r`; those
# aliases are the contract for reaching a specific side. That makes the
# "same query across N versions" comparison a function call:

# %%
db.append("trades", cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, start="2026-06-04", seed=8))


def per_symbol(version=None):
    return db.table("trades", version=version).group_by("symbol").agg(
        count_star().alias("n"), col("ts").max().alias("last_ts")
    )


drift = per_symbol(1).join(per_symbol(), on="symbol").select(
    symbol=col("symbol", relation="l"),
    trades_added=col("n", relation="r") - col("n", relation="l"),
)
print(drift.sql())

# %%
drift.sort("symbol").to_pandas()

# %% [markdown]
# `.join_asof()` lowers to the `asof_join` table function. That function takes
# *table names* and reads both at latest, so the builder refuses a side that
# already has verbs applied or a pin - rather than silently ignoring it.
# Filter after the join.

# %%
tape, quotes = cu.make_trades_and_quotes(days=2)  # shared base prices
for name, data in (("tape", tape), ("quotes", quotes)):
    db.create_table(name, data.schema, time_column="ts", sort_key=["ts", "symbol"])
    db.append(name, data)

try:
    db.table("tape").filter(col("symbol") == "AAPL").join_asof(db.table("quotes"), on="ts", by="symbol")
except h5i_db.InvalidInputError as e:
    print(f"{type(e).__name__}: {e}\nhint: {e.hint}")

# %%
tq = (
    db.table("tape")
    .join_asof(db.table("quotes"), on="ts", by="symbol", tolerance=5_000_000)
    .filter(col("symbol") == "AAPL")
    .select("ts", "symbol", "price", "bid", "ask", mid=(col("bid") + col("ask")) / 2)
)
print(tq.sql())

# %%
tq.to_pandas().head(4)

# %% [markdown]
# ## 6. The escape hatch, and the door back to SQL
#
# Full SQL coverage through verbs is deliberately not a goal. `sql_expr()`
# drops a raw fragment anywhere an expression is accepted - its text goes in
# verbatim, so it is the one place quoting is yours to get right.

# %%
tails = (
    db.table("prices")
    .group_by("symbol")
    .agg(
        p01=sql_expr("approx_percentile_cont(close, 0.01)"),
        p99=sql_expr("approx_percentile_cont(close, 0.99)"),
    )
    .sort("symbol")
    .limit(4)
)
tails.to_pandas()

# %% [markdown]
# The escape hatch you will reach for most is `lag`. There is no `.lag()`
# method, but a `sql_expr` fragment is windowable, so it takes `.over()` like
# any aggregate - and that covers `lag`, `lead`, `row_number` and the rest of
# the SQL window catalogue. Daily returns, the single most common shape in
# this cookbook:

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")

rets = (
    db.table("prices")
    .with_columns(prev_close=PREV_CLOSE)
    .with_columns(ret=col("close") / col("prev_close") - 1)
    .filter(col("ret").is_not_null())
    .select("ts", "symbol", "ret")
)
print(rets.sql())

# %%
rets.sort(["ts", "symbol"]).to_pandas().head(4)

# %% [markdown]
# Note the two-step `with_columns`: `ret` reads `prev_close`, which the stage
# before it computed, so the builder closes a level rather than emit SQL that
# would not resolve. Binding the fragment to a Python name (`PREV_CLOSE`)
# once and reusing it is the habit that keeps a factor library honest.

# %% [markdown]
# And when a pipeline outgrows the builder, `.sql()` hands you the query to
# paste into `db.sql()` and keep going from there. The two surfaces are one
# system with a door in the middle, and the generated SQL is deterministic -
# safe to snapshot-test or diff.
#
# Some things stay in `db.sql()` because there is no verb for them and the
# string is genuinely clearer: `UNION ALL`, deep multi-CTE chains, scalar
# subqueries, and the table functions `gapfill` / `resample` / `tail`.
# Stacking two read points into one labelled result is the everyday example:

# %%
db.sql(
    """
    SELECT 'version 1' AS read_point, count(*) AS rows FROM h5i('trades', 1)
    UNION ALL
    SELECT 'latest',                  count(*)         FROM trades
    """
).to_pandas()

# %% [markdown]
# ## Takeaways
#
# - `db.table(...)` is a lazy query; verbs return new frames and nothing runs
#   until `.collect()` / `.to_pandas()`. `.sql()` shows the compiled SQL.
# - The builder is a compiler over `db.sql()`, not a second engine - same
#   session, same table functions, same `h5i()` version pins.
# - Use `&`, `|`, `~` for boolean logic, and remember expressions carry SQL
#   semantics: integer `/` truncates.
# - Reach for it when queries are **generated** - a sweep over windows or
#   columns in a loop - where f-string SQL means quoting bugs. For a query you
#   write once, plain SQL is often shorter.
# - `rolling_*` and `cs_*` methods carry a real `PARTITION BY`, unlike the
#   `rolling_avg` SQL sugar, which is a global trailing row window.
# - `sql_expr()` is the escape hatch and `.sql()` is the door back; neither
#   surface is second-class.

# %%
db.close()
