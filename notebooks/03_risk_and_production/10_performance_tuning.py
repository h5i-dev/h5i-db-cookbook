# %% [markdown]
# # Performance tuning: pruning, projection, commit granularity, caches
#
# h5i-db stores each table as immutable, time-sorted Parquet segments under a
# versioned manifest. That layout dictates what makes queries fast - and what
# quietly makes them slow:
#
# 1. **time-range pruning** - the manifest records each segment's time range,
#    so time predicates skip whole segments before any I/O;
# 2. **column projection** - Parquet is columnar; touch fewer columns, read
#    fewer bytes;
# 3. **commit granularity** - every commit writes a manifest and at least one
#    segment; a thousand tiny commits means a thousand tiny files until you
#    `compact()`;
# 4. **caches** - the first query after opening a database pays for segment
#    footer/metadata reads that later queries do not.
#
# We measure all four on a ~3M-row tick table, then close with the resource
# guards (`timeout`, `max_rows`, `memory_limit`) that keep a shared research
# box polite. Absolute numbers are machine-specific; the *ratios* are the
# lesson.

# %%
import time

import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_perf"), create=True)


def bench(fn, repeat=3):
    """Best-of-3 wall time - min filters out scheduler/GC noise."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


timings_ms = {}  # collected for the chart at the end

# %% [markdown]
# ## 1. The dataset: 10 days of ticks, one commit per day
#
# ~3M rows appended day-by-day - the batching pattern h5i-db expects (one
# commit per publication event, not per row and not one monolith). Ten
# commits leave roughly one segment per day, which is what gives the time
# predicates below something to prune.

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=10, trades_per_day=100_000, seed=42)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

trades_df_ts = trades["ts"].to_pandas()
day_starts = trades_df_ts.dt.normalize().unique()
t0 = time.perf_counter()
for day in day_starts:
    day_end = day + pd.Timedelta(days=1)
    mask_lo = trades_df_ts.searchsorted(day)
    mask_hi = trades_df_ts.searchsorted(day_end)
    db.append("trades", trades.slice(mask_lo, mask_hi - mask_lo), note=f"day {day.date()}")
print(f"{len(trades):,} rows in {len(day_starts)} daily commits ({time.perf_counter() - t0:.1f}s)")

# %% [markdown]
# ## 2. Time-range pruning: scan a day, not the decade
#
# Same shape of query, two spans: materialize *everything* vs one day out of
# ten. The one-day scan touches ~10% of the rows and the manifest lets the
# engine skip the other nine segments without opening them. Note the two
# flavors: raw materialization (`SELECT *`) shows pruning at full strength;
# a `GROUP BY` aggregation is compute-light per row, so its wall-clock gain
# is smaller - pruning saves I/O, not hash-table work.

# %%
lo = int(pd.Timestamp("2026-06-05", tz="UTC").value // 1000)
hi = int(pd.Timestamp("2026-06-06", tz="UTC").value // 1000)
day_pred = f"ts >= to_timestamp_micros({lo}) AND ts < to_timestamp_micros({hi})"
agg = "SELECT symbol, count(*) AS n, vwap(price, size) AS vw FROM trades {} GROUP BY symbol"

timings_ms["SELECT * (10 days)"] = 1e3 * bench(lambda: db.sql("SELECT * FROM trades").to_arrow())
timings_ms["SELECT * (1 day)"] = 1e3 * bench(lambda: db.sql(f"SELECT * FROM trades WHERE {day_pred}").to_arrow())
timings_ms["agg (10 days)"] = 1e3 * bench(lambda: db.sql(agg.format("")).to_arrow())
timings_ms["agg (1 day)"] = 1e3 * bench(lambda: db.sql(agg.format(f"WHERE {day_pred}")).to_arrow())

print(f"materialize: {timings_ms['SELECT * (10 days)']:.0f} ms full -> "
      f"{timings_ms['SELECT * (1 day)']:.0f} ms one day "
      f"({timings_ms['SELECT * (10 days)'] / timings_ms['SELECT * (1 day)']:.1f}x)")
print(f"aggregate:   {timings_ms['agg (10 days)']:.1f} ms full -> "
      f"{timings_ms['agg (1 day)']:.1f} ms one day "
      f"({timings_ms['agg (10 days)'] / timings_ms['agg (1 day)']:.1f}x)")

# %% [markdown]
# ## 3. Projection: name your columns
#
# `SELECT *` on a 6-column tick table drags symbol/exchange/side strings
# through every operator. Naming the two columns you need reads only their
# Parquet column chunks - the classic columnar win, and the reason
# `SELECT *` belongs in exploration, not in scheduled jobs.

# %%
timings_ms["SELECT ts, price"] = 1e3 * bench(lambda: db.sql("SELECT ts, price FROM trades").to_arrow())
print(f"SELECT *:         {timings_ms['SELECT * (10 days)']:.0f} ms")
print(f"SELECT ts, price: {timings_ms['SELECT ts, price']:.0f} ms "
      f"({timings_ms['SELECT * (10 days)'] / timings_ms['SELECT ts, price']:.1f}x)")

# %% [markdown]
# ## 4. Commit granularity - and `compact()` as the antidote
#
# Every commit is durable: manifest write, fsync, new segment file. Committing
# a day of ticks in 300 micro-batches pays that price 300 times on ingest
# *and* leaves 300 small segments for every later query to open. `compact()`
# rewrites them into one segment as a new version - history intact, old
# versions still readable.
#
# One day of ticks (~300k rows), ingested both ways:

# %%
day1 = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, trades_per_day=100_000, seed=43)

db.create_table("t_one", day1.schema, time_column="ts", sort_key=["ts", "symbol"])
t0 = time.perf_counter()
db.append("t_one", day1)
ingest_one = time.perf_counter() - t0

db.create_table("t_many", day1.schema, time_column="ts", sort_key=["ts", "symbol"])
n, k = len(day1), 300
chunk = n // k
t0 = time.perf_counter()
for i in range(k):
    db.append("t_many", day1.slice(i * chunk, chunk if i < k - 1 else n - (k - 1) * chunk))
ingest_many = time.perf_counter() - t0

print(f"ingest {n:,} rows: 1 commit {ingest_one:.2f}s vs {k} commits {ingest_many:.2f}s "
      f"({ingest_many / ingest_one:.0f}x slower)")

# %%
agg1 = "SELECT symbol, count(*) AS n, vwap(price, size) AS vw FROM {} GROUP BY symbol"
timings_ms["agg, 1 segment"] = 1e3 * bench(lambda: db.sql(agg1.format("t_one")).to_arrow())
timings_ms["agg, 300 segments"] = 1e3 * bench(lambda: db.sql(agg1.format("t_many")).to_arrow())

t0 = time.perf_counter()
c = db.compact("t_many")
compact_s = time.perf_counter() - t0
timings_ms["agg, compacted"] = 1e3 * bench(lambda: db.sql(agg1.format("t_many")).to_arrow())

print(f"aggregation: {timings_ms['agg, 300 segments']:.1f} ms on 300 segments, "
      f"{timings_ms['agg, 1 segment']:.1f} ms on 1 segment")
print(f"compact() took {compact_s:.2f}s -> segments_total={c['segments_total']}, "
      f"now {timings_ms['agg, compacted']:.1f} ms "
      f"({timings_ms['agg, 300 segments'] / timings_ms['agg, compacted']:.1f}x recovered)")

# %% [markdown]
# ## 5. Cold vs warm: the first query pays the metadata bill
#
# Reopen the database and run the same aggregation twice. The first run reads
# segment footers and builds metadata caches; repeats skip that. On a local
# NVMe the gap is modest - over NFS or object storage it is the difference
# that makes long-lived readers worth keeping around.

# %%
db.close()
db = h5i_db.Database(cu.db_path("prod_perf"))  # fresh handle, cold caches

t0 = time.perf_counter()
db.sql(agg.format("")).to_arrow()
timings_ms["agg cold (1st run)"] = 1e3 * (time.perf_counter() - t0)
timings_ms["agg warm (best of 3)"] = 1e3 * bench(lambda: db.sql(agg.format("")).to_arrow())

print(f"cold: {timings_ms['agg cold (1st run)']:.1f} ms, warm: {timings_ms['agg warm (best of 3)']:.1f} ms "
      f"({timings_ms['agg cold (1st run)'] / timings_ms['agg warm (best of 3)']:.1f}x)")

# %% [markdown]
# ## 6. The scoreboard

# %%
labels = list(timings_ms)[::-1]
values = [timings_ms[k] for k in labels]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(labels, values, height=0.62, color="#4477aa")
ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
ax.set_title(f"Query wall time on ~{len(trades):,} rows (best of 3)")
ax.set_xlabel("milliseconds")
ax.margins(x=0.12)
ax.grid(axis="x", lw=0.3, alpha=0.5)
ax.set_axisbelow(True)
fig.tight_layout()

# %% [markdown]
# ## 7. Resource guards: production etiquette
#
# A shared box (or an agent with SQL access) needs queries that fail fast and
# loud rather than eat the machine. Every `db.sql` accepts `timeout=` seconds,
# `max_rows=`, and `memory_limit=` bytes; violations raise typed exceptions
# (`TimeoutError`, `LimitError`) with a `.code` you can alert on. Catch them,
# don't `except Exception`.

# %%
try:
    db.sql("SELECT * FROM trades", max_rows=100_000)
except h5i_db.LimitError as e:
    print(f"max_rows    -> LimitError   code={e.code} hint={e.hint}")

try:
    db.sql("SELECT * FROM trades", timeout=0.001)
except h5i_db.TimeoutError as e:
    print(f"timeout     -> TimeoutError code={e.code}")

try:
    db.sql(agg.format(""), memory_limit=100_000)  # 100 kB budget: not enough
except h5i_db.LimitError as e:
    print(f"memory_limit-> LimitError   code={e.code}")

# %% [markdown]
# ## 8. What the engine costs you - honest numbers
#
# Per the design docs, h5i-db's generic scan path carries roughly **20%
# overhead versus querying the same Parquet files with raw DataFusion** -
# the price of version resolution, manifest bookkeeping, and the pruning
# machinery. For that you get time travel, atomic commits and time-series
# operators; if a one-off bulk scan is truly latency-critical, you can always
# point another Parquet reader at a table's segment files. The tuning levers
# that actually move production numbers, in the order to reach for them:
#
# 1. put a time predicate on every query that has a natural window;
# 2. name your columns;
# 3. batch appends (one commit per feed batch), `compact()` after bursty
#    ingestion;
# 4. keep reader processes alive so caches stay warm;
# 5. set `timeout`/`max_rows`/`memory_limit` on anything user- or
#    agent-generated.
#
# ## Takeaways
#
# - Time predicates are the cheapest speedup in the system: the manifest
#   prunes whole segments before I/O - ~8x here on materializing scans
#   (aggregations gain less; their cost is hash work, not I/O).
# - Projection matters on wide tick tables: 2 columns ran ~3x faster than
#   `SELECT *` over 3M rows.
# - Commit granularity is a real dial: 300 micro-commits made ingest ~60x
#   slower and queries ~3-4x slower than one batched commit; `compact()`
#   wins the query time back in under a second.
# - Cold-open cost is paid once per process - long-lived readers amortize it.
# - Guards (`timeout`, `max_rows`, `memory_limit`) turn resource blowups into
#   typed, catchable errors - mandatory for shared boxes and agent access.

# %%
db.close()
