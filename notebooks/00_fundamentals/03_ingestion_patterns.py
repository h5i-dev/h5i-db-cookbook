# %% [markdown]
# # Ingestion patterns: five sources, one table
#
# No desk gets its data from one place. The tick feed hands you Arrow batches,
# research notebooks live in pandas or polars, vendors drop Parquet, and some
# legacy process still emails CSV.
#
# h5i-db's ingestion surface is deliberately small. `append` extends the feed,
# `write` replaces the contents, and both accept anything Arrow-shaped.
#
# In this recipe we:
#
# 1. ingest five consecutive sessions from five different sources into one
#    `trades` table,
# 2. contrast `write` and `append` semantics,
# 3. make concurrent loaders safe with optimistic locking,
# 4. batch commits and compact the segments they leave behind.

# %%
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import h5i_db
from h5i_db import col, count_star, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_ingestion"), create=True)

# %% [markdown]
# ## The data
#
# One continuous 11-session tape from `cu.make_trades`, one row per print.
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
tape = cu.make_trades(days=11, trades_per_day=4_000, start="2026-06-01", seed=7)
print(f"{tape.num_rows:,} rows x {tape.num_columns} columns")
tape.to_pandas().head()

# %% [markdown]
# We slice that tape into per-day batches so each "delivery" arrives in time
# order. `append` requires every batch to start at or after the table's stored
# maximum timestamp, which is what feed semantics means in practice. A staging
# directory under `data/dbs` stands in for the vendor drop zone.

# %%
dates = tape["ts"].to_pandas().dt.date
sessions = sorted(dates.unique())
by_day = {d: tape.filter(pa.array((dates == d).to_numpy())) for d in sessions}
print(f"{len(sessions)} sessions, {len(tape):,} trades:", sessions[0], "→", sessions[-1])

staging = Path("data/dbs/00_ingestion_staging")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

# %% [markdown]
# The destination is a single table. Every source below lands in it.

# %%
SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", SCHEMA, time_column="ts", sort_key=["ts", "symbol"])

# %% [markdown]
# ## 1. From a pyarrow Table
#
# The native path, with zero conversion. Every `append` returns the **commit
# dict**: the new version number (`sequence`), plus row and segment totals after
# the commit. Log it in your loaders. It is the receipt that ties a delivery to
# a version.

# %%
commit = db.append("trades", by_day[sessions[0]], note="day 1: arrow feed")
commit

# %% [markdown]
# ## 2. From a pandas DataFrame
#
# `pa.Table.from_pandas` does the heavy lifting. Always pass `schema=` as well:
# depending on your pandas version, datetimes round-trip as nanoseconds rather
# than the table's microseconds, and strict append refuses the mismatch. Passing
# the target schema makes the conversion do the cast.

# %%
df = by_day[sessions[1]].to_pandas()  # pretend this came from research code
commit = db.append(
    "trades",
    pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
    note="day 2: pandas",
)
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 3. From a polars DataFrame
#
# Polars speaks Arrow natively, with one wrinkle. Its `to_arrow()` emits
# `large_string` columns, which strict append rejects as a schema mismatch.
# `.cast(SCHEMA)` fixes that at the metadata level for free. Make it a habit on
# any polars to h5i-db boundary.

# %%
pldf = pl.from_arrow(by_day[sessions[2]])  # pretend this came from a polars pipeline
commit = db.append("trades", pldf.to_arrow().cast(SCHEMA), note="day 3: polars")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 4. From a Parquet file
#
# Parquet preserves types exactly, so a vendor Parquet drop is a read and an
# append. If a vendor's row order is suspect, sort before appending. Strict
# append will tell you either way.

# %%
pq.write_table(by_day[sessions[3]], staging / "vendor_day4.parquet")

commit = db.append("trades", pq.read_table(staging / "vendor_day4.parquet"), note="day 4: parquet drop")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 5. From CSV
#
# CSV keeps values but drops types. Read it naively and `ts` comes back as
# whatever the parser guessed. `ConvertOptions(column_types=...)` pins the time
# column to `timestamp[us, tz=UTC]` at parse time, which gives back an exact
# schema match.

# %%
pacsv.write_csv(by_day[sessions[4]], staging / "legacy_day5.csv")

from_csv = pacsv.read_csv(
    staging / "legacy_day5.csv",
    convert_options=pacsv.ConvertOptions(column_types={"ts": pa.timestamp("us", tz="UTC")}),
)
assert from_csv.schema.equals(by_day[sessions[4]].schema)
commit = db.append("trades", from_csv, note="day 5: legacy csv")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## `write` vs `append`
#
# - **`append`** extends the feed. It is strictly time-ordered and never touches
#   existing rows. Use it for anything that behaves like a tape.
# - **`write`** replaces the table's contents with the data you give it, as a
#   *new version*, with all history kept. Use it for reference data that gets
#   re-stated wholesale: universe membership, symbol mappings, risk limits.
#
# Neither is ever destructive. Old versions stay readable through
# `db.read(..., version=n)` in Python and `h5i('table', n)` in SQL.

# %%
universe_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("symbol", pa.string())]
)
db.create_table("universe", universe_schema, time_column="ts")

asof = pa.scalar(pd.Timestamp("2026-06-01", tz="UTC"), type=pa.timestamp("us", tz="UTC"))
db.write(
    "universe",
    pa.table({"ts": pa.array([asof] * 3), "symbol": pa.array(["AAPL", "MSFT", "NVDA"])}),
    note="June universe",
)
db.write(
    "universe",
    pa.table({"ts": pa.array([asof] * 4), "symbol": pa.array(["AAPL", "MSFT", "NVDA", "AVGO"])}),
    note="June universe, AVGO added",
)
print("head :", db.read("universe")["symbol"].to_pylist())
print("v1   :", db.read("universe", version=1)["symbol"].to_pylist())
[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("universe")]

# %% [markdown]
# ## Optimistic locking with `expected_version`
#
# When two loaders share a table, "append whatever, whenever" silently
# interleaves deliveries. `append(..., expected_version=n)` turns the commit into
# a compare-and-swap: it lands only if the table head is still at version `n`.
# Otherwise you get a `ConflictError`, with `retryable=True` and a hint spelling
# out the recovery. The retry itself is mechanical: re-read the head, re-append.

# %%
day6 = by_day[sessions[5]]
try:
    db.append("trades", day6, expected_version=1)  # stale: head is already at v5
except h5i_db.ConflictError as e:
    print(f"code      {e.code}")
    print(f"retryable {e.retryable}")
    print(f"hint      {e.hint}")

# retry pattern: re-read the head version, then re-append against it
head = db.versions("trades")[-1]["sequence"]
commit = db.append("trades", day6, expected_version=head, note="day 6: CAS append")
print(f"\nretried against v{head} -> committed v{commit['sequence']}")

# %% [markdown]
# ## Batching and compaction
#
# Every commit writes a manifest and at least one segment, so commit *batches*:
# a day, an hour, a few thousand rows. Never one row at a time.
#
# Even day-sized commits accumulate small segments, and query planning touches
# every one of them. The loop below is the normal rhythm: many small append
# commits during the week, then one `compact` to merge the segments. Compaction
# is itself just another commit, holding the same data in fewer segments with
# the full history retained.

# %%
for d in sessions[6:]:
    commit = db.append("trades", by_day[d], note=f"daily load {d}")
    print(f"v{commit['sequence']}: +{len(by_day[d]):>6,} rows "
          f"-> {commit['segments_total']:>2} segments total")

# %%
before = db.versions("trades")[-1]
commit = db.compact("trades")
print(f"compacted: {before['segments']} segments -> {commit['segments_total']}, "
      f"rows unchanged: {commit['rows_total']:,}")

[
    {k: v[k] for k in ("sequence", "op", "rows", "segments", "note") if k in v}
    for v in db.versions("trades")
]

# %%
# One tape, five formats, eleven commits - and a query sees a single clean table.
(
    db.table("trades")
    .group_by(time_bucket("1d", col("ts")).alias("session"))
    .agg(
        trades=count_star(),
        notional_mm=((col("price") * col("size")).sum() / 1e6).round(1),
    )
    .sort("session")
    .to_pandas()
)

# %% [markdown]
# ## Takeaways
#
# - Everything Arrow-shaped appends directly. The boundary gotchas are pandas
#   nanoseconds (`from_pandas(schema=...)`), polars `large_string`
#   (`.cast(schema)`) and CSV's type amnesia (`ConvertOptions`).
# - `append` extends the feed in strict time order. `write` restates the whole
#   contents as a new version. Neither destroys history.
# - The commit dict (`sequence`, `rows_total`, `segments_total`) is your
#   ingestion receipt. Log it.
# - `expected_version` turns append into compare-and-swap. On a retryable
#   `ConflictError`, re-read the head and re-append.
# - Batch your commits, then `compact` after many small appends. Compaction is
#   just another version that merges segments without touching history.

# %%
db.close()
