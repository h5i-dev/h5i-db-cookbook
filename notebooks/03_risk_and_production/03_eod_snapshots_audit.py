# %% [markdown]
# # EOD snapshots and the audit trail regulators actually ask for
#
# The question that arrives eighteen months later is never "what is the price
# now". It is *"what did your systems know at the close of business on
# 2026-06-02?"*. If your answer involves restoring backup tapes, you have
# already lost the meeting.
#
# In h5i-db an end-of-day cut is a named snapshot: a checksummed pin of the
# table manifest, O(1) to create, addressable forever from SQL as
# `h5i('trades', 'eod-2026-06-02')`.
#
# In this recipe we:
#
# 1. build a small EOD pipeline that appends the day's feed, runs data-quality
#    checks, snapshots and logs,
# 2. answer the regulator's question three ways,
# 3. diff two EOD cuts in SQL,
# 4. close with integrity attestation via `verify(deep=True)` and retention via
#    `vacuum`.

# %%
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star, sql_expr

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_eod"), create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.make_trades` gives three sessions of ticks, one row per print.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | trade timestamp, ascending |
# | `symbol` | `string` | ticker |
# | `price` | `float64` | trade price |
# | `size` | `int64` | shares traded |
# | `exchange` | `string` | reporting venue |
# | `side` | `string` | `B` buyer-initiated, `S` seller-initiated |
#
# We will feed it in one session at a time, as a production loader would.

# %%
all_trades = cu.make_trades(days=3, trades_per_day=20_000).to_pandas()
print(f"{len(all_trades):,} rows x {all_trades.shape[1]} columns, "
      f"{all_trades['ts'].dt.date.nunique()} sessions")
all_trades.head()

# %% [markdown]
# ## 2. Tables: the feed, and our own snapshot log
#
# One honest API note up front. The Python API currently has snapshot
# *creation* only, via `db.snapshot(name, tables=[...], note=...)`. Listing
# snapshots is a CLI feature, `h5i-db snapshot list`.
#
# Production pipelines want the catalog queryable next to the data, so we keep
# our own `snapshot_log` table. Each EOD run appends one row with the name,
# pinned sequence and manifest checksum returned by `snapshot()`.
#
# The log is itself versioned and append-only, which is exactly what an auditor
# wants a log to be.

# %%
trade_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])

log_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("snapshot_name", pa.string()),
        pa.field("table_name", pa.string()),
        pa.field("pinned_sequence", pa.int64()),
        pa.field("manifest_checksum", pa.string()),
        pa.field("note", pa.string()),
    ]
)
db.create_table("snapshot_log", log_schema, time_column="ts")
db.tables()

# %% [markdown]
# ## 3. The EOD pipeline: append, check, snapshot, log
#
# Three simulated production days. Each day runs one atomic `append` of that
# day's feed with a note, then a minimal data-quality gate covering row count,
# price positivity and universe completeness. Only if the gate passes does a
# named snapshot and a log row follow.
#
# The snapshot dict returns, per table, the pinned version and a manifest
# checksum. That is cryptographic evidence of exactly what the name refers to.

# %%
all_trades["day"] = all_trades["ts"].dt.date

def eod_checks(day: str) -> dict:
    row = (
        db.table("trades")
        .filter(col("ts") >= f"{day}T00:00:00Z", col("ts") < f"{day}T23:59:59Z")
        .select(
            rows=count_star(),
            symbols=col("symbol").n_unique(),
            px_min=col("price").min(),
            px_max=col("price").max(),
        )
        .to_pandas()
        .iloc[0]
    )
    return {
        "rows": int(row["rows"]),
        "symbols_ok": int(row["symbols"]) == 3,
        "prices_ok": bool(row["px_min"] > 0),
        "nonempty": int(row["rows"]) > 0,
    }

last_snap = None
for day, chunk in all_trades.groupby("day", sort=True):
    data = pa.Table.from_pandas(
        chunk.drop(columns="day").sort_values(["ts", "symbol"]), schema=trade_schema,
        preserve_index=False,
    )
    commit = db.append("trades", data, note=f"{day} feed")
    checks = eod_checks(str(day))
    assert all(v for k, v in checks.items() if k != "rows"), f"EOD gate failed: {checks}"

    last_snap = db.snapshot(f"eod-{day}", tables=["trades"], note=f"EOD risk cut {day}")
    entry = next(iter(last_snap["entries"].values()))
    db.append("snapshot_log", pa.table({
        "ts": pa.array([pd.Timestamp(last_snap["created_at_ns"], unit="ns", tz="UTC")],
                       type=pa.timestamp("us", tz="UTC")),
        "snapshot_name": [last_snap["name"]],
        "table_name": [entry["table_name"]],
        "pinned_sequence": [entry["sequence"]],
        "manifest_checksum": [entry["manifest_checksum"]],
        "note": [last_snap["note"]],
    }))
    print(f"{day}: appended {data.num_rows:,} rows as v{commit['sequence']}, "
          f"checks {checks}, snapshot 'eod-{day}'")

# %% [markdown]
# Here is the snapshot dict itself: name, creation time, and per-table pins with
# checksums. This is what lands in the log.

# %%
last_snap

# %%
(
    db.table("snapshot_log")
    .select("ts", "snapshot_name", "pinned_sequence",
            checksum=sql_expr("substr(manifest_checksum, 1, 12)"))
    .sort("ts")
    .to_pandas()
)

# %% [markdown]
# ## 4. "What did you know on 2026-06-02?" Three ways to answer
#
# The same read point is addressable three ways.
#
# - By **snapshot name**, which carries the business meaning: the EOD cut.
# - By **version number**, taken from the log's `pinned_sequence`.
# - By **as-of commit time**, meaning whatever was committed before T.
#
# All three are O(1) manifest lookups rather than replays.

# %%
by_name = (
    db.table("trades", snapshot="eod-2026-06-02")
    .select(
        rows=count_star(),
        last_trade=col("ts").max(),
        vwap_all=((col("price") * col("size")).sum() / col("size").sum()).round(2),
    )
    .to_pandas()
)
by_name

# %%
# By version number: the log row tells us which sequence the name pins.
pinned = (
    db.table("snapshot_log")
    .filter(col("snapshot_name") == "eod-2026-06-02")
    .select("pinned_sequence")
    .to_pandas()["pinned_sequence"]
    .iloc[0]
)
by_version = db.read("trades", version=int(pinned))

# By commit wall-clock time: the append's committed_at_ns from versions().
v2 = [v for v in db.versions("trades") if v["sequence"] == pinned][0]
as_of = pd.Timestamp(v2["committed_at_ns"], unit="ns", tz="UTC").isoformat()
by_time = db.read("trades", as_of=as_of)

print(f"by snapshot name: {int(by_name['rows'].iloc[0]):,} rows")
print(f"by version {pinned}:     {by_version.num_rows:,} rows")
print(f"by as_of {as_of}: {by_time.num_rows:,} rows")
assert by_version.num_rows == by_time.num_rows == int(by_name["rows"].iloc[0])

# %% [markdown]
# ## 5. Diff two EOD cuts in one SQL statement
#
# Because snapshots are relations, "what changed between the 2nd and the 3rd"
# is a join rather than an ETL job. Per symbol, it gives rows added and where
# the last print moved.

# %%
def per_symbol(snapshot: str):
    return (
        db.table("trades", snapshot=snapshot)
        .group_by("symbol")
        .agg(n=count_star(), last_px=col("price").last("ts"))
    )


d3, d2 = per_symbol("eod-2026-06-03"), per_symbol("eod-2026-06-02")
n3, n2 = col("n", relation="l"), col("n", relation="r")
px3, px2 = col("last_px", relation="l"), col("last_px", relation="r")

(
    d3.join(d2, on="symbol")
    .select(
        symbol=col("symbol", relation="l"),
        trades_added=n3 - n2,
        close_jun02=px2,
        close_jun03=px3,
        px_chg=(px3 / px2 - 1).round(4),
    )
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# ## 6. Integrity attestation
#
# `verify(deep=True)` re-reads every manifest and segment and checks the
# checksum chain from head to genesis.
#
# Combined with the manifest checksums recorded in `snapshot_log`, this is an
# attestation you can hand over. The EOD cut named in the log resolves to a
# manifest whose checksum matches, and every byte under it verifies.

# %%
report = db.verify("trades", deep=True)
report

# %%
assert not report["problems"], f"integrity check failed: {report['problems']}"
print(f"checked {report['manifests_checked']} manifests, "
      f"{report['segments_checked']} segments, "
      f"{report['bytes_checked']:,} bytes - no problems")

# %% [markdown]
# ## 7. Retention: vacuum respects history and snapshots
#
# `vacuum` reclaims storage that nothing references. Here every segment is still
# reachable, both through the version chain and through the three EOD
# snapshots.
#
# So even an aggressive `grace_seconds=0` run deletes nothing, and the oldest
# EOD cut remains readable afterwards. Snapshots are the retention policy: pin
# what you must keep, vacuum the rest.

# %%
print("dry run:", db.vacuum(apply=False))
print("applied:", db.vacuum(grace_seconds=0, apply=True))
print("oldest EOD cut still readable:",
      f"{len(db.read('trades', snapshot='eod-2026-06-01')):,} rows")

# %% [markdown]
# ## Takeaways
#
# - An EOD cut is `db.snapshot(name, tables=[...], note=...)`. It is O(1), no
#   data is copied, and it stays addressable from SQL forever as
#   `h5i('table', 'name')`.
# - "As known on date X" has three equivalent spellings: snapshot name, version
#   number and `as_of` commit time. All resolve without replay.
# - The Python API does not list snapshots yet, though the CLI does. An
#   append-only `snapshot_log` table closes the gap and gives you a *queryable,
#   versioned* catalog with the pinned sequence and manifest checksum per cut.
# - `verify(deep=True)` plus logged checksums turns "trust us" into an
#   attestation, and `vacuum` cannot touch anything a snapshot still pins.

# %%
db.close()
