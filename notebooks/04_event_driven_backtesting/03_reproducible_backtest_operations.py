# %% [markdown]
# # Operate reproducible backtests
#
# Reproducibility is an operational property, not a random seed. This recipe
# pins a market-data cut, appends late data, proves that pinned runs remain
# unchanged, and inspects the run manifests needed for review.

# %% [markdown]
# Build one 240-second tape, then split it into an approved 180-second cut and
# a late-arriving tail. This models a research database that continues to
# ingest after a run is certified.
#
# | input | rows | role |
# |---|---:|---|
# | `instruments` | 2 | Stable venue and contract metadata |
# | initial `book_deltas` | 360 | Approved L2 cut |
# | initial `trades` | 180 | Approved print cut |
# | late tail | 180 book rows + 60 trades | Subsequent ingestion |

# %%
import datetime as dt

import pandas as pd
import pyarrow.compute as pc

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

full = cu.make_backtest_fixture(steps=240)
base = dt.datetime(2026, 6, 1, 14, 0, 0)
cutoff = base + dt.timedelta(seconds=181)

initial_book = full["book_deltas"].filter(
    pc.less(full["book_deltas"]["ts_init"], cutoff)
)
late_book = full["book_deltas"].filter(
    pc.greater_equal(full["book_deltas"]["ts_init"], cutoff)
)
initial_trades = full["trades"].filter(pc.less(full["trades"]["ts_init"], cutoff))
late_trades = full["trades"].filter(
    pc.greater_equal(full["trades"]["ts_init"], cutoff)
)

print(f"initial book: {initial_book.num_rows:,} rows")
print(f"late book: {late_book.num_rows:,} rows")
initial_book.to_pandas().tail()

# %% [markdown]
# Create and load the approved cut. The table schemas are identical across
# initial and late batches, which lets append preserve one logical history.

# %%
db = h5i_db.Database(
    cu.fresh_db("04_reproducible_backtest_operations"),
    create=True,
)
db.create_table(
    "instruments",
    full["instruments"].schema,
    time_column="ts_init",
)
db.create_table("book_deltas", initial_book.schema, time_column="ts_init")
db.create_table("trades", initial_trades.schema, time_column="ts_init")
db.append("instruments", full["instruments"], note="reference data")
db.append("book_deltas", initial_book, note="approved 180-second book cut")
db.append("trades", initial_trades, note="approved 180-second trade cut")
db.snapshot(
    "approved-cut",
    tables=["instruments", "book_deltas", "trades"],
    note="Input approved by research controls",
)
db.versions("book_deltas")

# %% [markdown]
# The strategy opens inside the approved cut and closes in the late tail. The
# same signals table is used for every run, so only the market-data read point
# changes.
#
# | column | type | meaning |
# |---|---|---|
# | `ts` | `timestamp[ns]` | Order-intent arrival time |
# | `instrument_id` | `string` | Contract identifier |
# | `side` | `string` | Buy to open, sell to close |
# | `quantity` | `float64` | Units requested |
# | `tag` | `string` | Stable audit label |

# %%
signals = backtest.signal_table(
    [
        {
            "ts": base + dt.timedelta(seconds=30),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy",
            "quantity": 50.0,
            "tag": "open-approved",
        },
        {
            "ts": base + dt.timedelta(seconds=210),
            "instrument_id": "RATE-CUT-YES",
            "side": "sell",
            "quantity": 50.0,
            "tag": "close-late",
        },
    ]
)
print(f"{signals.num_rows:,} rows x {signals.num_columns} columns")
signals.to_pandas()

# %% [markdown]
# Store strategy intent after creating the market snapshot. The run fork pins
# the strategy table separately, so the snapshot remains a pure market-data
# approval point.

# %%
backtest.create_signal_table(db)
db.append("signals", signals, note="approved strategy intent")

# %% [markdown]
# Run against the approved cut before late ingestion. Only the opening signal
# is reached because replay ends with the pinned data.

# %%
first = backtest.run(
    db,
    "approved-before-late-data",
    starting_cash=10_000.0,
    signals="signals",
    snapshot="approved-cut",
    equity_interval_nanos=10_000_000_000,
)
first

# %% [markdown]
# Append the late tail as normal ingestion. Existing versions and the named
# snapshot remain readable; no data is copied into the snapshot.

# %%
db.append("book_deltas", late_book, note="late-arriving final minute")
db.append("trades", late_trades, note="late-arriving final minute")
print(db.versions("book_deltas")[-1])

# %% [markdown]
# Re-run the approved snapshot and run once at latest. The pinned result must
# remain identical. The latest run can reach the closing signal and therefore
# represents a different research input.

# %%
pinned_again = backtest.run(
    db,
    "approved-after-late-data",
    starting_cash=10_000.0,
    signals="signals",
    snapshot="approved-cut",
    equity_interval_nanos=10_000_000_000,
)
latest = backtest.run(
    db,
    "latest-after-late-data",
    starting_cash=10_000.0,
    signals="signals",
    equity_interval_nanos=10_000_000_000,
)

comparison = pd.DataFrame(
    [
        {"run": "pinned before append", **first},
        {"run": "pinned after append", **pinned_again},
        {"run": "latest after append", **latest},
    ]
).set_index("run")
comparison[
    [
        "fills",
        "orders",
        "records_processed",
        "final_cash",
        "realized_pnl",
    ]
]

# %% [markdown]
# Assert the evidence reviewers care about. Pinned runs have identical
# economics and event counts. Latest intentionally differs.

# %%
stable_fields = (
    "fills",
    "orders",
    "records_processed",
    "final_cash",
    "realized_pnl",
    "commissions",
)
assert all(first[field] == pinned_again[field] for field in stable_fields)
assert latest["records_processed"] > first["records_processed"]
assert latest["fills"] > first["fills"]
print("Pinned result survived subsequent ingestion unchanged.")

# %% [markdown]
# Coverage gates turn incomplete data into a hard failure. Here the approved
# cut cannot satisfy a window extending through the late tail.

# %%
try:
    backtest.run(
        db,
        "coverage-must-fail",
        starting_cash=10_000.0,
        signals="signals",
        snapshot="approved-cut",
        window=(
            base + dt.timedelta(seconds=1),
            base + dt.timedelta(seconds=240),
        ),
        minimum_coverage=0.95,
    )
except h5i_db.InvalidInputError as error:
    print(f"Rejected as intended: {error}")
else:
    raise AssertionError("the incomplete approved cut passed its coverage gate")

# %% [markdown]
# Each run fork carries a one-row manifest and detailed result tables. Keep
# the manifest, source snapshot name, strategy version, and configuration in
# the review packet. The fill table remains the execution source of truth.

# %%
audit_rows = []
for label, report in (
    ("pinned-before", first),
    ("pinned-after", pinned_again),
    ("latest", latest),
):
    run_db = db.fork(report["fork"])
    manifest = run_db.read("bt_run").to_pandas().iloc[0].to_dict()
    manifest["label"] = label
    manifest["fork"] = report["fork"]
    audit_rows.append(manifest)
    run_db.close()
audit = pd.DataFrame(audit_rows).set_index("label")
audit[
    [
        "run_id",
        "config_digest",
        "records_processed",
        "final_cash",
        "realized_pnl",
        "fork",
    ]
]

# %% [markdown]
# ## Takeaways
#
# - Pin market data before accepting a backtest result.
# - Keep strategy intent versioned separately from the historical data cut.
# - Re-running a named snapshot is stable after later appends.
# - Coverage gates reject truncated windows before they produce plausible metrics.
# - Preserve the run fork, manifest, digest, and fills as one reviewable artifact.

# %%
db.close()
