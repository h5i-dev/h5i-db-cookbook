# %% [markdown]
# # Your first event-driven backtest
#
# The backtests in section 02 are vectorized. Compute a signal for every date,
# multiply it by the return that followed, and sum. That is the right tool for
# a monthly rebalance, and it quietly assumes you traded at a price nobody
# quoted you.
#
# An event-driven backtest drops that assumption. Orders are submitted at an
# instant, they meet the order book as it was recorded at that instant, and the
# engine decides whether they filled, at what price, and after what fees and
# delay. It costs more to run. It is also the only way to ask whether a
# strategy survives its own execution.
#
# This recipe turns timestamped order intent into an auditable simulation.
# Market data is pinned before the strategy is written. Orders, fills,
# positions, and equity are ordinary versioned tables on an isolated fork.

# %% [markdown]
# ## Terms used here
#
# | term                  | meaning |
# | --------------------- | --- |
# | event-driven backtest | orders meet recorded market data event by event, not a bar price |
# | signal                | here the order intent: what to trade, which way, how much, and at what instant |
# | fill                  | an execution against your order, produced by the engine rather than assumed |
# | position              | the running quantity held in an instrument, with its cost basis |
# | equity curve          | cumulative account value over time |
# | run fork              | an isolated branch of the database a run writes its results into, so runs never collide |
# | pin                   | fixing the run's market data to an exact snapshot, before the strategy is written |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %% [markdown]
# The fixture contains reference data, atomic L2 snapshots, and prints for one
# prediction market. One book row is one price level within an `event_index`.
# `is_last` marks the atomic event boundary.
#
# | table | rows | time column | purpose |
# |---|---:|---|---|
# | `instruments` | 2 | `ts_init` | Venue rules and outcome metadata |
# | `book_deltas` | 360 | `ts_init` | Bid/ask snapshots used for matching |
# | `trades` | 180 | `ts_init` | Prints used by queue-aware fill models |

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pyarrow as pa

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

fixture = cu.make_backtest_fixture(steps=180)
for name, table in fixture.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")
fixture["book_deltas"].to_pandas().head()

# %% [markdown]
# The canonical schemas distinguish event time from arrival time. This example
# makes them equal. Production loaders should retain both when feeds expose
# them, because replay order follows `ts_init`.

# %%
fixture["book_deltas"].schema

# %% [markdown]
# Create one time-indexed table per event type. The named snapshot is the
# immutable market-data cut used by every run in this recipe.

# %%
db = h5i_db.Database(cu.fresh_db("04_first_event_driven_run"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="deterministic cookbook fixture")
db.snapshot(
    "market-cut-2026-06-01",
    tables=["instruments", "book_deltas", "trades"],
    note="Approved input for the first event-driven run",
)
db.tables()

# %% [markdown]
# The signals table is the strategy boundary. Each row states intent, not a
# fill. The engine decides the executable price after applying market data,
# fees, latency, and fill rules.
#
# | column | type | meaning |
# |---|---|---|
# | `ts` | `timestamp[ns]` | Earliest arrival time at the strategy boundary |
# | `instrument_id` | `string` | Instrument to trade |
# | `side` | `string` | `buy` or `sell` |
# | `quantity` | `float64` | Requested units |
# | `kind` | `string` | `market` or `limit` |
# | `tag` | `string` | Stable research label carried into fills |

# %%
base = dt.datetime(2026, 6, 1, 14, 0, 0)
signals = backtest.signal_table(
    [
        {
            "ts": base + dt.timedelta(seconds=20),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy",
            "quantity": 50.0,
            "tag": "open",
        },
        {
            "ts": base + dt.timedelta(seconds=120),
            "instrument_id": "RATE-CUT-YES",
            "side": "sell",
            "quantity": 50.0,
            "tag": "close",
        },
    ]
)
print(f"{signals.num_rows:,} rows x {signals.num_columns} columns")
signals.to_pandas()

# %% [markdown]
# Store the strategy after the market snapshot. Signals are versioned
# independently from the market-data pin, so research created later can replay
# the approved historical cut without pretending it existed at that time.

# %%
backtest.create_signal_table(db)
db.append("signals", signals, note="two-leg demonstration strategy")

# %% [markdown]
# Run with an explicit fee model and equity sampling interval. A run writes
# only to `bt-first-run`; the source database remains unchanged.

# %%
report = backtest.run(
    db,
    "first-run",
    starting_cash=10_000.0,
    signals="signals",
    snapshot="market-cut-2026-06-01",
    fee_rate=0.02,
    equity_interval_nanos=5_000_000_000,
)
report

# %% [markdown]
# The report is a compact run manifest. The fork contains the detailed audit
# trail. `bt_fills` is authoritative; positions can always be rebuilt from it.

# %%
run_db = db.fork(report["fork"])
fills = run_db.read("bt_fills").to_pandas()
orders = run_db.read("bt_orders").to_pandas()
positions = run_db.read("bt_positions").to_pandas()
print(orders[["order_id", "side", "quantity", "filled", "status", "tag"]])
fills

# %% [markdown]
# Validate the contract before analysing performance. Both tagged orders must
# fill, and the closing order must leave no net exposure.

# %%
assert report["fills"] == 2
assert fills["tag"].tolist() == ["open", "close"]
assert positions.empty or abs(positions["quantity"].sum()) < 1e-9
print(f"digest={report['digest']}")
print(f"commissions={report['commissions']:.6f}")
print(f"realized P&L={report['realized_pnl']:.6f}")

# %% [markdown]
# Equity is sampled in simulated time, not per input row. This bounds output
# size on tick data while retaining a useful risk series.

# %%
equity = run_db.read("bt_equity").to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(equity["ts"], equity["equity"], linewidth=1.8)
ax.set_title("Event-driven backtest equity")
ax.set_xlabel("Simulated time")
ax.set_ylabel("Portfolio value")
fig.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - The strategy is a versioned table of order intent.
# - A named snapshot fixes every market-data input to the run.
# - Matching, fees, and portfolio accounting happen inside one deterministic kernel.
# - Run outputs are queryable `bt_*` tables on an isolated fork.
# - The digest and source pin provide the minimum evidence needed to reproduce a result.

# %%
run_db.close()
db.close()
