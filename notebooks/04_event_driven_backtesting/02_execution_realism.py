# %% [markdown]
# # Stress-test execution assumptions
#
# A backtest is fragile when one fill assumption determines the conclusion.
# This recipe runs one signal set through fees, adverse slippage, latency, and
# queue position, then compares the resulting fills and cash.

# %% [markdown]
# The input tape contains 240 L2 snapshots and 240 prints. Prices trend and
# oscillate, so delayed orders receive a measurably different book.
#
# | table | key columns | meaning |
# |---|---|---|
# | `instruments` | `instrument_id`, `tick_size` | Venue constraints |
# | `book_deltas` | `ts_init`, `side`, `price`, `size` | Atomic L2 snapshots |
# | `trades` | `ts_init`, `price`, `size`, `aggressor` | Queue-consuming prints |

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

fixture = cu.make_backtest_fixture(steps=240)
for name, table in fixture.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")
fixture["trades"].to_pandas().head()

# %% [markdown]
# Store and pin the tape once. Every scenario below reads exactly this
# snapshot, so differences come from execution configuration rather than data.

# %%
db = h5i_db.Database(cu.fresh_db("04_execution_realism"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="execution sensitivity fixture")
db.snapshot(
    "execution-input-v1",
    tables=["instruments", "book_deltas", "trades"],
    note="Common tape for execution sensitivity",
)

# %% [markdown]
# The active strategy alternates market buys and sells. Market intent isolates
# fees, slippage, and latency from passive-fill uncertainty.
#
# | column | type | meaning |
# |---|---|---|
# | `ts` | `timestamp[ns]` | Submission time |
# | `side` | `string` | Trade direction |
# | `quantity` | `float64` | Requested units |
# | `kind` | `string` | Market order for this experiment |
# | `tag` | `string` | Scenario-independent order label |

# %%
base = dt.datetime(2026, 6, 1, 14, 0, 0)
active_rows = []
for index, second in enumerate((20, 50, 80, 110, 140, 170)):
    active_rows.append(
        {
            "ts": base + dt.timedelta(seconds=second),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy" if index % 2 == 0 else "sell",
            "quantity": 75.0,
            "tag": f"active-{index + 1}",
        }
    )
active_signals = backtest.signal_table(active_rows)
print(f"{active_signals.num_rows:,} rows x {active_signals.num_columns} columns")
active_signals.to_pandas()

# %% [markdown]
# Store active intent once. Each run receives a unique ID and therefore a
# separate output fork.

# %%
backtest.create_signal_table(db, "active_signals")
db.append("active_signals", active_signals, note="active execution experiment")

# %% [markdown]
# The friction matrix changes one assumption at a time. Prediction-market fees
# use the venue-shaped curve. Proportional fees use maker/taker rates.
# Slippage and queue position are separate modes in the current API.

# %%
scenarios = {
    "frictionless": {},
    "prediction_fee": {"fee_rate": 0.03},
    "proportional_fee": {
        "fee_kind": "proportional",
        "fee_rate": 0.001,
        "maker_rebate": -0.0001,
    },
    "20_tick_slippage": {"slippage_ticks": 20},
    "3_second_latency": {"latency_nanos": 3_000_000_000},
}

rows = []
for name, config in scenarios.items():
    report = backtest.run(
        db,
        f"active-{name}",
        starting_cash=10_000.0,
        signals="active_signals",
        snapshot="execution-input-v1",
        equity_interval_nanos=5_000_000_000,
        **config,
    )
    rows.append(
        {
            "scenario": name,
            "fills": report["fills"],
            "final_cash": report["final_cash"],
            "realized_pnl": report["realized_pnl"],
            "commissions": report["commissions"],
        }
    )
active_comparison = pd.DataFrame(rows).set_index("scenario")
active_comparison

# %% [markdown]
# Convert each scenario to an implementation-shortfall view. The frictionless
# run is the control, not a claim about achievable execution.

# %%
control_cash = active_comparison.loc["frictionless", "final_cash"]
active_comparison["cash_shortfall_vs_control"] = (
    control_cash - active_comparison["final_cash"]
)
active_comparison.sort_values("cash_shortfall_vs_control", ascending=False)

# %% [markdown]
# Passive intent needs a different experiment. The limit is the displayed bid
# at submission time. Queue-aware matching places it behind displayed size and
# requires later sell-aggressor prints to reach it.

# %%
book = fixture["book_deltas"].to_pandas()
arrival = book[(book["event_index"] == 30) & (book["side"] == "buy")].iloc[0]
passive_signals = backtest.signal_table(
    [
        {
            "ts": arrival["ts_init"].to_pydatetime(),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy",
            "quantity": 20.0,
            "kind": "limit",
            "limit_price": float(arrival["price"]),
            "time_in_force": "gtc",
            "tag": "passive-entry",
        }
    ]
)
print(f"{passive_signals.num_rows:,} rows x {passive_signals.num_columns} columns")
passive_signals.to_pandas()

# %% [markdown]
# Compare ordinary book matching with conservative and optimistic queue modes.
# Optimistic mode only changes trades whose aggressor is unknown. This tape
# names every aggressor, so both queue runs should agree.

# %%
backtest.create_signal_table(db, "passive_signals")
db.append("passive_signals", passive_signals, note="passive queue experiment")

queue_rows = []
for name, config in {
    "book_only": {},
    "queue_conservative": {"queue_position": True},
    "queue_optimistic": {
        "queue_position": True,
        "optimistic_queue": True,
    },
}.items():
    report = backtest.run(
        db,
        f"passive-{name}",
        starting_cash=10_000.0,
        signals="passive_signals",
        snapshot="execution-input-v1",
        **config,
    )
    run_db = db.fork(report["fork"])
    fills = run_db.read("bt_fills").to_pandas()
    run_db.close()
    queue_rows.append(
        {
            "scenario": name,
            "fills": report["fills"],
            "fill_time": None if fills.empty else fills.iloc[0]["ts"],
            "fill_price": None if fills.empty else fills.iloc[0]["price"],
            "is_taker": None if fills.empty else fills.iloc[0]["is_taker"],
        }
    )
queue_comparison = pd.DataFrame(queue_rows).set_index("scenario")
queue_comparison

# %% [markdown]
# Plot cash shortfall rather than raw cash. This makes small execution costs
# visible without truncating the portfolio-value axis.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
active_comparison["cash_shortfall_vs_control"].plot.bar(ax=ax, color="#4472C4")
ax.set_title("Execution-model cash shortfall")
ax.set_xlabel("Scenario")
ax.set_ylabel("Cash shortfall versus frictionless")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - Change one execution assumption at a time against a pinned tape.
# - Report implementation shortfall relative to a clearly labelled control.
# - Use market orders to study active frictions and limit orders to study queues.
# - Queue position requires prints with trustworthy aggressor classification.
# - Treat sensitivity across plausible models as model risk, not as error bars.

# %%
db.close()
