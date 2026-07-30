# %% [markdown]
# # Order lifecycle and account risk
#
# Most backtests model an order as a single event. It is sent and it fills.
# Real orders have a life. They rest, they get repriced as the market moves,
# they lose queue priority when they do, they get cancelled when they go
# stale, and they get rejected outright when they would breach a limit.
#
# Modelling only the fill hides two things that cost real money. Repricing
# sends you to the back of the queue. And a risk limit that lives in your
# notebook is not a risk limit.
#
# A production backtest needs more than timestamped entries. Quotes are
# amended, stale orders are cancelled, and account limits must reject unsafe
# intent before it reaches the simulated venue. This recipe exercises that
# complete lifecycle with stable client order IDs and inspects the resulting
# audit trail.

# %% [markdown]
# ## Terms used here
#
# | term            | meaning |
# | --------------- | --- |
# | order lifecycle | submit, amend, cancel, fill, expire: everything an order does after it is sent |
# | amend           | changing the price or size of a resting order rather than replacing it |
# | client order ID | your own stable identifier for an order, so its whole history joins up |
# | preflight       | a check that rejects unsafe or unsupported intent before it reaches the venue |
# | account limit   | a cap on exposure, order size or cash that the engine enforces natively |
# | audit trail     | the record of what was requested, what was rejected, and why |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import datetime as dt

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

INSTRUMENT_ID = "RATE-CUT-YES"
MARKET_CUT = "lifecycle-market-cut"
SECOND = 1_000_000_000

fixture = cu.make_backtest_fixture(steps=120, instrument_id=INSTRUMENT_ID)
db = h5i_db.Database(cu.fresh_db("06_order_lifecycle_and_risk"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="deterministic lifecycle fixture")
db.snapshot(
    MARKET_CUT,
    tables=["instruments", "book_deltas", "trades"],
    note="Approved market-data cut for lifecycle examples",
)

# %% [markdown]
# ## Declare the lifecycle
#
# `client_order_id` belongs to the strategy, not the engine. Later rows use
# that stable name to address the exact order created by `submit`. Submit
# fields are nullable in the storage schema because cancel rows only need an
# ID; the builder validates the fields required by each action.

# %%
base = dt.datetime(2026, 6, 1, 14, 0, 0)
lifecycle = backtest.command_table(
    [
        {
            "ts": base + dt.timedelta(seconds=10),
            "action": "submit",
            "client_order_id": "yes-quote-001",
            "instrument_id": INSTRUMENT_ID,
            "side": "buy",
            "quantity": 20.0,
            "kind": "limit",
            "limit_price": 0.25,
            "tag": "passive-quote",
        },
        {
            "ts": base + dt.timedelta(seconds=30),
            "action": "amend",
            "client_order_id": "yes-quote-001",
            "quantity": 10.0,
            "limit_price": 0.26,
        },
        {
            "ts": base + dt.timedelta(seconds=60),
            "action": "cancel",
            "client_order_id": "yes-quote-001",
        },
    ]
)
backtest.create_command_table(db, "lifecycle_commands")
db.append(
    "lifecycle_commands",
    lifecycle,
    note="submit, reprice/resize, then cancel one quote",
)
lifecycle.to_pandas()

# %% [markdown]
# The typed configuration captures every material assumption. Preflight checks
# the market pin, schemas, coverage, and the strongest fidelity supported by
# the feed before the expensive replay starts.

# %%
lifecycle_config = backtest.BacktestConfig(
    run_id="lifecycle",
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    data=backtest.DataConfig(
        commands="lifecycle_commands",
        snapshot=MARKET_CUT,
    ),
    execution=backtest.ExecutionConfig(
        fee_kind="prediction_market",
        fee_rate=0.02,
        latency_nanos=2_000_000,
    ),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=50.0,
        max_open_orders=4,
    ),
    output=backtest.OutputConfig(equity_interval_nanos=5 * SECOND),
    metadata={"research_ticket": "PM-142", "owner": "market-making"},
)
inspection = backtest.inspect(db, lifecycle_config)
inspection.to_dict()

# %%
inspection.raise_for_errors()
lifecycle_result = backtest.execute(db, lifecycle_config)
orders = lifecycle_result.orders.to_pandas()
orders[
    [
        "order_id",
        "side",
        "limit_price",
        "quantity",
        "filled",
        "status",
        "reject_reason",
        "tag",
    ]
]

# %% [markdown]
# The quote was intentionally away from the market, so the expected outcome is
# cancellation rather than a fill. `explain()` makes silence inspectable, and
# `verify()` reruns the persisted config and compares every authoritative
# output table.

# %%
assert lifecycle_result["fills"] == 0
assert orders["status"].tolist() == ["cancelled"]
explanation = lifecycle_result.explain()
verification = lifecycle_result.verify()
assert verification["verified"]
explanation

# %% [markdown]
# ## Prove that risk rejects before venue execution
#
# Risk controls are native engine constraints, not notebook-side filters. The
# oversized market order is recorded as rejected, never enters latency or
# matching, and carries a durable reason in `bt_orders`.

# %%
risk_commands = backtest.command_table(
    [
        {
            "ts": base + dt.timedelta(seconds=20),
            "action": "submit",
            "client_order_id": "oversized-entry",
            "instrument_id": INSTRUMENT_ID,
            "side": "buy",
            "quantity": 100.0,
            "tag": "must-reject",
        }
    ]
)
backtest.create_command_table(db, "risk_commands")
db.append("risk_commands", risk_commands, note="risk rejection demonstration")

risk_config = backtest.BacktestConfig(
    run_id="risk-rejection",
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    data=backtest.DataConfig(commands="risk_commands", snapshot=MARKET_CUT),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=50.0,
        max_open_orders=4,
    ),
)
risk_result = backtest.execute(db, risk_config)
risk_order = risk_result.orders.to_pylist()[0]
assert risk_result["fills"] == 0
assert risk_order["status"] == "rejected"
assert "max_order_quantity" in risk_order["reject_reason"]
risk_result.explain()

# %% [markdown]
# ## Takeaways
#
# - Stable client IDs make amend/cancel workflows independent of engine IDs.
# - Amendments follow venue-like queue rules; repricing or increasing size
#   loses priority.
# - Position limits include all live orders, not only already-filled exposure.
# - Rejection reasons are persisted and queryable, so a zero-fill run is
#   diagnosable.
# - Typed configs, preflight, and semantic verification form one reproducible
#   operational contract.

# %%
db.close()
