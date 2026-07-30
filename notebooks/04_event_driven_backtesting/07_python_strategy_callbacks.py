# %% [markdown]
# # Path-dependent Python strategies
#
# Some strategies cannot be written as a table of order intent. A rule that
# enters only once the previous position is confirmed closed, or that waits
# thirty seconds after a fill before acting, depends on what already happened
# rather than on the current row.
#
# That is what path-dependent means, and it needs the strategy to carry state
# between events. The cost is a crossing into Python for every event the
# strategy sees, which is why this recipe closes by naming the cheaper
# boundaries and when to prefer them.
#
# Signals and command tables are the fastest strategy boundary because the
# replay stays entirely in Rust. Some strategies genuinely need state,
# fill-driven decisions, or timers. This recipe uses the opt-in Python callback
# surface without giving strategy code direct access to borrowed engine
# internals.

# %% [markdown]
# ## Terms used here
#
# | term              | meaning |
# | ----------------- | --- |
# | callback          | your Python function, called by the engine when something happens |
# | path-dependent    | a decision that depends on what already happened, not only on current data |
# | stateful          | the strategy carries variables between events |
# | timer             | a callback scheduled for a future instant rather than triggered by data |
# | strategy identity | a stable name for a strategy, so its runs stay comparable across reruns |
# | determinism       | the same inputs producing the same outputs, every time, which reruns verify |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

INSTRUMENT_ID = "RATE-CUT-YES"
MARKET_CUT = "callback-market-cut"
SECOND = 1_000_000_000

fixture = cu.make_backtest_fixture(steps=150, instrument_id=INSTRUMENT_ID)
db = h5i_db.Database(cu.fresh_db("07_python_strategy_callbacks"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="deterministic callback fixture")
db.snapshot(
    MARKET_CUT,
    tables=["instruments", "book_deltas", "trades"],
    note="Approved market-data cut for callback examples",
)

# %% [markdown]
# ## Write a state machine with explicit effects
#
# Callback inputs are ordinary dictionaries. A callback returns `None`, one
# action mapping, or an iterable of mappings. The engine applies those actions
# after the callback, preserving causal event ordering. This strategy:
#
# 1. schedules an entry timer from the first observable market event;
# 2. submits a market buy when that timer fires;
# 3. schedules an exit only after the entry fill is confirmed; and
# 4. submits a reduce-only sell from the exit timer.

# %%
class TimedRoundTrip(backtest.EventStrategy):
    def __init__(self):
        self.entry_scheduled = False
        self.exit_scheduled = False
        self.fill_log = []

    def on_event(self, context, event):
        assert context["now"] == event["ts_init"]
        if not self.entry_scheduled:
            self.entry_scheduled = True
            return {
                "action": "timer",
                "name": "enter",
                "ts": context["now"] + 20 * SECOND,
            }
        return None

    def on_timer(self, context, event):
        if event["name"] == "enter":
            return {
                "action": "submit",
                "client_order_id": "entry",
                "instrument_id": INSTRUMENT_ID,
                "side": "buy",
                "quantity": 25.0,
                "tag": "timed-entry",
            }
        if event["name"] == "exit":
            return {
                "action": "submit",
                "client_order_id": "exit",
                "instrument_id": INSTRUMENT_ID,
                "side": "sell",
                "quantity": 25.0,
                "reduce_only": True,
                "tag": "fill-driven-exit",
            }
        raise ValueError(f"unexpected timer {event['name']!r}")

    def on_fill(self, context, event):
        self.fill_log.append(event)
        if event["tag"] == "timed-entry" and not self.exit_scheduled:
            self.exit_scheduled = True
            return {
                "action": "timer",
                "name": "exit",
                "ts": event["ts"] + 60 * SECOND,
            }
        return None


strategy = TimedRoundTrip()

# %% [markdown]
# `strategy_id` is persisted with the run. In packaged research code,
# `run_strategy` can derive it from class source; an explicit version is often
# preferable in notebooks and production because code review can tie it to a
# release or commit.

# %%
result = backtest.run_strategy(
    db,
    "timed-round-trip",
    strategy,
    strategy_id="cookbook.TimedRoundTrip:v1",
    starting_cash=10_000.0,
    data=backtest.DataConfig(
        snapshot=MARKET_CUT,
        minimum_coverage=0.95,
    ),
    execution=backtest.ExecutionConfig(
        fee_kind="prediction_market",
        fee_rate=0.02,
        latency_nanos=1_000_000,
    ),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=25.0,
        max_open_orders=2,
    ),
    output=backtest.OutputConfig(equity_interval_nanos=5 * SECOND),
    metadata={"purpose": "callback and timer contract demonstration"},
)
result

# %% [markdown]
# Fill callbacks ran on the same strategy object under the GIL. The persisted
# tables remain the source of truth; local state is useful for decisions and
# diagnostics, not as the audit record.

# %%
orders = result.orders.to_pandas()
fills = result.fills.to_pandas()
assert result["fills"] == 2
assert [fill["tag"] for fill in strategy.fill_log] == [
    "timed-entry",
    "fill-driven-exit",
]
assert fills["tag"].tolist() == ["timed-entry", "fill-driven-exit"]
assert result.positions.to_pandas()["quantity"].abs().sum() < 1e-9
orders[["order_id", "side", "quantity", "filled", "status", "tag"]]

# %% [markdown]
# Callback runs are reproducible when the same strategy implementation is
# supplied. Verification creates an isolated rerun, compares metrics and all
# authoritative output tables, then removes the temporary fork.

# %%
verification = result.verify(strategy=TimedRoundTrip())
assert verification["verified"]
verification

# %% [markdown]
# ## When to choose each strategy boundary
#
# | boundary | best for | performance | lifecycle |
# |---|---|---|---|
# | signals | vectorized entries/exits and target positions | native hot loop | submit |
# | commands | quoting and predetermined execution schedules | native hot loop | submit/amend/cancel |
# | Python callbacks | state machines, fill reactions, timers | one GIL crossing per callback | full |
#
# Prefer the simplest boundary that expresses the strategy. Callback
# flexibility is valuable, but it should be an explicit choice rather than an
# accidental tax on every backtest.

# %%
db.close()
