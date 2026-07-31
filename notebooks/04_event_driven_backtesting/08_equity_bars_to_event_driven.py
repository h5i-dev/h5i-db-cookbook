# %% [markdown]
# # From a vectorized equity backtest to an event-driven one
#
# Section 02 backtests a monthly momentum rule by multiplying a signal by the
# return that followed. Section 04 has so far replayed prediction markets. The
# recipe between them is the one a desk actually needs: take the equity strategy
# you already believe, run it through the matching engine, and find out what the
# multiplication was hiding.
#
# Three assumptions hide in every vectorized equity backtest. You traded at the
# close you formed the signal on. You paid that close, not the offer. You bought
# 17.4 shares. None of them survive contact with an order book, and two of them
# only ever move the P&L one way.
#
# This recipe reconciles the two backtests to the cent. The gap decomposes into
# timing, spread and fees, with the residual asserted to zero, so the question
# stops being "which number do I trust" and becomes "which of these three do I
# want to argue with".

# %% [markdown]
# ## Terms used here
#
# | term                | meaning |
# | ------------------- | --- |
# | vectorized backtest | P&L computed as signal times subsequent return, with no order book involved |
# | bar                 | one period of open/high/low/close/volume, here one session |
# | order book          | the resting bids and offers an order actually meets |
# | half-spread         | half the gap between bid and offer; what crossing costs per share |
# | target position     | the share count a strategy wants to hold, as opposed to the order that gets there |
# | formation date      | the date whose data produced the signal |
# | reconciliation      | accounting for every cent of difference between two calculations of the same thing |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, col, sql_expr, time_bucket
import cookbook_utils as cu

# %% [markdown]
# ## 1. The data
#
# Real daily bars for ten large caps, cached from Yahoo Finance. One row per
# symbol per session.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | session date |
# | `symbol` | `string` | ticker |
# | `open`, `high`, `low`, `close` | `float64` | session prices |
# | `adj_close` | `float64` | close adjusted for splits and dividends |
# | `volume` | `int64` | shares traded |

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")
print(f"{daily.num_rows:,} rows x {daily.num_columns} columns")
daily.to_pandas().head()

# %% [markdown]
# ## 2. Bar data contains no order book
#
# The engine accepts a canonical `bars` table, and a bar updates the *mark* an
# open position is valued at. It does not create a book, so an order arriving
# against bars alone meets no liquidity and never fills. That is not a gap in
# the engine: a daily bar genuinely does not record what was quoted.
#
# A book therefore has to be assumed, and the assumption belongs in the code
# rather than inside a fill price. `cu.make_equity_market` builds the smallest
# honest one: a two-sided quote `spread_bps` wide around each bar's close, with
# a slice of the bar's volume displayed on each side.

# %%
tape = daily.to_pandas()
tape = tape[tape["ts"] >= pd.Timestamp("2022-01-01", tz="UTC")]
tape = pa.Table.from_pandas(tape, preserve_index=False)
market = cu.make_equity_market(tape, spread_bps=5.0, depth_fraction=0.002)
for name, table in market.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")
market["book_deltas"].to_pandas().head(4)

# %% [markdown]
# One `event_index` is one atomic book event: the bid row and the offer row of
# one symbol at one instant, terminated by `is_last`. The engine reads a whole
# event before any order sees it.
#
# The canonical time column is `ts_init`, nanoseconds and tz-naive, which is why
# these tables look different from the `ts` columns elsewhere in the cookbook.

# %%
db = h5i_db.Database(cu.fresh_db("04_equity_bars_to_event_driven"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="daily bars, 10 large caps")
for name in ("instruments", "book_deltas", "trades"):
    table = market[name]
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="book synthesized from bars at 5bp")
db.snapshot(
    "tape-5bp",
    tables=["prices", "instruments", "book_deltas", "trades"],
    note="Prices plus the 5bp synthetic book that every run here reads",
)
db.tables()

# %% [markdown]
# ## 3. The signal, in the engine
#
# Classic 12-1 momentum on a monthly grid: the return from twelve months ago to
# one month ago, so the most recent month is skipped. The month-end close is a
# `last()` aggregate and the two lags are window functions over the monthly
# frame, so the whole signal is one query.

# %%
monthly = (
    db.table("prices", snapshot="tape-5bp")
    .with_columns(month=time_bucket("1mo", col("ts")))
    .group_by("symbol", "month")
    .agg(close=col("close").last("ts"), formation_ts=col("ts").max())
    .with_columns(
        lag_1=sql_expr("lag(close, 1)").over(partition_by="symbol", order_by="month"),
        lag_12=sql_expr("lag(close, 12)").over(partition_by="symbol", order_by="month"),
    )
    .with_columns(momentum=col("lag_1") / col("lag_12") - 1)
    .filter(col("momentum").is_not_null())
    .sort(["month", "symbol"])
)
signal = monthly.to_pandas()
signal = signal[signal["formation_ts"] >= pd.Timestamp("2022-02-01", tz="UTC")]
print(f"{len(signal):,} symbol-months over {signal['month'].nunique()} rebalances")
signal.head()

# %% [markdown]
# ## 4. Targets, not orders
#
# Selection and sizing are per-date arithmetic over ten names, so they belong in
# Python. Each month the top three names by momentum get an equal share of a
# fixed notional, rounded down to whole shares; everything else is flat.
#
# The rounding happens once, here, and every backtest below trades the same
# integer share counts. A term that appears in all four calculations cannot
# explain the difference between them.
#
# The last two months are dropped. Their orders would be released at or after
# the final record in the tape, and an order the replay never reaches is not a
# trade the vectorized version gets to book either.

# %%
CAPITAL = 300_000.0
TOP_N = 3

rows = []
for month, group in signal.groupby("month", sort=True):
    picks = group.nlargest(TOP_N, "momentum")
    formation = picks["formation_ts"].max()
    for pick in picks.itertuples():
        rows.append(
            {
                "formation_ts": formation,
                "symbol": pick.symbol,
                "shares": float(int(CAPITAL / TOP_N / pick.close)),
            }
        )
picked = pd.DataFrame(rows)
sessions = pd.DatetimeIndex(daily.to_pandas()["ts"].unique()).sort_values()
formation_index = sessions.searchsorted(picked["formation_ts"])
picked = picked[formation_index + 2 < len(sessions)]

grid = pd.MultiIndex.from_product(
    [sorted(picked["formation_ts"].unique()), sorted(picked["symbol"].unique())],
    names=["formation_ts", "symbol"],
)
positions = (
    picked.set_index(["formation_ts", "symbol"])
    .reindex(grid, fill_value=0.0)
    .reset_index()
    .sort_values(["formation_ts", "symbol"])
)
print(f"{len(positions):,} target rows over {positions['formation_ts'].nunique()} rebalances")
positions.head(6)

# %% [markdown]
# `backtest.target_positions` turns a desired holding into the minimum sequence
# of orders that reaches it, tracking each instrument's running position itself.
# A target that repeats produces no order at all.
#
# The stamp is the load-bearing decision. An intent released at an instant meets
# the last book the venue has already processed, so **the stamp chooses the
# price**, and the fill is recorded at the next event. Stamping one microsecond
# after the formation close would therefore trade *at* the formation close --
# the very assumption this recipe set out to drop.
#
# So each order is stamped one microsecond after its own instrument's quote in
# the *following* session: decided on Monday's close, sent while Tuesday's quote
# is the newest one, filled there. Nothing after the formation date enters the
# decision; the stamp only says when the decision reached the market.
#
# `cu.make_equity_market` staggers symbols within a session by a microsecond
# each, so "that instrument's quote" is a single unambiguous instant. Without
# the stagger, an order released into a batch of same-timestamp events would
# meet the new book for whichever symbol the merge happened to reach first.

# %%
book = market["book_deltas"].to_pandas()
book["session"] = book["ts_init"].dt.floor("s").dt.tz_localize("UTC")
quote_at = {(row.session, row.instrument_id): row.ts_init for row in book.itertuples()}
touch_at = {
    (row.session, row.instrument_id, row.side): row.price for row in book.itertuples()
}
depth_at = {
    (row.session, row.instrument_id, row.side): row.size for row in book.itertuples()
}

positions["execution_ts"] = sessions[
    sessions.searchsorted(positions["formation_ts"], side="right")
]
stamps = [
    quote_at[(row.execution_ts, row.symbol)] + dt.timedelta(microseconds=1)
    for row in positions.itertuples()
]
signals = backtest.target_positions(
    stamps,
    positions["shares"].tolist(),
    instrument_id=positions["symbol"].tolist(),
    tag="momentum-12-1",
)
backtest.create_signal_table(db)
db.append("signals", signals, note="12-1 momentum, top 3, whole shares")

orders = signals.to_pandas()
orders["execution_session"] = pd.DatetimeIndex(orders["ts"]).tz_localize("UTC").floor("s")
orders["formation_session"] = sessions[
    sessions.searchsorted(orders["execution_session"]) - 1
]
print(f"{signals.num_rows:,} orders from {len(positions):,} targets")
orders.head()[["ts", "instrument_id", "side", "quantity", "formation_session"]]

# %% [markdown]
# ## 5. The event-driven run
#
# Market data is pinned to the snapshot, fees are charged proportionally, and
# the run writes into its own fork. Nothing about the strategy is restated
# here: the strategy is the table.

# %%
FEE_RATE = 0.0005

report = backtest.run(
    db,
    "momentum-5bp",
    starting_cash=CAPITAL,
    signals="signals",
    snapshot="tape-5bp",
    fee_kind="proportional",
    fee_rate=FEE_RATE,
    equity_interval_nanos=86_400_000_000_000,
)
run_db = db.fork(report["fork"])
fills = run_db.read("bt_fills").to_pandas()
print(f"orders {report['orders']}  fills {report['fills']}  "
      f"commissions {report['commissions']:,.2f}")
print(f"final cash {report['final_cash']:,.2f}")
fills.head()

# %% [markdown]
# Every order filled in full, and that is a property of this book rather than of
# the strategy. The check worth keeping is each order against the depth it
# actually met: the largest here takes a single-digit percentage of the size
# displayed at its own instant. Once an order is a large fraction of that, the
# fill price stops being the touch and the reconciliation below stops holding.
# Recipe 04/10 is where orders start losing that argument.

# %%
consumed = orders.apply(
    lambda order: order["quantity"]
    / depth_at[
        (
            order["execution_session"],
            order["instrument_id"],
            "sell" if order["side"] == "buy" else "buy",
        )
    ],
    axis=1,
)
assert len(fills) == report["orders"], "a partial fill would break the reconciliation"
assert consumed.max() < 1.0, "an order asked for more than the book showed"
print(f"orders filled in full     {len(fills)} of {report['orders']}")
print(f"largest order vs depth    {consumed.max():.1%}")
print(f"median order vs depth     {consumed.median():.1%}")

# %% [markdown]
# ## 6. The same trades, three vectorized ways
#
# Each calculation below trades the identical integer share counts at the
# identical instants. Only the price changes:
#
# 1. **formation close** -- what a vectorized backtest usually assumes;
# 2. **execution close** -- the next session, because that is when the order
#    could actually have been sent;
# 3. **execution touch** -- the bid or the offer, not the mid.
#
# Cash is tracked directly. Buying costs cash, selling returns it, and whatever
# is still held at the end is marked at the final close.

# %%
closes = daily.to_pandas().pivot(index="ts", columns="symbol", values="close").sort_index()
close_at = closes.stack().to_dict()
orders.head(3)[["ts", "instrument_id", "side", "quantity", "execution_session"]]

# %% [markdown]
# Before pricing anything, check that the engine agrees about which book each
# order met. Every fill should carry the touch of its execution session, and be
# *recorded* one session later, because a released intent fills against the last
# processed book and is stamped at the next event.

# %%
executions = fills.sort_values("order_id").reset_index(drop=True)
assert (executions["instrument_id"] == orders["instrument_id"]).all()
assert (executions["side"] == orders["side"]).all()
assert (executions["quantity"] == orders["quantity"]).all()

expected = [
    touch_at[(row.execution_session, row.instrument_id, "sell" if row.side == "buy" else "buy")]
    for row in orders.itertuples()
]
recorded = pd.DatetimeIndex(executions["ts"]).tz_localize("UTC")
assert (executions["price"] - expected).abs().max() < 1e-9, "a fill missed the touch"
assert (recorded > orders["execution_session"]).all(), "a fill was recorded too early"
print(f"all {len(orders)} fills took the touch of their execution session,")
print("and every one was recorded at the following session")


# %%
def ending_equity(price_of) -> float:
    """Ending equity from the order sequence, priced by `price_of`."""
    cash, held = CAPITAL, {}
    for order in orders.itertuples():
        signed = order.quantity if order.side == "buy" else -order.quantity
        cash -= signed * price_of(order)
        held[order.instrument_id] = held.get(order.instrument_id, 0.0) + signed
    return cash + sum(
        quantity * close_at[(sessions[-1], name)] for name, quantity in held.items()
    )


formation_close = ending_equity(
    lambda o: close_at[(o.formation_session, o.instrument_id)]
)
execution_close = ending_equity(
    lambda o: close_at[(o.execution_session, o.instrument_id)]
)
execution_touch = ending_equity(
    lambda o: touch_at[
        (o.execution_session, o.instrument_id, "sell" if o.side == "buy" else "buy")
    ]
)
for label, value in (
    ("formation close", formation_close),
    ("execution close", execution_close),
    ("execution touch", execution_touch),
):
    print(f"{label:<18} {value:>14,.2f}")

# %% [markdown]
# ## 7. Reconciliation
#
# The run's own ending equity is its final cash plus whatever it still holds,
# marked at the same final close. Every step between the vectorized answer and
# that number is a named quantity, and the residual is asserted rather than
# eyeballed.

# %%
holdings = run_db.read("bt_positions").to_pandas()
event_driven = float(report["final_cash"]) + sum(
    row.quantity * close_at[(sessions[-1], row.instrument_id)]
    for row in holdings.itertuples()
)

waterfall = pd.DataFrame(
    [
        {"step": "vectorized (formation close)", "equity": formation_close, "delta": 0.0},
        {"step": "timing: trade the next close", "equity": execution_close,
         "delta": execution_close - formation_close},
        {"step": "spread: pay the touch", "equity": execution_touch,
         "delta": execution_touch - execution_close},
        {"step": "fees: proportional commission",
         "equity": execution_touch - report["commissions"],
         "delta": -report["commissions"]},
        {"step": "event-driven run", "equity": event_driven,
         "delta": event_driven - (execution_touch - report["commissions"])},
    ]
)
residual = waterfall["delta"].iloc[-1]
assert abs(residual) < 0.05, f"unexplained {residual:,.4f}"
waterfall.round(2)

# %% [markdown]
# The residual is zero to the cent, which is the claim worth making: the
# difference between the two backtests is not noise or engine detail, it is
# three decisions, each of which can be defended or changed.
#
# Note the sign of the first step. Trading a day later is not a cost by
# construction, and a momentum signal formed on a close is often chasing a name
# that gives some of it back. Spread and fees are one-directional; timing is
# not, and one combined number cannot tell you which you are exposed to.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
steps = waterfall.iloc[1:-1]
ax.bar(steps["step"], steps["delta"], color=["#4c78a8", "#e45756", "#f58518"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("What the vectorized backtest was hiding")
ax.set_xlabel("Assumption dropped")
ax.set_ylabel("Effect on ending equity")
ax.tick_params(axis="x", labelrotation=10)
fig.tight_layout()

# %% [markdown]
# ## 8. The spread was a guess, so vary it
#
# `spread_bps` was chosen, not measured, so a conclusion that depends on it is a
# conclusion about the guess. Re-ingesting one book per assumption gives the
# sensitivity directly; recipe 04/11 replaces the guess with a number fitted
# from the fills themselves.

# %%
trials = []
for spread_bps in (1.0, 5.0, 20.0, 50.0):
    scenario = cu.make_equity_market(tape, spread_bps=spread_bps, depth_fraction=0.002)
    trial = h5i_db.Database(cu.fresh_db(f"04_equity_spread_{int(spread_bps)}"), create=True)
    for name in ("instruments", "book_deltas", "trades"):
        table = scenario[name]
        trial.create_table(name, table.schema, time_column="ts_init")
        trial.append(name, table)
    trial.snapshot("tape", tables=["instruments", "book_deltas", "trades"])
    backtest.create_signal_table(trial)
    trial.append("signals", signals)
    trial_report = backtest.run(
        trial,
        f"momentum-{int(spread_bps)}bp",
        starting_cash=CAPITAL,
        signals="signals",
        snapshot="tape",
        fee_kind="proportional",
        fee_rate=FEE_RATE,
        equity_interval_nanos=86_400_000_000_000,
    )
    trial_fork = trial.fork(trial_report["fork"])
    still_held = trial_fork.read("bt_positions").to_pandas()
    ending = float(trial_report["final_cash"]) + sum(
        row.quantity * close_at[(sessions[-1], row.instrument_id)]
        for row in still_held.itertuples()
    )
    trials.append(
        {
            "spread_bps": spread_bps,
            "ending_equity": ending,
            "return_pct": 100 * (ending / CAPITAL - 1),
            "commissions": trial_report["commissions"],
        }
    )
    trial_fork.close()
    trial.close()
sensitivity = pd.DataFrame(trials)
sensitivity.round(2)

# %% [markdown]
# At the assumed 5 basis points the spread costs about half the commission bill.
# Moving that assumption to 50 basis points costs several times the entire
# commission schedule, and a strategy's ranking against a benchmark can turn on
# it. This is the number a vectorized backtest never has to state.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(sensitivity["spread_bps"], sensitivity["return_pct"], marker="o", linewidth=1.8)
ax.set_title("Ending return against the assumed spread")
ax.set_xlabel("Assumed spread (basis points)")
ax.set_ylabel("Total return (%)")
fig.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - Bars set marks; they do not make a book. An event-driven equity backtest
#   needs one, and writing the assumption as data makes it arguable.
# - `backtest.target_positions` is the bridge from a research position vector to
#   order intent, and it emits only the orders that change something.
# - The stamp chooses the price: an intent meets the last book already processed
#   and is recorded at the next event. Stamping after the *following* session's
#   quote is what executes at that session's close, with no look-ahead.
# - The gap between a vectorized and an event-driven result decomposes into
#   timing, spread and fees, and the residual should be asserted, not eyeballed.
# - Timing can help or hurt; spread and fees only hurt. One combined number
#   hides which of them you are running.
# - A named snapshot pins prices and the synthetic book together, so every run
#   in this recipe reads the same bytes.

# %%
run_db.close()
db.close()
