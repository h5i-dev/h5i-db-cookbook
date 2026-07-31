# %% [markdown]
# # Replaying an account's ledger
#
# Every other recipe in this section asks what a strategy would have done. This
# one asks the strictest question a backtester can be asked: here are the trades
# an account *actually* took, published on a public ledger. Does the engine
# reproduce that portfolio against the historical book?
#
# The answer is usually no, and that is the finding rather than the failure. A
# simulator that forced the fills would reproduce any ledger by construction and
# would have tested nothing. So the ledger is compiled into **intent** -- a limit
# order at the price the account got, immediate-or-cancel, with sells marked
# reduce-only -- and the recorded book is allowed to refuse it.
#
# Where it refuses is the useful output. It is either a fact about your market
# data (the book you have is not the book they traded against) or a fact about
# the ledger (that fill did not happen the way it is reported).

# %% [markdown]
# ## Terms used here
#
# | term            | meaning |
# | --------------- | --- |
# | ledger          | a public record of trades an account executed |
# | intent          | an order as submitted, before any venue decided whether it fills |
# | immediate-or-cancel | fill whatever is available right now and cancel the rest |
# | reduce-only     | an order that may only shrink a position, never open the opposite one |
# | reconciliation  | comparing what was reported against what the replay produced, per market |
# | fill ratio      | the share of ledger quantity the replay managed to reproduce |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, venues
import cookbook_utils as cu

TICK = 0.001

# %% [markdown]
# ## 1. The market data the account traded against
#
# A panel of binary event contracts, with the book, the prints and the
# resolutions. This is the historical record; the ledger below is what somebody
# claims they did inside it.
#
# The fixture updates every market at the same instant, and each market is given
# its own millisecond before anything is ingested. A real feed does not quote
# twelve markets at one nanosecond either, and the stagger removes an ambiguity
# that would otherwise decide the answer: an order released into a batch of
# same-timestamp events meets the *new* book for whichever market the merge
# reached first, and the previous one for all the others.

# %%
raw = cu.make_prediction_markets(n_markets=12, steps=24, seed=5)
offsets = {
    name: pd.Timedelta(milliseconds=index)
    for index, name in enumerate(sorted(set(raw["instruments"].column("instrument_id").to_pylist())))
}


def stagger(table):
    """Give every market its own instant, as a real feed would."""
    frame = table.to_pandas()
    if "instrument_id" not in frame.columns:
        return table
    frame["ts_init"] = frame["ts_init"] + frame["instrument_id"].map(offsets)
    if "ts_event" in frame.columns:
        frame["ts_event"] = frame["ts_init"]
    frame = frame.sort_values("ts_init", kind="stable")
    return pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False)


panel = {name: stagger(table) for name, table in raw.items()}
db = h5i_db.Database(cu.fresh_db("04_replaying_an_account_ledger"), create=True)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel fixture, one instant per market")
db.snapshot("books-v1", tables=list(panel), note="The book the ledger is checked against")
for name, table in panel.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")

specs = venues.polymarket_markets_from_json(cu.polymarket_market_payloads(raw))
print(f"\n{len(specs)} market specs, tokens resolved positionally to outcomes")
panel["book_deltas"].to_pandas().head(4)

# %% [markdown]
# ## 2. A ledger, including the parts that will not replay
#
# A real ledger is downloaded. This one is constructed, so that the
# reconciliation can be checked against a known answer: three kinds of row go in
# and the comparison should find exactly the two kinds that cannot replay.
#
# | kind | what it is | what the book should do |
# | --- | --- | --- |
# | at the touch | a buy at the offer, smaller than the displayed size | fill it |
# | better than the touch | a buy reported below the best bid | refuse: nothing was resting that low |
# | oversized | a buy at the offer, larger than the displayed size | fill part of it |

# %%
book = panel["book_deltas"].to_pandas()
quotes = (
    book.pivot_table(
        index=["ts_init", "instrument_id", "outcome"],
        columns="side",
        values=["price", "size"],
        aggfunc="last",
    )
    .dropna()
    .reset_index()
)
quotes.columns = ["ts_init", "instrument_id", "outcome", "bid", "ask", "bid_size", "ask_size"]
quotes = quotes[quotes["outcome"] == 0].sort_values("ts_init")
print(f"{len(quotes):,} usable two-sided quotes")
quotes.head(3)

# %%
rows, kinds = [], []
for index, quote in enumerate(quotes.iloc[40:100:6].itertuples()):
    kind = ("at the touch", "better than the touch", "oversized")[index % 3]
    # Prices stay on the market's 0.001 grid; an off-grid order is refused
    # before it ever meets the book.
    price = (
        round(float(quote.bid) - TICK, 3)
        if kind == "better than the touch"
        else float(quote.ask)
    )
    quantity = float(quote.ask_size) * (3.0 if kind == "oversized" else 0.4)
    rows.append(
        venues.LedgerRow(
            ts_ns=int(pd.Timestamp(quote.ts_init).value),
            instrument_id=quote.instrument_id,
            outcome=0,
            side="buy",
            quantity=round(quantity, 2),
            price=price,
            trade_id=f"0x{index:04d}",
        )
    )
    kinds.append(kind)

ledger = venues.ledger_table(rows).to_pandas().assign(kind=kinds)
print(f"{len(rows)} ledger rows over {ledger['instrument_id'].nunique()} markets")
ledger.head(6)

# %% [markdown]
# `LedgerRow` refuses the things that would silently invert a position: a price
# outside `(0, 1)` is not a probability, a non-positive quantity is not a fill,
# and a side that is neither buy nor sell is not a trade.

# %%
for description, build in (
    ("price of 1.4", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="buy", quantity=10.0, price=1.4)),
    ("quantity of 0", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="buy", quantity=0.0, price=0.5)),
    ("side 'long'", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="long", quantity=10.0, price=0.5)),
):
    try:
        build()
    except ValueError as error:
        print(f"{description:16} refused: {str(error).split(';')[0][:70]}")

# %% [markdown]
# ## 3. Compiling the ledger into intent
#
# `commands_from_ledger` turns each row into a limit order at the price the
# ledger reports, immediate-or-cancel, stamped a microsecond after the instant
# it claims. Sells become reduce-only, so a replay cannot invent short exposure
# the ledger never showed.
#
# Vendor dictionaries work too: pass the raw rows and the market specs, and
# tokens are resolved to outcomes positionally. Typed rows are used here because
# the reconciliation below wants them anyway.

# %%
commands = venues.commands_from_ledger(rows, specs)
backtest.create_command_table(db, "commands")
db.append("commands", commands, note="one account's published trades, as intent")
frame = commands.to_pandas()
print(f"{commands.num_rows} commands")
frame[["ts", "action", "instrument_id", "side", "quantity", "kind", "limit_price",
       "time_in_force", "reduce_only"]].head(4)

# %% [markdown]
# ## 4. The replay
#
# `DataConfig(commands=...)` replaces the signals table with the command stream.
# Everything else is an ordinary run against the pinned book.

# %%
result = backtest.execute(
    db,
    backtest.BacktestConfig(
        run_id="ledger-replay",
        portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
        data=backtest.DataConfig(commands="commands", snapshot="books-v1"),
        execution=backtest.ExecutionConfig(fee_kind="prediction_market", fee_rate=0.02),
        output=backtest.OutputConfig(equity_interval_nanos=900_000_000_000),
    ),
)
print(f"orders {result['orders']}  fills {result['fills']}")
result.orders.to_pandas()[["order_id", "instrument_id", "side", "quantity", "filled", "status"]].head(6)

# %% [markdown]
# ## 5. Reconciliation, per market
#
# `compare_to_ledger` reports where the replay and the ledger disagree rather
# than a single pass or fail, because *where* the book refused is the finding.

# %%
comparison = venues.compare_to_ledger(result, rows)
print(f"ledger rows        {comparison['ledger_rows']}")
print(f"replay fills       {comparison['replay_fills']}")
print(f"ledger quantity    {comparison['ledger_quantity']:,.2f}")
print(f"replay quantity    {comparison['replay_quantity']:,.2f}")
print(f"fill ratio         {comparison['fill_ratio']:.1%}")
print(f"markets reproduced {comparison['markets_reproduced']} of {len(comparison['markets'])}")
pd.DataFrame(comparison["markets"]).round(3).head(8)

# %% [markdown]
# The rows that failed to replay are the ones that were built to fail. Joining
# the per-row outcome back to the kind of row it was turns the reconciliation
# into a statement about the data rather than a number.

# %%
submitted = result.orders.to_pandas().sort_values("order_id").reset_index(drop=True)
assert (submitted["quantity"].round(2).to_numpy() == ledger["quantity"].round(2).to_numpy()).all(), (
    "orders and ledger rows are no longer in the same order"
)
ledger["replayed"] = submitted["filled"].to_numpy()
by_kind = ledger.groupby("kind").agg(
    rows=("quantity", "size"),
    ledger_quantity=("quantity", "sum"),
    replayed_quantity=("replayed", "sum"),
)
by_kind["ratio"] = by_kind["replayed_quantity"] / by_kind["ledger_quantity"]
by_kind.round(3)

# %% [markdown]
# Read the three rows as three different conversations. "At the touch"
# reproduced, so nothing is wrong. "Better than the touch" did not fill at all:
# on real data that is either hidden liquidity, a venue you are not looking at,
# or a mid-quoted report. "Oversized" filled partially, which says the displayed
# book could not have supported that trade and is a statement about the depth in
# your archive.

# %%
assert by_kind.loc["at the touch", "ratio"] > 0.99
assert by_kind.loc["better than the touch", "ratio"] < 0.01
assert by_kind.loc["oversized", "ratio"] < 1.0
print("the reconciliation found exactly the rows that were built to fail")

# %% [markdown]
# ## 6. Why the orders that did not fill are worth reading
#
# The run's own explanation carries the count of orders that produced no fill,
# which is the number to watch when the ledger is real and nobody planted the
# failures.

# %%
explanation = result.explain()
print(f"orders without fills {explanation.get('orders_without_fills')}")
unfilled = result.orders.to_pandas()
unfilled = unfilled[unfilled["filled"] == 0]
print(f"{len(unfilled)} of {result['orders']} orders never filled")
unfilled[["instrument_id", "side", "quantity", "limit_price", "status"]].head(5)

# %% [markdown]
# ## Takeaways
#
# - A ledger replays as *intent*, never as forced fills. Forcing them would
#   reproduce the ledger by construction and test nothing.
# - `commands_from_ledger` chooses limit, immediate-or-cancel and reduce-only on
#   purpose: each one refuses a way the replay could invent a position.
# - `compare_to_ledger` reports per-market shortfalls, because the useful output
#   is which market disagreed and by how much.
# - A row that will not replay is evidence: hidden liquidity, a missing venue, a
#   mid-quoted report, or an archive without the depth to support the trade.
# - The command stream is an ordinary pinned table, so a reconciliation is as
#   reproducible as any other run in this section.

# %%
db.close()
