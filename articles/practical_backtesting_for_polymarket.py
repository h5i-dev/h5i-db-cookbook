# %% [markdown]
# # Practical backtesting for Polymarket, with h5i-db
#
# Prediction markets are the friendliest asset class to reason about and one of
# the least forgiving to backtest. A Polymarket share pays exactly one dollar if
# the event happens and zero if it does not, so the price is a probability, the
# maximum loss is the price you paid, and every profit-and-loss question has an
# arithmetic answer rather than a modelling one. That clarity is exactly what
# makes the failure mode so easy to walk into: because the payoff is obvious,
# people write backtests that assume they bought at the price they saw on the
# screen, and the entire difference between a strategy that works on paper and
# one that works in an account lives in that assumption.
#
# This article builds the whole loop, end to end, and every number in it is
# produced by code you can run. It goes in three parts.
#
# 1. **How a Polymarket price is actually made.** The central limit order book,
#    the price ladder, and what happens mechanically when you send an order into
#    it. We will walk a market order through a real ladder by hand and count the
#    cost of a round trip before writing a single line of strategy code.
# 2. **A twelve-event toy market.** One instrument, a book small enough to print
#    in full, and four strategies that all see exactly the same data and end up
#    in four very different places. The differences are entirely execution, and
#    the toy is small enough that you can check the engine's arithmetic against
#    your own.
# 3. **Real tick-level Polymarket books.** The same machinery on a public capture
#    of real order books with real UMA-verified outcomes, running eleven standard
#    strategies, decomposing where the money went, and validating the result with
#    walk-forward windows and a deflated Sharpe ratio.
#
# The third part has a negative result: on this data, every one of the eleven
# standard rules loses money. That is deliberate. An article that showed you a
# winning strategy on one day of six markets would be teaching you to fool
# yourself, and the machinery you need to *establish* that a rule does not work
# is precisely the machinery you need before you believe one that does.
#
# The database underneath is [h5i-db](https://github.com/h5i-dev/h5i-db), an
# embedded, versioned time-series store with an event-driven backtest engine
# built into it. Nothing here needs a server, an account, or a cluster: it is a
# directory on disk plus a Python import.

# %% [markdown]
# ## Vocabulary, once, up front
#
# Prediction markets borrow their jargon from equities trading, and most of it is
# obvious once someone says it out loud. Every term below is used later in the
# article, and each one is also explained again in context where it first
# matters.
#
# | term | meaning |
# |---|---|
# | **share / contract** | the tradeable unit; one YES share pays \$1.00 if the event happens, \$0.00 otherwise |
# | **YES token / NO token** | the two sides of a binary market, each a separately tradeable asset |
# | **price** | what you pay for one share, between 0 and 1, which is also the market's implied probability |
# | **order book (CLOB)** | the list of everyone's resting buy and sell orders, sorted by price |
# | **bid** | the highest price anyone is currently willing to *pay* for a share |
# | **ask (offer)** | the lowest price anyone is currently willing to *sell* a share for |
# | **spread** | ask minus bid, the gap between the two, and the cost of an instant round trip |
# | **mid** | the midpoint `(bid + ask) / 2`, the usual "the price is X" number |
# | **depth / size** | how many shares are resting at a given price level |
# | **level** | one row of the book: a price plus the size resting at it |
# | **maker** | someone who posts an order and waits for a counterparty |
# | **taker** | someone who crosses the spread and trades immediately against a resting order |
# | **market order** | trade now at whatever the book offers |
# | **limit order** | trade only at this price or better, and wait if necessary |
# | **fill** | an execution; the actual price and quantity you got, as opposed to what you asked for |
# | **slippage** | the difference between the price you decided on and the price you got |
# | **resolution / settlement** | the market's outcome being finalised, after which shares pay \$1 or \$0 |
# | **round trip** | one buy and the sell that closes it, which is where costs are paid |
# | **realized P&L** | profit from positions you actually closed |
# | **settlement P&L** | profit from positions still open when the market resolved |

# %%
import datetime as dt
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest, quant, venues

print(f"h5i-db {h5i_db.__version__}")

# %% [markdown]
# ---
#
# # Part 1 — How a Polymarket price is made
#
# ## What you are actually buying
#
# Polymarket runs binary event markets. Take a market like *"Will NYC record
# measurable rain on 10 March 2026?"*. Behind it there are two tokens, YES and
# NO, and each one is an ordinary tradeable asset with its own order book. When
# the event resolves, one token is worth exactly \$1.00 and the other exactly
# \$0.00, forever.
#
# That gives every price an immediate interpretation. If YES trades at 0.62, the
# market is saying the event has a 62% chance, and buying one share at 0.62 costs
# you 62 cents to win a dollar. Two consequences follow directly and are worth
# internalising before anything else:
#
# **Your maximum loss is the price you paid.** There is no leverage, no margin
# call, and no way to lose more than you put in. A 100-share position at 0.62
# risks \$62 and can make \$38.
#
# **YES and NO must sum to one.** A protocol operation lets anyone deposit \$1
# and *mint* one YES plus one NO share, or hand back one of each and redeem \$1.
# So if YES ever trades at 0.62 while NO trades at 0.35, you can buy both for
# \$0.97, redeem the pair for \$1.00, and pocket three cents with no view on the
# weather at all. This is the parity constraint, and in practice it is enforced
# by exactly the people looking for that three cents.
#
# The parity constraint is why the two books stay consistent, and it is also why
# a serious backtest wants *both* books. We will come back to this in Part 3,
# where the capture we use carries only the YES side and the parity trade is
# therefore untestable — a limit of the data, not of the method.

# %% [markdown]
# ## The ladder
#
# Polymarket is a **central limit order book**, or CLOB — the same design the
# NYSE and every crypto exchange use, rather than the automated market maker that
# most on-chain trading uses. There is no pricing formula anywhere in it. The
# price is simply whatever two people last agreed on, and the book is the
# standing record of what everyone is currently willing to do.
#
# The book has two sides. **Bids** are resting buy orders, sorted best (highest)
# first, because a buyer paying more is more attractive to a seller. **Asks** are
# resting sell orders, sorted best (lowest) first. Every resting order carries a
# price and a size, and the pair is called a **level**.
#
# Printed as a ladder, with asks above bids and the gap between them in the
# middle, one instant of a real book looks like this:
#
# ```
#            price     size
#   ASK      0.4900  x  520      <- third-best offer
#   ASK      0.4880  x  310
#   ASK      0.4866  x   80      <- best ask (the touch): cheapest share on sale
#   ------------------------------  spread = 0.0100
#   BID      0.4766  x   80      <- best bid (the touch): highest price anyone pays
#   BID      0.4700  x   50      <- your order would rest here
#   BID      0.4650  x 1200
# ```
#
# Three things are worth reading off this picture straight away.
#
# **Nothing trades in the gap.** The best ask is 0.4866 and the best bid is
# 0.4766, so right now there is no price both a buyer and a seller accept. The
# one-cent gap is the **spread**, and it is the single most important number in
# this article. Everyone quotes the market as "about 0.48" — the **mid**, halfway
# between — but 0.48 is a price at which precisely nothing happens.
#
# **You choose which side of the spread to be on.** You can *take*: send a market
# buy, cross to 0.4866, and own shares within a second. Or you can *make*: post a
# limit buy at 0.4700, join the queue at that level, and wait for a seller to come
# to you. Taking costs money and is certain; making saves money and is not.
#
# **Depth is finite and it is small.** There are only 80 shares available at
# 0.4866. Ask for 200 and you get 80 at 0.4866, then the next 120 at 0.4880 and
# above, and your average price is worse than the number you saw on the screen.
# This is **slippage**, and on prediction markets it bites early, because event
# books are thin compared with equities.

# %% [markdown]
# ## Walking an order through the book, by hand
#
# Before delegating anything to an engine, it is worth doing the arithmetic once
# manually, because it is the arithmetic the engine will do and you want to be
# able to check it. Here is that ladder as a table.
#
# | column | type | meaning |
# |---|---|---|
# | `side` | `string` | `bid` = resting buy order, `ask` = resting sell order |
# | `price` | `float64` | the price of this level, as a probability between 0 and 1 |
# | `size` | `float64` | shares resting at this level, i.e. how much you can trade there |

# %%
LADDER = pd.DataFrame(
    [
        {"side": "ask", "price": 0.4900, "size": 520.0},
        {"side": "ask", "price": 0.4880, "size": 310.0},
        {"side": "ask", "price": 0.4866, "size": 80.0},
        {"side": "bid", "price": 0.4766, "size": 80.0},
        {"side": "bid", "price": 0.4700, "size": 50.0},
        {"side": "bid", "price": 0.4650, "size": 1200.0},
    ]
)
print(f"{len(LADDER)} rows x {LADDER.shape[1]} columns")
LADDER.head(6)

# %% [markdown]
# A market buy consumes ask levels from the best price upward until it has the
# quantity it asked for. The function below is the whole matching rule for a
# taker order, and there is genuinely nothing more to it.

# %%
def walk_the_book(ladder: pd.DataFrame, side: str, quantity: float) -> pd.DataFrame:
    """Consume levels best-price-first and return the fills a taker would get."""
    # A buyer consumes asks cheapest first; a seller consumes bids dearest first.
    levels = ladder[ladder.side == ("ask" if side == "buy" else "bid")]
    levels = levels.sort_values("price", ascending=(side == "buy"))
    remaining, fills = quantity, []
    for level in levels.itertuples():
        if remaining <= 0:
            break
        taken = min(remaining, level.size)
        fills.append({"price": level.price, "quantity": taken, "cost": taken * level.price})
        remaining -= taken
    return pd.DataFrame(fills)


best_ask = LADDER[LADDER.side == "ask"].price.min()
best_bid = LADDER[LADDER.side == "bid"].price.max()
mid = (best_bid + best_ask) / 2
print(f"best bid {best_bid:.4f}   best ask {best_ask:.4f}   "
      f"mid {mid:.4f}   spread {best_ask - best_bid:.4f}")

for size in (80.0, 200.0, 500.0, 1000.0):
    fills = walk_the_book(LADDER, "buy", size)
    got = fills.quantity.sum()
    average = fills.cost.sum() / got
    print(f"\nmarket buy {size:>6.0f} shares -> {len(fills)} level(s), filled {got:.0f}, "
          f"average price {average:.4f}, slippage vs mid {(average - mid) * 100:+.2f} cents")
    print(fills.to_string(index=False))

# %% [markdown]
# An 80-share order fills entirely at the touch and pays half the spread relative
# to the mid. A 200-share order exhausts the touch and reaches one level deeper,
# so its average price is worse. A 500-share order walks all three levels. A
# 1,000-share order asks for more than the 910 shares displayed on the whole ask
# side, and gets what is there and no more — which is exactly what the
# backtest engine does with it later, rather than inventing liquidity that was
# never in the book.
#
# Now the number that governs everything downstream: the cost of a **round trip**,
# meaning one buy and the sell that closes it. If you buy at the ask and later
# sell at the bid with the book unchanged, you have paid the full spread.

# %%
QUANTITY = 100.0
entry = walk_the_book(LADDER, "buy", QUANTITY)
exit_ = walk_the_book(LADDER, "sell", QUANTITY)
entry_price = entry.cost.sum() / entry.quantity.sum()
exit_price = exit_.cost.sum() / exit_.quantity.sum()
round_trip = pd.DataFrame(
    [
        {"leg": "buy (take the ask)", "price": entry_price, "cash": -entry.cost.sum()},
        {"leg": "sell (hit the bid)", "price": exit_price, "cash": exit_.cost.sum()},
    ]
)
round_trip.loc[len(round_trip)] = {
    "leg": "net", "price": exit_price - entry_price, "cash": round_trip.cash.sum()
}
print(f"a {QUANTITY:.0f}-share round trip with the book unchanged:")
print(round_trip.round(4).to_string(index=False))
print(f"\ncost as a fraction of the ~${entry.cost.sum():.0f} you put at risk: "
      f"{-round_trip.cash.iloc[-1] / entry.cost.sum() * 100:.2f}%")
print(f"the mid must move {(entry_price - exit_price) * 100:.2f} cents in your favour "
      f"before this trade breaks even")

# %% [markdown]
# That is the hurdle, and it is the reason most prediction-market strategies fail
# in a way that has nothing to do with forecasting skill. The mid did not move at
# all, yet a hundred-share round trip lost about 2% of the capital committed. A
# rule that trades ten times a day needs to be right by more than a cent, ten
# times a day, just to stay level.

# %% [markdown]
# ## Fees, and why they are shaped like a parabola
#
# Polymarket's headline trading fee has historically been zero, which is unusual
# and genuinely favourable. Everything in Part 3 is therefore run at a zero fee
# by default, so the costs you see there are pure spread. It is still worth
# understanding the fee model, because it is the standard on the venues that do
# charge — Kalshi most prominently — and because h5i-db implements it and we use
# it as a sensitivity.
#
# The event-contract convention is a fee proportional to `p * (1 - p)`:
#
# ```
# fee = rate * quantity * price * (1 - price)
# ```
#
# The shape is not arbitrary. `p * (1 - p)` is the variance of a Bernoulli
# outcome, so the fee is largest where the contract is genuinely uncertain and
# smallest at the tails, which is roughly proportional to the risk the venue is
# intermediating. It peaks at 0.50 and falls to nearly nothing near 0.01 or 0.99.

# %%
FEE_RATE = 0.07  # Kalshi's standard rate, used here as the sensitivity case
prices = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
fee_table = pd.DataFrame(
    [
        {
            "price": price,
            "fee_per_100_shares": FEE_RATE * 100 * price * (1 - price),
            "fee_in_cents_per_share": FEE_RATE * price * (1 - price) * 100,
            "round_trip_fee_cents": 2 * FEE_RATE * price * (1 - price) * 100,
        }
        for price in prices
    ]
)
print(f"quadratic fee at rate {FEE_RATE}:")
print(fee_table.round(3).to_string(index=False))

# %% [markdown]
# Read the last column against the spread cost we just computed. At a price near
# 0.50 the round-trip fee is about 3.5 cents per share, which on this book is
# larger than the one-cent spread — so on a fee-charging venue the fee, not the
# spread, is the dominant cost. On Polymarket at a zero fee, the spread is the
# whole story. Either way the arithmetic is the same: add up what a round trip
# costs, and require the strategy to beat it.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
grid = [index / 100 for index in range(1, 100)]
axes[0].plot(grid, [FEE_RATE * p * (1 - p) * 100 for p in grid], color="#c0392b", lw=1.6)
axes[0].set_title(f"Quadratic fee curve (rate {FEE_RATE}), one leg")
axes[0].set_xlabel("price / implied probability")
axes[0].set_ylabel("cents per share")
axes[1].barh(
    [f"{row.side} {row.price:.4f}" for row in LADDER.itertuples()],
    [row.size for row in LADDER.itertuples()],
    color=["#c0392b" if row.side == "ask" else "#2c7fb8" for row in LADDER.itertuples()],
)
axes[1].invert_yaxis()
axes[1].set_title("The ladder: asks (red) above, bids (blue) below")
axes[1].set_xlabel("shares resting at this level")
fig.tight_layout()

# %% [markdown]
# ---
#
# # Part 2 — Four strategies, one book
#
# Part 1 established what a trade costs. Part 2 makes the cost visible by running
# four strategies over a market small enough to print in its entirety, so that
# every fill the engine produces can be checked against the ladder it came from.
#
# The market is invented, deliberately. Fifteen book updates five minutes apart,
# one instrument, three price levels on each side. YES resolves true. The mid
# drifts from 0.51 up to 0.73 with three pullbacks in it, which is the shape that
# separates strategies: a trend follower gets whipsawed by the pullbacks and a
# buy-and-hold does not.

# %% [markdown]
# ## The canonical shape of market data
#
# h5i-db's backtest engine reads a small set of canonical tables, and the one that
# carries the book is `book_deltas`. Every row is one price level of one book
# update; rows sharing an `event_index` form a single atomic event, and the last
# row of the event is flagged with `is_last`. That structure is what lets the
# replay loop know when a book is fully applied and safe to trade against.
#
# | column | type | meaning |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | when your recorder received the update; this is what replay is ordered by |
# | `ts_event` | `timestamp[ns]` | when the venue says it happened; keep both, they differ |
# | `instrument_id` | `string` | the market |
# | `outcome` | `uint16` | which token: 0 = YES, 1 = NO |
# | `action` | `string` | `snapshot` for a full book, `delta` for an incremental change |
# | `side` | `string` | `buy` is a bid, `sell` is an ask |
# | `price` | `float64` | the level's price, as a probability |
# | `size` | `float64` | shares resting at that level |
# | `event_index` | `int64` | groups the rows of one atomic book update |
# | `is_last` | `bool` | true on the final row of an event |
# | `source_vendor` | `string` | provenance, so a mixed-source table stays auditable |

# %%
START = pd.Timestamp("2026-03-09T14:00:00Z")
STEP = pd.Timedelta(minutes=5)
MARKET = "RAIN-NYC-MAR10"
TICK = 0.01

# (best bid, best ask, size at the touch on each side)
TOUCH_PATH = [
    (0.50, 0.52, 800.0, 600.0),
    (0.47, 0.49, 700.0, 500.0),
    (0.44, 0.46, 900.0, 400.0),   # the dip: a patient bid at 0.46 gets hit here
    (0.46, 0.48, 500.0, 700.0),
    (0.51, 0.53, 600.0, 800.0),
    (0.49, 0.51, 700.0, 500.0),   # first pullback
    (0.55, 0.57, 900.0, 400.0),
    (0.58, 0.60, 600.0, 700.0),
    (0.54, 0.56, 800.0, 500.0),   # second pullback
    (0.61, 0.63, 900.0, 600.0),
    (0.65, 0.67, 700.0, 500.0),
    (0.64, 0.66, 600.0, 700.0),
    (0.70, 0.72, 800.0, 500.0),
    (0.71, 0.73, 900.0, 600.0),
    (0.72, 0.74, 850.0, 550.0),
]
DEPTH_LEVELS = 3

rows: dict[str, list] = {name: [] for name in cu.BOOK_DELTAS_SCHEMA.names}
stamps = []
for event_index, (bid, ask, bid_size, ask_size) in enumerate(TOUCH_PATH):
    ts = (START + event_index * STEP).to_pydatetime()
    stamps.append(ts)
    # Deeper levels sit one tick further out with half again as much size, which
    # is a stylised but reasonable shape for an event book.
    levels = []
    for depth in range(DEPTH_LEVELS):
        levels.append(("buy", round(bid - depth * TICK, 4), bid_size * (1.5 ** depth)))
        levels.append(("sell", round(ask + depth * TICK, 4), ask_size * (1.5 ** depth)))
    for level_index, (side, price, size) in enumerate(levels):
        rows["ts_init"].append(ts)
        rows["ts_event"].append(ts)
        rows["instrument_id"].append(MARKET)
        rows["outcome"].append(0)  # 0 = YES
        rows["action"].append("snapshot")
        rows["side"].append(side)
        rows["price"].append(price)
        rows["size"].append(size)
        rows["event_index"].append(event_index + 1)
        rows["is_last"].append(level_index == len(levels) - 1)
        rows["source_vendor"].append("handmade-toy")

toy_book = pa.table(rows, schema=cu.BOOK_DELTAS_SCHEMA)
print(f"{toy_book.num_rows:,} rows x {toy_book.num_columns} columns "
      f"({len(TOUCH_PATH)} events x {DEPTH_LEVELS * 2} levels)")
toy_book.to_pandas().head()

# %% [markdown]
# Printed as the ladder from Part 1, the third book update — the dip — looks like
# this. This is the same data as the rows above, just drawn the way a trader
# reads it.

# %%
def print_ladder(book: pa.Table, event_index: int) -> None:
    frame = book.to_pandas().query("event_index == @event_index")
    when = frame.ts_init.iloc[0]
    asks = frame[frame.side == "sell"].sort_values("price", ascending=False)
    bids = frame[frame.side == "buy"].sort_values("price", ascending=False)
    print(f"{MARKET}  YES book at {when:%H:%M:%S}\n")
    for row in asks.itertuples():
        print(f"      ASK   {row.price:.4f}  x {row.size:7.1f}")
    print(f"      {'-' * 30}   spread {asks.price.min() - bids.price.max():.4f}")
    for row in bids.itertuples():
        print(f"      BID   {row.price:.4f}  x {row.size:7.1f}")


print_ladder(toy_book, 3)

# %% [markdown]
# ## Two more tables: what the instrument is, and how it ended
#
# The book alone does not say what happens at the end. Two more canonical tables
# carry that: `instruments` describes the contract and `resolutions` records the
# outcome. `venues.MarketSpec` writes both from one declaration, which matters
# because the pairing of outcome labels to token identifiers is positional and
# getting it backwards would attribute every fill to the wrong side.
#
# Two fields on the spec do real work later:
#
# - `expiration_ns` is when the contract stops trading. Orders arriving after it
#   are rejected rather than filled, which is the correct behaviour and one you
#   want enforced rather than remembered.
# - `settlement_observable_ns` is when the outcome became *knowable*. The engine
#   refuses to settle a position unless the replay actually reached that instant,
#   so a backtest cannot quietly mark a position to an outcome that had not been
#   published yet. This is the single most common source of accidental lookahead
#   in prediction-market research and it is worth having a machine enforce it.

# %%
last_ns = int(pd.Timestamp(stamps[-1]).value)
toy_spec = venues.MarketSpec(
    instrument_id=MARKET,
    venue="polymarket",
    outcome_labels=("YES", "NO"),
    tokens=("token-yes-toy", "token-no-toy"),
    tick_size=TICK,
    lot_size=1.0,
    expiration_ns=last_ns,
    settlement_observable_ns=last_ns,
    winner_outcome=0,  # index into outcome_labels: YES wins
    metadata={"question": "Will NYC record measurable rain on 2026-03-10?"},
)

db = h5i_db.Database(cu.fresh_db("article_polymarket_toy"), create=True)
written = venues.write_markets(db, [toy_spec], note="toy market definition")
db.create_table("book_deltas", toy_book.schema, time_column="ts_init")
db.append("book_deltas", toy_book, note="14 book updates, 3 levels a side")
db.snapshot("toy-v1", tables=["instruments", "book_deltas", "resolutions"],
            note="the pinned input for every toy run")
print(written)
print("\ninstruments (one row per tradeable outcome):")
print(db.read("instruments").to_pandas().to_string(index=False))
print("\nresolutions (kind = winner | split | void):")
print(db.read("resolutions").to_pandas().to_string(index=False))

# %% [markdown]
# `db.snapshot(...)` is the reason the four runs below are comparable. It pins a
# named, immutable view of the input tables, so every strategy provably reads the
# same bytes and the only thing varying between runs is the strategy itself. If
# you later append more data, the snapshot still resolves to what it always did.
#
# ## The quote panel
#
# Strategies do not read `book_deltas` directly. They read a **quote panel**: one
# row per instrument per instant, carrying the best price and displayed size on
# each side. Collapsing a book to its touch is a `row_number()` window — rank
# each side by price, best first, and keep rank one.
#
# The `CASE` in the `ORDER BY` is the only subtle part, and it encodes the
# asymmetry from Part 1: the best bid is the *highest* buy price, while the best
# ask is the *lowest* sell price. Getting that backwards quietly hands every
# strategy the worst price in the book instead of the best one, which is a bug
# that produces plausible-looking output rather than an error.
#
# | column | type | meaning |
# |---|---|---|
# | `ts` | `timestamp[ns]` | the instant this book was current |
# | `instrument_id` | `string` | the market |
# | `bid` / `ask` | `float64` | best prices on each side |
# | `bid_size` / `ask_size` | `float64` | displayed depth at the touch |

# %%
TOP_OF_BOOK = """
WITH ranked AS (
    SELECT instrument_id, ts_init AS ts, side, price, size,
           row_number() OVER (
               PARTITION BY instrument_id, ts_init, side
               ORDER BY CASE WHEN side = 'buy' THEN -price ELSE price END
           ) AS depth_rank
    FROM h5i('book_deltas', 'toy-v1')
    WHERE outcome = 0
)
SELECT instrument_id, ts,
       max(CASE WHEN side = 'buy'  THEN price END) AS bid,
       max(CASE WHEN side = 'sell' THEN price END) AS ask,
       max(CASE WHEN side = 'buy'  THEN size  END) AS bid_size,
       max(CASE WHEN side = 'sell' THEN size  END) AS ask_size
FROM ranked
WHERE depth_rank = 1
GROUP BY instrument_id, ts
ORDER BY instrument_id, ts
"""
toy_panel = db.sql(TOP_OF_BOOK).to_pandas()
toy_panel["mid"] = (toy_panel.bid + toy_panel.ask) / 2
print(f"{len(toy_panel)} rows x {toy_panel.shape[1]} columns")
toy_panel.head()

# %% [markdown]
# `h5i('book_deltas', 'toy-v1')` in the `FROM` clause is how a query reads a
# pinned snapshot rather than the live table, so this panel is derived from
# exactly the bytes the backtests will replay.
#
# h5i-db also ships `backtest.quote_panel`, which is the convenience version of
# this query and what Part 3 uses. It assumes one level per side, which is true
# of the real capture there because the ingest truncates to top of book, and is
# not true of this toy book — hence the explicit query here.

# %% [markdown]
# ## The one timing rule that matters
#
# Before the strategies, the single most important mechanical fact about this
# engine, because it silently removes the most common bug in homemade backtests.
#
# **An order stamped after a book update does not trade against that update. It
# trades against the next one.**
#
# The reasoning is that market data reaches the venue before the strategy sees it,
# and strategy commands are queued rather than executed inside the callback that
# produced them. So an intent decided from the 14:00 book is released into a queue
# and meets the book the venue processes next, at 14:05. You never get to trade at
# the price that caused you to trade — which is exactly the constraint reality
# imposes and exactly the one a naive `df.shift(-1)` backtest forgets.
#
# The demonstration below stamps the same buy at three different instants inside
# the same five-minute gap and shows all three receiving the same fill.

# %%
MICRO = dt.timedelta(microseconds=1)


def run_toy(name: str, intents: list[dict], **execution) -> backtest.BacktestResult:
    """Append a signals table and replay it against the pinned toy snapshot."""
    table = backtest.signal_table(intents)
    signal_table = f"signals_{name}"
    db.create_table(signal_table, table.schema, time_column="ts")
    db.append(signal_table, table)
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=name,
            data=backtest.DataConfig(signals=signal_table, snapshot="toy-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
            execution=backtest.ExecutionConfig(**execution),
            output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
        ),
    )


timing = []
probes = {
    "one microsecond after 14:00": stamps[0] + MICRO,
    "midway, at 14:02:30": stamps[0] + dt.timedelta(minutes=2, seconds=30),
    "one microsecond before 14:05": stamps[1] - MICRO,
}
for index, (label, when) in enumerate(probes.items()):
    result = run_toy(
        f"timing{index}",
        [{"ts": when, "instrument_id": MARKET, "outcome": 0, "side": "buy", "quantity": 100.0}],
    )
    fill = result.fills.to_pandas().iloc[0]
    timing.append({"decided at": label, "filled at": fill.ts, "price": fill.price})
print(f"the 14:00 book quoted {toy_panel.ask.iloc[0]:.2f} on the ask, "
      f"the 14:05 book quoted {toy_panel.ask.iloc[1]:.2f}\n")
print(pd.DataFrame(timing).to_string(index=False))

# %% [markdown]
# All three fill at 0.49, the ask of the *following* book, and all three are
# recorded at 14:05. The engine will not let you buy at the 0.52 you were looking
# at when you decided. Every signal generator in h5i-db stamps its orders one
# microsecond after the quote they were derived from for exactly this reason.

# %% [markdown]
# ## Strategy 1 — cross the spread on momentum
#
# The first strategy is the most common thing anyone writes: buy when the price
# has been going up, sell when it turns down, and use market orders so the trade
# actually happens. h5i-db ships a pack of reference implementations of standard
# rules, and `threshold_momentum` is precisely this — buy after the mid rises by
# `threshold` over `lookback` samples, exit when it falls back.
#
# The generator returns a `SignalPlan`, which is a table of order intents plus the
# parameters that produced it. Strategies as *data* rather than as callbacks is a
# deliberate design choice here: a signals table has a content hash, so a run can
# be identified, reproduced and compared, which a Python closure cannot.
#
# | column | type | meaning |
# |---|---|---|
# | `ts` | `timestamp[ns]` | when the order is released |
# | `instrument_id` | `string` | the market |
# | `outcome` | `uint16` | 0 = YES |
# | `side` | `string` | `buy` or `sell` |
# | `quantity` | `float64` | shares requested, not necessarily shares filled |
# | `kind` | `string` | `market` or `limit` |
# | `limit_price` | `float64` | the worst acceptable price, for limit orders |
# | `time_in_force` | `string` | `ioc` fills now or cancels; `gtc` rests until filled |
# | `post_only` | `bool` | refuse to execute as a taker, i.e. insist on being a maker |

# %%
momentum = backtest.STRATEGIES["threshold_momentum"](
    toy_panel, lookback=1, threshold=0.01, quantity=100.0
)
print(f"strategy: {momentum.strategy}   parameters: {dict(momentum.parameters)}")
print(f"{momentum.num_signals} signals\n")
momentum.signals.to_pandas().head(8)

# %% [markdown]
# ## Strategies 2 to 4 — the same market, three other ways to be in it
#
# **Buy and hold.** One market buy at the start, then nothing. No exit, so the
# position is still open when the market resolves and it settles at \$1.00 a
# share. This is the strategy with the fewest round trips and therefore the
# lowest cost, and its P&L is entirely about the forecast rather than the
# execution.
#
# **Patient maker.** A limit buy at 0.46 posted at the start with
# `time_in_force="gtc"` and `post_only=True`. It does not cross the spread; it
# waits at 0.46 for the market to come to it, which it does during the dip at
# 14:10. `post_only` is what makes the fill a maker fill: the engine marks it
# `is_taker=False`, so on a fee-charging venue it would pay the maker rate or earn
# a rebate rather than paying the taker fee.
#
# **Delayed taker.** Strategy 1's signals, but with ten minutes of latency
# configured, meaning every order reaches the venue ten minutes after the decision
# that produced it. Latency is an execution setting rather than a strategy
# setting, so this is the identical signals table run under a different
# `ExecutionConfig` — which is precisely the comparison you want to be able to
# make cheaply.

# %%
db.create_table("signals_momentum", momentum.signals.schema, time_column="ts")
db.append("signals_momentum", momentum.signals)


def execute_toy(run_id: str, signals: str, **execution) -> backtest.BacktestResult:
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=run_id,
            data=backtest.DataConfig(signals=signals, snapshot="toy-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
            execution=backtest.ExecutionConfig(**execution),
            output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
            metadata=momentum.to_metadata() if signals == "signals_momentum" else {},
        ),
    )


runs = {
    "1. taker momentum": execute_toy("toy-momentum", "signals_momentum"),
    "2. buy and hold": run_toy(
        "hold",
        [{"ts": stamps[0] + MICRO, "instrument_id": MARKET, "outcome": 0,
          "side": "buy", "quantity": 100.0, "tag": "entry"}],
    ),
    "3. patient maker": run_toy(
        "maker",
        [{"ts": stamps[0] + MICRO, "instrument_id": MARKET, "outcome": 0,
          "side": "buy", "quantity": 100.0, "kind": "limit", "limit_price": 0.46,
          "time_in_force": "gtc", "post_only": True, "tag": "rest-at-46"}],
    ),
    "4. taker, 10 min late": execute_toy(
        "toy-momentum-late", "signals_momentum", latency_nanos=10 * 60 * 1_000_000_000
    ),
}

def summarise(label: str, result: backtest.BacktestResult) -> dict:
    """Read the run's own result tables rather than trusting a summary dict.

    `realized_pnl` covers positions the strategy closed itself; `settlement_pnl`
    covers positions still open when the market resolved. A strategy that never
    exits earns all of its money in the second column, so both are needed before
    two strategies can be compared at all.
    """
    positions = result.positions.to_pandas()
    orders = result.orders.to_pandas()
    realized = float(result.summary()["realized_pnl"])
    settlement = float(positions.settlement_pnl.fillna(0).sum()) if len(positions) else 0.0
    return {
        "strategy": label,
        "orders": len(orders),
        "fills": len(result.fills.to_pandas()),
        "realized_pnl": realized,
        "settlement_pnl": settlement,
        "total_pnl": realized + settlement,
    }


scoreboard = pd.DataFrame([summarise(label, result) for label, result in runs.items()])
print("100 shares a trade, zero fees, identical book and identical outcome:\n")
print(scoreboard.round(2).to_string(index=False))

# %% [markdown]
# Four strategies, one book, one outcome, and a spread of results that has almost
# nothing to do with forecasting. All four were right about the direction — the
# mid went from 0.51 to 0.73 and the market then resolved YES — and the best one
# finishes with roughly double the profit of the worst.
#
# Note also which column each strategy earned its money in. The two that never
# sold have zero `realized_pnl` and all of their profit in `settlement_pnl`,
# because they were still holding when the market resolved. Comparing strategies
# on realized P&L alone would have scored buy-and-hold at exactly zero.
#
# The fills explain the rest completely. Here is every execution the engine
# produced, for every run.

# %%
for label, result in runs.items():
    fills = result.fills.to_pandas()
    if fills.empty:
        print(f"\n{label}: no fills\n")
        continue
    view = fills[["ts", "side", "price", "quantity", "commission", "is_taker", "tag"]]
    print(f"\n{label}  ({len(fills)} fills)")
    print(view.to_string(index=False))

# %% [markdown]
# Read the entry prices against each other. The buy-and-hold entered once at 0.49.
# The patient maker entered at 0.46, three cents better, because it refused to
# cross the spread and was rewarded for waiting. The momentum rule entered and
# exited three times, each time buying at the ask *after* the move that triggered
# it — 0.53, then 0.60, then 0.67, plus a fourth entry at 0.73 it was still
# holding at the close — so it chased the same rally up the ladder and sat flat
# through a third of it. Its round trips were individually profitable and
# collectively almost pointless: it was right every single time and still finished
# behind the strategy that did nothing at all.
#
# The delayed run is the clearest of the four. Identical signals, ten minutes
# later, in a market moving a couple of cents per five-minute update, and the
# round trips flip from a small gain to a small loss. Its last two orders are also
# worth looking at: one arrived after the contract stopped trading and was
# rejected, and one was still in flight when the replay ended. Neither is an
# error. Both are what happens to a strategy whose decisions are stale.
#
# The comparison worth making explicit is entry price against final value, because
# it isolates execution from forecasting.

# %%
comparison = []
for label, result in runs.items():
    fills = result.fills.to_pandas()
    if fills.empty:
        continue
    buys = fills[fills.side == "buy"]
    comparison.append(
        {
            "strategy": label,
            "buys": len(buys),
            "avg_buy_price": float((buys.price * buys.quantity).sum() / buys.quantity.sum()),
            "shares_bought": float(buys.quantity.sum()),
            "shares_sold": float(fills[fills.side == "sell"].quantity.sum()),
        }
    )
detail = pd.DataFrame(comparison)
detail["cents_worse_than_best_entry"] = (detail.avg_buy_price - detail.avg_buy_price.min()) * 100
print(detail.round(3).to_string(index=False))
print("\nevery run bought the same asset, which settled at $1.00 per share")

print("\norders that did not fill:")
for label, result in runs.items():
    orders = result.orders.to_pandas()
    unfilled = orders[orders.status != "filled"]
    for row in unfilled.itertuples():
        print(f"  {label:<22} {row.ts}  {row.side:<4} {row.status}"
              f"{'  ' + str(row.reject_reason)[:60] if isinstance(row.reject_reason, str) else ''}")

# %% [markdown]
# ## What the maker gave up
#
# The patient maker looks unambiguously best in the table above, and on this path
# it was. That conclusion is not general, and the reason is worth stating plainly
# because it is where maker strategies actually fail: a resting order that the
# market never reaches simply does not trade.
#
# Re-run the same strategy with the limit two cents lower, at 0.44, in a book
# whose ask never fell below 0.46.

# %%
missed = run_toy(
    "maker-too-low",
    [{"ts": stamps[0] + MICRO, "instrument_id": MARKET, "outcome": 0, "side": "buy",
      "quantity": 100.0, "kind": "limit", "limit_price": 0.44,
      "time_in_force": "gtc", "post_only": True, "tag": "rest-at-44"}],
)
print(missed.orders.to_pandas()[
    ["ts", "side", "kind", "limit_price", "quantity", "filled", "status", "reject_reason"]
].to_string(index=False))
print(f"\nfills: {len(missed.fills.to_pandas())}, "
      f"total P&L: {missed.summary()['realized_pnl']:.2f}")
for warning in missed.summary()["warnings"] or []:
    print(f"warning: {warning}")

# %% [markdown]
# Two cents of greed cost the entire position, and the engine reports it as a
# cancelled order with a stated reason rather than as a fill that never happened.
# That asymmetry — takers pay a known cost with certainty, makers pay an unknown
# cost with probability — is the real trade-off, and it is why the maker/taker
# choice deserves to be a variable in your research rather than an assumption
# baked into it.
#
# One honest caveat on the maker result. This book is a periodic *snapshot*, not
# a stream of every individual change, so the engine can tell you that the ask
# reached your price but cannot tell you whether you were at the front of the
# queue when it did. It fills you optimistically. Modelling queue position
# properly needs every book delta between snapshots, and h5i-db will refuse a
# `queue_position=True` configuration on snapshot data rather than return a
# plausible-looking number — which is the behaviour you want from a backtester.

# %%
try:
    backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id="queue-claim",
            data=backtest.DataConfig(signals="signals_momentum", snapshot="toy-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
            execution=backtest.ExecutionConfig(queue_position=True),
        ),
    )
except ValueError as error:
    print(f"refused: {str(error)[:220]}")

# %% [markdown]
# ## Adding the fee back
#
# Finally, the same momentum signals under three fee regimes. The signals table
# does not change; only the `ExecutionConfig` does. This is the sensitivity that
# tells you whether a rule is fragile to a cost assumption, and it takes three
# lines to run.
#
# Worth knowing while reading the output: re-executing a configuration that has
# already run against the same pin returns the *recorded* result rather than
# replaying it. The zero-fee row below is therefore the identical run from the
# scoreboard, served from the trial ledger, which is a property of the pinning
# rather than a coincidence.

# %%
fee_rows = []
for label, execution in {
    "Polymarket (no fee)": {},
    "quadratic fee, rate 0.02": {"fee_kind": "kalshi", "fee_rate": 0.02},
    "quadratic fee, rate 0.07": {"fee_kind": "kalshi", "fee_rate": FEE_RATE},
}.items():
    result = execute_toy(f"toy-fee-{len(fee_rows)}", "signals_momentum", **execution)
    row = summarise(label, result)
    row["commissions"] = float(result.fills.to_pandas().commission.sum())
    fee_rows.append(row)
fees = pd.DataFrame(fee_rows).rename(columns={"strategy": "fee regime"})
print(fees[["fee regime", "fills", "commissions", "realized_pnl", "total_pnl"]]
      .round(2).to_string(index=False))

# %% [markdown]
# ## Part 2, in one line
#
# Same book, same outcome, same direction called correctly by all four, and the
# best result is about double the worst — on nothing but how the orders were sent.
# A backtest that does not model entry price, exit price, latency and fill
# probability is not measuring the strategy. It is measuring the price path, which
# you already knew.
#
# It is worth being clear about what this toy does *not* establish. Buy-and-hold
# wins here because this particular path went up and resolved YES, and a path that
# went the other way would have made it the worst of the four. What generalises is
# not the ranking; it is the size of the gap that execution alone opens up between
# strategies with identical views.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(toy_panel.ts, toy_panel.ask, color="#c0392b", lw=1.2, label="ask")
axes[0].plot(toy_panel.ts, toy_panel.bid, color="#2c7fb8", lw=1.2, label="bid")
axes[0].fill_between(toy_panel.ts, toy_panel.bid, toy_panel.ask, color="grey", alpha=0.25)
for label, result in runs.items():
    fills = result.fills.to_pandas()
    if fills.empty:
        continue
    axes[0].scatter(fills.ts, fills.price, s=42, zorder=3, label=label)
axes[0].set_title("Where each strategy actually traded")
axes[0].set_ylabel("price / implied probability")
axes[0].tick_params(axis="x", labelrotation=30)
axes[0].legend(fontsize=7)
axes[1].barh(scoreboard.strategy, scoreboard.total_pnl,
             color=["#2c7fb8" if value >= 0 else "#c0392b" for value in scoreboard.total_pnl])
axes[1].axvline(0.0, color="black", lw=0.8)
axes[1].invert_yaxis()
axes[1].set_title("Total P&L on an identical book and outcome")
axes[1].set_xlabel("dollars")
fig.tight_layout()

# %%
db.close()

# %% [markdown]
# ---
#
# # Part 3 — The same machinery on real Polymarket books
#
# The toy proves the plumbing and nothing about the world. Part 3 runs the same
# pipeline over real tick-level Polymarket order books, real markets, and real
# UMA-verified resolutions, and asks the only question that matters: **do any of
# the standard rules survive real costs?**
#
# ## The data
#
# A bounded, non-commercial sample of tick-level Polymarket books, published on
# Kaggle. It is not redistributed with this article; the cell below prints the
# download commands if the files are absent. The dataset is CC BY-NC 4.0, so
# review the licence before using anything derived from it commercially.
#
# If you would rather work with live data than a capture, the cookbook ships
# `scripts/fetch_polymarket.py`, which pulls market definitions and current books
# straight from Polymarket's public read-only endpoints into the same layout the
# ingest path reads. Be aware of what that path can and cannot give you: the API
# serves *current* books for live markets and historical *mid points* for any
# market, but not historical depth. A study of execution quality needs an archive
# for that reason, which is why this part uses a capture.

# %%
CACHE = Path("data/cache/kaggle-polymarket")
missing = cu.kaggle_missing_files(CACHE)
if missing:
    print("This section needs the Kaggle sample. Missing:", missing)
    print(f"\nDataset: {cu.KAGGLE_POLYMARKET_DATASET}  (licence {cu.KAGGLE_POLYMARKET_LICENSE})")
    for line in cu.kaggle_download_commands(CACHE):
        print("  " + line)
    raise SystemExit("download the sample, then re-run")

snapshots_path = CACHE / "snapshots_2026-03-09.parquet"
targets_path = CACHE / "market_targets.parquet"
print(f"snapshots  {pq.ParquetFile(snapshots_path).metadata.num_rows:,} rows "
      f"({snapshots_path.stat().st_size / 1e6:.0f} MB)")
print(f"targets    {pq.ParquetFile(targets_path).metadata.num_rows:,} markets")
print(f"licence    {cu.KAGGLE_POLYMARKET_LICENSE} (non-commercial)")

# %% [markdown]
# ### What one raw row looks like
#
# The capture is a websocket recording written straight to Parquet, so the
# interesting content is a JSON string in a single column rather than a set of
# typed fields.
#
# | column | type | meaning |
# |---|---|---|
# | `timestamp_received` | `int64` | milliseconds since epoch, when the recorder got the message |
# | `timestamp_created_at` | `int64` | milliseconds since epoch, the venue's own stamp |
# | `market_id` | `string` | the market's condition id, and the join key to the labels |
# | `update_type` | `string` | `book_snapshot` for a full book |
# | `data` | `string` | the raw websocket payload, as JSON |

# %%
raw = pq.ParquetFile(snapshots_path)
first_batch = next(raw.iter_batches(batch_size=3)).to_pandas()
print(f"{raw.metadata.num_rows:,} rows x {raw.metadata.num_columns} columns")
print(first_batch.drop(columns=["data"]).to_string())

event = json.loads(first_batch.data.iloc[0])
print(f"\none payload carries: {sorted(event)}")
print(f"  side       {event['side']}   (which token's book this is)")
print(f"  token_id   {str(event['token_id'])[:28]}...")
print(f"  best_bid   {event['best_bid']}   best_ask {event['best_ask']}")
print(f"  depth      {len(event['bids'])} bid levels / {len(event['asks'])} ask levels")
print(f"  first bid  {event['bids'][0]}  (price and size arrive as strings)")

# %% [markdown]
# ### Three limits, all properties of the capture
#
# Every study inherits the shape of its data, and it is better to state the
# constraints up front than to discover them in a result.
#
# **It covers one day.** Most of these markets resolve later, so almost nothing
# can be held to resolution inside the window. The study is therefore intraday and
# `realized_pnl` is the honest scoreboard, not settlement. How many markets
# actually resolve inside the window gets computed below rather than assumed.
#
# **It carries the YES side only.** The NO token's book is not in the capture, so
# the YES-plus-NO parity trade from Part 1 is not testable here, and any rule must
# trade the YES book. We check this against the data rather than trusting it.
#
# **It carries full depth**, which on an active market runs to tens or hundreds
# of levels a side. A rule that trades at the touch reads two of them, so we
# truncate to top of book at ingest — with the truncation opt-in and reported,
# because a silently shallower book is a different book.
#
# One more caveat about the sample rather than the capture: the six markets
# selected below include three questions about the same Federal Reserve meeting.
# Their outcomes are mechanically related, so the effective number of independent
# bets here is smaller than six, and no result on this panel should be read as
# having six markets' worth of evidence behind it.

# %% [markdown]
# ### The labels
#
# `market_targets.parquet` carries one row per market including the
# UMA-verified outcome. UMA is the decentralised oracle protocol Polymarket
# resolves through, and taking the outcome from the dataset's own label file
# rather than inferring it from the final price is the difference between a label
# and a guess.
#
# | column | type | meaning |
# |---|---|---|
# | `condition_id` | `string` | the market, and the join key to the book capture |
# | `question` | `string` | what was actually being traded |
# | `end_date` | `string` | ISO-8601, when the market closes |
# | `closed` | `bool` | whether trading has stopped |
# | `volume` / `liquidity` | `float64` | vendor-reported activity, useful for picking markets |
# | `clob_token_id_yes` / `_no` | `string` | the per-outcome token identifiers |
# | `target` | `int8` | 1 = YES wins, 0 = NO wins, null = not yet resolved |

# %%
targets = pq.read_table(targets_path).to_pandas()
print(f"{len(targets):,} rows x {targets.shape[1]} columns")
targets[["question", "end_date", "closed", "volume", "target"]].head()

# %% [markdown]
# ### Choosing markets
#
# We need markets that are both present in the book capture and resolved in the
# labels, and among those we take the six with the most book activity, because a
# strategy needs events to act on. That selection is not neutral and it is stated
# here rather than buried: picking the most active markets biases toward markets
# close to resolving, which are easier to forecast. It affects the calibration
# result at the end, and it is called out again there.

# %%
ids = pq.read_table(snapshots_path, columns=["market_id"]).column("market_id").combine_chunks()
coverage = pd.DataFrame(pc.value_counts(ids).to_pylist()).rename(
    columns={"values": "condition_id", "counts": "snapshot_rows"}
)
labelled = coverage.merge(targets, on="condition_id")
pool = labelled.query("target.notna()")
print(f"markets in the capture:     {len(coverage):,}")
print(f"also present in the labels: {len(labelled):,}")
print(f"and resolved:               {len(pool):,}")

chosen = pool.sort_values("snapshot_rows", ascending=False).head(6).reset_index(drop=True)
print("\nthe six we will trade:")
print(chosen[["question", "snapshot_rows", "end_date", "target"]].to_string(index=False))

# %% [markdown]
# ### Ingest
#
# Two declarations do the whole job. `MarketSpec` describes each contract, exactly
# as in Part 2 but now with values read from the real label file. `ArchiveLayout`
# describes the *file shape* — where the timestamp is, what unit it is in, which
# column holds the payload, which JSON field names the token — so that reading a
# new vendor's format is a literal rather than a new code path.
#
# `max_levels=1` is the top-of-book truncation, and the ingest report says how
# much it dropped.

# %%
specs = [
    venues.MarketSpec(
        instrument_id=row.condition_id,
        venue="polymarket",
        outcome_labels=("YES", "NO"),
        tokens=(row.clob_token_id_yes, row.clob_token_id_no),
        tick_size=0.001,
        lot_size=1.0,
        expiration_ns=int(pd.Timestamp(row.end_date).value),
        settlement_observable_ns=int(pd.Timestamp(row.end_date).value),
        winner_outcome=0 if int(row.target) == 1 else 1,
        metadata={"question": row.question},
    )
    for row in chosen.itertuples()
]

real = h5i_db.Database(cu.fresh_db("article_polymarket_real"), create=True)
markets = venues.write_markets(real, specs, note="UMA-verified labels")
print(f"instruments {markets.tables['instruments'].rows} rows "
      f"({len(specs)} markets x 2 outcomes)")
print(f"resolutions {markets.tables['resolutions'].rows} rows")
for spec in specs:
    print(f"  {spec.outcome_labels[spec.winner_outcome]:>3} won: {spec.metadata['question'][:62]}")

# %%
layout = venues.ArchiveLayout(
    name="kaggle-polymarket-top",
    timestamp_column="timestamp_received",
    timestamp_unit="ms",
    instrument_column="market_id",
    event_type_column="update_type",
    snapshot_events=("book_snapshot",),
    levels=venues.LevelLayout(style="payload"),
    payload_column="data",
    payload_token_field="token_id",
    payload_outcome_field="side",
    outcome_labels=("YES", "NO"),
    max_levels=1,
)
started = time.time()
ingest = venues.ingest_archive(
    real, files=[snapshots_path], markets=specs, layout=layout,
    note="tick capture, truncated to top of book",
)
print(f"ingested in {time.time() - started:.1f}s")
print(ingest)
pin = real.snapshot("real-v1", tables=["instruments", "book_deltas", "resolutions"],
                    note="real Polymarket books, top of book")
print(f"pinned as {pin['name']}, checksum {pin['checksum'][:16]}")

# %% [markdown]
# The canonical `book_deltas` table now holds real books in exactly the schema
# the toy used, which is the point of a canonical layer: everything downstream is
# identical whether the input was six hand-typed events or a websocket capture.

# %%
real_book = real.read("book_deltas")
print(f"{real_book.num_rows:,} rows x {real_book.num_columns} columns")
real_book.to_pandas().head()

# %% [markdown]
# The YES-only claim, checked against the data rather than asserted. Only outcome
# 0 has any book at all, so the NO side of these markets is untradeable here.
# The second query verifies that every book event is well formed: exactly one
# outcome per event, and exactly one row flagged `is_last`.

# %%
sides = real.sql(
    """
    SELECT outcome,
           count(DISTINCT event_index)   AS events,
           count(DISTINCT instrument_id) AS markets,
           min(price) AS lowest_price, max(price) AS highest_price
    FROM book_deltas GROUP BY outcome ORDER BY outcome
    """
).to_pandas()
print(sides.to_string(index=False))

malformed = real.sql(
    """
    SELECT count(*) AS malformed FROM (
        SELECT event_index FROM book_deltas GROUP BY event_index
        HAVING count(DISTINCT outcome) > 1
            OR sum(CASE WHEN is_last THEN 1 ELSE 0 END) <> 1)
    """
).to_pandas()
print(f"\nmalformed book events: {int(malformed.malformed.iloc[0])}")

# %% [markdown]
# ### The panel, and what real spreads look like
#
# Same call as the toy, and the same shape out. What differs is the content:
# these spreads are not uniform, and a market trading at 0.002 and one trading at
# 0.47 are entirely different instruments for execution purposes even though both
# are one contract paying one dollar.

# %%
panel = backtest.quote_panel(real, snapshot="real-v1")
panel["spread"] = panel.ask - panel.bid
panel["mid"] = (panel.bid + panel.ask) / 2
print(f"{len(panel):,} rows x {panel.shape[1]} columns, "
      f"{panel.instrument_id.nunique()} markets")
panel.head()

# %%
print(panel[["bid", "ask", "spread", "mid"]].describe().round(4).to_string())
print("\nhalf-spread as a share of the mid, by price level:")
levels = panel.assign(level=lambda frame: frame.mid.round(1)).groupby("level").agg(
    quotes=("mid", "size"),
    mean_mid=("mid", "mean"),
    half_spread=("spread", lambda series: series.mean() / 2),
)
# Divide by the mean mid inside each bucket, not by the bucket's label: the
# 0.0 bucket holds prices near a tenth of a cent and dividing by zero would
# print `inf` where the most interesting number is.
levels["half_spread_pct_of_mid"] = levels.half_spread / levels.mean_mid * 100
print(levels.round(4).to_string())

# %% [markdown]
# The last column is the round-trip hurdle from Part 1, computed on real books and
# expressed as a percentage of the price paid. The shape is the important part:
# the cost is under half a percent in the crowded middle of the book and around
# the near-certain end, and it explodes on the cheap contracts, where crossing the
# spread costs a double-digit percentage of the position before anything has
# happened. A longshot strategy therefore has to be right about the *direction*
# far more often than it has to be right about the *probability*.

# %% [markdown]
# ## Running eleven standard rules
#
# h5i-db ships eleven reference strategies — momentum in three flavours, mean
# reversion in three, a microstructure imbalance rule, a breakout, a deep-value
# longshot buyer, a panic fade and a late-favourite hold. They are not tuned. They
# are the rules people actually write first, implemented plainly, and running all
# of them is a cheap way to ask whether *any* standard shape of idea survives.
#
# One pin, one fee model, nothing else varying. Fees are set to zero, matching
# Polymarket's actual schedule, so what follows is a pure test of whether these
# rules beat the spread. Ranking is on `realized_pnl` because almost nothing
# settles inside a one-day window: closed round trips are the only honest
# scoreboard here.
#
# The generators default to ten shares a trade, so the dollar figures below are
# small by construction. Read the per-share column that follows rather than the
# totals, because that is the number that compares directly against the
# half-spread from Part 1 and the one that scales.

# %%
started = time.time()
results, rows = {}, []
for name, generator in sorted(backtest.STRATEGIES.items()):
    plan = generator(panel)
    if not plan.num_signals:
        rows.append({"strategy": name, "signals": 0, "fills": 0, "realized_pnl": None})
        continue
    signal_table = f"signals_{name}"
    real.create_table(signal_table, plan.signals.schema, time_column="ts")
    real.append(signal_table, plan.signals)
    result = backtest.execute(
        real,
        backtest.BacktestConfig(
            run_id=f"pack-{name}",
            data=backtest.DataConfig(signals=signal_table, snapshot="real-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(),  # no fee: Polymarket's schedule
            output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
            metadata=plan.to_metadata(),
        ),
    )
    results[name] = result
    executions = result.fills.to_pandas()
    rows.append(
        {
            "strategy": name,
            "signals": plan.num_signals,
            "fills": len(executions),
            "realized_pnl": float(result.summary()["realized_pnl"]),
            "fees": float(executions.commission.sum()) if len(executions) else 0.0,
        }
    )
pack = pd.DataFrame(rows).sort_values("realized_pnl", ascending=False, na_position="last")
print(f"{len(results)} runs in {time.time() - started:.1f}s, zero fees throughout\n")
print(pack.to_string(index=False))

# %% [markdown]
# Not one of the eleven made money, at zero fees, and the ordering is close to
# monotone in the number of trades. Three groups are worth separating before
# reading anything into that.
#
# `panic_fade` and `vwap_reversion` generated signals that never filled, so they
# have nothing to say. `deep_value` and `late_favorite_hold` bought and never
# sold, so their realized P&L is exactly zero — they are not the winners the sort
# order makes them look like, they simply never closed anything and their result
# is sitting unrealized in open positions. Everything that actually completed a
# round trip lost money.
#
# Among those, the loss scales with activity, which is the signature of a cost
# problem rather than of seven separately bad ideas: each round trip pays the
# spread, and the rules that trade most pay most.

# %%
traded = pack.dropna(subset=["realized_pnl"])
traded = traded[traded.fills > 0].assign(
    pnl_per_fill=lambda frame: frame.realized_pnl / frame.fills
)
shares = {
    name: float(result.fills.to_pandas().quantity.sum()) for name, result in results.items()
}
traded["shares"] = traded.strategy.map(shares)
traded["cents_per_share"] = traded.realized_pnl / traded.shares * 100
print(traded[["strategy", "fills", "shares", "realized_pnl", "cents_per_share"]]
      .round(4).to_string(index=False))
print(f"\ncorrelation between trade count and P&L: "
      f"{traded.fills.corr(traded.realized_pnl):+.3f}")

# %% [markdown]
# The `cents_per_share` column is the one to hold on to. The busiest rules lose
# on the order of a tenth of a cent per share traded, on a book whose median
# spread is a tenth of a cent wide. That is not a coincidence, and the next
# section shows it is not one.

# %% [markdown]
# ## Where the money went
#
# A correlation is suggestive; the accounting is conclusive. A taker pays half the
# spread on entry and half on exit, so the total spread cost of a run is
# computable directly from the fills the engine produced and the books they met.
# Comparing that budget against the realized loss answers a question that matters
# a great deal: was the rule *unlucky*, or was it merely *expensive*?

# %%
worst = pack.dropna(subset=["realized_pnl"]).iloc[-1].strategy
fills = results[worst].fills.to_pandas()
# A fill is stamped at the event that matched it, which need not be a quote
# instant, so this is an as-of join backwards onto the book it actually met.
paired = pd.merge_asof(
    fills.sort_values("ts"),
    panel[["instrument_id", "ts", "bid", "ask", "mid"]].sort_values("ts"),
    on="ts", by="instrument_id", direction="backward",
)
paired["half_spread_cost"] = (paired.ask - paired.bid).abs() / 2 * paired.quantity
realized = float(results[worst].summary()["realized_pnl"])
budget = pd.DataFrame(
    [
        {"component": "fees paid", "dollars": float(fills.commission.sum())},
        {"component": "half spread crossed", "dollars": float(paired.half_spread_cost.sum())},
    ]
)
budget.loc[len(budget)] = {"component": "total cost", "dollars": budget.dollars.sum()}
budget.loc[len(budget)] = {"component": "realized P&L", "dollars": realized}
budget.loc[len(budget)] = {
    "component": "implied gross edge (cost + P&L)",
    "dollars": float(budget.dollars.iloc[2]) + realized,
}
traded_shares = float(fills.quantity.sum())
budget["cents_per_share"] = budget.dollars / traded_shares * 100
print(f"{worst}: {len(fills)} fills, {traded_shares:,.0f} shares, "
      f"{len(paired.dropna(subset=['bid']))} fills matched to a quote\n")
print(budget.round(4).to_string(index=False))

# %% [markdown]
# Read the last line, and read it as the per-share number. Crossing the spread
# cost this rule more than its entire loss, which means the gross edge — what it
# would have made if trading were free — was *positive*. It was simply nowhere
# near large enough: the spread bill came to several times the gross edge, so a
# signal that was genuinely pointing the right way still produced a loss.
#
# That is a more precise and more useful diagnosis than "the strategy is bad". A
# rule with no signal at all would show a gross edge indistinguishable from zero
# and a loss matching its cost budget almost exactly. This one instead says: the
# idea has something in it, and the way it is being traded destroys several times
# more value than the idea creates. The response that follows is to attack the
# execution — trade less often, trade passively, trade only where the spread is
# narrow — rather than to discard the signal.
#
# Two honest caveats before anyone acts on that. One day of six markets is not
# evidence that the gross edge is real; it is a number computed on a sample far
# too small to distinguish from noise. And a passive version of this rule cannot
# be evaluated on this data at all, for the queue-position reason from Part 2.

# %% [markdown]
# ## Was it the fee or the spread?
#
# On this run the fee was zero, so the question is already answered for
# Polymarket. It is worth answering it in the general form anyway, because the
# distinction is operationally important: a fee is negotiable at volume and a
# spread is not.
#
# `backtest.study` runs a parameter sweep with proper validation attached. We take
# the least-bad rule that traded enough to be scored, split the day into two
# walk-forward windows, and vary the fee rate across them. A **walk-forward** window
# pairs a training period with a later holdout period, so a parameter chosen on the
# first is scored on data it never saw.
#
# One necessary caveat on "least bad": the rule with the smallest loss is usually
# the one that barely traded, and scoring a rule with two fills across four windows
# is meaningless. So we rank among rules with at least thirty fills and say exactly
# what the cut was.

# %%
MIN_FILLS = 30
scored = pack.dropna(subset=["realized_pnl"])
eligible = scored.query("fills >= @MIN_FILLS")
excluded = scored.query("fills < @MIN_FILLS")
best_name = eligible.iloc[0].strategy
print(f"rules with at least {MIN_FILLS} fills: {len(eligible)}")
print(f"excluded as too quiet to score:    {list(excluded.strategy)}")
print(f"least bad among the rest: {best_name} "
      f"({int(eligible.iloc[0].fills)} fills, {eligible.iloc[0].realized_pnl:+.2f})")

stamps_real = list(
    real.sql("SELECT DISTINCT ts_init FROM book_deltas ORDER BY ts_init").to_pandas().ts_init
)
cuts = [0, len(stamps_real) // 3, len(stamps_real) // 2,
        2 * len(stamps_real) // 3, len(stamps_real) - 1]
walk = backtest.WalkForward.of(
    backtest.ValidationWindows(train=(stamps_real[cuts[0]], stamps_real[cuts[1]]),
                               holdout=(stamps_real[cuts[1]], stamps_real[cuts[2]])),
    backtest.ValidationWindows(train=(stamps_real[cuts[2]], stamps_real[cuts[3]]),
                               holdout=(stamps_real[cuts[3]], stamps_real[cuts[4]])),
)
study = backtest.study(
    real,
    study_id="fee-sensitivity",
    base=backtest.BacktestConfig(
        run_id="fee-sensitivity",
        data=backtest.DataConfig(signals=f"signals_{best_name}", snapshot="real-v1"),
        portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
        execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=0.0),
        output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
    ),
    parameters={"execution.fee_rate": [0.0, 0.02, FEE_RATE]},
    validation=walk,
    selection=backtest.TopK(k=2, metric="realized_pnl"),
)
columns = ["trial", "parameters", "train_median_realized_pnl", "holdout_median_realized_pnl"]
print()
print(pd.DataFrame(study.ranked())[columns].to_string(index=False))
print(f"\ntrials {len(study.trials)}, reached the holdout {len(study.selected)}")

# %% [markdown]
# The fee axis behaves exactly as the arithmetic says it should: the training
# median gets worse as the rate rises, monotonically, with a gap between the
# zero-fee and the 0.02 trial that is roughly the fee bill on that many round
# trips. The fee is a real cost and it is well behaved.
#
# The holdout column is the more instructive one, and it should be read for what
# it does *not* establish. The zero-fee trial's holdout median comes out slightly
# positive while its training median is negative, and the temptation is to call
# that a rule that generalises. It is not. These medians are fractions of a dollar
# on a few dozen trades in a few hours of one day, which is well inside the range
# a coin flip would produce, and the sign flip between train and holdout is itself
# the signature of noise rather than of signal. What the walk-forward split buys
# here is the ability to see that, rather than a verdict.

# %% [markdown]
# ## The bar a winner would have had to clear
#
# We tried eleven rules and picked the best. That is a *search*, and a search
# inflates whatever it selects: the maximum of eleven noisy numbers is
# systematically higher than any one of them, so the winner's apparent performance
# is partly an artefact of having looked eleven times.
#
# The **deflated Sharpe ratio** corrects for exactly this. It takes the number of
# trials that produced the winner and computes the Sharpe a lucky-but-worthless
# strategy would be expected to reach by chance, then asks how likely it is that
# the observed Sharpe beats that benchmark. It is the single cheapest defence
# against fooling yourself, and it should be attached to every result that came
# out of a comparison.

# %%
equity = results[best_name].equity.to_pandas().sort_values("ts")
curve = equity.equity.tolist()
returns = [
    (curve[index] - curve[index - 1]) / curve[index - 1]
    for index in range(1, len(curve))
    if curve[index - 1]
]
deflated = quant.deflated_sharpe(returns, trials=len(backtest.STRATEGIES))
print(f"observed Sharpe                        {deflated.sharpe:+.3f}")
print(f"benchmark after {deflated.trials} rules were tried   {deflated.benchmark:+.3f}")
print(f"P(true Sharpe exceeds the benchmark)   {deflated.probability:.3f}")

# %% [markdown]
# ## Settlement, and the gate that refuses it
#
# Part 2 introduced `settlement_observable_ns`, the instant an outcome became
# knowable. Here it does real work. Only one of these six markets resolves inside
# the one-day capture, so a position held in any of the other five *cannot* be
# settled without using information the replay never reached. The engine refuses
# each one individually rather than marking it to the eventual winner.
#
# This is the gate that stops the single most valuable-looking bug in
# prediction-market research. Settling every open position at the known outcome
# would have handed each of these rules the full move to \$1.00 or \$0.00 on
# markets whose results were published weeks after the last book event in the
# data, and the resulting backtest would have looked spectacular.
#
# Read the `refused` column rather than the `settlement_applied` flag. The flag is
# true when settlement reached *any* position, so on a mixed panel a `True`
# alongside several refusals is normal rather than a contradiction. The
# authoritative per-position signal is whether `settlement_pnl` is null.

# %%
capture_end = int(pd.Timestamp(panel.ts.max()).value)
inside = [spec for spec in specs if spec.settlement_observable_ns <= capture_end]
print(f"markets resolving inside the capture: {len(inside)} of {len(specs)}")
for spec in inside:
    print(f"  {spec.metadata['question'][:60]}")

audit = []
for name, result in sorted(results.items()):
    manifest = result.run.to_pandas().iloc[0]
    held = result.positions.to_pandas()
    audit.append(
        {
            "strategy": name,
            "open_positions": len(held),
            "settled": int(held.settlement_pnl.notna().sum()) if len(held) else 0,
            "settlement_applied": bool(manifest.settlement_applied),
        }
    )
audit = pd.DataFrame(audit)
audit["refused"] = audit.open_positions - audit.settled
print()
print(audit.to_string(index=False))
print(f"\npositions held at the end:  {audit.open_positions.sum()}")
print(f"settled:                    {audit.settled.sum()}")
print(f"refused as not yet knowable:{audit.refused.sum():>4}")

# %% [markdown]
# Every position is refused and `settlement_applied` is false everywhere, which is
# stronger than the mixed case the previous paragraph describes: no run happened
# to be holding the one market that resolved inside the window, so settlement
# reached nothing at all. Twenty-nine open positions were left unvalued rather
# than marked to outcomes the replay never saw.
#
# This is also why the scoreboard ranked on `realized_pnl` from the beginning. On
# this capture there is no honest settlement number to rank on.

# %% [markdown]
# ## What the real outcomes are still good for
#
# Settlement is out of reach for most of the panel, but the resolutions are not
# useless: they let us score the *market's own* forecast, which is a question
# about the data rather than about any strategy. The **Brier score** is the mean
# squared error of a probabilistic forecast — lower is better, 0.25 is what you
# get by always saying 50% — and it decomposes into reliability (are your stated
# probabilities honest?), resolution (do you discriminate between outcomes?) and
# the irreducible uncertainty of the events themselves.

# %%
final_quotes = panel.sort_values("ts").groupby("instrument_id").last()
outcomes = {spec.instrument_id: 1.0 if spec.winner_outcome == 0 else 0.0 for spec in specs}
calibration = final_quotes.assign(
    outcome=lambda frame: frame.index.map(outcomes)
).dropna(subset=["mid", "outcome"])
parts = quant.brier_decomposition(calibration.mid.tolist(), calibration.outcome.tolist())
print(f"markets scored   {int(parts['observations'])}")
print(f"Brier            {parts['brier']:.4f}")
print(f"  reliability    {parts['reliability']:.4f}   (lower is better)")
print(f"  resolution     {parts['resolution']:.4f}   (higher is better)")
print(f"  uncertainty    {parts['uncertainty']:.4f}   (irreducible)")

questions = {spec.instrument_id: spec.metadata["question"] for spec in specs}
print("\nlast quoted mid against the realized outcome:")
print(
    calibration.assign(question=lambda frame: [questions[key][:44] for key in frame.index])[
        ["question", "mid", "outcome"]
    ].sort_values("mid").round(3).to_string(index=False)
)
print("\nSix markets is far too few to conclude anything, and these six were")
print("selected for book activity, which favours markets close to resolving.")
print("Read this Brier score as a property of the sample, not of Polymarket.")

# %% [markdown]
# ## Reproduce it
#
# One claim has to hold whatever the result is. `verify()` re-executes the stored
# configuration against the same data pin and compares every result table row by
# row, so a negative finding is exactly as reproducible as a positive one would
# have been. This is what a *pin* buys you: the snapshot name resolves to the same
# bytes forever, so the run can be repeated in six months and either match or fail
# loudly.

# %%
verified = results[best_name].verify()
print(f"verified:        {verified['verified']}")
print(f"tables compared: {list(verified['tables_equal'])}")
print(f"data pin:        {results[best_name].config.data.snapshot}")
print(f"trial digest:    {results[best_name].config.trial_digest[:16]}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for name in ("microprice_imbalance", "rsi_reversion", best_name):
    if name not in results:
        continue
    line = results[name].equity.to_pandas().sort_values("ts")
    axes[0].plot(line.ts, line.equity - line.equity.iloc[0], lw=1.2, label=name)
axes[0].axhline(0.0, color="black", lw=0.8)
axes[0].set_title("Real books, zero fees: every rule pays the spread")
axes[0].set_xlabel("time")
axes[0].set_ylabel("change in equity ($)")
axes[0].tick_params(axis="x", labelrotation=30)
axes[0].legend(fontsize=8)
axes[1].scatter(traded.fills, traded.realized_pnl, s=32, color="#c0392b")
for row in traded.itertuples():
    axes[1].annotate(row.strategy[:14], (row.fills, row.realized_pnl),
                     textcoords="offset points", xytext=(4, 3), fontsize=7)
axes[1].axhline(0.0, color="black", lw=0.8)
axes[1].set_title("Loss scales with the number of round trips")
axes[1].set_xlabel("fills")
axes[1].set_ylabel("realized P&L ($)")
fig.tight_layout()

# %%
real.close()

# %% [markdown]
# ---
#
# # What to take away
#
# **On how the price is made.** Polymarket is a plain central limit order book:
# bids and asks with sizes, and a spread between them at which nothing trades. A
# share pays one dollar or nothing, so the price is a probability and YES plus NO
# must sum to one. The mid is a summary, not a price you can transact at.
#
# **On what a trade costs.** Buy at the ask and sell at the bid and you have paid
# the spread, whatever happened to the mid in between. On the real books above,
# half the spread runs from under one percent of the mid on liquid markets to
# double digits at the cheap end. Any fee sits on top of that, and on the venues
# that charge one it follows a `p * (1 - p)` curve that peaks exactly where most
# trading happens.
#
# **On execution being the strategy.** Part 2's four runs saw the same book and
# the same outcome and landed in four different places, purely on how the orders
# were sent. Entry price, exit price, latency and fill probability are not details
# to be added later; they are most of the result.
#
# **On the negative result.** Eleven standard rules on real tick-level Polymarket
# books, at a zero fee, and not one of them made money. The loss tracked the trade
# count almost perfectly, and for the busiest rule the spread bill came to several
# times whatever gross edge the signal had. The useful form of that finding is not
# "these rules are bad" but "the cost of expressing them this way exceeds what
# they are worth", which points at execution rather than at the ideas.
#
# **On believing your own results.** Whatever the sign of the answer, the same
# machinery applies: pin the input so every run reads identical bytes, gate
# settlement on when the outcome was actually knowable, run the walk-forward split
# so a choice is scored on data it never saw, deflate the Sharpe by the number of
# things you tried, and verify that the run reproduces.
#
# **Where to go next.** The obvious research directions this article does not
# take: obtain both token books and test the parity trade that Part 1 describes;
# capture full depth rather than the top level and study the size you can actually
# trade; capture over months rather than a day so that positions can be held to
# resolution and settlement P&L becomes the scoreboard; and, most promising given
# what the cost budget says, stop crossing the spread at all and study whether
# the maker side is where the money is — which requires delta-level data, because
# it requires queue position.
#
# The full cookbook this article is drawn from lives at
# [h5i-db-cookbook](https://github.com/h5i-dev/h5i-db-cookbook), with a longer
# treatment of each of these in `notebooks/05_prediction_markets/`.
