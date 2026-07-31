# %% [markdown]
# # Calibrating costs from your own fills
#
# Recipe 04/02 varies fees, slippage and latency to see whether a conclusion
# survives them. That is the right first move, and it leaves an obvious
# question open: which setting is *true*? A strategy whose edge is smaller than
# the error in your slippage guess is not a strategy you have tested.
#
# The number is measurable. A replay against a book with depth produces fills
# that walked that book, and the distance each fill travelled from the mid, as a
# function of how much of the displayed size it took, is a cost model. This
# recipe fits one with `quant.costs`, then validates it the only way a cost
# model can be validated: by charging it in a cheaper backtest and checking that
# the cheaper backtest reproduces what the full book actually did.

# %% [markdown]
# ## Terms used here
#
# | term                | meaning |
# | ------------------- | --- |
# | slippage            | the gap between the price you referenced and the price you got |
# | participation       | order size as a fraction of the size displayed at the touch |
# | effective spread    | twice the average cost of crossing, one side measured |
# | market impact       | how much worse the price gets as the order gets bigger |
# | square-root law     | the usual empirical shape: cost grows with the square root of size |
# | top of book         | a backtest that sees only the best bid and offer, with no depth behind them |
# | tick                | the smallest price increment a venue quotes |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest
from h5i_db.quant import costs
import cookbook_utils as cu

SYMBOL = "ACME"
TICK = 0.01
PROBE_SIZES = (100, 250, 500, 1_000, 2_000, 4_000, 8_000)

# %% [markdown]
# ## 1. A book with something behind the touch
#
# Every earlier recipe in this section quoted one level a side, which is enough
# to fill a small order and useless for measuring what a large one costs. This
# tape carries eight levels, one tick apart, each holding more size than the one
# in front of it.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts_init` | `timestamp[ns]` | the instant the book event arrived |
# | `side` | `string` | `buy` for bids, `sell` for offers |
# | `price` | `float64` | the level's price |
# | `size` | `float64` | shares displayed at that price |
# | `event_index` | `int64` | rows sharing one value are one atomic book |

# %%
quotes = cu.make_quotes(symbols=[SYMBOL], days=1, quotes_per_day=2_000, seed=23)
tape = cu.make_equity_tape(
    quotes, levels=8, level_growth=1.6, print_every=3, tick_size=TICK
)
print(f"{tape['book_deltas'].num_rows:,} book rows, "
      f"{tape['book_deltas'].num_rows // (2 * 8):,} events")
tape["book_deltas"].to_pandas().head(9)[["ts_init", "side", "price", "size", "is_last"]]

# %%
db = h5i_db.Database(cu.fresh_db("04_calibrating_costs_from_fills"), create=True)
for name, table in tape.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="eight-level ladder")
db.snapshot("l2-v1", tables=list(tape), note="The full book the cost model is fitted on")

book = tape["book_deltas"].to_pandas()
bids, asks = book[book["side"] == "buy"], book[book["side"] == "sell"]
top_bid = bids.loc[bids.groupby("ts_init")["price"].idxmax()].set_index("ts_init")
top_ask = asks.loc[asks.groupby("ts_init")["price"].idxmin()].set_index("ts_init")
quote = pd.DataFrame(
    {
        "bid": top_bid["price"],
        "bid_size": top_bid["size"],
        "bid_depth": bids.groupby("ts_init")["size"].sum(),
        "ask": top_ask["price"],
        "ask_size": top_ask["size"],
        "ask_depth": asks.groupby("ts_init")["size"].sum(),
    }
).sort_index()
quote["mid"] = (quote["bid"] + quote["ask"]) / 2
quote.head()

# %% [markdown]
# ## 2. Probe orders
#
# A cost model needs fills across a range of sizes. Here they come from probe
# orders: the same size ladder, alternating side, spread through the session so
# each one meets a different book.
#
# On a desk you would use your own historical fills instead, and the arithmetic
# below is unchanged. Probes are used here because a fixture has no trading
# history, and because a deliberately spaced ladder makes the fit legible.

# %%
instants = pd.DatetimeIndex(book["ts_init"].unique()).sort_values()
probes = []
for index, size in enumerate(PROBE_SIZES * 8):
    at = instants[150 + index * 20]
    probes.append(
        {
            "ts": at.floor("us").to_pydatetime() + dt.timedelta(microseconds=1),
            "instrument_id": SYMBOL,
            "side": "buy" if index % 2 == 0 else "sell",
            "quantity": float(size),
            "kind": "market",
            "tag": f"probe-{size}",
        }
    )
signals = backtest.signal_table(sorted(probes, key=lambda row: row["ts"]))
backtest.create_signal_table(db)
db.append("signals", signals, note="size ladder, both sides")
print(f"{signals.num_rows} probe orders, {len(PROBE_SIZES)} sizes x 8 repeats")
signals.to_pandas().head()

# %% [markdown]
# Fees are set to zero for the probe run. The question is what the *book* costs;
# a commission schedule is known exactly and does not need fitting.

# %%
probe_run = backtest.run(
    db,
    "cost-probe",
    starting_cash=10_000_000.0,
    signals="signals",
    snapshot="l2-v1",
    fee_kind="proportional",
    fee_rate=0.0,
    equity_interval_nanos=600_000_000_000,
)
probe_fork = db.fork(probe_run["fork"])
fills = probe_fork.read("bt_fills").to_pandas()
orders = probe_fork.read("bt_orders").to_pandas()
print(f"{len(orders)} orders produced {len(fills)} fills")
print(f"fills per order: {len(fills) / len(orders):.1f} on average, "
      f"{fills.groupby('order_id').size().max()} at most")
fills.head()

# %% [markdown]
# ## 3. One sample per order
#
# A `SlippageSample` is one execution measured against the price that stood when
# it was decided. Several fills belong to one decision, so the fills of an order
# are collapsed into their size-weighted average price first. Measuring each
# partial fill separately would count the same decision many times and weight
# the deepest levels least.

# %%
executed = (
    fills.assign(notional=lambda frame: frame["price"] * frame["quantity"])
    .groupby("order_id")
    .agg(
        ts=("ts", "first"),
        side=("side", "first"),
        quantity=("quantity", "sum"),
        notional=("notional", "sum"),
        levels=("price", "nunique"),
    )
)
executed["avg_price"] = executed["notional"] / executed["quantity"]
decision = pd.merge_asof(
    executed.sort_values("ts").reset_index(),
    quote.reset_index()[["ts_init", "mid", "bid_size", "ask_size"]],
    left_on="ts",
    right_on="ts_init",
)
decision["displayed"] = np.where(
    decision["side"] == "buy", decision["ask_size"], decision["bid_size"]
)
samples = [
    costs.SlippageSample(
        direction=1 if row.side == "buy" else -1,
        fill_price=float(row.avg_price),
        reference_price=float(row.mid),
        quantity=float(row.quantity),
        reference_size=float(row.displayed),
    )
    for row in decision.itertuples()
]
decision["slippage"] = [sample.slippage for sample in samples]
decision["participation"] = [sample.participation for sample in samples]
decision[["side", "quantity", "levels", "avg_price", "mid", "slippage", "participation"]].head()

# %% [markdown]
# Two headline numbers before any fitting. The effective spread is what crossing
# costs on a round trip; the implementation shortfall is what this particular
# set of orders paid per share, weighted by size so that the large orders count
# for what they were.

# %%
print(f"effective spread          {costs.effective_spread(samples):.4f}")
print(f"implementation shortfall  {costs.implementation_shortfall(samples):.4f} per share")
print(f"quoted spread (median)    {(quote['ask'] - quote['bid']).median():.4f}")

# %% [markdown]
# ## 4. Fitting the shape
#
# `fit_impact` regresses slippage on size, either against the square root of
# participation or against participation itself. The square root is the default
# because it is the shape impact studies keep recovering, and because fitting a
# straight line to a concave process extrapolates badly exactly where it
# matters: in the large orders that decide whether a strategy has capacity.

# %%
sqrt_fit = costs.fit_impact(samples, shape="sqrt")
linear_fit = costs.fit_impact(samples, shape="linear")
pd.DataFrame([sqrt_fit.to_dict(), linear_fit.to_dict()]).round(5)

# %% [markdown]
# `usable` is not a statistical test. It is a reminder that a cost model fitted
# to eleven fills is not a cost model, and thirty is the conventional floor for
# pretending otherwise.

# %%
assert sqrt_fit.is_usable, "not enough fills to claim a cost model"
for label, fit in (("sqrt", sqrt_fit), ("linear", linear_fit)):
    print(f"{label:<7} R^2 {fit.r_squared:.3f}  residual {fit.residual_std:.4f}  "
          f"cost at 1x displayed {fit.predict(1.0):.4f}")

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(decision["participation"], decision["slippage"], s=18, alpha=0.6, label="orders")
grid = np.linspace(0.01, decision["participation"].max(), 100)
ax.plot(grid, [sqrt_fit.predict(x) for x in grid], linewidth=1.8, label="square-root fit")
ax.plot(grid, [linear_fit.predict(x) for x in grid], linewidth=1.4, linestyle="--", label="linear fit")
ax.set_title("What each order paid, against how much of the book it took")
ax.set_xlabel("Participation (order size / displayed size)")
ax.set_ylabel("Slippage vs mid ($ per share)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 5. The shortcut, and what it assumes
#
# `costs.fit_from_fills` does all of the above from a run's fork with no
# arguments. It has to choose a reference price without being told one, and it
# uses the size-weighted average of the other fills at the same instant, which
# measures how far through the book each fill reached rather than how far from
# the mid the order ended up.
#
# That makes it a measure of *depth consumed*, available whenever an order
# walked more than one level, and it is the right tool when the decision price
# was never recorded. When you do know the decision price, build the samples
# yourself as above.

# %%
shortcut = costs.fit_from_fills(probe_fork, shape="sqrt")
print(f"shortcut  intercept {shortcut.intercept:+.5f}  coefficient {shortcut.coefficient:+.5f}  "
      f"observations {shortcut.observations}")
print(f"explicit  intercept {sqrt_fit.intercept:+.5f}  coefficient {sqrt_fit.coefficient:+.5f}  "
      f"observations {sqrt_fit.observations}")
print("\nThey answer different questions and should not be expected to agree.")

# %% [markdown]
# ## 6. Charging the model somewhere cheaper
#
# The reason to fit a cost model is to use it where the data cannot support the
# measurement: a daily bar backtest, or any replay with only a top-of-book
# quote. Collapsing this book to one level per side is exactly that backtest --
# all the depth, none of the walking, everything filled at the touch.

# %%
sides = pd.concat(
    [
        quote.reset_index()[["ts_init", "bid", "bid_depth"]]
        .rename(columns={"bid": "price", "bid_depth": "size"})
        .assign(side="buy"),
        quote.reset_index()[["ts_init", "ask", "ask_depth"]]
        .rename(columns={"ask": "price", "ask_depth": "size"})
        .assign(side="sell"),
    ]
).sort_values(["ts_init", "side"])
sides["ts_event"] = sides["ts_init"]
sides["instrument_id"] = SYMBOL
sides["outcome"] = 0
sides["action"] = "snapshot"
sides["event_index"] = sides.groupby("ts_init", sort=False).ngroup()
sides["is_last"] = sides["side"] == "sell"
sides["source_vendor"] = "collapsed"
collapsed = pa.Table.from_pandas(
    sides[cu.BOOK_DELTAS_SCHEMA.names], schema=cu.BOOK_DELTAS_SCHEMA, preserve_index=False
)
print(f"{collapsed.num_rows:,} rows, one level a side, all the size at the touch")
collapsed.to_pandas().head(4)

# %%
cheap = h5i_db.Database(cu.fresh_db("04_calibrating_costs_top_of_book"), create=True)
for name, table in (
    ("instruments", tape["instruments"]),
    ("book_deltas", collapsed),
    ("trades", tape["trades"]),
):
    cheap.create_table(name, table.schema, time_column="ts_init")
    cheap.append(name, table)
cheap.snapshot("top-v1", tables=["instruments", "book_deltas", "trades"])
backtest.create_signal_table(cheap)
cheap.append("signals", signals, note="the same probe orders, cheaper book")
print(f"{collapsed.num_rows:,} book rows and the same {signals.num_rows} orders")

# %% [markdown]
# One thing to check against your own build before charging anything.
# `slippage_ticks` shifts the book by a *fixed* 0.0001 per unit in the current
# Python binding, not by the instrument's `tick_size`. On a prediction market
# those are the same number; on a 290-dollar share they differ by a hundred
# times. The charge below is therefore expressed in units of 0.0001, and the
# assertion at the end of this section is what would catch it if that changed.

# %%
SLIPPAGE_UNIT = 0.0001
probe = backtest.run(
    cheap,
    "top-unit-check",
    starting_cash=10_000_000.0,
    signals="signals",
    snapshot="top-v1",
    fee_kind="proportional",
    fee_rate=0.0,
    slippage_ticks=100,
    equity_interval_nanos=600_000_000_000,
)
probe_prices = cheap.fork(probe["fork"]).read("bt_fills").to_pandas()
plain = backtest.run(
    cheap,
    "top-unit-base",
    starting_cash=10_000_000.0,
    signals="signals",
    snapshot="top-v1",
    fee_kind="proportional",
    fee_rate=0.0,
    equity_interval_nanos=600_000_000_000,
)
plain_prices = cheap.fork(plain["fork"]).read("bt_fills").to_pandas()
shift = (
    probe_prices[probe_prices["side"] == "buy"]["price"].mean()
    - plain_prices[plain_prices["side"] == "buy"]["price"].mean()
)
print(f"100 slippage units moved the average buy by {shift:.5f}")
assert abs(shift - 100 * SLIPPAGE_UNIT) < 1e-6, "slippage_ticks is not what this recipe assumes"

typical = float(decision["participation"].median())
charge = int(round(sqrt_fit.predict(typical) / SLIPPAGE_UNIT))
print(f"median participation {typical:.2f} -> predicted cost "
      f"{sqrt_fit.predict(typical):.4f} = {charge} slippage units")

rows = []
for label, ticks in (("uncharged", 0), ("fitted charge", charge)):
    run = backtest.run(
        cheap,
        f"top-{label.split()[0]}",
        starting_cash=10_000_000.0,
        signals="signals",
        snapshot="top-v1",
        fee_kind="proportional",
        fee_rate=0.0,
        slippage_ticks=ticks,
        equity_interval_nanos=600_000_000_000,
    )
    fork = cheap.fork(run["fork"])
    got = fork.read("bt_fills").to_pandas()
    priced = pd.merge_asof(
        got.sort_values("ts"),
        quote.reset_index()[["ts_init", "mid"]],
        left_on="ts",
        right_on="ts_init",
    )
    signed = np.where(priced["side"] == "buy", 1.0, -1.0)
    cost = float(
        (signed * (priced["price"] - priced["mid"]) * priced["quantity"]).sum()
        / priced["quantity"].sum()
    )
    rows.append({"book": label, "slippage units": ticks, "cost per share": cost,
                 "shares": float(priced["quantity"].sum())})
    fork.close()

truth = costs.implementation_shortfall(samples)
rows.append({"book": "full ladder (truth)", "slippage units": None,
             "cost per share": truth, "shares": float(decision["quantity"].sum())})
comparison = pd.DataFrame(rows)
comparison.round(4)

# %% [markdown]
# The uncharged top-of-book run pays half a tick on every order regardless of
# size, which is the error a cost model exists to remove. Charging the fitted
# number moves it into the neighbourhood of what the real book did.
#
# It does not land exactly, and it cannot: a constant tick charge is a flat
# approximation to a curve, so it overcharges the small orders and undercharges
# the largest. What the fit buys is a defensible average and, more importantly,
# a statement about *which* orders the backtest is still lying about.

# %%
uncharged = comparison.loc[comparison["book"] == "uncharged", "cost per share"].iloc[0]
charged = comparison.loc[comparison["book"] == "fitted charge", "cost per share"].iloc[0]
print(f"error before charging  {abs(uncharged - truth):.4f} per share")
print(f"error after charging   {abs(charged - truth):.4f} per share")
assert abs(charged - truth) < abs(uncharged - truth), "the charge made it worse"

# %% [markdown]
# ## Takeaways
#
# - Slippage is measurable from a replay that had depth to walk; it does not
#   have to be a parameter someone chose.
# - Collapse the fills of one order into one sample. Each decision is one
#   observation, not one per level it reached.
# - `fit_impact` defaults to the square root because a linear fit extrapolates
#   badly into the large orders that decide capacity.
# - `CostFit.is_usable` is a sample-size reminder, and thirty fills is a floor
#   rather than a licence.
# - `fit_from_fills` needs no reference price and therefore measures depth
#   consumed rather than shortfall against a decision. Know which you asked for.
# - The point of the fit is to charge it where the data is thinner. Validate it
#   by running the cheaper backtest against the richer one, not by admiring the
#   R-squared.

# %%
probe_fork.close()
cheap.close()
db.close()
