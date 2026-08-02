# %% [markdown]
# # Practical backtesting for Polymarket, with h5i-db
#
# The companion article ended on the sentence this notebook starts from:
#
# > That gap between the displayed probability and the price at which a strategy
# > can really trade is where realistic backtesting begins.
#
# So this picks up there. It assumes you already know what a YES token is, why
# YES plus NO must sum to a dollar, how a limit order book turns competing orders
# into a price, and why "the price" is at least five different numbers depending
# on which question you are asking. If any of that is unfamiliar, read section 1
# of that article first. None of it is repeated here.
#
# What follows is the practical half, and it does three things:
#
# 1. **Get real Polymarket order books into a database**, from a tick-level
#    capture of markets that have since resolved.
# 2. **Write a strategy** as an event-driven callback, which is the form that
#    generalises to whatever you actually want to test.
# 3. **Run it and read the result**, including the one decomposition that tells
#    you whether an idea is worth pursuing.
#
# The database is [h5i-db](https://github.com/h5i-dev/h5i-db), an embedded,
# versioned time-series store with an event-driven backtest engine built in.
# Nothing here needs a server or an account: it is a directory on disk and a
# Python import.


# %%
import collections
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
from IPython.display import HTML, display

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest, quant, venues

print(f"h5i-db {h5i_db.__version__}")

# %% [markdown]
# ---
#
# # 1. Real Polymarket books
#
# ## The data
#
# A bounded, non-commercial sample of tick-level Polymarket books, published on
# Kaggle. It is not redistributed here; the cell below prints the download
# commands if the files are absent. The dataset is CC BY-NC 4.0, so review the
# licence before using anything derived from it commercially.

# %%
CACHE = Path("data/cache/kaggle-polymarket")
missing = cu.kaggle_missing_files(CACHE)
if missing:
    print("This article needs the Kaggle sample. Missing:", missing)
    print(f"\nDataset: {cu.KAGGLE_POLYMARKET_DATASET}  (licence {cu.KAGGLE_POLYMARKET_LICENSE})")
    for line in cu.kaggle_download_commands(CACHE):
        print("  " + line)
    raise SystemExit("download the sample, then re-run")

snapshots_path = CACHE / "snapshots_2026-03-09.parquet"
targets_path = CACHE / "market_targets.parquet"
print(f"books    {pq.ParquetFile(snapshots_path).metadata.num_rows:,} rows "
      f"({snapshots_path.stat().st_size / 1e6:.0f} MB)")
print(f"labels   {pq.ParquetFile(targets_path).metadata.num_rows:,} markets")
print(f"licence  {cu.KAGGLE_POLYMARKET_LICENSE} (non-commercial)")

# %% [markdown]
# The capture is a websocket recording written straight to Parquet, so the
# interesting content is a JSON string in one column rather than typed fields.
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
print(f"  best_bid   {event['best_bid']}   best_ask {event['best_ask']}")
print(f"  depth      {len(event['bids'])} bid levels / {len(event['asks'])} ask levels")
print(f"  first bid  {event['bids'][0]}  (price and size arrive as strings)")

# %% [markdown]
# ### Three limits, all properties of the capture
#
# Every study inherits the shape of its data, and it is better to state that up
# front than to discover it in a result.
#
# It covers one day. Most of these markets resolve later, so almost nothing can be
# held to resolution inside the window. This is an intraday study, and closed
# round trips are the only honest scoreboard.
#
# It carries the YES side only. The NO token's book is absent, so the parity trade
# is not testable here and every rule must trade the YES book.
#
# It carries full depth, which we truncate to the touch at ingest, because a rule
# that trades at the touch reads two levels. The truncation is opt-in and
# reported, since a silently shallower book is a different book.
#
# ## Picking markets
#
# `market_targets.parquet` carries one row per market including the UMA-verified
# outcome. UMA is the decentralised oracle Polymarket resolves through. Taking the
# outcome from the label file, instead of inferring it from the final price, is the
# difference between a label and a guess.
#
# | column | type | meaning |
# |---|---|---|
# | `condition_id` | `string` | the market, and the join key to the book capture |
# | `question` | `string` | what was being traded |
# | `end_date` | `string` | ISO-8601, when the market closes |
# | `clob_token_id_yes` / `_no` | `string` | the per-outcome token identifiers |
# | `target` | `int8` | 1 = YES wins, 0 = NO wins, null = not yet resolved |
#
# Two of those are identifiers rather than data, and both matter at ingest.
# `condition_id` names the market, and appears in the book capture as `market_id`.
# The token ids name the two sides of it, and are what an individual book message
# is matched against. Both are long enough to wreck a table layout, so the preview
# truncates them.

# %%
targets = pq.read_table(targets_path).to_pandas()
print(f"{len(targets):,} rows x {targets.shape[1]} columns")
print(f"condition_id is {targets.condition_id.str.len().max()} characters, "
      f"a token id up to {targets.clob_token_id_yes.str.len().max()}")
targets.assign(
    condition_id=targets.condition_id.str[:18] + "...",
    clob_token_id_yes=targets.clob_token_id_yes.str[:14] + "...",
)[["condition_id", "question", "end_date", "clob_token_id_yes", "target"]].head()

# %% [markdown]
# We need markets present in the capture *and* resolved in the labels, and among
# those we take the six with the most book activity, because a strategy needs
# events to act on. That selection is not neutral, and it is stated here rather
# than buried: picking the busiest markets favours markets close to resolving.

# %%
ids = pq.read_table(snapshots_path, columns=["market_id"]).column("market_id").combine_chunks()
coverage = pd.DataFrame(pc.value_counts(ids).to_pylist()).rename(
    columns={"values": "condition_id", "counts": "snapshot_rows"}
)
pool = coverage.merge(targets, on="condition_id").query("target.notna()")
print(f"markets in the capture: {len(coverage):,}   resolved: {len(pool):,}")
chosen = pool.sort_values("snapshot_rows", ascending=False).head(6).reset_index(drop=True)
print()
print(chosen[["question", "snapshot_rows", "end_date", "target"]].to_string(index=False))

# %% [markdown]
# ## Ingest
#
# Two declarations do the whole job.
#
# `MarketSpec` describes each contract, and it is where the two identifiers above
# get their meaning. `instrument_id` is whatever you choose to call the market;
# passing `condition_id` is what makes `book_deltas.instrument_id` join back to the
# label file later. `outcome_labels` and `tokens` are paired **positionally**, so
# the YES token must sit at the same index as the `YES` label, and a book message
# carrying that token becomes `outcome=0`. Get the pairing backwards and every fill
# is attributed to the wrong side.
# `settlement_observable_ns` is when the outcome became *knowable*, and the engine
# refuses to settle a position unless the replay actually reached that instant.
# That one field stops the most common form of accidental lookahead in
# prediction-market research, and we watch it work at the end.
#
# `ArchiveLayout` describes the *file shape*: where the timestamp is, what unit it
# is in, which column holds the payload, which JSON field names the token. Reading
# a new vendor's format is a literal, not a new code path.

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
    max_levels=1,  # top of book; the ingest report says how much this dropped
)

db = h5i_db.Database(cu.fresh_db("article_polymarket"), create=True)
venues.write_markets(db, specs, note="UMA-verified labels")
ingest = venues.ingest_archive(
    db, files=[snapshots_path], markets=specs, layout=layout, note="tick capture, top of book"
)
pin = db.snapshot("real-v1", tables=["instruments", "book_deltas", "resolutions"],
                  note="the pinned input for every run below")
print(ingest)
print(f"\npinned as '{pin['name']}', checksum {pin['checksum'][:16]}")
for spec in specs:
    print(f"  {spec.outcome_labels[spec.winner_outcome]:>3} won: {spec.metadata['question'][:60]}")

# %% [markdown]
# Everything downstream leans on `db.snapshot(...)`. It pins a named, immutable
# view of the input tables, so every run provably reads the same bytes and a result
# can be reproduced months later. Append more data tomorrow and `real-v1` still
# resolves to exactly what it does today.
#
# The canonical `book_deltas` table now holds real books. Every row is one price
# level of one book update; rows sharing an `event_index` are one atomic event,
# and the last row of an event is flagged `is_last`.
#
# | column | type | meaning |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | when the recorder received it; replay is ordered by this |
# | `ts_event` | `timestamp[ns]` | when the venue says it happened; keep both, they differ |
# | `instrument_id` | `string` | the market |
# | `outcome` | `uint16` | which token: 0 = YES, 1 = NO |
# | `action` | `string` | `snapshot` for a full book, `delta` for an incremental change |
# | `side` | `string` | `buy` is a bid, `sell` is an ask |
# | `price` / `size` | `float64` | the level's price as a probability, and shares resting there |
# | `event_index` | `int64` | groups the rows of one atomic update |
# | `is_last` | `bool` | true on the final row of an event |

# %%
book = db.read("book_deltas")
print(f"{book.num_rows:,} rows x {book.num_columns} columns")
book.to_pandas().head()

# %% [markdown]
# ---
#
# # 2. What these particular books cost to trade
#
# The article laid out the mechanics: cross the spread and you pay it, and a large
# order walks up the book to worse levels. What it could not say is how big those
# costs are on any real market, because that is a property of the data. Here it is
# for this capture.
#
# `backtest.quote_panel` collapses `book_deltas` to the top of book: one row per
# instrument per instant, which is also the shape the strategy will see.

# %%
panel = backtest.quote_panel(db, snapshot="real-v1")
panel["spread"] = panel.ask - panel.bid
panel["mid"] = (panel.bid + panel.ask) / 2
print(f"{len(panel):,} quotes across {panel.instrument_id.nunique()} markets\n")
print(panel[["bid", "ask", "spread", "mid"]].describe().round(4).to_string())

# %% [markdown]
# One book from the capture, drawn as the ladder from the article. It has two rows
# rather than five because `max_levels=1` truncated the ingest to the touch, which
# is all a taker at this size ever reaches. These numbers came out of a venue feed,
# not an example.

# %%
questions = {spec.instrument_id: spec.metadata["question"] for spec in specs}
sample = panel.loc[panel.spread.idxmin()]
depth = pd.DataFrame(
    [
        {"side": "ask", "price": sample.ask, "size": sample.ask_size},
        {"side": "bid", "price": sample.bid, "size": sample.bid_size},
    ]
)
heading = (f"{questions[sample.instrument_id][:48]} — YES book at "
           f"{sample.ts:%H:%M:%S}")
display(HTML(
    '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;max-width:560px">'
    f'<div style="padding:4px 10px;opacity:.75;font-size:.9em">{heading}</div>'
    '<table style="border-collapse:collapse;width:100%">'
    + "".join(
        '<tr>'
        f'<td style="padding:3px 10px;opacity:.55;font-size:.85em">{row.side.upper()}</td>'
        '<td style="padding:3px 10px;text-align:right;font-variant-numeric:tabular-nums">'
        f'{row.price:.4f}</td>'
        '<td style="padding:3px 10px;text-align:right;opacity:.7;'
        f'font-variant-numeric:tabular-nums">{row.size:,.0f}</td>'
        '<td style="padding:3px 10px;width:55%"><div style="background:'
        f'{"rgba(192,57,43,.55)" if row.side == "ask" else "rgba(44,127,184,.55)"};'
        f'width:{row.size / depth["size"].max() * 100:.1f}%;height:14px;'
        'border-radius:3px"></div></td></tr>'
        + ('<tr><td colspan="4" style="padding:6px 10px;border-top:1px dashed '
           'currentColor;border-bottom:1px dashed currentColor;opacity:.65;'
           f'font-size:.85em">spread {sample.spread:.4f} &nbsp;·&nbsp; mid '
           f'{sample.mid:.4f} &nbsp;·&nbsp; nothing trades in here</td></tr>'
           if row.side == "ask" else "")
        for row in depth.itertuples()
    )
    + "</table></div>"
))

# %% [markdown]
# ## The hurdle, by price level
#
# Half the spread is what a taker gives up on each leg, so it is the natural unit
# for "what does trading here cost". Expressed as a percentage of the price paid,
# it varies across the book by more than two orders of magnitude.

# %%
levels = panel.assign(level=lambda frame: frame.mid.round(1)).groupby("level").agg(
    quotes=("mid", "size"),
    mean_mid=("mid", "mean"),
    half_spread=("spread", lambda series: series.mean() / 2),
)
# Divide by the mean mid inside each bucket, not by the bucket label, or the 0.0
# bucket prints `inf` exactly where the most interesting number is.
levels["half_spread_pct_of_mid"] = levels.half_spread / levels.mean_mid * 100
print(levels.round(4).to_string())

# %% [markdown]
# Crossing the spread costs well under half a percent in the crowded middle of the
# book and near the certain end. On cheap contracts it explodes: a single round
# trip there costs a double-digit percentage of the position before anything has
# happened. A longshot strategy has to be right about *direction* far more often
# than it has to be right about *probability*.
#
# Keep the middle-of-the-book number in mind. The strategy below trades there, and
# we will compare its earnings directly against it: roughly 0.14 cents a share,
# paid on entry and again on exit.
#
# One more cost, briefly. Polymarket's headline trading fee has historically been
# zero, which is unusual, so every run below charges nothing and the costs you see
# are pure spread. Venues that do charge, Kalshi most prominently, use a fee
# proportional to `p * (1 - p)`, the variance of a coin flip, so it peaks where the
# contract is most uncertain. h5i-db implements that curve as `fee_kind="kalshi"`
# if you need it.

# %% [markdown]
# ---
#
# # 3. Writing a strategy
#
# A strategy here is an **event-driven callback**: an object whose methods are
# called as the replay walks the data forward, returning order commands. This is
# the shape most backtesting frameworks use, and it generalises. Anything
# path-dependent, anything reacting to its own fills or current position, fits
# inside it.
#
# The rule we will write is breakout, about the simplest trend-following idea there
# is:
#
# > Buy when the price makes a new high for the last `lookback` observations. Sell
# > out when it makes a new low. Hold nothing otherwise.
#
# It is deliberately simple. The subject of this article is the machinery around a
# strategy, not the strategy itself, so replace the body of `on_event` and
# everything else in the notebook still works.
#
# ## What a callback receives
#
# Subclass `backtest.EventStrategy` and implement the callbacks you want; the ones
# you leave alone are never called at all. A market event arrives as a plain
# mapping:
#
# | key | meaning |
# |---|---|
# | `type` | `market`, `timer`, or `fill` |
# | `ts_init` / `ts_event` | recorder and venue timestamps, in nanoseconds |
# | `instrument_id`, `outcome` | which market, and which token |
# | `best_bid`, `best_ask` | the touch **as the venue currently has it**, or `None` |
# | `position` | how many shares you are holding right now |
#
# Return `None` to do nothing, or a command mapping to act. `submit` needs a
# `client_order_id` unique within the run, which is how a later `amend` or
# `cancel` refers back to it.
#
# ## Three bugs you cannot write
#
# Here is where the callback form earns its place in a first tutorial. Writing this
# same rule against a precomputed table of signals invites three classic mistakes,
# and none of them is even expressible here.
#
# You cannot see the future. A table-based rule computes a rolling window over the
# whole history at once and has to remember to exclude the current row from its own
# window. A callback is handed one event at a time and physically cannot read the
# next one.
#
# You cannot trade on a price the venue has not processed. The engine shows market
# data to the venue before the strategy, and queues strategy commands instead of
# executing them inside the callback that produced them. An order decided from this
# event meets the *next* book, which is the constraint reality imposes and the one a
# naive backtest forgets.
#
# You cannot lose track of your position. `event["position"]` is the engine's own
# number, not a flag you maintain, so the rule cannot double up on an entry it has
# forgotten making.

# %%
class Breakout(backtest.EventStrategy):
    """Buy a new `lookback`-observation high; sell out on a new low."""

    def __init__(self, lookback=24, quantity=10.0):
        self.lookback = lookback
        self.quantity = quantity
        self.history = collections.defaultdict(collections.deque)
        self.orders_sent = 0

    def on_event(self, event):
        if event["type"] != "market":
            return None
        bid, ask = event["best_bid"], event["best_ask"]
        if bid is None or ask is None:
            return None  # one side of the book is empty; nothing to act on
        mid = (bid + ask) / 2.0
        window = self.history[(event["instrument_id"], event["outcome"])]

        command = None
        if len(window) == self.lookback:  # only act once the window is full
            if event["position"] == 0 and mid > max(window):
                command = self._order(event, "buy", self.quantity, "breakout-entry")
            elif event["position"] > 0 and mid < min(window):
                command = self._order(event, "sell", event["position"], "breakout-exit")

        # Record *after* deciding, so `window` always describes the past.
        window.append(mid)
        if len(window) > self.lookback:
            window.popleft()
        return command

    def _order(self, event, side, quantity, tag):
        self.orders_sent += 1
        return {
            "action": "submit",
            "client_order_id": f"breakout-{self.orders_sent}",
            "instrument_id": event["instrument_id"],
            "outcome": event["outcome"],
            "side": side,
            "quantity": quantity,
            "reduce_only": side == "sell",  # an exit may only shrink a position
            "tag": tag,
        }


print("strategy defined:", Breakout.__doc__)

# %% [markdown]
# ---
#
# # 4. Running it
#
# A run is a typed configuration plus a strategy instance. `DataConfig` names the
# pin and gives the strategy an id; `PortfolioConfig` sets the starting cash; and
# `ExecutionConfig` carries the venue assumptions, empty here because Polymarket
# charges no fee and we are making no latency claim.

# %%
LOOKBACK = 24
config = backtest.BacktestConfig(
    run_id=f"breakout-{LOOKBACK}",
    data=backtest.DataConfig(strategy_id="breakout", snapshot="real-v1"),
    portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
    execution=backtest.ExecutionConfig(),
    output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
    metadata={"strategy": "breakout", "lookback": LOOKBACK},
)
result = backtest.execute(db, config, strategy=Breakout(lookback=LOOKBACK))
fills = result.fills.to_pandas()
summary = result.summary()
print(f"orders sent   {summary['orders']}")
print(f"fills         {len(fills)}")
print(f"shares traded {fills.quantity.sum():,.0f}")
print(f"realized P&L  {summary['realized_pnl']:+.4f}")
print(f"fees paid     {fills.commission.sum():.4f}")

# %% [markdown]
# Every execution is a table. Look at the first few instead of trusting the total,
# and note `is_taker` on each row: the rule sends market orders, so it crossed the
# spread every single time, and section 2 said what that costs.

# %%
fills[["ts", "instrument_id", "side", "price", "quantity", "is_taker", "tag"]].head(8)

# %% [markdown]
# ### 174 orders, 104 fills
#
# The order table says exactly what happened to each one.

# %%
orders = result.orders.to_pandas()
print(orders.status.value_counts().to_string())
print(f"\n{len(fills)} fills from {int((orders.status == 'filled').sum())} filled orders, "
      "because an order big enough to walk two levels produces two")
refused = orders[orders.status == "rejected"]
print(f"\nwhy the {len(refused)} rejections, in the engine's own words:")
print(f"  {refused.reject_reason.iloc[0]}")

# %% [markdown]
# Nearly all of them were rejected because the contract had already stopped
# trading. One of the six markets closed on 2026-03-08 while the capture covers
# 2026-03-09, so its book keeps ticking in the data after the contract itself
# expired. The strategy has no way to know that. The engine does, from
# `expiration_ns` on the `MarketSpec`, and refuses every order.
#
# This is the sort of thing a hand-rolled backtest gets wrong silently. It would
# have happily traded a dead market for a full day and booked the results. Here it
# is 70 rejections in a status column, which is a much better place for it.

# %%
expired = db.sql(
    """
    SELECT i.instrument_id,
           to_timestamp_nanos(min(i.expiration_ns)) AS stopped_trading,
           max(b.ts_init)                           AS last_book_event,
           count(*)                                 AS book_rows_after_expiry
    FROM instruments AS i
    JOIN book_deltas AS b ON b.instrument_id = i.instrument_id
    WHERE b.ts_init > to_timestamp_nanos(i.expiration_ns)
    GROUP BY i.instrument_id
    """
).to_pandas()
print(f"{len(expired)} market(s) keep quoting after they stop trading:\n")
print(expired.assign(question=lambda f: [questions[k][:44] for k in f.instrument_id])[
    ["question", "stopped_trading", "last_book_event", "book_rows_after_expiry"]
].to_string(index=False))

# %% [markdown]
# ---
#
# # 5. Reading the result
#
# ## Where the money went
#
# The strategy lost money. The useful question is *why*, because "the idea is
# wrong" and "the idea is fine and the trading is too expensive" call for completely
# different responses, and the fills can tell them apart.
#
# A taker pays half the spread on entry and half on exit, so the spread bill is
# computable directly from the fills and the books they met. Subtract it and what
# remains is the **gross edge**: what the rule would have made if trading were
# free.

# %%
def decompose(outcome):
    """Split a run into gross edge, spread paid, and what is left."""
    executions = outcome.fills.to_pandas()
    # A fill is stamped at the event that matched it, which need not be a quote
    # instant, so this is an as-of join backwards onto the book it actually met.
    matched = pd.merge_asof(
        executions.sort_values("ts"),
        panel[["instrument_id", "ts", "bid", "ask"]].sort_values("ts"),
        on="ts", by="instrument_id", direction="backward",
    )
    cost = float(((matched.ask - matched.bid).abs() / 2 * matched.quantity).sum())
    pnl = float(outcome.summary()["realized_pnl"])
    traded = float(executions.quantity.sum())
    return {
        "fills": len(executions), "shares": traded, "matched": int(matched.bid.notna().sum()),
        "gross_edge": cost + pnl, "spread_cost": cost, "net_pnl": pnl,
        "fees": float(executions.commission.sum()),
        "gross_c_per_share": (cost + pnl) / traded * 100,
        "cost_c_per_share": cost / traded * 100,
    }


parts = decompose(result)
budget = pd.DataFrame(
    [
        {"component": "gross edge (before costs)", "dollars": parts["gross_edge"]},
        {"component": "half spread crossed", "dollars": -parts["spread_cost"]},
        {"component": "fees paid", "dollars": -parts["fees"]},
        {"component": "= realized P&L", "dollars": parts["net_pnl"]},
    ]
)
budget["cents_per_share"] = budget.dollars / parts["shares"] * 100
print(f"{parts['fills']} fills, {parts['shares']:,.0f} shares, "
      f"{parts['matched']} matched to a quote\n")
print(budget.round(4).to_string(index=False))

# %% [markdown]
# So the rule *does* have a gross edge. It made about 0.11 cents a share before
# costs, meaning the signal pointed the right way more often than not. It then
# paid about 0.15 cents a share to cross the spread, and 0.15 is bigger than 0.11.
#
# That is far more useful than "it loses". The idea is not worthless; it is being
# expressed in a way that costs slightly more than it is worth, which suggests an
# obvious lever. Trade less often. If the edge per share survives while the number
# of trades falls, the arithmetic flips.
#
# ## Trading less often
#
# `lookback` is exactly that lever: a longer window is a higher bar for what counts
# as a breakout, so fewer and more selective trades. Each setting needs a fresh
# strategy instance, because a callback carries state.

# %%
def run_breakout(lookback):
    """One run per setting. Each fork is named after the setting that made it."""
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=f"breakout-{lookback}",
            data=backtest.DataConfig(strategy_id=f"breakout-{lookback}", snapshot="real-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(),
            output=backtest.OutputConfig(equity_interval_nanos=60 * 1_000_000_000),
            metadata={"strategy": "breakout", "lookback": lookback},
        ),
        strategy=Breakout(lookback=lookback),
    )


LOOKBACKS = (6, 12, 24, 48, 96)
runs = {LOOKBACK: result}  # the run above is already one of the settings
rows = []
for lookback in LOOKBACKS:
    if lookback not in runs:
        runs[lookback] = run_breakout(lookback)
    rows.append({"lookback": lookback, **decompose(runs[lookback])})
sweep = pd.DataFrame(rows)
columns = ["lookback", "fills", "shares", "gross_edge", "spread_cost", "net_pnl",
           "gross_c_per_share", "cost_c_per_share"]
print(sweep[columns].round(4).to_string(index=False))
print(f"\ncorrelation between fills and P&L: {sweep.fills.corr(sweep.net_pnl):+.3f}")

# %% [markdown]
# Four columns to read, in order.
#
# `spread_cost` falls with the trade count, from about \$4 to \$0.5. Fewer shares
# crossed, less spread paid; no surprise.
#
# `gross_edge` does not follow it. It wanders around a dollar with no trend, and
# the busiest setting has the *least* of it. Trading sixteen times as often did
# not find sixteen times as much signal; it found the same signal and buried it in
# noise.
#
# `cost_c_per_share` is flat, at about 0.14 cents. It has to be: that is the
# half-spread, a property of the book and not of the rule.
#
# `gross_c_per_share` climbs steeply. The trades a short lookback adds are worse
# than the ones it already had.
#
# The sign of `net_pnl` flips exactly where those last two columns cross.
# **A strategy is profitable when its gross edge per share exceeds the half-spread
# it pays, and everything else is bookkeeping.**

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for lookback in LOOKBACKS:
    curve = runs[lookback].equity.to_pandas().sort_values("ts")
    axes[0].plot(curve.ts, curve.equity - curve.equity.iloc[0], lw=1.2,
                 label=f"lookback {lookback}")
axes[0].axhline(0.0, color="black", lw=0.8)
axes[0].set_title("One rule, five settings, real books at zero fees")
axes[0].set_xlabel("time")
axes[0].set_ylabel("change in equity ($)")
axes[0].tick_params(axis="x", labelrotation=30)
axes[0].legend(fontsize=8)
axes[1].plot(sweep.lookback, sweep.gross_c_per_share, marker="o", color="#2c7fb8",
             label="gross edge earned")
axes[1].plot(sweep.lookback, sweep.cost_c_per_share, marker="s", color="#c0392b",
             label="half spread paid")
axes[1].fill_between(sweep.lookback, sweep.gross_c_per_share, sweep.cost_c_per_share,
                     where=sweep.gross_c_per_share >= sweep.cost_c_per_share,
                     color="#2c7fb8", alpha=0.18)
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(list(LOOKBACKS))
axes[1].set_xticklabels([str(value) for value in LOOKBACKS])
axes[1].set_title("Profitable exactly where the lines cross")
axes[1].set_xlabel("lookback (observations)")
axes[1].set_ylabel("cents per share")
axes[1].legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## Before believing any of it
#
# One setting makes money, and this is where a research session most easily fools
# the person running it. We tried five things and picked the best one. The winner
# made 49 cents on 33 fills in a single day.
#
# The **deflated Sharpe ratio** prices exactly that mistake. It computes the Sharpe
# a worthless strategy would be expected to reach by luck given how many trials
# produced it, then asks how likely the observed Sharpe is to beat that. Run it
# twice, pretending we tried one setting and then telling the truth, because the
# difference between those two numbers is the lesson.

# %%
best_lookback = int(sweep.sort_values("net_pnl", ascending=False).iloc[0].lookback)
best = runs[best_lookback]
curve = best.equity.to_pandas().sort_values("ts").equity.tolist()
returns = [
    (curve[i] - curve[i - 1]) / curve[i - 1] for i in range(1, len(curve)) if curve[i - 1]
]
winner = sweep.set_index("lookback").loc[best_lookback]
print(f"best setting: lookback {best_lookback}, "
      f"{int(winner.fills)} fills, {winner.net_pnl:+.2f}\n")
for claimed in (1, len(LOOKBACKS)):
    deflated = quant.deflated_sharpe(returns, trials=claimed)
    label = "if we pretend we tried 1" if claimed == 1 else f"admitting we tried {claimed}"
    print(f"{label:<28} sharpe {deflated.sharpe:+.4f}  "
          f"benchmark {deflated.benchmark:+.4f}  P(genuine) {deflated.probability:.3f}")

# %% [markdown]
# Same run, same equity curve, same Sharpe. Claim one setting and it looks like a
# 76% chance of being real; admit five and it falls to 29%. Nobody applies this
# correction for you. The trial count is a fact about your afternoon, not about
# your data.
#
# There is a limit on what one day of six markets could show even in principle.
# The panel includes three questions about the same Federal Reserve meeting, whose
# outcomes are mechanically related, so the effective number of independent bets
# is smaller than six.
#
# ## Settlement, and the gate that refuses it
#
# One more thing the engine does on your behalf. Only one of these six markets
# resolves inside the one-day capture, so a position held in any of the others
# cannot be valued without information the replay never reached. The engine
# refuses each one individually rather than marking it to the eventual winner.

# %%
capture_end = int(pd.Timestamp(panel.ts.max()).value)
inside = [spec for spec in specs if spec.settlement_observable_ns <= capture_end]
positions = best.positions.to_pandas()
print(f"markets resolving inside the capture: {len(inside)} of {len(specs)}")
print(f"positions still open at the end:      {len(positions)}")
print(f"of those, settled:                    {int(positions.settlement_pnl.notna().sum())}")
warning = best.run.to_pandas().warnings.iloc[0]
if isinstance(warning, str) and warning:
    print(f"\nthe engine's own words:\n  {warning.split(';')[0][:190]}")

# %% [markdown]
# Had those positions been marked to their eventual outcomes, this backtest would
# have looked spectacular. The markets resolved weeks after the last book event in
# the data. That is the most valuable-looking bug in prediction-market research,
# and better refused by a machine than remembered by a person.
#
# ## The run as one page
#
# Everything above was assembled by hand: a table here, a chart there.
# `result.report()` does it in one call, rendering the run as a self-contained
# HTML document with no network access when opened and no dependencies, so it can
# be attached to a review or opened in five years.
#
# It is not a tearsheet. A tearsheet answers *how did the equity curve behave*;
# this answers *should I believe these numbers, and what actually happened*, so it
# leads with the replay fidelity, the pin, and the run's warnings before any
# performance figure. Read the status banner first: it says `periodic L2
# snapshots` rather than something reassuring, because that is the honest
# description of what this capture supports.

# %%
report_path = Path(f"data/cache/polymarket-breakout-{best_lookback}.html")
document = best.report(report_path, title=f"Polymarket breakout, lookback {best_lookback}")
print(f"wrote {report_path} ({len(document) / 1024:.0f} KB, self-contained)")
display(best)

# %% [markdown]
# ## Reproducing it
#
# The one claim that has to hold whatever the result is. `verify()` re-executes
# the stored configuration against the same pin and compares every result table
# row by row. A callback carries state, so it needs a fresh instance. The one that
# just ran has a full history buffer and would not start from the same place.

# %%
verified = best.verify(strategy=Breakout(lookback=best_lookback))
print(f"verified:        {verified['verified']}")
print(f"tables compared: {list(verified['tables_equal'])}")
print(f"data pin:        {best.config.data.snapshot}")

# %%
db.close()

# %% [markdown]
# ---
#
# # What to take away
#
# The mid is not a price. Polymarket is a plain limit order book: bids and asks
# with sizes, and a gap between them where nothing trades. Buy at the ask and sell
# at the bid and you have paid the spread, whatever the mid did in between.
#
# One number decides whether a strategy works: gross edge per share against the
# half-spread it pays. The sweep varied only trading frequency, and the sign of the
# result flipped exactly where those two crossed. Trading more often did not find
# more signal. It found the same signal and paid for it repeatedly.
#
# A losing backtest is diagnostic. Splitting the loss into gross edge and spread
# said the idea had something in it and the execution was destroying it, which
# points at trading less, or trading passively, instead of abandoning the signal.
#
# The trial count is part of the result. The winning setting looked like a 76%
# chance of being real until we admitted five settings were tried, at which point
# it fell to 29%. That correction costs one line and nobody applies it for you.
#
# Let the machine hold the caveats. The settlement gate refused to value positions
# whose outcomes were not yet knowable, `expiration_ns` refused 70 orders in a
# market that had already closed, the pin makes every run read identical bytes, and
# `report()` puts the fidelity warning at the top of the page instead of in a
# footnote the author has to remember.
#
# ## Where to go next
#
# Four things this tutorial deliberately left out:
#
# - **Signals tables versus callbacks:** a strategy can also be expressed as a
#   table of timestamped order intents. That form gives up path-dependence but
#   gains a content hash, which unlocks trial deduplication and `backtest.study`
#   parameter searches, neither of which accepts a callback. The sweep above
#   would have been a one-liner in that form.
# - **Execution as a variable:** the same strategy under added latency, or posting
#   passively instead of crossing, moves the result more than most strategy
#   changes do.
# - **Maker strategies and queue position**, which need delta-level data rather
#   than the periodic snapshots used here. The engine refuses a
#   `queue_position=True` claim on this capture instead of inventing a number.
# - **Both sides of the book**, which would make the YES/NO parity trade testable.
#
# The cookbook this article is drawn from is at
# [h5i-db-cookbook](https://github.com/h5i-dev/h5i-db-cookbook), with longer
# treatments of each in `notebooks/05_prediction_markets/`.
