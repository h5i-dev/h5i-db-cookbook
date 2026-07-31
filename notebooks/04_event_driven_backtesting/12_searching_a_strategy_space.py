# %% [markdown]
# # Searching a strategy space without fooling yourself
#
# A backtest that has been run once is a measurement. A backtest that has been
# run four hundred times and reported once is a selection, and the number on the
# slide is the maximum of four hundred draws. Everything in this recipe exists
# to keep those two things apart.
#
# There are two searches here and they are not the same shape. Strategy
# parameters change what was traded, so each candidate is a different signals
# table and a different run. Execution parameters change only what the trades
# cost, so those can be swept by `backtest.study`, which knows how to hold a
# holdout back and how to score a candidate on several windows.
#
# Recipe 05/07 does this on prediction markets. This one does it on equities,
# and adds the parts that only matter when the search is wide: random search,
# the trial ledger that makes a repeated run free, and one report across every
# surviving run.

# %% [markdown]
# ## Terms used here
#
# | term            | meaning |
# | --------------- | --- |
# | grid search     | trying every combination of the parameters you listed |
# | random search   | drawing parameter sets at random from ranges you gave |
# | walk-forward    | scoring a candidate on several train/holdout splits in time order |
# | holdout         | data used once, to score candidates that were already chosen |
# | trial ledger    | a record of every scored configuration, keyed by what it read and did |
# | basket report   | one document assembling many runs from their stored tables |
# | in-sample       | measured on the same data used to choose the parameters |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import datetime as dt
import itertools

import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, quant
import cookbook_utils as cu

CAPITAL = 250_000.0
LOT = 100.0

# %% [markdown]
# ## 1. Market data, once
#
# Ten large caps since 2022, with a book synthesized from the bars exactly as in
# recipe 04/08. Every candidate below reads this one snapshot, so no comparison
# is ever between two different tapes.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")
frame = daily.to_pandas()
frame = frame[frame["ts"] >= pd.Timestamp("2022-01-01", tz="UTC")]
market = cu.make_equity_market(
    pa.Table.from_pandas(frame, preserve_index=False), spread_bps=4.0
)

db = h5i_db.Database(cu.fresh_db("04_searching_a_strategy_space"), create=True)
for name in ("instruments", "book_deltas", "trades"):
    table = market[name]
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="4bp synthetic book")
db.snapshot("tape-v1", tables=["instruments", "book_deltas", "trades"], note="One tape for every trial")

closes = frame.pivot(index="ts", columns="symbol", values="close").sort_index()
sessions = closes.index
book = market["book_deltas"].to_pandas()
book["session"] = book["ts_init"].dt.floor("s").dt.tz_localize("UTC")
quote_at = {(row.session, row.instrument_id): row.ts_init for row in book.itertuples()}
print(f"{len(sessions):,} sessions, {closes.shape[1]} names")
closes.tail(3)

# %% [markdown]
# ## 2. One signals table per candidate
#
# The rule is a moving-average crossover: hold a fixed lot while the fast
# average is above the slow one, hold nothing otherwise. The parameters are the
# two window lengths.
#
# Those cannot be swept by `backtest.study`, and the refusal is deliberate.
# `data.signals` identifies *what was traded*, so a study that varied it would
# be comparing different strategies while claiming to compare settings of one.
# Each candidate therefore gets its own table, and the table is versioned data
# like everything else.

# %%
def crossover_signals(fast: int, slow: int) -> pa.Table:
    """Target one lot per name while the fast average leads the slow one."""
    averages = {
        "fast": closes.rolling(fast).mean(),
        "slow": closes.rolling(slow).mean(),
    }
    holding = (averages["fast"] > averages["slow"]).astype(float) * LOT
    # The last session has no session after it to execute in, so it cannot
    # carry a target. The one before it is set flat, which makes final cash the
    # whole answer instead of cash plus an open position to mark.
    holding = holding.iloc[slow:-1]
    holding.iloc[-1] = 0.0

    stamps, targets, names = [], [], []
    for session, row in holding.iterrows():
        index = sessions.get_loc(session)
        execution = sessions[index + 1]
        for symbol, target in row.items():
            stamps.append(quote_at[(execution, symbol)] + dt.timedelta(microseconds=1))
            targets.append(float(target))
            names.append(symbol)
    order = sorted(range(len(stamps)), key=lambda i: (stamps[i], names[i]))
    return backtest.target_positions(
        [stamps[i] for i in order],
        [targets[i] for i in order],
        instrument_id=[names[i] for i in order],
        tag=f"ma-{fast}-{slow}",
    )


CANDIDATES = [
    (fast, slow)
    for fast, slow in itertools.product((10, 20, 50), (50, 100, 200))
    if fast < slow
]
for fast, slow in CANDIDATES:
    table = crossover_signals(fast, slow)
    name = f"signals_{fast}_{slow}"
    db.create_table(name, table.schema, time_column="ts")
    db.append(name, table, note=f"MA {fast}/{slow}")
db.snapshot("plans-v1", tables=[f"signals_{f}_{s}" for f, s in CANDIDATES])
print(f"{len(CANDIDATES)} candidates, one signals table each")
db.table(f"signals_{CANDIDATES[0][0]}_{CANDIDATES[0][1]}").head(3).to_pandas()

# %% [markdown]
# ## 3. The in-sample leaderboard, and why it is not a result
#
# Running every candidate on the whole history gives a ranking. It is the
# ranking a search produces before any of the discipline is applied, and it is
# reported here so the later numbers have something to be compared against.

# %%
def config(run_id: str, signals: str, **execution) -> backtest.BacktestConfig:
    return backtest.BacktestConfig(
        run_id=run_id,
        portfolio=backtest.PortfolioConfig(starting_cash=CAPITAL),
        data=backtest.DataConfig(signals=signals, snapshot="tape-v1"),
        execution=backtest.ExecutionConfig(
            fee_kind="proportional", fee_rate=execution.get("fee_rate", 0.0005)
        ),
        output=backtest.OutputConfig(equity_interval_nanos=86_400_000_000_000),
    )


leaderboard = []
runs = {}
for fast, slow in CANDIDATES:
    label = f"ma-{fast}-{slow}"
    result = backtest.execute(db, config(label, f"signals_{fast}_{slow}"))
    runs[label] = result
    leaderboard.append(
        {
            "candidate": label,
            "fills": result["fills"],
            "commissions": result["commissions"],
            "net": result["final_cash"] - CAPITAL,
        }
    )
board = pd.DataFrame(leaderboard).sort_values("net", ascending=False)
board.round(2)

# %% [markdown]
# ## 4. A repeated trial is free, and does not count twice
#
# Every pinned, declarative config is a trial with a digest over everything that
# affects the replay. Submitting the same one again returns the recorded result
# instead of running it, and does not increase the trial count. That matters for
# two reasons: a retried job cannot silently multiply your effective number of
# trials, and an agent looping over configurations cannot pay twice for the same
# question.

# %%
before = backtest.trial_count(db)
again = backtest.execute(db, config("ma-20-100", "signals_20_100"))
after = backtest.trial_count(db)
print(f"cached          {again['cached']}")
print(f"trial count     {before} -> {after}")
print(f"fork reused     {again['fork'] == runs['ma-20-100']['fork']}")
print(f"same final cash {again['final_cash'] == runs['ma-20-100']['final_cash']}")
assert again["cached"] and after == before
assert again["fork"] == runs["ma-20-100"]["fork"]

# %% [markdown]
# ## 5. Sweeping what execution costs, honestly
#
# The surviving question for the leading rule is whether its edge is bigger than
# the cost of trading it. That *is* a study: the signals are fixed and only
# `execution` moves.
#
# Three pieces do the work. `RandomSearch` draws from a range instead of
# enumerating a grid, which is the better bet when a space is wide and most axes
# do not matter. `WalkForward` scores each candidate on several train/holdout
# splits so one lucky window cannot carry it. `TopK` makes the holdout a second
# stage: candidates are ranked on train, and only the best few are ever run out
# of sample.

# %%
best_label = board.iloc[0]["candidate"]
best_fast, best_slow = best_label.split("-")[1:]
edges = [sessions[i] for i in (0, len(sessions) // 3, 2 * len(sessions) // 3, len(sessions) - 1)]
folds = backtest.WalkForward.of(
    *[
        backtest.ValidationWindows(
            train=(edges[i].tz_localize(None), edges[i + 1].tz_localize(None)),
            holdout=(edges[i + 1].tz_localize(None), edges[i + 2].tz_localize(None)),
        )
        for i in range(len(edges) - 2)
    ]
)
study = backtest.study(
    db,
    study_id="cost-tolerance",
    base=config(f"study-{best_label}", f"signals_{best_fast}_{best_slow}"),
    parameters={"execution.fee_rate": backtest.Range(0.0, 0.004)},
    search=backtest.RandomSearch(trials=8, seed=5),
    validation=folds,
    selection=backtest.TopK(k=3, metric="final_cash"),
)
ranked = pd.DataFrame(study.ranked())
columns = [c for c in ("trial", "parameters", "train_median_final_cash",
                       "holdout_median_final_cash") if c in ranked.columns]
print(f"trials {len(study.trials)}, reached the holdout {len(study.selected)}")
ranked[columns].round(2)

# %% [markdown]
# The property to check in your own studies is the one this table shows: most
# candidates have no holdout column at all. A holdout that every candidate
# touched is a second training set with a nicer name.

# %%
with_holdout = sum(
    1 for row in study.trials if any(key.startswith("fold0_holdout_") for key in row)
)
print(f"{with_holdout} of {len(study.trials)} trials ever saw the holdout")
assert with_holdout == len(study.selected) <= 3

# %% [markdown]
# `TPESearch` is the third search shape. It proposes each point from the results
# so far, so it runs sequentially and needs the optional `optuna` extra. The
# call is otherwise identical:
#
# ```python
# search=backtest.TPESearch(trials=40, seed=7)
# ```
#
# Duplicate draws are kept rather than resampled, in every search shape, because
# dropping them would quietly change the trial count that the deflated-Sharpe
# correction depends on.

# %%
try:
    import optuna  # noqa: F401

    available = True
except ImportError:
    available = False
print(f"optuna installed: {available}")
print(f"trials recorded in this database so far: {backtest.trial_count(db)}")

# %% [markdown]
# ## 6. One document for the whole search
#
# Twenty forks are not a comparison. `quant.basket_payload` assembles a single
# report from the stored tables with no re-simulation: portfolio panels for the
# basket, per-run panels while the basket is small enough to read, and a record
# of anything it refused to draw.

# %%
basket = {label: result for label, result in list(runs.items())[:6]}
report = quant.basket_payload(
    db,
    basket,
    basket_id="ma-crossover search",
    panels=quant.PORTFOLIO_PANELS + ("equity",),
    snapshot="tape-v1",
)
print(f"runs      {report.totals['runs']}")
print(f"net       {report.totals['net']:,.2f}")
print(f"drawdown  {report.totals.get('max_drawdown', 0):.4f}")
print(f"panels    {', '.join(report.drawn)}")
if report.skipped:
    print(f"not drawn {[item['panel'] for item in report.skipped]}")
path = report.to_html("data/cache/ma-search-basket.html")
print(f"\nwrote {len(path):,} bytes of self-contained HTML")

# %% [markdown]
# ## 7. What the search costs the conclusion
#
# Nine candidates and eight cost draws is a small search, and it is still a
# search. The correction belongs with the result: recipe 05/05 computes the
# deflated Sharpe ratio and the probability of backtest overfitting for exactly
# this situation, and recipe 06/03 applies the same idea to a purged
# cross-validation on an equity panel.
#
# The number worth carrying out of this recipe is the trial count, because every
# honest correction needs it and only the ledger knows it.

# %%
print(f"strategy candidates run   {len(CANDIDATES)}")
print(f"execution trials run      {len(study.trials)}")
print(f"total trials in ledger    {backtest.trial_count(db)}")
print(f"best in-sample candidate  {best_label}")

# %% [markdown]
# ## Takeaways
#
# - Strategy parameters and execution parameters need different machinery: one
#   signals table per strategy candidate, `backtest.study` for the rest.
# - `RandomSearch` beats a grid when the space is wide; `TPESearch` needs the
#   `optuna` extra and runs sequentially by construction.
# - `WalkForward` stops one window carrying a candidate, and `TopK` keeps the
#   holdout for the shortlist. Check that most trials have no holdout columns.
# - The trial ledger makes a resubmitted config free and keeps the trial count
#   honest, which is what every overfitting correction is a function of.
# - `quant.basket_payload` compares runs from stored tables with no
#   re-simulation, and records what it refused to draw rather than thinning it.
# - A leaderboard is not a result until something out of sample has seen it.

# %%
db.close()
