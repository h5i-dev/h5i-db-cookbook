# %% [markdown]
# # Cross-validation that a time series survives
#
# A backtest measures what happened. Cross-validation is supposed to answer a
# harder question: was the parameter that won chosen for a reason, or was it the
# luckiest of the ones tried? Ordinary k-fold cannot answer it on market data,
# for two reasons that have nothing to do with the model.
#
# The first is overlap. A signal held for a month has a label that spans a month,
# so an observation in the training set whose holding period reaches into the
# test block has already seen the test period. The second is serial correlation:
# the observations either side of a fold boundary are nearly the same
# observation, so a test set adjacent to its training set is not out of sample in
# any useful way.
#
# `quant.validation` fixes both by construction -- purging the training rows
# whose labels overlap, and embargoing the ones immediately after each test block
# -- and `quant.overfitting` prices what is left of the result after the search
# that produced it.

# %% [markdown]
# ## Terms used here
#
# | term            | meaning |
# | --------------- | --- |
# | label           | the outcome an observation is scored against, here a forward return |
# | horizon         | how many observations forward a label reaches |
# | purging         | dropping training rows whose label overlaps the test block |
# | embargo         | dropping training rows just after a test block, because they are nearly the same data |
# | walk-forward    | training on the past and testing on what came next, repeatedly |
# | CPCV            | combinatorial purged cross-validation: many test paths instead of one |
# | PBO             | probability of backtest overfitting: how often the in-sample winner loses out of sample |
# | deflated Sharpe | a Sharpe ratio corrected for how many strategies were tried |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

HOLD = 21

# %% [markdown]
# ## 1. Candidates to choose between
#
# The search is over one parameter: the momentum lookback. Each candidate
# becomes a daily return series for the long-short spread between its top and
# bottom quantile, computed in the engine through `quant.build_panel`.
#
# One series per candidate over the same dates is exactly the matrix every tool
# below wants.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_cross_validation_without_leakage"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="Every candidate reads this cut")

LOOKBACKS = (21, 42, 63, 126, 189, 252)
pinned = db.table("prices", snapshot="prices-v1")
price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))

series = {}
for lookback in LOOKBACKS:
    factor = (
        pinned.with_columns(
            past=sql_expr(f"lag(adj_close, {lookback})").over(
                partition_by="symbol", order_by="ts"
            )
        )
        .with_columns(value=col("adj_close") / col("past") - 1)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("value"))
    )
    panel = quant.build_panel(db, factor, price_frame, periods=(HOLD,), quantiles=5)
    spread = panel.spread().to_pandas()[["ts", f"spread_{HOLD}"]]
    series[f"mom_{lookback}"] = spread.set_index("ts")[f"spread_{HOLD}"]

candidates = pd.DataFrame(series).dropna()
candidates.index = pd.DatetimeIndex(candidates.index)
print(f"{len(candidates):,} dates x {candidates.shape[1]} candidates")
candidates.tail(3).round(4)

# %% [markdown]
# Each column is the return of holding the top quantile against the bottom for
# `HOLD` days, dated by the day the position was formed. That dating is what
# makes the overlap explicit: the label of every row reaches 21 rows forward.

# %%
scores = candidates.mean() / candidates.std() * np.sqrt(252)
print("in-sample Sharpe of each candidate, whole history:")
print(scores.round(3).to_string())
print(f"\nbest in sample: {scores.idxmax()}")

# %% [markdown]
# ## 2. What purging removes
#
# `purged_kfold` cuts the sample into contiguous blocks and, for each, drops the
# training rows whose labels reach into the test block or fall inside the
# embargo. The count it drops is the size of the leak an unpurged split would
# have had.

# %%
n = len(candidates)
horizons = [HOLD] * n
splits = list(quant.purged_kfold(n, folds=5, horizons=horizons, embargo=0.01))
folds = pd.DataFrame(
    [
        {
            "fold": index,
            "train": split.train_size,
            "test": split.test_size,
            "purged": len(split.purged),
            "purged %": len(split.purged) / (split.train_size + len(split.purged)),
        }
        for index, split in enumerate(splits)
    ]
)
folds.round(4)

# %% [markdown]
# The purged rows are the leak, and they are not evenly spread. An interior fold
# has a test block on both sides of its training data, so it loses twice as much
# as the first or last one. Naive k-fold keeps every one of these rows.

# %%
fig, ax = plt.subplots(figsize=(9, 3.2))
for index, split in enumerate(splits):
    ax.scatter(split.test, [index] * split.test_size, s=2, color="#e45756", label="test" if index == 0 else None)
    ax.scatter(split.purged, [index] * len(split.purged), s=6, color="#f58518", label="purged" if index == 0 else None)
    ax.scatter(split.train, [index] * split.train_size, s=1, color="#4c78a8", alpha=0.4, label="train" if index == 0 else None)
ax.set_title(f"Purged 5-fold with a {HOLD}-day horizon and a 1% embargo")
ax.set_xlabel("Observation index")
ax.set_ylabel("Fold")
ax.legend(loc="upper right", markerscale=4)
fig.tight_layout()

# %% [markdown]
# ## 3. Does the leak change the answer?
#
# Selecting on unpurged folds and selecting on purged ones are two procedures.
# Running both and comparing the choice is the only way to find out whether the
# leak mattered on this data, and how much of the in-sample score it was worth.

# %%
def sharpe(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or values.std(ddof=1) == 0:
        return float("nan")
    return float(values.mean() / values.std(ddof=1) * np.sqrt(252))


naive = [
    quant.validation.Split(
        train=tuple(i for i in range(n) if not (split.test[0] <= i <= split.test[-1])),
        test=split.test,
    )
    for split in splits
]

comparison = []
for label, chosen in (("naive k-fold", naive), ("purged + embargo", splits)):
    train_scores, test_scores = {}, {}
    for name in candidates.columns:
        column = candidates[name].to_numpy()
        train_scores[name] = np.mean([sharpe(column[list(s.train)]) for s in chosen])
        test_scores[name] = np.mean([sharpe(column[list(s.test)]) for s in chosen])
    winner = max(train_scores, key=train_scores.get)
    comparison.append(
        {
            "procedure": label,
            "train rows": sum(s.train_size for s in chosen),
            "picked": winner,
            "train Sharpe of pick": train_scores[winner],
            "test Sharpe of pick": test_scores[winner],
        }
    )
pd.DataFrame(comparison).round(3)

# %% [markdown]
# On this data both procedures pick the same candidate, and the purged version
# reaches that decision on about two per cent fewer training rows. That is a
# result worth having and it is not the same as "purging does not matter": the
# only way to say the leak did not change the answer is to compute the answer
# both ways. When the two disagree, the unpurged one is the one that saw the
# test set.

# %% [markdown]
# ## 4. Many paths, not one number
#
# Five folds give one estimate of out-of-sample performance, and that estimate
# has its own error bars that nobody ever quotes. Combinatorial purged
# cross-validation tests on every combination of blocks instead, so the output is
# a *distribution* of results for the same candidate.

# %%
paths = list(quant.combinatorial_purged(n, groups=6, test_groups=2, horizons=horizons, embargo=0.01))
picked = comparison[-1]["picked"]
column = candidates[picked].to_numpy()
path_scores = np.array([sharpe(column[list(split.test)]) for split in paths])
print(f"{len(paths)} paths from C(6, 2)")
print(f"median   {np.nanmedian(path_scores):.3f}")
print(f"quartiles{np.nanpercentile(path_scores, 25):8.3f} to {np.nanpercentile(path_scores, 75):.3f}")
print(f"worst    {np.nanmin(path_scores):.3f}")
print(f"share of paths below zero: {(path_scores < 0).mean():.0%}")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(path_scores, bins=12, color="#4c78a8", alpha=0.8)
ax.axvline(np.nanmedian(path_scores), color="black", linewidth=1.2, label="median")
ax.axvline(0, color="#e45756", linewidth=1.0, linestyle="--")
ax.set_title(f"Out-of-sample Sharpe of {picked} across {len(paths)} CPCV paths")
ax.set_xlabel("Annualized Sharpe on the held-out blocks")
ax.set_ylabel("Paths")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 5. Walk-forward, which is what production looks like
#
# Cross-validation reuses the future as training data for earlier folds. That is
# legitimate for *evaluating a procedure* and dishonest as a description of what
# a desk could have run. Walk-forward only ever trains on the past.

# %%
rolling = list(quant.walk_forward(n, train_size=500, test_size=125, horizons=horizons, embargo=0.01))
expanding = list(
    quant.walk_forward(n, train_size=500, test_size=125, horizons=horizons, embargo=0.01, expanding=True)
)
def walk_selection(chosen) -> pd.DataFrame:
    """Pick on each fold's training window, score on the block that follows."""
    picks = []
    for index, split in enumerate(chosen):
        train_pick = max(
            candidates.columns,
            key=lambda name: sharpe(candidates[name].to_numpy()[list(split.train)]),
        )
        picks.append(
            {
                "fold": index,
                "picked": train_pick,
                "test Sharpe": sharpe(candidates[train_pick].to_numpy()[list(split.test)]),
            }
        )
    return pd.DataFrame(picks)


walk = pd.DataFrame(
    [
        {
            "window": label,
            "folds": len(chosen),
            "first train size": chosen[0].train_size,
            "last train size": chosen[-1].train_size,
            "changed its mind": walk_selection(chosen)["picked"].nunique(),
            "mean test Sharpe of its pick": walk_selection(chosen)["test Sharpe"].mean(),
        }
        for label, chosen in (("rolling", rolling), ("expanding", expanding))
    ]
)
walk.round(3)

# %% [markdown]
# The two differ only in whether the training window drops its oldest data. That
# choice is a claim about how fast the market changes, and it belongs in the
# writeup rather than in a default.

# %%
walk_choices = walk_selection(rolling)
print(f"the rolling procedure changed its mind {walk_choices['picked'].nunique()} times")
print(f"mean test Sharpe of what it picked: {walk_choices['test Sharpe'].mean():.3f}")
walk_choices.round(3)

# %% [markdown]
# Two things in that table are worth more than the average. The procedure keeps
# changing its mind, which is what a parameter with no stable edge looks like
# from the inside. And the individual fold Sharpes swing across a range no
# annualized number should: a 125-day block is far too short for the statistic
# to mean anything on its own, which is the entire argument for looking at the
# distribution in section 4 rather than at any single fold.

# %% [markdown]
# ## 6. Pricing the search
#
# Every number above came out of a search over six candidates. `quant.overfitting`
# turns that count into a correction rather than a caveat.
#
# `probability_of_backtest_overfitting` takes the whole matrix -- one column per
# candidate, over the same dates -- and asks how often the in-sample winner
# lands in the bottom half out of sample. A PBO near one half means selection
# carried no information at all.

# %%
pbo = quant.probability_of_backtest_overfitting(candidates.to_numpy(), partitions=8)
print(f"PBO                  {pbo.pbo:.3f}")
print(f"splits evaluated     {pbo.splits}")
print(f"strategies compared  {pbo.strategies}")

# %% [markdown]
# The deflated Sharpe asks the complementary question about the winner alone:
# given that six candidates were tried, how much of its Sharpe is the expected
# maximum of six draws?

# %%
best = candidates[picked].to_numpy()
deflated = quant.deflated_sharpe(best, trials=len(LOOKBACKS), trials_source="counted")
print(f"observed Sharpe (per day)  {deflated.sharpe:.4f}")
print(f"benchmark from {deflated.trials} trials     {deflated.benchmark:.4f}")
print(f"probability it is real     {deflated.probability:.3f}")
print(f"significant at 95%         {deflated.is_significant}")

needed = quant.minimum_track_record_length(best, benchmark=0.0, confidence=0.95)
print(f"\nobservations available     {len(best):,}")
print(f"observations needed        {needed:,.0f}")

# %% [markdown]
# `trials_source="counted"` is not decoration. A trial count that was counted
# from a ledger and one that was asserted by the analyst deserve different
# amounts of trust, and recipe 04/12 shows where the count comes from when the
# search ran through `backtest.study`.

# %% [markdown]
# The two answers point in opposite directions and both are correct, because
# they are different questions. The deflated Sharpe asks whether *this series*
# beats what six coin flips would have produced, and it does. PBO asks whether
# picking the in-sample winner is a good way to choose, and it is not: the
# candidates are variations of one idea, highly correlated and mostly positive,
# so the family carries an edge while the ranking within it is noise.
#
# The practical reading is to trade the family rather than the winner, or to
# report both numbers and let the reader see the tension.

# %%
verdict = pd.DataFrame(
    [
        {"question": "does selection carry information?", "number": f"PBO {pbo.pbo:.2f}",
         "answer": "yes" if pbo.pbo < 0.5 else "no"},
        {"question": "is the winner's Sharpe real?", "number": f"p {deflated.probability:.2f}",
         "answer": "yes" if deflated.is_significant else "not yet"},
        {"question": "is the track record long enough?",
         "number": f"{len(best):,} of {needed:,.0f}",
         "answer": "yes" if len(best) >= needed else "no"},
    ]
)
verdict

# %% [markdown]
# ## Takeaways
#
# - Overlapping labels leak. `horizons` is how you tell the splitter how far a
#   label reaches, and leaving it out is a claim that labels are instantaneous.
# - The embargo handles what purging does not: observations adjacent to a test
#   block are nearly the test block.
# - CPCV returns a distribution of out-of-sample results. The width of that
#   distribution is the honest error bar on a backtest.
# - Walk-forward is the only scheme that describes what a desk could have run;
#   cross-validation evaluates a procedure, not a track record.
# - PBO, the deflated Sharpe and the minimum track record length are three
#   different questions, and a result should answer all three.
# - Every candidate here read one pinned snapshot, so the comparison is between
#   parameters rather than between data.

# %%
db.close()
