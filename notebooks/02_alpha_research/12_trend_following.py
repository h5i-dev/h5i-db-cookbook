# %% [markdown]
# # Trend following: each asset against its own past
#
# Recipe 02/01 ranks assets against each other and buys the winners. This one
# never compares two assets at all: each is measured against its own history and
# held long, short, or not at all on that basis alone. The difference sounds
# small and changes everything about the resulting portfolio. Cross-sectional
# momentum is roughly market-neutral by construction; time-series momentum takes
# a directional position in every asset it looks at, and its whole reason for
# existing is that the position can be short.
#
# This is the core of a managed-futures book, and the two things that make it
# work are not the signal. They are volatility scaling, so that a quiet asset and
# a violent one contribute the same risk, and a turnover budget, because a fast
# trend signal trades a lot and section 04 has already priced what that costs.

# %% [markdown]
# ## Terms used here
#
# | term                 | meaning |
# | -------------------- | --- |
# | time-series momentum | going long an asset that has risen against its own past, short one that has fallen |
# | volatility scaling   | sizing each position so that every asset contributes similar risk |
# | volatility target    | the annualized volatility the whole portfolio aims at |
# | turnover             | how much of the portfolio is traded, as a fraction of its value |
# | crisis alpha         | the claim that trend following pays off in the worst months for markets |
# | plateau              | a range of parameter values that all work, as opposed to one that does |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

LOOKBACK = 252
VOL_WINDOW = 60
TARGET_VOL = 0.10
COST_BPS = 10.0
# A numpy scalar inside a builder expression renders as a function call, so the
# annualization factor is kept as a plain Python float.
ANNUALIZE = float(np.sqrt(252))

# %% [markdown]
# ## 1. The universe
#
# Thirty large caps, which is not what a CTA trades. A real book runs this
# across futures in equities, rates, currencies and commodities, precisely so
# that the positions are not all the same bet. Everything below is identical on
# that data; the diversification is the part this fixture cannot supply, and the
# result section says so rather than pretending otherwise.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("alpha_trend_following"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="Everything here reads this cut")
print(f"{prices.num_rows:,} rows, {daily.to_pandas()['symbol'].nunique()} names")
daily.to_pandas().head(3)

# %% [markdown]
# ## 2. Signal and risk, in one query
#
# Three things are computed per name per day: the daily return, the trailing
# volatility that will size the position, and the trend signal itself. The
# volatility is an exponentially weighted standard deviation, which reacts to a
# regime change faster than a rolling window of the same length.
#
# Every one of these looks backwards only. The position implied by row *t* is
# traded on *t+1*, and that shift happens explicitly below rather than being
# assumed.

# %%
previous = sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")
past = sql_expr(f"lag(adj_close, {LOOKBACK})").over(partition_by="symbol", order_by="ts")
features = (
    db.table("prices", snapshot="prices-v1")
    .with_columns(previous=previous, past=past)
    .with_columns(
        ret=col("adj_close") / col("previous") - 1,
        trend=col("adj_close") / col("past") - 1,
    )
    .filter(col("ret").is_not_null() & col("trend").is_not_null())
)
scaled = (
    features.with_columns(
        vol=col("ret").rolling_std(VOL_WINDOW, order_by="ts", partition_by="symbol")
    )
    .with_columns(annual_vol=col("vol") * ANNUALIZE)
    .select(
        ts=col("ts"),
        symbol=col("symbol"),
        ret=col("ret"),
        trend=col("trend"),
        annual_vol=col("annual_vol"),
    )
    .sort(["ts", "symbol"])
)
panel = scaled.to_pandas().dropna()
print(f"{len(panel):,} symbol-days with a signal and a volatility estimate")
panel.head(3).set_index("ts").round(4)

# %% [markdown]
# ## 3. Positions
#
# The signal decides the sign; the volatility decides the size. Each name gets a
# target risk of `TARGET_VOL / n`, so a name whose volatility doubles has its
# position halved and contributes the same amount of risk as before.
#
# Weights are capped. Without a cap, a name that went quiet takes a position
# larger than the portfolio, which is arithmetic rather than conviction.

# %%
signal = np.sign(panel["trend"])
names = panel.groupby("ts")["symbol"].transform("size")
panel["weight"] = (
    signal * (TARGET_VOL / np.sqrt(names)) / panel["annual_vol"]
).clip(-0.2, 0.2)

base = panel.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)
returns = panel.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)
unscaled = (base.shift(1).fillna(0.0) * returns).sum(axis=1)
print(f"realized volatility before the overlay: "
      f"{unscaled.std() * ANNUALIZE:.1%} against a {TARGET_VOL:.0%} target")

# %% [markdown]
# Sizing each name at `TARGET_VOL / sqrt(n)` assumes the names are independent,
# and they are not: thirty large caps move together, so the portfolio lands
# somewhere else entirely. Rather than argue about the right constant, measure
# the book's own trailing volatility and scale to the target. Every input to
# that scale is lagged, so no day's position uses that day's outcome.

# %%
realized = unscaled.rolling(VOL_WINDOW).std() * ANNUALIZE
scale = (TARGET_VOL / realized).shift(1).clip(upper=3.0).fillna(0.0)
weights = base.mul(scale, axis=0)

held = weights.shift(1).fillna(0.0)  # decided yesterday, held today
gross = (held * returns).sum(axis=1)
turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
net = gross - turnover * COST_BPS / 10_000

print(f"realized volatility after   {gross.std() * ANNUALIZE:.1%}")
print(f"average gross exposure      {held.abs().sum(axis=1).mean():.2f}x")
print(f"average net exposure        {held.sum(axis=1).mean():+.2f}x")
print(f"average daily turnover      {turnover.mean():.3f} of book")
print(f"annual cost at {COST_BPS:.0f}bp         {turnover.mean() * 252 * COST_BPS / 10_000:.2%}")

# %% [markdown]
# Net exposure is the number that says whether this is a trend book or a
# disguised long. It should spend real time on both sides of zero; if it never
# goes short, the strategy is the market with extra steps.

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(held.index, held.sum(axis=1), linewidth=1.0)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Net exposure through time")
ax.set_xlabel("Date")
ax.set_ylabel("Sum of weights")
fig.tight_layout()

# %% [markdown]
# ## 4. Performance, through the shared machinery
#
# The returns series is stored as a table and analysed with `quant.perf`, the
# same code recipe 06/01 uses. Two strategies that report their Sharpe from two
# different implementations are two numbers, not a comparison.

# %%
returns_table = pa.Table.from_pandas(
    pd.DataFrame({"ts": net.index, "ret": net.to_numpy()}), preserve_index=False
)
db.create_table("trend_returns", returns_table.schema, time_column="ts")
db.append("trend_returns", returns_table, note=f"{LOOKBACK}-day trend, {TARGET_VOL:.0%} vol target")

benchmark_table = pa.Table.from_pandas(
    pd.DataFrame({"ts": returns.index, "ret": returns.mean(axis=1).to_numpy()}),
    preserve_index=False,
)
db.create_table("universe_returns", benchmark_table.schema, time_column="ts")
db.append("universe_returns", benchmark_table, note="equal-weight universe")
db.snapshot("returns-v1", tables=["trend_returns", "universe_returns"])

strategy = quant.returns(db, "trend_returns", snapshot="returns-v1")
universe = quant.returns(db, "universe_returns", snapshot="returns-v1")
comparison = pd.DataFrame(
    {"trend (net)": strategy.stats(benchmark=universe), "universe": universe.stats()}
)
comparison.round(4)

# %% [markdown]
# Read the volatility line before the return line. The point of a volatility
# target is that the number it produces is roughly the number that was asked
# for; if the realized volatility is nowhere near `TARGET_VOL`, the sizing is
# not doing its job and the return is a different strategy's return.
#
# The universe beats the strategy over this sample, and that is the expected
# result rather than a bug. A trend book on thirty correlated mega-caps during a
# long bull market is being asked to do the one thing it is worst at. Section 7
# quantifies why.

# %%
print(f"target volatility   {TARGET_VOL:.1%}")
print(f"realized volatility {strategy.stats()['annual_volatility']:.1%}")
print(f"beta to the universe {strategy.stats(benchmark=universe)['beta']:+.2f}")

# %%
curve = strategy.equity_curve().to_pandas()
bench_curve = universe.equity_curve().to_pandas()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(curve["ts"], curve["cumulative_return"], linewidth=1.6, label="trend, net of costs")
ax.plot(bench_curve["ts"], bench_curve["cumulative_return"], linewidth=1.2, label="equal-weight universe")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title(f"{LOOKBACK}-day time-series momentum at a {TARGET_VOL:.0%} volatility target")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 5. The claim worth testing: what happens on the bad days
#
# Trend following is sold on its behaviour in drawdowns, not on its average
# return. The testable version of that claim is narrow: on the worst days for
# the universe, is the strategy's return better than a long position would have
# been?

# %%
joined = pd.DataFrame({"strategy": net, "universe": returns.mean(axis=1)}).dropna()
buckets = pd.qcut(joined["universe"], 5, labels=["worst", "poor", "flat", "good", "best"])
by_bucket = joined.groupby(buckets, observed=True).agg(
    days=("strategy", "size"),
    universe=("universe", "mean"),
    strategy=("strategy", "mean"),
)
by_bucket["difference"] = by_bucket["strategy"] - by_bucket["universe"]
by_bucket.round(5)

# %% [markdown]
# On daily data the honest answer is usually "it cushions, it does not hedge".
# A day is far shorter than the horizon this signal reacts on, so the protection
# such a book actually provides shows up over months of a sustained decline, not
# over the single worst sessions. Quoting the daily version as crisis alpha is
# the usual overstatement.

# %%
monthly = joined.resample("ME").apply(lambda column: (1 + column).prod() - 1)
worst_months = monthly.nsmallest(6, "universe")
print("the six worst months for the universe:")
print((worst_months * 100).round(2).to_string())
print(f"\nstrategy beat the universe in {int((worst_months['strategy'] > worst_months['universe']).sum())} of 6")

# %% [markdown]
# ## 6. Is the lookback a choice or a fit?
#
# A parameter that works at 252 days and nowhere else is a fitted parameter. A
# plateau across neighbouring horizons is the weakest evidence worth having, and
# recipe 06/03 turns the same question into a probability.

# %%
def run_lookback(days: int) -> dict:
    lag = sql_expr(f"lag(adj_close, {days})").over(partition_by="symbol", order_by="ts")
    frame = (
        db.table("prices", snapshot="prices-v1")
        .with_columns(previous=previous, past=lag)
        .with_columns(
            ret=col("adj_close") / col("previous") - 1,
            trend=col("adj_close") / col("past") - 1,
        )
        .filter(col("ret").is_not_null() & col("trend").is_not_null())
        .with_columns(
            vol=col("ret").rolling_std(VOL_WINDOW, order_by="ts", partition_by="symbol")
        )
        .select(
            ts=col("ts"),
            symbol=col("symbol"),
            ret=col("ret"),
            trend=col("trend"),
            annual_vol=col("vol") * ANNUALIZE,
        )
        .sort(["ts", "symbol"])
        .to_pandas()
        .dropna()
    )
    count = frame.groupby("ts")["symbol"].transform("size")
    frame["weight"] = (
        np.sign(frame["trend"]) * (TARGET_VOL / np.sqrt(count)) / frame["annual_vol"]
    ).clip(-0.2, 0.2)
    b = frame.pivot(index="ts", columns="symbol", values="weight").fillna(0.0)
    r = frame.pivot(index="ts", columns="symbol", values="ret").fillna(0.0)
    raw = (b.shift(1).fillna(0.0) * r).sum(axis=1)
    overlay = (TARGET_VOL / (raw.rolling(VOL_WINDOW).std() * ANNUALIZE)).shift(1).clip(upper=3.0).fillna(0.0)
    w = b.mul(overlay, axis=0)
    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    series = (w.shift(1).fillna(0.0) * r).sum(axis=1) - turn * COST_BPS / 10_000
    return {
        "lookback": days,
        "sharpe": float(series.mean() / series.std() * np.sqrt(252)),
        "annual turnover": float(turn.mean() * 252),
        "days": len(series),
    }


plateau = pd.DataFrame([run_lookback(days) for days in (63, 126, 189, 252, 315, 378)])
plateau.round(3)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(plateau["lookback"], plateau["sharpe"], marker="o", linewidth=1.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Net Sharpe against trend lookback")
ax.set_xlabel("Lookback (trading days)")
ax.set_ylabel("Net Sharpe")
fig.tight_layout()

# %% [markdown]
# ## 7. What this fixture cannot show
#
# Thirty large-cap equities are one bet wearing thirty hats. A managed-futures
# book earns its reputation from holding trends in rates, currencies and
# commodities at the same time, which is where the correlation between positions
# is low enough for the volatility target to mean something.
#
# The average pairwise correlation below is the number that says so.

# %%
correlation = returns.corr().to_numpy()
off_diagonal = correlation[~np.eye(len(correlation), dtype=bool)]
print(f"average pairwise correlation of the universe: {off_diagonal.mean():.2f}")
print(f"effective independent bets (1/(1+(n-1)r)) * n: "
      f"{len(correlation) / (1 + (len(correlation) - 1) * off_diagonal.mean()):.1f} "
      f"of {len(correlation)} names")

# %% [markdown]
# ## Takeaways
#
# - Time-series momentum compares an asset to its own past, so it takes
#   directional positions and can be short. That is the whole difference from
#   recipe 02/01.
# - Volatility scaling is what makes positions comparable; check that realized
#   volatility lands near the target before reading the return.
# - Turnover is a cost, and the cost belongs in the returns series rather than
#   in a footnote. Recipe 04/11 is where the number comes from.
# - "Crisis alpha" is a monthly claim, not a daily one, and the daily table says
#   so plainly.
# - Judge the lookback by the plateau, not by the peak.
# - A universe of correlated equities cannot show what diversification does for
#   this strategy. Say what the fixture cannot support.

# %%
db.close()
