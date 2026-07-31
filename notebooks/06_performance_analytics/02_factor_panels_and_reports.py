# %% [markdown]
# # Factor panels: IC, quantiles, turnover
#
# Recipe 02/05 builds a factor and scores it with pandas and scipy. That is the
# right way to learn what an information coefficient is, and the wrong way to
# run a factor library: every desk that does it ends up with three
# implementations of the same rank correlation and no agreement about which is
# authoritative.
#
# `quant.build_panel` is the shared implementation. It joins a factor to forward
# returns, assigns quantiles, and exposes the whole `alphalens` surface -- IC,
# decay, quantile returns, spreads, turnover, factor-portfolio alpha -- as
# queries against the engine. The panel is a pinned query rather than a frame,
# so the numbers carry the data version they came from and a hundred thousand
# rows never leave the database.

# %% [markdown]
# ## Terms used here
#
# | term                | meaning |
# | ------------------- | --- |
# | factor              | a number per asset per date that is supposed to predict returns |
# | forward return      | the return over the next *n* bars, the thing a factor is scored against |
# | information coefficient | the rank correlation between the factor and the return that followed |
# | quantile bucket     | assets split into equal-count groups by factor value each date |
# | spread              | the return of the top bucket minus the bottom one |
# | turnover            | how much of a bucket's membership changes from one date to the next |
# | rank autocorrelation| how stable an asset's factor rank is over time |
# | group neutral       | comparing assets only against others in their own sector |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

# %% [markdown]
# ## 1. Prices and a factor, both long
#
# Both inputs are long format: `(ts, asset, value)`. Here the factor is 12-1
# momentum and the price is the adjusted close, and both come from one table.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_factor_panels_and_reports"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="The cut the panel is pinned to")
print(f"{prices.num_rows:,} rows, {daily.to_pandas()['symbol'].nunique()} names")
daily.to_pandas().head(3)

# %% [markdown]
# A lazy frame passed to `build_panel` is taken as given, which means **the pin
# has to be on the frame**. `db.table(name, snapshot=...)` puts it there. Passing
# an unpinned frame together with `snapshot=` would produce a panel that claims
# a pin in its provenance and reads the latest version anyway.

# %%
pinned = db.table("prices", snapshot="prices-v1")
price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
factor_frame = (
    pinned.with_columns(
        month_ago=sql_expr("lag(adj_close, 21)").over(partition_by="symbol", order_by="ts"),
        year_ago=sql_expr("lag(adj_close, 252)").over(partition_by="symbol", order_by="ts"),
    )
    .with_columns(momentum=col("month_ago") / col("year_ago") - 1)
    .select(ts=col("ts"), asset=col("symbol"), factor=col("momentum"))
)
print(factor_frame.sql())

# %% [markdown]
# ## 2. The panel
#
# `periods` are forward-return horizons in bars, so on daily data these are one
# day, one week and one month. `quantiles=5` splits each date into five
# equal-count buckets. `filter_zscore` drops factor values more than twenty
# standard deviations from the mean, which is a data-error filter rather than a
# modelling choice.

# %%
panel = quant.build_panel(
    db,
    factor_frame,
    price_frame,
    periods=(1, 5, 21),
    quantiles=5,
    filter_zscore=20.0,
    max_loss=0.35,
)
print(panel)
panel.collect().to_pandas().head()

# %% [markdown]
# ## 3. What the panel threw away
#
# `loss_report` is the accounting `alphalens` prints and most users skip. Rows
# are lost joining to forward returns (no price far enough ahead, a non-finite
# factor value) and lost again in binning. `max_loss` refuses the panel outright
# when too much of the factor never made it, because a factor scored on the
# surviving 40% of its own values is a different factor.

# %%
report = panel.loss_report()
print(f"factor rows          {report['initial']:,}")
print(f"after forward returns{report['after_forward_returns']:>8,}  "
      f"({report['forward_returns']:.2%} lost)")
print(f"after binning        {report['after_binning']:>8,}  ({report['binning']:.2%} lost)")
print(f"total loss           {report['total']:.2%} against a limit of {report['max_loss']:.0%}")

try:
    quant.build_panel(db, factor_frame, price_frame, periods=(1, 5, 21), max_loss=0.001)
except quant.MaxLossExceededError as error:
    print(f"\nrefused at a 0.1% limit: {str(error)[:120]}...")

# %% [markdown]
# ## 4. Information coefficient
#
# The per-date rank correlation between the factor and each forward return.
# Ranks use the engine's `cs_rank`, which follows pandas' tie rule, so these
# match `scipy.stats.spearmanr` rather than SQL's `percent_rank`.

# %%
ic = panel.ic().to_pandas()
print(f"{len(ic):,} dates scored")
print(pd.DataFrame({"mean IC": panel.mean_ic().to_pandas().iloc[0]}).round(4).to_string())
ic.tail(3).set_index("ts").round(4)

# %% [markdown]
# `ic_decay` is the same question asked across horizons in one query: how long
# does the signal survive? The `t_stat` column is what says whether a mean IC of
# 0.02 is a signal or a rounding error.

# %%
decay = panel.ic_decay().to_pandas()
decay.round(4)

# %% [markdown]
# Monthly resampling turns a noisy daily series into something you can look at.
# It does not make a weak factor stronger; it makes a weak factor legible.

# %%
monthly_ic = panel.mean_ic(by="1mo").to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(monthly_ic["bucket"], monthly_ic["ic_21"], width=20)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Monthly mean IC, 21-day horizon")
ax.set_xlabel("Month")
ax.set_ylabel("Information coefficient")
fig.tight_layout()

# %% [markdown]
# ## 5. Quantile returns and the spread
#
# The IC says the ranking is informative. The quantile table says whether the
# information is in the part of the distribution you can trade, and the standard
# errors say whether the difference between two buckets is a difference at all.

# %%
quantiles = panel.quantile_returns().to_pandas()
quantiles.round(5)

# %% [markdown]
# `spread` is the top bucket minus the bottom one, per date, with a joint
# standard error. It is the closest thing in the panel to a tradeable series,
# and it is still gross of every cost in section 04.

# %%
spread = panel.spread().to_pandas()
summary = {
    f"spread_{period}": {
        "mean": spread[f"spread_{period}"].mean(),
        "t-stat": spread[f"spread_{period}"].mean()
        / spread[f"spread_{period}"].std()
        * len(spread) ** 0.5,
    }
    for period in (1, 5, 21)
}
pd.DataFrame(summary).round(4)

# %%
cumulative = panel.cumulative_returns(period=21).to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(cumulative["ts"], cumulative["cumulative_return"], linewidth=1.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor-weighted portfolio, 21-day horizon")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return")
fig.tight_layout()

# %% [markdown]
# `alpha_beta` regresses the factor portfolio against the equally weighted
# universe, one row per horizon, so the question "is this just market exposure"
# gets an answer rather than an argument.

# %%
pd.DataFrame(panel.alpha_beta()).round(4)

# %% [markdown]
# ## 6. What it costs to hold
#
# Turnover and rank autocorrelation are the two numbers that decide whether an
# IC survives contact with recipe 04/11. A factor whose top bucket changes
# completely every day needs an enormous IC to pay for itself.

# %%
turnover = panel.turnover(period=1).to_pandas()
autocorrelation = panel.rank_autocorrelation(period=1).to_pandas()
print(f"mean daily turnover by bucket:")
print(turnover.groupby("factor_quantile")["turnover"].mean().round(4).to_string())
print(f"\nmean rank autocorrelation: {autocorrelation['autocorrelation'].mean():.4f}")
print(f"implied average holding period: "
      f"{1 / max(turnover['turnover'].mean(), 1e-9):.1f} days")

# %% [markdown]
# ## 7. Sectors, and what neutrality changes
#
# A momentum factor that loads on one sector is a sector bet with extra steps.
# `group` accepts a mapping from asset to group; `by_group` scores each sector
# separately, and `group_adjust` demeans the forward returns within a sector
# before scoring, which is the question "does this work *within* a sector".

# %%
SECTORS = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "META": "tech",
    "CSCO": "tech", "IBM": "tech", "V": "financials", "JPM": "financials",
    "BAC": "financials", "GS": "financials", "BRK-B": "financials",
    "XOM": "energy", "CVX": "energy", "UNH": "health", "JNJ": "health",
    "MRK": "health", "ABBV": "health", "PG": "staples", "PEP": "staples",
    "KO": "staples", "COST": "staples", "WMT": "staples", "AMZN": "discretionary",
    "HD": "discretionary", "MCD": "discretionary", "DIS": "discretionary",
    "CAT": "industrials", "GE": "industrials", "T": "telecom",
}
grouped = quant.build_panel(
    db,
    factor_frame,
    price_frame,
    periods=(1, 5, 21),
    quantiles=5,
    group=SECTORS,
    max_loss=0.35,
)
by_sector = grouped.mean_ic(by_group=True).to_pandas()
print(by_sector.round(4).to_string(index=False))

# %% [markdown]
# `telecom` has one member, and a group of one has no cross-section to rank, so
# its IC is undefined rather than zero. That is the correct answer and a useful
# reminder that group statistics inherit the group's size.

# %%
plain = grouped.mean_ic().to_pandas().iloc[0]
neutral = grouped.mean_ic(group_adjust=True).to_pandas().iloc[0]
pd.DataFrame({"raw": plain, "sector-neutral": neutral}).round(4)

# %% [markdown]
# The answer here is uncomfortable and worth stating plainly: most of this
# factor's information coefficient is a sector bet. Demeaning forward returns
# within sectors cuts the one-day IC by more than half and turns the one-month
# IC negative. A momentum book run against this universe would have been long
# whichever sector had been running.

# %% [markdown]
# ## 8. The report
#
# `quant.factor_report` renders the same page `alphalens` prints, as a
# self-contained HTML file with the provenance header attached.

# %%
html = quant.factor_report(panel, path="data/cache/momentum-factor-report.html")
payload = quant.report_payload(panel)
print(f"wrote {len(html):,} bytes")
print(f"tables {[table['id'] for table in payload.get('tables', [])]}")
print(f"charts {[chart['id'] for chart in payload.get('charts', [])]}")

# %% [markdown]
# ## 9. A panel is a query, not a result
#
# Nothing above materialised the panel in Python. `frame` exposes it as a lazy
# frame so a downstream query can filter, join or aggregate it inside the
# engine, and `sql()` prints exactly what would run.

# %%
recent = (
    panel.frame.filter(col("ts") >= "2026-01-01")
    .group_by("factor_quantile")
    .agg(days=col("fwd_21").count(), mean_fwd_21=col("fwd_21").mean())
    .sort("factor_quantile")
)
print(f"panel SQL is {len(panel.sql()):,} characters; nothing was collected to get here")
recent.to_pandas().round(5)

# %% [markdown]
# ## Takeaways
#
# - `build_panel` joins factor to forward returns and buckets once; everything
#   else is a query against that one definition.
# - Read `loss_report` before reading the IC. A factor scored on the rows that
#   survived is not the factor you built.
# - `ic_decay` gives horizon and t-statistic in one query, which is the
#   difference between "there is a signal" and "there is a number".
# - Turnover and rank autocorrelation decide whether an IC can pay for itself.
# - `group` turns a sector bet into a testable claim, and `group_adjust` asks
#   whether the factor works inside a sector at all.
# - The panel is a pinned query. Nothing leaves the engine until an aggregate
#   small enough to read comes back.

# %%
db.close()
