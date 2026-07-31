# %% [markdown]
# # Tearsheets and performance statistics
#
# Every recipe so far ends with a P&L number and a chart drawn by hand. That is
# fine for one strategy and hopeless for twenty, and it is how two people end up
# quoting two different Sharpe ratios for the same run.
#
# `h5i_db.quant` is the shared answer: a returns series is an object with a pin
# under it, the statistics are queries against the engine rather than a pandas
# frame in someone's notebook, and the arithmetic matches `empyrical`, which is
# what `pyfolio` was a wrapper over. The part neither of those libraries could
# offer is the header: every number carries the data version it was computed
# from, and `quant.verify` refuses to bless a result that was not pinned.

# %% [markdown]
# ## Terms used here
#
# | term              | meaning |
# | ----------------- | --- |
# | returns series    | one simple (non-cumulative) return per period, the input every statistic needs |
# | annualization     | how many periods make a year: 252 for daily bars, 12 for monthly |
# | Sharpe ratio      | mean return divided by its volatility, annualized |
# | Sortino ratio     | the same idea counting only downside volatility |
# | drawdown          | how far below its running peak an equity curve is |
# | Calmar ratio      | annual return divided by the worst drawdown |
# | alpha and beta    | return not explained by a benchmark, and sensitivity to it |
# | tearsheet         | the standard one-page performance report |
# | provenance        | the pin, parameters and query that produced a number |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, col, quant, sql_expr, time_bucket
import cookbook_utils as cu

# %% [markdown]
# ## 1. Two return series
#
# The benchmark is an equally weighted portfolio of the whole universe,
# rebalanced daily. The strategy is the 12-1 momentum rule from recipe 02/01:
# each month, hold the top three names in equal weight.
#
# Both are computed in the engine and stored as tables, because a returns series
# is data. Recomputing it in a notebook every time is how two tearsheets of the
# same strategy come to disagree.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_tearsheets_and_performance_stats"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="The price cut every number here reads")
print(f"{prices.num_rows:,} rows x {prices.num_columns} columns, "
      f"{daily.to_pandas()['symbol'].nunique()} names")
daily.to_pandas().head()

# %% [markdown]
# Daily returns per name, then the cross-sectional average, is the benchmark.
# `lag` needs the window escape hatch; everything else is a verb.

# %%
previous = sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")
returns_frame = (
    db.table("prices", snapshot="prices-v1")
    .with_columns(previous=previous)
    .with_columns(ret=col("adj_close") / col("previous") - 1)
    .filter(col("ret").is_not_null())
)
benchmark_table = (
    returns_frame.group_by("ts").agg(ret=col("ret").mean()).sort("ts").to_arrow()
)
db.create_table("benchmark_returns", benchmark_table.schema, time_column="ts")
db.append("benchmark_returns", benchmark_table, note="equal-weight universe")
print(f"{benchmark_table.num_rows:,} daily observations")
benchmark_table.to_pandas().tail(3)

# %% [markdown]
# The strategy's monthly holdings come from the same query, and the daily return
# of an equal-weight basket is the average return of whatever it held that day.

# %%
monthly = (
    db.table("prices", snapshot="prices-v1")
    .with_columns(month=time_bucket("1mo", col("ts")))
    .group_by("symbol", "month")
    .agg(close=col("adj_close").last("ts"))
    .with_columns(
        lag_1=sql_expr("lag(close, 1)").over(partition_by="symbol", order_by="month"),
        lag_12=sql_expr("lag(close, 12)").over(partition_by="symbol", order_by="month"),
    )
    .with_columns(momentum=col("lag_1") / col("lag_12") - 1)
    .filter(col("momentum").is_not_null())
    .to_pandas()
)
picks = (
    monthly.sort_values("momentum", ascending=False)
    .groupby("month")
    .head(3)[["month", "symbol"]]
)
daily_returns = returns_frame.select(
    ts=col("ts"), symbol=col("symbol"), ret=col("ret")
).to_pandas()
daily_returns["month"] = (
    daily_returns["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    .dt.tz_localize("UTC")
)
held = daily_returns.merge(
    picks.assign(month=lambda frame: frame["month"] + pd.offsets.MonthBegin(1)),
    on=["month", "symbol"],
)
strategy_table = pa.Table.from_pandas(
    held.groupby("ts", as_index=False)["ret"].mean().sort_values("ts"),
    preserve_index=False,
)
db.create_table("strategy_returns", strategy_table.schema, time_column="ts")
db.append("strategy_returns", strategy_table, note="12-1 momentum, top three, monthly")
db.snapshot("returns-v1", tables=["strategy_returns", "benchmark_returns"])
print(f"{strategy_table.num_rows:,} daily observations while invested")
strategy_table.to_pandas().tail(3)

# %% [markdown]
# ## 2. Opening a series
#
# `quant.returns` takes a table (or a lazy frame) with a timestamp and a return
# column and gives back a `ReturnSeries`. Nothing is computed yet: the series is
# a pinned query, and each statistic below runs in the engine.
#
# `annualization` is the number of periods in a year. The constants cover the
# usual cases, and anything else is a number: `24 * 365` for hourly crypto,
# `78` for five-minute bars in an equity session.

# %%
strategy = quant.returns(db, "strategy_returns", snapshot="returns-v1",
                         annualization=quant.DAILY)
benchmark = quant.returns(db, "benchmark_returns", snapshot="returns-v1",
                          annualization=quant.DAILY)
print(strategy)
print(f"pinned: {strategy.provenance.pin.is_pinned}")
print(f"digest: {strategy.provenance.digest[:16]}")
print(f"\nconstants: DAILY={quant.DAILY} WEEKLY={quant.WEEKLY} "
      f"MONTHLY={quant.MONTHLY} YEARLY={quant.YEARLY}")

# %% [markdown]
# ## 3. The headline statistics
#
# One row of SQL produces the whole `perf_stats` table that `pyfolio` prints.
# Passing a benchmark joins the two series on their timestamps and adds alpha
# and beta, so only overlapping days contribute.

# %%
stats = pd.DataFrame(
    {
        "strategy": strategy.stats(),
        "with benchmark": strategy.stats(benchmark=benchmark),
        "benchmark": benchmark.stats(),
    }
)
stats.round(4)

# %% [markdown]
# Three of those deserve reading together rather than separately. `stability` is
# the R-squared of the cumulative log return against time, so a high Sharpe with
# low stability is a strategy that made its money in a few weeks. `tail_ratio`
# compares the 95th percentile of returns to the 5th, so below one means the
# losses have the fatter tail. `daily_value_at_risk` is a two-sigma daily loss,
# which is a statement about ordinary days and not about the bad ones.

# %%
alpha = strategy.stats(benchmark=benchmark)
print(f"annualized alpha over the universe  {alpha['alpha']:+.2%}")
print(f"beta to the universe                {alpha['beta']:.2f}")
print(f"stability                           {alpha['stability']:.2f}")
print(f"tail ratio                          {alpha['tail_ratio']:.2f}")

# %% [markdown]
# ## 4. Drawdowns as episodes
#
# A single maximum drawdown number hides the thing a risk committee asks about:
# how long it lasted. `drawdown_table` segments the underwater series into
# non-overlapping episodes the way `pyfolio` does, each with its peak, valley
# and recovery.

# %%
episodes = pd.DataFrame(strategy.drawdown_table(top=5))
episodes["net_drawdown"] = episodes["net_drawdown"].round(4)
episodes

# %% [markdown]
# An episode with no recovery date is one the series never climbed out of before
# the data ended, and its duration is unknown rather than zero.

# %%
underwater = strategy.underwater().to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(underwater["ts"], underwater["drawdown"], 0, alpha=0.6, color="#e45756")
ax.set_title("Drawdown")
ax.set_xlabel("Date")
ax.set_ylabel("Below the running peak")
fig.tight_layout()

# %% [markdown]
# ## 5. Rolling statistics
#
# Rolling windows answer the question a single number cannot: was this the same
# strategy throughout? `rolling_beta` uses the engine's `ts_cov` rather than a
# sliding covariance aggregate, because DataFusion cannot retract from the
# built-in one.

# %%
window = 126
rolling = (
    strategy.rolling_sharpe(window).to_pandas()
    .merge(strategy.rolling_volatility(window).to_pandas(), on="ts")
    .merge(strategy.rolling_beta(benchmark, window).to_pandas(), on="ts")
    .dropna()
)
print(f"{len(rolling):,} complete {window}-day windows")
rolling.tail(3).round(3)

# %%
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(rolling["ts"], rolling["rolling_sharpe"], linewidth=1.4)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"Rolling {window}-day Sharpe")
axes[0].set_ylabel("Sharpe")
axes[1].plot(rolling["ts"], rolling["rolling_beta"], linewidth=1.4, color="#f58518")
axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title(f"Rolling {window}-day beta to the universe")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Beta")
fig.tight_layout()

# %% [markdown]
# ## 6. The tearsheet
#
# `quant.tearsheet` renders the standard page: headline statistics, cumulative
# return, drawdown, rolling Sharpe, the drawdown table, and the provenance
# header. The charts are inline SVG with no external requests, because a report
# that needs a plotting library installed in order to be *read* is not a report.

# %%
html = quant.tearsheet(
    strategy,
    path="data/cache/momentum-tearsheet.html",
    benchmark=benchmark,
    title="12-1 momentum, top three",
    rolling_window=window,
    top_drawdowns=5,
)
print(f"wrote {len(html):,} bytes of self-contained HTML")
payload = quant.report_payload(strategy, benchmark=benchmark)
print(f"headline rows {len(payload['headline'])}, charts {len(payload['charts'])}")
print([chart["id"] for chart in payload["charts"]])

# %% [markdown]
# ## 7. From a backtest run to a tearsheet
#
# A run writes `bt_equity` into its fork, which is a *level* series rather than
# a return series. `quant.from_levels` is the bridge, and it drops the first bar
# rather than calling it a zero return, because a fake flat bar at the start of
# every curve is a lie that compounds.

# %%
frame = daily.to_pandas()
frame = frame[
    (frame["symbol"].isin(["AAPL", "MSFT", "JPM"]))
    & (frame["ts"] >= pd.Timestamp("2024-01-01", tz="UTC"))
]
market = cu.make_equity_market(pa.Table.from_pandas(frame, preserve_index=False))
run_db = h5i_db.Database(cu.fresh_db("06_tearsheet_run"), create=True)
for name in ("instruments", "book_deltas", "trades"):
    table = market[name]
    run_db.create_table(name, table.schema, time_column="ts_init")
    run_db.append(name, table)
run_db.snapshot("tape-v1", tables=["instruments", "book_deltas", "trades"])

book = market["book_deltas"].to_pandas()
book["session"] = book["ts_init"].dt.floor("s").dt.tz_localize("UTC")
first = book.groupby("instrument_id")["ts_init"].min()
signals = backtest.signal_table(
    [
        {
            "ts": stamp + dt.timedelta(microseconds=1),
            "instrument_id": symbol,
            "side": "buy",
            "quantity": 100.0,
            "tag": "buy-and-hold",
        }
        for symbol, stamp in first.items()
    ]
)
backtest.create_signal_table(run_db)
run_db.append("signals", signals)
report = backtest.run(
    run_db,
    "buy-and-hold",
    starting_cash=200_000.0,
    signals="signals",
    snapshot="tape-v1",
    fee_kind="proportional",
    fee_rate=0.0005,
    equity_interval_nanos=86_400_000_000_000,
)
fork = run_db.fork(report["fork"])
run_series = quant.from_levels(fork, "bt_equity", level="equity", annualization=quant.DAILY)
run_stats = run_series.stats()
print(f"{report['fills']} fills, {len(fork.read('bt_equity')):,} equity samples")
pd.Series(run_stats).to_frame("buy and hold").round(4)

# %% [markdown]
# ## 8. A number you can cite
#
# `quant.verify` re-runs the computation and checks both halves: the provenance
# digest and the recomputed values. An unpinned series is reported as
# *unverifiable* rather than passed, because two runs against "latest" agreeing
# proves only that nothing changed in the seconds between them.

# %%
verified = quant.verify(strategy, rerun=lambda: quant.returns(
    db, "strategy_returns", snapshot="returns-v1", annualization=quant.DAILY))
print(f"verified {verified['verified']}, pinned {verified['pinned']}")

unpinned = quant.returns(db, "strategy_returns", annualization=quant.DAILY)
try:
    quant.verify(unpinned)
except quant.VerificationError as error:
    print(f"\nunpinned refused: {str(error)[:110]}...")
relaxed = quant.verify(unpinned, strict=False)
print(f"non-strict report: verified={relaxed['verified']} reason={relaxed['reason']!r}")

# %% [markdown]
# The provenance header is what makes the refusal meaningful. It records the
# pin, the parameters and the SQL, so a tearsheet regenerated from the same pin
# reproduces the same numbers rather than merely similar ones.

# %%
provenance = strategy.provenance
print(f"kind        {provenance.kind}")
print(f"digest      {provenance.digest}")
print(f"pin         {provenance.pin}")
print(f"parameters  {provenance.parameters}")
print(f"warnings    {list(provenance.warnings()) or 'none'}")

# %% [markdown]
# ## Takeaways
#
# - A returns series is the input every performance statistic needs; store it as
#   a table so two reports cannot disagree about it.
# - `ReturnSeries.stats()` matches `empyrical`, and a benchmark adds alpha and
#   beta over the overlapping days only.
# - Drawdowns are episodes with durations, not a single worst number.
# - `from_levels` turns a run's `bt_equity` into the same object, so a backtest
#   and a research series are analysed by identical code.
# - `quant.tearsheet` renders a self-contained page with a provenance header.
# - `quant.verify` refuses to verify an unpinned computation. That refusal is
#   the feature.

# %%
fork.close()
run_db.close()
db.close()
