# Glossary

Every term the recipes use without stopping to define it. Each recipe also
carries a **Terms used here** cell listing the handful it needs, so you should
rarely have to come looking. This page is the fuller version, and the place to
find a term you met somewhere else.

Definitions are written for someone new to quant finance. If you already know
the field, the entries you want are the h5i-db ones at the bottom.

- [Reading paths](#reading-paths)
- [Market data and microstructure](#market-data-and-microstructure)
- [Returns, volatility and risk](#returns-volatility-and-risk)
- [Alpha research and backtesting](#alpha-research-and-backtesting)
- [Prediction markets](#prediction-markets)
- [Options, rates and other asset classes](#options-rates-and-other-asset-classes)
- [h5i-db and storage](#h5i-db-and-storage)

## Reading paths

**New to quant.** Read 00/01, 00/02 and 00/05 for the database, then
01/01 (bars), 01/04 (joining trades to quotes) and 02/01 (a first backtest).
Those six give you the vocabulary the rest of the cookbook assumes.

**New to h5i-db, not to quant.** Read 00/09 for the query builder and 00/05 for
versioning, then jump to whatever asset class you work in. The **Takeaways**
cell at the end of each recipe tells you whether the rest is worth your time.

## Market data and microstructure

Microstructure is the study of how prices actually form, at the level of
individual orders and trades, rather than the daily closes most models use.

- **Tick.** One event on a market data feed. Usually a trade or a quote update.
- **Tape.** The stream of trades, in time order. "The tape" is the trade
  history of an instrument.
- **Print.** A single trade on the tape. "A 10,000-share print" is one trade of
  that size.
- **Trade.** An executed transaction, with a price, a size and a timestamp.
- **Quote.** A dealer's or venue's current willingness to trade, published as a
  bid and an ask.
- **Bid.** The highest price a buyer is currently willing to pay.
- **Ask** (also **offer**). The lowest price a seller is currently willing to
  accept.
- **Mid.** The average of the bid and the ask. The usual stand-in for "the
  price" when you need one number.
- **Spread.** The ask minus the bid. It is the cost of an immediate round trip,
  and the market's charge for providing liquidity.
- **Quoted spread.** The spread as displayed, before any trade happens.
- **Effective spread.** Twice the distance from the trade price to the mid that
  prevailed at the time. It measures what a trade actually paid, which can beat
  the quoted spread when the trade executes inside it.
- **Realized spread.** The effective spread measured against a mid some minutes
  *after* the trade. It separates the liquidity provider's earnings from the
  loss they take when the price keeps moving against them.
- **Order book.** All resting buy and sell orders for an instrument, sorted by
  price. Also just "the book".
- **Top of book** (also **L1**). The best bid and best ask, with their sizes.
  The cheapest level of market data, and the most widely available.
- **L2** (also **depth**). Several price levels on each side, with the size
  resting at each. Needed for any claim about trading more than the top level.
- **Queue position.** Where your order sits in the line at its price level.
  Orders at one price fill in the order they arrived, so position decides
  whether you fill at all. Reconstructing it needs every order book event, not
  periodic snapshots.
- **Venue.** One exchange or trading system. US equities trade on more than a
  dozen simultaneously.
- **Fragmentation.** The same instrument trading on many venues at once, so no
  single venue's book is the whole market.
- **NBBO.** National Best Bid and Offer. The highest bid and lowest ask across
  all US venues, computed continuously. Execution quality is measured against
  it by regulation.
- **Locked market.** Bid equals ask across venues. **Crossed** means bid is
  above ask. Both are transient artifacts of fragmentation.
- **Bar** (also **candle**). Ticks aggregated into a fixed time interval.
- **OHLCV.** The five fields of a bar: open, high, low, close and volume.
- **Session.** One trading day as the exchange defines it, for example 09:30 to
  16:00 New York for US equities. Not a UTC calendar day.
- **DST.** Daylight saving time. Exchange sessions are fixed in local wall-clock
  time, so their UTC offset shifts twice a year.
- **VWAP.** Volume-weighted average price, the total value traded divided by the
  total volume. The standard benchmark for an execution, because it is the
  average price the whole market got.
- **TWAP.** Time-weighted average price, ignoring volume. Diverges from VWAP
  whenever volume is concentrated at particular times, which it always is.
- **Arrival price.** The price at the instant the order was handed to the desk.
  Benchmarking against it charges the trader for the delay, not just the fills.
- **Slippage.** The gap between the price you expected and the price you got.
- **Implementation shortfall.** Total cost of turning a decision into a
  position, including slippage, fees and the part of the order never filled.
- **Basis point** (**bp**). One hundredth of a percent, 0.01%. Execution costs
  and spreads are quoted in bp because they are small.
- **Market impact.** The price moving against you *because* you traded. Large
  orders pay it, and it is why backtests that assume unlimited size lie.
- **Latency.** The delay between a decision and its arrival at the venue. Long
  enough that the book you decided against may be gone.
- **Maker / taker.** A maker posts a resting order and waits. A taker crosses
  the spread and trades immediately. Venues usually charge them differently.
- **Fill.** An execution against your order. A **partial fill** is less than the
  full quantity. **Fill ratio** is the fraction you got.
- **Signed volume.** Trade volume with a sign: positive when the buyer was the
  aggressor, negative when the seller was.
- **Buyer-initiated / seller-initiated.** Which side crossed the spread to make
  the trade happen. Feeds rarely say, so it is inferred.
- **Lee-Ready.** The standard rule for inferring that side. Compare the trade
  price to the prevailing mid: above means buyer-initiated, below means
  seller-initiated, and ties fall back to the tick test.
- **Tick test.** Sign a trade by comparing it to the previous trade price. Up
  means a buy, down means a sell.
- **Order flow imbalance (OFI).** Buyer-initiated volume minus seller-initiated
  volume, over a window. The workhorse short-horizon microstructure signal.
- **Microprice.** The mid weighted by the size on each side, so a book with far
  more bid than ask size sits above the mid. A better one-number estimate of
  where the price is going than the plain mid.
- **Imbalance.** Bid size relative to ask size at the top of book.
- **Bid-ask bounce.** Consecutive trades alternating between bid and ask, which
  looks like volatility but is not. It inflates any variance estimate sampled
  fast enough to see it.
- **Microstructure noise.** The general term for that effect. The observed price
  is the true price plus a trading artifact, and the artifact dominates at short
  sampling intervals.
- **Stale quote.** A quote old enough that it no longer describes the market. A
  **tolerance** on a join refuses to match beyond a chosen age.
- **Corporate action.** A change to the security itself rather than to its
  price: a split, a dividend, a merger.
- **Split.** A share count multiplication, for example 4:1. The price divides by
  the same factor, so every historical price must be rescaled to keep returns
  correct.
- **Adjusted close.** A close price rescaled for all later splits and dividends.
  Comparable across time, but it changes whenever a new action lands.
- **Adjustment factor.** The multiplier that converts raw prices to adjusted
  ones. Storing it separately keeps the raw tape intact.

## Returns, volatility and risk

- **Return.** The fractional price change over a period. A **simple return** is
  `p_t / p_{t-1} - 1`. A **log return** is `ln(p_t / p_{t-1})`, which adds
  across time and is preferred for multi-period arithmetic.
- **Volatility** (**vol**). The standard deviation of returns. The default proxy
  for how risky a position is.
- **Annualized.** A per-period figure scaled to a one-year horizon. Multiply
  daily returns by 252 and daily volatility by `sqrt(252)`, the count of trading
  days in a year.
- **Realized volatility (RV).** Volatility measured from observed returns,
  usually the square root of the sum of squared intraday returns. No model
  needed, which is why it is called nonparametric.
- **Signature plot.** Realized variance plotted against the sampling interval
  used to compute it. It slopes up at fast sampling because bid-ask bounce
  inflates the estimate, which is how you choose a sampling frequency.
- **Bipower variation.** A variance estimate built from products of *adjacent*
  absolute returns. One large jump contaminates only two terms, so subtracting
  it from realized variance isolates the jump component.
- **Jump.** A price move too large to be part of the continuous diffusion, for
  example a gap on news.
- **EWMA.** Exponentially weighted moving average. Each new observation gets
  weight `alpha` and the running estimate keeps `1 - alpha`, so recent data
  counts more and old data fades smoothly.
- **RiskMetrics lambda.** The same recursion written as
  `sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_t^2`, with
  `lambda = 0.94` for daily data. Note that h5i-db's `alpha` is `1 - lambda`,
  so 0.94 becomes `ewma(x, 0.06)`.
- **Center of mass / half-life.** Two ways to express how long an EWMA
  remembers. Daily `lambda = 0.94` has a center of mass near 16 trading days.
- **Vol targeting.** Scaling position size so realized risk stays near a fixed
  level, cutting exposure when volatility rises. A risk-shaping tool first and a
  return-improving one only sometimes.
- **Leverage.** Exposure as a multiple of capital. 2x means two dollars of
  position per dollar of equity.
- **Gross exposure.** The total size of all positions, long plus short,
  ignoring sign.
- **Equity curve.** Cumulative value of a strategy over time, usually plotted as
  the growth of one dollar.
- **NAV.** Net asset value. The book's total worth at a point in time.
- **Drawdown.** The percentage fall from the equity curve's running peak.
  **Max drawdown** is the worst one in the sample, and it is what actually gets
  a strategy shut down.
- **Sharpe ratio.** Annualized return divided by annualized volatility. Reward
  per unit of risk, and the field's default comparison number.
- **P&L.** Profit and loss. **Realized** P&L is booked from closed trades,
  **unrealized** is the paper gain on open ones.
- **Mark to market.** Valuing an open position at the current market price
  rather than what you paid.
- **VaR.** Value at Risk. The loss a book will not exceed on a given day with a
  chosen confidence, for example "1-day 99% VaR of $2.4M". It says nothing about
  how bad the worse days are.
- **Expected Shortfall (ES**, also **CVaR).** The average loss on the days that
  do breach VaR. It answers the question VaR ducks.
- **Historical VaR.** VaR read straight off the empirical distribution of past
  returns. **Parametric VaR** assumes a distribution, usually normal, and reads
  the quantile from it.
- **Exception** (also **breach**). A day whose loss exceeded the VaR forecast.
  A 99% VaR should produce them about 1% of the time.
- **Kupiec POF test.** A statistical test of whether the observed count of
  exceptions is consistent with the confidence level claimed.
- **Beta.** How much an asset moves for a one-unit move in the market. The
  slope from regressing asset returns on market returns.
- **Market model.** The regression `r_asset = alpha + beta * r_market + e`, used
  to strip out market-wide moves.
- **Abnormal return.** What is left of a return after removing the market model
  prediction. The part attributable to the event you are studying.
- **CAR.** Cumulative abnormal return, summed over an event window.

## Alpha research and backtesting

- **Alpha.** Return not explained by known risk exposures. Loosely, the part of
  performance that is skill.
- **Signal.** A number computed from data that indicates a position to take.
- **Factor.** A characteristic shared across many assets that explains returns,
  for example value, momentum or quality. A **factor library** is the stored,
  versioned panel of them.
- **Cross-sectional.** Comparing assets against each other at one point in time.
  **Time-series** means comparing one asset against its own history.
- **Momentum (12-1).** Buy what has gone up. The standard version uses the
  return over the last twelve months excluding the most recent one, because that
  last month tends to reverse.
- **Value (B/P).** Book value divided by price. High means cheap relative to
  accounting worth.
- **Quality.** A family of profitability and stability measures. This cookbook
  uses revenue-growth stability as a proxy.
- **Panel.** A dataset with two dimensions, typically one row per asset per
  date.
- **Rebalance.** Recomputing target weights and trading to them, on a schedule.
- **Turnover.** How much of the book is traded at each rebalance. It is the
  direct driver of transaction costs.
- **Information coefficient (IC).** The correlation between a signal and the
  return that followed. Monthly ICs around 0.03 to 0.05 are a real equity
  factor, so calibrate your expectations low.
- **Quintile spread.** Sort assets by signal into five buckets, then take the
  return of the top bucket minus the bottom. A model-free readout of whether a
  signal separates winners from losers.
- **Cointegration.** Two prices that individually wander but whose difference
  does not. It is the statistical basis for pairs trading.
- **Engle-Granger test.** The standard two-step test for it: regress one series
  on the other, then test whether the residual is stationary.
- **Hedge ratio.** How many units of the second leg to hold against one unit of
  the first, so the pair position is insensitive to the common move.
- **Spread (pairs trading).** The residual of that hedged combination. A
  different meaning from the bid-ask spread above.
- **Z-score.** How many standard deviations a value sits from its rolling mean.
  Converts a spread into a comparable trade trigger.
- **Mean reversion.** The tendency to return to an average. Its **half-life** is
  how long half the deviation takes to decay.
- **Backtest.** Simulating a strategy on historical data.
- **Event-driven backtest.** A backtest where orders meet the recorded market
  data event by event, rather than assuming a fill at a bar price.
- **Lookahead bias.** Using information the strategy could not have had at the
  time. The most common and most flattering backtest bug.
- **Point-in-time.** Data stored and queried as it was known at each moment,
  which is what makes lookahead bias avoidable.
- **Reporting lag.** The gap between the period a fundamental figure describes
  and the date it was published, typically 25 to 55 days.
- **Survivorship bias.** Testing only on assets that still exist today, which
  quietly deletes every failure from the sample.
- **Restatement.** A vendor correcting historical data after the fact. It
  changes what a rerun of the same backtest sees.
- **In-sample / out-of-sample.** Data used to develop the strategy, versus data
  held back to test it.
- **Holdout.** Out-of-sample data deliberately untouched until the end. It is
  only worth what its scarcity makes it, so it can be spent once.
- **Walk-forward.** Repeatedly fit on a window and test on the window after it,
  rolling forward. Closer to how a strategy is actually run.
- **Overfitting.** Finding a rule that describes the noise in your sample rather
  than any repeatable structure.
- **Multiple testing.** Trying many variants, so the best one looks good partly
  by luck. The more you try, the higher the bar the winner must clear.
- **PBO.** Probability of backtest overfitting. Estimated by repeatedly
  splitting the trial set and asking how often the in-sample winner is below
  median out of sample. Above 0.5 means the selection procedure is worse than
  random.
- **Deflated Sharpe ratio.** A Sharpe adjusted downward for how many variants
  were tried, plus the skew and kurtosis of the returns. It answers whether the
  headline number survives the size of the search.
- **Minimum track record length.** How long a track record must be before a
  claimed Sharpe is statistically distinguishable from zero.
- **Embargo.** A gap between train and test data, so information does not leak
  across the boundary through overlapping windows.
- **Event study.** Measuring abnormal returns in a window around a dated event.
  The **estimation window** fits the market model, the **event window** is where
  the effect is measured.
- **Lead-lag.** One instrument's moves systematically preceding another's.
- **Cross-correlation function (CCF).** Correlation between two series at a
  range of time offsets. The peak offset is the estimated lead.
- **Epps effect.** Measured correlation between two assets falling toward zero
  as the sampling interval shrinks. An artifact of asynchronous trading, not a
  real decoupling.
- **Hayashi-Yoshida.** A covariance estimator that pairs overlapping tick
  intervals directly, so it needs no time grid and avoids the Epps effect.
- **Seasonality.** Repeating patterns by time of day, week or year. Intraday
  volume follows a U-shape, heavy at the open and close.
- **Transaction costs.** Everything that makes the traded price worse than the
  decision price: commissions, fees, spread and impact.
- **Commission.** The broker's or venue's explicit per-trade charge.

## Prediction markets

Also called event contracts. A contract that pays 1.00 if a stated event happens
and 0.00 if it does not, so its price reads directly as a probability.

- **Binary contract.** The two-outcome case. **YES** pays on the event
  happening, **NO** pays on it not happening.
- **Resolution.** The determination of which outcome occurred.
- **Settlement.** Paying out on that determination. Positions held to settlement
  are worth exactly 1.00 or 0.00.
- **Expiration.** When trading stops. Note that resolution can come later, and
  the gap matters.
- **Observability.** When the outcome actually became knowable. A backtest may
  only use a resolution after this instant, never at the moment the event
  occurred.
- **Settlement lag.** The delay between expiration and the result becoming
  observable.
- **Parity.** YES and NO must sum to 1.00, because exactly one of them pays. Any
  gap is an arbitrage, subject to fees.
- **Basis.** The size of that gap. "A two-cent basis" means the pair costs 0.98.
- **Fee curve.** Prediction-market fees typically scale with `p * (1 - p)`, so
  they are largest at 0.50 and near zero at the extremes. As a fraction of the
  capital committed, the fee is `rate * (1 - p)`, which falls as the contract
  gets more expensive.
- **Favorite-longshot bias.** The oldest documented anomaly in event markets.
  Cheap contracts (**longshots**) are systematically overpriced and expensive
  ones (**favorites**) underpriced.
- **Calibration.** Whether quoted probabilities match observed frequencies.
  Perfectly calibrated means events priced at 0.30 happen 30% of the time.
- **Reliability curve.** Observed frequency plotted against quoted probability.
  The diagonal is perfect calibration.
- **Brier score.** Mean squared error of a probability forecast, so lower is
  better. It decomposes into **reliability** (calibration error), **resolution**
  (how far forecasts move from the base rate, higher is better) and
  **uncertainty** (a property of the events, not the forecaster).
- **Log loss.** An alternative score that punishes confident errors far more
  harshly than the Brier score does.
- **Base rate.** How often the event happens overall. The benchmark any forecast
  has to beat.
- **UMA.** The decentralized oracle protocol Polymarket uses to determine
  outcomes.

## Options, rates and other asset classes

- **Implied volatility (IV).** The volatility that makes a pricing model return
  the option's observed market price. The market's forecast, quoted as a number
  traders can compare.
- **IV surface.** Implied volatility across strikes and expiries, as one object.
- **Smile / skew.** The shape of IV across strikes at one expiry. Equity index
  options are skewed, with downside strikes priced at higher volatility.
- **Term structure.** The shape across expiries at one strike, typically ATM.
- **ATM.** At the money. A strike near the current price.
- **Delta.** How much the option price moves for a one-unit move in the
  underlying. Also used as a moneyness label, so "25-delta" identifies a strike.
- **Risk reversal.** The IV difference between a call and a put at equal delta.
  A one-number readout of skew.
- **Butterfly.** The IV of the wings against the ATM. A one-number readout of
  smile curvature.
- **Expiry.** The date an option contract expires. **Tenor** is the time
  remaining, which is why it should be derived rather than stored.
- **Chain.** All listed options on one underlying at one moment.
- **Yield curve.** Interest rate plotted against maturity. A **par curve**
  quotes it as the coupon that prices a bond at 100.
- **Tenor (rates).** The maturity point on the curve, for example 2y or 10y.
- **Slope.** A long tenor's yield minus a short one's, for example 10y minus 2y.
- **Curvature** (also **butterfly**). The middle of the curve against its ends.
- **Carry.** What a position earns just from time passing, if the curve does not
  move.
- **Rolldown.** The gain from a bond aging into a lower point on an upward
  sloping curve.
- **Mark.** An official price for a position, set once per day or more often.

## h5i-db and storage

- **Embedded database.** A database that runs inside your process, with no
  server to start. A database is a directory on disk, like SQLite or DuckDB.
- **Arrow.** The in-memory columnar format h5i-db speaks. pandas, polars and
  pyarrow all convert to it cheaply.
- **Parquet.** The on-disk columnar file format. Columnar means a query that
  touches three columns reads only those three.
- **Schema.** The table's column names and types, fixed at creation.
- **Time column.** The column h5i-db sorts and prunes on. Every table declares
  one.
- **Sort key.** The physical row order inside each segment. It must start with
  the time column, and it decides how well joins and rollups stream.
- **Commit.** One atomic write. Every `append`, `write`, `delete` and `restore`
  produces one.
- **Version.** The state of a table after a given commit. Versions are immutable
  and stay readable forever.
- **Manifest.** The small file listing which segments make up a version.
  Reading an old version is a manifest lookup, not a log replay.
- **Segment.** One immutable Parquet file holding a time range of rows.
- **Time travel.** Reading a table as it was at an earlier version, wall-clock
  time (`as_of`), or named snapshot. Written `h5i('trades', 7)` in SQL.
- **Snapshot.** A named, checksummed pin of one or more table versions. Costs
  O(1) because it references manifests rather than copying data.
- **Pin.** Fixing a run's inputs to an exact version or snapshot, so a rerun
  reads the same bytes.
- **Append vs write.** `append` adds rows to the end. `write` replaces the whole
  contents, as a new version, with the old one still readable.
- **Optimistic concurrency.** Writers do not lock. A commit carries the version
  it expected to extend, and is rejected if the head has moved.
- **ConflictError.** That rejection. It is a retry, not a lost update.
- **Plan / apply.** Deletes and replacements are staged as a plan carrying row
  counts and before/after samples, reviewed, then applied as a commit. This is
  the only delete path in the Python API.
- **Mutation policy.** A database-level setting that blocks direct destructive
  writes, so the reviewed plan/apply flow is the only way through.
- **compact.** Merge many small segments into fewer large ones, as a new
  version. Streaming ingestion is what makes it necessary.
- **vacuum.** Reclaim unreferenced storage. It never removes committed version
  history.
- **verify.** Walk the checksum chain. `verify(deep=True)` re-checksums every
  stored byte.
- **Pruning.** Skipping whole segments whose time range cannot match the query
  predicate, before any I/O.
- **Projection.** Reading only the columns the query needs.
- **DataFusion.** The Apache SQL engine h5i-db queries through. It provides full
  SQL, and h5i-db adds the time-series operators.
- **DataFrame builder.** `db.table(...)` plus verbs, a lazy query you hold as a
  Python value. It compiles to SQL, and `.sql()` prints what it produced.
- **Lazy.** Nothing executes until a terminal call such as `.collect()`.
- **time_bucket.** Truncate timestamps to a grid, optionally in a named
  timezone, so bucket edges follow local wall-clock time through DST.
- **Window function.** SQL that computes across a set of rows related to the
  current one, for example `lag()` or a rolling mean, without collapsing them.
- **CTE.** A named subquery introduced by `WITH`, so a multi-step query stays
  readable.
- **ASOF join.** Join each left row to the most recent right row at or before
  its timestamp. The operation behind attaching quotes to trades, fundamentals
  to prices, and marks to positions.
- **gapfill / resample.** Turn an irregular series onto a regular grid.
  `'null'` leaves holes visible, `'locf'` carries the last observation forward,
  and `'interpolate'` draws a line between neighbours.
- **locf.** Last observation carried forward. Convenient and dangerous, because
  a repeated stale price prints a zero return that was never tradeable.
- **tail.** Read exactly the rows committed after a version you name. A
  versioned table doubles as a message log, with no timestamp-cursor guesswork.
- **High-water mark.** The last version a consumer processed, stored so it can
  resume.
- **arrival_delta.** Run one query at the current head and at a past decision
  point, and report the difference. The part that moves is the part that
  depended on data which had not arrived yet.
- **Run fork.** An isolated branch of the database that a backtest writes its
  orders, fills, positions and equity into, so runs never collide.
- **Preflight.** A check that rejects an unsupported or unsafe request before it
  reaches the engine, for example a queue-position claim on snapshot-only data.
