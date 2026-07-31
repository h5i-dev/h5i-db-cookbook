# Writing cookbook recipes

Style guide + h5i-db API cheatsheet for recipe authors. Every recipe is a
jupytext `py:percent` file under `notebooks/<section>/`; `scripts/build_notebooks.py`
executes it and writes the paired `.ipynb`.

## Recipe format

```python
# %% [markdown]
# # <Recipe title>
#
# One-paragraph pitch: the professional problem this solves and why h5i-db's
# features (versioning / ASOF / time_bucket / plan-apply / ...) matter for it.

# %% [markdown]
# ## Terms used here
#
# | term | meaning |
# | --- | --- |
# | <term> | <one line, no jargon inside the definition> |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import pyarrow as pa
import h5i_db
from h5i_db import col, time_bucket, vwap
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("<section>_<recipe>"), create=True)
```

Rules:
- **Two audiences, one prose register.** The body is written for professional
  quants: don't explain what VWAP *is* beyond one line, show how to compute it
  well. Beginners are served by the **Terms used here** cell instead, so the
  prose never slows down for them. Explain h5i-db concepts on first use in that
  recipe.
- **Every recipe carries a "Terms used here" cell**, second cell, right after
  the pitch. 5-10 rows, only terms that recipe actually leans on, ordered as the
  recipe meets them. One line each, written for someone new to quant finance and
  containing no jargon of its own. Backticks for API and SQL identifiers, plain
  text for concepts. Anything defined there must also be in `GLOSSARY.md`, and
  the link line closing the cell is verbatim across all recipes.
- **The intro states the motivation**, not just the mechanics: what goes wrong
  without this, or what question a desk is actually asking. A recipe whose
  opening only describes its own steps is under-written.
- **Show the data before you query it.** Every primary input table gets a
  markdown paragraph (what the generator returns, what one row is), a
  `column | type | meaning` table, and a code cell that loads, reports size and
  shows the head:

  ```python
  trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=30_000, seed=7)
  print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
  trades.to_pandas().head()
  ```

  Plain pyarrow/pandas, no helper - the reader should be able to copy it. Types
  in the table match the `pa.field(...)` constructors (`timestamp[us, tz=UTC]`,
  `string`, `float64`, `int64`). Put the preview *before* `create_table`, so the
  schema reads as a response to data the reader has seen. Recipes that build
  their input locally preview the constructed table. With several inputs, the
  second may be prose + head if its shape is obvious.
- **Prose register: documentation, not essay.** One idea per sentence, ~25 words
  max. No clause stuffed between dashes - if it is worth setting off, give it
  its own sentence or a colon. Lead with what the code does, then why it is
  written that way. The Polars user guide is the reference register.
- **Queries are builder-first**: `db.table(...)` + verbs, not a SQL string,
  wherever the builder expresses the query cleanly. See the DataFrame builder
  cheatsheet for what deliberately stays in `db.sql()`.
- Markdown cell before every code cell: *why*, not *what*. Code comments only
  where genuinely non-obvious.
- Each recipe is standalone and idempotent: `cu.fresh_db("name")` (wipes prior
  runs), unique db name = `<section-number>_<recipe-stem>`.
- Deterministic: fixed seeds; real data via `cu.fetch_daily` / `cu.fetch_intraday`
  (cached to parquet). Never un-cached network calls in hot paths.
- Keep runtime under ~60s per recipe. Synthetic data sized accordingly
  (e.g. `make_trades(days=3, trades_per_day=20_000)`).
- Plots: matplotlib, no seaborn, one clear figure where a picture earns its
  place; always `ax.set_title/xlabel/ylabel`; `fig.tight_layout()`.
- End with a short "Takeaways" markdown cell: 3-5 bullets, incl. which h5i-db
  features did the heavy lifting.
- Close the database at the end: `db.close()`.
- **Mirror it into `notebooks_ja/`.** Markdown cells are translated, code cells
  stay byte-identical. The terms cell becomes `## ここで使う用語` with
  `| 用語 | 意味 |` and links to `../../GLOSSARY.ja.md`. Verify with
  `diff <(grep -v '^#' notebooks/S/R.py) <(grep -v '^#' notebooks_ja/S/R.py)`.
  Japanese register is ですます調, tech genre; lint the extracted markdown with
  the `natural-japanese` skill's `lint.py --genre tech`, never the `.py`. The
  recurring finding is `antithesis_repetition`: 「〜ではなく」 trips at three per
  document, so keep at most two and rephrase the rest.

## cookbook_utils cheatsheet

```python
import cookbook_utils as cu
cu.fresh_db(name) -> str            # wiped db dir path; cu.db_path(name) keeps
trades = cu.make_trades(symbols=["AAPL","MSFT","NVDA"], days=5, trades_per_day=20_000,
                        start="2026-06-01", seed=7)   # ts,symbol,price,size,exchange,side
quotes = cu.make_quotes(...)                          # ts,symbol,bid,ask,bid_size,ask_size
t, q  = cu.make_trades_and_quotes(days=3)             # shared base px; walks are
                                                      # independent - derive trades
                                                      # from quotes for micro-consistent tapes
daily = cu.make_daily_prices(symbols=[...], days=750) # ts,symbol,open,high,low,close,volume
fx    = cu.make_fx_ticks(pairs=["EURUSD","USDJPY"], hours=72)     # ts,pair,bid,ask
chain = cu.make_option_chain(snapshots=5)             # ts,underlier,expiry,strike,cp,iv,mid,delta
curves= cu.make_yield_curves(days=250)                # ts,tenor_years,yield_pct
funda = cu.make_fundamentals(quarters=12)             # ts(=report time),period_end,symbol,eps,revenue_m,book_value_m
real  = cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")
                                                      # ts,symbol,open,high,low,close,adj_close,volume
bars  = cu.fetch_intraday(["SPY","QQQ"], period="30d", interval="1h")
```
All tables: `ts` is `timestamp[us, tz=UTC]`, sorted ascending.

Backtest fixtures are separate: their time column is `ts_init` and it is
`timestamp[ns]`, tz-naive, because that is the canonical backtest schema. Every
raw-unit argument against them (`read(time_end=...)`, plan ranges) is therefore
**nanoseconds**, not microseconds.

```python
fix = cu.make_backtest_fixture(steps=180)             # one market: instruments, book_deltas, trades
panel = cu.make_prediction_markets(n_markets=240, steps=48, seed=11)
#   -> instruments, book_deltas, trades, resolutions for a panel of binary markets.
#   Quotes carry an injected favorite-longshot bias (longshot_bias=), the NO book is
#   the complement plus an oscillating basis (basis_amplitude=), and winners are
#   assigned by systematic sampling so calibration is measurable at panel size.
#   Trading runs to expiry; the result becomes observable settlement_lag_minutes
#   later; tail_steps further snapshots quote the resolved book at ~1.00/~0.00 so a
#   full-window replay reaches settlement and a truncated one does not.
truth = cu.market_truth(panel)                        # instrument_id, yes_won - the answer key

# Vendor-shaped fixtures, for recipes that exercise the ingest path (05/06, 05/07).
payloads = cu.polymarket_market_payloads(panel)        # public market-endpoint JSON,
#   including the awkward parts: list fields as JSON-encoded strings, resolution as
#   settled outcomePrices plus a closed flag, ISO-8601 times.
files = cu.write_polymarket_archive(panel, "data/cache/mirror")   # hourly Parquet in the
#   full-feed archive shape: event_type, timestamp (ms), market, asset_id, nested
#   bids/asks, flat price/size/side. The inverse of h5i_db.venues, for teaching only.
```

Equities and other continuous instruments use two more builders. Both emit the
same canonical tables, with `kind="spot"` instruments and prices that are not
probabilities.

```python
market = cu.make_equity_market(bars, spread_bps=5.0, depth_fraction=0.002,
                               stagger_us=1)
#   -> instruments, bars, book_deltas, trades. `bars` is any (ts, symbol, open,
#   high, low, close, volume) table: cu.fetch_daily, cu.fetch_intraday and
#   cu.make_daily_prices all qualify. Bar data records no book, so one is
#   assumed: a two-sided quote `spread_bps` wide around each bar's close with
#   `depth_fraction` of the bar's volume a side. That assumption is the subject
#   of 04/08, not a detail to hide.
tape = cu.make_equity_tape(quotes, action="set", levels=8, level_growth=1.6,
                           print_every=2, print_size=400.0)
#   -> instruments, book_deltas, trades from a cu.make_quotes stream. Prints are
#   derived FROM the quotes, so a trade always happens at a price the book was
#   showing, which is what queue-aware fills need. `action="set"` emits
#   delete/set deltas instead of snapshots: a Python callback is told the price
#   of a delta and only the level *count* of a snapshot, so a quoting strategy
#   needs deltas (04/09). `levels` builds a ladder for measuring what a large
#   order costs (04/11) and requires `action="snapshot"`.
```

**Stagger is load-bearing, not cosmetic.** `make_equity_market` gives each
symbol its own microsecond within a bar. An intent released into a batch of
same-timestamp events meets the *new* book for whichever instrument the merge
reached first and the previous one for all the others, so without the stagger
the answer depends on the shape of the panel. Recipe 04/13 staggers the
prediction-market panel by hand for the same reason.

**The stamp chooses the price.** An intent released at an instant meets the last
book the venue has already processed, and the fill is *recorded* at the next
event. Stamping one microsecond after a bar's close therefore trades at that
close. To trade at the next session's close, stamp one microsecond after that
session's own quote instant (04/08 does this; the book table carries the exact
instants).

One invariant of that fixture is load-bearing: rows sharing an `event_index`
must describe ONE outcome of ONE instrument. One event is one book, so a
snapshot spanning both outcomes describes a book that never existed, holding
both sides' levels with a best ask belonging to the other outcome.

h5i-db refuses this as of the fix in `store.rs::read_book_events`
("one event describes one outcome of one instrument"). Older builds accepted it
silently and filled against the wrong side: a YES buy paid the NO ask, with no
error anywhere. If you are on a Python extension built before that fix, the
failure is silent, so check the rule rather than relying on the engine to.

`event_index` values need only *change* between events; they do not have to
increase with `ts_init`. Grouping is by row contiguity terminated by `is_last`,
and an unterminated event is caught explicitly. A generator that emits
instrument-major (so the index walks backwards through time) still produces
correct books.

**Signal timing.** Stamp a signal strictly after the quote it was decided from
(`decision_ts + timedelta(microseconds=1)`). A signal sharing a timestamp with a
book event may match against the *previous* snapshot, and which one it gets
depends on merge order among equal timestamps. Submitting after the decision
quote fills at exactly the bid/ask you decided from, which is both deterministic
and free of look-ahead. Recipes 05/01-05/05 all rely on this.

**Post-resolution rows.** `make_prediction_markets` keeps quoting after the result
is observable, so any scan measuring price behaviour must stop at
`expiration_ns`; otherwise the resolution jump is scored as a price move and
dominates every volatility estimate.

Real-data calls are cached by exact argument set. These four are pre-cached -
recipes should use one of them verbatim (then subset in SQL/pandas), not
invent new argument combinations:

```python
cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")       # 30 names
cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")  # 10 names
cu.fetch_intraday(["SPY", "QQQ"], period="60d", interval="1h")
cu.fetch_intraday(["SPY"], period="30d", interval="30m")
```

## h5i-db API cheatsheet

**No HDF5 anywhere** - storage is immutable Parquet segments + versioned
manifests. Never describe it as HDF5-based.

```python
import h5i_db
db = h5i_db.Database(path, create=True)          # context manager works too

db.create_table("trades", schema, time_column="ts", sort_key=["ts", "symbol"])
# NOTE: when sort_key is given, it MUST start with the time column.
db.append("trades", arrow_table, note="day 1 load")   # strict: schema match,
                                   #    time-sorted, min ts >= table max ts
# append/write return the commit dict:
#   {"table","sequence","op","rows_total","segments_total","segments_added",
#    "segments_deduped","committed_at_ns"}
db.append("trades", data, expected_version=3)  # optimistic lock -> ConflictError if head moved
# (plain append already auto-retries pure-append conflicts; on ConflictError
#  with expected_version, re-read the head and retry against it)
db.write("trades", arrow_table)    # replace contents -> new version (history kept)
db.tables(); db.schema("trades")
db.versions("trades")              # list of dicts: sequence (=version number), op
                                   # ('create','append','write','delete_range','replace_range',
                                   #  'restore','compact'), committed_at_ns, rows, bytes,
                                   # segments, and note (only when one was given)
# committed_at_ns -> as_of string:
#   pd.Timestamp(v["committed_at_ns"], unit="ns", tz="UTC").isoformat()

res = db.sql("SELECT ...", timeout=60, max_rows=1_000_000)  # QueryResult
res.to_pandas(); res.to_arrow(); res.to_polars(); len(res)

db.read("trades", version=3)                       # O(1) time travel (pa.Table)
db.read("trades", as_of="2026-07-01T00:00:00Z")    # by commit wall-clock time
db.read("trades", columns=["ts","price"], time_start=us, time_end=us, limit=100)
db.restore("trades", version=3)                    # rollback = new head, history kept

# Previewable mutations - start/end are raw int64 in the time column's unit (us here)
plan = db.plan_delete_range("trades", start_us, end_us, note="bad prints")
plan = db.plan_replace_range("trades", start_us, end_us, data=fixed_table, note="...")
plan.summary                       # {"rows_before","rows_after","rows_affected",
                                   #  "segments_reused","segments_added","added_bytes",
                                   #  "affected_time_range"}
plan.before_sample; plan.after_sample   # pa.Table previews
plan.apply()                       # ConflictError if head moved since planning
plan.discard(); db.list_plans("trades")
# Ranges are half-open [start, end) in raw time-column units. A replace plan
# replaces the WHOLE window across all rows in it - replacement data must carry
# the innocent rows through. Plans expire after 7 days (raw["expires_at_ns"]).
# plan_* are the ONLY delete/replace paths in Python (no direct methods exist);
# a replace plan with an empty data table silently becomes op 'delete_range'.
# Raw-us conversion footgun: pd.Timestamp.value is NANOSECONDS (divide by
# 1_000), but a pandas datetime64[us, UTC] Series .astype("int64") is already
# MICROSECONDS.
# Policy flags gate only the direct paths; plan.apply() is the sanctioned flow
# and succeeds even when the corresponding direct_* flag is False.

db.policy()                        # {"direct_append","direct_write","direct_replace",
                                   #  "direct_delete","direct_restore","direct_compact"} (bools)
db.set_policy(direct_delete=False) # gated ops then require the plan/apply flow
# Look-ahead diagnostics: run one query at head AND at a decision read point,
# and report the delta - the share of a result that was not knowable then.
rep = db.arrival_delta(sql, version=1)          # or as_of="..." / snapshot="..."
rep["changed"]; rep["max_abs_delta"]; rep["withheld_versions"]
rep["columns"][0]                               # {"name","head","asof","delta","delta_pct",...}
#   (head/asof/delta/delta_pct are present only when both results are 1 row)
rep["vacuous"]      # True => both read points resolved to the SAME version, so
                    # the delta is arithmetically zero and means nothing. The
                    # normal state of a single-bulk-ingest database. CHECK THIS
                    # BEFORE READING THE NUMBER.
rep["notes"]        # caveats to surface alongside the numbers
# Scope: this measures the ARRIVAL axis (late/restated rows across commits). It
# is blind to event-time look-ahead inside one snapshot (a same-bar signal, a
# window overrunning forward) - those show a near-zero delta. The structural
# fix for that axis is CLI-only today: `h5i-db query ... --decision-time <ts>
# [--embargo <dur>]` bounds every scan in the session.

db.snapshot("eod-2026-07-21", tables=["trades"], note="EOD risk cut")
# (create-only: there is no list-snapshots call in the Python API)
db.compact("trades")               # merge small segments (do this after many small appends)
db.vacuum(apply=False)             # dry-run space reclaim; verify(deep=True) checks checksums
# vacuum NEVER prunes committed version history - every version stays readable.
# It only reclaims truly unreferenced objects (e.g. discarded-plan staging
# segments), and only ones older than grace_seconds.
db.close()
```

Exceptions: `H5iError` base with `.code`, `.retryable`, `.hint`; subclasses
`ConflictError, NotFoundError, InvalidInputError, PolicyError, CorruptionError,
LimitError, TimeoutError, StorageError`.

## Event-driven backtest cheatsheet

```python
from h5i_db import backtest

signals = backtest.signal_table([
    {"ts": ts, "instrument_id": "market", "side": "buy", "quantity": 10.0},
    {"ts": ts2, "instrument_id": "market", "side": "sell", "quantity": 10.0,
     "kind": "limit", "limit_price": 0.55, "time_in_force": "gtc",
     "tag": "exit", "reduce_only": True},
])
backtest.create_signal_table(db, "signals")
db.append("signals", signals)

report = backtest.run(
    db, "run-001", starting_cash=10_000, signals="signals",
    snapshot="approved-input",
    fee_kind="proportional", fee_rate=0.001, maker_rebate=-0.0001,
    latency_nanos=2_000_000, slippage_ticks=2,
    equity_interval_nanos=1_000_000_000,
    window=(start, end), minimum_coverage=0.95,
)
run_db = db.fork(report["fork"])
run_db.read("bt_run")       # one-row run manifest
run_db.read("bt_orders")    # every accepted/rejected/cancelled order
run_db.read("bt_fills")     # authoritative executions
run_db.read("bt_positions") # final portfolio state
run_db.read("bt_equity")    # sampled equity curve
```

Market inputs use the canonical `instruments`, `book_deltas`, `trades`,
`bars`, `funding`, and `resolutions` schemas. Replay order follows `ts_init`.
Rows sharing a `book_deltas.event_index` are one atomic event and end where
`is_last=True`. A named snapshot pins market data; signals are versioned on
their own axis. `slippage_ticks` currently takes precedence over queue mode,
so recipes should test those assumptions as separate scenarios.

For external L2 data, pin two layers: hash the exact source Parquets, then
create a named h5i-db snapshot after normalization. Preserve both event and
arrival timestamps, count any clock repairs, and keep future resolution labels
out of the feature table. Periodic full-book snapshots can test market-order
execution and depth sensitivity, but cannot support exact queue-position
claims between snapshots. Recipes 04/04 and 04/05 apply this contract to a
bounded, non-commercial Kaggle Polymarket sample.

## DataFrame builder cheatsheet

**Recipes are builder-first.** `db.table(...)` starts a lazy query you build
with verbs; it compiles to SQL run through `db.sql()`, so it sees the same
session, table functions and version pins. Prefer it over a SQL string
wherever it expresses the query cleanly - it reads better and, for generated
queries, removes a class of quoting bugs. Recipe 00/09 teaches it; 00/04 stays
pure SQL on purpose (its subject *is* SQL).

```python
from h5i_db import col, lit, when, sql_expr, count_star, vwap, wavg, time_bucket

db.table("trades")                          # FROM "trades" (snapshot-bound)
db.table("trades", version=42)               # -> h5i('trades', 42); also
db.table("trades", as_of="2026-07-01T00:00:00Z")   # snapshot="eod-..."

.filter(*preds)        # ANDed; use & | ~ , NOT and/or/not, and parenthesize
                       #   each comparison (& binds tighter than >)
.select(*exprs, **named)          # replace projection; kwargs name the result
.with_columns(*e, replace=None, **named)   # ADD columns; naming an existing
                       #   one is an error unless you pass replace="name"
.group_by(*keys).agg(*e, **named) / .count(name="count")
.sort(by, descending=False)  .limit(n, offset=0)  .head(n)  .unique()
.join(other, on=/left_on=/right_on=, how="inner", predicate=sql_expr(...))
.join_asof(other, on=, by=, direction="backward", tolerance=us)
.pipe(fn, *args)
# terminals: .collect(timeout=, max_rows=) .to_pandas() .to_arrow()
#            .to_polars() .sql() .explain() .schema()

col("x"), lit(3), col("x", relation="l")     # l / r are the join-side aliases
col("a").is_in([...]) .is_null() .is_not_null() .between(lo, hi) .like("A%")
col("a").cast("DOUBLE") .abs() .log() .exp() .sqrt() .round(2) .coalesce(0)
col("a").sum() .mean() .min() .max() .count() .n_unique() .std() .var()
        .median() .quantile(q) .first("ts") .last("ts")   # bar open/close
when(c).then(a).otherwise(b)      # a when/then chain is already an expression
count_star(), vwap(price, size), wavg(weight, value)
time_bucket("5m", col("ts"), timezone="America/New_York")   # or origin=...
col("x").over(partition_by=, order_by=, rows=n | (pre, post), duration="30m")
col("x").rolling_mean(20, order_by="ts", partition_by="symbol")
#   also rolling_sum/min/max/std/var/count/mad/skew/kurt/rank/idxmax/idxmin,
#   rolling_corr(other, ...), rolling_cov(other, ...), .ewma(alpha, order_by,
#   partition_by). Unlike the rolling_* SQL sugar these DO partition.
col("x").cs_rank(partition_by="ts") .cs_zscore(...) .cs_demean(...)
        .cs_winsorize(lo, hi, partition_by="ts")
sql_expr("approx_percentile_cont(price, 0.99)")     # raw fragment, verbatim
```

Gotchas that bite in recipes:
- **Never alias a computed group key to an existing column name.**
  `group_by(time_bucket('5m', col('ts')).alias('ts'))` renders `GROUP BY "ts"`,
  which binds to the *raw* `ts` column - one group per row, silently wrong and
  no error. Bucket to `bar`, then `.select(col("bar").alias("ts"), ...)` a
  level down. (Verified 2026-07-26: 5,660 groups instead of 156.)
- **`agg -> select(rename) -> sort -> select(aggregate)` generates invalid
  SQL** - the rename level is dropped but its `ORDER BY` survives, so the
  outer query references a name that no longer exists. Keep the trailing
  `.sort()` at the call site instead of inside a reusable frame-builder
  function. (h5i-db builder bug, found 2026-07-26, not yet reported upstream.)
- **`time_bucket()` parses a narrower interval grammar than the SQL function.**
  Only `<n>` + `s|m|h|d|w|mo|y`: `'1s'` yes, `'1sec'` and `'250ms'` raise
  `InvalidInputError`. Sub-second widths need `sql_expr("time_bucket('250ms',
  ts)")` or plain SQL.
- **No `.lag()` / `.row_number()`.** Use the windowable escape hatch:
  `sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")`. Bind it
  to a name and reuse it.
- Expressions carry **SQL** semantics: `/` between integer columns truncates.
  `col("size").cast("DOUBLE") / 100`.
- A stage reading a column an earlier stage *computed* gets its own subquery
  level, so `with_columns(a=...)` then `with_columns(b=col("a") ...)` is two
  stages, not one. Aggregation, `LIMIT` and `DISTINCT` also close a level, and
  a stage only sees what the one before it emitted.
- `join()` aliases both sides `l` / `r` and does **not** dedupe column names -
  project explicitly.
- `join_asof()` needs plain, unpinned tables on both sides (the `asof_join`
  table function takes names and reads at latest). Filter *after* the join.
- `from h5i_db import col` collides with loop variables named `col` - rename
  them (`for name in (...)`) in any recipe that imports it.
- **A numpy scalar in an expression renders as a function call.**
  `col("vol") * np.sqrt(252)` compiles to `"vol" * np.float64(15.87)` and fails
  planning. Bind the constant with `float(...)` first.
- **`.select()` after `.sort()` on the same frame is refused** when the
  projection drops the ordering columns (`_reject_shadowed_ordering`). Keep the
  unsorted frame around and sort at the point you collect.

## Quant layer cheatsheet

```python
from h5i_db import quant

series = quant.returns(db, "strategy_returns", snapshot="v1",
                       annualization=quant.DAILY)     # or from_levels(fork, "bt_equity")
series.stats(benchmark=other)      # empyrical's perf_stats, plus alpha/beta
series.drawdown_table(top=5); series.underwater(); series.equity_curve()
series.rolling_sharpe(63); series.rolling_volatility(63); series.rolling_beta(other, 126)
quant.tearsheet(series, path="...", benchmark=other)  # self-contained HTML

panel = quant.build_panel(db, factor_frame, price_frame, periods=(1, 5, 21),
                          quantiles=5, group=SECTORS, max_loss=0.35)
panel.loss_report(); panel.ic(); panel.mean_ic(by="1mo", by_group=True)
panel.ic_decay(); panel.quantile_returns(); panel.spread(); panel.turnover()
panel.rank_autocorrelation(); panel.alpha_beta(); panel.cumulative_returns(period=21)
quant.factor_report(panel, path="...")

quant.purged_kfold(n, folds=5, horizons=[21]*n, embargo=0.01)
quant.combinatorial_purged(n, groups=6, test_groups=2, horizons=..., embargo=...)
quant.walk_forward(n, train_size=500, test_size=125, expanding=False)
quant.probability_of_backtest_overfitting(matrix, partitions=8)
quant.deflated_sharpe(returns, trials=8, trials_source="counted")
quant.minimum_track_record_length(returns)

quant.sweep(db, {"lookback": [63, 126]}, fn, prefix="mom", keep_going=True)
quant.verify(subject, rerun=lambda: rebuild())        # refuses anything unpinned
quant.restatement_impact(build, db, before={"version": 1}, after={}, metric=fn)
quant.basket_payload(db, {"label": result}, panels=quant.PORTFOLIO_PANELS + ("equity",))
costs.SlippageSample(...); costs.effective_spread(s); costs.implementation_shortfall(s)
costs.fit_impact(s, shape="sqrt"); costs.fit_from_fills(fork)
```

Gotchas that bite here too:
- **A lazy frame passed to `build_panel` is taken as given**, so the pin has to
  be on the frame (`db.table(name, snapshot=...)`). Passing an unpinned frame
  with `snapshot=` produces a panel whose provenance claims a pin it did not use.
- **`fit_from_fills` measures depth consumed, not shortfall.** Without a stored
  decision price it references the same-instant VWAP of the other fills, so it
  needs orders that walked more than one level. Build `SlippageSample`s yourself
  when you know the decision price.
- **`slippage_ticks` is denominated in a fixed 0.0001 in the Python binding**,
  not in the instrument's `tick_size`. On a prediction market those coincide; on
  a 290-dollar share they differ by a hundred times. 04/11 asserts the unit
  before charging anything.
- **`backtest.signal_table` rounds timestamps through a float**, so a stamp can
  land a few nanoseconds off. `pd.DatetimeIndex(...).floor("us")` before joining
  against microsecond data, or the cast raises.
- **A study cannot vary `data.signals`** - that field is data identity. Search a
  strategy parameter with one signals table per candidate (04/12) and use
  `backtest.study` for execution, portfolio and risk axes.

Keep in `db.sql()` (no verb, and the string is clearer): `UNION ALL`, deep
multi-CTE chains, scalar subqueries, `gapfill` / `resample` / `tail`, the
`ASOF JOIN` keyword form over pinned tables, and anything whose *subject* is
the SQL text itself (00/04, 03/10).

## SQL cheatsheet (DataFusion + h5i extensions)

```sql
-- Time travel: latest / version / as-of commit time / named snapshot
SELECT * FROM trades;
SELECT * FROM h5i('trades', 42);
SELECT * FROM h5i('trades', '2026-07-01T00:00:00Z');
SELECT * FROM h5i('trades', 'eod-2026-07-21');

-- OHLCV + VWAP rollup (streams on sorted storage - no sort)
SELECT time_bucket('5m', ts) AS bar, symbol,
       first_value(price ORDER BY ts) AS open, max(price) AS high,
       min(price) AS low,  last_value(price ORDER BY ts) AS close,
       sum(size) AS volume, vwap(price, size) AS vwap
FROM trades GROUP BY bar, symbol ORDER BY bar;
-- time_bucket widths: '5s','250ms','1m','1h','1d','7d','1mo','1y' (plural
-- spellings like '30 seconds' work too); optional 3rd arg: origin timestamp
-- or IANA tz ('America/New_York') for session-aligned days.

-- ASOF join (sort-free): for each left row, latest right row at/before it
SELECT * FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol');
-- direction/tolerance: asof_join('t','q','ts','ts','symbol','backward', 5000000)
-- (tolerance in raw time units = us). Keyword form also works:
--   FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts)
--     ON trades.symbol = quotes.symbol
-- (bare table names only - aliases are rejected by the planner)
-- Unmatched left rows keep NULLs; colliding right columns get _right suffix.
-- String-family by-keys coerce across encodings (utf8 / large_string /
-- dictionary), so pandas-built tables join stored ones directly; other
-- type mismatches still error.
-- Hygiene worth keeping in any ASOF pipeline: assert len(output) ==
-- len(left) (a LEFT ASOF join is 1:1 with its left side) and cross-check
-- one row against a point query. (Builds before 2026-07-23 silently
-- truncated joins beyond 8,192 rows per side - upgrade if you see that.)

-- Regular grids for illiquid series: step is raw units (us!), fill mode:
SELECT * FROM gapfill('bars_1m', 'ts', 60000000, 'locf');      -- also 'null','interpolate'
-- resample(...) is an exact alias; <= 1M generated rows.
-- CAUTION: gapfill has NO per-key grouping - on a multi-symbol table locf
-- carries whichever symbol last ticked. Gapfill single-instrument tables (or
-- filter into one first). 'null' mode only emits values where an observation
-- lands exactly on a grid instant.

-- Window functions
SELECT ts, symbol, price,
       ewma(price, 0.06) OVER (PARTITION BY symbol ORDER BY ts) AS px_smooth,
       rolling_avg(price, ts, 20) AS ma20,          -- sugar; also rolling_sum/min/max
       lag(price) OVER (PARTITION BY symbol ORDER BY ts) AS prev_px
FROM trades;
-- CAUTION: rolling_* sugar is NOT partitioned - it is a trailing n-ROW window
-- in global time order, so on a multi-symbol table it mixes symbols. Use it on
-- single-symbol subsets only; for per-symbol windows write the explicit
-- AVG(x) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN n-1 PRECEDING AND CURRENT ROW).
-- rolling_* also cannot take its own OVER clause.

-- Streaming tail (unbounded - ALWAYS use LIMIT in notebooks)
SELECT * FROM tail('trades', 5, 50) LIMIT 100;   -- versions after 5, poll 50ms
-- tail requires a PURE-APPEND version chain: any delete/replace/restore/write
-- in the range makes it error with an informative hint. Use append-only tables
-- for streaming demos. tail BLOCKS until LIMIT rows arrive - if LIMIT exceeds
-- what's available it polls until the query timeout= fires (TimeoutError), so
-- size LIMIT from versions() row deltas and pass a timeout as a backstop.
-- Aggregations can't run directly over the unbounded stream - consume rows
-- first, aggregate client-side.
```

Full SQL otherwise: joins, CTEs, subqueries, `date_trunc`, `date_bin`,
`stddev`, `corr`, `approx_percentile_cont`, etc. `EXTRACT(hour FROM ts)`,
`to_timestamp_micros`, arithmetic on timestamps via `INTERVAL`.

Gotchas:
- `ts` columns are `timestamp[us, UTC]` → every raw-unit argument
  (plan ranges, gapfill step, asof tolerance, read time_start/end) is
  **microseconds**. `int(pd.Timestamp("2026-06-02", tz="UTC").value // 1000)`.
- `append` requires strictly ordered data - later timestamps than what's
  stored. Simulating multi-day feeds: append day by day, in order.
- `append` enforces the *full* `sort_key`, not just the time column: with
  `sort_key=["ts","symbol"]`, data must be sorted by ts *then* symbol
  (`table.sort_by([("ts","ascending"),("symbol","ascending")])`) or it is
  rejected with `sort_order_violation`.
- String literals in SQL use single quotes; identifiers are case-insensitive.
- `last_value(x ORDER BY ts)` inside GROUP BY is the h5i/DataFusion idiom for
  "closing" values (no need for self-joins).
- Batch appends (one commit per chunk of rows, not per row) - every commit
  writes a manifest. After many small appends, `db.compact(...)`.
```
