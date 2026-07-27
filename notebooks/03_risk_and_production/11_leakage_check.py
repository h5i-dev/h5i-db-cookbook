# %% [markdown]
# # Leakage checks: which of last night's backtests actually held up?
#
# A research agent - or a parameter sweep, or a junior with a for-loop - runs
# forty backtests overnight. In the morning there are forty Sharpes. Reviewing
# them properly means re-deriving each one against the data as it stood at the
# decision instant, and nobody does that forty times, so in practice the top of
# the list gets promoted and the rest get deleted. That is exactly the selection
# procedure that promotes leaks.
#
# `arrival_delta` turns that review into a number. It runs one query twice -
# against the current head, and against a decision read point - and reports the
# delta. The part of a metric that moves is the part that depended on data which
# had not arrived when the trade would have been placed: the alpha that
# evaporates in production.
#
# This recipe measures a real one, then spends equal time on the two ways the
# check can mislead you - a *vacuous* result, and an event-time leak it is blind
# to by construction. A diagnostic you trust incorrectly is worse than none.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_leakage"), create=True)
rng = np.random.default_rng(11)

# %% [markdown]
# ## 1. A panel with an arrival history
#
# The point of the exercise depends on the database being a record of *when
# things arrived*, not just of what they say now. So we load the vendor's first
# publication, then let corrections land later as their own commits - which is
# what a vendor feed actually looks like once you keep the history.

# %%
symbols = [f"STK{i:03d}" for i in range(60)]
panel = cu.make_daily_prices(symbols=symbols, days=500)

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])

first_publication = (
    panel.select(["ts", "symbol", "close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
)
db.append("prices", first_publication, note="vendor delivery, first publication")

pdf = first_publication.to_pandas()
print(f"{len(pdf):,} rows, {pdf.ts.min().date()} .. {pdf.ts.max().date()}, version 1")

# %% [markdown]
# ## 2. The corrections arrive
#
# Three restatements, each a separate commit: a fat-finger print, a missed
# split, and a late adjustment. Because a range replacement rewrites the whole
# window, the replacement frame has to carry the innocent rows of that day
# through unchanged - the helper below reads the window, patches it, and hands
# the full window back.

# %%
def restate(day: pd.Timestamp, symbol: str, factor: float, note: str) -> int:
    """Replace one trading day, correcting a single symbol's close."""
    start_us = day.value // 1_000
    end_us = (day + pd.Timedelta(days=1)).value // 1_000
    window = db.read("prices", time_start=start_us, time_end=end_us).to_pandas()
    window.loc[window.symbol == symbol, "close"] *= factor
    fixed = pa.Table.from_pandas(
        window.sort_values(["ts", "symbol"])[["ts", "symbol", "close"]],
        schema=schema,
        preserve_index=False,
    )
    plan = db.plan_replace_range("prices", start_us, end_us, data=fixed, note=note)
    return plan.apply()["sequence"]


days = pd.DatetimeIndex(sorted(pdf.ts.unique()))
# Corrections land against days inside the sample, the way real ones do.
for day, sym, factor, why in [
    (days[-40], "STK003", 0.5, "missed 2:1 split"),
    (days[-25], "STK011", 0.97, "fat-finger print, vendor corrected"),
    (days[-12], "STK007", 1.02, "late dividend adjustment"),
]:
    seq = restate(day, sym, factor, why)
    print(f"v{seq}: {why} on {day.date()} ({sym})")

for v in db.versions("prices"):
    print(f"  v{v['sequence']:>2} {v['op']:<13} {v.get('note', '')}")

# %% [markdown]
# ## 3. The metric, written once, as a single query
#
# `arrival_delta` re-runs a *query*, so the whole strategy has to be expressible
# as one. That constraint is a feature: a metric you can hand to the database is
# a metric you can hand to any read point, including a past one.
#
# Below is a plain cross-sectional momentum book - 21-day momentum, demeaned
# each day, weighted by conviction, held one day. The signal deliberately uses
# `prev`, the close *before* the return being earned, so the strategy itself
# contains no look-ahead. Everything we are about to measure comes from the
# data moving, not from the code cheating.

# %%
# arrival_delta() takes a SQL string (it re-runs the same text at two read
# points), so this study stays a string end to end.
SHARPE_SQL = """
WITH d AS (
  SELECT ts, symbol, close,
         lag(close, 1)  OVER (PARTITION BY symbol ORDER BY ts) AS prev,
         lag(close, 22) OVER (PARTITION BY symbol ORDER BY ts) AS prev22
  FROM prices
),
sig AS (
  SELECT ts,
         close / prev - 1.0   AS r,
         prev / prev22 - 1.0  AS mom
  FROM d
  WHERE prev IS NOT NULL AND prev22 IS NOT NULL
),
w AS (
  SELECT ts, r, mom - avg(mom) OVER (PARTITION BY ts) AS wt
  FROM sig
),
daily AS (
  SELECT ts, sum(wt * r) / nullif(sum(abs(wt)), 0) AS pnl
  FROM w GROUP BY ts
)
SELECT avg(pnl) / nullif(stddev(pnl), 0) * sqrt(252.0) AS sharpe FROM daily
"""

head_sharpe = db.sql(SHARPE_SQL).to_pandas().sharpe.iloc[0]
print("head Sharpe:", round(head_sharpe, 3))

# The data is a synthetic random walk with a common factor, so this number is
# noise around zero and its *level* means nothing. Everything below is about
# how much it moves when the read point changes, which is a property of the
# data's history rather than of the strategy.

# %% [markdown]
# ## 4. The check itself
#
# The decision point is version 1 - the state of the world when the research was
# done, before any correction had arrived. Everything after it is hindsight.

# %%
report = db.arrival_delta(SHARPE_SQL, version=1)

print(f"changed          : {report['changed']}")
print(f"vacuous          : {report['vacuous']}")
metric = report["columns"][0]
print(f"head             : {metric['head']:.3f}   (what the backtest reported)")
print(f"as-of v1         : {metric['asof']:.3f}   (what it could actually have known)")
print(f"delta            : {metric['delta']:+.3f} Sharpe")
print("\nwithheld commits:")
for w in report["withheld_versions"]:
    print(f"  {w['table']}: v{w['asof_version']} -> v{w['head_version']}")

# %% [markdown]
# Read the *absolute* delta, in the metric's own units: this book's Sharpe moved
# by about half a point between what could be known and what is known now. The
# report also carries `delta_pct`, but a ratio metric sitting near zero makes
# that percentage explode - a Sharpe that goes from -0.36 to -0.85 is "-133%",
# which is arithmetically true and useless. Percentages are for metrics with a
# meaningful base; use the raw delta for the rest.
#
# It is not a verdict that the strategy is broken. The restatements are genuine
# improvements to the data, and the corrected Sharpe is the more honest one.
# What the delta says is: *this much of the result was not knowable at decision
# time*, so a live run starting that day would not have traded this book. Use it
# as a haircut, and as a ranking key when you have forty of these and time to
# examine three.

# %%
print("\n".join(report["notes"]))

# %% [markdown]
# ## 5. Triage across a sweep
#
# Which is the actual workflow. Run the same check across every variant that
# came out of the overnight sweep and sort by exposure, not by Sharpe.

# %%
def with_lookback(n: int) -> str:
    """Same book, different momentum window. Asserted so a silent no-op in the
    substitution cannot quietly turn the sweep into three copies of one query."""
    sql = SHARPE_SQL.replace("lag(close, 22)", f"lag(close, {n})").replace("prev22", f"prev{n}")
    assert (n == 22) or (sql != SHARPE_SQL), f"lookback substitution failed for {n}"
    return sql


VARIANTS = {f"mom_{n}d": with_lookback(n) for n in (22, 66, 5)}

rows = []
for name, sql in VARIANTS.items():
    rep = db.arrival_delta(sql, version=1)
    c = rep["columns"][0]
    rows.append(
        {
            "variant": name,
            "reported": c["head"],
            "knowable": c["asof"],
            "delta": c["delta"],
            "vacuous": rep["vacuous"],
        }
    )

triage = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False)
print(triage.round(3).to_string(index=False))

# %% [markdown]
# A gate follows directly: promote nothing whose reported edge survives only
# with hindsight. The threshold is a house parameter; the discipline is that it
# exists and is computed rather than eyeballed.

# %%
TOLERATED_SHARPE_MOVE = 0.15
promoted = triage[triage.delta.abs() <= TOLERATED_SHARPE_MOVE]
print(
    f"promoted {len(promoted)} of {len(triage)} variants "
    f"at |delta| <= {TOLERATED_SHARPE_MOVE} Sharpe"
)
if promoted.empty:
    print("  (none survive: a single missed split moves every variant in this book)")

# %% [markdown]
# ## 6. Failure mode 1: a zero that means nothing
#
# Here is the part that decides whether this diagnostic helps you or lulls you.
# `arrival_delta` compares two read points. If both resolve to the same version
# - which is the case for any database loaded in a single bulk ingest, the
# normal cold start - then it compares identical data and the delta is *forced*
# to zero. It has measured nothing, and a zero that means nothing looks exactly
# like a clean bill of health.
#
# The report says so, in `vacuous`. Check that field before you read the number.

# %%
bulk = h5i_db.Database(cu.fresh_db("prod_leakage_bulk"), create=True)
bulk.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])
bulk.append("prices", first_publication, note="ten years, one commit")

bulk_report = bulk.arrival_delta(SHARPE_SQL, version=1)
print(f"changed          : {bulk_report['changed']}   <- looks clean")
print(f"vacuous          : {bulk_report['vacuous']}   <- but it checked nothing")
print(f"withheld commits : {len(bulk_report['withheld_versions'])}")
print()
print(bulk_report["notes"][0])
bulk.close()

# %% [markdown]
# ## 7. Failure mode 2: the leak it cannot see
#
# `arrival_delta` measures the *arrival* axis: rows that exist now but had not
# been published yet. It is blind to the other axis - rows that were always in
# the table, read at a moment you should not have been reading them.
#
# The classic version is a signal that uses the same day's close to trade that
# same day's close. Below, `mom` is built from `close` instead of `prev`, which
# makes the strategy trade on information it earns simultaneously. The Sharpe is
# absurd, the leak is total, and the check reports essentially nothing.

# %%
LEAKY_SQL = SHARPE_SQL.replace("prev / prev22 - 1.0  AS mom", "close / prev22 - 1.0 AS mom")
assert LEAKY_SQL != SHARPE_SQL, "the leaky-signal substitution did not apply"

leaky_report = db.arrival_delta(LEAKY_SQL, version=1)
lc = leaky_report["columns"][0]
inflation = lc["head"] - metric["head"]
print(f"leaky Sharpe     : {lc['head']:.2f}  (honest signal: {metric['head']:.2f})")
print(f"inflation from the leak : {inflation:+.2f} Sharpe")
print(f"arrival delta           : {lc['delta']:+.2f} Sharpe")
print(f"vacuous                 : {leaky_report['vacuous']}")

# %% [markdown]
# Note what the check did and did not tell you. It reports a real arrival delta
# for this query, because the restatements move the leaky signal too - but that
# delta is measuring the restatements, not the peeking. Nothing in the report
# separates a signal that reads its own bar from one that does not, because
# that is not what it measures. Whatever the delta comes out to, large or
# small, it is not a verdict on look-ahead, which is what the report's own
# `notes` say on every run.
#
# The event-time axis needs a different tool: a cutoff applied to the scan
# itself.
#
# In SQL you can write the bound by hand. Each step of a walk-forward
# evaluation is exactly this: decide at T, seeing only rows stamped at or
# before T. The check below is deliberately a row count rather than another
# Sharpe - comparing Sharpes across the bound would conflate the cutoff with
# the shorter sample it produces, which is its own way of lying with numbers.

# %%
decision = days[-60]
cutoff = decision.strftime("%Y-%m-%dT%H:%M:%SZ")

n_rows = lambda frame: frame.select(count_star().alias("n")).to_pandas().n.iloc[0]
visible = n_rows(db.table("prices").filter(col("ts") <= cutoff))
total = n_rows(db.table("prices"))
print(f"decision date {decision.date()}: {visible:,} of {total:,} rows readable")
print(f"future rows hidden: {total - visible:,}")

# %% [markdown]
# The trouble with writing that bound by hand is that you have to remember it
# every single time, in every subquery of every variant, forever - and a leaked
# backtest looks like a good one, so the day you forget is the day you get
# promoted.

# %% [markdown]
# The CLI makes that bound structural instead of remembered - the session is
# pinned and *no* query inside it can reach past the instant, including one that
# explicitly asks:
#
# ```bash
# h5i-db query prod_leakage.db "<the same SQL>" \
#   --decision-time 2026-05-02T00:00:00Z \
#   --embargo 1d
# ```
#
# Use both: the cutoff to stop event-time leaks happening, `arrival_delta` to
# measure the arrival leaks that no cutoff can prevent, because they are caused
# by the world learning something after you did.

# %% [markdown]
# ## Takeaways
#
# - **`arrival_delta` prices hindsight.** One query, two read points; the delta
#   is how much of a result was not knowable at decision time. Read it in the
#   metric's own units and rank a sweep by it rather than by Sharpe -
#   `delta_pct` explodes on any metric whose base sits near zero, which a
#   Sharpe does.
# - **Read `vacuous` before you read the number.** A single-commit database
#   makes the check compare identical data, and its zero is arithmetic rather
#   than evidence. This is the normal state of a freshly bulk-loaded store.
# - **It only sees the arrival axis.** A same-bar look-ahead is invisible to
#   it: the delta it reports for a peeking signal is measuring restatements,
#   not the peeking, so no value of that delta clears a signal of look-ahead.
#   The report's `notes` say so on every run; believe them.
# - **One restated split can outweigh a whole book.** A single corrected symbol
#   moved every variant here by more than 0.4 Sharpe. Corporate actions are not
#   a rounding error in the arrival history; they are usually the whole story.
# - **The two tools are complementary, not alternatives.** `--decision-time`
#   prevents event-time leaks structurally; `arrival_delta` measures arrival
#   leaks, which are caused by the data changing and so cannot be prevented at
#   all - only quantified.
# - **h5i-db features doing the work:** immutable per-commit versioning (the
#   arrival history that makes the question answerable), O(1) time travel (both
#   runs are cheap), and previewable `plan_replace_range` (the restatements are
#   auditable rather than silent overwrites).

# %%
db.close()
