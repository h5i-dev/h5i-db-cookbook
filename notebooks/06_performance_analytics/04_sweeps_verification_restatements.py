# %% [markdown]
# # Sweeps, verification, and what a restatement did to your alpha
#
# Three questions arrive in order at the end of a piece of research. What did
# the parameter grid say? Can somebody else reproduce the number? And when the
# vendor revises the data next month, does the number change?
#
# The first is ordinary work. The second and third are where a versioned store
# stops being a storage detail: `quant.verify` refuses to bless a computation
# that was not pinned, and `quant.restatement_impact` answers "did the revision
# move my answer" by running the same computation at two read points, which is a
# question a directory of Parquet files cannot be asked.

# %% [markdown]
# ## Terms used here
#
# | term          | meaning |
# | ------------- | --- |
# | sweep         | running one computation over a grid of parameters |
# | fork          | a copy-on-write branch of the database a trial writes into |
# | provenance    | the pin, parameters and SQL that produced a number |
# | digest        | a hash of that provenance, so two results can be compared exactly |
# | restatement   | a vendor correcting data you have already used |
# | as-of read    | reading the database as it stood at an earlier moment |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import json

import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

# %% [markdown]
# ## 1. The data, and a version of it
#
# Daily prices for thirty names. The important part is the version history:
# every commit is readable afterwards, which is what makes the last section of
# this recipe possible.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_sweeps_verification_restatements"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="vendor load, first delivery")
db.snapshot("prices-v1", tables=["prices"], note="What the research below read")
first_version = db.versions("prices")[-1]["sequence"]
print(f"prices is at version {first_version}")
print(f"{prices.num_rows:,} rows")
daily.to_pandas().head(3)


# %% [markdown]
# ## 2. A sweep is one fork per trial
#
# `quant.sweep` calls a function once per parameter combination, with a database
# opened on that trial's own fork. Anything the function writes lands in the
# fork and nowhere else, so trials cannot contaminate each other or the base
# data, and a failed trial leaves no mess to clean up.
#
# The function returns metrics; the sweep writes them into the fork alongside
# the parameters that produced them.

# %%
def score(fork, params) -> dict:
    """Build a factor panel inside this trial's fork and score it."""
    pinned = fork.table("prices", snapshot="prices-v1")
    factor = (
        pinned.with_columns(
            past=sql_expr(f"lag(adj_close, {params['lookback']})").over(
                partition_by="symbol", order_by="ts"
            )
        )
        .with_columns(value=col("adj_close") / col("past") - 1)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("value"))
    )
    price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
    panel = quant.build_panel(
        fork,
        factor,
        price_frame,
        periods=(params["horizon"],),
        quantiles=params["quantiles"],
    )
    decay = panel.ic_decay().to_pandas().iloc[0]
    spread = panel.spread().to_pandas()[f"spread_{params['horizon']}"]
    return {
        "mean_ic": float(decay["mean_ic"]),
        "ic_t_stat": float(decay["t_stat"]),
        "spread_sharpe": float(spread.mean() / spread.std() * np.sqrt(252)),
        "observations": int(decay["n"]),
    }


swept = quant.sweep(
    db,
    {"lookback": [63, 126, 252], "quantiles": [3, 5], "horizon": [21]},
    score,
    prefix="momentum",
    note="momentum lookback and bucket count",
)
print(swept)
compared = swept.to_pandas()
board = pd.concat(
    [
        pd.json_normalize(compared["_params"].map(json.loads)),
        compared[["mean_ic", "ic_t_stat", "spread_sharpe", "observations"]],
    ],
    axis=1,
).sort_values("spread_sharpe", ascending=False)
board.round(4)

# %% [markdown]
# `compare()` is one query across every fork at once. The forks share their
# base's segments, so comparing a hundred trials costs about what reading one
# does, plus what each trial actually wrote. Parameters travel as a JSON string
# in `_params`, so a trial's identity survives whatever columns its metrics
# happened to have.

# %%
best = swept.best("spread_sharpe")
print(f"forks created  {len(swept.forks)}")
print(f"best trial     lookback={best['lookback']:.0f} quantiles={best['quantiles']:.0f}")
print(f"               spread Sharpe {best['spread_sharpe']:.3f}, IC t-stat {best['ic_t_stat']:.2f}")
print(f"comparison table columns: {swept.compare().to_arrow().column_names}")

# %% [markdown]
# A sweep that stops at the first failure wastes the trials that already ran.
# `keep_going` records the failure and continues, which is the behaviour you
# want overnight and the behaviour you must not have when the failure means the
# data is wrong.

# %%
def brittle(fork, params) -> dict:
    if params["lookback"] > 1_000:
        raise ValueError("no history that long")
    return {"lookback": float(params["lookback"])}


partial = quant.sweep(
    db, {"lookback": [63, 5_000]}, brittle, prefix="brittle", keep_going=True
)
print(f"{len(partial)} trials succeeded, {len(partial.failures)} failed")
print(partial.failures)

# %% [markdown]
# ## 3. Verification is a refusal, not a checkmark
#
# `quant.verify` re-executes the computation and checks two things: the
# provenance digest is unchanged, and the recomputed numbers agree. An unpinned
# computation is reported as unverifiable rather than passed, because two runs
# against "latest" agreeing proves only that nothing changed in between.

# %%
def build_panel(pin: dict):
    pinned = db.table("prices", **pin)
    factor = (
        pinned.with_columns(
            past=sql_expr("lag(adj_close, 126)").over(partition_by="symbol", order_by="ts")
        )
        .with_columns(value=col("adj_close") / col("past") - 1)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("value"))
    )
    price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
    return quant.build_panel(db, factor, price_frame, periods=(21,), quantiles=5, **pin)


panel = build_panel({"snapshot": "prices-v1"})
report = quant.verify(panel, rerun=lambda: build_panel({"snapshot": "prices-v1"}))
print(f"verified {report['verified']}  pinned {report['pinned']}  digest {report['digest'][:16]}")

unpinned = build_panel({})
relaxed = quant.verify(unpinned, strict=False)
print(f"unpinned: verified={relaxed['verified']} reason={relaxed['reason']!r}")

# %% [markdown]
# ## 4. The vendor restates
#
# A correction arrives: one session of one symbol was wrong. The fix is an
# ordinary previewable mutation, so the old version stays readable and the new
# one is what "latest" now means.

# %%
target_symbol, target_day = "GE", pd.Timestamp("2024-03-14", tz="UTC").date()
frame = daily.to_pandas()
# The session stamp carries a time, so a whole day is a half-open range rather
# than an equality test.
window = frame[frame["ts"].dt.date == target_day].copy()
print("as delivered:")
print(window[window["symbol"] == target_symbol][["ts", "symbol", "close", "adj_close"]].to_string(index=False))

corrected = window.copy()
mask = corrected["symbol"] == target_symbol
corrected.loc[mask, ["close", "adj_close"]] = (
    corrected.loc[mask, ["close", "adj_close"]] * 0.82
).values
day_start = int(pd.Timestamp(target_day, tz="UTC").value // 1_000)
day_end = day_start + 24 * 60 * 60 * 1_000_000
plan = db.plan_replace_range(
    "prices",
    day_start,
    day_end,
    data=pa.Table.from_pandas(
        corrected.sort_values("symbol"), schema=prices.schema, preserve_index=False
    ),
    note="vendor restatement: GE close was 18% too high",
)
print(f"\nrows affected {plan.summary['rows_affected']}, "
      f"rows after {plan.summary['rows_after']:,}")
plan.apply()
print(f"prices is now at version {db.versions('prices')[-1]['sequence']}")

# %% [markdown]
# ## 5. Did the restatement change the answer?
#
# `restatement_impact` runs one computation at two read points and reports the
# differences per metric. This is the question a versioned store exists to
# answer: not "what is my alpha" but "did the revision move it, and by how
# much".

# %%
def headline(built) -> dict:
    decay = built.ic_decay().to_pandas().iloc[0]
    return {
        "mean_ic": float(decay["mean_ic"]),
        "ic_t_stat": float(decay["t_stat"]),
        "observations": float(decay["n"]),
    }


impact = quant.restatement_impact(
    build_panel,
    db,
    before={"version": first_version},
    after={},
    metric=headline,
)
print(f"changed: {impact['changed']}")
pd.DataFrame(impact["metrics"]).T.round(6)

# %% [markdown]
# One corrected close, in one name out of thirty, on one session out of two
# thousand, moved the mean information coefficient in the fifth decimal place
# and its t-statistic by about a hundredth. Small, and not zero, which is the
# point: the size was measured rather than assumed. A restatement that touched a
# whole sector, or a survivorship correction that removed a name, will not read
# like this.
#
# The reason to run it every time is that the cost is one query, and the
# alternative is discovering the sensitivity from a production difference.

# %%
sensitive = quant.restatement_impact(
    build_panel,
    db,
    before={"version": first_version},
    after={},
    metric=lambda built: {
        "top_bucket_21d": float(
            built.quantile_returns().to_pandas().iloc[-1]["mean_21"]
        )
    },
)
pd.DataFrame(sensitive["metrics"]).T.round(8)

# %% [markdown]
# ## 6. Cleaning up
#
# A sweep's forks are cheap but not free, and research databases accumulate
# them. `drop()` removes a sweep's forks and the comparison table with them.

# %%
print(f"forks before {len([name for name in db.fork_names()])}")
dropped = partial.drop()
print(f"dropped {dropped} brittle-sweep forks")
print(f"forks after  {len([name for name in db.fork_names()])}")

# %% [markdown]
# ## Takeaways
#
# - `quant.sweep` gives every trial its own fork, so trials cannot contaminate
#   each other and `compare()` reads them all back in one query.
# - `keep_going` is right for an overnight grid and wrong when a failure means
#   the data is broken.
# - `quant.verify` checks the provenance digest *and* the recomputed values, and
#   refuses to verify anything unpinned.
# - A restatement is an ordinary versioned commit, so the old answer stays
#   readable rather than being overwritten.
# - `restatement_impact` turns "the vendor revised the data" into a number per
#   metric, which is the only form in which that fact is actionable.
# - Drop sweep forks when the research is done; the comparison table goes with
#   them.

# %%
db.close()
