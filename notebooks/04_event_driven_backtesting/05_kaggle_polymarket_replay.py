# %% [markdown]
# # Causal signal replay on real Polymarket books
#
# This recipe builds a deliberately modest microstructure strategy on the
# bounded [Kaggle Polymarket sample](https://www.kaggle.com/datasets/marvingozo/polymarket-tick-level-orderbook-dataset).
# The professional lesson is the experimental design:
#
# - features become observable only after their minute closes;
# - the eventual resolution label is excluded;
# - orders meet real recorded books, not feature-bar prices;
# - fees, latency, and slippage are sensitivity dimensions;
# - every run is pinned and queryable on its own fork.

# %%
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

CACHE = Path("data/cache/kaggle-polymarket")
sample = cu.load_kaggle_sample(CACHE, depth_levels=10)
print(sample.question)
print(pd.Series(sample.audit, name="value").to_frame())

# %% [markdown]
# ## Build a causal feature
#
# `depth_imbalance` at minute *t* is stamped at *t + 1 minute* by the loader.
# Its baseline uses only the preceding 120 observations: both rolling moments
# are shifted before the current z-score is calculated. The first absolute
# two-sigma event opens one position; a fixed two-hour horizon closes it.
#
# This is an execution tutorial, not a mined trading rule. One market and one
# day are far too little evidence for model selection.

# %%
features = sample.features.to_pandas()
depth = features["depth_imbalance"]
features["history_mean"] = depth.rolling(120, min_periods=60).mean().shift(1)
features["history_std"] = depth.rolling(120, min_periods=60).std().shift(1)
features["depth_z"] = (
    (depth - features["history_mean"]) / features["history_std"].replace(0, np.nan)
)

book_times = sample.book_deltas.column("ts_init")
book_start = pd.Timestamp(book_times[0].as_py())
book_end = pd.Timestamp(book_times[-1].as_py())
eligible = features[
    features["ts_init"].between(
        book_start, book_end - pd.Timedelta(hours=2), inclusive="both"
    )
    & features["depth_z"].abs().ge(2.0)
]
assert not eligible.empty
entry = eligible.iloc[0]
exit_time = entry["ts_init"] + pd.Timedelta(hours=2)
direction = "buy" if entry["depth_z"] > 0 else "sell"
close_direction = "sell" if direction == "buy" else "buy"
print(
    f"{direction=} at {entry['ts_init']} z={entry['depth_z']:.2f}; "
    f"close at {exit_time}"
)

# %% [markdown]
# Signals are timestamped intent. They contain no execution price. That
# prevents the common mistake of filling at the same bar close that generated
# the signal.

# %%
quantity = 20.0
signals = backtest.signal_table(
    [
        {
            "ts": entry["ts_init"].to_pydatetime(),
            "instrument_id": sample.market_id,
            "side": direction,
            "quantity": quantity,
            "tag": "depth-z-entry",
        },
        {
            "ts": exit_time.to_pydatetime(),
            "instrument_id": sample.market_id,
            "side": close_direction,
            "quantity": quantity,
            "tag": "time-exit",
        },
    ]
)
signals.to_pandas()

# %% [markdown]
# ## Pin data before running

# %%
db = h5i_db.Database(cu.fresh_db("05_kaggle_polymarket_replay"), create=True)
for name, table in {
    "instruments": sample.instruments,
    "book_deltas": sample.book_deltas,
    "trades": sample.trades,
    "features_1m": sample.features,
}.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="bounded CC BY-NC Kaggle Polymarket sample")
db.snapshot(
    "approved-kaggle-cut",
    tables=["instruments", "book_deltas", "trades", "features_1m"],
    note="Real snapshot replay inputs; resolution label excluded",
)
backtest.create_signal_table(db)
db.append("signals", signals, note="causal depth-imbalance experiment")

# %% [markdown]
# ## Treat execution assumptions as model risk
#
# The snapshot feed cannot support exact queue depletion, so this experiment
# uses market orders and does not enable queue-position fills. We vary the
# curved Polymarket fee rate, latency, and adverse slippage. The baseline is
# intentionally not presented alone.

# %%
scenarios = {
    "frictionless": {
        "fee_rate": 0.0,
        "latency_nanos": 0,
        "slippage_ticks": 0,
    },
    "base": {
        "fee_rate": 0.02,
        "latency_nanos": 50_000_000,
        "slippage_ticks": 0,
    },
    "stressed": {
        "fee_rate": 0.035,
        "latency_nanos": 250_000_000,
        "slippage_ticks": 2,
    },
}
reports = {}
inspections = {}
for name, assumptions in scenarios.items():
    config = backtest.BacktestConfig(
        run_id=f"kaggle-{name}",
        portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
        data=backtest.DataConfig(
            signals="signals",
            snapshot="approved-kaggle-cut",
            minimum_coverage=0.75,
        ),
        execution=backtest.ExecutionConfig(
            fee_kind="prediction_market",
            fee_rate=assumptions["fee_rate"],
            latency_nanos=assumptions["latency_nanos"],
            slippage_ticks=assumptions["slippage_ticks"],
        ),
        risk=backtest.RiskConfig(
            max_order_quantity=quantity,
            max_abs_position=quantity,
            max_open_orders=1,
        ),
        output=backtest.OutputConfig(
            equity_interval_nanos=60_000_000_000,
        ),
        metadata={
            "dataset": cu.KAGGLE_POLYMARKET_DATASET,
            "scenario": name,
            "signal": "causal depth imbalance",
        },
    )
    inspections[name] = backtest.inspect(db, config)
    inspections[name].raise_for_errors()
    reports[name] = backtest.execute(db, config)

summary = pd.DataFrame(reports).T[
    ["fills", "commissions", "realized_pnl", "final_cash", "digest", "fork"]
]
summary

# %% [markdown]
# Preflight classifies this source as snapshot L2 rather than tick-delta L2.
# That fidelity statement is part of the result and prevents the notebook from
# quietly making queue-position claims the source cannot support.

# %%
{
    name: {
        "fidelity": inspection.fidelity.value,
        "warnings": [issue.message for issue in inspection.warnings],
    }
    for name, inspection in inspections.items()
}

# %% [markdown]
# A valid comparison fully executes the same two tagged orders in every
# scenario. One order may generate multiple fills when it walks book levels;
# counting fills as trades would be a subtle analysis error. The digest changes
# because execution configuration is part of run identity.

# %%
assert summary["fills"].ge(2).all()
assert summary["digest"].nunique() == len(summary)
assert reports["base"].verify()["verified"]

fill_frames = []
for scenario, report in reports.items():
    run_db = db.fork(report["fork"])
    scenario_fills = run_db.read("bt_fills").to_pandas()
    filled_by_tag = scenario_fills.groupby("tag")["quantity"].sum()
    assert set(filled_by_tag.index) == {"depth-z-entry", "time-exit"}
    assert filled_by_tag.eq(quantity).all()
    scenario_fills["scenario"] = scenario
    fill_frames.append(scenario_fills)
    run_db.close()
fills = pd.concat(fill_frames, ignore_index=True)
fills[["scenario", "ts", "side", "price", "quantity", "commission", "tag"]]

# %% [markdown]
# Plot equity from the most conservative scenario. Output frequency is one
# minute even though the engine consumes every retained book event.

# %%
stressed_db = db.fork(reports["stressed"]["fork"])
equity = stressed_db.read("bt_equity").to_pandas()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(equity["ts"], equity["equity"], linewidth=1.6)
ax.set(
    title="Stressed execution scenario",
    xlabel="simulated time",
    ylabel="portfolio value",
)
fig.tight_layout()

# %% [markdown]
# ## Production checklist
#
# - Repeat across many markets and non-overlapping dates; this single trade is
#   a pipeline test, not statistical evidence.
# - Pin the external file hashes alongside the h5i-db snapshot.
# - Calibrate fee rates to the venue and market date.
# - Use tick deltas, not periodic snapshots, before enabling queue-position
#   claims.
# - Report rejected orders, coverage, turnover, and sensitivity—not only P&L.
# - Respect the source dataset's non-commercial license.

# %%
stressed_db.close()
db.close()
