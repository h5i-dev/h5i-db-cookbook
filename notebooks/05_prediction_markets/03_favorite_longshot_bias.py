# %% [markdown]
# # Trading the favorite-longshot bias
#
# The oldest documented anomaly in event contracts is that longshots are
# overpriced and favorites underpriced. Recipe 05/02 measured it as a calibration
# gap; this one turns it into a position and asks whether it survives the fee
# curve. Bucketed hold-to-resolution returns come first, then two engine runs that
# take opposite sides of the same bias, then the part that decides it: at which
# price levels the edge is larger than the cost of trading there.

# %%
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest

db = h5i_db.Database(cu.fresh_db("05_favorite_longshot_bias"), create=True)
FEE_RATE = 0.07
QUANTITY = 20.0

# %% [markdown]
# ## The panel
#
# One row of `book_deltas` is one price level of one atomic book event. This
# recipe needs only the YES ask, which is what a buyer pays, and the resolution.
#
# | column | type | meaning |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | recorder arrival; replay order |
# | `instrument_id` | `string` | the market |
# | `outcome` | `uint16` | 0 = YES, 1 = NO |
# | `side` | `string` | `buy` is the bid, `sell` the ask |
# | `price` | `float64` | probability on a 0.001 grid |
# | `size` | `float64` | contracts at that level |

# %%
panel = cu.make_prediction_markets(n_markets=180, steps=32, seed=11)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel load")
db.snapshot("panel-v1", tables=list(panel), note="favorite-longshot study")
print(f"{panel['book_deltas'].num_rows:,} book rows, {panel['resolutions'].num_rows} markets")
panel["book_deltas"].to_pandas().head()

# %% [markdown]
# ## Hold-to-resolution returns, by price
#
# Fix a decision instant, take the YES ask there, and hold to resolution. The
# payoff is 1 if YES won and 0 otherwise, so the return on capital is
# `(payoff - ask) / ask`. Bucketing by the ask is the classic presentation: each
# bucket is a portfolio of contracts that all looked equally likely.

# %%
stamps = sorted({value.as_py() for value in panel["book_deltas"].column("ts_init")})
decision = stamps[8]
decision_ns = int(pd.Timestamp(decision, tz="UTC").value)
quotes = db.sql(
    f"""
    SELECT instrument_id, outcome,
           max(CASE WHEN side = 'buy'  THEN price END) AS bid,
           max(CASE WHEN side = 'sell' THEN price END) AS ask
    FROM h5i('book_deltas', 'panel-v1')
    WHERE ts_init = to_timestamp_nanos({decision_ns})
    GROUP BY instrument_id, outcome
    """
).to_pandas()
truth = cu.market_truth(panel).to_pandas()

yes = quotes[quotes.outcome == 0].merge(truth, on="instrument_id", validate="one_to_one")
yes["payoff"] = yes.yes_won.astype(float)
yes["gross_return"] = (yes.payoff - yes.ask) / yes.ask
yes["fee_per_contract"] = FEE_RATE * yes.ask * (1.0 - yes.ask)
yes["net_return"] = (yes.payoff - yes.ask - yes.fee_per_contract) / yes.ask
print(f"decision instant {decision}, {len(yes)} markets")

EDGES = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0]
yes["bucket"] = pd.cut(yes.ask, EDGES)
buckets = (
    yes.groupby("bucket", observed=True)
    .agg(
        n=("ask", "size"),
        mean_ask=("ask", "mean"),
        hit_rate=("payoff", "mean"),
        gross=("gross_return", "mean"),
        net=("net_return", "mean"),
    )
    .assign(edge_pp=lambda f: (f.hit_rate - f.mean_ask) * 100)
)
print(buckets.round(3).to_string())

# %% [markdown]
# Buying the cheap buckets loses money and buying the expensive ones makes it,
# and the sign flips somewhere in the middle. The `edge_pp` column is the same
# calibration gap from recipe 05/02, expressed per contract; `gross` divides it by
# the price, which is why a small gap on a 15-cent contract is a large percentage
# loss.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
centres = buckets.mean_ask
width = 0.05
ax.bar(centres - width / 2, buckets.gross * 100, width=width, label="gross", color="#2c7fb8")
ax.bar(centres + width / 2, buckets.net * 100, width=width, label="net of fees", color="#c0392b")
ax.axhline(0.0, color="black", lw=0.8)
ax.set_title("Hold-to-resolution return on a long YES position, by price")
ax.set_xlabel("YES ask at the decision instant")
ax.set_ylabel("return on capital, %")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## The cost of trading there
#
# The fee is `rate * q * p * (1-p)`, so it is cheapest exactly where the
# favorite edge lives. As a *fraction of capital* it is `rate * (1-p)`, which
# falls monotonically as the contract gets more expensive. Both effects point the
# same way: the favorite side of this bias is the affordable side.

# %%
levels = pd.DataFrame({"price": [0.10, 0.25, 0.50, 0.75, 0.90]})
levels["fee_per_contract"] = FEE_RATE * levels.price * (1 - levels.price)
levels["fee_pct_of_capital"] = levels.fee_per_contract / levels.price * 100
print(levels.round(4).to_string(index=False))

# %% [markdown]
# ## Take both sides in the engine
#
# Two runs over the same pin. One buys the favorites, one buys the longshots, and
# nothing else differs. Signals are stamped one microsecond after the decision
# quote, so each order transacts at the ask it was chosen from.

# %%
FAVORITE, LONGSHOT = 0.75, 0.30
submit = decision + dt.timedelta(microseconds=1)


def signals_for(members: pd.DataFrame, tag: str) -> pa.Table:
    rows = [
        {
            "ts": submit,
            "instrument_id": row.instrument_id,
            "outcome": 0,
            "side": "buy",
            "quantity": QUANTITY,
            "tag": tag,
        }
        for row in members.itertuples()
    ]
    return backtest.signal_table(rows).sort_by([("ts", "ascending")])


legs = {
    "favorites": yes[yes.ask >= FAVORITE],
    "longshots": yes[yes.ask <= LONGSHOT],
}
for name, members in legs.items():
    table = signals_for(members, name)
    db.create_table(f"signals_{name}", table.schema, time_column="ts")
    db.append(f"signals_{name}", table)
    print(f"{name:10} {len(members):>3} markets, mean ask {members.ask.mean():.3f}")

# %%
def run(name: str, fee_kind: str | None) -> backtest.BacktestResult:
    execution = (
        backtest.ExecutionConfig(fee_kind=fee_kind, fee_rate=FEE_RATE)
        if fee_kind
        else backtest.ExecutionConfig()
    )
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=f"flb-{name}-{fee_kind or 'gross'}",
            data=backtest.DataConfig(signals=f"signals_{name}", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=execution,
            metadata={"study": "favorite-longshot", "leg": name},
        ),
    )


def account(result: backtest.BacktestResult) -> dict[str, float]:
    """Positions are held to resolution, so the result is in settlement_pnl."""
    positions = result.positions.to_pandas()
    fills = result.fills.to_pandas()
    settled = float(positions.settlement_pnl.fillna(0.0).sum())
    fees = float(result.summary()["commissions"])
    capital = float((fills.price * fills.quantity).sum())
    return {
        "fills": len(fills),
        "capital": capital,
        "settled": settled,
        "fees": fees,
        "net": settled - fees,
        "net_return_pct": (settled - fees) / capital * 100 if capital else float("nan"),
    }


report = {}
for name in legs:
    for fee_kind in (None, "kalshi"):
        label = f"{name} / {'gross' if fee_kind is None else 'net'}"
        report[label] = account(run(name, fee_kind))
print(pd.DataFrame(report).T.round(2).to_string())

# %% [markdown]
# The favorite leg is positive gross and stays positive after fees. The longshot
# leg is negative gross and gets worse. The asymmetry in how much fees take is the
# point of the table above: the longshot leg pays a larger fee *relative to the
# capital it commits*, so it is the more expensive way to be wrong.

# %% [markdown]
# ## Where the edge actually is
#
# One threshold is a choice, and choices are where backtests go wrong. Sweeping it
# shows whether the finding is a knife-edge or a plateau. Everything here is one
# decision instant on one panel, so treat the shape as the result and the level as
# noise; recipe 05/05 puts a number on that distinction.

# %%
sweep = []
for threshold in np.arange(0.40, 0.92, 0.04):
    members = yes[yes.ask >= threshold]
    if len(members) < 5:
        continue
    sweep.append(
        {
            "threshold": round(float(threshold), 2),
            "markets": len(members),
            "mean_ask": members.ask.mean(),
            "gross_pct": members.gross_return.mean() * 100,
            "net_pct": members.net_return.mean() * 100,
        }
    )
sweep = pd.DataFrame(sweep)
print(sweep.round(2).to_string(index=False))

# %% [markdown]
# ## Volatility scales with p(1-p)
#
# A practical consequence of the same arithmetic. A contract at 0.05 cannot move
# as far as one at 0.50, because it is bounded below at zero. Position sizing that
# ignores this over-risks the middle of the book and under-risks the tails.
#
# The scan stops at `expiration_ns`. The panel keeps quoting after the result is
# observable, and those snapshots sit at ~1.00 or ~0.00, so including them would
# score the resolution jump as a price move. It is the single largest "move" in
# the data and it belongs to no volatility estimate.

# %%
expiry_ns = int(db.sql("SELECT max(expiration_ns) AS e FROM instruments").to_pandas()["e"][0])
paths = db.sql(
    f"""
    SELECT instrument_id, ts_init,
           (max(CASE WHEN side = 'buy'  THEN price END)
          + max(CASE WHEN side = 'sell' THEN price END)) / 2 AS mid
    FROM h5i('book_deltas', 'panel-v1')
    WHERE outcome = 0 AND ts_init <= to_timestamp_nanos({expiry_ns})
    GROUP BY instrument_id, ts_init
    ORDER BY instrument_id, ts_init
    """
).to_pandas()
paths["step_move"] = paths.groupby("instrument_id").mid.diff()
paths["level"] = paths.groupby("instrument_id").mid.transform("mean").round(1).clip(0.1, 0.9)
vol = (
    paths.dropna(subset=["step_move"])
    .groupby("level")
    .agg(observations=("step_move", "size"), step_vol_bp=("step_move", lambda s: s.std() * 10_000))
    .assign(sqrt_p1mp=lambda f: np.sqrt(f.index * (1 - f.index)))
)
vol["vol_over_sqrt"] = vol.step_vol_bp / vol.sqrt_p1mp
print(vol.round(1).to_string())
print("\nthe last column is roughly flat, which is what p*(1-p) scaling looks like")

# %% [markdown]
# ## Takeaways
#
# - The favorite-longshot bias shows up as a signed calibration gap and survives
#   translation into a position: long favorites made money net of fees on this
#   panel, long longshots lost it.
# - Divide by the price. A three-point gap is a 20% return on a 15-cent contract
#   and a 3% return on an 85-cent one, and the fee as a share of capital is
#   `rate * (1-p)`, which moves the opposite way.
# - Sweep the threshold before believing it. A plateau is a finding; a spike is a
#   fitted parameter.
# - Price volatility scales with `sqrt(p*(1-p))`. Size positions on that, not on a
#   constant.
# - h5i-db features doing the work: one named snapshot behind four runs, so the
#   gross/net and favorite/longshot comparisons differ only in the field being
#   studied, and `bt_fills` supplied the capital actually committed rather than
#   the capital intended.

# %%
db.close()
