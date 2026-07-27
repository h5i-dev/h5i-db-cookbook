# %% [markdown]
# # Event studies: CARs with ASOF-aligned announcement dates
#
# The classic event-study pipeline computes market-model abnormal returns and
# cumulative abnormal returns around announcements. One step is perennially
# fiddly: announcements do not land on trading days.
#
# Earnings drop on Saturday, M&A leaks on a holiday, and every study needs "the
# first trading session at or after the announcement". That is exactly an ASOF
# join with `'forward'` direction, which h5i-db does in one SQL call, per
# symbol, with a staleness tolerance.
#
# In this recipe we:
#
# 1. build a 100-name daily panel and inject a *known* announcement shock into a
#    treated half of 100 events, so the study has a ground truth to recover,
# 2. store `prices`, `events` and the trading calendar as tables,
# 3. align events to sessions with `asof_join(..., 'forward', tolerance)`,
# 4. estimate market-model betas and compute CAR[-10,+10], treated against
#    control, with confidence bands.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_events"), create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.make_daily_prices` gives a factor-driven panel, a common market factor
# plus idiosyncratic noise, with no events in it. One row per symbol per
# session.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | session close, 20:00 UTC |
# | `symbol` | `string` | ticker, `STK000` … `STK099` |
# | `open`, `high`, `low`, `close` | `float64` | session prices |
# | `volume` | `int64` | shares traded |

# %%
N_SYMBOLS, N_DAYS, N_EVENTS = 100, 650, 100
DAY0_SHOCK, DRIFT_PER_DAY, DRIFT_DAYS = 0.02, 0.0015, 10

symbols = [f"STK{i:03d}" for i in range(N_SYMBOLS)]
prices = cu.make_daily_prices(symbols=symbols, days=N_DAYS).to_pandas()
print(f"{len(prices):,} rows x {prices.shape[1]} columns, {N_SYMBOLS} symbols")
prices.head()

# %% [markdown]
# We add the events ourselves, in pandas, *before* storing. There are 100
# events, one per symbol, at random calendar timestamps, deliberately including
# weekends.
#
# For the treated half we scale the price path from the effective session
# onward: a +2% level shift on day 0 and a +0.15% per day drift over the next 10
# sessions, a caricature of post-announcement drift.
#
# Controls get nothing, so their CAR should be flat at zero. That is a built-in
# placebo check.

# %%
sess_us = np.sort(prices["ts"].astype("int64").unique())  # session closes, epoch us
rng = np.random.default_rng(7)

# One event per symbol: pick the *effective* session j, then draw the
# announcement uniformly between the previous close and that close - Monday
# sessions naturally pick up weekend announcements.
event_sym = rng.permutation(symbols)[:N_EVENTS]
event_j = rng.integers(150, len(sess_us) - 30, N_EVENTS)
ann_us = sess_us[event_j - 1] + (rng.random(N_EVENTS) * (sess_us[event_j] - sess_us[event_j - 1])).astype("int64")
treated = rng.random(N_EVENTS) < 0.5

ts_us_all = prices["ts"].astype("int64").to_numpy()
for sym, j, is_t in zip(event_sym, event_j, treated):
    if not is_t:
        continue
    k = np.arange(len(sess_us))
    factor = np.where(k < j, 1.0, (1 + DAY0_SHOCK) * (1 + DRIFT_PER_DAY) ** np.clip(k - j, 0, DRIFT_DAYS))
    mask = prices["symbol"].to_numpy() == sym
    idx = np.searchsorted(sess_us, ts_us_all[mask])
    for field in ("open", "high", "low", "close"):
        prices.loc[mask, field] = prices.loc[mask, field].to_numpy() * factor[idx]

print(f"{treated.sum()} treated / {(~treated).sum()} control events")
print("weekend announcements:", (pd.to_datetime(ann_us, unit="us", utc=True).dayofweek >= 5).sum())

# %% [markdown]
# ## 2. Store `prices`, `events`, and the trading calendar itself
#
# All three are h5i-db tables with `ts` as the time column.
#
# The events table carries the *raw announcement timestamp*, not a trading day.
# Resolving that mapping is the database's job, not the ingest script's.
#
# The third table, `sessions`, is the trading calendar: one row per session
# close, derived from the panel we actually have, so holidays are whatever the
# data says they are. Materializing the calendar as data is what lets the
# alignment be a join instead of `BusinessDay` arithmetic.

# %%
price_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
db.create_table("prices", price_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    pa.Table.from_pandas(prices.sort_values(["ts", "symbol"]), preserve_index=False).cast(price_schema),
    note="synthetic panel with injected event shocks",
)

events = (
    pd.DataFrame(
        {
            "ts": pd.to_datetime(ann_us, unit="us", utc=True),
            "symbol": event_sym,
            "cal": "XNYS",  # which trading calendar governs this event
            "event_id": np.arange(N_EVENTS),
            "treated": treated,
        }
    )
    .sort_values("ts")
    .reset_index(drop=True)
)
event_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("cal", pa.string()),
        pa.field("event_id", pa.int64()),
        pa.field("treated", pa.bool_()),
    ]
)
db.create_table("events", event_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("events", pa.Table.from_pandas(events, preserve_index=False).cast(event_schema), note="100 announcements")

session_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("cal", pa.string()),
        pa.field("session_id", pa.int64()),
    ]
)
db.create_table("sessions", session_schema, time_column="ts")
db.append(
    "sessions",
    pa.table(
        {
            "ts": pa.array(sess_us, pa.timestamp("us", tz="UTC")),
            "cal": pa.array(["XNYS"] * len(sess_us)),
            "session_id": pa.array(np.arange(len(sess_us))),
        }
    ),
    note="trading calendar derived from the panel",
)
db.tables()

# %% [markdown]
# ## 3. Align announcements to trading sessions with a forward ASOF join
#
# `asof_join(left, right, lts, rts, key, 'forward', tolerance)` matches each
# event to the *first* session at or after the announcement. The key here is the
# calendar id: one calendar in this study, though a global book would carry
# several.
#
# The tolerance is in raw time units, microseconds, matching the
# `timestamp[us, UTC]` column. Seven days is a generous cap that surfaces
# calendar problems as NULLs instead of silently matching weeks later.
#
# The right table's colliding `ts` comes back as `ts_right`, putting
# announcement and effective session side by side. And because `asof_join(...)`
# is just a relation, we can equi-join its output straight back to `prices` for
# the day-0 close, all in one statement.
#
# One operational note. Joining the 100-row events table against the 650-row
# calendar, rather than the 65k-row price panel, keeps the join tiny. Derive
# compact, purpose-built tables and join those.
#
# We assert one output row per event and cross-check the session mapping against
# a numpy `searchsorted` on the calendar. Cheap asserts like these turn
# alignment mistakes into loud failures instead of quietly shifted event
# windows.

# %%
# Forward ASOF: each announcement snaps to the next session actually present
# in the calendar - then join that session's close off the price panel.
next_session = (
    db.table("events")
    .join_asof(
        db.table("sessions"), on="ts", by="cal",
        direction="forward", tolerance=7 * 86_400 * 1_000_000,
    )
    .select("event_id", "symbol", "treated", announced=col("ts"),
            effective_session=col("ts_right"))
)

aligned = (
    next_session.join(
        db.table("prices"),
        left_on=["effective_session", "symbol"], right_on=["ts", "symbol"],
    )
    .select(
        "event_id", "treated", "announced", "effective_session",
        symbol=col("symbol", relation="l"),
        day0_close=col("close", relation="r"),
    )
    .sort("event_id")
    .to_pandas()
)
assert len(aligned) == N_EVENTS, "one output row per event"
expected_us = sess_us[np.searchsorted(sess_us, ann_us)]  # independent check
got_us = aligned.sort_values("event_id")["effective_session"].astype("int64").to_numpy()
assert (got_us == expected_us).all(), "ASOF session mapping mismatch"
print("unmatched events:", aligned["effective_session"].isna().sum())
weekend = aligned[pd.to_datetime(aligned["announced"]).dt.dayofweek >= 5]
weekend.assign(
    ann_dow=pd.to_datetime(weekend["announced"]).dt.day_name(),
    eff_dow=pd.to_datetime(weekend["effective_session"]).dt.day_name(),
).head(6)

# %% [markdown]
# Every weekend announcement lands on the following Monday's session. No
# calendar arithmetic, no `BusinessDay` offsets, and it would work unchanged for
# holidays, because the join targets the sessions *actually present* in the
# price table.
#
# ## 4. Returns and the market factor
#
# Simple returns per symbol via `lag()`, and the equal-weight market return as a
# window average over each session. One statement, computed on sorted storage.

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")

rets = (
    db.table("prices")
    .with_columns(ret=col("close") / PREV_CLOSE - 1)
    .with_columns(mkt=col("ret").mean().over(partition_by="ts"))
    .filter(col("ret").is_not_null())
    .select("ts", "symbol", "ret", "mkt")
    .sort(["ts", "symbol"])
    .to_pandas()
)
rets.head(3)

# %% [markdown]
# ## 5. Market model and CAR[-10,+10]
#
# For each event we estimate alpha and beta on trading days [-130, -11] relative
# to the effective session, then cumulate abnormal returns
# `AR = r - (a + b*mkt)` over [-10, +10].
#
# Cross-sectional averaging uses a normal-approximation band,
# `1.96 * sd / sqrt(n)`.

# %%
panel = rets.pivot(index="ts", columns="symbol", values="ret")
mkt = rets.groupby("ts")["mkt"].first().loc[panel.index]
pos = {ts.value: i for i, ts in enumerate(panel.index)}  # ns epoch -> row

EST, EVT = (-130, -11), (-10, 10)
rel_days = np.arange(EVT[0], EVT[1] + 1)
cars = {}
for _, ev in aligned.iterrows():
    j = pos[ev["effective_session"].value]
    est = slice(j + EST[0], j + EST[1] + 1)
    r_est, m_est = panel[ev["symbol"]].iloc[est], mkt.iloc[est]
    beta = np.cov(r_est, m_est)[0, 1] / np.var(m_est, ddof=1)
    alpha = r_est.mean() - beta * m_est.mean()
    win = slice(j + EVT[0], j + EVT[1] + 1)
    ar = panel[ev["symbol"]].iloc[win].to_numpy() - (alpha + beta * mkt.iloc[win].to_numpy())
    cars[ev["event_id"]] = np.cumsum(ar)

car = pd.DataFrame(cars, index=rel_days).T
car["treated"] = aligned.set_index("event_id").loc[car.index, "treated"]

summary = {}
for grp, g in car.groupby("treated"):
    m = g[rel_days].mean()
    ci = 1.96 * g[rel_days].std() / np.sqrt(len(g))
    summary["treated" if grp else "control"] = (m, ci, len(g))

def group_diff_t(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))

t_end, c_end = car[car["treated"]][rel_days[-1]], car[~car["treated"]][rel_days[-1]]
t_d0 = car[car["treated"]][0].sub(car[car["treated"]][-1])
c_d0 = car[~car["treated"]][0].sub(car[~car["treated"]][-1])
print(
    f"mean CAR[+10]  treated {t_end.mean():+.2%}   control {c_end.mean():+.2%}"
    f"   diff t-stat {group_diff_t(t_end, c_end):.1f}"
)
print(
    f"mean day-0 AR  treated {t_d0.mean():+.2%}   control {c_d0.mean():+.2%}"
    f"   diff t-stat {group_diff_t(t_d0, c_d0):.1f}"
)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5))
for name, color in (("treated", "tab:red"), ("control", "tab:blue")):
    m, ci, n = summary[name]
    ax.plot(rel_days, m * 100, color=color, label=f"{name} (n={n})")
    ax.fill_between(rel_days, (m - ci) * 100, (m + ci) * 100, color=color, alpha=0.15)
ax.axvline(0, color="k", lw=0.7, ls="--")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Average CAR around announcement (market model, 95% band)")
ax.set_xlabel("trading days relative to effective session")
ax.set_ylabel("CAR (%)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# The study recovers what we injected. The treated group jumps about +2% on day
# 0, a sharp and highly significant one-day difference, then drifts toward about
# +3.5% by day +10. The control CAR stays inside its band around zero, so the
# placebo behaves.
#
# Note how much weaker the CAR[+10] t-stat is than the day-0 one. Cumulating 21
# days of idiosyncratic noise on roughly 50 events erodes power quickly, which
# is exactly why real event studies live and die by sample size.
#
# Pre-event CARs are flat for both groups, confirming the alignment introduced
# no lookahead. Announcements only touch prices from the effective session
# onward.
#
# ## Takeaways
#
# - `asof_join(..., 'forward', tolerance)` against a materialized trading
#   calendar is the right primitive for "first session at or after the
#   announcement". It is holiday-agnostic, and past the tolerance it returns
#   NULLs rather than silent garbage. The tolerance is raw microseconds,
#   matching the `timestamp[us, UTC]` time column.
# - `asof_join(...)` composes like any relation. One statement chained the
#   forward alignment into an equi-join back to `prices` for day-0 closes.
# - Keeping `events` as a first-class h5i-db table, with the time column set to
#   raw announcement time, means the calendar mapping lives in the query. Re-run
#   against a revised price panel and it re-resolves automatically.
# - Returns and the equal-weight market factor came from one SQL window
#   statement. Only the per-event OLS loop lives in pandas.
# - Injecting a known effect into synthetic data, with a control group left
#   untouched, turns the recipe into a self-verifying test of the pipeline.

# %%
db.close()
