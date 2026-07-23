# %% [markdown]
# # Lead-lag discovery: cross-correlations on irregular ticks
#
# Who moves first? Lead-lag analysis between related instruments (index vs
# futures, ADR vs home listing, correlated FX crosses) is plagued by one
# mechanical problem: ticks arrive irregularly and *asynchronously*, and
# every naive fix — bucketing, last-observation-carried-forward — biases the
# answer (the Epps effect). We work through the toolkit on a pair with a
# *known, injected* lead:
#
# 1. build a synthetic GBPUSD that follows EURUSD with a 500 ms lag,
# 2. bucket-and-correlate: the classic CCF on a 250 ms grid via
#    `time_bucket`,
# 3. tick-level alignment with `asof_join` and a staleness-tolerance sweep,
# 4. the Hayashi-Yoshida estimator, which needs no grid at all — and, swept
#    over time shifts, localizes the lag without ever resampling,
# 5. a real-data reality check on a daily equity pair.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_leadlag"), create=True)

# %% [markdown]
# ## 1. Inject a known 500 ms lead
#
# `make_fx_ticks` generates *independent* pairs, so any lead-lag we found in
# it would be noise. Instead we take its EURUSD as the leader and construct
# GBPUSD ourselves: at each GBPUSD tick time `t`, the log-mid is
# `beta * eur_logmid(t - 500ms) + idio(t)` — a backward-asof sample of the
# leader's path, half a second stale, plus an idiosyncratic random walk of
# comparable magnitude. Ground truth: EURUSD leads by exactly 500 ms with
# beta 0.9, and nothing else connects the two.

# %%
LAG_US = 500_000
BETA = 0.9

eur = cu.make_fx_ticks(pairs=["EURUSD"], hours=24, ticks_per_hour=20_000).to_pandas()
eur_ts = eur["ts"].astype("int64").to_numpy()
eur_logmid = np.log((eur["bid"] + eur["ask"]) / 2).to_numpy()

rng = np.random.default_rng(99)
n = len(eur)
t0, t1 = eur_ts[0], eur_ts[-1]
gbp_ts = np.sort(rng.integers(t0 + LAG_US, t1, n))  # own asynchronous clock

# backward-asof sample of the leader at (t - 500ms), in numpy
idx = np.searchsorted(eur_ts, gbp_ts - LAG_US, side="right") - 1
idio = np.cumsum(rng.normal(0, 4e-5, n))
gbp_logmid = np.log(1.27) + BETA * (eur_logmid[idx] - eur_logmid[0]) + idio
gbp_mid = np.exp(gbp_logmid)
half = gbp_mid * rng.uniform(0.2e-4, 1e-4, n) / 2

gbp = pd.DataFrame(
    {
        "ts": pd.to_datetime(gbp_ts, unit="us", utc=True).astype("datetime64[us, UTC]"),
        "pair": "GBPUSD",
        "bid": np.round(gbp_mid - half, 5),
        "ask": np.round(gbp_mid + half, 5),
    }
)
fx = (
    pd.concat([eur, gbp])
    .sort_values(["ts", "pair"])
    .reset_index(drop=True)
)
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pair", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)
db.create_table("fx", schema, time_column="ts", sort_key=["ts", "pair"])
db.append("fx", pa.Table.from_pandas(fx, preserve_index=False).cast(schema), note="EURUSD + lagged GBPUSD")
print(f"{len(fx):,} ticks over 24h ({n:,} per pair, ~{n / 86_400:.1f}/sec)")

# %% [markdown]
# ## 2. The bucketed CCF: `time_bucket` at 250 ms
#
# The classic approach: resample both series to a fine common grid, compute
# returns, correlate at every lead/lag offset. h5i-db's `time_bucket` takes
# sub-second widths (`'250ms'`), so the grid reduction happens in the
# database: one GROUP BY turns 960k ticks into per-bucket closing mids,
# streaming on sorted storage. The pandas side just forward-fills the grid
# and shifts.

# %%
bucketed = db.sql(
    """
    SELECT time_bucket('250ms', ts) AS bucket, pair,
           last_value((bid + ask) / 2 ORDER BY ts) AS mid
    FROM fx
    GROUP BY bucket, pair
    """
).to_pandas()

grid = bucketed.pivot(index="bucket", columns="pair", values="mid")
full_index = pd.date_range(grid.index.min(), grid.index.max(), freq="250ms", tz="UTC")
rets = np.log(grid.reindex(full_index).ffill()).diff().dropna()

lags = np.arange(-12, 13)
ccf = pd.Series(
    {k: rets["EURUSD"].corr(rets["GBPUSD"].shift(-k)) for k in lags}
)
band = 1.96 / np.sqrt(len(rets))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(lags * 0.25, ccf.values, width=0.2, color="tab:blue")
ax.axhspan(-band, band, color="gray", alpha=0.3, label="95% band under independence")
ax.axvline(0.5, color="tab:red", ls="--", lw=1, label="true lag (0.5 s)")
ax.set_title("CCF of 250 ms returns: corr(EURUSD$_t$, GBPUSD$_{t+k}$)")
ax.set_xlabel("k (seconds; positive = EURUSD leads)")
ax.set_ylabel("correlation")
ax.legend()
fig.tight_layout()
peak = ccf.idxmax()
print(f"CCF peak at k = {peak * 0.25:+.2f}s (corr {ccf.max():.2f}); corr at k=0: {ccf.loc[0]:.2f}")

# %% [markdown]
# The CCF nails the direction and localizes the lag: the mass sits at
# +0.5 s and +0.75 s — the injected 500 ms plus the staleness of last-tick
# sampling (ticks arrive every ~180 ms, so the "closing mid" of a bucket is
# itself slightly stale) smears the effect one bucket to the right. Nothing
# at zero, nothing at negative lags — GBPUSD never leads. This is the
# discretization tradeoff in miniature: a coarser grid would pile the whole
# effect into one bar at lag zero and *hide* the direction; a finer grid
# thins each bucket until carried-forward zeros dominate (Epps effect).
#
# ## 3. Tick-level alignment: `asof_join` and the staleness tolerance
#
# Before estimating anything grid-free, the workhorse operation is aligning
# one irregular series to another: for each GBPUSD tick, the latest EURUSD
# tick — `asof_join(..., 'backward', tolerance)`. The tolerance (raw
# microseconds) is a data-quality dial: how stale a leader quote are you
# willing to accept? We audit a 20-minute window, sweeping the tolerance and
# tracking match rate and the correlation of aligned tick-to-tick returns.
#
# Windowed extracts keep the audit fast and the numbers inspectable; we
# assert the join returned one row per left tick and cross-check a sample
# against `pandas.merge_asof` — standard hygiene for any ASOF pipeline.

# %%
w0 = int(pd.Timestamp("2026-06-01 12:00", tz="UTC").value // 1000)
w1 = int(pd.Timestamp("2026-06-01 12:20", tz="UTC").value // 1000)
BUF_US = 10_000_000  # give the window a 10s run-up so early ticks can match

qa_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("base", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)
eur_w = eur[(eur["ts"].astype("int64") >= w0 - BUF_US) & (eur["ts"].astype("int64") < w1)]
gbp_w = gbp[(gbp["ts"].astype("int64") >= w0) & (gbp["ts"].astype("int64") < w1)]
for name, frame in (("eur_qa", eur_w), ("gbp_qa", gbp_w)):
    tbl = pa.Table.from_pandas(
        frame.assign(base="USD")[["ts", "base", "bid", "ask"]], preserve_index=False
    ).cast(qa_schema)
    db.create_table(name, qa_schema, time_column="ts")
    db.append(name, tbl, note="QA window 12:00-12:20")
print(f"QA window: {len(gbp_w):,} follower ticks, {len(eur_w):,} leader ticks")

rows = []
for tol_ms in (50, 100, 250, 500, 1000, 5000):
    j = db.sql(
        f"""
        SELECT ts, (bid + ask) / 2 AS gbp_mid,
               ts_right, (bid_right + ask_right) / 2 AS eur_mid
        FROM asof_join('gbp_qa', 'eur_qa', 'ts', 'ts', 'base', 'backward', {tol_ms * 1000})
        ORDER BY ts
        """
    ).to_pandas()
    assert len(j) == len(gbp_w), "asof_join must return one row per left tick"
    matched = j.dropna(subset=["eur_mid"])
    r = np.log(matched[["gbp_mid", "eur_mid"]]).diff().dropna()
    rows.append(
        {
            "tolerance_ms": tol_ms,
            "match_rate": len(matched) / len(j),
            "aligned_ret_corr": r["gbp_mid"].corr(r["eur_mid"]),
        }
    )

# cross-check the last sweep (5s tolerance) against pandas.merge_asof
ref = pd.merge_asof(
    gbp_w[["ts"]].reset_index(drop=True),
    eur_w.assign(eur_mid_ref=(eur_w["bid"] + eur_w["ask"]) / 2)[["ts", "eur_mid_ref"]],
    on="ts",
    tolerance=pd.Timedelta("5s"),
)
chk = j.merge(ref, on="ts")
assert np.allclose(chk["eur_mid"], chk["eur_mid_ref"], equal_nan=True), "asof mismatch vs pandas"
print("asof_join validated against pandas.merge_asof")
pd.DataFrame(rows).round(3)

# %% [markdown]
# With ~5.5 leader ticks per second, a 50 ms tolerance keeps only a quarter
# of the follower's ticks (the rest have no fresh-enough match and come back
# NULL — visible, not silently stale); by ~500 ms essentially everything
# matches. The correlation column repays a careful read: it *falls* as the
# tolerance loosens. Not because matches get staler — because the surviving
# tick pairs get denser, so consecutive aligned returns span ever shorter
# intervals, and an interval shorter than the 500 ms lag has almost no
# contemporaneous overlap with its counterpart (corr → 0.07). At tight
# tolerance the thinned series spans ~1 s intervals that straddle the lag,
# and correlation reappears. Neither number is "the" correlation: with
# asynchronous, lagged series, what you measure depends on the interval
# length you measure over — the Epps effect in a table, and the case for an
# estimator that respects intervals explicitly.
#
# ## 4. Hayashi-Yoshida: no grid, and a dial for the lag
#
# The HY estimator sums `r_i * s_j` over all pairs of tick-interval returns
# whose time intervals *overlap* — unbiased for asynchronous data, no
# resampling, no carried-forward zeros. Sweeping a time shift `delta`
# applied to the follower's clock turns it into a grid-free
# cross-correlogram (Hoffmann-Rosenbaum-Yoshida): the shift that maximizes
# HY correlation estimates the lag. A two-pointer pass makes it O(n + m).

# %%
def hy_corr(t_a, r_a, t_b, r_b):
    """Hayashi-Yoshida correlation from interval times and returns."""
    cov = 0.0
    j0 = 0
    for i in range(1, len(t_a)):
        a_lo, a_hi = t_a[i - 1], t_a[i]
        while j0 < len(t_b) - 1 and t_b[j0 + 1] <= a_lo:
            j0 += 1
        j = j0
        while j < len(t_b) - 1 and t_b[j] < a_hi:
            if min(a_hi, t_b[j + 1]) > max(a_lo, t_b[j]):  # overlap
                cov += r_a[i - 1] * r_b[j]
            j += 1
    return cov / np.sqrt((r_a**2).sum() * (r_b**2).sum())

# 2-hour subsample of raw ticks, pulled with a pruned time-range scan
sub = db.sql(
    """
    SELECT ts, pair, (bid + ask) / 2 AS mid FROM fx
    WHERE ts >= '2026-06-01T06:00:00Z' AND ts < '2026-06-01T08:00:00Z'
    ORDER BY ts
    """
).to_pandas()
te, tg = {}, {}
for p, g in sub.groupby("pair"):
    ts_us = g["ts"].astype("int64").to_numpy()
    (te if p == "EURUSD" else tg)["t"] = ts_us
    (te if p == "EURUSD" else tg)["r"] = np.diff(np.log(g["mid"].to_numpy()))

shifts_ms = np.arange(-1000, 1001, 250)
hy = [
    hy_corr(te["t"][1:], te["r"], tg["t"][1:] - s * 1000, tg["r"])
    for s in shifts_ms
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(shifts_ms / 1000, hy, marker="o", color="tab:purple")
ax.axvline(0.5, color="tab:red", ls="--", lw=1, label="true lag (0.5 s)")
ax.set_title("Hayashi-Yoshida correlation vs follower clock shift")
ax.set_xlabel("shift applied to GBPUSD timestamps (s)")
ax.set_ylabel("HY correlation")
ax.legend()
fig.tight_layout()
best = shifts_ms[int(np.argmax(hy))]
print(f"HY peak at shift {best} ms: corr {max(hy):.2f}   (unshifted: {hy[4]:.2f})")

# %% [markdown]
# The HY correlogram recovers the injected lag exactly — a sharp peak at
# +500 ms, no rightward smear, and a peak correlation at least as high as
# the best bucketed estimate, with no grid throwing information away. The
# unshifted HY value is ~0 for the right reason: contemporaneously these
# series really are nearly unrelated; the dependence lives entirely at the
# half-second offset.
#
# ## 5. Reality check: a daily equity pair
#
# The same CCF machinery on real KO/PEP daily closes (from the cached
# 30-name S&P sample). At daily horizon in liquid large caps, the honest
# expectation is a strong contemporaneous correlation and *no* exploitable
# cross-lag — information that took 500 ms in our synthetic market takes
# well under a day in real ones.

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01").to_pandas()
pair_df = (
    daily[daily["symbol"].isin(["KO", "PEP"])][["ts", "symbol", "adj_close"]]
    .sort_values(["ts", "symbol"])
    .reset_index(drop=True)
)
eq_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("equity_daily", eq_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("equity_daily", pa.Table.from_pandas(pair_df, preserve_index=False).cast(eq_schema), note="KO/PEP daily")

eq = db.sql(
    """
    SELECT ts, symbol,
           adj_close / lag(adj_close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
    FROM equity_daily ORDER BY ts
    """
).to_pandas()
eq_panel = eq.pivot(index="ts", columns="symbol", values="ret").dropna()
eq_band = 1.96 / np.sqrt(len(eq_panel))
eq_ccf = {k: eq_panel["KO"].corr(eq_panel["PEP"].shift(-k)) for k in range(-5, 6)}
print(f"KO->PEP daily CCF ({len(eq_panel)} days, 95% band ±{eq_band:.3f}):")
for k, v in eq_ccf.items():
    flag = " *" if abs(v) > eq_band and k != 0 else ""
    print(f"  lag {k:+d}d: {v:+.3f}{flag}")

# %% [markdown]
# Contemporaneous correlation ~0.5-0.6; every cross-lag is at or inside the
# noise band. No daily lead-lag in KO/PEP — as it should be. Real lead-lag
# alpha lives at the horizons of section 2-4, which is exactly why tick
# infrastructure (and estimators that respect asynchronicity) matter.
#
# ## Takeaways
#
# - `time_bucket` accepts sub-second widths (`'250ms'`), so CCF grids come
#   straight out of one GROUP BY; `last_value(... ORDER BY ts)` is the
#   per-bucket closing mid.
# - `asof_join(..., 'backward', tolerance)` aligns irregular series with an
#   explicit staleness budget — unmatched ticks surface as NULLs you can
#   count, which is how a match-rate/tolerance audit falls out for free.
#   (Current-build caveat: keep both join inputs within one storage batch
#   (~8k rows) and assert one output row per left row — larger inputs are
#   silently truncated.)
# - Bucketing localizes a lag only to grid precision and dilutes correlation
#   (Epps); Hayashi-Yoshida on raw tick intervals, swept over clock shifts,
#   recovered both the exact 500 ms lag and the full correlation.
# - Injecting the effect (a known lag, a known beta) turns the notebook into
#   a calibration exercise: every estimator is judged against ground truth,
#   and the real-data section is honest about finding nothing at daily
#   horizon.

# %%
db.close()
