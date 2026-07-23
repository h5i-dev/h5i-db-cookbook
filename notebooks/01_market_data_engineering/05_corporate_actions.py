# %% [markdown]
# # Corporate actions: split adjustment without losing the tape
#
# A stock split rewrites history: every price before the effective date must be
# scaled, or every return computed across it is garbage. That forces an
# engineering choice - do you *restate* the stored series (and lose the raw
# tape), or keep the raw tape canonical and adjust at read time? With h5i-db
# you don't have to lose anything either way: a restatement is a `write()`
# commit with a note, the pre-restatement series stays readable forever via
# time travel, and the adjustment-factor pattern is a small SQL join. This
# recipe builds both patterns on real AAPL and NVDA data covering three splits:
# AAPL 4:1 (2020-08-31), NVDA 4:1 (2021-07-20) and NVDA 10:1 (2024-06-10).

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_corpactions"), create=True)

# %% [markdown]
# ## 1. Reconstruct the raw tape
#
# We pull real daily data (cached to parquet, so this is deterministic and
# offline). One wrinkle: Yahoo's `close` column is already split-adjusted -
# `adj_close` only adds dividend adjustments on top (the AAPL close/adj_close
# ratio of ~1.07 in 2018 is pure dividends). A real exchange feed would have
# printed AAPL at ~$499 on 2020-08-28, not ~$125. So we *un-adjust* using the
# known split schedule to recover the as-printed tape - exactly what a raw
# vendor feed delivers, and the honest starting point for this recipe.

# %%
real = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
px = (
    real.to_pandas()
    .query("symbol in ['AAPL', 'NVDA']")[["ts", "symbol", "close", "adj_close", "volume"]]
    .sort_values(["ts", "symbol"])
    .reset_index(drop=True)
)

SPLITS = pd.DataFrame(
    {
        "ts": pd.to_datetime(["2020-08-31", "2021-07-20", "2024-06-10"], utc=True),
        "symbol": ["AAPL", "NVDA", "NVDA"],
        "ratio": [4.0, 4.0, 10.0],  # shares multiply by ratio, price divides
    }
)

# Cumulative product of all splits *after* each row's date = the factor that
# was later divided out of the price. Multiply it back to get the raw print.
px["future_factor"] = 1.0
for s in SPLITS.itertuples():
    mask = (px["symbol"] == s.symbol) & (px["ts"] < s.ts)
    px.loc[mask, "future_factor"] *= s.ratio
px["close_raw"] = (px["close"] * px["future_factor"]).round(2)

px[(px["symbol"] == "AAPL") & px["ts"].between("2020-08-27", "2020-09-02")]

# %% [markdown]
# ## 2. Store the raw tape as the canonical table
#
# The raw series is what the exchange actually printed - it should be the
# immutable source of truth. Everything downstream (adjusted series, returns,
# factors) is derived and reproducible from it.

# %%
PRICE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
db.create_table("prices", PRICE_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])

raw_table = pa.Table.from_pandas(
    px[["ts", "symbol", "close_raw", "volume"]].rename(columns={"close_raw": "close"}),
    preserve_index=False,
).cast(PRICE_SCHEMA)
db.append("prices", raw_table, note="raw unadjusted tape, AAPL+NVDA 2018-2026")

# %% [markdown]
# ## 3. The classic screwup: naive returns across a split
#
# Compute daily returns straight off the raw closes with a `lag()` window and
# look at the worst days on record. Three of them are not crashes - they are
# splits masquerading as -75% and -90% "returns". Any risk model, momentum
# signal or stop-loss fed this series would have fired on a non-event.

# %%
worst = db.sql(
    """
    WITH r AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
        FROM prices
    )
    SELECT ts, symbol, round(ret * 100, 1) AS ret_pct
    FROM r
    WHERE ret IS NOT NULL
    ORDER BY ret
    LIMIT 5
    """
).to_pandas()
worst

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for sym, g in px.groupby("symbol"):
    axes[0].plot(g["ts"], g["close_raw"], lw=0.9, label=sym)
for s in SPLITS.itertuples():
    axes[0].axvline(s.ts, color="crimson", ls="--", lw=0.8)
axes[0].set_yscale("log")
axes[0].set_title("Raw tape: split cliffs (dashed = effective dates)")
axes[0].set_xlabel("date")
axes[0].set_ylabel("close (USD, log)")
axes[0].legend()

nvda = px[px["symbol"] == "NVDA"]
axes[1].plot(nvda["ts"], nvda["close_raw"] / nvda["close_raw"].iloc[0], lw=0.9, label="naive (raw closes)")
axes[1].plot(nvda["ts"], nvda["close"] / nvda["close"].iloc[0], lw=0.9, label="split-adjusted")
axes[1].set_yscale("log")
axes[1].set_title("NVDA cumulative growth: naive vs adjusted")
axes[1].set_xlabel("date")
axes[1].set_ylabel("growth of $1 (log)")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# ## 4. Pattern A - adjustment factors joined at read time
#
# Keep `prices` raw forever; store the split schedule as its own tiny h5i-db
# table and derive the adjusted series in SQL. A row's adjustment factor is the
# product of all split ratios with a *later* effective date - `exp(sum(ln ...))`
# turns the product into an aggregate, so one join + group-by does it. When the
# next split is announced, ingestion is a one-row append; no historical data is
# touched.

# %%
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("ratio", pa.float64()),
    ]
)
db.create_table("splits", SPLIT_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append("splits", pa.Table.from_pandas(SPLITS, preserve_index=False).cast(SPLIT_SCHEMA),
          note="split schedule through 2024-06-10")

adjusted = db.sql(
    """
    SELECT p.ts, p.symbol, p.close AS close_raw,
           round(p.close / coalesce(exp(sum(ln(s.ratio))), 1.0), 4) AS close_adj
    FROM prices p
    LEFT JOIN splits s
           ON s.symbol = p.symbol AND s.ts > p.ts
    GROUP BY p.ts, p.symbol, p.close
    ORDER BY p.ts, p.symbol
    """
).to_pandas()
adjusted[(adjusted["symbol"] == "AAPL") & adjusted["ts"].between("2020-08-27", "2020-09-02")]

# %% [markdown]
# Sanity check: our SQL-adjusted series must agree with the vendor's
# split-adjusted `close` to the penny (the only noise is the cent-rounding we
# applied when reconstructing the raw tape).

# %%
check = adjusted.merge(px[["ts", "symbol", "close"]], on=["ts", "symbol"])
max_err = (check["close_adj"] - check["close"]).abs().max()
print(f"max |SQL-adjusted - vendor adjusted| = ${max_err:.4f} across {len(check):,} rows")
assert max_err < 0.01

# %% [markdown]
# The `splits` table is a feed like any other: appends must arrive in time
# order. A backfilled 2019 split cannot be appended after the 2024 row - h5i-db
# rejects it rather than silently corrupting the sort order. (Historical
# corrections go through `write()` or the plan/apply flow instead.)

# %%
stale_split = pa.table(
    {
        "ts": pa.array([pd.Timestamp("2019-06-01", tz="UTC")], pa.timestamp("us", tz="UTC")),
        "symbol": ["FAKE"],
        "ratio": [2.0],
    }
).cast(SPLIT_SCHEMA)
try:
    db.append("splits", stale_split)
except h5i_db.H5iError as e:
    print(f"rejected: code={e.code}\nhint: {e.hint}")

# %% [markdown]
# ## 5. Pattern B - restate in place, keep history via versioning
#
# Many desks prefer the adjusted series to *be* the table, so every consumer
# gets clean prices with zero join logic. The usual cost is that the raw tape
# is gone. With versioned storage it isn't: `write()` replaces the head but
# every prior version remains readable, so restatement becomes an audited,
# reversible operation rather than a destructive one.

# %%
restated = px.copy()
restated["close"] = (restated["close_raw"] / restated["future_factor"]).round(4)
db.write(
    "prices",
    pa.Table.from_pandas(restated[["ts", "symbol", "close", "volume"]], preserve_index=False).cast(PRICE_SCHEMA),
    note="restated: split-adjusted (AAPL 4:1 2020-08-31, NVDA 4:1 2021-07-20, NVDA 10:1 2024-06-10)",
)

[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("prices")]

# %% [markdown]
# The version history *is* the audit trail. `h5i('prices', 1)` queries the
# pre-restatement tape alongside the live table - here, AAPL around its split,
# with the implied factor recovered from the two versions:

# %%
db.sql(
    """
    SELECT now.ts, now.symbol,
           was.close                       AS raw_close,
           now.close                       AS adj_close,
           round(was.close / now.close, 2) AS implied_factor
    FROM prices now
    JOIN h5i('prices', 1) was
      ON now.ts = was.ts AND now.symbol = was.symbol
    WHERE now.symbol = 'AAPL' AND now.ts BETWEEN '2020-08-26' AND '2020-09-03'
    ORDER BY now.ts
    """
).to_pandas()

# %% [markdown]
# "What did the price series look like before the restatement?" also works by
# wall-clock commit time - useful when a downstream job only logged *when* it
# read, not which version:

# %%
raw_commit = db.versions("prices")[1]  # sequence 1 = the raw-tape append
as_of = pd.Timestamp(raw_commit["committed_at_ns"], unit="ns", tz="UTC").isoformat()
pre = db.read("prices", as_of=as_of)
aapl_pre = pre.to_pandas().query("symbol == 'AAPL' and ts == '2020-08-28 20:00:00+00:00'")
print(f"as of {as_of}:")
print(aapl_pre[["ts", "symbol", "close"]].to_string(index=False))

# %% [markdown]
# And with the head restated, the split artifacts are gone - the worst days in
# the series are now genuine market moves (COVID-era selloffs, earnings), not
# corporate-action ghosts:

# %%
db.sql(
    """
    WITH r AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
        FROM prices
    )
    SELECT ts, symbol, round(ret * 100, 1) AS ret_pct
    FROM r WHERE ret IS NOT NULL
    ORDER BY ret
    LIMIT 5
    """
).to_pandas()

# %% [markdown]
# ## Choosing between the patterns
#
# - **Adjustment-factor join** (Pattern A): raw tape stays canonical; adjusted
#   prices are always consistent with the *current* split schedule; new actions
#   are one-row appends. Cost: every consumer must apply the join (or read a
#   derived table you maintain).
# - **Restate in place** (Pattern B): consumers get clean prices with no join;
#   the restatement is one audited commit, reversible via `restore()` and
#   inspectable via `h5i('prices', v)`. Cost: between action announcement and
#   restatement the head is stale, and each restatement rewrites the table.
# - They compose: keep `prices` raw + a `splits` table as ground truth, and
#   publish a restated `prices_adj` table for convenience - versioning gives
#   you the audit trail on both.

# %% [markdown]
# ## Takeaways
#
# - Vendor "close" columns are often already adjusted - know what your feed
#   delivers before you build on it. Here we reconstructed the true raw tape
#   from the split schedule.
# - Naive returns across a split print -75% (AAPL 4:1) and -90% (NVDA 10:1)
#   artifacts; one `lag()` window query makes them jump out.
# - The adjustment-factor pattern is a one-join SQL derivation
#   (`exp(sum(ln(ratio)))`) off an append-only `splits` table.
# - `write()` + `note` turns restatement into an audited commit;
#   `h5i('prices', 1)` and `read(as_of=...)` keep every pre-restatement view
#   queryable - you never have to choose between clean data and the raw tape.

# %%
db.close()
