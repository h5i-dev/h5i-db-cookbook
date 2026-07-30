# %% [markdown]
# # Corporate actions: split adjustment without losing the tape
#
# A stock split rewrites history. Every price before the effective date must be
# scaled, or every return computed across it is garbage.
#
# That forces an engineering choice. Do you *restate* the stored series and lose
# the raw tape, or keep the raw tape canonical and adjust at read time?
#
# With h5i-db you do not have to lose anything either way. A restatement is a
# `write()` commit with a note, the pre-restatement series stays readable
# forever through time travel, and the adjustment-factor pattern is a small SQL
# join.
#
# This recipe builds both patterns on real AAPL and NVDA data covering three
# splits: AAPL 4:1 (2020-08-31), NVDA 4:1 (2021-07-20) and NVDA 10:1
# (2024-06-10).

# %% [markdown]
# ## Terms used here
#
# | term              | meaning |
# | ----------------- | --- |
# | corporate action  | a change to the security itself rather than to its price |
# | split             | a share count multiplication, e.g. 4:1; the price divides by the same factor |
# | adjusted close    | a close rescaled for all later splits and dividends, so returns compare across time |
# | adjustment factor | the multiplier converting raw prices to adjusted ones |
# | restatement       | rewriting stored history, here as a `write()` commit carrying a note |
# | time travel       | reading a table as it was at an earlier version |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_corpactions"), create=True)

# %% [markdown]
# ## 1. The data
#
# `cu.fetch_daily` pulls real daily bars from Yahoo Finance and caches them to
# parquet, so this recipe is deterministic and works offline after the first
# run. One row per symbol per session.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | session date |
# | `symbol` | `string` | ticker |
# | `open`, `high`, `low`, `close` | `float64` | session prices |
# | `adj_close` | `float64` | close adjusted for splits *and* dividends |
# | `volume` | `int64` | shares traded |

# %%
real = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
print(f"{real.num_rows:,} rows x {real.num_columns} columns, "
      f"{len(set(real['symbol'].to_pylist()))} symbols")
real.to_pandas().head()

# %% [markdown]
# One wrinkle decides the whole recipe. Yahoo's `close` column is **already
# split-adjusted**, and `adj_close` only adds dividend adjustments on top. The
# AAPL close/adj_close ratio of about 1.07 in 2018 is pure dividends.
#
# A real exchange feed would have printed AAPL at roughly \\$499 on 2020-08-28,
# not \\$125. So we *un-adjust* using the known split schedule to recover the
# as-printed tape. That is what a raw vendor feed delivers, and the honest
# starting point here.

# %%
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
# There is the cliff: \\$499 on the 28th, \\$129 on the 31st.
#
# ## 2. Store the raw tape as the canonical table
#
# The raw series is what the exchange actually printed, so it should be the
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
# Compute daily returns straight off the raw closes with a `lag()` window, then
# look at the worst days on record.
#
# Three of them are not crashes. They are splits masquerading as -75% and -90%
# returns. Any risk model, momentum signal or stop-loss fed this series would
# have fired on a non-event.

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")


def worst_days(version=None, n: int = 5):
    """The n most negative daily returns - re-run after the restatement."""
    return (
        db.table("prices", version=version)
        .with_columns(ret=col("close") / PREV_CLOSE - 1)
        .filter(col("ret").is_not_null())
        .select("ts", "symbol", ret_pct=(col("ret") * 100).round(1))
        .sort("ret_pct")
        .limit(n)
        .to_pandas()
    )


worst = worst_days()
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
# ## 4. Pattern A: adjustment factors joined at read time
#
# Keep `prices` raw forever. Store the split schedule as its own tiny h5i-db
# table and derive the adjusted series in SQL.
#
# A row's adjustment factor is the product of all split ratios with a *later*
# effective date. `exp(sum(ln ...))` turns that product into an aggregate, so
# one join plus a group-by does the whole job. When the next split is announced,
# ingestion is a one-row append and no historical data is touched.

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

# A non-equi LEFT JOIN (`s.ts > p.ts`) whose whole condition would have to be
# a raw fragment anyway - one of the places a SQL string stays clearer than
# the builder.
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
# Sanity check. Our SQL-adjusted series must agree with the vendor's
# split-adjusted `close` to the penny. The only noise is the cent-rounding we
# applied when reconstructing the raw tape.

# %%
check = adjusted.merge(px[["ts", "symbol", "close"]], on=["ts", "symbol"])
max_err = (check["close_adj"] - check["close"]).abs().max()
print(f"max |SQL-adjusted - vendor adjusted| = ${max_err:.4f} across {len(check):,} rows")
assert max_err < 0.01

# %% [markdown]
# The `splits` table is a feed like any other, so appends must arrive in time
# order. A backfilled 2019 split cannot be appended after the 2024 row. h5i-db
# rejects it rather than silently corrupting the sort order, and historical
# corrections go through `write()` or the plan/apply flow instead.

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
# ## 5. Pattern B: restate in place, keep history via versioning
#
# Many desks prefer the adjusted series to *be* the table, so every consumer
# gets clean prices with zero join logic. The usual cost is that the raw tape is
# gone.
#
# With versioned storage it is not. `write()` replaces the head, but every prior
# version remains readable, so restatement becomes an audited, reversible
# operation rather than a destructive one.

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
# pre-restatement tape alongside the live table. Below, AAPL around its split,
# with the implied factor recovered from the two versions:

# %%
now_px = col("close", relation="l")
was_px = col("close", relation="r")

(
    db.table("prices")  # head: restated
    .join(db.table("prices", version=1), on=["ts", "symbol"])  # raw tape
    .filter(
        col("symbol", relation="l") == "AAPL",
        col("ts", relation="l").between("2020-08-26", "2020-09-03"),
    )
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        raw_close=was_px,
        adj_close=now_px,
        implied_factor=(was_px / now_px).round(2),
    )
    .sort("ts")
    .to_pandas()
)

# %% [markdown]
# "What did the price series look like before the restatement?" also works by
# wall-clock commit time. That is useful when a downstream job only logged
# *when* it read, not which version.

# %%
raw_commit = db.versions("prices")[1]  # sequence 1 = the raw-tape append
as_of = pd.Timestamp(raw_commit["committed_at_ns"], unit="ns", tz="UTC").isoformat()
pre = db.read("prices", as_of=as_of).to_pandas()
# Compare against a Timestamp, not a string: pandas parses a string literal to
# datetime64[ns] and the unit mismatch with our [us] column matches nothing.
split_eve = pd.Timestamp("2020-08-28 20:00", tz="UTC")
aapl_pre = pre[(pre["symbol"] == "AAPL") & (pre["ts"] == split_eve)]
print(f"as of {as_of}:")
print(aapl_pre[["ts", "symbol", "close"]].to_string(index=False))

# %% [markdown]
# And with the head restated, the split artifacts are gone. The worst days in
# the series are now genuine market moves, COVID-era selloffs and earnings,
# rather than corporate-action ghosts.

# %%
worst_days()

# %% [markdown]
# ## Choosing between the patterns
#
# - **Adjustment-factor join** (Pattern A). The raw tape stays canonical,
#   adjusted prices are always consistent with the *current* split schedule, and
#   new actions are one-row appends. The cost is that every consumer must apply
#   the join, or read a derived table you maintain.
# - **Restate in place** (Pattern B). Consumers get clean prices with no join,
#   the restatement is one audited commit, reversible via `restore()` and
#   inspectable via `h5i('prices', v)`. The cost is that the head is stale
#   between action announcement and restatement, and each restatement rewrites
#   the table.
# - They compose. Keep `prices` raw plus a `splits` table as ground truth, and
#   publish a restated `prices_adj` table for convenience. Versioning gives you
#   the audit trail on both.

# %% [markdown]
# ## Takeaways
#
# - Vendor "close" columns are often already adjusted, so know what your feed
#   delivers before you build on it. Here we reconstructed the true raw tape
#   from the split schedule.
# - Naive returns across a split print -75% (AAPL 4:1) and -90% (NVDA 10:1)
#   artifacts, and one `lag()` window query makes them jump out.
# - The adjustment-factor pattern is a one-join SQL derivation,
#   `exp(sum(ln(ratio)))`, off an append-only `splits` table.
# - `write()` plus a `note` turns restatement into an audited commit.
#   `h5i('prices', 1)` and `read(as_of=...)` keep every pre-restatement view
#   queryable, so you never have to choose between clean data and the raw tape.

# %%
db.close()
