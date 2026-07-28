# %% [markdown]
# # NBBO consolidation: best bid/offer across fragmented venues
#
# US equities trade on a dozen or more venues, each publishing its own top of
# book. The consolidated best bid and offer, the number your execution quality
# is measured against, has to be *derived*: at any instant, take each venue's
# latest quote and best them cross-sectionally.
#
# "Latest quote per venue at time t" is exactly an ASOF join. So a
# research-grade sampled NBBO is a few SQL statements: a time grid ASOF-joined
# against per-venue quotes, then `max(bid)` and `min(ask)` per instant.
#
# In this recipe we:
#
# 1. fan a consolidated quote stream out to three venues with distinct
#    microstructure personalities,
# 2. snapshot each venue's prevailing quote on a 10-second grid,
# 3. consolidate and store the NBBO as its own table,
# 4. measure which venue sets the inside, how often the market locks or
#    crosses, and how much tighter the consolidated spread is than any single
#    book.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_nbbo"), create=True)

# %% [markdown]
# ## 1. The data
#
# One session of consolidated-style AAPL quotes from `cu.make_quotes`, one row
# per change in the best bid or offer.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | quote timestamp, ascending |
# | `symbol` | `string` | ticker, `AAPL` only here |
# | `bid`, `ask` | `float64` | best bid and offer |
# | `bid_size`, `ask_size` | `int64` | displayed depth at each side |

# %%
quotes = cu.make_quotes(
    symbols=["AAPL"], days=1, quotes_per_day=2_600, seed=11, base_prices={"AAPL": 320.0}
).to_pandas()
print(f"{len(quotes):,} rows x {quotes.shape[1]} columns")
quotes.head()

# %% [markdown]
# We fan that single stream out to three venues, each with its own personality.
# A venue re-quotes on the consolidated updates but occasionally *drops* some,
# which is where genuine staleness comes from. It adds its own gateway latency,
# and it skews its price a tick away from, or occasionally inside, the
# reference.
#
# That mix is what makes one venue more competitive at the inside than another,
# and what lets brief locked and crossed states appear. Seeds are fixed
# throughout.
#
# The fanned-out table has one row per venue quote: `ts`, `venue`, `bid`, `ask`,
# `bid_size`, `ask_size`.

# %%
rng = np.random.default_rng(5)
TICK = 0.01

VENUES = {
    #        P(miss update), latency (ms), P(1 tick worse), P(1 tick better)
    "ARCA": dict(drop=0.04, lat_ms=3.0, worse=0.25, better=0.02),
    "EDGX": dict(drop=0.08, lat_ms=10.0, worse=0.35, better=0.005),
    "XNAS": dict(drop=0.02, lat_ms=1.5, worse=0.15, better=0.03),
}

parts = []
for venue, v in VENUES.items():
    d = quotes[rng.random(len(quotes)) >= v["drop"]].copy()
    n = len(d)
    d["ts"] = d["ts"] + pd.to_timedelta(rng.exponential(v["lat_ms"], n), unit="ms").round("us")
    d["bid"] = d["bid"] - TICK * (rng.random(n) < v["worse"]) + TICK * (rng.random(n) < v["better"])
    d["ask"] = d["ask"] + TICK * (rng.random(n) < v["worse"]) - TICK * (rng.random(n) < v["better"])
    d["venue"] = venue
    parts.append(d)

venue_quotes = (
    pd.concat(parts)[["ts", "venue", "bid", "ask", "bid_size", "ask_size"]]
    .assign(bid=lambda x: x["bid"].round(2), ask=lambda x: x["ask"].round(2))
    .sort_values(["ts", "venue"], kind="stable")
    .reset_index(drop=True)
)
venue_quotes.assign(spread=venue_quotes["ask"] - venue_quotes["bid"]).groupby("venue").agg(
    quotes=("ts", "count"), avg_spread=("spread", "mean")
).round(4)

# %%
VQ_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("venue", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
    ]
)
db.create_table("venue_quotes", VQ_SCHEMA, time_column="ts", sort_key=["ts", "venue"])
db.append("venue_quotes", pa.Table.from_pandas(venue_quotes, preserve_index=False).cast(VQ_SCHEMA),
          note="per-venue AAPL quotes, 1 session, 3 venues")

# %% [markdown]
# ## 2. A sampling grid, ASOF-joined per venue
#
# An event-accurate NBBO re-evaluates on *every* quote update. For research, a
# sampled NBBO on a regular grid is the standard approximation, with error
# bounded by the sampling interval. Consolidated updates arrive roughly every 9
# seconds here and we sample every 10.
#
# The grid is itself a tiny h5i-db table, one row per (timestamp, venue). So a
# single `asof_join(grid, venue_quotes, ..., 'venue')` snapshots each venue's
# prevailing quote at each instant. The 60-second `tolerance`, in raw
# microseconds, declares a venue's quote stale and returns NULL, rather than
# letting a dead book linger at the inside.
#
# One habit is worth keeping in any ASOF pipeline: assert the join returned
# exactly one row per grid row. A LEFT ASOF join is 1:1 with its left side, so a
# sizing or key mistake then fails loudly instead of skewing the book.

# %%
open_ts = venue_quotes["ts"].min().ceil("10s")
close_ts = pd.Timestamp("2026-06-01 20:00", tz="UTC")
grid_times = pd.date_range(open_ts, close_ts, freq="10s", inclusive="left")

grid = pd.DataFrame(
    {
        "ts": np.repeat(grid_times, len(VENUES)),
        "venue": np.tile(sorted(VENUES), len(grid_times)),
    }
)
GRID_SCHEMA = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("venue", pa.string())]
)
db.create_table("nbbo_grid", GRID_SCHEMA, time_column="ts", sort_key=["ts", "venue"])
db.append("nbbo_grid", pa.Table.from_pandas(grid, preserve_index=False).cast(GRID_SCHEMA))
print(f"grid: {len(grid_times)} instants x {len(VENUES)} venues = {len(grid):,} rows")

# %%
STALE_US = 60 * 1_000_000

# One frame, reused for every question below: each grid instant x venue gets
# the venue's prevailing quote, or NULL if the last one is older than STALE_US.
SNAPSHOT = db.table("nbbo_grid").join_asof(
    db.table("venue_quotes"), on="ts", by="venue",
    direction="backward", tolerance=STALE_US,
)

(
    SNAPSHOT.select("ts", "venue", "bid", "ask", quoted_at=col("ts_right"))
    .sort(["ts", "venue"])
    .limit(6)
    .to_pandas()
)

# %%
# Trust, but verify: one output row per grid row, and the SQL ASOF must agree
# with an independent pandas merge_asof on a venue's book.
state = SNAPSHOT.select("ts", "venue", "bid", "ask").sort(["ts", "venue"]).to_pandas()
assert len(state) == len(grid), f"ASOF join returned {len(state)} rows for {len(grid)} grid rows"

ref = pd.merge_asof(
    pd.DataFrame({"ts": grid_times}),
    venue_quotes[venue_quotes["venue"] == "XNAS"][["ts", "bid", "ask"]],
    on="ts", tolerance=pd.Timedelta("60s"),
)
chk = state[state["venue"] == "XNAS"].reset_index(drop=True)
assert chk["bid"].equals(ref["bid"]) and chk["ask"].equals(ref["ask"])
print(f"validated: {len(state):,} snapshot rows, XNAS column matches pandas merge_asof exactly")

# %% [markdown]
# ## 3. Consolidate and store the NBBO
#
# Bettering across venues is now a plain GROUP BY per grid instant.
#
# We materialize the result as its own table. Downstream recipes, such as
# execution benchmarks and effective-spread studies, read `nbbo_10s` without
# re-running the consolidation.

# %%
nbbo = (
    SNAPSHOT.group_by("ts")
    .agg(nbb=col("bid").max(), nbo=col("ask").min(), venues_quoting=col("bid").count())
    .filter(col("nbb").is_not_null(), col("nbo").is_not_null())  # the HAVING
    .sort("ts")
    .to_arrow()
)

NBBO_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("nbb", pa.float64()),
        pa.field("nbo", pa.float64()),
        pa.field("venues_quoting", pa.int64()),
    ]
)
db.create_table("nbbo_10s", NBBO_SCHEMA, time_column="ts")
db.append("nbbo_10s", nbbo.cast(NBBO_SCHEMA), note="10s-sampled NBBO from 3 venues")
db.table("nbbo_10s").limit(5).to_pandas()

# %% [markdown]
# ## 4. Who sets the inside, and how often does it lock or cross?
#
# Window functions over the per-venue snapshot answer the venue-share question:
# at each instant, is this venue's bid equal to the consolidated best? Ties
# count for every venue at the inside.
#
# The same snapshot flags *locked* states, where NBB equals NBO, and *crossed*
# states, where NBB exceeds NBO. With independent per-venue latencies and price
# skews, brief locks and crossings are a fact of consolidated life, and any NBBO
# pipeline has to decide how to treat them.

# %%
QUOTING = SNAPSHOT.filter(col("bid").is_not_null(), col("ask").is_not_null())

share = lambda flag: (when(flag).then(lit(1.0)).otherwise(lit(0.0)).mean() * 100).round(1)

(
    QUOTING.with_columns(
        nbb=col("bid").max().over(partition_by="ts"),
        nbo=col("ask").min().over(partition_by="ts"),
    )
    .group_by("venue")
    .agg(
        pct_at_best_bid=share(col("bid") >= col("nbb")),
        pct_at_best_ask=share(col("ask") <= col("nbo")),
        avg_own_spread=(col("ask") - col("bid")).mean().round(4),
    )
    .sort("venue")
    .to_pandas()
)

# %%
count_if = lambda flag: when(flag).then(lit(1)).otherwise(lit(0)).sum()

(
    db.table("nbbo_10s")
    .select(
        instants=count_star(),
        locked=count_if(col("nbb") == col("nbo")),
        crossed=count_if(col("nbb") > col("nbo")),
        pct_locked_or_crossed=(
            when(col("nbb") >= col("nbo")).then(lit(1.0)).otherwise(lit(0.0)).mean() * 100
        ).round(2),
    )
    .to_pandas()
)

# %% [markdown]
# ## 5. The consolidation dividend: spread vs any single venue
#
# Here is the point of consolidating. The inside spread is tighter than every
# individual book, because the best bid and the best ask usually live on
# *different* venues.
#
# One UNION ALL puts the books side by side, and the session plot shows the NBBO
# spread hugging the floor below each venue's own spread.
#
# There is no `UNION` verb, so this is the moment to walk through the door.
# `.sql()` renders the per-venue half we already built, and the string supplies
# the union. No f-string interpolation of the tolerance, and no hand-rewriting
# of a query that already exists.

# %%
venue_spreads = QUOTING.group_by(col("venue").alias("book")).agg(
    avg_spread_usd=(col("ask") - col("bid")).mean().round(4)
)

db.sql(
    f"""
    {venue_spreads.sql()}
    UNION ALL
    SELECT 'NBBO', round(avg(nbo - nbb), 4) FROM nbbo_10s
    ORDER BY avg_spread_usd
    """
).to_pandas()

# %%
import matplotlib.pyplot as plt

MINUTE = time_bucket("1m", col("ts"))

venue_min = (
    QUOTING.group_by(MINUTE.alias("m"), "venue")
    .agg(spread=(col("ask") - col("bid")).mean())
    .sort("m")
    .to_pandas()
)
nbbo_min = (
    db.table("nbbo_10s")
    .group_by(MINUTE.alias("m"))
    .agg(spread=(col("nbo") - col("nbb")).mean())
    .sort("m")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
for venue, g in venue_min.groupby("venue"):
    ax.plot(g["m"], g["spread"] * 100, lw=0.8, alpha=0.7, label=venue)
ax.plot(nbbo_min["m"], nbbo_min["spread"] * 100, lw=1.6, color="black", label="NBBO")
ax.set_title("Quoted spread through the session: each venue vs consolidated NBBO")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("spread (cents)")
ax.legend(ncols=4)
fig.tight_layout()

# %% [markdown]
# ## Takeaways
#
# - NBBO construction is "latest quote per venue, then best across venues",
#   which maps one-to-one onto `asof_join(grid, venue_quotes, ..., 'venue')`
#   plus a GROUP BY. No per-venue loops, no manual state machines.
# - The ASOF `tolerance` doubles as a staleness rule. A venue that stops quoting
#   drops out of the consolidation instead of pinning a dead price at the
#   inside.
# - A sampled NBBO is an approximation. Honest work states its grid, 10s here,
#   and remembers that locks and crosses between samples go unseen.
# - Materializing `nbbo_10s` as a table makes the consolidation a build
#   artifact: versioned, noted, and instantly queryable by every downstream
#   execution study.

# %%
db.close()
