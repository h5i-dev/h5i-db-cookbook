# %% [markdown]
# # ASOF joins: trades vs quotes, signing and spreads
#
# Attaching the prevailing quote to every trade is *the* microstructure
# join: it powers trade signing, effective/realized spread measurement, TCA
# and toxicity analytics. In h5i-db `asof_join` is a native SQL table
# function over time-sorted storage — no round-trip through pandas — with
# direction (`'backward'` / `'forward'`) and a staleness tolerance built in,
# and it composes with CTEs and window functions like any other table.
#
# On a two-session tape with known ground truth we attach quotes to every
# trade (cross-checked against `pandas.merge_asof`), sign trades Lee-Ready
# style and score the signing, measure quoted / effective / realized
# spreads, and use tolerance to refuse stale quotes instead of silently
# marking against them.

# %%
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_asof"), create=True)

# %% [markdown]
# ## 1. A tape with known ground truth
#
# To *score* trade signing we need trades whose true aggressor side we know,
# so we derive the printed tape from the quote stream: each trade takes the
# prevailing bid or ask (10% execute at mid — the hard case for
# classification), 0.2–5ms of exchange reporting latency, and keeps its true
# `side`. We also store a `marks` table — each trade's timestamp shifted
# +5 minutes — which will drive the realized-spread lookup later.

# %%
quotes = cu.make_quotes(
    symbols=["AAPL", "MSFT", "NVDA"], days=2, quotes_per_day=1_000, seed=11
)

qp = quotes.to_pandas()
rng = np.random.default_rng(23)
is_trade = rng.random(len(qp)) < 0.6
tr = qp.loc[is_trade, ["ts", "symbol", "bid", "ask"]].reset_index(drop=True)
n = len(tr)
buy = rng.random(n) < 0.5
at_mid = rng.random(n) < 0.10
px = np.where(buy, tr["ask"], tr["bid"])
px = np.where(at_mid, (tr["bid"] + tr["ask"]) / 2, px)
tr["price"] = px
tr["size"] = np.maximum(1, rng.lognormal(4.0, 1.2, n) // 100 * 100).astype("int64")
tr["side"] = np.where(buy, "B", "S")
tr["ts"] = (tr["ts"] + pd.to_timedelta(rng.uniform(0.2, 5.0, n), unit="ms")).dt.floor("us")
tr = tr.sort_values(["ts", "symbol"]).reset_index(drop=True)
tr["trade_id"] = np.arange(n, dtype="int64")

ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("trade_id", pa.int64()),
     pa.field("price", pa.float64()), pa.field("size", pa.int64()),
     pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(
    tr[["ts", "symbol", "trade_id", "price", "size", "side"]], preserve_index=False
).cast(trade_schema))

quote_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("bid", pa.float64()),
     pa.field("ask", pa.float64()), pa.field("bid_size", pa.int64()),
     pa.field("ask_size", pa.int64())]
)
db.create_table("quotes", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes.cast(quote_schema))

marks = tr[["ts", "symbol", "trade_id"]].copy()
marks["ts"] = marks["ts"] + pd.Timedelta(minutes=5)
marks = marks.sort_values(["ts", "symbol"])
mark_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("trade_id", pa.int64())]
)
db.create_table("marks", mark_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("marks", pa.Table.from_pandas(marks, preserve_index=False).cast(mark_schema))

print(f"{len(tr):,} trades, {len(qp):,} quotes over 2 sessions")

# %% [markdown]
# ## 2. The join itself
#
# `asof_join(left, right, left_ts, right_ts, by_key)`: for each trade, the
# latest quote at or before it, per symbol. Right-side columns that collide
# get a `_right` suffix — here the quote's own timestamp arrives as
# `ts_right`, so trade-time minus `ts_right` is the quote's age. Because
# both tables are stored time-ordered there is no sort phase to pay for:

# %%
t0 = time.perf_counter()
joined = db.sql(
    """
    SELECT ts, symbol, trade_id, price, side, ts_right, bid, ask
    FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ORDER BY trade_id
    """
).to_pandas()
elapsed_ms = (time.perf_counter() - t0) * 1e3
print(f"{len(joined):,} trades matched against {len(qp):,} quotes "
      f"in {elapsed_ms:.1f} ms")
joined.head(5)

# %% [markdown]
# Trust, but verify: the same alignment through `pandas.merge_asof` must
# produce identical quotes for every trade.

# %%
left = tr[["ts", "symbol", "trade_id"]].sort_values("ts")
left["ts"] = left["ts"].astype("datetime64[us, UTC]")  # match arrow's us resolution
ref = pd.merge_asof(
    left, qp[["ts", "symbol", "bid", "ask"]],
    on="ts", by="symbol", direction="backward",
)
cmp = joined.merge(ref[["trade_id", "bid", "ask"]], on="trade_id", suffixes=("", "_pd"))
assert len(cmp) == len(tr)
assert np.allclose(cmp["bid"], cmp["bid_pd"]) and np.allclose(cmp["ask"], cmp["ask_pd"])
print(f"asof_join matches pandas.merge_asof on all {len(cmp):,} trades")

# %% [markdown]
# There is also a keyword spelling — handy when the ASOF join is one clause
# of a larger statement. Two current limitations to know about: it requires
# bare table names (no aliases), and the output columns are referenced
# unqualified:

# %%
db.sql(
    """
    SELECT ts, symbol, price, bid, ask
    FROM trades ASOF JOIN quotes
    MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol
    LIMIT 3
    """
).to_pandas()

# %% [markdown]
# ## 3. Lee-Ready trade signing
#
# Quote rule first — above the mid is a buy, below is a sell — then the tick
# test for exact-mid prints (here via `lag(price)`; strictly Lee-Ready wants
# the last *different* price, a refinement that matters little on this
# tape). All in one statement: the asof join feeds a window function feeds a
# `CASE`.

# %%
signed = db.sql(
    """
    WITH j AS (
        SELECT ts, symbol, trade_id, price, size, side, bid, ask,
               (bid + ask) / 2 AS mid
        FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ),
    t AS (
        SELECT *, lag(price) OVER (PARTITION BY symbol ORDER BY ts) AS prev_px
        FROM j
    )
    SELECT *,
           CASE WHEN price > mid THEN 'B'
                WHEN price < mid THEN 'S'
                WHEN price > prev_px THEN 'B'
                WHEN price < prev_px THEN 'S'
                ELSE 'U' END AS lr_side
    FROM t
    """
).to_pandas()

overall = (signed["lr_side"] == signed["side"]).mean()
quote_rule = signed[signed["price"] != signed["mid"]]
mid_prints = signed[signed["price"] == signed["mid"]]
print(f"overall accuracy      : {overall:.1%}")
print(f"quote-rule prints     : {(quote_rule['lr_side'] == quote_rule['side']).mean():.1%} "
      f"of {len(quote_rule):,}")
print(f"mid prints (tick test): {(mid_prints['lr_side'] == mid_prints['side']).mean():.1%} "
      f"of {len(mid_prints):,}")
pd.crosstab(signed["lr_side"], signed["side"], margins=True)

# %% [markdown]
# The quote rule is near-perfect — its only failure mode is a quote update
# landing inside the trade's few-ms reporting latency — while mid prints
# fall back to the roughly coin-flip tick test. On real TAQ data the
# quote-rule share is lower and overall accuracy lands around 85–90%; the
# split above shows exactly where the errors live.
#
# ## 4. Quoted, effective and realized spread
#
# - quoted: `ask - bid` at the prevailing quote,
# - effective: `2·|price - mid|` — what aggressors actually paid,
# - realized: `2·d·(price - mid₊₅ₘ)` — what liquidity providers actually
#   kept, using the mid 5 minutes later.
#
# The 5-minutes-later mid is the `marks` table asof-joined **forward**: the
# first quote at or after trade-time + 5m. Everything lands in one
# statement, in basis points of the mid.

# %%
spreads = db.sql(
    """
    WITH j AS (
        SELECT symbol, trade_id, price, side, bid, ask, (bid + ask) / 2 AS mid
        FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ),
    m AS (
        SELECT trade_id, (bid + ask) / 2 AS mid_5m
        FROM asof_join('marks', 'quotes', 'ts', 'ts', 'symbol', 'forward')
    )
    SELECT j.symbol,
           avg((j.ask - j.bid) / j.mid) * 1e4                    AS quoted_bps,
           avg(2 * abs(j.price - j.mid) / j.mid) * 1e4           AS effective_bps,
           avg(2 * (CASE WHEN j.side = 'B' THEN 1 ELSE -1 END)
                 * (j.price - m.mid_5m) / j.mid) * 1e4           AS realized_bps,
           count(*)                                              AS n_trades
    FROM j JOIN m USING (trade_id)
    WHERE m.mid_5m IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
).to_pandas()
spreads

# %%
x = np.arange(len(spreads))
w = 0.27
fig, ax = plt.subplots(figsize=(8, 4))
for i, col in enumerate(("quoted_bps", "effective_bps", "realized_bps")):
    ax.bar(x + (i - 1) * w, spreads[col], width=w, label=col.replace("_bps", ""))
ax.set_xticks(x, spreads["symbol"])
ax.set_title("Spread measures by symbol (bps of mid)")
ax.set_xlabel("symbol")
ax.set_ylabel("basis points")
ax.legend()
fig.tight_layout()

# %% [markdown]
# Effective sits a notch below quoted because 10% of prints execute at mid.
# Realized shows no *systematic* gap to effective — this synthetic flow
# carries no information, so on average the mid does not drift against the
# liquidity provider after the trade (the per-symbol estimates scatter
# around effective because 5-minute mid volatility dwarfs the spread; that
# noisiness is faithful to the real-world estimator too). On a real tape
# realized sits systematically *below* effective, and the gap
# (2·d·(mid₊₅ₘ − mid), the price impact) is the adverse-selection cost —
# this decomposition is the standard way to measure it.
#
# ## 5. Tolerance: refuse stale quotes
#
# By default the asof join reaches arbitrarily far back — against a gappy
# quote feed (outages, subsampled vendor files) that silently marks trades
# against minutes-old NBBO. The optional tolerance (raw microseconds, like
# every raw time argument in h5i-db) bounds the lookback: unmatched trades
# keep NULL quote columns, so staleness becomes *visible* and countable. We
# thin the quote feed to 25% to simulate the problem:

# %%
qs = qp[rng.random(len(qp)) < 0.25].sort_values(["ts", "symbol"])
db.create_table("quotes_sparse", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_sparse", pa.Table.from_pandas(qs, preserve_index=False).cast(quote_schema))

for label, tol in [("no tolerance", None), ("60s tolerance", 60_000_000),
                   ("5s tolerance", 5_000_000)]:
    args = "'ts', 'ts', 'symbol'" + (f", 'backward', {tol}" if tol else "")
    r = db.sql(
        f"SELECT count(*) AS n, count(bid) AS matched "
        f"FROM asof_join('trades', 'quotes_sparse', {args})"
    ).to_pandas()
    print(f"{label:>13}: {r['matched'][0]:,} of {r['n'][0]:,} trades matched "
          f"({r['matched'][0] / r['n'][0]:.1%})")

# %% [markdown]
# Without a tolerance every trade gets *some* quote, however stale; with a
# freshness requirement, the trades whose prevailing quote is too old
# honestly report NULL instead. In production that NULL share is a
# data-quality metric worth alerting on.
#
# ## Takeaways
#
# - `asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')` is the whole
#   trades-vs-quotes alignment — per-key, time-correct, streaming on sorted
#   storage, and it agrees with `pandas.merge_asof` row for row.
# - Direction and tolerance are first-class: `'forward'` fetched the
#   5-minutes-later mid for realized spread; a microsecond tolerance turned
#   stale-quote marking into visible, countable NULLs. Colliding right
#   columns get `_right`.
# - The join composes with the rest of SQL — Lee-Ready signing and the full
#   quoted/effective/realized decomposition are each one statement (CTEs +
#   `lag()` + `CASE`).
# - Scoring against ground truth requires a tape where trades and quotes
#   share one price process — worth remembering before benchmarking signing
#   accuracy on any synthetic data.

# %%
db.close()
