# %% [markdown]
# # Order flow imbalance: does signed volume predict returns?
#
# Order flow imbalance is the excess of buyer-initiated over seller-initiated
# volume, and it is the workhorse microstructure signal. It correlates strongly
# with contemporaneous price moves and, in real markets at short horizons, is
# mildly predictive of the next move.
#
# In this recipe we build both flavours:
#
# 1. **trade OFI**: audit signing rules against ground truth, using Lee-Ready
#    via an ASOF join to the prevailing quote with the tick test as fallback,
#    then sign the tape and bucket signed volume per minute,
# 2. **quote OFI** (Cont, Kukanov and Stoikov 2014): order-book pressure from
#    changes in best bid and ask prices and sizes, via `lag()` windows,
# 3. an honest look at contemporaneous versus *predictive* correlation with
#    1-minute returns.

# %% [markdown]
# ## Terms used here
#
# | term                 | meaning |
# | -------------------- | --- |
# | order flow imbalance | buyer-initiated volume minus seller-initiated volume over a window |
# | buyer-initiated      | the buyer crossed the spread to make the trade happen; feeds rarely say, so it is inferred |
# | signed volume        | trade volume carrying that sign |
# | Lee-Ready            | the standard signing rule: compare the trade price to the prevailing mid |
# | tick test            | the fallback rule: compare the trade to the previous trade price |
# | quote OFI            | book pressure from changes in best bid and ask prices and sizes |
# | contemporaneous      | measured over the same window as the return, so it explains rather than predicts |
# | predictive           | measured over the window before the return, which is the only useful kind |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_ofi"), create=True)

# %% [markdown]
# ## 1. The data
#
# Two sessions of top-of-book quotes from `cu.make_quotes`, one row per change
# in the best bid or offer.
#
# | column | type | meaning |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | quote timestamp, ascending |
# | `symbol` | `string` | ticker, `AAPL` or `MSFT` |
# | `bid`, `ask` | `float64` | best bid and offer |
# | `bid_size`, `ask_size` | `int64` | displayed depth at each side |

# %%
quotes = cu.make_quotes(symbols=["AAPL", "MSFT"], days=2)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# One generator caveat shapes what follows. `make_trades` prints follow a random
# walk only loosely coupled to `make_quotes` mids, which would make quote-based
# trade signing meaningless.
#
# So we keep the synthetic quote stream and *print our own tape off it*, the way
# a matching engine would. A random 25% of quote updates trigger a trade 1 to 50
# ms later, buyers lift the offer, sellers hit the bid, and the true aggressor
# side is recorded.
#
# The tape is price-consistent with the quotes by construction, so every signing
# rule below can be scored against ground truth. That is a luxury real data
# never affords. The printed tape has columns `ts`, `symbol`, `price`, `size`
# and `side`.

# %%
q = quotes.to_pandas()

rng = np.random.default_rng(21)
picks = q[rng.random(len(q)) < 0.25].copy()
side = rng.choice(np.array([1, -1]), size=len(picks))
picks["price"] = np.where(side > 0, picks["ask"], picks["bid"])
picks["ts"] = (
    picks["ts"] + pd.to_timedelta(rng.integers(1_000, 50_000, len(picks)), unit="us")
).astype("datetime64[us, UTC]")
picks["size"] = np.maximum(100, (rng.lognormal(4.0, 1.2, len(picks)) // 100 * 100)).astype("int64")
picks["side"] = np.where(side > 0, "B", "S")

trade_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("side", pa.string()),
    ]
)
trades_df = picks[["ts", "symbol", "price", "size", "side"]].sort_values(["ts", "symbol"])
trades = pa.Table.from_pandas(trades_df, preserve_index=False).cast(trade_schema)
print(f"{len(trades_df):,} trades printed off {len(q):,} quotes")
trades_df.head()

# %%
db.create_table("quotes", quotes.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes, note="2-day synthetic NBBO")
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades, note="tape printed off the quote stream")
print({t: len(db.read(t)) for t in db.tables()})

# %% [markdown]
# ## 2. Audit the signing rules on a QA window
#
# Lee-Ready says: above the prevailing mid is a buy, below is a sell, and at the
# mid you fall back to the tick test, where an uptick is a buy.
#
# The prevailing-quote lookup is `asof_join(..., 'backward', 5_000_000)`: the
# latest quote at most 5 seconds old, per symbol, with the tolerance in raw
# microseconds.
#
# We run the audit on a midday 15-minute window, carved out with time-ranged
# `db.read` calls using raw microsecond bounds, pruned at the segment level, and
# stored as small workbench tables. A windowed audit is fast to iterate on and
# easy to eyeball.
#
# We assert the join returns exactly one row per trade, and cross-check the
# attached quotes against `pandas.merge_asof`.

# %%
w0 = int(pd.Timestamp("2026-06-01 16:00", tz="UTC").value // 1000)
w1 = int(pd.Timestamp("2026-06-01 16:15", tz="UTC").value // 1000)

trades_w = db.read("trades", time_start=w0, time_end=w1)
quotes_w = db.read("quotes", time_start=w0 - 60_000_000, time_end=w1)  # 60s run-up
db.create_table("trades_qa", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades_qa", trades_w, note="midday QA window")
db.create_table("quotes_qa", quotes.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_qa", quotes_w, note="midday QA window + run-up")
print(f"QA window: {len(trades_w):,} trades, {len(quotes_w):,} quotes")

def sign_of(x, prev):
    """+1 uptick, -1 downtick, 0 unchanged - the tick test."""
    return when(x > prev).then(lit(1)).when(x < prev).then(lit(-1)).otherwise(lit(0))


PREV_PX = sql_expr("lag(price)").over(partition_by="symbol", order_by="ts")
TRUE_SIGN = when(col("side") == "B").then(lit(1)).otherwise(lit(-1))

audit = (
    db.table("trades_qa")
    .join_asof(db.table("quotes_qa"), on="ts", by="symbol", tolerance=5_000_000)
    .select("ts", "symbol", "price", "size", "side",
            mid=(col("bid") + col("ask")) / 2)
    .with_columns(prev_px=PREV_PX)
    .select(
        "ts", "symbol", "mid",
        lr_sign=when(col("price") > col("mid")).then(lit(1))
        .when(col("price") < col("mid")).then(lit(-1))
        .when(col("price") > col("prev_px")).then(lit(1))
        .when(col("price") < col("prev_px")).then(lit(-1))
        .otherwise(lit(0)),
        tick_sign=sign_of(col("price"), col("prev_px")),
        true_sign=TRUE_SIGN,
    )
    .sort(["ts", "symbol"])
    .to_pandas()
)
assert len(audit) == len(trades_w), "asof_join must return one row per trade"

# cross-check the ASOF lookup itself against pandas.merge_asof
ref = pd.merge_asof(
    trades_w.to_pandas().sort_values("ts"),
    quotes_w.to_pandas().sort_values("ts").assign(mid_ref=lambda d: (d["bid"] + d["ask"]) / 2),
    on="ts",
    by="symbol",
    tolerance=pd.Timedelta("5s"),
)
chk = audit.sort_values("ts").reset_index(drop=True)
assert np.allclose(chk["mid"], ref["mid_ref"], equal_nan=True), "asof mismatch vs pandas"
print("asof_join validated against pandas.merge_asof")

for rule in ("lr_sign", "tick_sign"):
    scored = audit[audit[rule] != 0]
    acc = (scored[rule] == scored["true_sign"]).mean()
    print(f"  {rule:9s}: {acc:.1%} accuracy vs true aggressor side ({len(scored):,} classified)")

# %% [markdown]
# On a price-consistent tape, Lee-Ready earns its reputation. The quote rule is
# nearly perfect, with errors coming from quotes that updated in the 1 to 50 ms
# between quote and print, while the pure tick test trails well behind. That is
# the same ordering seen on real TAQ, where Lee-Ready scores around 85%.
#
# ## 3. Sign the full tape
#
# The audited quote rule would need the trade-quote ASOF at full scale. For this
# recipe the tick test is accurate enough to carry the OFI study, and it is a
# pure `lag()` window over `trades` with no alignment needed. We keep the
# generator's true side as an oracle upper bound.
#
# The signed tape is persisted as a first-class table. Signing is expensive
# enough that you want it done once, committed and versioned.

# %%
signed = (
    db.table("trades")
    .with_columns(prev_px=PREV_PX)
    .select(
        "ts", "symbol", "price", "size",
        tick_sign=sign_of(col("price"), col("prev_px")),
        true_sign=TRUE_SIGN,
    )
    .sort(["ts", "symbol"])
    .to_arrow()
)

full = signed.to_pandas()
scored = full[full["tick_sign"] != 0]
print(f"full tape: {(scored['tick_sign'] == scored['true_sign']).mean():.1%} tick-test accuracy on {len(scored):,} prints")

# %%
signed_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("tick_sign", pa.int32()),
        pa.field("true_sign", pa.int32()),
    ]
)
db.create_table("signed_trades", signed_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("signed_trades", signed.cast(signed_schema), note="tick-test + oracle signs")

# %% [markdown]
# ## 4. Trade OFI per minute
#
# One `time_bucket` aggregation gives signed buy and sell volume, the normalized
# imbalance `(buys - sells) / total`, and the bucket's closing print via the
# `last_value(... ORDER BY ts)` idiom.

# %%
BAR = time_bucket("1m", col("ts"))
size_when = lambda flag: when(flag).then(col("size")).otherwise(lit(0)).sum()

trade_ofi = (
    db.table("signed_trades")
    .group_by(BAR.alias("bar"), "symbol")
    .agg(
        buy_vol=size_when(col("tick_sign") > 0),
        sell_vol=size_when(col("tick_sign") < 0),
        oracle_net=when(col("true_sign") > 0).then(col("size")).otherwise(-col("size")).sum(),
        volume=col("size").sum(),
        close=col("price").last("ts"),
    )
    .sort(["bar", "symbol"])
    .to_pandas()
)
trade_ofi["ofi"] = (trade_ofi["buy_vol"] - trade_ofi["sell_vol"]) / trade_ofi["volume"]
trade_ofi["ofi_oracle"] = trade_ofi["oracle_net"] / trade_ofi["volume"]
trade_ofi.head(4)

# %% [markdown]
# ## 5. Quote OFI (Cont et al.)
#
# Book-pressure OFI needs *changes* at the best. A bid price rising, or size
# growing at an unchanged bid, adds buy pressure. The mirror on the ask side
# subtracts.
#
# That is four `lag()` comparisons per quote, pure SQL window machinery, summed
# per minute, with the bucket's closing mid for returns.

# %%
def prev(name: str):
    return sql_expr(f"lag({name})").over(partition_by="symbol", order_by="ts")


# Cont et al.: size added at the bid counts positive, size pulled counts
# negative, mirrored on the ask. Four lagged comparisons per quote.
keep = lambda flag, x: when(flag).then(x).otherwise(lit(0))

quote_ofi = (
    db.table("quotes")
    .with_columns(pb=prev("bid"), pa=prev("ask"), pbs=prev("bid_size"), pas=prev("ask_size"))
    .filter(col("pb").is_not_null())
    .group_by(BAR.alias("bar"), "symbol")
    .agg(
        qofi=(
            keep(col("bid") >= col("pb"), col("bid_size"))
            - keep(col("bid") <= col("pb"), col("pbs"))
            - keep(col("ask") <= col("pa"), col("ask_size"))
            + keep(col("ask") >= col("pa"), col("pas"))
        ).sum(),
        mid_close=((col("bid") + col("ask")) / 2).last("ts"),
    )
    .sort(["bar", "symbol"])
    .to_pandas()
)
quote_ofi.head(4)

# %% [markdown]
# ## 6. Contemporaneous vs predictive
#
# Here is the question that matters. Does this minute's OFI tell you about
# *this* minute's return, which is mechanical, or the *next* minute's, which
# would be alpha?
#
# We compute both correlations per symbol, per estimator. Returns are
# within-session only, with the `shift` taken inside each symbol and day, so
# there are no overnight artifacts.

# %%
def bucket_corrs(frame, ofi_col, px_col):
    rows = []
    frame = frame.copy()
    frame["day"] = frame["bar"].dt.date
    for (sym, _), g in frame.groupby(["symbol", "day"]):
        g = g.sort_values("bar")
        frame.loc[g.index, "ret"] = g[px_col].pct_change()
        frame.loc[g.index, "ret_next"] = g[px_col].pct_change().shift(-1)
    for sym, g in frame.groupby("symbol"):
        g = g.dropna(subset=["ret", "ret_next"])
        rows.append(
            {
                "symbol": sym,
                "n_buckets": len(g),
                "corr_same_min": g[ofi_col].corr(g["ret"]),
                "corr_next_min": g[ofi_col].corr(g["ret_next"]),
            }
        )
    return pd.DataFrame(rows)

results = pd.concat(
    [
        bucket_corrs(trade_ofi, "ofi", "close").assign(estimator="trade OFI (tick test)"),
        bucket_corrs(trade_ofi, "ofi_oracle", "close").assign(estimator="trade OFI (oracle)"),
        bucket_corrs(quote_ofi, "qofi", "mid_close").assign(estimator="quote OFI (Cont)"),
    ]
)[["estimator", "symbol", "n_buckets", "corr_same_min", "corr_next_min"]]
results.round(3)

# %%
import matplotlib.pyplot as plt

sym = "AAPL"
g = trade_ofi[trade_ofi["symbol"] == sym].copy()
g["day"] = g["bar"].dt.date
g["ret"] = g.groupby("day")["close"].pct_change()
g = g.dropna(subset=["ret"])

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(g["ofi"], g["ret"] * 1e4, s=8, alpha=0.35, color="tab:blue")
b, a = np.polyfit(g["ofi"], g["ret"] * 1e4, 1)
xs = np.linspace(-1, 1, 50)
ax.plot(xs, a + b * xs, color="tab:red", lw=1.5, label=f"OLS slope = {b:.1f} bps per unit OFI")
ax.set_title(f"{sym}: 1-minute trade OFI vs same-minute return")
ax.set_xlabel("OFI = (buy vol - sell vol) / total vol")
ax.set_ylabel("1-minute return (bps)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# Three honest lessons in one table.
#
# - **Quote OFI** shows the textbook contemporaneous link, around 0.5 to 0.6.
#   Book pressure and same-minute mid moves are near-mechanically related.
# - **Trade OFI with tick-test signs** also reads about 0.5 contemporaneous. The
#   **oracle** row exposes most of that as circular. With *true* aggressor signs
#   the correlation drops to roughly 0.07: in this market the mid is an
#   exogenous random walk, flow has no price impact, and true signed volume
#   relates to the minute's return only through the bounce of the closing print.
#   The tick rule manufactures the rest by deriving signs from the very price
#   changes we then correlate against. On real data, where flow does move price,
#   the oracle-equivalent number would be solidly positive. The comparison is
#   how you learn to ask.
# - **Predictive** correlations sit near zero with a negative tilt for trade OFI.
#   Bid-ask bounce means a buy-heavy minute tends to close at the ask and
#   mean-revert a touch. Even perfect signs would not help, because there is no
#   flow-to-future-price causality here to find, and the pipeline correctly
#   finds none.
#
# On real tick data quote OFI does carry short-horizon predictive power. The
# reality check still applies before calling it alpha: the effect lives at
# seconds-to-minutes horizons in size-constrained depth, and capturing a 1 bp
# move costs you the spread plus fees. OFI signals are best understood as
# execution-layer inputs, deciding when to cross and when to post, not as
# standalone strategies.
#
# ## Takeaways
#
# - `asof_join(..., 'backward', tolerance)` is the prevailing-quote lookup, and
#   time-ranged `db.read(time_start=, time_end=)` carves QA windows in
#   O(window). One current-build caveat: keep both ASOF inputs within one
#   storage batch, roughly 8k rows, and assert one output row per left row,
#   because larger inputs truncate silently. Cross-checking against
#   `pandas.merge_asof` costs three lines and buys certainty.
# - Signing rules are CASE expressions over that join plus `lag()` windows.
#   Persisting the result as a `signed_trades` table means the pass runs once,
#   and every downstream study reads versioned, committed signs.
# - Both OFI flavours reduce to one `time_bucket` GROUP BY each, with
#   `last_value(... ORDER BY ts)` giving bucket closes without self-joins.
# - Honest nulls matter. Solid contemporaneous correlations alongside predictive
#   correlations near zero, even with oracle signs, is the signature of a
#   mechanical rather than exploitable relationship.

# %%
db.close()
