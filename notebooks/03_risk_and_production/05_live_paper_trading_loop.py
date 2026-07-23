# %% [markdown]
# # A crash-safe paper-trading loop with full order attribution
#
# The hard part of a live loop is not the strategy - it is answering, a week
# later, *"why did we send that order?"*. This recipe builds a sequential,
# deterministic paper-trading loop where every stage is a versioned commit:
# each feed chunk is one `append` to `trades`, the signal is computed in SQL
# on exactly that head, and every order row **stores the trades version
# (sequence number) that produced it**. Attribution stops being forensics
# and becomes a join - and because `h5i('trades', v)` is O(1) time travel,
# you can replay any order's exact signal input on demand.
#
# Crash safety comes for free: every commit is an atomic manifest swap, so a
# `kill -9` mid-write leaves the previous head fully consistent - the loop
# restarts by asking the database where the feed stopped.

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_paper"), create=True)

# %% [markdown]
# ## 1. Tables: feed, orders, positions - all append-only
#
# Three pure-append tables. Append-only is not just a style choice: it keeps
# the version chain compatible with `tail()` streaming reads, and it means
# the order blotter is tamper-evident by construction.

# %%
trade_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
order_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("side", pa.string()),
        pa.field("qty", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("data_version", pa.int64()),   # trades sequence that produced this order
    ]
)
position_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("position", pa.int64()),
        pa.field("price", pa.float64()),
        pa.field("mkt_value", pa.float64()),
    ]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("orders", order_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("positions", position_schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %% [markdown]
# ## 2. The feed and the signal
#
# One session of tick data for two names, chunked into 5-minute deliveries -
# a deterministic stand-in for a live feed handler. The signal is a fast/slow
# EWMA crossover on 1-minute closes, computed entirely in SQL against
# whatever the `trades` head is at that moment: `time_bucket` builds the
# bars, `ewma` window functions smooth them, `row_number()` picks the latest
# bar per symbol.

# %%
feed = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=20_000).to_pandas()
feed["chunk"] = feed["ts"].dt.floor("5min")
chunks = list(feed.groupby("chunk", sort=True))
print(f"{len(feed):,} ticks -> {len(chunks)} five-minute deliveries")

def signal_sql(relation: str) -> str:
    return f"""
    WITH bars AS (
        SELECT time_bucket('1m', ts) AS bar, symbol,
               last_value(price ORDER BY ts) AS close
        FROM {relation}
        GROUP BY bar, symbol
    ), sig AS (
        SELECT bar, symbol, close,
               ewma(close, 0.30) OVER (PARTITION BY symbol ORDER BY bar) AS fast,
               ewma(close, 0.05) OVER (PARTITION BY symbol ORDER BY bar) AS slow
        FROM bars
    )
    SELECT symbol, close, fast, slow
    FROM (SELECT *, row_number() OVER (PARTITION BY symbol ORDER BY bar DESC) AS rn FROM sig)
    WHERE rn = 1
    ORDER BY symbol
    """

# %% [markdown]
# ## 3. The event loop: append chunk → SQL signal → append orders → mark
#
# Sequential and deterministic - no threads, no sleeps. Per chunk: one
# commit for the ticks, one signal query on that exact head, at most one
# commit for orders (a batch, never per-row), one for the marks. The commit
# dict returned by `append` hands us the sequence number we stamp on each
# order. Long/flat logic: fast above slow → hold 100 shares, else flat.

# %%
TARGET = 100
pos = {"AAPL": 0, "MSFT": 0}
cash = 0.0
equity_track = []

for chunk_ts, chunk in chunks:
    data = pa.Table.from_pandas(
        chunk.drop(columns="chunk").sort_values(["ts", "symbol"]),
        schema=trade_schema, preserve_index=False,
    )
    commit = db.append("trades", data)
    seq = commit["sequence"]

    sig = db.sql(signal_sql("trades")).to_pandas().set_index("symbol")
    mark_ts = chunk_ts + pd.Timedelta(minutes=5)

    order_rows = []
    for sym, row in sig.iterrows():
        desired = TARGET if row["fast"] > row["slow"] else 0
        delta = desired - pos[sym]
        if delta != 0:
            order_rows.append(
                {"ts": mark_ts, "symbol": sym, "side": "BUY" if delta > 0 else "SELL",
                 "qty": abs(delta), "price": row["close"], "data_version": seq}
            )
            cash -= delta * row["close"]
            pos[sym] = desired
    if order_rows:
        db.append("orders", pa.Table.from_pandas(
            pd.DataFrame(order_rows).sort_values(["ts", "symbol"]),
            schema=order_schema, preserve_index=False))

    marks = pd.DataFrame(
        {"ts": mark_ts, "symbol": list(sig.index),
         "position": [pos[s] for s in sig.index],
         "price": sig["close"].to_numpy(),
         "mkt_value": [pos[s] * sig.loc[s, "close"] for s in sig.index]}
    )
    db.append("positions", pa.Table.from_pandas(
        marks.sort_values(["ts", "symbol"]), schema=position_schema, preserve_index=False))
    equity_track.append((mark_ts, cash + marks["mkt_value"].sum()))

n_orders = db.sql("SELECT count(*) AS n FROM orders").to_pandas()["n"].iloc[0]
print(f"loop done: {len(chunks)} chunks, {n_orders} orders, "
      f"final positions {pos}, final equity {equity_track[-1][1]:,.2f} USD")

# %% [markdown]
# ## 4. Restart recovery: the database is the checkpoint
#
# If the process dies anywhere in the loop, no partial state exists - each
# commit either fully happened or didn't. On restart, the resume point is a
# query, not a checkpoint file that might itself be stale:

# %%
resume = db.sql(
    "SELECT max(ts) AS last_tick, count(*) AS rows_ingested FROM trades"
).to_pandas()
print("on restart, continue the feed after:")
resume

# %% [markdown]
# ## 5. Order attribution: replay the exact signal input
#
# Every order carries `data_version`. To audit one, point the *same* signal
# SQL at `h5i('trades', <version>)` - O(1) time travel to the head the loop
# saw - and confirm the decision. This is the difference between "we think
# the signal said buy" and "here is the signal, recomputed from the exact
# bytes, saying buy".

# %%
last_order = db.sql(
    "SELECT ts, symbol, side, qty, price, data_version FROM orders ORDER BY ts DESC LIMIT 1"
).to_pandas().iloc[0]
print("auditing order:", dict(last_order))

replayed = db.sql(
    signal_sql(f"h5i('trades', {int(last_order['data_version'])})")
).to_pandas().set_index("symbol")
row = replayed.loc[last_order["symbol"]]
replayed_desired = TARGET if row["fast"] > row["slow"] else 0

held_after = db.sql(
    f"""
    SELECT position FROM positions
    WHERE symbol = '{last_order["symbol"]}' AND ts = '{last_order["ts"].isoformat()}'
    """
).to_pandas()["position"].iloc[0]

print(f"replayed signal at v{int(last_order['data_version'])}: "
      f"fast={row['fast']:.4f} slow={row['slow']:.4f} -> desired position {replayed_desired}")
assert replayed_desired == held_after, "replayed decision must match the recorded position"
print(f"recorded position after order: {held_after} - attribution verified")

# %% [markdown]
# ## 6. The reader side: streaming new orders with `tail()`
#
# A downstream consumer (risk checks, an execution bridge) follows the order
# blotter with `tail('orders', after_version, poll_ms)`. Two production
# rules: `tail` requires a **pure-append** version chain (any write/delete/
# restore/compact in the range errors out - one reason our blotter is
# append-only), and it is **unbounded** - it blocks until `LIMIT` rows have
# arrived, so always attach a LIMIT no larger than what you know is there,
# or a query timeout.

# %%
overs = db.versions("orders")
after = overs[max(0, len(overs) - 6)]          # start ~5 commits behind head
available = overs[-1]["rows"] - after["rows"]
n = int(min(10, available))
print(f"tailing orders after v{after['sequence']} ({available} new rows, reading {n}):")
db.sql(f"SELECT * FROM tail('orders', {after['sequence']}, 50) LIMIT {n}",
       timeout=30).to_pandas()

# %% [markdown]
# ## 7. Session wrap-up: equity curve and blotter hygiene
#
# The equity curve comes from the recorded marks. Then maintenance: a day of
# 5-minute commits leaves ~78 small segments per table, so `compact()` merges
# them - as its own audited commit. (Note the trade-off: a compact commit
# breaks the pure-append chain for `tail()` ranges that span it, so compact
# *behind* your consumers' read positions.)

# %%
import matplotlib.pyplot as plt

eq = pd.DataFrame(equity_track, columns=["ts", "equity"]).set_index("ts")
posn = db.sql(
    "SELECT ts, symbol, position FROM positions ORDER BY ts"
).to_pandas().pivot(index="ts", columns="symbol", values="position")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(eq.index, eq["equity"], lw=1.2, color="tab:blue")
ax1.axhline(0, color="0.7", lw=0.6)
ax1.set_title("Paper strategy equity (marked every 5 minutes)")
ax1.set_ylabel("P&L (USD)")
for sym in posn.columns:
    ax2.step(posn.index, posn[sym], where="post", label=sym, lw=1.0)
ax2.set_title("Positions (shares)")
ax2.set_xlabel("time (UTC)")
ax2.set_ylabel("shares")
ax2.legend(fontsize=8)
fig.tight_layout()

# %%
for t in ("trades", "orders", "positions"):
    before = db.versions(t)[-1]["segments"]
    c = db.compact(t, note="post-session compaction")
    print(f"{t}: {before} segments -> {c['segments_total']} (v{c['sequence']}, op={c['op']})")

# %% [markdown]
# ## Takeaways
#
# - One chunk = one commit; the commit's sequence number stamped into each
#   order row makes attribution a join and replay a `h5i('trades', v)` query.
# - Atomic commits are the crash-safety story: `kill -9` mid-write leaves the
#   previous head intact, and the restart point is `SELECT max(ts)` - the
#   database is the checkpoint.
# - Keep the blotter pure-append: it stays `tail()`-compatible for
#   downstream consumers and tamper-evident for auditors. Always LIMIT (and
#   ideally timeout) a `tail()` read - it blocks until enough rows arrive.
# - Batch orders per cycle (never per-row commits), and `compact()` after the
#   session - behind your readers - to merge the day's small segments.

# %%
db.close()
