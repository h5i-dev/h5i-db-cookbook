# %% [markdown]
# # クラッシュに強いペーパートレードのループと、完全な発注根拠
#
# ライブのループで難しいのは戦略ではありません。1週間後に*「なぜあの注文を出したのか」*に
# 答えることです。このレシピでは、各工程がバージョン付きのコミットになる、逐次的で決定的な
# ペーパートレードのループを作ります。フィードのチャンク1つが `trades` への `append` 1回、
# シグナルはその先頭に対して SQL で計算し、注文の各行は**それを生んだ trades のバージョン
# （シーケンス番号）を保存します**。根拠の追跡は鑑識作業ではなくジョインになりますし、
# `h5i('trades', v)` が O(1) のタイムトラベルなので、どの注文についてもシグナルの入力を
# そのまま再生できます。
#
# クラッシュへの強さはただで付いてきます。どのコミットもアトミックなマニフェストの差し替え
# なので、書き込みの途中で `kill -9` されても、直前の先頭は完全に整合したまま残ります。ループは
# 「フィードがどこで止まったか」をデータベースに尋ねて再開します。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star, sql_expr, time_bucket

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_paper"), create=True)

# %% [markdown]
# ## 1. テーブル: フィード、注文、ポジション。すべて追記のみ
#
# 追記のみのテーブルが3つ。追記のみは趣味の問題ではありません。バージョン連鎖が `tail()` の
# ストリーミング読み出しと互換に保たれますし、注文の記録が作りからして改竄検知可能になります。

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
# ## 2. フィードとシグナル
#
# 2銘柄について1セッションぶんのティックデータを、5分ごとの納品に刻みます。ライブのフィード
# ハンドラの、決定的な代役です。シグナルは1分足終値に対する速い／遅い EWMA のクロスオーバーで、
# その瞬間の `trades` の先頭に対して、すべてデータベース側で計算します。`time_bucket` が足を作り、
# `ewma` のウィンドウ関数がそれを平滑化し、`row_number()` が銘柄ごとの最新の足を選びます。

# %%
feed = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=20_000).to_pandas()
feed["chunk"] = feed["ts"].dt.floor("5min")
chunks = list(feed.groupby("chunk", sort=True))
print(f"{len(feed):,} ticks -> {len(chunks)} five-minute deliveries")

def signal_frame(version=None):
    """Latest fast/slow EWMA crossover per symbol, at any read point."""
    bars = (
        db.table("trades", version=version)
        .group_by(time_bucket("1m", col("ts")).alias("bar"), "symbol")
        .agg(close=col("price").last("ts"))
    )
    return (
        bars.with_columns(
            fast=col("close").ewma(0.30, order_by="bar", partition_by="symbol"),
            slow=col("close").ewma(0.05, order_by="bar", partition_by="symbol"),
        )
        .with_columns(
            rn=sql_expr("row_number()").over(
                partition_by="symbol", order_by="bar", descending=True
            )
        )
        .filter(col("rn") == 1)
        .select("symbol", "close", "fast", "slow")
        .sort("symbol")
    )

# %% [markdown]
# ## 3. イベントループ: チャンクを append → シグナル → 注文を append → 評価
#
# 逐次的で決定的です。スレッドもスリープもありません。チャンクごとに、ティックのコミットが1回、
# その先頭に対するシグナルのクエリが1回、注文のコミットが最大1回（1行ずつ打たず、バッチで）、
# 評価のコミットが1回です。`append` が返すコミット辞書から、各注文に刻むシーケンス番号を受け
# 取ります。ロング／フラットの論理は、速い線が遅い線の上なら100株保有、そうでなければフラット。

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

    sig = signal_frame().to_pandas().set_index("symbol")
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

n_orders = db.table("orders").select(count_star().alias("n")).to_pandas()["n"].iloc[0]
print(f"loop done: {len(chunks)} chunks, {n_orders} orders, "
      f"final positions {pos}, final equity {equity_track[-1][1]:,.2f} USD")

# %% [markdown]
# ## 4. 再起動からの復帰: データベースがチェックポイント
#
# ループのどこでプロセスが死んでも、途中の状態は存在しません。どのコミットも完全に起きたか、
# 起きなかったかのどちらかです。再起動時の再開点はクエリで求まります。それ自体が古くなって
# いるかもしれないチェックポイントファイルは要りません。

# %%
resume = db.table("trades").select(
    last_tick=col("ts").max(), rows_ingested=count_star()
).to_pandas()
print("on restart, continue the feed after:")
resume

# %% [markdown]
# ## 5. 発注根拠: シグナルの入力をそのまま再生する
#
# どの注文も `data_version` を持っています。1件を監査するには、*同じ*シグナルのフレームを
# そのバージョンに向けるだけです。ループが見た先頭への O(1) のタイムトラベルで、
# 判断を確認できます。「シグナルは買いと言っていたはずだ」と「これがそのシグナルです。当時の
# バイトから計算し直して、買いと言っています」の違いがここにあります。

# %%
last_order = (
    db.table("orders")
    .select("ts", "symbol", "side", "qty", "price", "data_version")
    .sort("ts", descending=True)
    .limit(1)
    .to_pandas()
    .iloc[0]
)
print("auditing order:", dict(last_order))

replayed = (
    signal_frame(version=int(last_order["data_version"]))
    .to_pandas()
    .set_index("symbol")
)
row = replayed.loc[last_order["symbol"]]
replayed_desired = TARGET if row["fast"] > row["slow"] else 0

held_after = (
    db.table("positions")
    .filter(col("symbol") == last_order["symbol"], col("ts") == last_order["ts"].isoformat())
    .select("position")
    .to_pandas()["position"]
    .iloc[0]
)

print(f"replayed signal at v{int(last_order['data_version'])}: "
      f"fast={row['fast']:.4f} slow={row['slow']:.4f} -> desired position {replayed_desired}")
assert replayed_desired == held_after, "replayed decision must match the recorded position"
print(f"recorded position after order: {held_after} - attribution verified")

# %% [markdown]
# ## 6. 読み手の側: `tail()` で新しい注文を流し込む
#
# 下流の利用者（リスク検査、執行ブリッジ）は `tail('orders', after_version, poll_ms)` で注文簿を
# 追います。本番で守るべき規則が2つ。`tail` は**追記のみ**のバージョン連鎖を要求します（範囲内に
# write／delete／restore／compact が1つでもあるとエラーになります。注文簿を追記のみにしている
# 理由の1つです）。そして`tail` は**終わりがありません**。`LIMIT` の行数が届くまでブロックするので、
# 必ず、そこにあると分かっている以下の LIMIT か、クエリのタイムアウトを添えてください。

# %%
overs = db.versions("orders")
after = overs[max(0, len(overs) - 6)]          # start ~5 commits behind head
available = overs[-1]["rows"] - after["rows"]
n = int(min(10, available))
print(f"tailing orders after v{after['sequence']} ({available} new rows, reading {n}):")
db.sql(f"SELECT * FROM tail('orders', {after['sequence']}, 50) LIMIT {n}",
       timeout=30).to_pandas()

# %% [markdown]
# ## 7. セッションの後片付け: エクイティカーブと注文簿の衛生管理
#
# エクイティカーブは記録した評価から作ります。そのあとメンテナンスです。5分ごとのコミットを1日
# 続けると、テーブルあたり約78個の小さなセグメントが残るので、`compact()` で併合します。これも
# それ自体が監査対象のコミットです。（トレードオフに注意してください。compact のコミットは、
# それをまたぐ範囲について `tail()` の追記のみの連鎖を切ります。だから利用者の読み取り位置より
# *後ろ*で compact してください。）

# %%
import matplotlib.pyplot as plt

eq = pd.DataFrame(equity_track, columns=["ts", "equity"]).set_index("ts")
posn = (
    db.table("positions")
    .select("ts", "symbol", "position")
    .sort("ts")
    .to_pandas()
    .pivot(index="ts", columns="symbol", values="position")
)

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
# ## まとめ
#
# - チャンク1つ＝コミット1回。そのコミットのシーケンス番号を注文の各行に刻めば、根拠の追跡は
#   ジョインに、再生は `h5i('trades', v)` のクエリになります。
# - アトミックなコミットがクラッシュへの強さそのものです。書き込み途中の `kill -9` は直前の
#   先頭を無傷で残し、再開点は `SELECT max(ts)` で求まります。データベースがチェックポイントです。
# - 注文簿は追記のみに保ってください。下流の利用者にとって `tail()` 互換であり続けますし、監査
#   担当にとっては改竄検知可能なままです。`tail()` の読み出しには必ず LIMIT を（できればタイム
#   アウトも）付けてください。行がそろうまでブロックします。
# - 注文はサイクルごとにまとめてコミットし（1行ずつは禁物です）、セッション後に――読み手より
#   後ろで――`compact()` してその日の小さなセグメントを併合します。

# %%
db.close()
