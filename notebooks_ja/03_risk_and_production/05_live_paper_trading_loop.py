# %% [markdown]
# # クラッシュに強いペーパートレードのループと、注文の完全な帰属
#
# ライブのループで難しいのは戦略ではありません。1週間後に*「なぜあの注文を出したのか」*に
# 答えることです。
#
# このレシピは、どの段階もバージョン管理されたコミットになる、逐次的で決定的なペーパートレード
# のループを組み立てます。フィードのチャンク1つが `trades` への `append` 1件、シグナルはまさに
# その先頭に対して SQL で計算し、注文の各行が**それを生んだ trades のバージョンを保持**します。
#
# 帰属の追跡が鑑識作業ではなくジョインになります。そして `h5i('trades', v)` が O(1) の
# タイムトラベルなので、どの注文についても、そのシグナルの入力をそのまま再生できます。
#
# クラッシュ耐性はただで付いてきます。どのコミットもアトミックなマニフェストの差し替えなので、
# 書き込みの途中で `kill -9` が来ても直前の先頭は完全に整合したまま残り、ループはフィードが
# どこで止まったかをデータベースに聞いて再開します。

# %% [markdown]
# ## ここで使う用語
#
# | 用語         | 意味 |
# | ---------- | --- |
# | ペーパートレード   | 実際の注文を出さずに、ライブのループ全体を動かすこと |
# | アトリビューション  | どの入力がどの注文を生んだのかを、後から答えられるようにすること |
# | `tail`     | 指定したバージョンより後にコミットされた行だけを読む |
# | ハイウォーターマーク | ループが処理し終えた最後のバージョン。再開のために保存しておく |
# | クラッシュ耐性    | コミットはマニフェストのアトミックな差し替えなので、書き込み中に落ちても直前のヘッドが残る |
# | タイムトラベル    | 過去のバージョンの姿でテーブルを読むこと。ここでは注文の入力を再現するのに使う |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star, sql_expr, time_bucket

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_paper"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の、2銘柄・1セッションぶんのティックデータです。1行が1約定です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード、`AAPL` か `MSFT` |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |
#
# これを5分ごとの配信に切り分けます。ライブのフィードハンドラの、決定的な代役です。

# %%
feed = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=20_000).to_pandas()
print(f"{len(feed):,} rows x {feed.shape[1]} columns")
feed.head()

# %% [markdown]
# ## 2. テーブル: フィード、注文、ポジション、すべて追記専用
#
# 追記だけの3つのテーブルです。追記専用は好みの問題ではありません。バージョンの連鎖を
# `tail()` のストリーミング読み取りと互換に保ちますし、注文の記録を作りからして改竄が見える
# ものにします。

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
# ## 3. シグナル
#
# シグナルは1分足の終値に対する速い／遅い EWMA のクロスオーバーで、その時点の `trades` の先頭に
# 対して、すべてデータベースの中で計算します。
#
# `time_bucket` がバーを作り、`ewma` のウィンドウ関数が平滑化し、`row_number()` が銘柄ごとに
# 最新のバーを選びます。

# %%
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
# ## 4. イベントループ: チャンクを追記、シグナル、注文を追記、評価
#
# 逐次的で決定的です。スレッドもスリープもありません。
#
# チャンクごとに、ティックのコミットが1件、まさにその先頭に対するシグナルのクエリが1回、注文の
# コミットが多くても1件（行ごとではなくバッチで）、そして評価のコミットが1件です。`append` が
# 返すコミット辞書が、各注文に刻むシーケンス番号を渡してくれます。
#
# ロング／フラットのロジックは単純です。速いほうが遅いほうより上なら100株持ち、それ以外は
# フラット。

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
# ## 5. 再起動からの復旧: データベースがチェックポイント
#
# ループのどこでプロセスが死んでも、中途半端な状態は存在しません。どのコミットも、完全に起きたか
# 起きなかったかのどちらかです。
#
# 再起動時の再開点はクエリです。それ自体が古くなっているかもしれないチェックポイントファイル
# ではありません。

# %%
resume = db.table("trades").select(
    last_tick=col("ts").max(), rows_ingested=count_star()
).to_pandas()
print("on restart, continue the feed after:")
resume

# %% [markdown]
# ## 6. 注文の帰属: シグナルの入力をそのまま再生する
#
# どの注文も `data_version` を持っています。監査したい注文があれば、*同じ*シグナルのフレームを
# そのバージョンに向けます。ループが見ていた先頭への O(1) のタイムトラベルで、判断を確認できます。
#
# これが「たぶんシグナルは買いと言っていた」と「これがシグナルです。まさにそのバイト列から
# 計算し直したもので、買いと言っています」の差です。

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
# ## 7. 読み手の側: `tail()` で新しい注文を流し取る
#
# 下流の利用者、リスクチェックでも執行のブリッジでも、`tail('orders', after_version, poll_ms)`
# で注文の記録を追いかけます。
#
# 本番の規則が2つあります。`tail` は**追記だけの**バージョン連鎖を要求するので、範囲の中に
# write、削除、復元、圧縮があるとエラーになります。私たちの記録が追記専用である理由の1つです。
#
# そして `tail` は**境界がありません**。`LIMIT` の行数がそろうまでブロックするので、あると
# 分かっている行数を超えない LIMIT か、クエリのタイムアウトを必ず添えてください。

# %%
overs = db.versions("orders")
after = overs[max(0, len(overs) - 6)]          # start ~5 commits behind head
available = overs[-1]["rows"] - after["rows"]
n = int(min(10, available))
print(f"tailing orders after v{after['sequence']} ({available} new rows, reading {n}):")
db.sql(f"SELECT * FROM tail('orders', {after['sequence']}, 50) LIMIT {n}",
       timeout=30).to_pandas()

# %% [markdown]
# ## 8. セッションの締め: エクイティカーブと記録の手入れ
#
# エクイティカーブは記録した評価から出します。
#
# そのあとがメンテナンスです。5分ごとのコミットを1日続けると、テーブルあたりおよそ78個の小さな
# セグメントが残るので、`compact()` がそれを併合します。これも監査されるコミット1件です。
#
# トレードオフに注意してください。圧縮のコミットは、それをまたぐ `tail()` の範囲について追記
# だけの連鎖を壊します。だから圧縮は利用者の読み取り位置より*後ろ*でかけてください。

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
# - チャンク1つがコミット1件です。そのコミットのシーケンス番号を注文の各行に刻むことで、帰属は
#   ジョインに、再生は `h5i('trades', v)` のクエリになります。
# - アトミックなコミットがクラッシュ耐性の中身です。書き込みの途中で `kill -9` が来ても直前の
#   先頭は無傷で、再開点は `SELECT max(ts)`。データベースがチェックポイントです。
# - 注文の記録は追記専用に保ってください。下流の利用者に対しては `tail()` と互換のまま、監査人
#   に対しては改竄が見えるままになります。`tail()` の読み取りには必ず LIMIT を、できれば
#   タイムアウトも付けてください。行がそろうまでブロックするからです。
# - 注文はサイクルごとにまとめてコミットし、行ごとには打たないこと。そしてセッションのあと、
#   読み手より後ろで `compact()` をかけて、その日の小さなセグメントを併合してください。

# %%
db.close()
