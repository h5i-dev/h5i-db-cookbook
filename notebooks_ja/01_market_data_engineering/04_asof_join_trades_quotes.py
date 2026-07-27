# %% [markdown]
# # ASOF ジョイン: 約定と気配、サインとスプレッド
#
# 各約定にその時点の気配を貼り付ける処理は、マイクロストラクチャの*本命*のジョインです。
# 約定のサイン付け、実効スプレッドと実現スプレッドの測定、TCA、トキシシティ分析を支えます。
#
# h5i-db の `asof_join` は、時刻順ストレージの上で動くネイティブの SQL テーブル関数で、pandas
# への往復はありません。方向（`'backward'` か `'forward'`）と鮮度の許容差が組み込まれていて、
# 他のテーブルと同じように CTE やウィンドウ関数と組み合わせられます。
#
# このレシピでは、正解の分かっている2セッションぶんのテープを使って次の4つを進めます。
#
# 1. すべての約定に気配を貼り、`pandas.merge_asof` と突き合わせる
# 2. Lee-Ready 流に約定へサインを付け、その精度を採点する
# 3. クオート・実効・実現の3つのスプレッドを測る
# 4. 許容差を使って、古い気配を黙って使うかわりに拒否する

# %%
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_asof"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_quotes` の2セッションぶんのベストビッド・ベストオファーです。最良気配が動くたびに
# 1行です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 気配の時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `bid`、`ask` | `float64` | 最良のビッドとオファー |
# | `bid_size`、`ask_size` | `int64` | 各サイドの表示数量 |

# %%
quotes = cu.make_quotes(
    symbols=["AAPL", "MSFT", "NVDA"], days=2, quotes_per_day=1_000, seed=11
)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# サイン付けを*採点*するには、本当のアグレッサー側が分かっている約定が必要です。そこでテープを
# 気配ストリームから導きます。各約定はその時点のビッドかアスクを取り、10%はミッドで約定し
# （分類にとって難しいケースです）、報告の遅延は 0.2〜5ms、そしてすべてのプリントが本当の
# `side` を保持します。
#
# 導出したテープには `trade_id` も足しておきます。後のジョインで個々のプリントを指名できる
# ようにするためです。

# %%
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

print(f"{len(tr):,} trades derived from {len(qp):,} quotes over 2 sessions")
tr[["ts", "symbol", "trade_id", "price", "size", "side"]].head()

# %% [markdown]
# それと並べて `marks` テーブルも保存します。各約定のタイムスタンプを5分先にずらしたもので、
# 4節の実現スプレッドの参照に使います。

# %%
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

db.tables()

# %% [markdown]
# ## 2. ジョインそのもの
#
# `asof_join(left, right, left_ts, right_ts, by_key)` は、各約定に対して、その時刻以前で最新の
# 気配を銘柄ごとに返します。
#
# 右側でぶつかった列には `_right` が付きます。ここでは気配自身のタイムスタンプが `ts_right`
# として届くので、約定時刻から `ts_right` を引けば気配の古さになります。どちらのテーブルも
# 時刻順で保存されているので、ソートの工程に払うコストはありません。

# %%
t0 = time.perf_counter()
TQ = db.table("trades").join_asof(db.table("quotes"), on="ts", by="symbol")

joined = TQ.select(
    "ts", "symbol", "trade_id", "price", "side", "ts_right", "bid", "ask"
).sort("trade_id").to_pandas()
elapsed_ms = (time.perf_counter() - t0) * 1e3
print(f"{len(joined):,} trades matched against {len(qp):,} quotes "
      f"in {elapsed_ms:.1f} ms")
joined.head(5)

# %% [markdown]
# 信じてよい、ただし確かめること。`pandas.merge_asof` で同じ整列をかけたとき、すべての約定に
# ついて同じ気配が出てこなければなりません。

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
# キーワード構文もあります。ASOF ジョインが大きな文の一節になるときに便利です。いまのところ
# 制約が2つあります。テーブル名を素で書く必要があり（別名は使えません）、出力列は修飾なしで
# 参照します。

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
# ## 3. Lee-Ready のサイン付け
#
# まずクオートルール。ミッドより上なら買い、下なら売りです。ミッドちょうどのプリントはティック
# テストに落ち、ここでは `lag(price)` を使います。厳密な Lee-Ready は直近の*異なる*価格を求め
# ますが、このテープではほとんど差が出ない精緻化です。
#
# 全体が1つの文に収まります。asof ジョインがウィンドウ関数に流れ、それが `CASE` の階段に
# 流れます。

# %%
PREV_PX = sql_expr("lag(price)").over(partition_by="symbol", order_by="ts")

signed = (
    TQ.select(
        "ts", "symbol", "trade_id", "price", "size", "side", "bid", "ask",
        mid=(col("bid") + col("ask")) / 2,
    )
    .with_columns(prev_px=PREV_PX)
    .with_columns(
        lr_side=when(col("price") > col("mid")).then(lit("B"))
        .when(col("price") < col("mid")).then(lit("S"))
        .when(col("price") > col("prev_px")).then(lit("B"))
        .when(col("price") < col("prev_px")).then(lit("S"))
        .otherwise(lit("U"))
    )
    .to_pandas()
)

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
# クオートルールはほぼ完璧です。唯一の失敗は、約定の数ミリ秒の報告遅延の中に気配の更新が
# 入り込むケースです。ミッドのプリントは、ほぼコイン投げのティックテストに落ちます。
#
# 実際の TAQ データではクオートルールの割合はもっと低く、全体の精度は 85〜90% あたりに
# 落ち着きます。上の内訳は、誤りがどこに住んでいるかを正確に示しています。
#
# ## 4. クオート・実効・実現の3つのスプレッド
#
# 3つの尺度、3つの式です。
#
# - クオートはその時点の気配の `ask - bid`
# - 実効は `2·|price - mid|` で、アグレッサーが実際に払った額
# - 実現は `2·d·(price - mid₊₅ₘ)` で、流動性供給者が実際に手元に残した額。5分後のミッドを
#   使います
#
# その5分後のミッドは、`marks` テーブルを**前向きに** asof 結合して取ります。約定時刻の5分後
# 以降で最初の気配です。すべてが1つの文に収まり、ミッドに対するベーシスポイントで出てきます。

# %%
MID = (col("bid") + col("ask")) / 2

j = TQ.select("symbol", "trade_id", "price", "side", "bid", "ask", mid=MID)
m = (
    db.table("marks")
    .join_asof(db.table("quotes"), on="ts", by="symbol", direction="forward")
    .select("trade_id", mid_5m=MID)
)

# .join() aliases the sides l and r: l is the trade-time join, r the mark.
px = col("price", relation="l")
mid = col("mid", relation="l")
mid_5m = col("mid_5m", relation="r")
direction = when(col("side", relation="l") == "B").then(lit(1)).otherwise(lit(-1))

spreads = (
    j.join(m, on="trade_id")
    .filter(mid_5m.is_not_null())
    .group_by(col("symbol", relation="l").alias("symbol"))
    .agg(
        quoted_bps=((col("ask", relation="l") - col("bid", relation="l")) / mid).mean() * 1e4,
        effective_bps=(2 * (px - mid).abs() / mid).mean() * 1e4,
        realized_bps=(2 * direction * (px - mid_5m) / mid).mean() * 1e4,
        n_trades=count_star(),
    )
    .sort("symbol")
    .to_pandas()
)
spreads

# %%
x = np.arange(len(spreads))
w = 0.27
fig, ax = plt.subplots(figsize=(8, 4))
for i, measure in enumerate(("quoted_bps", "effective_bps", "realized_bps")):
    ax.bar(x + (i - 1) * w, spreads[measure], width=w, label=measure.replace("_bps", ""))
ax.set_xticks(x, spreads["symbol"])
ax.set_title("Spread measures by symbol (bps of mid)")
ax.set_xlabel("symbol")
ax.set_ylabel("basis points")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 実効がクオートよりひと回り小さいのは、プリントの10%がミッドで約定するからです。
#
# 実現と実効のあいだに*系統的な*差はありません。この合成フローには情報が乗っていないので、
# 約定のあと平均的にはミッドが流動性供給者に不利な方向へ動かないのです。銘柄ごとの推定値が
# 実効の周りに散らばるのは、5分のミッドのボラティリティがスプレッドを圧倒するからで、その
# ノイズの多さは現実の推定量にも忠実です。
#
# 実際のテープでは、実現は実効より系統的に*下*に来ます。その差 `2·d·(mid₊₅ₘ − mid)` が価格
# インパクトで、この分解が逆選択コストを測る標準的なやり方です。
#
# ## 5. 許容差: 古い気配を拒否する
#
# 既定では asof ジョインはいくらでも過去まで遡ります。気配フィードに穴があると、つまり障害や
# 間引かれたベンダーファイルがあると、何分も前の NBBO で約定を黙って評価してしまいます。
#
# 任意の許容差が遡りの範囲を区切ります。単位は生のマイクロ秒で、h5i-db の生の時刻引数はすべて
# これです。一致しなかった約定は気配の列が NULL のまま残るので、古さが*見える*ようになり、
# 数えられるようになります。以下では問題を再現するために、気配フィードを25%に間引きます。

# %%
qs = qp[rng.random(len(qp)) < 0.25].sort_values(["ts", "symbol"])
db.create_table("quotes_sparse", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_sparse", pa.Table.from_pandas(qs, preserve_index=False).cast(quote_schema))

for label, tol in [("no tolerance", None), ("60s tolerance", 60_000_000),
                   ("5s tolerance", 5_000_000)]:
    r = (
        db.table("trades")
        .join_asof(db.table("quotes_sparse"), on="ts", by="symbol", tolerance=tol)
        .select(n=count_star(), matched=col("bid").count())
        .to_pandas()
    )
    print(f"{label:>13}: {r['matched'][0]:,} of {r['n'][0]:,} trades matched "
          f"({r['matched'][0] / r['n'][0]:.1%})")

# %% [markdown]
# 許容差がなければ、どの約定にもどれだけ古かろうと*何かしらの*気配が付きます。鮮度の条件を
# 置けば、その時点の気配が古すぎる約定は正直に NULL を返します。本番ではこの NULL の割合が、
# アラートを張る価値のあるデータ品質の指標になります。
#
# ## まとめ
#
# - `.join_asof(other, on="ts", by="symbol")` が約定と気配の整列のすべてです。キーごとに、
#   時刻的に正しく、整列済みストレージの上をストリーミングで流れ、`pandas.merge_asof` と
#   1行ずつ一致します。両側は素のピン留めなしテーブルでなければなりません。土台の `asof_join`
#   がテーブル名を取るからで、絞り込みはジョインの*後*にやります。
# - 方向と許容差は一級の引数です。`direction="forward"` は実現スプレッド用に5分後のミッドを
#   取ってきました。マイクロ秒の `tolerance` は、古い気配での評価を目に見えて数えられる NULL
#   に変えました。許容差を振るのも、3本の文を手書きするのではなく Python のループです。
#   右側でぶつかった列には `_right` が付きます。
# - 結合したフレームは組み合わせられます。ここでは `TQ` という変数に持たせ、Lee-Ready の
#   サイン付けとクオート・実効・実現の分解の両方がその上に載っています。`CASE` の階段には
#   `when().then()`、ティックテストには `sql_expr("lag(price)").over(...)` を使いました。
# - 正解と突き合わせて採点するには、約定と気配が1つの価格過程を共有するテープが要ります。
#   合成データでサイン付けの精度を測る前に、思い出す価値のある前提です。

# %%
db.close()
