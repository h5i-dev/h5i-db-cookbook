# %% [markdown]
# # DataFrame ビルダ: クエリを Python のオブジェクトとして扱う
#
# `db.table(...)` は、メソッド呼び出しで組み立てる**遅延**クエリの入口です。SQL の文字列は
# 書きません。`.collect()` のような終端の呼び出しまで、何も走りません。
#
# 正体はコンパイラで、第2のエンジンではありません。どの動詞も `db.sql()` を通る SQL に落ちる
# ので、組み立てたクエリが見るセッションもテーブル関数もバージョンのピン留めも、手で書いた
# 文字列の場合とまったく同じです。`.sql()` を呼べば、生成されたものがそのまま見えます。
#
# リサーチのデスクにとっての見返りは、クエリを生成できることです。ウィンドウや列をループで
# 掃くファクターライブラリは、いまなら f-string で SQL を作っているでしょう。クオートの
# バグと `'` のインジェクションが住んでいるのはそこです。ビルダなら識別子のクオートは1か所
# に集まり、組み立て途中のパイプラインは、持ち回して伸ばして再利用できる普通の Python の値
# になります。

# %% [markdown]
# ## ここで使う用語
#
# | 用語       | 意味 |
# | -------- | --- |
# | 遅延（lazy） | `.collect()` のような終端呼び出しまで何も実行されない |
# | 動詞（verb） | `.filter()` や `.group_by()` など、クエリを伸ばすビルダのメソッド1つ |
# | CTE      | `WITH` で導入する名前付き副問い合わせ。多段のクエリが読みやすくなる |
# | ASOF 結合  | 左の各行を、そのタイムスタンプ以前で最も新しい右の行に結合する |
# | gapfill  | 不規則な系列を規則的な時間グリッドに載せる |
# | パネル      | 1行が1資産1日付になるデータセット |
# | Zスコア     | ある値がローリング平均から標準偏差いくつ分離れているか |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, vwap, when

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_dataframe_builder"), create=True)

# %% [markdown]
# ## データ
#
# テーブルは2つです。ビルダの動詞も同じ線で分かれるからです。`cu.make_trades` のティック
# レベルの `trades` は、バケット化と集約を試す側です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
print(f"trades: {trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# `cu.make_daily_prices` の日足 OHLCV パネル、50銘柄×500セッションのほうは、ウィンドウと
# クロスセクションの動詞を試す側です。列は `ts`、`symbol`、`open`、`high`、`low`、`close`、
# `volume` です。

# %%
prices = cu.make_daily_prices(days=500)  # 50 names x 500 sessions
print(f"prices: {prices.num_rows:,} rows x {prices.num_columns} columns")
prices.to_pandas().head()

# %%
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades)

db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices)

db.tables()

# %% [markdown]
# ## 1. フレームとは、まだ走っていないクエリのこと
#
# `db.table("trades")` はテーブル全体を出発点にします。動詞は**新しい**フレームを返すので、
# 何も書き換わりませんし、途中まで組んだパイプラインはそのまま再利用できます。`.sql()` が
# それを描き出し、`.collect()` が走らせます。

# %%
liquid = db.table("trades").filter(col("symbol").is_in(["AAPL", "NVDA"]))
print(liquid.sql())

# %% [markdown]
# 定番の OHLCV 集計を組んでみます。`group_by(...).agg(...)` は集約と並べてキーも射影します。
# `.first("ts")` と `.last("ts")` は `first_value(x ORDER BY ts)` の定型で、自己結合なしに
# バーの始値と終値を出します。

# %%
bars = (
    db.table("trades")
    .group_by(time_bucket("1m", col("ts")).alias("bar"), "symbol")
    .agg(
        col("price").first("ts").alias("open"),
        col("price").max().alias("high"),
        col("price").min().alias("low"),
        col("price").last("ts").alias("close"),
        col("size").sum().alias("volume"),
        vwap(col("price"), col("size")).alias("vwap"),
    )
    .sort(["bar", "symbol"])
)
print(bars.sql())

# %%
bars.to_pandas().head(6)

# %% [markdown]
# ## 2. 式
#
# `col(name)` が列、`lit(value)` が定数で、あとは算術と比較で積み上げます。早めに出会って
# おきたい罠が2つあります。
#
# - Python では `and`、`or`、`not` を多重定義できないので、真偽の論理は `&`、`|`、`~` を
#   使います。これらは比較より*強く*結合するので、比較のたびに括弧が要ります。
# - 式は Python ではなく **SQL** の意味論を保ちます。整数列どうしの `/` は整数除算です。
#   真の除算が欲しければキャストしてください。

# %%
signed = (
    db.table("trades")
    .filter((col("price") > 0) & (col("size") >= 100))
    .select(
        "ts",
        "symbol",
        "price",
        notional=col("price") * col("size"),
        lots=col("size").cast("DOUBLE") / 100,
        direction=when(col("side") == "B").then(lit(1)).otherwise(lit(-1)),
    )
)
print(signed.sql())

# %%
signed.to_pandas().head(4)

# %% [markdown]
# 識別子は常にクオートされるので、大文字小文字も残ります。`col("Symbol")` は `Symbol` という
# 名前のフィールドを見つけます。素の SQL なら小文字に畳まれるところです。そして文字列
# リテラルは常に文字列で、構文になることはありません。

# %%
print(db.table("trades").filter(col("symbol") == "'; DROP TABLE trades; --").sql())

# %% [markdown]
# ## 3. パイプラインが SQL になるまで
#
# たいていのパイプラインは1つの平らな `SELECT` にコンパイルされます。独立した `with_columns`
# は1つにまとまりますし、*ベース*の列に対する絞り込みは同じ `WHERE` に残ります。
#
# 前の段が**計算した**列を読む段は、自分の階層を持ちます。SQL は `WHERE` を、選択リストの
# 隣の項目ではなく `FROM` に対して解決するからです。集約と `LIMIT` と `DISTINCT` も階層を
# 閉じます。後に続くものは、その出力に対して働くからです。

# %%
movers = (
    db.table("prices")
    .with_columns(ret=col("close") / col("open") - 1)
    .filter(col("ret") > 0.01)  # reads a computed column -> subquery
    .sort("ret", descending=True)
    .limit(5)
)
print(movers.sql())

# %%
movers.to_pandas()

# %% [markdown]
# 階層が*どこで*閉じるかを体で覚えることが、いちばん効きます。次の段が何を見られるかを決める
# のがそれだからです。
#
# パイプラインが平らなあいだは、動詞はまだベーステーブルに届きます。
# `select("ts", "symbol").sort("close")` はちゃんと解決します。SQL の `ORDER BY` が読むのは
# `FROM` だからです。集約が階層を閉じてしまうと、その列は本当に消えます。エンジンもそう
# 言います。

# %%
try:
    db.table("prices").group_by("symbol").agg(count_star().alias("n")).sort("close").collect()
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: {str(e)[:180]}")

# %% [markdown]
# ## 4. 見返り: 生成するクエリ
#
# ビルダが場所代を稼ぐのはここです。複数のルックバックを掃く処理が、文字列の切り貼りでは
# なくフレームに対する Python のループになり、しかも各フレームは持てて名前を付けられて
# 再利用できる値です。下では、価格とそれ自身の移動平均との乖離を3つのウィンドウで出します。
#
# ローリングのメソッドは `window` と `order_by`、任意で `partition_by` を取ります。SQL の
# `rolling_avg` 糖衣と違って本物の `PARTITION BY` を持つので、複数銘柄のテーブルでも銘柄を
# 混ぜることは**ありません**。

# %%
WINDOWS = (5, 20, 60)

base = db.table("prices").filter(col("symbol").is_in(["STK000", "STK001", "STK002"]))

ma_gap = base.with_columns(
    **{
        f"gap_{n}d": col("close") / col("close").rolling_mean(n, order_by="ts", partition_by="symbol") - 1
        for n in WINDOWS
    }
)
print(ma_gap.sql())

# %%
ma_gap.select("ts", "symbol", *[f"gap_{n}d" for n in WINDOWS]).sort(["ts", "symbol"]).to_pandas().tail(6)

# %% [markdown]
# クロスセクションの演算子は、*同じ瞬間の*仲間内で値を順位づけるので、比較する単位を引数に
# 取ります。z 化した signal をいくつか1つの合成値にまとめる、というのがファクター構築の形
# そのものです。

# %%
combo = (
    db.table("prices")
    .with_columns(
        z_ret=(col("close") / col("open") - 1).cs_zscore(partition_by="ts"),
        z_vol=col("volume").cast("DOUBLE").cs_zscore(partition_by="ts"),
    )
    .with_columns(score=(col("z_ret") - col("z_vol")) / 2)
    .select("ts", "symbol", "z_ret", "z_vol", "score")
    .sort(["ts", "score"], descending=[False, True])
)
combo.to_pandas().head(5)

# %% [markdown]
# ## 5. バージョンのピン留めとジョイン
#
# 読み取り点はそのまま `db.table()` に渡って `h5i()` に落ちるので、ピン留めしたビルダの
# クエリは、手書きの SQL と同じくソースの時点で束縛されます。
#
# `.join()` は両側をサブクエリとして描き、`l` と `r` の別名を付けます。この別名が、特定の
# 側に手を伸ばすための約束事です。この2つが揃うと、「同じクエリを N 個のバージョンに対して」
# という比較が関数呼び出しになります。

# %%
db.append("trades", cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, start="2026-06-04", seed=8))


def per_symbol(version=None):
    return db.table("trades", version=version).group_by("symbol").agg(
        count_star().alias("n"), col("ts").max().alias("last_ts")
    )


drift = per_symbol(1).join(per_symbol(), on="symbol").select(
    symbol=col("symbol", relation="l"),
    trades_added=col("n", relation="r") - col("n", relation="l"),
)
print(drift.sql())

# %%
drift.sort("symbol").to_pandas()

# %% [markdown]
# `.join_asof()` は `asof_join` テーブル関数に落ちます。この関数は*テーブル名*を取って両方を
# 最新で読むので、すでに動詞が適用された側やピン留めされた側を、ビルダは黙って無視せず
# はっきり拒否します。絞り込みはジョインの後でやってください。

# %%
tape, quotes = cu.make_trades_and_quotes(days=2)  # shared base prices
for name, data in (("tape", tape), ("quotes", quotes)):
    db.create_table(name, data.schema, time_column="ts", sort_key=["ts", "symbol"])
    db.append(name, data)

try:
    db.table("tape").filter(col("symbol") == "AAPL").join_asof(db.table("quotes"), on="ts", by="symbol")
except h5i_db.InvalidInputError as e:
    print(f"{type(e).__name__}: {e}\nhint: {e.hint}")

# %%
tq = (
    db.table("tape")
    .join_asof(db.table("quotes"), on="ts", by="symbol", tolerance=5_000_000)
    .filter(col("symbol") == "AAPL")
    .select("ts", "symbol", "price", "bid", "ask", mid=(col("bid") + col("ask")) / 2)
)
print(tq.sql())

# %%
tq.to_pandas().head(4)

# %% [markdown]
# ## 6. 脱出口と、SQL へ戻る扉
#
# 動詞で SQL を全面的に覆うことは、意図的に目標にしていません。`sql_expr()` は、式が受け
# 付けられる場所ならどこにでも生の断片を落とします。テキストはそのまま入るので、クオートを
# 正しくするのが自分の仕事になる唯一の場所でもあります。

# %%
tails = (
    db.table("prices")
    .group_by("symbol")
    .agg(
        p01=sql_expr("approx_percentile_cont(close, 0.01)"),
        p99=sql_expr("approx_percentile_cont(close, 0.99)"),
    )
    .sort("symbol")
    .limit(4)
)
tails.to_pandas()

# %% [markdown]
# いちばんよく手が伸びる脱出口は `lag` でしょう。`.lag()` メソッドはありませんが、`sql_expr`
# の断片はウィンドウ化できるので、集約と同じように `.over()` を取ります。これで `lag`、
# `lead`、`row_number` をはじめ、SQL のウィンドウ関数がひととおり使えます。日次リターンは
# このクックブックでいちばん多く出てくる形です。

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")

rets = (
    db.table("prices")
    .with_columns(prev_close=PREV_CLOSE)
    .with_columns(ret=col("close") / col("prev_close") - 1)
    .filter(col("ret").is_not_null())
    .select("ts", "symbol", "ret")
)
print(rets.sql())

# %%
rets.sort(["ts", "symbol"]).to_pandas().head(4)

# %% [markdown]
# 2段に分けた `with_columns` に注目してください。`ret` は1つ前の段が計算した `prev_close` を
# 読むので、ビルダは解決できない SQL を吐くかわりに階層を閉じます。断片を Python の名前に
# 一度だけ束ねて、ここでは `PREV_CLOSE` として、それを使い回す。ファクターライブラリを正直に
# 保つ習慣です。

# %% [markdown]
# パイプラインがビルダに収まらなくなったら、`.sql()` がクエリを渡してくれます。`db.sql()` に
# 貼って、そこから先を続けてください。2つの面は、真ん中に扉のある1つのシステムです。生成
# される SQL は決定的なので、スナップショットテストや差分にもかけられます。
#
# 動詞がなく、文字列のほうが素直に読めるという理由で `db.sql()` に残るものもあります。
# `UNION ALL`、深い多段の CTE、スカラサブクエリ、そして `gapfill`／`resample`／`tail` の
# テーブル関数です。読み取り点を2つ積んでラベル付きの1つの結果にするのが、日常的な例です。

# %%
db.sql(
    """
    SELECT 'version 1' AS read_point, count(*) AS rows FROM h5i('trades', 1)
    UNION ALL
    SELECT 'latest',                  count(*)         FROM trades
    """
).to_pandas()

# %% [markdown]
# ## まとめ
#
# - `db.table(...)` は遅延クエリです。動詞は新しいフレームを返し、`.collect()` や
#   `.to_pandas()` まで何も走りません。`.sql()` がコンパイル結果の SQL を見せます。
# - ビルダは `db.sql()` の上のコンパイラであって第2のエンジンではありません。セッションも
#   テーブル関数も `h5i()` のバージョンピンも同じものです。
# - 真偽の論理には `&`、`|`、`~` を使います。式が SQL の意味論を持つことも忘れずに。整数の
#   `/` は切り捨てです。
# - 手を伸ばすべきなのは、クエリを**生成する**とき、たとえばウィンドウや列をループで掃く
#   ときです。f-string の SQL がクオートのバグを生む場面ですから。1度しか書かないクエリなら、
#   素の SQL のほうが短いことも多いでしょう。
# - `rolling_*` と `cs_*` のメソッドは本物の `PARTITION BY` を持ちます。全体の行ウィンドウを
#   取るだけの `rolling_avg` 糖衣とは違います。
# - `sql_expr()` が脱出口、`.sql()` が戻る扉です。どちらの面も二級市民ではありません。

# %%
db.close()
