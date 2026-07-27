# %% [markdown]
# # DataFrame ビルダ: クエリを Python の値として扱う
#
# `db.table(...)` はメソッド呼び出しで組み立てる**遅延**クエリの入口です。SQL 文字列を書く
# 代わりに動詞をつないでいき、`.collect()` のような終端の呼び出しまで何も走りません。これは
# 第2のエンジンではなくコンパイラです。どの動詞も `db.sql()` を通る SQL に落ちるので、
# 組み立てたクエリが見るセッションもテーブル関数もバージョンの固定も、自分で書いたであろう
# 文字列とまったく同じです。そして `.sql()` を呼べば、何が生成されたかがそのまま見えます。
#
# リサーチの現場での見返りは、生成するクエリにあります。窓幅や列をループで掃引する
# ファクターライブラリは、いまなら f 文字列で SQL を組み立てているはずです。クォートのバグ
# や `'` の混入が住み着くのはそこです。ビルダなら識別子のクォートは1か所に集約され、
# 組み立て途中のパイプラインは、持ち回して伸ばして再利用できるただの Python の値になります。

# %%
import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, vwap, when

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_dataframe_builder"), create=True)

trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades)

prices = cu.make_daily_prices(days=500)  # 50 names x 500 sessions
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices)

print(f"trades: {len(trades):,} rows   prices: {len(prices):,} rows")

# %% [markdown]
# ## 1. フレームとは、まだ走っていないクエリ
#
# `db.table("trades")` はテーブル全体を出発点にします。動詞は**新しい**フレームを返すので、
# 元は書き換わらず、途中まで組んだパイプラインをそのまま再利用できます。`.sql()` が文字列に
# し、`.collect()` が実行します。

# %%
liquid = db.table("trades").filter(col("symbol").is_in(["AAPL", "NVDA"]))
print(liquid.sql())

# %% [markdown]
# 定番の OHLCV ロールアップを組み立てるとこうなります。`group_by(...).agg(...)` はキーを
# 集約と並べて射影します。`.first("ts")` と `.last("ts")` は `first_value(x ORDER BY ts)`
# の書き方そのもので、自己結合なしにバーの始値と終値を返します。

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
# `col(name)` が列、`lit(value)` が定数で、四則演算と比較はそこから積み上がります。早めに
# 出会っておきたい罠が2つあります。
#
# - Python では `and` / `or` / `not` を多重定義できないので、論理は `&`、`|`、`~` を使います。
#   これらは比較より*強く*結合するため、比較のたびに括弧が要ります。
# - 式が保つのは Python ではなく **SQL** の意味です。整数列どうしの `/` は整数除算になります。
#   本当の除算がほしいならキャストしてください。

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
# 識別子は常にクォートされるので大文字小文字が保たれ（素の SQL なら小文字に畳まれる
# `Symbol` という名前のフィールドも `col("Symbol")` で見つかります）、文字列リテラルは
# あくまで文字列で、構文になることはありません。

# %%
print(db.table("trades").filter(col("symbol") == "'; DROP TABLE trades; --").sql())

# %% [markdown]
# ## 3. パイプラインが SQL になるまで
#
# たいていのパイプラインは平らな `SELECT` 1つに落ちます。独立した `with_columns` は1つに
# まとまり、*元からある*列での絞り込みは同じ `WHERE` に残ります。一方、前の段が**計算した**
# 列を読む段は自分の階層を持ちます。SQL は `WHERE` を選択リストの兄弟ではなく `FROM` に
# 対して解決するからです。集約と `LIMIT` と `DISTINCT` も階層を閉じます。後続はその出力に
# 対して働くためです。

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
# 身につける価値があるのは、階層が*どこで*閉じるかの感覚です。次の段に何が見えるかがそれで
# 決まります。パイプラインが平らなあいだは、動詞は元のテーブルまで届きます。
# `select("ts", "symbol").sort("close")` がきちんと解決するのは、SQL の `ORDER BY` が
# 見にいく先も同じく `FROM` だからです。集約が階層を閉じたあとは、その列は本当に消えていて、
# エンジンがそう言ってきます。

# %%
try:
    db.table("prices").group_by("symbol").agg(count_star().alias("n")).sort("close").collect()
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: {str(e)[:180]}")

# %% [markdown]
# ## 4. 見返り: 生成するクエリ
#
# ビルダが本領を発揮するのはここです。複数の遡及期間の掃引は、フレームを回すだけの Python の
# ループになります。文字列を切り貼りする必要はなく、それぞれのフレームは持って名前を付けて
# 再利用できる値です。ここでは、価格とその移動平均との乖離を3つの窓で見ます。
#
# rolling 系のメソッドは `window` と `order_by` を取り、任意で `partition_by` も取ります。
# SQL の糖衣構文 `rolling_avg` と違って本物の `PARTITION BY` を伴うので、複数銘柄のテーブル
# でも銘柄が**混ざりません**。

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
# クロスセクションの演算子は、値を*同じ時点の*仲間と比べて順位づけるので、比較する範囲を
# 引数に取ります。z スコア化したいくつかのシグナルを1つの合成値にまとめる形は、そのまま
# ファクター構築の骨格です。

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
# ## 5. バージョンの固定とジョイン
#
# 読み取り点は `db.table()` にそのまま渡り、`h5i()` に落ちます。固定したビルダのクエリは、
# 手で書いた SQL とまったく同じくソースの時点で束縛されます。`.join()` は両側を `l` と `r`
# という別名のサブクエリとして描き、この別名が特定の側に触れるための約束事になります。
# おかげで「同じクエリを N 個のバージョンに当てる」比較が、関数呼び出し1つで済みます。

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
# `.join_asof()` は `asof_join` テーブル関数に落ちます。この関数が取るのは*テーブル名*で、
# どちらも最新版として読むので、すでに動詞を当てた側や固定した側をビルダは黙って無視せず
# 拒否します。絞り込みはジョインのあとに置いてください。

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
# ## 6. 非常口と、SQL へ戻る扉
#
# 動詞だけで SQL の全機能を覆うことは、意図して目標にしていません。`sql_expr()` は式を
# 受け付ける場所ならどこにでも生の断片を落とせます。テキストはそのまま挿入されるので、
# クォートの責任が自分にある唯一の場所でもあります。

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
# 非常口のうち、いちばん手を伸ばすことになるのは `lag` です。`.lag()` メソッドはありません
# が、`sql_expr` の断片はウィンドウ化できるので、他の集約と同じように `.over()` を取ります。
# これで `lag`、`lead`、`row_number` をはじめ SQL のウィンドウ関数一式が使えます。日次
# リターンという、このクックブックで最も多く出てくる形を見てみます。

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
# `with_columns` が2段になっている点に注目してください。`ret` は前の段が計算した
# `prev_close` を読むので、ビルダは解決できない SQL を吐く代わりに階層を閉じます。断片を
# `PREV_CLOSE` のような Python の名前に1度だけ束ねて使い回す習慣が、ファクターライブラリを
# 正直に保ってくれます。

# %% [markdown]
# パイプラインがビルダの手に余るようになったら、`.sql()` がクエリを渡してくれるので、それを
# `db.sql()` に貼って続きを書けます。2つの面は、あいだに扉のある1つの仕組みです。生成される
# SQL は決定的なので、スナップショットテストにかけても差分を取っても構いません。
#
# 対応する動詞がなく、文字列のほうが素直に読める場面は `db.sql()` に残ります。`UNION ALL`、
# 深い多段 CTE、スカラサブクエリ、そして `gapfill` / `resample` / `tail` の各テーブル関数です。
# 2つの読み取り点をラベル付きで1つの結果に積むのが、日常的な例です。

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
#   `.to_pandas()` まで何も走りません。`.sql()` でコンパイル結果の SQL が見えます。
# - ビルダは `db.sql()` の上のコンパイラであって、第2のエンジンではありません。セッションも
#   テーブル関数も `h5i()` によるバージョン固定も、すべて同じものです。
# - 論理には `&`、`|`、`~` を使い、式が SQL の意味を持つことを忘れないでください。整数の
#   `/` は切り捨てます。
# - 手を伸ばすべきなのはクエリを**生成する**とき、つまり窓幅や列をループで掃引するような
#   場面です。f 文字列の SQL がクォートのバグを招くところが、ちょうどビルダの持ち場になります。
#   1度しか書かないクエリなら、素の SQL のほうが短いことも多いはずです。
# - `rolling_*` と `cs_*` のメソッドは本物の `PARTITION BY` を伴います。SQL の糖衣構文
#   `rolling_avg` のほうは全体をまたぐ行数ベースの窓で、そこが違います。
# - `sql_expr()` が非常口、`.sql()` が戻り道です。どちらの面も二級市民ではありません。

# %%
db.close()
