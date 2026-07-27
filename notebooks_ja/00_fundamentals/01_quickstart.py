# %% [markdown]
# # クイックスタート: 最初の h5i-db マーケットデータベース
#
# h5i-db は、クオンツの仕事に合わせて作られた、組み込み型でバージョン管理付きの時系列
# データベースです。動かすサーバはありません。SQLite や DuckDB と同じで、データベースの
# 実体はディスク上のディレクトリです。
#
# 汎用の組み込みストアと違うところが2つあります。1つは、書き込みが1回ごとにアトミックな
# コミットになり、書き換え不能なバージョンが積み上がること。そのバージョンは後からでも
# そのままクエリできます。もう1つは SQL レイヤ（Apache DataFusion）で、`time_bucket`、
# `vwap`、`ewma`、ASOF ジョイン、ギャップ補完といった、現場で実際に要る時系列演算子が
# 最初から入っています。
#
# これから5分で進めるのは次の4つです。
#
# 1. これから読み込むティックデータを眺める
# 2. データベースを作って取り込む
# 3. クエリ1つで VWAP 付きの分足を計算する
# 4. 直前の取り込みが起きる前の姿でテーブルを読む

# %%
import h5i_db
import pyarrow as pa
from h5i_db import col, count_star, time_bucket, vwap

import cookbook_utils as cu

print("h5i-db version:", h5i_db.__version__)

# %% [markdown]
# ## 1. データ
#
# このクックブックのレシピは、マーケットデータを `cookbook_utils` から取ります。シードを
# 固定すれば毎回同じものが出てくる、合成データの生成器です。`cu.make_trades` が返すのは
# ティックデータで、1行が1約定、3銘柄×3セッションぶんあります。約定の到着はセッションの
# 中でU字型に偏り、価格はビッドとアスクの間で跳ねるので、動きは実物に近くなります。
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
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# ## 2. データベースを作って取り込む
#
# テーブルは Arrow スキーマと `time_column` の組です。h5i-db はセグメントをこの列で整列
# させて保存し、プルーニングや ASOF ジョイン、バー集計にも同じ順序を使います。`sort_key`
# は同じ時刻の中での二次ソートを足すもので、先頭は必ず時刻列にします。

# %%
db = h5i_db.Database(cu.fresh_db("00_quickstart"), create=True)

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %% [markdown]
# `append` は pyarrow の Table でも RecordBatch でも受け取りますが、厳格です。スキーマが
# 一致していること、時刻順に並んでいること、先頭がテーブルの現在の最大タイムスタンプ以降
# であること。upsert ではなく、フィードを流し込む動きだと考えてください。
#
# 呼び出し1回がアトミックなコミット1件です。戻り値は受領証で、新しいバージョン番号と、
# コミット後の行数・セグメント数が入っています。

# %%
commit = db.append("trades", trades)
commit

# %% [markdown]
# ## 3. 問い合わせる
#
# 同じエンジンにクエリの面が2つあります。`db.table(...)` はメソッド呼び出しで組み立てる
# **遅延クエリ**の入口で、結果を回収するまで何も走りません。`db.sql(...)` のほうは文字列を
# 取ります。どちらも同じ DataFusion のプランにコンパイルされます。
#
# 分足を出すのに使う金融向け演算子は3つです。
#
# - `time_bucket('1m', ts)` は各約定をその分に切り下げ、バーの格子を作ります
# - `.first("ts")` と `.last("ts")` は `ts` 順の `first_value`／`last_value` です。始値と終値が
#   行の並び順ではなくイベント時刻で決まります
# - `vwap(price, size)` は組み込みの集約関数です
#
# セグメントはもともと `ts` 順で保存されているので、このクエリはソートを挟まずストリー
# ミングで流れます。

# %%
bar_query = (
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
bars = bar_query.to_pandas()
bars.head(8)

# %% [markdown]
# ビルダの正体はコンパイラで、第2のエンジンが増えるわけではありません。`.sql()` を呼べば
# DataFusion に渡したものがそのまま見えますし、その文字列は `db.sql()` に貼ればそのまま
# 動きます。
#
# ビルダ自体はレシピ09で扱います。以降のクックブックは既定でビルダを使い、文字列のほうが
# 読みやすい場面だけ SQL に降ります。

# %%
print(bar_query.sql())

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
for sym, g in bars.groupby("symbol"):
    ax.plot(g["bar"], g["vwap"] / g["vwap"].iloc[0], label=sym, lw=0.8)
ax.set_title("1-minute VWAP, normalized")
ax.set_xlabel("time")
ax.set_ylabel("VWAP (first bar = 1)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 4. バージョンとタイムトラベル
#
# `versions()` には、そのテーブルが積んだコミットがすべて並びます。4日目を追記しても、
# 3日ぶんのテーブルはどこにも行きません。バージョン1として読めるままです。古いバージョンを
# 開く操作は O(1) です。h5i-db は古いマニフェストを読むだけで、ログを再生したりはしません。

# %%
day4 = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, start="2026-06-04", seed=8)
db.append("trades", day4, note="day 4 feed")

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("trades")
]

# %%
v1_rows = len(db.read("trades", version=1))
v2_rows = len(db.read("trades", version=2))
latest = db.table("trades").select(count_star().alias("n")).to_pandas()["n"][0]
print(f"version 1: {v1_rows:,} rows\nversion 2: {v2_rows:,} rows\nlatest:    {latest:,} rows")

# %% [markdown]
# タイムトラベルはクエリの中でも効くので、過去のバージョンと最新のテーブルを1つの文で
# 突き合わせられます。読み取り点は `db.table(...)` に渡すだけで、内部では `h5i()` テーブル
# 関数に落ちます。クエリは Python の値ですから、バージョンの関数として一度書いて2回呼べば
# 済みます。

# %%
def per_symbol(version=None):
    return (
        db.table("trades", version=version)
        .group_by("symbol")
        .agg(count_star().alias("n"), col("ts").max().alias("mx"))
    )


was, now = per_symbol(1), per_symbol()
now.join(was, on="symbol").select(
    symbol=col("symbol", relation="l"),
    trades_added=col("n", relation="l") - col("n", relation="r"),
    ts_advanced_by=col("mx", relation="l") - col("mx", relation="r"),
).sort("symbol").to_pandas()

# %% [markdown]
# ## まとめ
#
# - データベースはディレクトリ、テーブルは Arrow スキーマと時刻列です。サーバもデーモンも
#   要りません。`pip install` して `Database(path, create=True)` を呼べば終わりです。
# - `append` は時刻順が厳密に守られるフィード型のアトミックコミットです。取り込みに失敗
#   しても、すでに保存されているものは壊れません。
# - OHLCV と VWAP のバーは集約1つで得られ、整列済みストレージの上をストリーミングで流れ
#   ます。`db.table(...)` の動詞で組み立てても、SQL で書いても構いません。両者をつなぐ扉が
#   `.sql()` です（レシピ09）。
# - タイムトラベルは一級の機能で、しかも O(1) です。Python なら `db.read(v)`、ビルダなら
#   `db.table(v)`、SQL なら `h5i('table', v)`。これが再現可能なリサーチの背骨になります。
#   レシピ05がそこを引き取ります。

# %%
db.close()
