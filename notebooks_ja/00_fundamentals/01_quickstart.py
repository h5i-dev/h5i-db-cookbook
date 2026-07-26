# %% [markdown]
# # クイックスタート: 最初の h5i-db マーケットデータベース
#
# h5i-db は、クオンツの仕事に合わせて作られた、組み込み型でバージョン管理付きの時系列
# データベースです。書き込みは1回ごとにアトミックなコミットになり、書き換え不能な
# バージョンが1つ積み上がります。SQL レイヤ（Apache DataFusion）には `time_bucket`、
# `vwap`、`ewma`、ASOF ジョイン、ギャップ補完といった時系列演算子が最初から入って
# います。動かすサーバはありません。SQLite や DuckDB と同じで、データベースの実体は
# ディレクトリです。
#
# これから5分で進めるのは次の4つです。
#
# 1. データベースと `trades` テーブルを作る
# 2. 数日分のティックデータを取り込む
# 3. SQL 1文で VWAP 付きの分足を計算する
# 4. テーブルを過去のバージョンへ巻き戻す

# %%
import h5i_db
import pyarrow as pa

import cookbook_utils as cu

print("h5i-db version:", h5i_db.__version__)

# %% [markdown]
# ## 1. データベースとテーブルを作る
#
# `Database` はディスク上のディレクトリです。テーブルは Arrow スキーマと `time_column`
# の指定で宣言します。h5i-db はセグメントをこの列で整列させて保存し、プルーニングや
# ASOF ジョイン、バー集計にも同じ順序を使います。`sort_key` は二次ソート（同じ時刻の
# 中での symbol 順）を足すもので、先頭は必ず時刻列にします。

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
# ## 2. ティックデータを取り込む
#
# `append` は pyarrow の Table でも RecordBatch でも受け取ります。ただし*厳格*です。
# データは時刻順に並んでいて、先頭がテーブルの現在の最大タイムスタンプ以降でなければ
# なりません。upsert ではなく、フィードを流し込む動きだと考えてください。1回の呼び出しが
# アトミックなコミット1件になり、書き換え不能なバージョンが1つ増えます。

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=3, trades_per_day=20_000)
commit = db.append("trades", trades)
commit

# %% [markdown]
# ## 3. SQL で問い合わせる
#
# DataFusion による完全な SQL に、金融向けの演算子が加わります。10万件を超えるティックが
# 1文で VWAP 付きの分足になり、しかもセグメントが時刻順で保存されているので、ソートを
# 挟まずストリーミングで流れます。

# %%
bars = db.sql(
    """
    SELECT time_bucket('1m', ts) AS bar,
           symbol,
           first_value(price ORDER BY ts) AS open,
           max(price)                     AS high,
           min(price)                     AS low,
           last_value(price ORDER BY ts)  AS close,
           sum(size)                      AS volume,
           vwap(price, size)              AS vwap
    FROM trades
    GROUP BY bar, symbol
    ORDER BY bar, symbol
    """
).to_pandas()
bars.head(8)

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
# コミットはすべて `versions()` に並びます。もう1日分を追記してから、*追記前の姿*の
# テーブルを読んでみましょう。ログの再生ではなく O(1) の操作です。

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
latest = db.sql("SELECT count(*) AS n FROM trades").to_pandas()["n"][0]
print(f"version 1: {v1_rows:,} rows\nversion 2: {v2_rows:,} rows\nlatest:    {latest:,} rows")

# %% [markdown]
# タイムトラベルは SQL の中でも効きます。`h5i()` テーブル関数を使えば、過去の
# バージョンと最新のテーブルを1つの文で突き合わせられます。

# %%
db.sql(
    """
    SELECT now.symbol,
           now.n  - was.n  AS trades_added,
           now.mx - was.mx AS ts_advanced_by
    FROM      (SELECT symbol, count(*) AS n, max(ts) AS mx FROM trades GROUP BY symbol) now
    JOIN      (SELECT symbol, count(*) AS n, max(ts) AS mx FROM h5i('trades', 1) GROUP BY symbol) was
    USING (symbol)
    ORDER BY symbol
    """
).to_pandas()

# %% [markdown]
# ## まとめ
#
# - データベースはディレクトリ、テーブルは Arrow スキーマと時刻列。サーバもデーモンも
#   要りません。`pip install` して `Database(path, create=True)` を呼べば終わりです。
# - `append` は時刻順が厳密に守られるフィード型のアトミックコミットです。取り込みに
#   失敗しても、それ以前のバージョンはすべて残っています。
# - OHLCV と VWAP のバーは SQL 1文で得られ、整列済みストレージの上をストリーミングで
#   流れます。
# - タイムトラベルは一級の機能です。Python なら `db.read(v)`、SQL なら
#   `h5i('table', v)` で、どちらも O(1)。これが再現可能なリサーチの背骨になります
#   （レシピ05を参照）。

# %%
db.close()
