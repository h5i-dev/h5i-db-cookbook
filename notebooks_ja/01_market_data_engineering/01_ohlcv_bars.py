# %% [markdown]
# # ティックデータから OHLCV バーを作る
#
# バーの構築は、どのティックデータセットも最初に通る変換で、微妙なバグが生まれる場所でも
# あります。バケットの境界がずれる、`last` がイベント時刻ではなくファイルの並び順で拾われる、
# セッションが UTC の日付境界で真っ二つになる、といった具合です。h5i-db ではこの工程が
# SQL 1文で済みます。`time_bucket` と順序指定つきの `first_value`／`last_value`、そして
# `vwap` が、時刻順の Parquet セグメントの上をソートなしで流れます。しかも出来上がった
# バーのテーブル自体がバージョン管理されたテーブルなので、保存も監査もタイムトラベルも
# できます。
#
# このレシピで進めるのは次の4つです。
#
# 1. 1週間ぶんのティックを 1分足／5分足／1時間足に集計する
# 2. セッションに揃えた日足を作る（タイムゾーンと夏時間に対応）
# 3. 5分足をバージョン管理された `bars_5m` テーブルとして保存する
# 4. SQL で作ったバーを pandas の参照実装とビット単位で突き合わせる

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, count_star, lit, time_bucket, vwap
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_ohlcv"), create=True)

# %% [markdown]
# ## 1. 1週間ぶんのティックを読み込む
#
# 約45万件の合成約定です。3銘柄・5セッション、日中の売買はU字型で、ビッド・アスク間の
# 跳ね返りも入っています。テーブルは `time_column="ts"` と `symbol` の二次ソートで宣言
# します。この物理的な並び順があるおかげで、以下のバー作成クエリはソートせずに
# ストリーミングで流れます。

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=30_000, seed=7
)

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
db.append("trades", trades, note="week of ticks")["rows_total"]

# %% [markdown]
# ## 2. 好きな幅のバー
#
# バーのロールアップは幅の関数1つです。だから1度だけ組み立てて、あとは呼ぶだけで済みます。
# クエリが Python の値として持ち回せることの見返りがこれです。押さえるべき書き方は
# 3つあります。
#
# - `time_bucket('<width>', ts)` が各ティックを自分のバケットへ切り下げます。幅は `'1m'`、
#   `'5m'`、`'1h'` など（`'1d'`、`'1mo'` なども使えます）。
# - `.first("ts")` と `.last("ts")` は `first_value(price ORDER BY ts)` とその対です。
#   行の並び順というたまたまの結果に左右されず、*イベント時刻*で始値と終値を返します。
#   自己結合は不要です。
# - `vwap(price, size)` はネイティブの集約関数です。
#
# 覚えておきたい命名の決まりが1つあります。計算したグループキーに、すでに存在する列と同じ
# 名前を付けてはいけません。`GROUP BY "ts"` はバケットではなく生の `ts` 列に結び付き、
# ティック1件ごとに1グループという結果を、エラーも出さずに返します。バケットは `bar` に
# しておき、名前が空いている1段下の `select` で改名します。

# %%
def bars(width: str):
    return (
        db.table("trades")
        .group_by(time_bucket(width, col("ts")).alias("bar"), "symbol")
        .agg(
            open=col("price").first("ts"),
            high=col("price").max(),
            low=col("price").min(),
            close=col("price").last("ts"),
            volume=col("size").sum(),
            vwap=vwap(col("price"), col("size")),
            n_trades=count_star(),
        )
        .select(
            col("bar").alias("ts"),
            "symbol", "open", "high", "low", "close", "volume", "vwap", "n_trades",
        )
    )


bars_5m = bars("5m").sort(["ts", "symbol"]).to_pandas()
bars_5m.head(6)

# %% [markdown]
# 幅ごとのバー本数を数えるのも、同じフレームを伸ばすだけです。集約が1段を閉じるので、その上に
# 載せた `count(*)` は文字列をいじらずにサブクエリとして入れ子になります。

# %%
for width in ("1m", "5m", "1h"):
    n = bars(width).select(count_star().alias("n")).to_pandas()["n"][0]
    print(f"{width:>3} bars: {n:,}")

# %% [markdown]
# ## 3. セッションに揃えた日足
#
# `time_bucket` は3つ目の引数を任意で取ります。IANA のタイムゾーンか、起点となる
# タイムスタンプです。`time_bucket('1d', ts, 'America/New_York')` は、UTC の深夜を離れて
# ニューヨークの深夜（夏時間かどうかで 04:00／05:00 UTC）で日を切ります。下の境界は
# ちょうど EDT のオフセットぶんずれています。13:30〜20:00 UTC の現物セッションなら、どちらの
# 切り方でもティックの*グループ分け*はたまたま同じになりますが、夏時間の切り替えをまたぐと
# 固定 UTC オフセットは1時間ずれてしまうのに対し、ニューヨーク版は正しいままです。

# %%
(
    db.table("trades")
    .group_by(
        time_bucket("1d", col("ts")).alias("day_utc"),
        time_bucket("1d", col("ts"), timezone="America/New_York").alias("day_new_york"),
    )
    .agg(n_trades=count_star())
    .sort("day_utc")
    .to_pandas()
)

# %% [markdown]
# バケットの選び方がラベルだけでなく*数字*まで変えるのは、UTC の深夜をまたぐセッション、
# つまり夜間の先物、FX、暗号資産です。Globex 型のセッションはニューヨーク18:00（22:00 UTC）に
# 始まるので、素朴な UTC 日はセッションを毎回2つに割ってしまいますし、ニューヨーク深夜の日でも
# 同じことが起きます。直し方は起点を指定する形で、`time_bucket('1d', ts, '<セッション開始>')`
# とすればバケットが取引セッション自体に揃います。違いをはっきりさせるため、夜間セッションを
# 3つ合成します。

# %%
rng = np.random.default_rng(42)
frames = []
for day in ("2026-06-01", "2026-06-02", "2026-06-03"):
    open_ts = pd.Timestamp(f"{day} 22:00", tz="UTC")  # 18:00 New York open
    n = 4_000
    secs = np.sort(rng.uniform(0, 22 * 3600, n))  # ~22h overnight session
    px = 5_000 * np.exp(np.cumsum(rng.normal(0, 2e-5, n)))
    frames.append(
        pd.DataFrame(
            {
                "ts": open_ts + pd.to_timedelta(secs, unit="s"),
                "price": np.round(px * 4) / 4,  # quarter-point ticks
                "size": rng.integers(1, 10, n),
            }
        )
    )
fut_pd = pd.concat(frames).sort_values("ts").reset_index(drop=True)
fut_pd["ts"] = fut_pd["ts"].dt.floor("us")  # pandas is ns-resolution; h5i time is us
fut = pa.Table.from_pandas(fut_pd, preserve_index=False)

fut_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
    ]
)
db.create_table("fut_trades", fut_schema, time_column="ts", sort_key=["ts"])
db.append("fut_trades", fut.cast(fut_schema))

schemes = {
    "UTC day": time_bucket("1d", col("ts")),
    "New York day": time_bucket("1d", col("ts"), timezone="America/New_York"),
    "session day": time_bucket("1d", col("ts"), origin="2026-06-01T22:00:00Z"),
}
pd.concat(
    db.table("fut_trades")
    .group_by(bucket.alias("bar_start"))
    .agg(n_trades=count_star())
    .select(scheme=lit(label), bar_start=col("bar_start"), n_trades=col("n_trades"))
    .sort("bar_start")
    .to_pandas()
    for label, bucket in schemes.items()
).reset_index(drop=True)

# %% [markdown]
# 1セッション4,000約定が3つ。3本のきれいな日足として復元できるのは、セッション起点の
# バケットだけです。暦日ベースの2つは、どのセッションも2つのバケットに割ってしまいます。
#
# ## 4. バーをバージョン管理されたテーブルとして保存する
#
# バーは何度も作り直す派生データセットです。`db.write` はまさにそのためにあります。
# テーブルの中身を1回のアトミックなコミットで*置き換え*ながら、それ以前のビルドを
# バージョン履歴に残します。ティックを訂正したあとにバーを作り直しても、古いバーは
# `h5i('bars_5m', <version>)` でクエリできます。

# %%
bar_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("n_trades", pa.int64()),
    ]
)
db.create_table("bars_5m", bar_schema, time_column="ts", sort_key=["ts", "symbol"])
commit = db.write(
    "bars_5m", bars("5m").sort(["ts", "symbol"]).to_arrow().cast(bar_schema),
    note="5m bars built from trades v1",
)
{k: commit[k] for k in ("table", "sequence", "op", "rows_total")}

# %% [markdown]
# ## 5. ローソク足で見る
#
# 1銘柄・1セッションを、保存したバーのテーブルからそのまま描きます。終値と VWAP に、
# 高値・安値のレンジをバンドとして重ねます。

# %%
one_day = (
    db.table("bars_5m")
    .filter(
        col("symbol") == "AAPL",
        time_bucket("1d", col("ts")) == "2026-06-02T00:00:00Z",
    )
    .select("ts", "high", "low", "close", "vwap")
    .sort("ts")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(one_day["ts"], one_day["low"], one_day["high"],
                alpha=0.25, label="high-low range")
ax.plot(one_day["ts"], one_day["close"], lw=1.2, label="close")
ax.plot(one_day["ts"], one_day["vwap"], lw=1.0, ls="--", label="bar VWAP")
ax.set_title("AAPL 5-minute bars, 2026-06-02")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. pandas と突き合わせる
#
# 信じてよい、ただし検証はする。同じティックから `pandas.Grouper` で5分足を組み直し、
# 全項目を比べます。どちらもタイムスタンプをバケット開始時刻へ切り下げ、始値と終値を
# イベント順で取るので、2つの構築結果は浮動小数点の精度まで一致するはずです。

# %%
tp = trades.to_pandas()
tp["pv"] = tp["price"] * tp["size"]
ref = (
    tp.groupby(["symbol", pd.Grouper(key="ts", freq="5min")])
    .agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        pv=("pv", "sum"),
        n_trades=("price", "count"),
    )
    .reset_index()
)
ref["vwap"] = ref["pv"] / ref["volume"]

stored = db.table("bars_5m").sort(["symbol", "ts"]).to_pandas()
merged = stored.merge(ref, on=["symbol", "ts"], suffixes=("", "_ref"))
assert len(merged) == len(stored) == len(ref)
for field in ("open", "high", "low", "close", "volume", "vwap", "n_trades"):
    assert np.allclose(merged[field], merged[f"{field}_ref"]), field
print(f"all {len(merged):,} bars match pandas across OHLC, volume, VWAP, count")

# %% [markdown]
# ## まとめ
#
# - ティックから OHLCV＋VWAP のバーまでは集約1つです。`time_bucket` と `.first("ts")`
#   `.last("ts")` と `vwap` が時刻順ストレージの上をストリーミングで流れ、結果は pandas の
#   参照実装とぴたり一致します。
# - SQL のテンプレートではなく `bars(width)` という関数として書いたので、どの幅のバーも、
#   その上に載せる本数のカウントも、たった1つの定義から出てきます。
# - `time_bucket` の3つ目の引数がセッションの揃えを担当します。夏時間に強い暦日には
#   `timezone=`、深夜をまたぐ夜間セッションには `origin=` を渡します。素朴な UTC 日は
#   Globex 型のセッションを黙って2つに割ります。
# - 派生したバーはデータベースに置きましょう。`bars_5m` への `db.write` なら、ビルド履歴を
#   丸ごと残したままアトミックに作り直せます。下流の利用者はどのバーのバージョンから計算したかを
#   固定できます。

# %%
db.close()
