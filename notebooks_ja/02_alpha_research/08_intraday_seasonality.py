# %% [markdown]
# # 日中の季節性: 出来高のU字、ボラティリティのスマイル、スプレッドの縮小
#
# 執行のモデルもアルファのモデルも、ほとんどが時刻を条件に取ります。出来高は寄りと引けに
# 集まり、ボラティリティは最初の1時間で高く、スプレッドは午前中に縮んでいきます。
#
# これらのカーブを正しく測るのは、統計の問題である前にタイムゾーンの問題です。「ニューヨーク
# 09:30」は夏時間の切り替わりごとに UTC に対してずれますし、UTC のタイムスタンプに素朴な
# `EXTRACT(hour)` をかけると、年2回、バケットが1時間ぶん滲みます。
#
# `time_bucket` は3つ目の引数に IANA のタイムゾーンを取るので、バケットはそのゾーンの*壁時計*
# に揃います。夏時間も込みです。
#
# カーブは2回測ります。生成器がどの効果を含み、どれを含まないかが分かっている合成ティック
# データと、実際の SPY／QQQ の1時間足です。

# %% [markdown]
# ## ここで使う用語
#
# | 用語           | 意味 |
# | ------------ | --- |
# | 日中の季節性       | 日付をまたいでではなく、時刻に沿って繰り返すパターン |
# | U字（U-shape）  | 出来高の曲線。寄りと引けが厚く、日中の真ん中が薄い |
# | ボラティリティ・スマイル | ここではボラティリティの同じU字のこと。オプションのスマイルとは無関係 |
# | スプレッドの縮小     | 不確実性が晴れるにつれ、午前中を通じて気配スプレッドが狭くなること |
# | セッション        | 取引所が定める1営業日。米国株ならニューヨーク時間 09:30〜16:00 |
# | DST（夏時間）     | セッションは現地時刻で固定なので、UTC からのオフセットが年2回ずれる |
# | IANA タイムゾーン  | `America/New_York` のような名前付きゾーン。DST の規則を自前で持つ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_seasonality"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades_and_quotes` からのフィードが2つ、5セッション、3銘柄です。約定は1行が1
# プリント、気配は1行が最良ビッド・アスクの1変化です。
#
# | テーブル | 列 |
# | --- | --- |
# | `trades` | `ts`、`symbol`、`price`、`size`、`exchange`、`side` |
# | `quotes` | `ts`、`symbol`、`bid`、`ask`、`bid_size`、`ask_size` |
#
# どちらも `ts` は `timestamp[us, tz=UTC]` で昇順です。

# %%
trades, quotes = cu.make_trades_and_quotes(days=5)
print(f"trades: {trades.num_rows:,} rows   quotes: {quotes.num_rows:,} rows")
trades.to_pandas().head()

# %%
quotes.to_pandas().head()

# %% [markdown]
# 実データ側は `cu.fetch_intraday` の60日ぶんの SPY と QQQ の1時間足です。`ts`、`symbol`、
# `open`、`high`、`low`、`close`、`volume` で、1行が1銘柄1時間です。

# %%
bars = cu.fetch_intraday(["SPY", "QQQ"], period="60d", interval="1h").sort_by(
    [("ts", "ascending"), ("symbol", "ascending")]
)
print(f"bars_1h: {bars.num_rows:,} rows x {bars.num_columns} columns")
bars.to_pandas().head()

# %%
for name, tbl in (("trades", trades), ("quotes", quotes)):
    db.create_table(name, tbl.schema, time_column="ts", sort_key=["ts", "symbol"])
    db.append(name, tbl, note="5-day synthetic feed")

# append enforces the declared sort key, so order by (ts, symbol) explicitly
db.create_table("bars_1h", bars.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("bars_1h", bars, note="real SPY/QQQ hourly bars")
{t: len(db.read(t)) for t in db.tables()}

# %% [markdown]
# ## 2. ニューヨークの1日を30分刻みで見る出来高
#
# `time_bucket('30m', ts, 'America/New_York')` はバケットの境界をニューヨークの壁時計に
# 固定します。
#
# 6月の1週間だけなら UTC のバケットと見分けがつきません。3月と11月の夏時間の切り替わりを
# またぐと、ニューヨークのバケットは 09:30、10:00 と貼り付いたままなのに対し、UTC のバケットは
# 標本の途中で1時間ずれ、時刻ごとの平均をどれも2つのぼやけた母集団に割ってしまいます。同じ
# クエリで、1年中正しいわけです。
#
# 壁時計のラベルは、バケットの開始時刻を pandas でニューヨーク時間に変換して作ります。

# %%
NY30 = time_bucket("30m", col("ts"), timezone="America/New_York")

vol30 = (
    db.table("trades")
    .group_by(NY30.alias("bucket"), "symbol")
    .agg(volume=col("size").sum(), n_trades=count_star())
    .sort(["bucket", "symbol"])
    .to_pandas()
)
vol30["tod"] = vol30["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")

profile = (
    vol30.groupby(["tod", "symbol"])["volume"].mean().unstack()  # avg across the 5 days
)
profile_share = profile / profile.sum()

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
for sym in profile_share.columns:
    ax.plot(profile_share.index, profile_share[sym] * 100, marker="o", ms=3, label=sym)
ax.set_title("Synthetic trades: share of daily volume by 30-min NY bucket")
ax.set_xlabel("bucket start (America/New_York)")
ax.set_ylabel("% of daily volume")
ax.tick_params(axis="x", rotation=45)
ax.legend()
fig.tight_layout()

# %% [markdown]
# U字ははっきり出ていますが、それは構造上そうなっています。生成器は到着の40%を寄り付近に、
# 20%を引け付近に集めていて、実際のパターンを真似ています。
#
# 執行スケジューラが欲しがる要約の数字はこちらです。

# %%
first_last = pd.DataFrame(
    {
        "first 30m share": profile_share.loc["09:30"],
        "last 30m share": profile_share.loc["15:30"],
        "midday hour share": profile_share.loc["12:00"] + profile_share.loc["12:30"],
    }
).T
(first_last * 100).round(1)

# %% [markdown]
# ## 3. 時刻ごとのボラティリティとスプレッド: 正直なヌル
#
# 同じバケット化を30分足のリターンにかければボラティリティのスマイル、気配にかければスプレッド
# の縮小が出ます。
#
# ここで合成データは現実の効果を*出せません*。理由をはっきりさせておく価値があります。生成器の
# 価格はカレンダー時間の一様な拡散過程なので、分散は約定の密度に関係なく経過時間に比例します
# し、スプレッドは銘柄・日ごとに1度だけ引かれます。
#
# だから正直な予想は*平らな*ボラティリティのカーブと*平らな*スプレッドのカーブです。実際に
# そうなりますし、これは有用なプラセボになります。ここでパイプラインが曲がりを見せたら、それは
# パイプラインのバグです。

# %%
smile = (
    db.table("trades")
    .group_by(NY30.alias("bucket"), "symbol")
    .agg(open=col("price").first("ts"), close=col("price").last("ts"))
    .select("bucket", "symbol", bar_ret=col("close") / col("open") - 1)
    .sort("bucket")
    .to_pandas()
)
smile["tod"] = smile["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
vol_by_tod = smile.groupby("tod")["bar_ret"].std() * 1e4

# each bucket std rests on only 15 bar returns (5 days x 3 symbols), so test
# flatness properly: standardize per symbol, then Levene's test across buckets
from scipy import stats

smile["z"] = smile.groupby("symbol")["bar_ret"].transform(lambda s: s / s.std())
levene = stats.levene(*[g["z"].values for _, g in smile.groupby("tod")])
print(f"Levene test for equal variance across buckets: p = {levene.pvalue:.2f}")

spread = (
    db.table("quotes")
    .group_by(NY30.alias("bucket"), "symbol")
    .agg(spread_bps=((col("ask") - col("bid")) / ((col("ask") + col("bid")) / 2)).mean() * 10000)
    .sort("bucket")
    .to_pandas()
)
spread["tod"] = spread["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
spread_by_tod = spread.groupby("tod")["spread_bps"].mean()

pd.DataFrame(
    {
        "bar-return std (bps)": vol_by_tod,
        "quoted spread (bps)": spread_by_tod,
    }
).loc[["09:30", "10:30", "12:30", "14:30", "15:30"]].round(2)

# %% [markdown]
# 予想どおり平らです。バケットごとの生の標準偏差は揺れますが、これは各バケットが15本のバー
# リターンに載っているためで、Levene 検定はバケット間の分散の不均一を支持しません。スプレッド
# の列はベーシスポイント単位で一定です。
#
# 手法は検証されましたし、効果はこの生成器に存在しません。実際の市場では最初の1時間の
# ボラティリティが2〜3倍になり、スプレッドは広く始まって数分で締まります。そのうちボラティリティ
# のほうは、次に実データで見ます。
#
# ついでにタイムゾーンの注意をもう1つ。`'1d'` の幅では、タイムゾーンの引数が*日*の境界の位置を
# 決めます。24時間動く資産クラスでは、それがどの約定を「月曜」に入れるかを決めます。

# %%
(
    db.table("trades")
    .group_by(
        time_bucket("1d", col("ts")).alias("day_utc"),
        time_bucket("1d", col("ts"), timezone="America/New_York").alias("day_ny"),
    )
    .agg(volume=col("size").sum())
    .sort("day_utc")
    .limit(3)
    .to_pandas()
)

# %% [markdown]
# 米国株ならセッションは1つの UTC 日の中に収まるので、どちらのグループ分けも一致し、違うのは
# ラベルだけです。ニューヨークの0時は 04:00 UTC にあたります。00:00 UTC をまたいで流れる FX や
# 暗号資産では、この選択が日々の数字をすべて変えます。24/7 市場のレシピがその場合を扱います。
#
# ## 4. 本物: SPY と QQQ、60日ぶんの1時間足
#
# バーごとのリターンは SQL の `lag()` から出し、そのあと同じ壁時計のグループ分けをかけます。
#
# 正直に書いておく注意が1つ。このキャッシュ標本は4月末から7月なので夏時間の切り替わりを
# またぎません。だからここでは UTC のバケット化でもたまたま通ります。ニューヨークに固定した版
# は、3月になっても通り続けるほうです。

# %%
real = (
    db.table("bars_1h")
    .select(
        "ts", "symbol", "volume",
        ret=col("close") / sql_expr("lag(close)").over(partition_by="symbol", order_by="ts") - 1,
    )
    .sort("ts")
    .to_pandas()
)
real["tod"] = real["ts"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
# first bar of each day carries an overnight return -> exclude from the smile
real.loc[real["tod"] == "09:30", "ret"] = np.nan

by_tod = real.groupby(["tod", "symbol"]).agg(
    avg_volume=("volume", "mean"), ret_std=("ret", "std")
)
share = by_tod["avg_volume"].unstack()
share = share / share.sum()
vol_curve = by_tod["ret_std"].unstack() * 1e4

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for sym in share.columns:
    axes[0].plot(share.index, share[sym] * 100, marker="o", label=sym)
    axes[1].plot(vol_curve.index, vol_curve[sym], marker="o", label=sym)
axes[0].set_title("Real hourly bars: volume share by NY hour")
axes[0].set_ylabel("% of daily volume")
axes[1].set_title("Real hourly bars: return std by NY hour")
axes[1].set_ylabel("hourly return std (bps)")
for ax in axes:
    ax.set_xlabel("bar start (America/New_York)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
fig.tight_layout()

print("SPY first-hour volume share: {:.1%}   last-hour: {:.1%}".format(
    share["SPY"].iloc[0], share["SPY"].iloc[-1]))

# %% [markdown]
# 実データのカーブは、合成ティックが出せなかったものを届けます。U字型の出来高プロファイルと、
# 寄り後に高く、正午へ向けて減衰していくボラティリティです。
#
# 09:30 のバーはスマイルから除いてあります。`lag()` のリターンがオーバーナイトのギャップを
# またぐからです。脚注ではなくパイプラインに書き込む価値のある、古典的な細部です。
#
# ## まとめ
#
# - `time_bucket(width, ts, 'America/New_York')` が、時刻別の統計を夏時間に強く作る方法です。
#   バケットが壁時計のセッションに貼り付くので、3月と11月が季節性のカーブを滲ませません。
#   `'1d'` の幅では、同じ引数が取引日の始まりを決めます。
# - パイプライン全体はカーブ1本あたり GROUP BY 2つです。出来高もバーリターンもスプレッドも、
#   数十万のティックからデータベースの中で数十のバケットに畳まれます。
# - 合成データはプラセボであって代用品ではありません。生成器に組み込まれた出来高のU字は再現し、
#   生成器に存在しないボラティリティとスプレッドのカーブは正しく*平ら*に出ました。ボラティリティ
#   のスマイルは実際の SPY／QQQ のバーが供給しました。
# - オーバーナイトのリターンを 09:30 のバケットから除くのは、使える季節性カーブと、微妙に
#   間違ったカーブを分ける1行の正しさです。

# %%
db.close()
