# %% [markdown]
# # 日中の季節性: 出来高のU字、ボラティリティのスマイル、スプレッドの収束
#
# 執行モデルもアルファモデルも、ほとんどが時刻を条件に取ります。出来高は寄り付きと引けに
# 集まり、ボラティリティは最初の1時間で山を作り、スプレッドは午前中に狭まっていく。これらの
# 曲線を正しく測る作業は、統計の問題である前にタイムゾーンの問題です。「ニューヨーク9時30分」は
# 夏時間の切り替えのたびに UTC に対してずれますし、UTC のタイムスタンプに素朴な
# `EXTRACT(hour)` をかけると、年に2回、バケットが1時間ぶんにじみます。h5i-db の `time_bucket` は
# 3つ目の引数に IANA のタイムゾーンを取るので、バケットはそのゾーンの*壁時計*時刻に、夏時間も
# 含めて揃います。
#
# 曲線は2回測ります。まず合成ティックデータで（生成器がどの効果を含み、どれを含まないかが
# 分かっているので、誠実さの検問所になります）、次に実際の SPY／QQQ の1時間足でです。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_seasonality"), create=True)

# %% [markdown]
# ## 1. 5日ぶんのティックと、60日ぶんの実データの足

# %%
trades, quotes = cu.make_trades_and_quotes(days=5)
for name, tbl in (("trades", trades), ("quotes", quotes)):
    db.create_table(name, tbl.schema, time_column="ts", sort_key=["ts", "symbol"])
    db.append(name, tbl, note="5-day synthetic feed")

# append enforces the declared sort key, so order by (ts, symbol) explicitly
bars = cu.fetch_intraday(["SPY", "QQQ"], period="60d", interval="1h").sort_by(
    [("ts", "ascending"), ("symbol", "ascending")]
)
db.create_table("bars_1h", bars.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("bars_1h", bars, note="real SPY/QQQ hourly bars")
{t: len(db.read(t)) for t in db.tables()}

# %% [markdown]
# ## 2. ニューヨーク時間30分刻みの出来高
#
# `time_bucket('30m', ts, 'America/New_York')` はバケットの境界をニューヨークの壁時計時刻に
# 固定します。6月の1週間の中では UTC でのバケットと区別が付きません。ところが3月と11月の
# 夏時間切り替えをまたぐと、ニューヨークのバケットは9:30、10:00……に留まり続けるのに対し、
# UTC のバケットはサンプルの途中で1時間ずれて、時刻ごとの平均をぼやけた2つの集団に割ってしまい
# ます。同じクエリで、1年中正しい。壁時計のラベルは、バケット開始時刻を pandas でニューヨーク
# 時間に変換して付けます。

# %%
vol30 = db.sql(
    """
    SELECT time_bucket('30m', ts, 'America/New_York') AS bucket,
           symbol,
           sum(size)  AS volume,
           count(*)   AS n_trades
    FROM trades
    GROUP BY bucket, symbol
    ORDER BY bucket, symbol
    """
).to_pandas()
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
# U字ははっきり出ています。構造上そうなります。生成器が到着の40%を寄り付き付近に、20%を引け
# 付近に集めていて、実際のパターンを真似ているからです。執行スケジューラにとって役に立つ要約は
# 次のとおりです。

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
# ## 3. 時刻別のボラティリティとスプレッド — 正直な「何もなし」
#
# 同じバケット分けを、30分足のリターン（ボラティリティのスマイル）とクオートスプレッドに
# 当てます。ここで合成データは現実世界の効果を*出せません*。その理由をはっきりさせておく価値が
# あります。生成器の価格は暦時間に対して一様な拡散過程（分散は経過時間に比例し、約定の
# 頻度には依りません）で、スプレッドは銘柄・日ごとに1回引くだけだからです。したがって誠実な
# 期待は、*平坦な*ボラティリティ曲線と*平坦な*スプレッド曲線です。実際そうなりますし、有用な
# プラセボにもなります。ここでパイプラインが曲率を見せたなら、それはパイプラインのバグです。

# %%
smile = db.sql(
    """
    WITH bars30 AS (
        SELECT time_bucket('30m', ts, 'America/New_York') AS bucket, symbol,
               first_value(price ORDER BY ts) AS open,
               last_value(price ORDER BY ts)  AS close
        FROM trades
        GROUP BY bucket, symbol
    )
    SELECT bucket, symbol, close / open - 1 AS bar_ret
    FROM bars30 ORDER BY bucket
    """
).to_pandas()
smile["tod"] = smile["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
vol_by_tod = smile.groupby("tod")["bar_ret"].std() * 1e4

# each bucket std rests on only 15 bar returns (5 days x 3 symbols), so test
# flatness properly: standardize per symbol, then Levene's test across buckets
from scipy import stats

smile["z"] = smile.groupby("symbol")["bar_ret"].transform(lambda s: s / s.std())
levene = stats.levene(*[g["z"].values for _, g in smile.groupby("tod")])
print(f"Levene test for equal variance across buckets: p = {levene.pvalue:.2f}")

spread = db.sql(
    """
    SELECT time_bucket('30m', ts, 'America/New_York') AS bucket, symbol,
           avg((ask - bid) / ((ask + bid) / 2)) * 10000 AS spread_bps
    FROM quotes
    GROUP BY bucket, symbol
    ORDER BY bucket
    """
).to_pandas()
spread["tod"] = spread["bucket"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
spread_by_tod = spread.groupby("tod")["spread_bps"].mean()

pd.DataFrame(
    {
        "bar-return std (bps)": vol_by_tod,
        "quoted spread (bps)": spread_by_tod,
    }
).loc[["09:30", "10:30", "12:30", "14:30", "15:30"]].round(2)

# %% [markdown]
# 予想どおり平坦です。バケットごとの生の標準偏差は揺れますが（それぞれ15本の足のリターンに
# 乗っているだけです）、Levene の検定はバケット間の分散が等しくないという証拠を見つけません。
# スプレッドの列はベーシスポイント単位で一定です。手法は検証できました。効果はこの生成器には
# ありません。実際の市場では最初の1時間のボラが2〜3倍になり、スプレッドは広く始まって数分で
# 締まります。そのうちボラティリティのほうは、次に実データで見ます。
#
# ついでにタイムゾーンの話をもう1つ。幅が `'1d'` のとき、タイムゾーン引数は*日*の境界がどこに
# 落ちるかを決めます。24時間動く資産クラスでは、これが「月曜」に属する約定を決めることになります。

# %%
db.sql(
    """
    SELECT time_bucket('1d', ts)                     AS day_utc,
           time_bucket('1d', ts, 'America/New_York') AS day_ny,
           sum(size)                                 AS volume
    FROM trades
    GROUP BY day_utc, day_ny
    ORDER BY day_utc
    LIMIT 3
    """
).to_pandas()

# %% [markdown]
# 米国株ならセッションは1つの UTC 日の内側に収まるので、どちらのグループ分けも一致し、違うのは
# ラベルだけです（ニューヨークの深夜＝04:00 UTC）。00:00 UTC をまたいで流れる FX や暗号資産では、
# この選択が日次の数字をすべて変えます。24/7 市場のレシピを参照してください。
#
# ## 4. 本物: SPY と QQQ の1時間足60日ぶん
#
# 足ごとのリターンは SQL の `lag()` で出し、そこに同じ壁時計のグループ分けを当てます。1つ正直に
# しておくと、このキャッシュ済みサンプル（4月下旬から7月）は夏時間の切り替えをまたぎません。
# だからここでは UTC でのバケット分けもたまたま動きます。3月になっても動き続けるのは、ニュー
# ヨークに固定したほうです。

# %%
real = db.sql(
    """
    SELECT ts, symbol, volume,
           close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
    FROM bars_1h
    ORDER BY ts
    """
).to_pandas()
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
# 実データの曲線は、合成ティックが出せなかったものを届けます。U字型の出来高プロファイル*と*、
# 寄り付き後に高く、正午へ向けて減衰していくボラティリティです（9:30 の足はスマイルから除いて
# います。その `lag()` リターンがオーバーナイトのギャップをまたぐからです。脚注ではなく
# パイプラインに書き込んでおくべき、古典的な細部です）。
#
# ## まとめ
#
# - `time_bucket(width, ts, 'America/New_York')` が、時刻別の統計を夏時間に強く作る方法です。
#   バケットが壁時計のセッションに固定されるので、3月と11月が季節性の曲線をにじませません。
#   幅が `'1d'` のときは、同じ引数が取引日の始まりを決めます。
# - パイプライン全体は、曲線1本につき GROUP BY のクエリ2つです。出来高も、足のリターンも、
#   スプレッドも、数十万のティックからデータベースの中で数十のバケットに畳まれます。
# - 合成データはプラセボであって代用品ではありません。出来高のU字は再現し（生成器に組み込まれて
#   います）、ボラティリティとスプレッドの曲線は正しく*平坦*でした（生成器にありません）。
#   ボラティリティのスマイルを供給したのは、実際の SPY／QQQ の足です。
# - 9:30 のバケットからオーバーナイトのリターンを除くこと。使える季節性の曲線と、微妙に間違った
#   曲線を分けるのは、この種の1行の正しさです。

# %%
db.close()
