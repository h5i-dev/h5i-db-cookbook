# %% [markdown]
# # FX と暗号資産: 取引所のセッションがない 24/7 のデータ
#
# 株式まわりの道具立てはセッションに寄りかかっています。取引所が「1日」を、寄りを、引けを
# 定義してくれます。
#
# FX と暗号資産にはそれがありません。EURUSD は週末の休みを挟みながら24時間動き、BTC は止まる
# ことがなく、「日次の終値」はデスクの*取り決め*です。ニューヨークの17時か、東京カットか、
# UTC の0時か。
#
# このレシピはその世界のための道具立てを見せます。セッションの視点を作る IANA タイムゾーン付き
# の `time_bucket`、24時間のローリングボラティリティを出す行ベースのウィンドウ、そして静かな
# 時間帯をまたいで規則正しい格子を作る `gapfill` です。
#
# **データについて正直に。** 合成のティック生成器は到着を一様に出すだけで、東京／ロンドン／
# ニューヨークの流動性の時計も、週末の休みも持ちません。FX の週末は明示的にモデル化し、FX ペア
# から金曜 21:00 〜 日曜 21:00 UTC のティックを取り除きつつ、BTC はそのまま動かします。以下では、
# 実データなら見えるはずの構造がこのデータには出せない箇所に印を付けます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語          | 意味 |
# | ----------- | --- |
# | セッション       | 取引所が定める1営業日。FX と暗号資産にはこれが存在しない |
# | 日次クローズの慣行   | デスクが終値と呼ぶことに決めた時刻。NY 17時、東京カット、UTC 0時など |
# | IANA タイムゾーン | `America/New_York` のような名前付きゾーン。DST の規則を自前で持つ |
# | ローリングウィンドウ  | 直近 N 行または N 時間の統計量を、行ごとに計算し直したもの |
# | gapfill     | 不規則な系列を規則的な時間グリッドに載せる |
# | 週末の休止       | FX は金曜夕方から日曜夕方まで止まる。暗号資産は止まらない |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_fx"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_fx_ticks` が返すのは 24/7 の FX・暗号資産型のティックで、1行が1回の気配更新です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 気配の時刻、昇順 |
# | `pair` | `string` | 銘柄。`EURUSD`、`USDJPY`、`BTCUSD` |
# | `bid`、`ask` | `float64` | 最良のビッドとオファー |
#
# ここでは金曜 2026-06-05 00:00 UTC から96時間ぶんを取ります。週末をまるごと含みます。

# %%
raw = cu.make_fx_ticks(pairs=["EURUSD", "USDJPY", "BTCUSD"], hours=96, start="2026-06-05").to_pandas()
print(f"{len(raw):,} rows x {raw.shape[1]} columns")
raw.head()

# %% [markdown]
# そのあと FX の週末を手で切り出します。生成器には週末がないからです。FX ペアは金曜 21:00 〜
# 日曜 21:00 UTC のティックを失い、BTC は動き続けます。

# %%
wk_open = pd.Timestamp("2026-06-05 21:00", tz="UTC")   # Fri 5pm ET
wk_close = pd.Timestamp("2026-06-07 21:00", tz="UTC")  # Sun 5pm ET
is_weekend = (raw["ts"] >= wk_open) & (raw["ts"] < wk_close)
ticks_df = raw[~(is_weekend & (raw["pair"] != "BTCUSD"))].reset_index(drop=True)
print(f"{len(raw):,} generated ticks -> {len(ticks_df):,} after removing the FX weekend")

# %% [markdown]
# ## 2. ティックを保存する
#
# 3銘柄ぶんをまとめて1つの `ticks` テーブルに入れます。FX の週末が日ごとの件数の穴として現れる
# 一方、BTC はその中をそのままプリントします。複数資産クラスの本が付き合わされる非対称性が、
# まさにこれです。

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pair", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)
db.create_table("ticks", schema, time_column="ts", sort_key=["ts", "pair"])
db.append("ticks", pa.Table.from_pandas(ticks_df, schema=schema, preserve_index=False),
          note="4 days of FX/crypto ticks, FX weekend removed")

MID = (col("bid") + col("ask")) / 2

(
    db.table("ticks")
    .group_by(time_bucket("1d", col("ts")).alias("day_utc"), "pair")
    .agg(ticks=count_star())
    .sort(["day_utc", "pair"])
    .to_pandas()
).pivot(index="day_utc", columns="pair", values="ticks")

# %% [markdown]
# ## 3. 「1日」とは何か。同じティックから3つの答え
#
# `time_bucket('1d', ts, tz)` は、日の境界を任意の IANA タイムゾーンの*現地の0時*に揃えます。
# 夏時間も面倒を見てくれます。
#
# 同じティックの流れが、UTC・東京・ニューヨークの取り決めごとに違う日次終値を返します。24/7 の
# 流れをそれぞれ違う瞬間で切るからです。
#
# FX でよく使われる「ニューヨーク17時」のカットは0時ではなく現地の 17:00 です。その取り決めが
# 必要なら、`time_bucket` の任意の origin 引数で境界をずらせます。

# %%
def daily_close(timezone, label: str) -> pd.DataFrame:
    out = (
        db.table("ticks")
        .filter(col("pair") == "EURUSD")
        .group_by(time_bucket("1d", col("ts"), timezone=timezone).alias("bucket"))
        .agg(close=MID.last("ts"))
        .sort("bucket")
        .to_pandas()
    )
    out[label + "_boundary_utc"] = out["bucket"].dt.strftime("%m-%d %H:%M")
    return out.rename(columns={"close": label})[[label + "_boundary_utc", label]]


utc = daily_close(None, "close_utc")
tokyo = daily_close("Asia/Tokyo", "close_tokyo")
ny = daily_close("America/New_York", "close_ny")
pd.concat([utc, tokyo, ny], axis=1).round(5)

# %% [markdown]
# 取り決めごとに1日の始まりが違う UTC の瞬間になります。UTC なら 00:00、東京の0時なら 15:00、
# 夏時間のニューヨークの0時なら 04:00。だから「同じ日」の終値が列によって違います。
#
# 実際のデスクではこれは細かすぎる話ではありません。日次の P&L も VaR の窓もキャリーの計上も、
# どのカットを選んだかを引き継ぎます。
#
# ## 4. 時間足を保存し、そのうえで24時間ローリングの実現ボラティリティ
#
# 時間足のミッドのバーを、それ自体のテーブルに保存します。派生テーブルはここでは安上がりで、
# コミット1件ずつですし、ローリングウィンドウや gapfill のような下流のクエリは*保存された*
# 規則正しい系列の上で働きたがります。
#
# 24時間ローリングのボラティリティは行ベースのウィンドウ、`ROWS BETWEEN 23 PRECEDING` です。
# どの1時間も存在する 24/7 のデータでは、これが自然な枠です。
#
# 注意点が1つ、FX の週末から来ます。欠けた時間は単に行として存在しないので、ウィンドウは黙って
# その穴をまたぎます。月曜の最初のバーは金曜の時間を「直近24時間」に混ぜてしまうのです。それが
# 効いてくる場面では、6節の規則正しい格子が答えになります。

# %%
bars = (
    db.table("ticks")
    .group_by(time_bucket("1h", col("ts")).alias("hr"), "pair")
    .agg(mid_close=MID.last("ts"), avg_spread=(col("ask") - col("bid")).mean(), ticks=count_star())
    .select(col("hr").alias("ts"), "pair", "mid_close", "avg_spread", "ticks")
    .sort(["ts", "pair"])
    .to_arrow()
)

bars_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)]
    + [bars.schema.field(i) for i in range(1, bars.num_columns)]
)
db.create_table("bars_1h", bars_schema, time_column="ts", sort_key=["ts", "pair"])
db.append("bars_1h", bars.cast(bars_schema), note="hourly mid bars")

PREV_MID = sql_expr("lag(mid_close)").over(partition_by="pair", order_by="ts")

rv = (
    db.table("bars_1h")
    .with_columns(lr=(col("mid_close") / PREV_MID).log())
    .select(
        "ts", "pair",
        rv_ann=(
            (col("lr") * col("lr")).rolling_sum(24, order_by="ts", partition_by="pair") * 365.0
        ).sqrt(),
    )
    .sort(["ts", "pair"])
    .to_pandas()
).dropna()
rv.tail(3)

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for pair, g in rv.groupby("pair"):
    ax.plot(g["ts"], 100 * g["rv_ann"], lw=1.0, label=pair)
ax.axvspan(wk_open, wk_close, color="0.85", zorder=0, label="FX weekend")
ax.set_title("Rolling 24h realized vol (hourly bars, annualized)")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("realized vol (%)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 5. 流動性の時計: 合成データでは手法、実データでは構造
#
# 24/7 の1週間にもリズムはあります。東京の朝、FX のスプレッドがいちばん締まる 12:00〜16:00 UTC
# あたりのロンドンとニューヨークの重なり、そしてニューヨーク後の凪。
#
# クエリはティックとスプレッドを時刻でバケットするだけですし、その時刻を取引拠点の現地時間に
# 直すのは pandas の `tz_convert` 1つです。
#
# **以下の線は平らになるはずです。** 一様な合成の到着に時計はありません。実際のフィードでは、
# この同じクエリがU字と重なりの山を描きます。持ち帰るべきはこの絵ではなく手法です。

# %%
clock = (
    db.table("ticks")
    .group_by(sql_expr("extract(hour FROM ts)").alias("hour_utc"), "pair")
    .agg(ticks=count_star(), avg_spread=(col("ask") - col("bid")).mean())
    .sort(["hour_utc", "pair"])
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
for pair, g in clock.groupby("pair"):
    ax.plot(g["hour_utc"], g["ticks"] / 4.0, lw=1.0, marker="o", ms=3, label=pair)
ax.axvspan(12, 16, color="0.85", zorder=0, label="London/NY overlap")
ax.set_title("Ticks per hour-of-day (avg/day) - flat by construction on synthetic data")
ax.set_xlabel("hour of day (UTC)")
ax.set_ylabel("avg ticks per hour")
ax.set_xticks(range(0, 24, 3))
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 6. `gapfill` による規則正しい格子と、幻のリターンの罠
#
# `gapfill(table, time_col, step, mode)` は*保存された*テーブルを規則正しい格子の上で標本化し
# ます。体に入れておきたいことがいくつかあります。
#
# - **刻みは生の時間単位**、つまり `timestamp[us]` の列ならマイクロ秒です。`1_000_000` が1秒の
#   格子、`60_000_000` が1分。生成される格子は100万行で頭打ちなので、期間に合うように刻みを
#   選んでください。
# - gapfill は**キーごとのグループ化を持たない**素の格子標本化です。複数銘柄のテーブルにかけると、
#   `locf` は*最後にティックしたペアが何であれ*それを持ち越します。必ず単一系列のテーブルを
#   gapfill してください。だからまず EURUSD のティックを単独で保存します。
# - `'null'` モードが値を出すのは、観測が格子の瞬間とちょうど一致したときだけです。不規則な
#   ティックではまず起きません。標本化には `'locf'` か `'interpolate'` を使い、「実際に約定した
#   分だけ」が欲しいときは素の `time_bucket` のバーを使ってください。

# %%
eur = ticks_df[ticks_df["pair"] == "EURUSD"]
db.create_table("eur_ticks", schema, time_column="ts", sort_key=["ts", "pair"])
db.append("eur_ticks", pa.Table.from_pandas(eur, schema=schema, preserve_index=False),
          note="EURUSD only, for gapfill")

# A research-grade 1-second grid: step = 1_000_000 us.
n_1s = db.sql("SELECT count(*) AS n FROM gapfill('eur_ticks', 'ts', 1000000, 'locf')").to_pandas()["n"][0]
print(f"1s locf grid over 96h: {n_1s:,} rows (cap is 1M - a 1s grid fits ~11 days)")

# A 1-minute locf grid, plus honest 1m bars (only minutes that actually traded):
grid_locf = db.sql("SELECT ts, bid, ask FROM gapfill('eur_ticks', 'ts', 60000000, 'locf')").to_pandas()
grid_locf["mid"] = (grid_locf["bid"] + grid_locf["ask"]) / 2
bars_1m = (
    db.table("eur_ticks")
    .group_by(time_bucket("1m", col("ts")).alias("minute"))
    .agg(mid=MID.last("ts"))
    .sort("minute")
    .to_pandas()
)
print(f"1m locf grid: {len(grid_locf):,} rows; traded minutes: {len(bars_1m):,}; "
      f"fabricated by locf: {len(grid_locf) - len(bars_1m):,}")

# %%
fig, ax = plt.subplots(figsize=(10, 4))
plot_bars = bars_1m.set_index("minute")["mid"].reindex(grid_locf["ts"])  # NaN = untraded minute
ax.plot(grid_locf["ts"], plot_bars.values, lw=0.8, label="1m mid (traded minutes only)")
wk = grid_locf[(grid_locf["ts"] >= wk_open) & (grid_locf["ts"] < wk_close)]
ax.plot(wk["ts"], wk["mid"], lw=1.4, ls="--", label="locf across the halt")
ax.axvspan(wk_open, wk_close, color="0.85", zorder=0)
ax.set_title("EURUSD mid on a 1-minute grid: locf carries Friday's quote across the weekend")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("mid")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# 破線の区間は*でっち上げの*平らな線です。それが `locf` の意味です。本を評価するには正しい選択
# ですが（最後の気配が最良の知識**である**ため）、リターンの統計への入力としては間違いです。
# 2,880個のゼロリターンが測定されるボラティリティを押し下げますし、週末の値動き全体が再開時の
# 1分足の偽のプリント1本に着地します。この引きでは控えめですが、荒れた週末のあとなら、断然
# 最大の「1分」リターンになります。

# %%
ret_locf = np.log(grid_locf["mid"]).diff().dropna()
ret_traded = np.log(bars_1m["mid"]).diff().dropna()  # untraded minutes never enter

ann = np.sqrt(365 * 24 * 60)
pd.DataFrame(
    {
        "series": ["locf grid", "traded minutes only"],
        "n_returns": [len(ret_locf), len(ret_traded)],
        "ann_vol_pct": [100 * ret_locf.std() * ann, 100 * ret_traded.std() * ann],
        "max_abs_1m_ret_bp": [1e4 * ret_locf.abs().max(), 1e4 * ret_traded.abs().max()],
    }
).round(2)

# %% [markdown]
# ゼロリターンの水増しは測定されるボラティリティを3分の1削り、週末のギャップは合成の1分に圧縮
# されます。約定した分だけの系列、つまりティックのあるところにしか存在しない素の `time_bucket`
# のバーが、ボラティリティ推定への正直な入力です。
#
# ## まとめ
#
# - 24/7 のデータでは「1日」は事実ではなくパラメータです。`time_bucket('1d', ts, '<IANA tz>')`
#   がどの取り決めにもセッションに揃った日を与えます。夏時間込みで。同じティック、違う日次終値。
# - 行ベースのウィンドウ（`ROWS BETWEEN 23 PRECEDING`）は連続データでのローリングの自然な枠です
#   が、データの*穴*を黙ってまたぎます。自分の市場の停止時間を把握してください。
# - `gapfill` には保存された単一系列のテーブルが要ります。刻みは生のマイクロ秒（`1_000_000` が
#   1秒）で、格子は100万行が上限です。
# - `locf` は評価のため、約定分のバーは測定のためです。停止をまたいで気配を持ち越すと、ゼロ
#   リターンをでっち上げてボラティリティを下に歪め、幻のギャップのプリント1本で裾を上に歪めます。
# - 合成のティックに流動性の時計はありません。ここでの時刻別のクエリは、実際のフィードに対して
#   走らせるべき手法です。そこでは東京・ロンドン・ニューヨークの構造が実際に現れます。

# %%
db.close()
