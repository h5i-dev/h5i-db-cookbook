# %% [markdown]
# # 債券: バージョン管理されたカーブの評価、訂正、キャリーとロールダウン
#
# 金利デスクの中核データセットは小さいのに容赦がありません。評価日ごとにパーカーブが1本、その
# 上のどの数字もリスクと P&L と顧客向けの評価に流れていきます。
#
# ここで自然な配置が*ロング形式*、`(ts, tenor_years, yield_pct)` で、評価日ごとにコミット1件
# です。これで3つが同時に手に入ります。あらゆるカーブ分析のための SQL のピボット、EOD の評価
# セットごとのバージョン（規制当局が実際に尋ねてくる監査証跡）、そして不良コントリビュータが
# すり抜けたときのプレビューできる訂正です。
#
# このレシピで進めるのは次の4つです。
#
# 1. 250営業日ぶんのカーブを保存する。わざと仕込んだ、桁を打ち間違えた10年の評価を含めて
# 2. 傾きと曲率の分析を走らせる
# 3. 悪い評価を SQL で*見つけ*、`plan_replace_range` で訂正する
# 4. 訂正前の眺めがいまも永久に引けることを示す

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_curves"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_yield_curves` が返すのは Nelson-Siegel のダイナミクスを持つ日次のパーイールドカーブ
# で、ロング形式です。1行が1評価日1テナーで、3か月から30年まで10のテナーがあります。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 評価日 |
# | `tenor_years` | `float64` | 年単位のテナー: 0.25、0.5、1、2、3、5、7、10、20、30 |
# | `yield_pct` | `float64` | パーイールド、パーセント |

# %%
curves = cu.make_yield_curves(days=250)
clean_df = curves.to_pandas()
print(f"{len(clean_df):,} rows x {clean_df.shape[1]} columns, "
      f"{clean_df['ts'].nunique()} mark dates")
clean_df.head()

# %% [markdown]
# ここで事故を仕込みます。評価日 #120 で10年が 25bp 高くプリントされます。古いままのコント
# リビュータならこうなる、という形です。データベースに実際に着地するのはこちらです。

# %%
mark_dates = clean_df["ts"].unique()
feed_df = clean_df.copy()
bad_date = pd.Timestamp(mark_dates[120])
bad_mask = (feed_df["ts"] == bad_date) & (feed_df["tenor_years"] == 10.0)
feed_df.loc[bad_mask, "yield_pct"] += 0.25
print(f"vendor feed: {len(feed_df):,} rows, bad 10y mark on {bad_date.date()}")

# %% [markdown]
# ## 2. 評価日1つがコミット1件
#
# その日の10テナーをまとめて追記するので、コミットが*そのまま*評価セットになります。
#
# `versions()` はデスクの評価日誌のように読めます。連番、行数、そして評価日を名指しするノート。
# ループで250コミットは構いません。公表イベントごとにコミット1件であることに変わりはなく、
# h5i-db が期待するまとめ方だからです。

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("tenor_years", pa.float64()),
        pa.field("yield_pct", pa.float64()),
    ]
)
db.create_table("curves", schema, time_column="ts", sort_key=["ts", "tenor_years"])

for day_ts, day_rows in feed_df.groupby("ts"):
    db.append("curves", pa.Table.from_pandas(day_rows, schema=schema, preserve_index=False),
              note=f"EOD marks {day_ts.date()}")

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("curves")[-3:]
]

# %% [markdown]
# ## 3. カーブ分析を SQL のピボットとして
#
# ロング形式は条件付き集約できれいにピボットできます。1行が1評価日、1列が1つのベンチマーク
# テナーです。
#
# そこから先、デスクの定番の時系列はただの算術です。2s10s の傾き、曲率を見る 2s5s10s の
# バタフライ、そして10年引く3か月という素朴なターム構造プレミアムの代理変数。本物のターム
# プレミアムには期待の モデルが要ります。これは長期側の傾きという観測量にすぎません。

# %%
# Long-to-wide by conditional aggregation. As a dict of tenors it is one
# definition instead of five near-identical CASE arms - add a tenor by adding
# an entry.
KEY_TENORS = {"y3m": 0.25, "y2": 2, "y5": 5, "y10": 10, "y30": 30}

key_rates = (
    db.table("curves")
    .group_by("ts")
    .agg(**{
        name: when(col("tenor_years") == t).then(col("yield_pct")).max()
        for name, t in KEY_TENORS.items()
    })
    .sort("ts")
    .to_pandas()
)
key_rates["s2s10"] = key_rates["y10"] - key_rates["y2"]
key_rates["fly_2_5_10"] = 2 * key_rates["y5"] - key_rates["y2"] - key_rates["y10"]
key_rates["slope_proxy"] = key_rates["y10"] - key_rates["y3m"]
num_cols = key_rates.columns.drop("ts")
key_rates.tail(3).round(dict.fromkeys(num_cols, 4))

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(key_rates["ts"], 100 * key_rates["s2s10"], lw=1.0, label="2s10s slope")
ax.plot(key_rates["ts"], 100 * key_rates["fly_2_5_10"], lw=1.0, label="2s5s10s fly")
ax.axvline(bad_date, color="0.6", ls=":", lw=1)
ax.annotate("suspicious print", xy=(bad_date, 100 * key_rates.loc[key_rates["ts"] == bad_date, "s2s10"].iloc[0]),
            xytext=(10, 15), textcoords="offset points", fontsize=8)
ax.set_title("Curve shape time series - note the one-day 2s10s spike")
ax.set_xlabel("mark date")
ax.set_ylabel("bp")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 4. ウィンドウ関数で悪い評価を見つける
#
# 古いままのコントリビュータは、1日の往復として現れます。前日比の大きな変化と、同じテナーでの
# その巻き戻しです。
#
# すべての (テナー, 日付) の組について |Δ利回り| を順位づければ、すぐ浮かび上がります。

# %%
PREV_Y = sql_expr("lag(yield_pct)").over(partition_by="tenor_years", order_by="ts")

(
    db.table("curves")
    .with_columns(d1=col("yield_pct") - PREV_Y)
    .filter(col("d1").is_not_null())
    .select(
        "ts", "tenor_years",
        yield_pct=col("yield_pct").round(4),
        chg_bp=(100 * col("d1")).round(1),
        abs_move=col("d1").abs(),
    )
    .sort("abs_move", descending=True)
    .limit(4)
    .select("ts", "tenor_years", "yield_pct", "chg_bp")
    .to_pandas()
)

# %% [markdown]
# 上位2行は連続する日付の同じテナーで、符号が逆です。相場の動きではなく、悪いプリントの署名
# です。
#
# ## 5. プレビューとノート付きで、履歴を丸ごと残して訂正する
#
# `plan_replace_range` が悪い評価日に対して訂正をステージします。範囲の境界は `ts` 単位の生の
# マイクロ秒で、終端は排他的です。10テナーはすべて同じタイムスタンプを共有しているので、
# `[day_us, day_us + 1)` がその日の10行をちょうど捉えます。
#
# 何かが変わる*前に*サマリとサンプルを確認し、そのうえでノート付きで適用します。デスクは履歴を
# 編集しません。訂正版のバージョンをその上に積むだけです。

# %%
day_us = int(bad_date.value // 1000)
corrected = clean_df[clean_df["ts"] == bad_date]  # the true marks for that day

plan = db.plan_replace_range(
    "curves", day_us, day_us + 1,
    data=pa.Table.from_pandas(corrected, schema=schema, preserve_index=False),
    note=f"restate {bad_date.date()}: 10y stale contributor, -25bp",
)
print("summary:", plan.summary)
print("\nbefore (10y row):")
print(plan.before_sample.to_pandas().query("tenor_years == 10"))
print("\nafter (10y row):")
print(plan.after_sample.to_pandas().query("tenor_years == 10"))

# %%
commit = plan.apply()
print(f"applied as v{commit['sequence']} ({commit['op']})")
[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("curves")[-2:]
]

# %% [markdown]
# ## 6. ポイントインタイム: あの晩、リスクは何で走ったのか
#
# 訂正のあとに効いてくる問いがこれです。*下流の利用者が実際に見た数字はどれか。*
#
# どのバージョンもいまだに指せるので、SQL の `h5i('curves', v)` が評価時点の眺めと訂正後の先頭を
# 1つのクエリで結合します。問題の日のオーバーナイトのリスクは 4.05% で走り、今日の帳簿は同じ
# 評価日について 3.80% を示します。

# %%
v_pre_restate = db.versions("curves")[-2]["sequence"]  # head just before the restatement
was_y, now_y = col("yield_pct", relation="l"), col("yield_pct", relation="r")

(
    db.table("curves", version=v_pre_restate)
    .join(db.table("curves"), on=["ts", "tenor_years"])
    .filter(
        col("ts", relation="l") == pd.Timestamp(day_us, unit="us", tz="UTC").isoformat(),
        col("tenor_years", relation="l").is_in([2, 5, 10, 30]),
    )
    .select(
        tenor_years=col("tenor_years", relation="l"),
        as_marked_that_night=was_y.round(4),
        restated=now_y.round(4),
        diff_bp=(100 * (now_y - was_y)).round(1),
    )
    .sort("tenor_years")
    .to_pandas()
)

# %% [markdown]
# ## 7. 最新カーブからのキャリーとロールダウン
#
# 1年の horizon にわたる5年ポジションについて、カーブ不変を仮定した標準的な算術です。保存された
# テナーを内挿して計算します。
#
# **ロールダウン**は、今日のカーブの上で5年から4年へ歳を取ることによる利回りの上乗せです。
# **キャリー**は3か月の調達に対する利回りで、ここではマイナスになります。このシミュレーションの
# カーブは年末に手前が逆イールドになるからです。
#
# どちらも利回り空間の近似です。価格空間にするにはデュレーションを掛けてください。相対価値の
# スクリーニングには十分ですが、P&L の予測には向きません。

# %%
# No verb for a scalar subquery: resolve the latest mark date, then filter.
latest_ts = db.table("curves").select(m=col("ts").max()).to_pandas()["m"][0]

latest = (
    db.table("curves")
    .filter(col("ts") == latest_ts.isoformat())
    .select("tenor_years", "yield_pct")
    .sort("tenor_years")
    .to_pandas()
)

tenors, ylds = latest["tenor_years"].values, latest["yield_pct"].values
y = lambda t: float(np.interp(t, tenors, ylds))

rolldown_bp = 100 * (y(5) - y(4))
carry_bp = 100 * (y(5) - y(0.25))
dur_4y = 3.7  # rough modified duration of the aged position
print(f"5y yield {y(5):.3f}%, 4y yield {y(4):.3f}%, 3m funding {y(0.25):.3f}%")
print(f"rolldown (1y horizon): {rolldown_bp:+.1f} bp  (~{rolldown_bp * dur_4y / 100:+.2f}% price)")
print(f"carry over funding:    {carry_bp:+.1f} bp")
print(f"carry + roll:          {carry_bp + rolldown_bp:+.1f} bp per year, curve unchanged")

# %%
fig, ax = plt.subplots(figsize=(9, 4))
show_dates = mark_dates[:: len(mark_dates) // 5][:6]
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(show_dates)))
for d, c in zip(show_dates, colors):
    g = clean_df[clean_df["ts"] == d]
    ax.plot(g["tenor_years"], g["yield_pct"], marker="o", ms=3, lw=1.1,
            color=c, label=str(pd.Timestamp(d).date()))
ax.set_xscale("log")
ax.set_xticks([0.25, 1, 2, 5, 10, 30])
ax.set_xticklabels(["3m", "1y", "2y", "5y", "10y", "30y"])
ax.set_title("Curve snapshots across the year")
ax.set_xlabel("tenor")
ax.set_ylabel("yield (%)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - ロング形式の `(ts, tenor_years, yield_pct)` と、評価日ごとのコミット1件。この2つがカーブの
#   履歴を、バージョン管理され SQL で引ける評価日誌に変えます。
# - 条件付き集約のピボットと `lag()` のウィンドウが、定番の分析、つまり傾き、フライ、前日比の
#   外れ値検出を賄います。データベースの外で形を作り替える必要はありません。
# - 訂正は `plan_replace_range` です。終端排他のマイクロ秒の範囲が評価日を選び、プレビューが何が
#   変わるかを正確に見せ、ノートが監査記録として `versions()` に残ります。
# - ポイントインタイムはただで手に入ります。先頭に対して結合した `h5i('curves', v)` が、「あの晩
#   リスクは何で走ったのか」に、考古学ではなく来歴で答えます。
# - キャリーやロールダウンのようなカーブの計算も、保存されたテナーから素直に読めます。データ
#   ベースが評価を出し、pandas が内挿をやります。

# %%
db.close()
