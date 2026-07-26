# %% [markdown]
# # 債券: バージョン付きのカーブ評価、訂正、キャリーとロールダウン
#
# 金利デスクの中核データセットは小さいのに、容赦がありません。マーク日ごとにパーカーブが1本、
# その上のどの数字もリスクと損益と顧客への評価に流れ込みます。h5i-db で自然な構成は*縦持ち*
# ――`(ts, tenor_years, yield_pct)`――で、マーク日ごとに1コミットです。これで3つが同時に手に
# 入ります。あらゆるカーブ分析のための SQL ピボット、EOD の評価セットごとのバージョン（規制
# 当局が実際に尋ねてくる監査証跡）、そして悪い提示値がすり抜けたときのプレビュー可能な訂正です。
#
# ここでは250営業日ぶんのカーブを保存し（意図的に仕込んだ10年の入力ミスも1つ含みます）、
# スロープと曲率の分析を走らせ、悪い評価を SQL で*見つけ*、`plan_replace_range` で訂正し、
# 訂正前の見え方が永久にクエリできることを示します。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_curves"), create=True)

curves = cu.make_yield_curves(days=250)  # ts, tenor_years, yield_pct - Nelson-Siegel dynamics
clean_df = curves.to_pandas()
mark_dates = clean_df["ts"].unique()

# Plant the incident: on mark date #120 the 10y prints 25bp too high
# (a stale contributor). This is what actually lands in the database.
feed_df = clean_df.copy()
bad_date = pd.Timestamp(mark_dates[120])
bad_mask = (feed_df["ts"] == bad_date) & (feed_df["tenor_years"] == 10.0)
feed_df.loc[bad_mask, "yield_pct"] += 0.25
print(f"vendor feed: {len(feed_df):,} rows, bad 10y mark on {bad_date.date()}")

# %% [markdown]
# ## 1. マーク日1つ＝コミット1回
#
# その日の10テナーをまとめて append します。コミット*こそが*評価セットです。すると
# `versions()` がデスクの評価日誌のように読めます。シーケンス、行数、そしてマーク日を書いた
# 注記です。（ループで250コミットは問題ありません。公表イベント1件につき1コミットという、
# h5i-db が期待するバッチ化のままです。）

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
# ## 2. カーブ分析を SQL のピボットで
#
# 縦持ちは条件付き集約できれいにピボットできます。マーク日ごとに1行、ベンチマークのテナー
# ごとに1列です。そこから先、デスクが日常的に見る時系列はただの算術になります。2s10s の
# スロープ、2s5s10s のバタフライ（曲率）、そして素朴なタームプレミアムの代理（10年 − 3か月。
# 本物のタームプレミアムには期待のモデルが要るので、これは長期側のスロープという観測量に
# すぎません）。

# %%
key_rates = db.sql(
    """
    SELECT ts,
           max(CASE WHEN tenor_years = 0.25 THEN yield_pct END) AS y3m,
           max(CASE WHEN tenor_years = 2    THEN yield_pct END) AS y2,
           max(CASE WHEN tenor_years = 5    THEN yield_pct END) AS y5,
           max(CASE WHEN tenor_years = 10   THEN yield_pct END) AS y10,
           max(CASE WHEN tenor_years = 30   THEN yield_pct END) AS y30
    FROM curves
    GROUP BY ts ORDER BY ts
    """
).to_pandas()
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
# ## 3. 悪い評価をウィンドウ関数で見つける
#
# 古い提示値は、1日の往復として現れます。前日比の大きな変化があり、同じテナーで翌日それが
# 巻き戻る形です。(テナー, 日付) の全組について |Δ利回り| を順位付ければ、すぐ浮かんできます。

# %%
db.sql(
    """
    WITH chg AS (
        SELECT ts, tenor_years, yield_pct,
               yield_pct - lag(yield_pct) OVER (PARTITION BY tenor_years ORDER BY ts) AS d1
        FROM curves
    )
    SELECT ts, tenor_years, round(yield_pct, 4) AS yield_pct, round(100 * d1, 1) AS chg_bp
    FROM chg
    WHERE d1 IS NOT NULL
    ORDER BY abs(d1) DESC
    LIMIT 4
    """
).to_pandas()

# %% [markdown]
# 上位2行は、同じテナーの連続する日付で符号が逆です。相場の変動ではなく、悪い提示値の署名です。
#
# ## 4. 訂正する — プレビューと注記と、完全な履歴とともに
#
# `plan_replace_range` が、問題のマーク日に対して訂正をステージングします（範囲の境界は `ts` の
# 単位で生のマイクロ秒、終端は含みません。10行すべてが同じタイムスタンプを共有するので、
# `[day_us, day_us + 1)` でちょうどその日の10行が入ります）。何かが変わる*前に*要約とサンプルを
# 検分し、それから注記付きで適用します。デスクは歴史を編集しません。訂正済みのバージョンを
# 上に積むだけです。

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
# ## 5. ポイントインタイム: あの晩、リスクは何で走ったのか
#
# 訂正のあとに効いてくる問いはこれです。*下流の利用者が実際に見た数字はどれか*。どのバージョンも
# 参照できるので、SQL の `h5i('curves', v)` が、評価当時の見え方と訂正後の先頭を1つのクエリで
# 突き合わせます。問題の日のオーバーナイトのリスクは4.05%で走っていて、今日の帳簿では同じマーク日が
# 3.80%になっています。

# %%
v_pre_restate = db.versions("curves")[-2]["sequence"]  # head just before the restatement
db.sql(
    f"""
    SELECT was.tenor_years,
           round(was.yield_pct, 4) AS as_marked_that_night,
           round(now.yield_pct, 4) AS restated,
           round(100 * (now.yield_pct - was.yield_pct), 1) AS diff_bp
    FROM h5i('curves', {v_pre_restate}) was
    JOIN curves now ON was.ts = now.ts AND was.tenor_years = now.tenor_years
    WHERE was.ts = to_timestamp_micros({day_us}) AND was.tenor_years IN (2, 5, 10, 30)
    ORDER BY was.tenor_years
    """
).to_pandas()

# %% [markdown]
# ## 6. 最新カーブからのキャリーとロールダウン
#
# 5年ポジションを1年の期間で見る、標準的な「カーブ不変」の算術です。保存済みのテナーを補間して
# 使います。**ロールダウン**は今日のカーブ上で5年から4年へ齢を重ねることによる利回りの上乗せ、
# **キャリー**は3か月調達に対する利回りです。ここでは負になります。このシミュレーションの
# カーブは、年末に手前が逆イールドになっているからです。どちらも利回り空間の近似なので（価格
# 空間ならデュレーションを掛けます）、相対価値のスクリーニングには十分でも、損益の予測には
# なりません。

# %%
latest = db.sql(
    "SELECT tenor_years, yield_pct FROM curves WHERE ts = (SELECT max(ts) FROM curves) ORDER BY tenor_years"
).to_pandas()

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
# - 縦持ちの `(ts, tenor_years, yield_pct)` とマーク日ごとの1コミットで、カーブの履歴が
#   バージョン管理された SQL でクエリできる評価日誌になります。
# - 条件付き集約のピボットと `lag()` のウィンドウで、日常的な分析――スロープ、フライ、前日比の
#   外れ値検出――はまかなえます。データベースの外でデータを組み替える必要はありません。
# - 訂正は `plan_replace_range` です。マイクロ秒の範囲境界（終端を含まない）がマーク日を選び、
#   プレビューが何が変わるかを正確に見せ、注記が `versions()` に監査記録として残ります。
# - ポイントインタイムはただで付いてきます。`h5i('curves', v)` を先頭と突き合わせれば、「あの晩、
#   リスクは何で走ったのか」に、発掘作業ではなく出所つきで答えられます。
# - キャリーとロールダウンのようなカーブの計算は、保存済みのテナーから素直に読めます。データ
#   ベースが評価を配り、補間は pandas がやります。

# %%
db.close()
