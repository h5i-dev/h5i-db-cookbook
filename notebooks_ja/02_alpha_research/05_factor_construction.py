# %% [markdown]
# # ポイントインタイムのファクターライブラリを作る
#
# 株式ファクターは先読みで死にます。市場がまだ見ていない純資産で計算した B/P レシオは、
# バックテストでは美しく、実運用ではひどい成績を出します。
#
# このレシピは定番のファクターを3つ作り、正直に評価します。
#
# 1. `asof_join` でファンダメンタルズを**報告日時点で**結合する
# 2. バリュー（B/P）、モメンタム（12-1）、そして売上成長の安定性から作る品質のプロキシを
#    組み立てる
# 3. 月次の情報係数と五分位スプレッドで評価する
# 4. 結果をバージョン管理してスナップショットしたファクターパネルとして保存する
#
# 最後の1つが「ファクターライブラリ」のパターンです。作り直すたびに、差分を取れてピン留め
# できるコミットが1つ増えます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語          | 意味 |
# | ----------- | --- |
# | ファクター       | 多数の資産に共通してリターンを説明する特性。バリューやモメンタムなど |
# | バリュー（B/P）   | 純資産を株価で割った値。高いほど会計上の価値に比べて割安 |
# | クオリティ       | 収益性や安定性の指標。ここでは売上成長の安定度を代理に使う |
# | モメンタム（12-1） | 直近12か月から直近1か月を除いたリターン |
# | パネル         | 1行が1資産1日付になるデータセット |
# | ポイントインタイム   | 各時点で知られていたとおりにデータを保存し、参照すること |
# | 先読みバイアス     | 当時は手に入らなかった情報を使うこと。ファクターを殺すバグ |
# | IC（情報係数）    | シグナルと、その後に実現したリターンとの相関 |
# | 五分位スプレッド    | シグナル上位1/5から下位1/5を引いたリターン。効くかをモデルなしで読む |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

import h5i_db
from h5i_db import col, count_star, sql_expr, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_factors"), create=True)

# %% [markdown]
# ## 1. データ
#
# 合成50銘柄ぶんのフィードが2つ。価格側は `cu.make_daily_prices` の日足 OHLCV パネルで、
# おおよそ3年ぶん、1行が1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、20:00 UTC |
# | `symbol` | `string` | 銘柄コード、`STK000` 〜 `STK049` |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `volume` | `int64` | 出来高（株数） |

# %%
prices = cu.make_daily_prices()   # 50 symbols, 750 sessions
print(f"prices: {prices.num_rows:,} rows x {prices.num_columns} columns")
prices.to_pandas().head()

# %% [markdown]
# ファンダメンタルズ側は `cu.make_fundamentals` の12四半期です。`ts` は**報告**のタイムスタンプ
# で、実際の開示と同じく `period_end` の25〜55日あとに来ます。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 報告時刻。数字が公になった瞬間 |
# | `period_end` | `timestamp[us, tz=UTC]` | 会計四半期の期末 |
# | `symbol` | `string` | 銘柄コード |
# | `eps` | `float64` | 報告された1株当たり利益 |
# | `revenue_m`、`book_value_m` | `float64` | 売上高と純資産、百万ドル |
#
# 2つのフィードを、それぞれの時間の意味を持たせたまま別テーブルに置くこと。それがそもそも
# ポイントインタイムの結合を可能にします。

# %%
funda = cu.make_fundamentals()    # 50 symbols, 12 quarters, report-lagged ts
print(f"fundamentals: {funda.num_rows:,} rows x {funda.num_columns} columns")
funda.to_pandas().head()

# %%
px_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)]
    + [prices.schema.field(i) for i in range(1, len(prices.schema))]
)
db.create_table("prices", px_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices.sort_by([("ts", "ascending"), ("symbol", "ascending")]))

f_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)]
    + [funda.schema.field(i) for i in range(1, len(funda.schema))]
)
db.create_table("fundamentals", f_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fundamentals", funda.sort_by([("ts", "ascending"), ("symbol", "ascending")]))
print(f"prices: {prices.num_rows:,} rows   fundamentals: {funda.num_rows:,} rows")

# %% [markdown]
# ## 2. 報告日ごとの派生ファンダメンタル指標
#
# ここでの品質とは、四半期売上成長の*安定性*です。直近4四半期の前四半期比成長率の標準偏差に
# マイナスを付けたものを使います。
#
# 成長率もその移動標準偏差も、報告時刻の系列に対するウィンドウ関数なので、これは SQL 1文です。
# 出力は `fund_signals` という独自のテーブルに保存します。派生データもデータであり、ほかと
# 同じようにバージョン管理されるべきだからです。

# %%
PREV_REV = sql_expr("lag(revenue_m)").over(partition_by="symbol", order_by="ts")

fund_sig = (
    db.table("fundamentals")
    .with_columns(rev_g=col("revenue_m") / PREV_REV - 1)
    .with_columns(
        rev_g_std=col("rev_g").rolling_std(4, order_by="ts", partition_by="symbol"),
        n_g=col("rev_g").rolling_count(4, order_by="ts", partition_by="symbol"),
    )
    .select("ts", "symbol", "book_value_m", "rev_g", "rev_g_std", "n_g")
    .sort(["ts", "symbol"])
    .to_arrow()
)

fs_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("book_value_m", pa.float64()),
        pa.field("rev_g", pa.float64()),
        pa.field("rev_g_std", pa.float64()),
        pa.field("n_g", pa.int64()),
    ]
)
db.create_table("fund_signals", fs_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fund_signals", fund_sig.cast(fs_schema), note="rev growth + 4q stability")

# %% [markdown]
# ## 3. モメンタム付きの月末価格パネル
#
# モメンタムのレシピと同じ形です。12-1 モメンタムには銘柄ごとの `lag(21)/lag(252)`、サンプリ
# ングには `time_bucket('1mo', ...)` で各月の最終営業日を使います。
#
# このパネルも保存します。`asof_join` はテーブルに対して働きますし、月末の観測日はまさに
# ファンダメンタルズを貼り付けたい左側だからです。

# %%
def lag_close(n: int):
    return sql_expr(f"lag(close, {n})").over(partition_by="symbol", order_by="ts")


sig = db.table("prices").with_columns(px_1m_ago=lag_close(21), px_12m_ago=lag_close(252))

month_end = (
    db.table("prices")
    .group_by(time_bucket("1mo", col("ts")).alias("month"), "symbol")
    .agg(month_end=col("ts").max())
    .select(col("month_end").alias("ts"), "symbol")
)

panel_tbl = (
    sig.join(month_end, on=["ts", "symbol"])
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        close=col("close", relation="l"),
        momentum=col("px_1m_ago", relation="l") / col("px_12m_ago", relation="l") - 1,
    )
    .sort(["ts", "symbol"])
    .to_arrow()
)

panel_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("momentum", pa.float64()),
    ]
)
db.create_table("panel_me", panel_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("panel_me", panel_tbl.cast(panel_schema), note="month-end closes + 12-1 momentum")
db.table("panel_me").select(rows=count_star(), months=col("ts").n_unique()).to_pandas()

# %% [markdown]
# ## 4. ポイントインタイムの結合
#
# `asof_join('panel_me', 'fund_signals', 'ts', 'ts', 'symbol')` は、月末の各行に対して、その
# 日付**以前に報告された**最新のファンダメンタルズ行を銘柄ごとに返します。
#
# カレンダーの計算も、「四半期を45日ずらす」といった経験則もありません。結合キーが報告の
# タイムスタンプなので、訂正の遅れは構造的に処理されます。
#
# `ts_right` は各月がどの開示を使ったかを正確に示しますし、その銘柄の最初の報告より前の月は
# NULL になります。そうあるべきです。

# %%
pit = (
    db.table("panel_me")
    .join_asof(db.table("fund_signals"), on="ts", by="symbol")
    .select("ts", "symbol", "close", "momentum", "book_value_m", "rev_g_std", "n_g",
            report_ts=col("ts_right"))
    .sort(["ts", "symbol"])
    .to_pandas()
)
pit["staleness_days"] = (pit["ts"] - pit["report_ts"]).dt.days
pit[["ts", "symbol", "close", "momentum", "book_value_m", "report_ts", "staleness_days"]].tail(4)

# %% [markdown]
# ## 5. 3つのファクターを組み立てる
#
# - **バリュー**: B/P、つまり純資産を時価総額で割ったもの。銘柄ごとに固定の合成株式数を使い
#   ます。生成器に発行済株式数がないので、シードを固定した静的な株式数でクロスセクションを
#   意味のある、決定的なものに保ちます。
# - **モメンタム**: 12-1 を、パネルからそのまま。
# - **品質**: 直近4四半期の売上成長率の標準偏差にマイナスを付けたもの。4四半期そろっている
#   こと（`n_g = 4`）を条件にします。
#
# 各ファクターは月ごとにクロスセクションで z 化し、±3 でクリップします。こうすると比較も合成
# もできるようになります。

# %%
rng = np.random.default_rng(123)
symbols = sorted(pit["symbol"].unique())
shares_m = dict(zip(symbols, rng.uniform(100, 2000, len(symbols)).round(1)))

pit["value"] = pit["book_value_m"] / (pit["close"] * pit["symbol"].map(shares_m))
pit["quality"] = np.where(pit["n_g"] == 4, -pit["rev_g_std"], np.nan)

FACTORS = ["value", "momentum", "quality"]


def zscore_by_month(df: pd.DataFrame, field: str) -> pd.Series:
    g = df.groupby("ts")[field]
    return ((df[field] - g.transform("mean")) / g.transform("std")).clip(-3, 3)


for f in FACTORS:
    pit[f"z_{f}"] = zscore_by_month(pit, f)
pit["z_combo"] = pit[[f"z_{f}" for f in FACTORS]].mean(axis=1)

coverage = pit.groupby("ts")[[f"z_{f}" for f in FACTORS]].count()
coverage.tail(3)

# %% [markdown]
# ## 6. 評価: 情報係数と五分位スプレッド
#
# 情報係数は、月末 *t* のファクター値と *t+1* 月のリターンとの、月次のスピアマン順位相関です。
# シグナルはリターンが始まる前に完全に観測できています。
#
# 平均情報係数の |t 値| が概ね2を超えるのが、「このファクターは何かを予測している」の通例の
# 基準です。

# %%
close_piv = pit.pivot(index="ts", columns="symbol", values="close")
fwd_ret = close_piv.pct_change().shift(-1)
pit = pit.merge(
    fwd_ret.stack().rename("fwd_ret").reset_index().rename(columns={"level_1": "symbol"}),
    on=["ts", "symbol"],
    how="left",
)


def monthly_ic(df: pd.DataFrame, field: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[field, "fwd_ret"]).groupby("ts"):
        if len(g) >= 20:
            out[ts] = stats.spearmanr(g[field], g["fwd_ret"])[0]
    return pd.Series(out, name=field)


ics = pd.concat([monthly_ic(pit, f"z_{f}") for f in FACTORS + ["combo"]], axis=1)
ic_summary = pd.DataFrame(
    {
        "mean_IC": ics.mean(),
        "IC_vol": ics.std(),
        "t_stat": ics.mean() / ics.std() * np.sqrt(ics.notna().sum()),
        "months": ics.notna().sum(),
    }
)
ic_summary.round(3)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
for ax, f in zip(axes, FACTORS):
    s = ics[f"z_{f}"].dropna()
    ax.bar(s.index, s.values, width=18, color=np.where(s.values >= 0, "tab:blue", "tab:red"))
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(s.mean(), color="tab:green", lw=1.0, ls="--", label=f"mean {s.mean():+.3f}")
    ax.set_ylabel(f"{f} IC")
    ax.legend(loc="upper left")
axes[0].set_title("Monthly rank IC by factor (Spearman, forward 1-month return)")
axes[-1].set_xlabel("month end")
fig.tight_layout()

# %% [markdown]
# 五分位スプレッドは同じ情報をリターンの空間で見せます。毎月10銘柄ずつ5つのバケットに分け、
# 等ウェイトで1か月持ち、Q5 から Q1 を引きます。

# %%
def quintile_spread(df: pd.DataFrame, field: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[field, "fwd_ret"]).groupby("ts"):
        if len(g) >= 25:
            q = pd.qcut(g[field].rank(method="first"), 5, labels=False)
            out[ts] = g["fwd_ret"][q == 4].mean() - g["fwd_ret"][q == 0].mean()
    return pd.Series(out)


spreads = pd.DataFrame({f: quintile_spread(pit, f"z_{f}") for f in FACTORS + ["combo"]})
pd.DataFrame(
    {
        "ann_spread": spreads.mean() * 12,
        "t_stat": spreads.mean() / spreads.std() * np.sqrt(spreads.notna().sum()),
    }
).round(3)

# %% [markdown]
# 数字を出たとおりに読んでください。合成データがいちばん教えてくれるのはここです。
#
# - **モメンタム**は本当にプラスです。情報係数の t 値がおよそ 2.7、年率の Q5−Q1 スプレッドは
#   二桁。生成器が各銘柄に持続的なドリフトを与えているので、過去の勝者は勝ち続けます。
#   パイプラインは本物の効果を検出しました。
# - **バリュー**は正しくヌルです。クロスセクションの B/P の順位は各銘柄の静的な純資産と株式数
#   の引きで決まっていて、将来のリターンについて何の情報も持ちません。
# - **品質**が罠です。平均情報係数の |t| はおよそ3。それでいてこのファクターは*純粋なノイズ*
#   から作られています。構造上、どの銘柄も売上成長のボラティリティは同じだからです。4四半期の
#   移動窓のせいで今月のファクターは先月とほぼ同じになり、18ほどの月次情報係数は、実効的には
#   1つか2つの観測が18の帽子をかぶっているだけです。i.i.d. を仮定した t 値は作り話です。
#   ゆっくり動くシグナルは、Newey-West にせよ重ならない窓にせよ、重複に頑健な推論を通してから
#   でないと t 値を信じてはいけません。
#
# 合成値は、平均するノイズをそのまま引き継ぎます。20〜30か月では、どれも実データなら研究の
# 基準を通りません。成果物はアルファではなくパイプラインです。
#
# ## 7. パネルを保存する: ファクターライブラリのパターン
#
# 出来上がったパネル、つまり生のファクターと z スコアが `factor_panel` テーブルになります。
# スナップショットは、それを生んだ*すべての*テーブルの状態に名前を付けます。
#
# 来月ライブラリを作り直せば、差分を取れる新しいバージョンが1つ増えます。論文や本番のモデルは、
# 学習に使ったスナップショットに固定できます。

# %%
out_cols = ["ts", "symbol", "value", "momentum", "quality", "z_value", "z_momentum", "z_quality", "z_combo"]
panel_out = pit[out_cols].sort_values(["ts", "symbol"])

fp_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("symbol", pa.string())]
    + [pa.field(c, pa.float64()) for c in out_cols[2:]]
)
db.create_table("factor_panel", fp_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "factor_panel",
    pa.Table.from_pandas(panel_out, preserve_index=False).cast(fp_schema),
    note="build 001: value/momentum/quality + combo",
)
db.snapshot(
    "factor-build-001",
    tables=["prices", "fundamentals", "fund_signals", "panel_me", "factor_panel"],
    note="monthly factor library build",
)

(
    db.table("factor_panel", snapshot="factor-build-001")
    .group_by("ts")
    .agg(names=count_star(), combo_mean=col("z_combo").mean(), mom_z_std=col("z_momentum").std())
    .sort("ts", descending=True)
    .limit(3)
    .to_pandas()
).set_index("ts").round(3)

# %% [markdown]
# ## まとめ
#
# - **報告のタイムスタンプ**に対する `asof_join` が、ポイントインタイムの仕掛けのすべてです。
#   月末の各行は市場が実際に見ていた最新の開示を拾い、`ts_right` がそれがどれだったかを
#   記録します。
# - 派生データもデータです。成長と安定性の指標も、月末のパネルも、最終的なファクターライブラリ
#   も、すべてバージョン管理された h5i のテーブルとして生き、`lag`、`stddev ... OVER ROWS`、
#   `time_bucket('1mo')` という SQL のウィンドウで作られます。
# - 合成データに対する評価は正直です。モメンタムは構造上ドリフトが持続するので効き、バリュー
#   はヌル、そして品質の「有意な」t 値は純粋なノイズから生まれた重複窓の副作用です。ファクター
#   研究の推論について、これ以上安く学べる教材はありません。
# - `db.snapshot(...)` 1つが、ファクター構築の入力の*系譜まるごと*を固定します。「私のモデルが
#   学習したパネルは、どのファンダメンタルズから出たのか」が、調査ではなくクエリになります。

# %%
db.close()
