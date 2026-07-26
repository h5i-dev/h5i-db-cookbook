# %% [markdown]
# # ポイントインタイムのファクターライブラリを作る
#
# 株式ファクターは先読みで死にます。市場がまだ見ていない簿価で計算した B/P は、バックテストでは
# 見事に、実売買では惨めに振る舞います。このレシピでは古典的なファクターを3つ――バリュー
# （B/P）、モメンタム（12-1）、クオリティの代理（売上成長の安定性）――作ります。ファンダメンタルズは
# h5i-db の `asof_join` で**報告日時点**として結合し、月次の IC と五分位スプレッドで評価し、
# 最後にバージョン管理してスナップショットを打ったファクターパネルとして保存します。作り直しの
# たびに差分を取って固定できるコミットが積み上がる、「ファクターライブラリ」のパターンです。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_factors"), create=True)

# %% [markdown]
# ## 1. 価格とファンダメンタルズを別々のフィードとして
#
# 合成50銘柄です。日次価格が約3年ぶん、ファンダメンタルズが12四半期ぶんで、後者の `ts` は
# **報告**（公知になった）タイムスタンプです。実際の開示と同じく `period_end` の25〜55日後に
# なっています。2つのフィードを、それぞれの時刻の意味を持つ別テーブルに保つこと。これがそもそも
# ポイントインタイムのジョインを可能にします。

# %%
prices = cu.make_daily_prices()   # 50 symbols, 750 sessions
funda = cu.make_fundamentals()    # 50 symbols, 12 quarters, report-lagged ts

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
# ## 2. 報告日ごとの、派生ファンダメンタル指標
#
# ここでのクオリティは四半期売上成長率の*安定性*で、直近4四半期の前四半期比成長率の標準偏差に
# マイナスを付けたものです。成長率も、その直近の標準偏差も、報告時刻の系列に対するウィンドウ
# 関数なので、全体が SQL 1文になります。その出力を `fund_signals` という独立したテーブルとして
# 保存します。派生データもデータであり、他と同じようにバージョン管理されるべきだからです。

# %%
fund_sig = db.sql(
    """
    WITH g AS (
        SELECT ts, symbol, book_value_m,
               revenue_m / lag(revenue_m) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS rev_g
        FROM fundamentals
    )
    SELECT ts, symbol, book_value_m, rev_g,
           stddev(rev_g) OVER (PARTITION BY symbol ORDER BY ts
                               ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS rev_g_std,
           count(rev_g)  OVER (PARTITION BY symbol ORDER BY ts
                               ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS n_g
    FROM g
    ORDER BY ts, symbol
    """
).to_arrow()

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
# モメンタムのレシピと同じ形です。銘柄ごとの `lag(21)/lag(252)` で12-1モメンタムを取り、
# `time_bucket('1mo', ...)` で各月の最終立会日に標本を取ります。このパネルも保存します。
# `asof_join` はテーブルに対して働きますし、月末の観測日はまさに、ファンダメンタルズを貼り
# 付けたい左側だからです。

# %%
panel_tbl = db.sql(
    """
    WITH sig AS (
        SELECT ts, symbol, close,
               lag(close, 21)  OVER w AS px_1m_ago,
               lag(close, 252) OVER w AS px_12m_ago
        FROM prices
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ),
    month_end AS (
        SELECT time_bucket('1mo', ts) AS month, symbol, max(ts) AS ts
        FROM prices GROUP BY month, symbol
    )
    SELECT s.ts, s.symbol, s.close,
           s.px_1m_ago / s.px_12m_ago - 1 AS momentum
    FROM sig s JOIN month_end USING (ts, symbol)
    ORDER BY s.ts, s.symbol
    """
).to_arrow()

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
db.sql("SELECT count(*) AS rows, count(DISTINCT ts) AS months FROM panel_me").to_pandas()

# %% [markdown]
# ## 4. ポイントインタイムのジョイン
#
# `asof_join('panel_me', 'fund_signals', 'ts', 'ts', 'symbol')` は、各月末の行に対して、
# その日**以前に報告された**最新のファンダメンタルズ行を銘柄ごとに返します。カレンダーの
# 計算も、「四半期を45日ずらす」といった経験則も要りません。ジョインのキーが報告タイム
# スタンプなので、訂正の遅れも作りからして扱えます。`ts_right` を見れば、その月がどの開示を
# 使ったかが正確に分かります。銘柄の初回報告より前の月は NULL になります。そうあるべきです。

# %%
pit = db.sql(
    """
    SELECT ts, symbol, close, momentum,
           book_value_m, rev_g_std, n_g,
           ts_right AS report_ts
    FROM asof_join('panel_me', 'fund_signals', 'ts', 'ts', 'symbol')
    ORDER BY ts, symbol
    """
).to_pandas()
pit["staleness_days"] = (pit["ts"] - pit["report_ts"]).dt.days
pit[["ts", "symbol", "close", "momentum", "book_value_m", "report_ts", "staleness_days"]].tail(4)

# %% [markdown]
# ## 5. 3つのファクターを組み立てる
#
# - **バリュー**: B/P ＝ 簿価 / 時価総額。発行株数は銘柄ごとに固定した合成値を使います
#   （生成器は発行株数を持たないので、シードを固定した静的な株数でクロスセクションを意味の
#   あるものにし、かつ決定的に保ちます）。
# - **モメンタム**: 12-1 を、パネルからそのまま。
# - **クオリティ**: 直近4四半期の売上成長率の標準偏差にマイナスを付けたもの。4四半期そろって
#   いること（`n_g = 4`）を要求します。
#
# 各ファクターは月ごとにクロスセクションで z スコア化し（±3 で切り詰め）、比較も合成もできる
# ようにします。

# %%
rng = np.random.default_rng(123)
symbols = sorted(pit["symbol"].unique())
shares_m = dict(zip(symbols, rng.uniform(100, 2000, len(symbols)).round(1)))

pit["value"] = pit["book_value_m"] / (pit["close"] * pit["symbol"].map(shares_m))
pit["quality"] = np.where(pit["n_g"] == 4, -pit["rev_g_std"], np.nan)

FACTORS = ["value", "momentum", "quality"]


def zscore_by_month(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("ts")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std")).clip(-3, 3)


for f in FACTORS:
    pit[f"z_{f}"] = zscore_by_month(pit, f)
pit["z_combo"] = pit[[f"z_{f}" for f in FACTORS]].mean(axis=1)

coverage = pit.groupby("ts")[[f"z_{f}" for f in FACTORS]].count()
coverage.tail(3)

# %% [markdown]
# ## 6. 評価: 情報係数と五分位スプレッド
#
# IC は、月末 *t* のファクター値と、月 *t+1* のリターンとの月次スピアマン順位相関です。
# シグナルはリターンが始まる前に完全に観測できています。平均 IC の |t 値| が約2を超えることが、
# 「このファクターは何かを予測している」と言うための通常の基準です。

# %%
close_piv = pit.pivot(index="ts", columns="symbol", values="close")
fwd_ret = close_piv.pct_change().shift(-1)
pit = pit.merge(
    fwd_ret.stack().rename("fwd_ret").reset_index().rename(columns={"level_1": "symbol"}),
    on=["ts", "symbol"],
    how="left",
)


def monthly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[col, "fwd_ret"]).groupby("ts"):
        if len(g) >= 20:
            out[ts] = stats.spearmanr(g[col], g["fwd_ret"])[0]
    return pd.Series(out, name=col)


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
# 五分位スプレッドは同じ情報をリターンの空間で見せます。毎月10銘柄ずつ5つのバケットに並べ、
# 等ウェイトで1か月保有し、Q5 − Q1 を見ます。

# %%
def quintile_spread(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for ts, g in df.dropna(subset=[col, "fwd_ret"]).groupby("ts"):
        if len(g) >= 25:
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
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
# 数字はそのまま読んでください。合成データがいちばん多くを教えてくれるのは、ここです。
#
# - **モメンタム**は本当にプラスです（IC の t 値が約2.7、年率換算した Q5−Q1 スプレッドは
#   二桁）。生成器が各銘柄に持続的なドリフトを与えているので、過去の勝者が勝ち続けます。
#   パイプラインは実在する効果を検出できています。
# - **バリュー**は正しく無です。クロスセクションの B/P 順位は各銘柄の静的な簿価と株数の
#   引きに支配され、そこに将来リターンの情報はありません。
# - **クオリティ**が罠です。平均 IC の |t| は約3。ところがこのファクターは*純粋なノイズ*から
#   作られています（構造上、全銘柄の売上成長ボラは同一です）。直近4四半期の窓を使う以上、
#   今月のファクターは先月とほとんど同じです。つまり約18個の月次 IC は、実効的には1つか2つの
#   観測が18の顔をかぶっているだけで、iid を前提とした t 値は絵空事です。ゆっくり動く
#   シグナルは、t 値を信じる前に重なりに頑健な推測（Newey-West、重ならない窓）が要ります。
#
# 合成ファクターは、平均に混ぜたノイズをそのまま受け継ぎます。20〜30か月では、実データで
# あればどれもリサーチの基準を通りません。成果物はアルファではなく、パイプラインのほうです。
#
# ## 7. パネルを保存する — ファクターライブラリのパターン
#
# 出来上がったパネル（生のファクターと z スコア）は `factor_panel` テーブルになり、スナップ
# ショットがそれを生んだ*すべての*テーブルの状態に名前を付けます。来月ライブラリを作り直せば
# 差分を取れる新しいバージョンが手に入りますし、論文や本番モデルは学習に使ったスナップショットに
# 固定できます。

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

db.sql(
    """
    SELECT ts, count(*) AS names, avg(z_combo) AS combo_mean, stddev(z_momentum) AS mom_z_std
    FROM h5i('factor_panel', 'factor-build-001')
    GROUP BY ts ORDER BY ts DESC LIMIT 3
    """
).to_pandas().set_index("ts").round(3)

# %% [markdown]
# ## まとめ
#
# - **報告タイムスタンプ**への `asof_join` が、ポイントインタイムの仕掛けのすべてです。月末の
#   各行は市場が実際に見ていた最新の開示を拾い、`ts_right` がどれを使ったかを記録します。
# - 派生データもデータです。成長率と安定性の指標、月末パネル、最終的なファクターライブラリは
#   どれもバージョン管理された h5i テーブルとして存在し、SQL のウィンドウ（`lag`、
#   `stddev ... OVER ROWS`、`time_bucket('1mo')`）で組み立てられます。
# - 合成データに対する評価は誠実です。モメンタムは効き（構造上ドリフトが持続します）、バリューは
#   無で、クオリティの「有意な」t 値は純粋なノイズから生まれた重なり窓の作り物でした。ファクター
#   リサーチの推測について、これほど安く学べる教訓はそうありません。
# - `db.snapshot(...)` 1回で、ファクター構築の入力系譜*全体*が固定されます。「私のモデルが
#   学習したパネルは、どのファンダメンタルズから生まれたのか」は調査ではなくクエリになります。

# %%
db.close()
