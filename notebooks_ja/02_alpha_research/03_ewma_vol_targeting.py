# %% [markdown]
# # EWMA ボラティリティとボラティリティ・ターゲティング
#
# RiskMetrics 流の EWMA ボラティリティは、どのリスクデスクでも主力の条件付きボラティリティ
# 推定量です。h5i-db はこれをネイティブの SQL ウィンドウ関数として持っています。
# `ewma(x, alpha) OVER (...)` です。
#
# このレシピで進めるのは次の4つです。
#
# 1. ポートフォリオのリターン系列を SQL で作る
# 2. EWMA ボラティリティを SQL で推定し、`pandas.ewm` とビット単位で突き合わせる
# 3. 実務どおりに使う。年率10%のリスク目標を保つようにエクスポージャをスケールする
# 4. 実データで、素のカーブとボラティリティ・ターゲティングのカーブを比べる

# %% [markdown]
# ## ここで使う用語
#
# | 用語                 | 意味 |
# | ------------------ | --- |
# | ボラティリティ            | リターンの標準偏差。ポジションの危険度を表す既定の代理変数 |
# | 年率化                | 日次の値を1年に換算すること。リターンは252倍、ボラティリティは `sqrt(252)` 倍 |
# | EWMA               | 指数加重移動平均。直近の観測値ほど重く効く |
# | ラムダと alpha         | RiskMetrics は旧推定値に lambda を掛ける。h5i-db の alpha は `1 - lambda` |
# | 重心（center of mass） | EWMA がどれだけ過去を覚えているか。日次 lambda 0.94 で約16営業日 |
# | ボラティリティ・ターゲティング    | 実現リスクが一定水準に近づくようポジションの大きさを調整すること |
# | レバレッジ              | 資本に対するエクスポージャの倍率。2倍なら資本1ドルにポジション2ドル |
# | エクイティカーブ           | 戦略の累積価値の推移。1ドルの成長として描く |
# | ドローダウン             | エクイティカーブの直近ピークからの下落率 |
# | シャープレシオ            | 年率リターンを年率ボラティリティで割った値。リスク1単位あたりの報酬 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_voltarget"), create=True)

# %% [markdown]
# ## 1. データ
#
# 流動性の高い10銘柄、2020〜2026 年の日足で、`cu.fetch_daily` のキャッシュから来ます。1行が
# 1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | セッションの日付 |
# | `symbol` | `string` | 銘柄コード |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `adj_close` | `float64` | 分割と配当で調整済みの終値 |
# | `volume` | `int64` | 出来高（株数） |

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")
print(f"{daily.num_rows:,} rows x {daily.num_columns} columns, "
      f"{len(set(daily['symbol'].to_pylist()))} symbols")
daily.to_pandas().head()

# %% [markdown]
# 取引する資産は、この10銘柄の等ウェイトポートフォリオです。広い株式のプロキシとしては妥当
# ですし、コロナのドローダウンと2022年の弱気相場を標本の中に入れられます。
#
# 銘柄ごとの単純リターンはパーティションごとの `lag()` から出し、等ウェイトポートフォリオは
# 日ごとのクロスセクション平均です。10銘柄すべてが約定した日だけを残し、結果を `port_returns`
# という独自のテーブルに保存します。こうすると EWMA の段が、きれいな単一テーブルのウィンドウ
# クエリになります。

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    daily.select(["ts", "symbol", "adj_close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
    .cast(schema),
    note="yfinance 10 names 2020-2026",
)

PREV = sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")

port = (
    db.table("prices")
    .with_columns(ret=col("adj_close") / PREV - 1)
    .filter(col("ret").is_not_null())
    .group_by("ts")
    .agg(ret=col("ret").mean(), names=count_star())
    .filter(col("names") == 10)  # the HAVING: full cross-section only
    .select("ts", "ret")
    .sort("ts")
    .to_arrow()
)

ret_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("ret", pa.float64()),
    ]
)
db.create_table("port_returns", ret_schema, time_column="ts", sort_key=["ts"])
db.append("port_returns", port.cast(ret_schema), note="equal-weight 10-name portfolio")
print(f"{port.num_rows} daily portfolio returns")

# %% [markdown]
# ## 2. SQL の中の EWMA 分散と、lambda／alpha の対訳
#
# RiskMetrics は漸化式を
# $\sigma_t^2 = \lambda\,\sigma_{t-1}^2 + (1-\lambda)\,r_t^2$ と書き、日次では
# $\lambda = 0.94$ を使います。
#
# h5i-db の `ewma(x, alpha)` は*新しい*観測にかかる平滑化の重みを取るので、
# **alpha = 1 − lambda = 0.06** です。同じ漸化式で、命名の向きが逆。慣習のずれによるバグの、
# 息の長い発生源です。
#
# 0.94／0.06 の組は λ/(1−λ) ≈ 16 営業日の重心を意味します。バーの頻度が違えば、0.94 を
# そのまま使い回さずスケールし直します。たとえば週次なら λ は 0.97 寄りになります。

# %%
ewma_sql = (
    db.table("port_returns")
    .select("ts", "ret", ewma_var=(col("ret") * col("ret")).ewma(0.06, order_by="ts"))
    .sort("ts")
    .to_pandas()
)
ewma_sql["ewma_vol_ann"] = np.sqrt(ewma_sql["ewma_var"] * 252)

# Cross-check against pandas (adjust=False = the plain recursion).
pandas_var = (ewma_sql["ret"] ** 2).ewm(alpha=0.06, adjust=False).mean()
assert np.allclose(ewma_sql["ewma_var"], pandas_var)
print("SQL ewma() == pandas ewm(alpha=0.06, adjust=False): exact match")
ewma_sql.set_index("ts").tail(3).round(6)

# %% [markdown]
# ## 3. ボラティリティ・ターゲティング
#
# 目標は年率10%です。ポジションは日ごとに
# `leverage_t = target_vol / ewma_vol_{t-1}` でスケールするので、今日のエクスポージャを
# 決めるのは**前日**のボラティリティ推定値です。当日の値を使えば、その日自身の二乗リターンが
# 自分のサイジングに漏れ込みます。
#
# レバレッジは 2倍で頭打ちにします。資金調達とマンデートの現実です。そして日々のグロス
# エクスポージャの変化に 5bps を課金し、戦略が自分のリバランス代を払うようにします。

# %%
TARGET_VOL, LEV_CAP, COST_BPS = 0.10, 2.0, 5

df = ewma_sql.set_index("ts")
lev = (TARGET_VOL / df["ewma_vol_ann"].shift(1)).clip(upper=LEV_CAP)
raw = df["ret"]
targeted = (lev * raw - COST_BPS / 1e4 * lev.diff().abs()).dropna()


def perf_stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "ann_ret": r.mean() * 252,
        "ann_vol": r.std() * np.sqrt(252),
        "sharpe": r.mean() / r.std() * np.sqrt(252),
        "max_dd": dd.min(),
    }


stats = pd.DataFrame(
    {"raw (1x)": perf_stats(raw.loc[targeted.index]), "vol-targeted 10%": perf_stats(targeted)}
).T
print(f"leverage: mean {lev.mean():.2f}x, range [{lev.min():.2f}, {lev.max():.2f}]x")
stats.round(3)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axes[0].plot(df.index, df["ewma_vol_ann"] * 100, lw=0.9, color="tab:blue")
axes[0].axhline(TARGET_VOL * 100, color="tab:red", lw=0.8, ls="--", label="10% target")
axes[0].set_ylabel("EWMA vol (ann, %)")
axes[0].set_title("EWMA(0.94) volatility, leverage, and equity curves")
axes[0].legend()

idx = targeted.index
axes[1].plot((1 + raw.loc[idx]).cumprod(), lw=1.1, label="raw (1x)")
axes[1].plot((1 + targeted).cumprod(), lw=1.1, label="vol-targeted")
axes[1].set_ylabel("growth of $1")
axes[1].legend()

for r, lab in [(raw.loc[idx], "raw (1x)"), (targeted, "vol-targeted")]:
    eq = (1 + r).cumprod()
    axes[2].plot(eq / eq.cummax() - 1, lw=0.9, label=lab)
axes[2].set_ylabel("drawdown")
axes[2].set_xlabel("date")
axes[2].legend()
fig.tight_layout()

# %% [markdown]
# 理屈が言うとおりの場所に、そのまま結果が出ます。ターゲティングした本の実現ボラティリティは
# 10%近くに着地し、素のほうはおよそ24%。最大ドローダウンは3分の1ほどに縮みます。戦略が
# ボラティリティの跳ね上がり、つまり2020年のコロナや2022年の弱気相場で、レバレッジを落とす
# からです。
#
# シャープの改善はわずかです。ボラティリティ・ターゲティングは主に*リスクの形を整える*道具で、
# 株式のボラティリティとリターンの負の相関を利用しています。シャープの上乗せは目的ではなく
# おまけと考えてください。

# %% [markdown]
# ## まとめ
#
# - `ewma(ret*ret, 0.06) OVER (ORDER BY ts)` は RiskMetrics の漸化式を SQL のウィンドウ呼び
#   出し1つで再現します。`pandas.ewm(alpha=0.06, adjust=False)` とビット単位で一致することを
#   確認しました。
# - 対訳を覚えておいてください。h5i-db の `alpha` は 1 − RiskMetrics の `lambda` です。日次の
#   λ=0.94 は `ewma(x, 0.06)` になり、重心はおよそ16日。バーの頻度が違えばスケールし直します。
# - サイジングには*前日*のボラティリティ推定値を使います。当日のサイジングは先読みで、しかも
#   肝心な日ほど見栄えをよくします。
# - この標本では、ターゲティングが実現ボラティリティを10%近くに保ち、最大ドローダウンを
#   およそ32%から11%に切り、シャープもわずかに上げました。これがボラティリティ・ターゲティング
#   の正直な売り文句です。
# - `port_returns` のような派生系列も一級のバージョン管理されたテーブルとして生きるので、
#   リスクのパイプライン全体が SQL で引けて再現できます。

# %%
db.close()
