# %% [markdown]
# # VaR と期待ショートフォールを、監査できるリスクテーブルとともに
#
# 誰にも再現できないリスクの数字は負債です。
#
# このレシピで進めるのは次の3つです。
#
# 1. 固定の株式ポートフォリオについて、実際の日次データでヒストリカルとパラメトリックの
#    VaR・期待ショートフォールを計算する
# 2. Kupiec の POF 検定で VaR をバックテストする
# 3. 日次のリスク指標を、本番日1日につきコミット1件でバージョン管理されたテーブルに書き出す
#
# 3つ目こそ、たいていのノートブックが飛ばす部分です。「X日に報告した VaR は何で、どの入力
# から出たのか」を、メールのやり取りではなくクエリに変えます。
#
# 働くのは h5i-db の3つの機能です。リターンのパイプラインを担う SQL のウィンドウ関数、監査
# 証跡になる追記専用のコミット（`versions()` が EOD の実行をノートと実時刻つきで見せます）、
# そして小さな日次追記が積み重なったテーブルを健全に保つ `compact()` です。

# %% [markdown]
# ## ここで使う用語
#
# | 用語              | 意味 |
# | --------------- | --- |
# | VaR             | ある信頼水準のもとで、1日の損失がこれを超えないという水準 |
# | 期待ショートフォール      | VaR を超えた日の損失の平均。VaR がかわした問いに答える |
# | ヒストリカル VaR      | 過去リターンの経験分布からそのまま読む方法 |
# | パラメトリック VaR     | 分布（ふつう正規分布）を仮定して分位点を読む方法 |
# | 信頼水準            | VaR の記述に出てくる 95% や 99% のこと |
# | 超過（exception）   | 損失が予測を超えた日。99% VaR なら全体の約1%で起きるはず |
# | Kupiec の POF 検定 | 観測された超過回数が、掲げた信頼水準と整合するかの検定 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_var"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.fetch_daily` が返すのは、流動性の高い米国株30銘柄の8年超の日次データです。Yahoo Finance
# からキャッシュしてあります。1行が1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | セッションの日付 |
# | `symbol` | `string` | 銘柄コード |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `adj_close` | `float64` | 分割と配当で調整済みの終値 |
# | `volume` | `int64` | 出来高（株数） |

# %%
real = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
print(f"{real.num_rows:,} rows x {real.num_columns} columns, "
      f"{len(set(real['symbol'].to_pylist()))} symbols")
real.to_pandas().head()

# %% [markdown]
# 保存するのは `adj_close` です。おおむねトータルリターンの系列で、P&L のリスクはこれで計算
# すべきものです。1回の一括 `append` がアトミックなコミット1件になります。
#
# 厳格さを1つ知っておいてください。`sort_key=["ts", "symbol"]` の場合、`append` は入力が
# *キー全体*で整列していることを要求します。日次パネルは同じタイムスタンプを30銘柄で共有して
# いるので、`ts` だけの整列では足りません。append の前に (ts, symbol) で並べてください。

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
    real.select(["ts", "symbol", "adj_close"]).sort_by([("ts", "ascending"), ("symbol", "ascending")]),
    note="vendor backfill 2018-2026",
)

db.table("prices").select(rows=count_star(), names=col("symbol").n_unique()).to_pandas()

# %% [markdown]
# ## 2. ポートフォリオの P&L 系列を SQL 1文で
#
# ポートフォリオは12銘柄の固定構成です。ウェイトの合計は1、想定元本は \\$10M。
#
# 銘柄ごとの日次単純リターンは `lag()` のウィンドウから出します。ポートフォリオのリターンは
# 加重和で、インラインの `VALUES` によるウェイトの関係との結合で作ります。
#
# すべてが時刻順ストレージの上を流れます。ローリング分位点が実際に要るまで、pandas での
# 形の作り替えはありません。

# %%
NOTIONAL = 10_000_000.0

WEIGHTS_SQL = """
    (VALUES ('AAPL', 0.10), ('MSFT', 0.10), ('NVDA', 0.08), ('AMZN', 0.08),
            ('GOOGL', 0.08), ('JPM', 0.10), ('XOM', 0.08), ('UNH', 0.08),
            ('PG', 0.08), ('KO', 0.06), ('CAT', 0.08), ('GS', 0.08)
    ) AS w(symbol, weight)
"""

# An inline VALUES relation has no builder verb, so this one stays SQL.
port = db.sql(
    f"""
    WITH rets AS (
        SELECT ts, symbol,
               adj_close / lag(adj_close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
        FROM prices
    )
    SELECT r.ts, sum(r.ret * w.weight) AS port_ret, count(*) AS names
    FROM rets r
    JOIN {WEIGHTS_SQL} ON r.symbol = w.symbol
    WHERE r.ret IS NOT NULL
    GROUP BY r.ts
    ORDER BY r.ts
    """
).to_pandas()

# Keep only days where all 12 names have a return (guards against partial days).
port = port[port["names"] == 12].set_index("ts")
port["pnl"] = NOTIONAL * port["port_ret"]
print(f"{len(port):,} trading days, "
      f"ann. vol {port['port_ret'].std() * np.sqrt(252):.1%}, "
      f"worst day {port['pnl'].min():,.0f} USD")

# %% [markdown]
# ## 3. ローリングのヒストリカル VaR／ES と、パラメトリックの派生形
#
# どの尺度も過去252日の窓を使い、それを**1日ずらします**。つまり *t* 日について報告される
# VaR は、*t−1* 日までの情報しか使いません。
#
# このずらしが、リスクモデルと後知恵モデルを分けます。そして監査人がまさに尋ねてくる細部
# でもあります。
#
# - **ヒストリカル。** 窓の経験分位点。ES はその分位点より外側の裾の平均です。
# - **パラメトリック正規。** ローリングの平均とボラティリティに、正規分位点を組み合わせます。
# - **パラメトリック Student-t。** 同じものを、単位分散にスケールし直した t 分位点で。自由度は
#   標準化した全標本に対して1度だけ推定します。安価な裾の厚み対策です。

# %%
from scipy import stats

r = port["port_ret"]
W = 252

def hist_var(x, q):
    return -np.quantile(x, q)

def hist_es(x, q):
    cut = np.quantile(x, q)
    return -x[x <= cut].mean()

roll = r.rolling(W)
port["var95"] = roll.apply(hist_var, args=(0.05,), raw=True).shift(1)
port["var99"] = roll.apply(hist_var, args=(0.01,), raw=True).shift(1)
port["es95"] = roll.apply(hist_es, args=(0.05,), raw=True).shift(1)
port["es99"] = roll.apply(hist_es, args=(0.01,), raw=True).shift(1)

mu, sd = roll.mean().shift(1), roll.std().shift(1)
port["var95_norm"] = -(mu + sd * stats.norm.ppf(0.05))

# Fit t df on the full standardized history, rescale to unit variance.
z = ((r - r.mean()) / r.std()).to_numpy()
nu = stats.t.fit(z, floc=0)[0]
t_scale = np.sqrt((nu - 2) / nu)
port["var95_t"] = -(mu + sd * stats.t.ppf(0.05, nu) * t_scale)

port[["var95", "es95", "var95_norm", "var95_t"]].dropna().tail(3).mul(NOTIONAL).round(0)

# %% [markdown]
# ## 4. Kupiec の POF バックテスト: 95% は本当に 95% か
#
# 実現損失が前日の VaR を超えた日を数え、Kupiec の failure 比率の尤度比検定を走らせます。
# 漸近的に自由度1のカイ二乗分布に従います。
#
# きちんと較正された 95% の VaR なら、超過はおよそ5%の日に起きるはずです。帰無仮説が棄却
# されれば、モデルのカバレッジが間違っているということです。

# %%
def kupiec_pof(returns: pd.Series, var: pd.Series, p: float) -> dict:
    mask = var.notna()
    ret, v = returns[mask], var[mask]
    n = len(ret)
    x = int((ret < -v).sum())
    pi = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = ((n - x) * np.log(1 - pi) + x * np.log(pi)) if 0 < x < n else 0.0
    lr = -2 * (ll_null - ll_alt)
    return {"n_days": n, "breaches": x, "expected": round(n * p, 1),
            "breach_rate": round(pi, 4), "LR_pof": round(lr, 2),
            "p_value": round(float(stats.chi2.sf(lr, 1)), 4)}

backtest = pd.DataFrame(
    {
        "hist 95%": kupiec_pof(r, port["var95"], 0.05),
        "hist 99%": kupiec_pof(r, port["var99"], 0.01),
        "normal 95%": kupiec_pof(r, port["var95_norm"], 0.05),
        "t 95%": kupiec_pof(r, port["var95_t"], 0.05),
    }
).T
backtest

# %% [markdown]
# 表は出たとおりに読んでください。
#
# ヒストリカルと正規のモデルは 95% でカバレッジを保っていますが、99% のヒストリカル VaR は
# カバー不足です。超過はボラティリティのレジーム転換に固まり、過去の経験分位点はそれを拾うのが
# 遅いのです。Kupiec が検定するのはカバレッジだけで、独立性は見ません。固まり方を拾うなら
# Christoffersen の検定です。
#
# そして「裾の厚み対策」の罠にも注目してください。単位分散にスケールし直した Student-t は、
# 5% の水準では正規よりも*穏やかな*分位点になります。裾が逆転するのは 1% あたりからです。
# 95% で素朴に t モデルに差し替えると保守性が下がりますし、検定はそれを捕まえます。

# %% [markdown]
# ## 5. 監査証跡: バージョン管理された `risk_metrics` テーブル
#
# 本番のパターンは、**履歴をコミット1件で埋め戻し、そのあとは EOD の実行ごとにコミット1件**
# です。
#
# 日次の append はそれぞれアトミックで、ノートを持ちます。`versions()` は「どのリスクの数字が
# いつ公表され、正確には何時だったか」に、実時刻のコミット時刻つきで答えます。追加のログ基盤は
# 要りません。
#
# 以下では直近60営業日を個別のコミットとして再現します。

# %%
metrics = port.dropna(subset=["var95", "var99", "es95", "es99"])[
    ["pnl", "var95", "es95", "var99", "es99"]
].copy()
for c in ["var95", "es95", "var99", "es99"]:
    metrics[c] *= NOTIONAL

mschema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pnl", pa.float64()),
        pa.field("var95", pa.float64()),
        pa.field("es95", pa.float64()),
        pa.field("var99", pa.float64()),
        pa.field("es99", pa.float64()),
    ]
)
db.create_table("risk_metrics", mschema, time_column="ts")

hist_part = metrics.iloc[:-60].reset_index()
db.append("risk_metrics", pa.Table.from_pandas(hist_part, schema=mschema, preserve_index=False),
          note="backfill through model go-live")

for day, row in metrics.iloc[-60:].iterrows():
    one = pa.Table.from_pandas(
        pd.DataFrame([row]).rename_axis("ts").reset_index(), schema=mschema, preserve_index=False
    )
    db.append("risk_metrics", one, note=f"eod risk {day.date()}")

[
    {k: v.get(k) for k in ("sequence", "op", "rows", "note")}
    for v in db.versions("risk_metrics")[-4:]
]

# %% [markdown]
# 小さなコミット60件は、小さな Parquet セグメント60個でもあります。`compact()` がそれを効率の
# よいセグメントに併合します。これも監査されるコミット1件なので、メンテナンスまで履歴に残り
# ます。

# %%
before = db.versions("risk_metrics")[-1]
compacted = db.compact("risk_metrics", note="post-EOD-loop compaction")
print(f"segments: {before['segments']} -> {compacted['segments_total']}, "
      f"op={compacted['op']}, sequence={compacted['sequence']}")

# %% [markdown]
# リスクのテーブルは、ほかと同じように引けるようになりました。直近2年の超過日を、SQL から
# そのまま出します。

# %%
(
    db.table("risk_metrics")
    .filter(col("pnl") < -col("var95"), col("ts") >= "2024-07-01T00:00:00Z")
    .select(
        "ts",
        pnl=col("pnl").round(),
        var95_floor=(-col("var95")).round(),
        var99_floor=(-col("var99")).round(),
    )
    .sort("ts", descending=True)
    .limit(8)
    .to_pandas()
)

# %% [markdown]
# ## 6. VaR のバンドに対する P&L、超過を印つきで

# %%
import matplotlib.pyplot as plt

plot_df = metrics.loc["2024-07-01":]
breach95 = plot_df[plot_df["pnl"] < -plot_df["var95"]]
breach99 = plot_df[plot_df["pnl"] < -plot_df["var99"]]

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(plot_df.index, plot_df["pnl"], lw=0.6, color="0.4", label="daily P&L")
ax.plot(plot_df.index, -plot_df["var95"], lw=1.2, color="tab:orange", label="-VaR 95%")
ax.plot(plot_df.index, -plot_df["var99"], lw=1.2, color="tab:red", label="-VaR 99%")
ax.scatter(breach95.index, breach95["pnl"], s=28, color="tab:orange", zorder=3,
           label=f"95% breach ({len(breach95)})")
ax.scatter(breach99.index, breach99["pnl"], s=36, color="tab:red", marker="x", zorder=4,
           label=f"99% breach ({len(breach99)})")
ax.set_title("Daily P&L vs rolling 252d historical VaR ($10M book)")
ax.set_xlabel("date")
ax.set_ylabel("USD")
ax.legend(loc="lower left", ncols=3, fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - リターンのパイプライン全体、つまり銘柄ごとの `lag()` リターンと加重集約は、時刻順
#   ストレージの上の SQL 1文です。pandas が出てくるのはローリング分位点のところだけです。
# - VaR の窓は1日ずらしてください。そうすれば Kupiec の POF 検定がカバレッジの成否を正直に
#   教えてくれますし、99% でのヒストリカルモデルの失敗は言い伝えではなく数字に現れます。
# - `risk_metrics` は*ただで手に入る監査証跡*です。EOD の実行ごとにノート付きの append を1件
#   積めば、`versions()` がこれまで報告したすべてのリスクの数字の正確な公表時刻を返します。
#   追加のログ基盤は不要です。
# - 小さな日次コミットが多くても構いません。定期的に `compact()` を走らせてください。それ自体
#   もバージョン管理されたノート付きのコミットです。

# %%
db.close()
