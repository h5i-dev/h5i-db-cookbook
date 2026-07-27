# %% [markdown]
# # クロスセクショナル・モメンタム: 誠実な月次バックテスト
#
# 古典的な12-1モメンタムファクターを、実際の価格で端から端まで作ります。シグナルの計算は
# SQL のウィンドウクエリ、リバランス日は `time_bucket('1mo', ...)` から取り、そして多くの
# バックテスト基盤が落とすところ――半年後にこの実行が*何を見たか*を正確に再現できるよう、
# シグナルのテーブルをバージョン管理してスナップショットを打ちます。戦略そのものはあえて
# 平凡です。主役はその周りにある h5i-db 流の作業手順のほうです。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_momentum"), create=True)

# %% [markdown]
# ## 1. 実際の価格をバージョン管理テーブルへ
#
# 大型株30銘柄の日次、2018〜2026年（キャッシュ済みの Yahoo Finance データ）。研究に必要な
# `ts, symbol, adj_close` だけを保存し、`sort_key` は `["ts", "symbol"]` にします。これで
# 銘柄別のウィンドウスキャンが順序どおりに流れます。`append` はこのソートキーに厳格で、入力は
# `ts` 順、同じタイムスタンプの中では `symbol` 順に並んでいる必要があります。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])

prices = (
    daily.select(["ts", "symbol", "adj_close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
    .cast(schema)
)
commit = db.append("prices", prices, note="yfinance 30 names 2018-2026")
print(f"version {commit['sequence']}: {commit['rows_total']:,} rows")

# %% [markdown]
# ## 2. シグナルを SQL 1文で
#
# 12-1モメンタムは、12か月前から1か月前までのリターンで、直近1か月は飛ばします（短期の
# リバーサルを避けるため）。立会日ベースの系列なら、銘柄ごとの `lag(px, 21) / lag(px, 252) - 1`
# です。リバランスの格子は `time_bucket('1mo', ts)` から、つまり各暦月の最終立会日から取ります。
# 2つをジョインすれば月末のシグナルパネルが出ます。pandas のリサンプルも、カレンダーの
# 例外処理も要りません。

# %%
def lag_px(n: int):
    """lag(adj_close, n) per symbol - there is no .lag() verb, so sql_expr."""
    return sql_expr(f"lag(adj_close, {n})").over(partition_by="symbol", order_by="ts")


sig = db.table("prices").with_columns(px_1m_ago=lag_px(21), px_12m_ago=lag_px(252))

month_end = (
    db.table("prices")
    .group_by(time_bucket("1mo", col("ts")).alias("month"), "symbol")
    .agg(month_end=col("ts").max())
    .select(col("month_end").alias("ts"), "symbol")
)

panel = (
    sig.join(month_end, on=["ts", "symbol"])
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        close=col("adj_close", relation="l"),
        momentum=col("px_1m_ago", relation="l") / col("px_12m_ago", relation="l") - 1,
    )
    .sort(["ts", "symbol"])
    .to_pandas()
)
panel.tail(4)

# %% [markdown]
# ## 3. ポートフォリオ構築と、誠実な損益
#
# 30銘柄しかないので「デシル」は3銘柄になってしまいます。そこで上位10銘柄をロング、下位
# 10銘柄をショートし、等ウェイトにします。三分位に近く、ノイズも小さくなります。守るべき
# 点は3つです。
#
# - **先読みなし**: 月末 *t* に観測したシグナルが、*t* から *t+1* までのリターンを取ります。
#   最終月には先のリターンがないので落とします。
# - **コスト**: 片道回転率に10bps を、ウェイトが変わるたびに（初回の構築も含めて）課します。
# - **採用条件**: ランク付けには252日の履歴が必要です。ランク付けできる銘柄が25未満の月は
#   飛ばします。

# %%
mom = panel.pivot(index="ts", columns="symbol", values="momentum")
close = panel.pivot(index="ts", columns="symbol", values="close")
fwd_ret = close.pct_change().shift(-1)  # month-end t -> t+1, aligned at t

N_SIDE, COST_BPS = 10, 10
ranks = mom.rank(axis=1)  # NaNs are left unranked
n_valid = mom.notna().sum(axis=1)

weights = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
weights[ranks.ge(ranks.max(axis=1).to_numpy()[:, None] - N_SIDE + 1)] = 1 / N_SIDE
weights[ranks.le(N_SIDE)] = -1 / N_SIDE
weights.loc[n_valid < 25] = 0.0

turnover = weights.diff().abs().sum(axis=1)
turnover.iloc[0] = weights.iloc[0].abs().sum()

gross = (weights * fwd_ret).sum(axis=1)
net = (gross - COST_BPS / 1e4 * turnover).iloc[:-1]  # drop month with no fwd ret
bench = fwd_ret.mean(axis=1).iloc[:-1]  # equal-weight long-only, no costs


def perf_stats(r: pd.Series, periods: int = 12) -> dict:
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    return {
        "ann_ret": r.mean() * periods,
        "ann_vol": r.std() * np.sqrt(periods),
        "sharpe": r.mean() / r.std() * np.sqrt(periods),
        "max_dd": dd.min(),
    }


stats = pd.DataFrame(
    {"momentum L/S (net)": perf_stats(net), "equal-weight bench": perf_stats(bench)}
).T
print(f"avg monthly one-way turnover: {turnover.mean():.0%}")
stats.round(3)

# %% [markdown]
# ## 4. エクイティカーブ
#
# ロング・ショートの帳簿はおおむね市場中立なので、ロングオンリーのベンチマークとの比較で
# 見るべきは水準ではなく*形*です。ベンチマークが2018〜2026年の上げ相場に乗るあいだ、
# モメンタムの帳簿はおおむね足踏みします。

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot((1 + net).cumprod(), label=f"12-1 momentum L/S, net {COST_BPS}bps", lw=1.4)
ax.plot((1 + bench).cumprod(), label="equal-weight 30 names (long-only)", lw=1.4)
ax.axhline(1.0, color="gray", lw=0.6, ls=":")
ax.set_title("Monthly 12-1 momentum on 30 large caps, 2019-2026")
ax.set_xlabel("rebalance date")
ax.set_ylabel("growth of $1")
ax.legend()
fig.tight_layout()

# %% [markdown]
# モメンタムの帳簿は、ここでは **Sharpe がほぼゼロ**に着地します。これはバグではありません。
# 超大型株30銘柄はモメンタムのユニバースとして最悪ですし（プレミアムは歴史的に、もっと広く
# 小型を含むクロスセクションに住んでいます）、2020〜2023年にはモメンタムの手ひどい崩壊が
# ありました。カーブが見栄えよくなるまでルックバックをいじりたくなる衝動は抑えてください。
# バックテストはそうやって死にます。出たとおりに報告しましょう。

# %% [markdown]
# ## 5. シグナルにバージョンを付ける — 再現性の層
#
# この数字を生んだパネルを `signals` テーブルとして h5i-db に戻し、**名前付きスナップ
# ショット**で `prices` と `signals` の両方をこの状態に固定します。両方のテーブルが先へ
# 進んだあとでも、誰でも `h5i('signals', 'mom-run-001')` をクエリでき、この実行が見た価格を
# 読み直せます。リサーチ基盤に本当に必要な監査証跡はこれです。

# %%
sig_out = (
    pd.concat(
        [
            mom.stack().rename("momentum"),
            weights.stack().rename("weight"),
        ],
        axis=1,
    )
    .dropna(subset=["momentum"])
    .reset_index()
    .rename(columns={"level_0": "ts", "level_1": "symbol"})
    .sort_values(["ts", "symbol"])
)

sig_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("momentum", pa.float64()),
        pa.field("weight", pa.float64()),
    ]
)
db.create_table("signals", sig_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "signals",
    pa.Table.from_pandas(sig_out, preserve_index=False).cast(sig_schema),
    note="12-1 momentum, top/bottom 10, 10bps",
)
db.snapshot("mom-run-001", tables=["prices", "signals"], note="momentum backtest run 001")

count_if = lambda flag: when(flag).then(lit(1)).otherwise(lit(0)).sum()

(
    db.table("signals", snapshot="mom-run-001")
    .group_by("ts")
    .agg(
        names=count_star(),
        longs=count_if(col("weight") > 0),
        shorts=count_if(col("weight") < 0),
    )
    .sort("ts", descending=True)
    .limit(3)
    .to_pandas()
)

# %%
[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("signals")
]

# %% [markdown]
# ## まとめ
#
# - `lag(...) OVER (PARTITION BY symbol ORDER BY ts)` と `time_bucket('1mo', ts)` があれば、
#   「月末の日付で12-1モメンタムを計算する」は SQL 1文になり、整列済みストレージの上を
#   ストリーミングで流れます。
# - バックテストの衛生管理は安く済みます。シグナルを1期ずらし、回転コストを課し、知りようの
#   ない最終月を落とす。毎回やってください。
# - 結果は――大型株モメンタムの Sharpe がほぼゼロ、月次回転率約70%が片側10bps を食う――
#   誠実なものです。超大型株30銘柄はデモであって、アルファの源泉ではありません。
# - `db.snapshot("mom-run-001", ...)` は価格*と*シグナルを、名前が付いてクエリできる1つの
#   状態に固定します。`h5i('signals', 'mom-run-001')` があれば、この実行を永久に再現できます。
#   CSV の発掘は要りません。

# %%
db.close()
