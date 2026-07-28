# %% [markdown]
# # クロスセクション・モメンタム: 正直な月次バックテスト
#
# 定番の 12-1 モメンタムファクターを、実際の価格で端から端まで。シグナルの計算は SQL の
# ウィンドウクエリ、月次のリバランス日は `time_bucket('1mo', ...)`、そしてシグナルの
# テーブル自体をバージョン管理してスナップショットで固定します。
#
# 最後の1つを、たいていのバックテスト基盤は落とします。半年後に、この実行が*何を見ていたか*
# を正確に再現できるべきです。
#
# 戦略そのものはあえて平凡にしてあります。その周りにある h5i-db 流のワークフローが主題です。
#
# このレシピで進めるのは次の4つです。
#
# 1. 実際の日足をバージョン管理されたテーブルに読み込む
# 2. 月次のリバランス格子の上で 12-1 シグナルを1つの文で計算する
# 3. ポートフォリオを組み、コストを引いた正直な P&L を出す
# 4. 価格とシグナルをまとめてスナップショットし、再現性の層を作る

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_momentum"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.fetch_daily` が返すのは、大型株30銘柄の 2018〜2026 年の実際の日足です。Yahoo Finance
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
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
print(f"{daily.num_rows:,} rows x {daily.num_columns} columns, "
      f"{len(set(daily['symbol'].to_pylist()))} symbols")
daily.to_pandas().head()

# %% [markdown]
# 保存するのはこの研究に要る `ts`、`symbol`、`adj_close` だけです。`sort_key` を
# `["ts", "symbol"]` にしておくと、銘柄ごとのウィンドウ走査が順序どおりに流れます。
#
# `append` はそのソートキーに厳格です。入力は `ts` 順、同じタイムスタンプの中では `symbol`
# 順に並んでいる必要があります。

# %%
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
# 12-1 モメンタムは、12か月前から1か月前までのリターンです。直近1か月を飛ばすのは短期リバー
# サルを避けるためです。営業日の系列なら、銘柄ごとに `lag(px, 21) / lag(px, 252) - 1` に
# なります。
#
# リバランスの格子は `time_bucket('1mo', ts)` から来ます。各暦月の最終営業日です。2つを結合
# すれば月末のシグナルパネルが出てきます。pandas のリサンプリングも、カレンダーの例外処理も
# 要りません。

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
# ## 3. ポートフォリオの構築と正直な P&L
#
# 30銘柄しかないので「デシル」は3銘柄になってしまいます。そこで上位10銘柄をロング、下位
# 10銘柄をショートし、等ウェイトで持ちます。ターシルに近く、ノイズも小さくなります。
#
# 規律のチェックリストはこうです。
#
# - **先読みなし。** 月末 *t* で観測したシグナルが、*t* から *t+1* のリターンを取ります。
#   最後の月には先のリターンがないので落とします。
# - **コスト。** 片道のターンオーバーに 10bps。ウェイトが動くたび、最初の組成も含めて課金
#   します。
# - **組入れ資格。** 順位付けの対象になるには252日の履歴が要ります。順位付けできる銘柄が
#   25未満の月は飛ばします。

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
# ロング／ショートの本はおおむねマーケットニュートラルなので、ロングオンリーのベンチマークと
# 比べるのは水準ではなく*形*の話です。ベンチマークは 2018〜2026 年の強気相場に乗り、モメン
# タムの本はおおむね足踏みします。

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
# モメンタムの本は**シャープレシオがほぼゼロ**に着地します。これはバグではありません。
# 大型株30銘柄はモメンタムのユニバースとしては最悪で、このプレミアムは歴史的にもっと広く、
# もっと小型のクロスセクションに住んでいます。加えて 2020〜2023 年にはモメンタムの
# クラッシュが何度もありました。
#
# カーブが見栄えよくなるまでルックバックをいじりたくなるのを我慢してください。バックテストは
# そうやって死にます。出たとおりに報告しましょう。

# %% [markdown]
# ## 5. シグナルをバージョン管理する: 再現性の層
#
# これらの数字を生んだパネルを `signals` テーブルとして h5i-db に戻し、**名前付きスナップ
# ショット**で `prices` と `signals` の両方をこの状態に固定します。
#
# あとから誰でも `h5i('signals', 'mom-run-001')` を引けますし、この実行が見た価格も読み直せ
# ます。両方のテーブルが先へ進んだあとでもです。リサーチ基盤に本当に要る監査証跡はこれです。

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
# - `lag(...) OVER (PARTITION BY symbol ORDER BY ts)` と `time_bucket('1mo', ts)` の組み合わせ
#   が、「月末日で 12-1 モメンタムを計算する」を、整列済みストレージの上を流れる SQL 1文に
#   します。
# - バックテストの衛生管理は安上がりです。シグナルを1期ずらし、ターンオーバーに課金し、
#   分からない最後の月を落とす。毎回やりましょう。
# - 出てきた結果が正直なものです。大型株モメンタムのシャープはほぼゼロ、月次ターンオーバーは
#   およそ70%で、片道 10bps を食っていきます。大型株30銘柄はデモであってアルファの源では
#   ありません。
# - `db.snapshot("mom-run-001", ...)` は価格*と*シグナルを、名前の付いた1つの引ける状態に
#   固定します。`h5i('signals', 'mom-run-001')` はこの実行を永久に再現しますし、CSV の
#   発掘作業も要りません。

# %%
db.close()
