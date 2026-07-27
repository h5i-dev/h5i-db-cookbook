# %% [markdown]
# # 不規則な市場に規則的な格子を: gapfill と resample
#
# 流動性の低い銘柄は毎分は約定しません。ところが下流のほとんど――共分散行列、流動性の
# 高いベンチマークとのジョイン、リスク評価――は、規則的な時間格子を欲しがります。h5i-db の
# `gapfill()`（別名 `resample()`）は、保存済みのバーのテーブルを SQL の1呼び出しで規則的な
# 格子に変えます。埋め方は3つ。`'null'`（穴を見えるまま残す）、`'locf'`（直近の観測値を
# 前方に持ち越す）、`'interpolate'` です。どれを選ぶかはモデリング上の判断で、P&L に
# はっきり効きます。このレシピの締めは古典的な罠、locf の陳腐化した価格が生む幻のリターン
# （ゼロリターン）です。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
from h5i_db import col, count_star, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_gapfill"), create=True)

# %% [markdown]
# ## 1. 流動性のあるテープの中の、流動性のない銘柄
#
# 3銘柄・5セッションぶんのティックを用意してから、NVDA を約定数の2%（1日あたり約400件）まで
# 間引きます。よくて1分に数回しか約定しない小型株の代役です。

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7
)
tp = trades.to_pandas()
rng = np.random.default_rng(5)
keep = (tp["symbol"] != "NVDA") | (rng.random(len(tp)) < 0.02)
tp = tp[keep]

ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64()), pa.field("exchange", pa.string()),
     pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(tp, preserve_index=False).cast(trade_schema))

db.table("trades").group_by("symbol").agg(n_trades=count_star()).sort("symbol").to_pandas()

# %% [markdown]
# ## 2. 穴の空いたバー
#
# NVDA の1分足終値です。h5i-db 固有の設計上のポイントが1つあります。`gapfill()` は
# **保存済みのテーブル名**に対して働き、テーブル全体を1本の系列として扱います。partition-by の
# 引数はありません。したがって手順は、銘柄ごとのバー系列を作り、それ自体を1つのテーブルとして
# 保存し、そのテーブルを gapfill する、という形になります。（複数銘柄のユニバースなら、系列
# 1本につきバーのテーブルを1つ書くか、絞り込んだコピーを gapfill します。）

# %%
bar_schema = pa.schema(
    [ts_field, pa.field("close", pa.float64()), pa.field("volume", pa.int64())]
)
nvda_bars = (
    db.table("trades")
    .filter(col("symbol") == "NVDA")
    .group_by(time_bucket("1m", col("ts")).alias("bar"))
    .agg(close=col("price").last("ts"), volume=col("size").sum())
    .select(col("bar").alias("ts"), "close", "volume")
    .sort("ts")
    .to_arrow()
)
db.create_table("bars_nvda_1m", bar_schema, time_column="ts", sort_key=["ts"])
db.append("bars_nvda_1m", nvda_bars.cast(bar_schema))

n_bars = len(nvda_bars)
session_minutes = 5 * 390  # 13:30-20:00 UTC, five sessions
print(f"{n_bars:,} one-minute bars observed out of {session_minutes:,} session minutes "
      f"({n_bars / session_minutes:.0%} coverage)")

# %% [markdown]
# ## 3. `gapfill` で規則的な格子に載せる
#
# 刻みは**生のマイクロ秒**（`ts` 列の単位）で、1分なら 60_000_000 です。格子はテーブルの
# 最小タイムスタンプから最大タイムスタンプまでを覆い、夜間も含みます。分析の都合で立会
# 時間だけが必要なら、下流で絞り込んでください。

# %%
locf = db.sql(
    "SELECT * FROM gapfill('bars_nvda_1m', 'ts', 60000000, 'locf')"
).to_pandas()
print(f"grid rows: {len(locf):,} (observed bars: {n_bars:,})")
locf.head(8)

# %% [markdown]
# 上の出力に注意書きが1つ見えています。locf は `volume` も含めて**すべて**の列を前方に
# 持ち越すので、幻の出来高が生まれます。持ち越しで評価するなら、価格だけの系列を保存するか
# （ここでの `close` のように）、`'null'` モードで出して `COALESCE(volume, 0)` を自分で
# 当ててください。
#
# ## 4. 3つの埋め方を並べて見る
#
# 同じ呼び出しを3つのモードで、穴の空いた午前の窓に当てます。`gapfill` は他のテーブルと
# 同じように合成できるので、外側の `WHERE` はその出力を絞り込むだけです。

# %%
window = {}
for mode in ("null", "locf", "interpolate"):
    window[mode] = db.sql(
        f"""
        SELECT ts, close FROM gapfill('bars_nvda_1m', 'ts', 60000000, '{mode}')
        WHERE ts >= '2026-06-02T14:00:00Z' AND ts < '2026-06-02T16:00:00Z'
        ORDER BY ts
        """
    ).to_pandas()

obs = window["null"].dropna(subset=["close"])
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(window["locf"]["ts"], window["locf"]["close"],
        drawstyle="steps-post", lw=1.2, label="locf (carry forward)")
ax.plot(window["interpolate"]["ts"], window["interpolate"]["close"],
        lw=1.2, ls="--", label="interpolate")
ax.scatter(obs["ts"], obs["close"], s=22, color="black", zorder=3,
           label="observed 1m closes")
ax.set_title("NVDA (thinned) 1m closes, 2026-06-02 14:00-16:00: fill policies")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 3つのうち2つは別々の意味で安全で、1つは危険です。`'null'` は穴を正直に残します。
# `'locf'` は因果的（過去のデータしか使わない）ですが陳腐化します。そして
# `'interpolate'` は**先読みします**。埋めた点はどれも*次の*観測値を使うので、補間した
# 系列をバックテストのシグナルに流し込んでは絶対にいけません。
#
# ## 5. `resample` という別名と、行数のガード
#
# `resample(...)` は `gapfill(...)` の完全な別名です。どちらも100万行を超える生成を
# 拒みます。刻みを打ち間違えた例（下では3日間を100ミリ秒刻みで）は、何百万行も黙って
# 実体化する代わりに `LimitError` を上げます。密で連続した系列なら真っ当な使い道になります。
# 72時間ぶんの EURUSD ティックを1秒格子に載せてみます。

# %%
fx = cu.make_fx_ticks(pairs=["EURUSD"], hours=72).to_pandas()
fx["mid"] = (fx["bid"] + fx["ask"]) / 2
fx_schema = pa.schema([ts_field, pa.field("mid", pa.float64())])
db.create_table("fx_mid", fx_schema, time_column="ts", sort_key=["ts"])
db.append("fx_mid", pa.Table.from_pandas(fx[["ts", "mid"]], preserve_index=False).cast(fx_schema))

grid_1s = db.sql(
    "SELECT count(*) AS n, count(mid) AS filled FROM resample('fx_mid', 'ts', 1000000, 'locf')"
).to_pandas()
print(f"1s grid: {grid_1s['n'][0]:,} rows from {len(fx):,} irregular ticks")

try:
    db.sql("SELECT count(*) FROM resample('fx_mid', 'ts', 100000, 'locf')")
except h5i_db.LimitError as e:
    print(f"100ms grid refused -> {type(e).__name__} [{e.code}]: {e}")

# %% [markdown]
# ## 6. 幻のリターン: 埋め方がモデリング上の判断である理由
#
# 流動性の低い銘柄を locf 格子で評価すると、分次リターンのほとんどがゼロになります。
# 陳腐化した価格が「動かなかった」の顔をしているのです。同じリターン系列を両方のやり方で
# 計算します（日中のペアのみ、夜間は除外）。

# %%
def session_returns(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    mins = d["ts"].dt.hour * 60 + d["ts"].dt.minute
    d = d[(mins >= 13 * 60 + 30) & (mins < 20 * 60)]
    ret = np.log(d["close"]).diff()
    ret[d["ts"].dt.date != d["ts"].shift().dt.date] = np.nan  # drop overnight pair
    return ret.dropna()

null_grid = db.sql(
    "SELECT ts, close FROM gapfill('bars_nvda_1m', 'ts', 60000000, 'null') ORDER BY ts"
).to_pandas()

ann = np.sqrt(390 * 252)  # per-minute -> annualized
r_locf = session_returns(locf)
r_null = session_returns(null_grid)  # defined only where adjacent minutes both traded

for name, r in (("locf grid", r_locf), ("null grid", r_null)):
    print(f"{name}: {len(r):>5,} minute returns | {(r == 0).mean():>4.0%} exactly zero | "
          f"ann. vol {r.std() * ann:.1%} | excess kurtosis {r.kurtosis():.1f}")

# %% [markdown]
# locf が何を歪め、何を歪めないかに注目してください。分散の合計はおおむね保たれます。
# 価格が据え置かれた区間は、本当の値動きを1回の遅れたジャンプに畳み込むので、見出しの
# 実現ボラティリティはほとんど動きません。壊れるのは*分布*のほうです。半数近くの分が
# きっちりゼロの変動を報告し、超過尖度が爆発します（約5.5 対 約0.5）。つまり locf の系列は
# 「何も起きなかった」の連続を、値動きの時点ではなく次の約定の時点に置かれた架空のジャンプが
# 区切る形になります。分次リターンを使うもの――短期リスク、ジャンプ検出、流動性の高い銘柄との
# 相互相関（Epps 効果）――は、その作り物をそのまま受け継ぎます。本番で無難なのは、観測された
# バーを保存し、`'null'` 格子を配り、持ち越しは利用側で明示的に記録された1工程として行う形です。
#
# ## まとめ
#
# - `gapfill('table', 'ts', step_us, mode)` と `resample(...)` は、保存済みのバーの
#   テーブルを1呼び出しで規則的な格子に変えます。刻みは生のマイクロ秒（1分＝60_000_000）で、
#   生成は100万行で打ち止めです（OOM ではなく `LimitError`）。
# - テーブルを1本の系列として扱うので、銘柄ごとにバーのテーブルを保存します（h5i-db では
#   バーのテーブルも安価なバージョン管理テーブルなので、自然な形です）。
# - `'locf'` は因果的ですが陳腐化し、しかも出来高を含む*すべて*の列を持ち越します。
#   `'interpolate'` は先読みするのでバックテストに流してはいけません。`'null'` は穴を
#   正直に残します。
# - 幻のリターンは測れます。locf 格子では約半数の分がゼロリターンになり、尖度は観測ペアの
#   系列の約10倍です。見出しのボラティリティは生き延びますが、リターンの*経路*に敏感なものは
#   すべて壊れます。

# %%
db.close()
