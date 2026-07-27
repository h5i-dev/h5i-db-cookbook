# %% [markdown]
# # 不規則な市場に規則正しい格子を: gapfill と resample
#
# 流動性の低い銘柄は毎分は約定しません。それでも下流のほとんどは規則正しい時間格子を欲しがり
# ます。共分散行列、流動性の高いベンチマークとの結合、リスクの評価。
#
# `gapfill()`、別名 `resample()` は、保存済みのバーのテーブルを SQL 呼び出し1つで規則正しい
# 格子に変えます。埋め方は3種類。`'null'` は穴を見えるまま残し、`'locf'` は直前の観測値を
# 持ち越し、`'interpolate'` は前後を直線で結びます。
#
# どれを選ぶかはモデリングの決定で、P&L にも効いてきます。このレシピの締めくくりは定番の罠、
# 古いままの locf 価格が生む幻のリターンです。

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
# ## 1. データ
#
# `cu.make_trades` の5セッションぶんのティックデータで、1行が1約定です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |

# %%
trades = cu.make_trades(
    symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7
)
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# 流動性の高いテープの中に流動性の低い銘柄を作るため、NVDA のプリントを2%、1日あたり
# およそ400約定まで間引きます。これで、よくても毎分数回しか約定しない小型株の代役になります。
# AAPL と MSFT はテープをそのまま保ちます。

# %%
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
# ## 2. 穴のあいたバー
#
# NVDA の1分足の終値です。h5i-db の設計上の性質が1つ、使い方を決めます。`gapfill()` は
# **保存済みのテーブル名**に対して働き、テーブル全体を1本の系列として扱います。partition-by の
# 引数はありません。
#
# だから手順はこうなります。銘柄ごとのバー系列を作り、それ自体のテーブルとして保存し、その
# テーブルを gapfill する。複数銘柄のユニバースなら、系列ごとに1つのバーテーブルを書くか、
# 絞り込んだコピーを gapfill します。

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
# ## 3. `gapfill` で規則正しい格子に乗せる
#
# 刻みは**生のマイクロ秒**で、`ts` 列の単位に合わせます。1分なら 60_000_000 です。
#
# 格子はテーブルの最小から最大のタイムスタンプまでを覆い、夜間も含みます。分析にセッション
# 時間だけが要るなら、下流で絞り込んでください。

# %%
locf = db.sql(
    "SELECT * FROM gapfill('bars_nvda_1m', 'ts', 60000000, 'locf')"
).to_pandas()
print(f"grid rows: {len(locf):,} (observed bars: {n_bars:,})")
locf.head(8)

# %% [markdown]
# 上の出力に細かい注意点が見えています。locf は `volume` を含めて**すべての**列を持ち越すので、
# 幻の出来高という副作用が出ます。持ち越しで評価するなら、ここでの `close` のように価格だけの
# 系列を保存するか、`'null'` モードで走らせて `COALESCE(volume, 0)` を自分で当ててください。
#
# ## 4. 3つの埋め方を並べて見る
#
# 同じ呼び出しを3モード、穴の多い午前の窓に当てます。`gapfill` は他のテーブルと同じように
# 組み合わせられるので、外側の `WHERE` はその出力を絞るだけです。

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
# このうち2つは別々の意味で安全で、1つは危険です。`'null'` は穴を正直に残します。`'locf'` は
# 過去のデータしか使わないので因果的ですが、古くなります。`'interpolate'` は**先を見ます**。
# 埋めた各点が*次の*観測値を使うので、内挿した系列をバックテストのシグナルに流してはいけません。
#
# ## 5. `resample` という別名と、行数のガード
#
# `resample(...)` は `gapfill(...)` の完全な別名です。どちらも100万行を超える生成を拒むので、
# 刻みを打ち間違えても、何百万行も黙って実体化するかわりに `LimitError` が上がります。下では
# 3日間に対する 100ms がそのガードに引っかかります。
#
# 密で連続した系列が、正直な使いどころを見せてくれます。72時間ぶんの EURUSD ティックを
# 1秒格子に乗せます。`cu.make_fx_ticks` が返すのは `ts`、`pair`、`bid`、`ask` で、ここからは
# ミッドだけを使います。

# %%
fx = cu.make_fx_ticks(pairs=["EURUSD"], hours=72).to_pandas()
fx["mid"] = (fx["bid"] + fx["ask"]) / 2
print(f"{len(fx):,} EURUSD ticks over 72h")
fx.head()

# %%
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
# ## 6. 幻のリターン: 埋め方がモデリングの決定である理由
#
# 流動性の低い銘柄を locf の格子で評価すると、分次リターンはほとんどゼロになります。古いままの
# 価格が「動きなし」のふりをするのです。
#
# 以下では同じリターン系列を両方のやり方で計算します。日中のペアだけを使い、オーバーナイトの
# ギャップは除きます。

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
# locf が歪めるものと歪めないものに注目してください。
#
# 分散の総量はおおむね保たれます。値が固まっている区間は本当の動きを1回の遅れたジャンプに
# まとめるので、表向きの実現ボラティリティはほとんど変わりません。
#
# 歪むのは*分布*です。分の半数近くがぴったりゼロの動きを報告し、超過尖度は 0.5 前後から
# 5.5 前後まで跳ね上がります。locf の系列は「何も起きていない」の連続を、動いた瞬間ではなく
# 次のプリントの時刻に置かれた架空のジャンプが区切る、という姿になります。
#
# 分次リターンを食べるものは、この副作用をそのまま受け継ぎます。短期のリスク、ジャンプ検出、
# 流動性の高い銘柄との相互相関（Epps 効果）。まともな本番のパターンは、観測されたバーを保存し、
# `'null'` の格子を公開して、持ち越しは使う側で明示的に、文書化したうえで行うことです。
#
# ## まとめ
#
# - `gapfill('table', 'ts', step_us, mode)` と `resample(...)` は、保存済みのバーのテーブルを
#   呼び出し1つで規則正しい格子に変えます。刻みは生のマイクロ秒（1分 = 60_000_000）で、生成は
#   100万行で頭打ちになり、OOM ではなく `LimitError` が上がります。
# - テーブルを1本の系列として扱うので、銘柄ごとに1つのバーテーブルを保存してください。
#   h5i-db ではバーのテーブルもバージョン管理された安いテーブルなので、無理のない形です。
# - `'locf'` は因果的ですが古くなり、しかも出来高を含む*すべての*列を持ち越します。
#   `'interpolate'` は先を見るので、バックテストに流してはいけません。`'null'` は穴を正直に
#   残します。
# - 幻のリターンは測れます。locf の格子では分のおよそ半数がゼロリターンになり、尖度は観測
#   ペアの系列のおよそ10倍です。表向きのボラティリティは生き残りますが、リターンの*経路*に
#   敏感なものはすべて壊れます。

# %%
db.close()
