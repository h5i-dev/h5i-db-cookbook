# %% [markdown]
# # NBBO の統合: 分断された取引所をまたぐ最良気配
#
# 米国株は十いくつもの取引所で売買され、それぞれが自分の板の最上段を配信します。執行品質が
# 測られる基準である統合最良気配（NBBO）は、*導出*しなければなりません。任意の瞬間に、各
# 取引所の直近の気配を取り、それをクロスセクションで比べて最良を選ぶのです。
#
# 「時刻 t における取引所ごとの直近の気配」は、まさに ASOF ジョインです。だから研究に使える
# サンプリング版の NBBO は SQL 数文で書けます。時間格子を取引所別の気配に ASOF 結合し、
# 瞬間ごとに `max(bid)` と `min(ask)` を取るだけです。
#
# このレシピで進めるのは次の4つです。
#
# 1. 統合された気配ストリームを、性格の異なる3つの取引所に展開する
# 2. 10秒格子の上で、各取引所のその時点の気配をスナップショットする
# 3. 統合した NBBO を、それ自体のテーブルとして保存する
# 4. どの取引所が最良を作るか、市場がどれくらいロックやクロスを起こすか、そして統合スプレッド
#    が単一の板よりどれだけ狭いかを測る

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_nbbo"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_quotes` の1セッションぶんの AAPL 気配で、統合フィード相当のものです。最良気配が
# 動くたびに1行です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 気配の時刻、昇順 |
# | `symbol` | `string` | 銘柄コード。ここでは `AAPL` のみ |
# | `bid`、`ask` | `float64` | 最良のビッドとオファー |
# | `bid_size`、`ask_size` | `int64` | 各サイドの表示数量 |

# %%
quotes = cu.make_quotes(
    symbols=["AAPL"], days=1, quotes_per_day=2_600, seed=11, base_prices={"AAPL": 320.0}
).to_pandas()
print(f"{len(quotes):,} rows x {quotes.shape[1]} columns")
quotes.head()

# %% [markdown]
# この1本のストリームを、それぞれ性格の違う3つの取引所に展開します。取引所は統合の更新に
# 合わせて出し直しますが、ときどき*取りこぼし*ます。ここが本物の古さの出どころです。さらに
# 自前のゲートウェイ遅延が乗り、価格は基準から1ティックずらされるか、ときには基準より内側に
# 入ります。
#
# この配合が、ある取引所を他よりも最良に近い存在にし、短いロックやクロスの状態を生みます。
# シードは通して固定です。
#
# 展開後のテーブルは1行が1取引所の気配で、列は `ts`、`venue`、`bid`、`ask`、`bid_size`、
# `ask_size` です。

# %%
rng = np.random.default_rng(5)
TICK = 0.01

VENUES = {
    #        P(miss update), latency (ms), P(1 tick worse), P(1 tick better)
    "ARCA": dict(drop=0.04, lat_ms=3.0, worse=0.25, better=0.02),
    "EDGX": dict(drop=0.08, lat_ms=10.0, worse=0.35, better=0.005),
    "XNAS": dict(drop=0.02, lat_ms=1.5, worse=0.15, better=0.03),
}

parts = []
for venue, v in VENUES.items():
    d = quotes[rng.random(len(quotes)) >= v["drop"]].copy()
    n = len(d)
    d["ts"] = d["ts"] + pd.to_timedelta(rng.exponential(v["lat_ms"], n), unit="ms").round("us")
    d["bid"] = d["bid"] - TICK * (rng.random(n) < v["worse"]) + TICK * (rng.random(n) < v["better"])
    d["ask"] = d["ask"] + TICK * (rng.random(n) < v["worse"]) - TICK * (rng.random(n) < v["better"])
    d["venue"] = venue
    parts.append(d)

venue_quotes = (
    pd.concat(parts)[["ts", "venue", "bid", "ask", "bid_size", "ask_size"]]
    .assign(bid=lambda x: x["bid"].round(2), ask=lambda x: x["ask"].round(2))
    .sort_values(["ts", "venue"], kind="stable")
    .reset_index(drop=True)
)
venue_quotes.assign(spread=venue_quotes["ask"] - venue_quotes["bid"]).groupby("venue").agg(
    quotes=("ts", "count"), avg_spread=("spread", "mean")
).round(4)

# %%
VQ_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("venue", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
    ]
)
db.create_table("venue_quotes", VQ_SCHEMA, time_column="ts", sort_key=["ts", "venue"])
db.append("venue_quotes", pa.Table.from_pandas(venue_quotes, preserve_index=False).cast(VQ_SCHEMA),
          note="per-venue AAPL quotes, 1 session, 3 venues")

# %% [markdown]
# ## 2. サンプリング格子と、取引所ごとの ASOF ジョイン
#
# イベント精度の NBBO は*すべての*気配更新で評価し直します。研究では、規則正しい格子の上で
# サンプリングした NBBO が標準的な近似で、誤差はサンプリング間隔で抑えられます。ここでは統合の
# 更新がおよそ9秒ごとに来るので、10秒でサンプリングします。
#
# 格子そのものが小さな h5i-db のテーブルで、(タイムスタンプ, 取引所) ごとに1行です。だから
# `asof_join(grid, venue_quotes, ..., 'venue')` 1つで、各瞬間の各取引所の気配をスナップショット
# できます。60秒の `tolerance` は生のマイクロ秒で、取引所の気配が古いと宣言して NULL を返し
# ます。死んだ板が最良に居座り続けるのを防ぐためです。
#
# ASOF のパイプラインでは習慣にしたいことが1つあります。ジョインが格子の行数ぴったりを返した
# と assert することです。LEFT の ASOF ジョインは左側と1対1なので、サイズやキーのミスがあれば
# 板を歪めるかわりに大きな音を立てて落ちます。

# %%
open_ts = venue_quotes["ts"].min().ceil("10s")
close_ts = pd.Timestamp("2026-06-01 20:00", tz="UTC")
grid_times = pd.date_range(open_ts, close_ts, freq="10s", inclusive="left")

grid = pd.DataFrame(
    {
        "ts": np.repeat(grid_times, len(VENUES)),
        "venue": np.tile(sorted(VENUES), len(grid_times)),
    }
)
GRID_SCHEMA = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("venue", pa.string())]
)
db.create_table("nbbo_grid", GRID_SCHEMA, time_column="ts", sort_key=["ts", "venue"])
db.append("nbbo_grid", pa.Table.from_pandas(grid, preserve_index=False).cast(GRID_SCHEMA))
print(f"grid: {len(grid_times)} instants x {len(VENUES)} venues = {len(grid):,} rows")

# %%
STALE_US = 60 * 1_000_000

# One frame, reused for every question below: each grid instant x venue gets
# the venue's prevailing quote, or NULL if the last one is older than STALE_US.
SNAPSHOT = db.table("nbbo_grid").join_asof(
    db.table("venue_quotes"), on="ts", by="venue",
    direction="backward", tolerance=STALE_US,
)

(
    SNAPSHOT.select("ts", "venue", "bid", "ask", quoted_at=col("ts_right"))
    .sort(["ts", "venue"])
    .limit(6)
    .to_pandas()
)

# %%
# Trust, but verify: one output row per grid row, and the SQL ASOF must agree
# with an independent pandas merge_asof on a venue's book.
state = SNAPSHOT.select("ts", "venue", "bid", "ask").sort(["ts", "venue"]).to_pandas()
assert len(state) == len(grid), f"ASOF join returned {len(state)} rows for {len(grid)} grid rows"

ref = pd.merge_asof(
    pd.DataFrame({"ts": grid_times}),
    venue_quotes[venue_quotes["venue"] == "XNAS"][["ts", "bid", "ask"]],
    on="ts", tolerance=pd.Timedelta("60s"),
)
chk = state[state["venue"] == "XNAS"].reset_index(drop=True)
assert chk["bid"].equals(ref["bid"]) and chk["ask"].equals(ref["ask"])
print(f"validated: {len(state):,} snapshot rows, XNAS column matches pandas merge_asof exactly")

# %% [markdown]
# ## 3. 統合して NBBO を保存する
#
# 取引所をまたいで最良を取るのは、格子の瞬間ごとの素の GROUP BY になりました。
#
# 結果はそれ自体のテーブルとして実体化します。執行のベンチマークや実効スプレッドの研究といった
# 下流のレシピは、統合をやり直さずに `nbbo_10s` を読むだけで済みます。

# %%
nbbo = (
    SNAPSHOT.group_by("ts")
    .agg(nbb=col("bid").max(), nbo=col("ask").min(), venues_quoting=col("bid").count())
    .filter(col("nbb").is_not_null(), col("nbo").is_not_null())  # the HAVING
    .sort("ts")
    .to_arrow()
)

NBBO_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("nbb", pa.float64()),
        pa.field("nbo", pa.float64()),
        pa.field("venues_quoting", pa.int64()),
    ]
)
db.create_table("nbbo_10s", NBBO_SCHEMA, time_column="ts")
db.append("nbbo_10s", nbbo.cast(NBBO_SCHEMA), note="10s-sampled NBBO from 3 venues")
db.table("nbbo_10s").limit(5).to_pandas()

# %% [markdown]
# ## 4. 最良を作るのは誰か、ロックやクロスはどれくらい起きるか
#
# 取引所別のスナップショットに対するウィンドウ関数が、シェアの問いに答えます。各瞬間で、
# その取引所のビッドは統合の最良と等しいか。同値なら最良にいる全取引所を数えます。
#
# 同じスナップショットが、NBB と NBO が等しい*ロック*状態と、NBB が NBO を上回る*クロス*状態も
# 拾います。取引所ごとに遅延も価格のずれも独立なので、短いロックとクロスは統合された世界の
# 現実であり、どの NBBO パイプラインもそれをどう扱うか決めなければなりません。

# %%
QUOTING = SNAPSHOT.filter(col("bid").is_not_null(), col("ask").is_not_null())

share = lambda flag: (when(flag).then(lit(1.0)).otherwise(lit(0.0)).mean() * 100).round(1)

(
    QUOTING.with_columns(
        nbb=col("bid").max().over(partition_by="ts"),
        nbo=col("ask").min().over(partition_by="ts"),
    )
    .group_by("venue")
    .agg(
        pct_at_best_bid=share(col("bid") >= col("nbb")),
        pct_at_best_ask=share(col("ask") <= col("nbo")),
        avg_own_spread=(col("ask") - col("bid")).mean().round(4),
    )
    .sort("venue")
    .to_pandas()
)

# %%
count_if = lambda flag: when(flag).then(lit(1)).otherwise(lit(0)).sum()

(
    db.table("nbbo_10s")
    .select(
        instants=count_star(),
        locked=count_if(col("nbb") == col("nbo")),
        crossed=count_if(col("nbb") > col("nbo")),
        pct_locked_or_crossed=(
            when(col("nbb") >= col("nbo")).then(lit(1.0)).otherwise(lit(0.0)).mean() * 100
        ).round(2),
    )
    .to_pandas()
)

# %% [markdown]
# ## 5. 統合の配当: 単一の取引所とのスプレッド比較
#
# 統合する意味がここにあります。最良のスプレッドはどの個別の板よりも狭くなります。最良の
# ビッドと最良のアスクは、たいてい*別々の*取引所にいるからです。
#
# UNION ALL 1つで板を横並びにでき、セッションのプロットでは NBBO のスプレッドが各取引所の
# スプレッドの下を這います。
#
# `UNION` の動詞はないので、ここが扉をくぐる場面です。すでに組んだ取引所別の半分は `.sql()` が
# 描き出し、文字列が union を足します。許容差を f-string で埋め込む必要も、すでにあるクエリを
# 手で書き直す必要もありません。

# %%
venue_spreads = QUOTING.group_by(col("venue").alias("book")).agg(
    avg_spread_usd=(col("ask") - col("bid")).mean().round(4)
)

db.sql(
    f"""
    {venue_spreads.sql()}
    UNION ALL
    SELECT 'NBBO', round(avg(nbo - nbb), 4) FROM nbbo_10s
    ORDER BY avg_spread_usd
    """
).to_pandas()

# %%
import matplotlib.pyplot as plt

MINUTE = time_bucket("1m", col("ts"))

venue_min = (
    QUOTING.group_by(MINUTE.alias("m"), "venue")
    .agg(spread=(col("ask") - col("bid")).mean())
    .sort("m")
    .to_pandas()
)
nbbo_min = (
    db.table("nbbo_10s")
    .group_by(MINUTE.alias("m"))
    .agg(spread=(col("nbo") - col("nbb")).mean())
    .sort("m")
    .to_pandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
for venue, g in venue_min.groupby("venue"):
    ax.plot(g["m"], g["spread"] * 100, lw=0.8, alpha=0.7, label=venue)
ax.plot(nbbo_min["m"], nbbo_min["spread"] * 100, lw=1.6, color="black", label="NBBO")
ax.set_title("Quoted spread through the session: each venue vs consolidated NBBO")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("spread (cents)")
ax.legend(ncols=4)
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - NBBO の構築は「取引所ごとの直近の気配、そして取引所をまたいで最良を取る」であり、それが
#   `asof_join(grid, venue_quotes, ..., 'venue')` と GROUP BY に1対1で対応します。取引所ごとの
#   ループも、手書きの状態機械も要りません。
# - ASOF の `tolerance` はそのまま古さの規則になります。気配を出さなくなった取引所は、死んだ
#   価格を最良に固定するかわりに統合から抜けます。
# - サンプリングした NBBO は近似です。正直な仕事は格子を明示し（ここでは10秒）、サンプルの
#   あいだのロックやクロスは見えていないことを忘れません。
# - `nbbo_10s` をテーブルとして実体化すると、統合はビルドの成果物になります。バージョン管理
#   され、ノートが付き、下流のあらゆる執行研究からすぐ引けます。

# %%
db.close()
