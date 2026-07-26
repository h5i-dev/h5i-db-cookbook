# %% [markdown]
# # NBBO の統合: 分断された取引所をまたぐ最良気配
#
# 米国株は十数以上の取引所で売買され、それぞれが自分の板の最良気配を出しています。執行品質を
# 測る基準になる統合最良気配（NBBO）は、*導出*しなければ手に入りません。任意の瞬間について、
# 各取引所の直近の気配を取り、横断的にいちばん良いものを選ぶ。この「時刻 t における取引所ごとの
# 直近気配」がまさに ASOF ジョインで、h5i-db なら数本の SQL でリサーチに使えるサンプリング
# NBBO を組み立てられます。時間格子を取引所別の気配に ASOF ジョインし、瞬間ごとに `max(bid)` と
# `min(ask)` を取るだけです。ついでに、内側の値を実際に作っているのはどの取引所か、市場が
# ロックやクロスをどれくらい起こすか、統合スプレッドが単独の取引所よりどれだけ狭いかも測ります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_nbbo"), create=True)

# %% [markdown]
# ## 1. 分断されたテープを合成する
#
# AAPL の統合形式の気配を1日ぶん用意し、マイクロストラクチャの性格が異なる3つの取引所へ
# 展開します。各取引所は統合側の更新に合わせて出し直しますが、ときどき*取りこぼし*（フィードの
# 穴。本物の陳腐化の源です）、自分のゲートウェイ遅延を加え、価格を基準から1ティックずらします
# （たまに内側に入ることもあります）。この組み合わせがあるから、内側での競争力に取引所ごとの差が
# 生まれ、短いロック／クロス状態も現れます。シードは全体を通して固定です。

# %%
quotes = cu.make_quotes(
    symbols=["AAPL"], days=1, quotes_per_day=2_600, seed=11, base_prices={"AAPL": 320.0}
).to_pandas()
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
# イベント精度の NBBO は*すべての*気配更新で再評価します。リサーチでは規則的な格子上の
# サンプリング NBBO が標準的な近似です（誤差はサンプリング間隔で抑えられます。ここでは統合側の
# 更新が約9秒ごとに届くので、10秒ごとに標本を取ります）。格子自体も小さな h5i-db テーブルで、
# （タイムスタンプ, 取引所）の組ごとに1行あります。だから
# `asof_join(grid, venue_quotes, ..., 'venue')` 1回で、各瞬間における各取引所のその場の気配が
# 撮れます。60秒の `tolerance`（生のマイクロ秒）は、死んだ板が内側に居座り続けるのを許さず、
# その取引所の気配を陳腐化（NULL）と宣言します。
#
# ASOF のパイプラインで習慣にする価値があるものが1つあります。ジョインが格子の行1つにつき
# ちょうど1行を返したことを表明することです（LEFT ASOF ジョインは左側と 1:1 です）。サイズや
# キーの誤りが、板を歪める代わりに大きな声で失敗するようになります。

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
db.sql(
    f"""
    SELECT ts, venue, bid, ask, ts_right AS quoted_at
    FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
    ORDER BY ts, venue
    LIMIT 6
    """
).to_pandas()

# %%
# Trust, but verify: one output row per grid row, and the SQL ASOF must agree
# with an independent pandas merge_asof on a venue's book.
state = db.sql(
    f"""
    SELECT ts, venue, bid, ask
    FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
    ORDER BY ts, venue
    """
).to_pandas()
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
# 取引所をまたいで良いほうを取る作業は、格子の瞬間ごとの単なる GROUP BY になりました。結果は
# それ自体のテーブルとして実体化します。下流のレシピ（執行ベンチマーク、実効スプレッドの研究）は
# 統合を走らせ直さずに `nbbo_10s` を読めます。

# %%
nbbo = db.sql(
    f"""
    SELECT ts,
           max(bid)   AS nbb,
           min(ask)   AS nbo,
           count(bid) AS venues_quoting
    FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
    GROUP BY ts
    HAVING max(bid) IS NOT NULL AND min(ask) IS NOT NULL
    ORDER BY ts
    """
).to_arrow()

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
db.sql("SELECT * FROM nbbo_10s LIMIT 5").to_pandas()

# %% [markdown]
# ## 4. 内側を作っているのは誰か、ロックやクロスはどれくらい起きるか
#
# 取引所別スナップショットへのウィンドウ関数が、シェアの問いに答えます。各瞬間について、この
# 取引所のビッドは統合ベストと等しいか（同着なら内側にいる全取引所を数えます）。同じスナップ
# ショットが*ロック*（NBB = NBO）と*クロス*（NBB > NBO）の状態にも印を付けます。取引所ごとに
# 遅延も価格のずれも独立している以上、短いロックやクロスは統合された世界の日常であり、どんな
# NBBO パイプラインもそれをどう扱うか決めなければなりません。

# %%
db.sql(
    f"""
    WITH state AS (
        SELECT ts, venue, bid, ask
        FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
        WHERE bid IS NOT NULL AND ask IS NOT NULL
    ),
    ranked AS (
        SELECT venue, bid, ask,
               max(bid) OVER (PARTITION BY ts) AS nbb,
               min(ask) OVER (PARTITION BY ts) AS nbo
        FROM state
    )
    SELECT venue,
           round(avg(CASE WHEN bid >= nbb THEN 1.0 ELSE 0.0 END) * 100, 1) AS pct_at_best_bid,
           round(avg(CASE WHEN ask <= nbo THEN 1.0 ELSE 0.0 END) * 100, 1) AS pct_at_best_ask,
           round(avg(ask - bid), 4)                                        AS avg_own_spread
    FROM ranked
    GROUP BY venue
    ORDER BY venue
    """
).to_pandas()

# %%
db.sql(
    """
    SELECT count(*)                                        AS instants,
           sum(CASE WHEN nbb = nbo THEN 1 ELSE 0 END)      AS locked,
           sum(CASE WHEN nbb > nbo THEN 1 ELSE 0 END)      AS crossed,
           round(avg(CASE WHEN nbb >= nbo THEN 1.0 ELSE 0.0 END) * 100, 2) AS pct_locked_or_crossed
    FROM nbbo_10s
    """
).to_pandas()

# %% [markdown]
# ## 5. 統合の配当: 単独の取引所とのスプレッド比較
#
# 統合する意味はここにあります。内側のスプレッドはどの単独の板よりも狭くなります。最良ビッドと
# 最良アスクは、たいてい*別々の*取引所にいるからです。UNION ALL 1つで板を並べれば、セッションの
# プロットで NBBO スプレッドが各取引所のスプレッドの下に貼り付いているのが見えます。

# %%
db.sql(
    f"""
    SELECT venue AS book, round(avg(ask - bid), 4) AS avg_spread_usd
    FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
    WHERE bid IS NOT NULL AND ask IS NOT NULL
    GROUP BY venue
    UNION ALL
    SELECT 'NBBO', round(avg(nbo - nbb), 4) FROM nbbo_10s
    ORDER BY avg_spread_usd
    """
).to_pandas()

# %%
import matplotlib.pyplot as plt

venue_min = db.sql(
    f"""
    SELECT time_bucket('1m', ts) AS m, venue, avg(ask - bid) AS spread
    FROM asof_join('nbbo_grid', 'venue_quotes', 'ts', 'ts', 'venue', 'backward', {STALE_US})
    WHERE bid IS NOT NULL AND ask IS NOT NULL
    GROUP BY m, venue ORDER BY m
    """
).to_pandas()
nbbo_min = db.sql(
    """
    SELECT time_bucket('1m', ts) AS m, avg(nbo - nbb) AS spread
    FROM nbbo_10s GROUP BY m ORDER BY m
    """
).to_pandas()

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
# - NBBO の構築は「取引所ごとの直近気配を取り、取引所をまたいで良いほうを選ぶ」だけで、これは
#   `asof_join(grid, venue_quotes, ..., 'venue')` と GROUP BY に 1:1 で対応します。取引所ごとの
#   ループも、手書きの状態機械も要りません。
# - ASOF の `tolerance` はそのまま陳腐化のルールになります。気配を出さなくなった取引所は、死んだ
#   価格を内側に貼り付ける代わりに統合から抜けます。
# - サンプリング NBBO は近似です。誠実な仕事は格子（ここでは10秒）を明記し、標本の間に起きた
#   ロックやクロスが見えないことを忘れません。
# - `nbbo_10s` をテーブルとして実体化すれば、統合はビルド成果物になります。バージョンが付き、
#   注記が付き、下流の執行研究からすぐクエリできます。

# %%
db.close()
