# %% [markdown]
# # ASOF ジョイン: 約定と気配、サイド判定とスプレッド
#
# すべての約定にその時点の気配を貼り付ける作業は、マイクロストラクチャの*本命*のジョイン
# です。約定のサイド判定、実効スプレッドと実現スプレッドの計測、TCA、トキシシティ分析は、
# どれもここから始まります。h5i-db の `asof_join` は、時刻順ストレージの上で働くネイティブの
# SQL テーブル関数です。pandas を往復する必要はありません。方向（`'backward'`／`'forward'`）と
# 陳腐化の許容差が組み込まれていて、他のテーブルと同じように CTE やウィンドウ関数と合成
# できます。
#
# 正解が分かっている2セッションぶんのテープを使って、全約定に気配を貼り（`pandas.merge_asof`
# と照合します）、Lee-Ready 流にサイドを判定してその精度を採点し、クオート／実効／実現の
# 各スプレッドを測り、最後に許容差を使って、古すぎる気配で黙って評価する代わりにきちんと
# 拒む形を作ります。

# %%
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_asof"), create=True)

# %% [markdown]
# ## 1. 正解が分かっているテープ
#
# サイド判定を*採点*するには、本当の攻撃側が分かっている約定が要ります。そこで印字される
# テープを気配ストリームから導きます。各約定はその場のビッドかアスクを取り（10%はミッドで
# 執行。分類にとっては難しいケースです）、0.2〜5ミリ秒の取引所報告遅延を挟み、本当の
# `side` を保持します。あわせて `marks` テーブルも保存します。各約定のタイムスタンプを
# 5分ずらしたもので、あとで実現スプレッドの参照に使います。

# %%
quotes = cu.make_quotes(
    symbols=["AAPL", "MSFT", "NVDA"], days=2, quotes_per_day=1_000, seed=11
)

qp = quotes.to_pandas()
rng = np.random.default_rng(23)
is_trade = rng.random(len(qp)) < 0.6
tr = qp.loc[is_trade, ["ts", "symbol", "bid", "ask"]].reset_index(drop=True)
n = len(tr)
buy = rng.random(n) < 0.5
at_mid = rng.random(n) < 0.10
px = np.where(buy, tr["ask"], tr["bid"])
px = np.where(at_mid, (tr["bid"] + tr["ask"]) / 2, px)
tr["price"] = px
tr["size"] = np.maximum(1, rng.lognormal(4.0, 1.2, n) // 100 * 100).astype("int64")
tr["side"] = np.where(buy, "B", "S")
tr["ts"] = (tr["ts"] + pd.to_timedelta(rng.uniform(0.2, 5.0, n), unit="ms")).dt.floor("us")
tr = tr.sort_values(["ts", "symbol"]).reset_index(drop=True)
tr["trade_id"] = np.arange(n, dtype="int64")

ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("trade_id", pa.int64()),
     pa.field("price", pa.float64()), pa.field("size", pa.int64()),
     pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(
    tr[["ts", "symbol", "trade_id", "price", "size", "side"]], preserve_index=False
).cast(trade_schema))

quote_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("bid", pa.float64()),
     pa.field("ask", pa.float64()), pa.field("bid_size", pa.int64()),
     pa.field("ask_size", pa.int64())]
)
db.create_table("quotes", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes.cast(quote_schema))

marks = tr[["ts", "symbol", "trade_id"]].copy()
marks["ts"] = marks["ts"] + pd.Timedelta(minutes=5)
marks = marks.sort_values(["ts", "symbol"])
mark_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("trade_id", pa.int64())]
)
db.create_table("marks", mark_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("marks", pa.Table.from_pandas(marks, preserve_index=False).cast(mark_schema))

print(f"{len(tr):,} trades, {len(qp):,} quotes over 2 sessions")

# %% [markdown]
# ## 2. ジョインそのもの
#
# `asof_join(left, right, left_ts, right_ts, by_key)` は、各約定に対して銘柄ごとに、その
# 時刻以前で最新の気配を返します。右側の列名が衝突すると `_right` が付きます。ここでは気配
# 自身のタイムスタンプが `ts_right` として届くので、約定時刻から `ts_right` を引けば気配の
# 古さになります。両方のテーブルが時刻順で保存されているので、ソートの工程を払う必要は
# ありません。

# %%
t0 = time.perf_counter()
joined = db.sql(
    """
    SELECT ts, symbol, trade_id, price, side, ts_right, bid, ask
    FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ORDER BY trade_id
    """
).to_pandas()
elapsed_ms = (time.perf_counter() - t0) * 1e3
print(f"{len(joined):,} trades matched against {len(qp):,} quotes "
      f"in {elapsed_ms:.1f} ms")
joined.head(5)

# %% [markdown]
# 信じてよい、ただし検証はする。`pandas.merge_asof` で同じ整列を行い、全約定について同一の
# 気配が付くことを確かめます。

# %%
left = tr[["ts", "symbol", "trade_id"]].sort_values("ts")
left["ts"] = left["ts"].astype("datetime64[us, UTC]")  # match arrow's us resolution
ref = pd.merge_asof(
    left, qp[["ts", "symbol", "bid", "ask"]],
    on="ts", by="symbol", direction="backward",
)
cmp = joined.merge(ref[["trade_id", "bid", "ask"]], on="trade_id", suffixes=("", "_pd"))
assert len(cmp) == len(tr)
assert np.allclose(cmp["bid"], cmp["bid_pd"]) and np.allclose(cmp["ask"], cmp["ask_pd"])
print(f"asof_join matches pandas.merge_asof on all {len(cmp):,} trades")

# %% [markdown]
# キーワード形式の書き方もあります。ASOF ジョインがもっと大きな文の一節になるときに便利です。
# 今のところ制約が2つあるので覚えておいてください。テーブル名は裸で書く必要があり（別名は
# 使えません）、出力列は修飾なしで参照します。

# %%
db.sql(
    """
    SELECT ts, symbol, price, bid, ask
    FROM trades ASOF JOIN quotes
    MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol
    LIMIT 3
    """
).to_pandas()

# %% [markdown]
# ## 3. Lee-Ready のサイド判定
#
# まずクオートルールです。ミッドより上なら買い、下なら売り。ちょうどミッドで印字された
# ものにはティックテストを当てます（ここでは `lag(price)` を使います。厳密な Lee-Ready は
# 直近の*異なる*価格を求めますが、このテープでは差はわずかです）。全体が1文で書けます。
# asof ジョインがウィンドウ関数に入り、それが `CASE` に入ります。

# %%
signed = db.sql(
    """
    WITH j AS (
        SELECT ts, symbol, trade_id, price, size, side, bid, ask,
               (bid + ask) / 2 AS mid
        FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ),
    t AS (
        SELECT *, lag(price) OVER (PARTITION BY symbol ORDER BY ts) AS prev_px
        FROM j
    )
    SELECT *,
           CASE WHEN price > mid THEN 'B'
                WHEN price < mid THEN 'S'
                WHEN price > prev_px THEN 'B'
                WHEN price < prev_px THEN 'S'
                ELSE 'U' END AS lr_side
    FROM t
    """
).to_pandas()

overall = (signed["lr_side"] == signed["side"]).mean()
quote_rule = signed[signed["price"] != signed["mid"]]
mid_prints = signed[signed["price"] == signed["mid"]]
print(f"overall accuracy      : {overall:.1%}")
print(f"quote-rule prints     : {(quote_rule['lr_side'] == quote_rule['side']).mean():.1%} "
      f"of {len(quote_rule):,}")
print(f"mid prints (tick test): {(mid_prints['lr_side'] == mid_prints['side']).mean():.1%} "
      f"of {len(mid_prints):,}")
pd.crosstab(signed["lr_side"], signed["side"], margins=True)

# %% [markdown]
# クオートルールはほぼ完璧で、唯一外すのは、約定の数ミリ秒の報告遅延の内側に気配更新が
# 入り込んだときだけです。ミッドでの印字はコイン投げに近いティックテストに落ちます。実際の
# TAQ データではクオートルールの適用割合はもっと低く、全体の精度は85〜90%あたりに着地します。
# 上の内訳を見れば、誤りがどこに住んでいるかがそのまま分かります。
#
# ## 4. クオート・実効・実現の各スプレッド
#
# - クオートスプレッド: その時点の気配の `ask - bid`
# - 実効スプレッド: `2·|price - mid|`。攻撃側が実際に払った額
# - 実現スプレッド: `2·d·(price - mid₊₅ₘ)`。5分後のミッドを使った、流動性供給側が実際に
#   残せた額
#
# 5分後のミッドは、`marks` テーブルを**forward** で asof ジョインして取ります。約定時刻＋5分
# 以降で最初の気配です。すべてが1文に収まり、単位はミッドに対するベーシスポイントです。

# %%
spreads = db.sql(
    """
    WITH j AS (
        SELECT symbol, trade_id, price, side, bid, ask, (bid + ask) / 2 AS mid
        FROM asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')
    ),
    m AS (
        SELECT trade_id, (bid + ask) / 2 AS mid_5m
        FROM asof_join('marks', 'quotes', 'ts', 'ts', 'symbol', 'forward')
    )
    SELECT j.symbol,
           avg((j.ask - j.bid) / j.mid) * 1e4                    AS quoted_bps,
           avg(2 * abs(j.price - j.mid) / j.mid) * 1e4           AS effective_bps,
           avg(2 * (CASE WHEN j.side = 'B' THEN 1 ELSE -1 END)
                 * (j.price - m.mid_5m) / j.mid) * 1e4           AS realized_bps,
           count(*)                                              AS n_trades
    FROM j JOIN m USING (trade_id)
    WHERE m.mid_5m IS NOT NULL
    GROUP BY 1 ORDER BY 1
    """
).to_pandas()
spreads

# %%
x = np.arange(len(spreads))
w = 0.27
fig, ax = plt.subplots(figsize=(8, 4))
for i, col in enumerate(("quoted_bps", "effective_bps", "realized_bps")):
    ax.bar(x + (i - 1) * w, spreads[col], width=w, label=col.replace("_bps", ""))
ax.set_xticks(x, spreads["symbol"])
ax.set_title("Spread measures by symbol (bps of mid)")
ax.set_xlabel("symbol")
ax.set_ylabel("basis points")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 実効はクオートより一段低くなります。印字の10%がミッドで執行されるからです。実現のほうは、
# 実効に対する*系統的な*差が出ていません。この合成フローは情報を運ばないので、約定後に
# ミッドが流動性供給側に不利な方へ平均的にドリフトすることがないためです（銘柄別の推定値が
# 実効の周りに散らばるのは、5分ミッドのボラティリティがスプレッドを圧倒するからで、この
# ばらつきは現実の推定量にも忠実です）。実際のテープでは、実現は実効より系統的に*下*に
# 座り、その差（2·d·(mid₊₅ₘ − mid)、つまりプライスインパクト）が逆選択コストになります。
# この分解が、それを測る標準的なやり方です。
#
# ## 5. 許容差: 古すぎる気配を拒む
#
# 既定では、asof ジョインはいくらでも過去へ手を伸ばします。穴の空いた気配フィード（配信
# 障害、間引かれたベンダーファイル）が相手だと、数分前の NBBO で約定を黙って評価してしまい
# ます。任意の許容差（h5i-db の生の時刻引数と同じくマイクロ秒）を渡せば、遡る範囲に上限が
# かかります。一致しなかった約定は気配の列が NULL のまま残るので、陳腐化が*見えて*数えられる
# ようになります。問題を再現するために、気配フィードを25%まで間引きます。

# %%
qs = qp[rng.random(len(qp)) < 0.25].sort_values(["ts", "symbol"])
db.create_table("quotes_sparse", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_sparse", pa.Table.from_pandas(qs, preserve_index=False).cast(quote_schema))

for label, tol in [("no tolerance", None), ("60s tolerance", 60_000_000),
                   ("5s tolerance", 5_000_000)]:
    args = "'ts', 'ts', 'symbol'" + (f", 'backward', {tol}" if tol else "")
    r = db.sql(
        f"SELECT count(*) AS n, count(bid) AS matched "
        f"FROM asof_join('trades', 'quotes_sparse', {args})"
    ).to_pandas()
    print(f"{label:>13}: {r['matched'][0]:,} of {r['n'][0]:,} trades matched "
          f"({r['matched'][0] / r['n'][0]:.1%})")

# %% [markdown]
# 許容差がなければ、どの約定にも*何かしらの*気配が付きます。どれだけ古かろうと関係なく。
# 新しさを要求すれば、その時点の気配が古すぎる約定は正直に NULL を返します。本番では、この
# NULL の割合は監視して警報を出す価値のあるデータ品質指標です。
#
# ## まとめ
#
# - `asof_join('trades', 'quotes', 'ts', 'ts', 'symbol')` だけで、約定と気配の整列は
#   完結します。キー別で、時刻的に正しく、整列済みストレージの上をストリーミングで流れ、
#   `pandas.merge_asof` と行単位で一致します。
# - 方向と許容差は一級の引数です。`'forward'` は実現スプレッド用に5分後のミッドを取って
#   きました。マイクロ秒の許容差は、古い気配での評価を目に見えて数えられる NULL に変えました。
#   右側の列名が衝突すると `_right` が付きます。
# - このジョインは SQL の残りと合成できます。Lee-Ready の判定も、クオート／実効／実現の
#   完全な分解も、それぞれ1文で書けます（CTE と `lag()` と `CASE`）。
# - 正解に対する採点には、約定と気配が1つの価格過程を共有するテープが要ります。合成データで
#   判定精度をベンチマークする前に、思い出す価値のある点です。

# %%
db.close()
