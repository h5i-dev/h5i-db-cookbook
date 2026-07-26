# %% [markdown]
# # クオンツのための SQL ツアー
#
# h5i-db のクエリ層は Apache DataFusion です。ジョインも CTE もウィンドウ関数も揃った
# 完全な SQL に、時刻順ストレージを活かす金融向けの演算子（`time_bucket`、
# `rolling_avg`、`ewma`、`vwap`、ASOF ジョイン）が足されています。このレシピはその
# 案内つきツアーです。50銘柄の日次パネルを題材に、1つの概念を1つの現実的な作業に当てて
# いき、最後は共有マシンを探索クエリで落とさないための安全装置（`timeout=`、`max_rows=`）
# で締めます。kdb/q や pandas を知っている人にとっては、対訳集として読めるはずです。

# %%
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_sql_tour"), create=True)

prices = cu.make_daily_prices(days=500)  # 50 symbols x 500 sessions
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices)

trades = cu.make_trades(days=1, trades_per_day=10_000)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades)

print(f"prices: {len(prices):,} rows   trades: {len(trades):,} rows")

# %% [markdown]
# あとでジョインするための静的な参照テーブルです。`time_column` は任意なので、セクター
# 対応表やシンボルマスタのような参照データには付けなくてかまいません。ただし付けない
# 以上、時刻ベースの仕掛け（プルーニング、ASOF、`time_bucket`）は一切効きません。

# %%
SECTORS = ["Tech", "Financials", "Energy", "Health Care", "Industrials"]
syms = sorted(set(prices["symbol"].to_pylist()))
db.create_table(
    "sectors",
    pa.schema([pa.field("symbol", pa.string()), pa.field("sector", pa.string())]),
)
db.append("sectors", pa.table({
    "symbol": pa.array(syms),
    "sector": pa.array([SECTORS[i % len(SECTORS)] for i in range(len(syms))]),
}))
db.sql("SELECT sector, count(*) AS names FROM sectors GROUP BY sector ORDER BY sector").to_pandas()

# %% [markdown]
# ## 1. 時間範囲のスキャン: マニフェストに仕事をさせる
#
# セグメントは `ts` で整列して保存され、マニフェストが各セグメントの時間範囲を記録して
# います。だから時刻の述語は、I/O が始まる前にセグメントを丸ごと読み飛ばします。身に
# つけたい癖が1つあります。大きなテーブルへの探索クエリは、**必ず**時刻フィルタから
# 始めることです。1週間だけ触るのか10年ぶんに触るのかの差になります。RFC3339 の文字列
# リテラルは、タイムスタンプ列とそのまま比較できます。

# %%
db.sql(
    """
    SELECT count(*) AS rows, min(ts) AS first_session, max(ts) AS last_session
    FROM prices
    WHERE ts >= '2024-01-01T00:00:00Z' AND ts < '2024-04-01T00:00:00Z'
    """
).to_pandas()

# %% [markdown]
# ## 2. 集約と GROUP BY: クロスセクションの要約

# %%
db.sql(
    """
    SELECT symbol,
           count(*)                              AS sessions,
           round(avg(close), 2)                  AS avg_close,
           round(sum(close * volume) / 1e9, 2)   AS dollar_vol_bn
    FROM prices
    GROUP BY symbol
    ORDER BY dollar_vol_bn DESC
    LIMIT 10
    """
).to_pandas()

# %% [markdown]
# ## 3. ウィンドウ関数: リターン、ランク、ローリングリスク
#
# 日次パネルの仕事の9割は、次の3つのウィンドウの型でまかなえます。
#
# - リターンには `lag()`。自己結合も pandas への往復も要りません。
# - 「銘柄ごとの直近N件」には `row_number()`（partition して降順に並べ、絞り込む）。
# - ローリングのモーメントには `ROWS BETWEEN` で枠を明示します。ここでは直近20日の
#   年率換算ボラティリティです。

# %%
db.sql(
    """
    SELECT ts, symbol, close,
           round(close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1, 5) AS ret_1d
    FROM prices
    WHERE symbol IN ('STK000', 'STK001')
    ORDER BY symbol, ts
    LIMIT 5
    """
).to_pandas()

# %%
db.sql(
    """
    SELECT ts, symbol, close, rn
    FROM (
        SELECT ts, symbol, close,
               row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
        FROM prices
    )
    WHERE rn <= 3 AND symbol IN ('STK000', 'STK001')
    ORDER BY symbol, rn
    """
).to_pandas()

# %%
db.sql(
    """
    WITH rets AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
    )
    SELECT ts, symbol,
           round(stddev(r) OVER (
               PARTITION BY symbol ORDER BY ts
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
           ) * sqrt(252) * 100, 2) AS vol20_ann_pct
    FROM rets
    WHERE symbol = 'STK000'
    ORDER BY ts DESC
    LIMIT 5
    """
).to_pandas()

# %% [markdown]
# ## 4. h5i の糖衣構文: `rolling_avg` と `ewma`
#
# `rolling_avg(x, ts, n)`（`rolling_sum/min/max` も同様）は、`ts` 順で直近 n 行の平均を
# 取る短縮形です。**ひとつ危ない角があります。** これは結果行を全体の時刻順になぞる
# だけで、銘柄で*分割されません*。複数銘柄のテーブルにかけると、AAPL と MSFT を平然と
# 平均します。先に1銘柄へ絞り込む（下の例）か、パネルなら
# `avg(...) OVER (PARTITION BY symbol ...)` と明示的に書いてください。
#
# `ewma(x, alpha)` のほうは本物のウィンドウ関数で、`PARTITION BY` を*ちゃんと*尊重します。
# RiskMetrics 流の平滑化が1行で書けます。

# %%
ma = db.sql(
    """
    SELECT ts, close,
           rolling_avg(close, ts, 20)                       AS ma20_sugar,
           avg(close) OVER (ORDER BY ts
               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)   AS ma20_standard,
           ewma(close, 0.06) OVER (ORDER BY ts)             AS ewma_px
    FROM prices
    WHERE symbol = 'STK007'
    ORDER BY ts
    """
).to_pandas()

import numpy as np

assert np.allclose(ma["ma20_sugar"], ma["ma20_standard"])
print("rolling_avg == explicit OVER frame, verified on", len(ma), "rows")
ma.tail(3)

# %% [markdown]
# ## 5. CTE とジョイン: リターン → ボラティリティ → セクターのパイプライン
#
# 多段のリサーチ SQL を読める状態に保つのが CTE です。各段が名前の付いた検証可能な中間
# 結果になり、パイプライン全体は1つのプランとして走ります（一時テーブルの実体化は
# ありません）。ここでは、日次リターン → 直近20日ボラティリティ → セクター対応表との
# ジョイン、と進んで、現在のセクター別リスク像に行き着きます。

# %%
sector_vol = db.sql(
    """
    WITH rets AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
    ),
    vol AS (
        SELECT ts, symbol,
               stddev(r) OVER (PARTITION BY symbol ORDER BY ts
                               ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * sqrt(252) AS v
        FROM rets
    ),
    latest AS (
        SELECT symbol, v
        FROM (SELECT symbol, v, row_number() OVER (PARTITION BY symbol ORDER BY ts DESC) rn FROM vol)
        WHERE rn = 1
    )
    SELECT s.sector,
           count(*)                    AS names,
           round(avg(l.v) * 100, 2)    AS avg_vol20_pct,
           round(max(l.v) * 100, 2)    AS max_vol20_pct
    FROM latest l
    JOIN sectors s USING (symbol)
    GROUP BY s.sector
    ORDER BY avg_vol20_pct DESC
    """
).to_pandas()
sector_vol

# %%
import matplotlib.pyplot as plt

vol_series = db.sql(
    """
    WITH rets AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
        WHERE symbol IN ('STK000', 'STK013', 'STK026')
    )
    SELECT ts, symbol,
           stddev(r) OVER (PARTITION BY symbol ORDER BY ts
                           ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) * sqrt(252) * 100 AS vol
    FROM rets ORDER BY ts
    """
).to_pandas()

fig, ax = plt.subplots(figsize=(10, 4))
for sym, g in vol_series.groupby("symbol"):
    ax.plot(g["ts"], g["vol"], lw=0.9, label=sym)
ax.set_title("Trailing 20-day annualized volatility (SQL window frame)")
ax.set_xlabel("date")
ax.set_ylabel("vol (% ann.)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. 日付まわりの道具: `time_bucket`、`date_trunc`、`EXTRACT`
#
# 主力は `time_bucket` です。ここでは GROUP BY の中で `last_value(... ORDER BY ts)` を
# 使って月末終値を出し（「終値」を取る定番の書き方で、自己結合は不要です）、`lag` に
# つないで月次リターンにします。`date_trunc` は同じ切り捨てを、幅やタイムゾーンの追加
# 指定なしで行います。`EXTRACT` は季節性の切り口に使うカレンダー要素を取り出します。

# %%
db.sql(
    """
    WITH monthly AS (
        SELECT time_bucket('1mo', ts) AS month, symbol,
               last_value(close ORDER BY ts) AS close_m
        FROM prices GROUP BY month, symbol
    )
    SELECT month, symbol,
           round(close_m / lag(close_m) OVER (PARTITION BY symbol ORDER BY month) - 1, 4) AS ret_1mo
    FROM monthly
    WHERE symbol = 'STK000'
    ORDER BY month DESC
    LIMIT 6
    """
).to_pandas()

# %%
# Day-of-week effect scan (EXTRACT dow: 0 = Sunday). A daily panel only has
# Mon-Fri; means this close to zero are exactly what an honest scan shows.
db.sql(
    """
    WITH rets AS (
        SELECT ts, close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
    )
    SELECT EXTRACT(dow FROM ts) AS dow,
           count(*) AS obs,
           round(avg(r) * 1e4, 2) AS mean_ret_bps
    FROM rets GROUP BY dow ORDER BY dow
    """
).to_pandas()

# %%
# Intraday cut on the tick table: hourly volume shows the session U-shape.
db.sql(
    """
    SELECT EXTRACT(hour FROM ts) AS hour_utc,
           count(*)              AS trades,
           sum(size)             AS shares
    FROM trades GROUP BY hour_utc ORDER BY hour_utc
    """
).to_pandas()

# %% [markdown]
# ## 7. 分布と依存関係: `approx_percentile_cont` と `corr`
#
# 50万行を pandas に引き出さずにリターンの分位点を出し、自己結合したリターンの CTE の上で
# 普通の集約として相関を取ります。

# %%
db.sql(
    """
    WITH rets AS (
        SELECT close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
    )
    SELECT round(approx_percentile_cont(r, 0.01) * 100, 3) AS p01_pct,
           round(approx_percentile_cont(r, 0.05) * 100, 3) AS p05_pct,
           round(approx_percentile_cont(r, 0.50) * 100, 3) AS p50_pct,
           round(approx_percentile_cont(r, 0.95) * 100, 3) AS p95_pct,
           round(approx_percentile_cont(r, 0.99) * 100, 3) AS p99_pct
    FROM rets
    """
).to_pandas()

# %%
db.sql(
    """
    WITH rets AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS r
        FROM prices
    )
    SELECT a.symbol AS sym_a, b.symbol AS sym_b,
           round(corr(a.r, b.r), 3) AS rho
    FROM rets a
    JOIN rets b USING (ts)
    WHERE (a.symbol, b.symbol) IN (('STK000','STK001'), ('STK000','STK025'), ('STK010','STK040'))
    GROUP BY a.symbol, b.symbol
    ORDER BY rho DESC
    """
).to_pandas()

# %% [markdown]
# ## 8. 資源の安全装置: 遅く落ちるより速く落ちる
#
# 共有のリサーチマシンで危ないのは、間違ったクエリではなく、うっかり巨大になったクエリの
# ほうです。`db.sql` はどの呼び出しでも `timeout=`（秒）と `max_rows=` を取り、どちらかを
# 超えると型の付いたエラー（`TimeoutError`、`LimitError`）が上がります。ツール側で分岐
# できる `.code` も付いています。共有ノートブックでは控えめな既定値を置いておき、必要な
# ときに意識して上げましょう。

# %%
try:
    db.sql("SELECT * FROM prices", max_rows=1_000)  # 25,000-row table
except h5i_db.LimitError as e:
    print(f"LimitError  code={e.code}")
    print(f"message     {e}")

# %%
try:
    # 25k x 25k row cross join: a typo away from a real query. The deadline
    # cancels it instead of letting it own the box.
    db.sql(
        "SELECT a.symbol, corr(a.close, b.close) FROM prices a CROSS JOIN prices b GROUP BY a.symbol",
        timeout=1,
    )
except h5i_db.TimeoutError as e:
    print(f"TimeoutError  code={e.code}")
    print(f"hint          {e.hint}")

# %% [markdown]
# ## まとめ
#
# - どのクエリも時刻の述語から始めます。整列済みセグメントとマニフェストの時間範囲が
#   あるので、そのフィルタは行だけでなく I/O を削ります。
# - `lag`、`row_number`、`ROWS BETWEEN` の枠があれば、リターン・直近N件・ローリング
#   リスクは SQL の中で片付きます。GROUP BY の中の `last_value(x ORDER BY ts)` は終値を
#   取る定番です。
# - 糖衣構文の `rolling_avg` は全体の時刻順で直近N行を見るだけなので、先に1銘柄へ絞ります。
#   `ewma` は本物のウィンドウ関数で `PARTITION BY` を尊重します。
# - CTE のパイプライン（リターン → ボラティリティ → セクタージョイン）は1つのプランとして
#   走ります。静的な参照テーブルなら `time_column` は省いてかまいません。
# - `timeout=` と `max_rows=` は、暴走したクエリを型の付いた捕捉可能なエラーに変えます。
#   共有データベースでの作法です。

# %%
db.close()
