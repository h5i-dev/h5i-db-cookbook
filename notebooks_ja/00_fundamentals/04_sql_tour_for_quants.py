# %% [markdown]
# # クオンツのための SQL ツアー
#
# h5i-db のクエリ層は Apache DataFusion です。ジョインも CTE もウィンドウ関数も揃った、
# 一通りの SQL が使えます。その上に、時刻順ストレージを活かす金融向けの演算子
# （`time_bucket`、`rolling_avg`、`ewma`、`vwap`、ASOF ジョイン）が載っています。
#
# このレシピはガイド付きのツアーです。各停留所で概念を1つ取り上げ、50銘柄の日足パネルに
# 対する現実的な作業に当てはめます。kdb/q や pandas を知っているなら、対訳集として読んで
# ください。
#
# ここでは意図的にすべて SQL の文字列で書きます。SQL が主題だからです。クックブックの他の
# 部分は遅延 DataFrame ビルダ（`db.table(...)` と動詞、レシピ09）を好みますが、それも
# まさにここに出てくる文にコンパイルされます。組み立てたクエリに `.sql()` を呼べば、生成
# された SQL がそのまま見えます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語            | 意味 |
# | ------------- | --- |
# | パネル           | 1行が1資産1日付になるデータセット |
# | クロスセクション      | 同じ時点で資産どうしを比べること。自分の過去と比べる時系列の対義 |
# | ウィンドウ関数       | 行をまとめずに近傍の行にまたがって計算する SQL。`lag()` など |
# | CTE           | `WITH` で導入する名前付き副問い合わせ。多段のクエリが読みやすくなる |
# | `time_bucket` | タイムスタンプをグリッドに丸める。タイムゾーン名も渡せる |
# | VWAP          | 出来高加重平均価格。市場全体が得た平均価格 |
# | EWMA          | 指数加重移動平均。直近の観測値ほど重く効く |
# | ASOF 結合       | 左の各行を、そのタイムスタンプ以前で最も新しい右の行に結合する |
# | 分位（quantile）  | 行を等分のバケットに割り振ること。シグナルを評価する定番の方法 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_sql_tour"), create=True)

# %% [markdown]
# ## データ
#
# 主役のテーブルは `cu.make_daily_prices` の日足 OHLCV パネルです。合成の50銘柄×500セッ
# ションで、1行が1銘柄1セッションぶん。リターンには共通のマーケットファクターと固有の
# ノイズが載っているので、クロスセクションのクエリにも見つけるものがあります。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、20:00 UTC |
# | `symbol` | `string` | 銘柄コード、`STK000` 〜 `STK049` |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `volume` | `int64` | 出来高（株数） |

# %%
prices = cu.make_daily_prices(days=500)  # 50 symbols x 500 sessions
print(f"prices: {prices.num_rows:,} rows x {prices.num_columns} columns")
prices.to_pandas().head()

# %% [markdown]
# 日中の例のために、1セッションぶんのティックデータも連れてきます。列は `ts`、`symbol`、
# `price`、`size`、`exchange`、`side` で、1行が1約定です。

# %%
trades = cu.make_trades(days=1, trades_per_day=10_000)
print(f"trades: {trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %%
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices)

db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades)

db.tables()

# %% [markdown]
# あとで使うジョインのために、静的な参照テーブルも要ります。`time_column` は任意です。
# セクター表や銘柄マスタのような参照データには必要ありません。ただしその場合、時刻まわりの
# 仕掛けも一切付いてきません。プルーニングも ASOF も `time_bucket` も効きません。

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
# セグメントは `ts` 順で保存され、マニフェストには各セグメントの時刻範囲が入っています。
# だから時刻の述語は、I/O が始まる前にセグメントを丸ごと飛ばします。
#
# 身につけるべき習慣はこうです。大きなテーブルに対する探索的なクエリは、**必ず**時刻の
# フィルタから始める。1週間ぶんを触るか10年ぶんを触るかの差になります。RFC3339 の文字列
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
# ## 3. ウィンドウ関数: リターン、順位、ローリングのリスク
#
# 日足パネルの仕事は、だいたい3つのウィンドウのパターンで足ります。
#
# - リターンには `lag()`。自己結合も pandas への往復も要りません
# - 「銘柄ごとの直近N件」には `row_number()`。パーティションを切り、降順に並べ、絞ります
# - ローリングのモーメントには `ROWS BETWEEN` で明示的にフレームを書きます。ここでは
#   過去20日の年率ボラティリティです

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
# ## 4. h5i の糖衣: `rolling_avg` と `ewma`
#
# `rolling_avg(x, ts, n)` は、`ts` 順で過去 n 行の平均を取る書き方の短縮形です。
# `rolling_sum`、`rolling_min`、`rolling_max` も同じ形をしています。
#
# **1つだけ鋭い角があります。** これは結果行を全体の時刻順になめるだけで、銘柄では
# *パーティションされません*。複数銘柄のテーブルにかけると、AAPL と MSFT を平気で混ぜて
# 平均します。下のように先に1銘柄へ絞るか、パネルなら明示的な
# `avg(...) OVER (PARTITION BY symbol ...)` の形を使ってください。
#
# `ewma(x, alpha)` のほうは正真正銘のウィンドウ関数で、`PARTITION BY` を尊重します。
# RiskMetrics 式の平滑化が1行で書けます。

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
# ## 5. CTE とジョイン: リターンからボラティリティ、そしてセクターへ
#
# CTE は多段のリサーチ SQL を読めるものにします。各段が名前の付いた、検証できる中間結果に
# なり、それでいてパイプライン全体は1つのプランとして走ります。実体化された一時テーブルは
# できません。
#
# ここでの4段はこうです。日次リターン、過去20日のボラティリティ、銘柄ごとの最新の1行、
# そしてセクター表とのジョイン。最後に現在のリスクの見取り図が出ます。

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
# 主力は `time_bucket` です。下では GROUP BY の中で `last_value(... ORDER BY ts)` と組んで
# 月次の終値を作っています。これが「終値」を取る定型で、自己結合は要りません。その終値は
# `lag` でそのまま月次リターンにつながります。
#
# `date_trunc` は同じ切り下げを、幅やタイムゾーンの追加オプションなしでやります。`EXTRACT`
# はカレンダーの成分を取り出すので、季節性の切り口に使えます。

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
# 50万行を pandas に引き上げずにリターンの分位点を出します。ペアの相関は、自己結合した
# リターンの CTE に対する普通の集約として出てきます。

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
# ## 8. リソースのガード: 遅く失敗せず、速く失敗する
#
# 共有のリサーチマシンで危ないのは、間違ったクエリではありません。うっかり巨大になった
# クエリです。
#
# `db.sql` の呼び出しはどれも `timeout=`（秒）と `max_rows=` を取ります。どちらかを超えると
# 型付きのエラー、`TimeoutError` か `LimitError` が上がり、ツール側で分岐できる `.code` が
# 付いてきます。共有のノートブックでは控えめな既定値を置き、上げるときは意識して上げて
# ください。

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
# - クエリは必ず時刻の述語から始めます。整列済みセグメントとマニフェストの時刻範囲のおかげ
#   で、そのフィルタは行だけでなく I/O を削ります。
# - `lag`、`row_number`、`ROWS BETWEEN` のフレームで、リターンも直近N件もローリングのリスク
#   も SQL の中で片付きます。GROUP BY の中の `last_value(x ORDER BY ts)` が終値の定型です。
# - `rolling_avg` の糖衣は全体の時刻順で過去N行を取るので、先に1銘柄へ絞ってください。
#   `ewma` は本物のウィンドウ関数で、`PARTITION BY` を尊重します。
# - CTE のパイプラインは1つのプランとして走ります。静的な参照テーブルは `time_column` を
#   丸ごと省けます。
# - `timeout=` と `max_rows=` は、暴走するクエリを型付きで捕まえられるエラーに変えます。
#   共有データベースでの作法です。

# %%
db.close()
