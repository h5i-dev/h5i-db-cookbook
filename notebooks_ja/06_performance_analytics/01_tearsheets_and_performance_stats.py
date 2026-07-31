# %% [markdown]
# # ティアシートとパフォーマンス統計
#
# ここまでのレシピはどれも、損益の数字と手描きのチャートで終わっていました。戦略が1つなら
# それで十分ですが、20あるとどうにもなりませんし、同じ実行について2人が別々のシャープレシオを
# 口にする原因もそこにあります。
#
# `h5i_db.quant` が共通の答えです。リターン系列はピンを持ったオブジェクトで、統計量は誰かの
# ノートブック上の pandas フレームではなくエンジンへのクエリで、計算は `empyrical` と一致します。
# `pyfolio` はその薄いラッパーでした。どちらのライブラリにも用意できなかったのがヘッダー部分
# です。どの数字も、それを計算したデータバージョンを持ち歩き、ピン留めされていない結果に
# `quant.verify` は保証を与えません。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | リターン系列 | 1期間につき1つの単純（非累積）リターン。すべての統計量の入力 |
# | 年率換算係数 | 1年が何期間か。日次バーなら252、月次なら12 |
# | シャープレシオ | 平均リターンをその標準偏差で割り、年率換算したもの |
# | ソルティノレシオ | 同じ考え方で、下振れのボラティリティだけを使うもの |
# | ドローダウン | エクイティカーブが直近の高値からどれだけ下にいるか |
# | カルマーレシオ | 年率リターンを最大ドローダウンで割ったもの |
# | アルファとベータ | ベンチマークで説明できないリターンと、ベンチマークへの感応度 |
# | ティアシート | 1ページにまとめた標準的なパフォーマンスレポート |
# | 来歴（プロビナンス） | その数字を生んだピン、パラメータ、クエリ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, col, quant, sql_expr, time_bucket
import cookbook_utils as cu

# %% [markdown]
# ## 1. 2つのリターン系列
#
# ベンチマークはユニバース全体の等ウェイト・ポートフォリオで、毎日リバランスします。戦略は
# レシピ 02/01 の12-1モメンタムで、毎月、上位3銘柄を等ウェイトで持ちます。
#
# どちらもエンジンで計算してテーブルに保存します。リターン系列はデータだからです。毎回
# ノートブックで計算し直すことが、同じ戦略のティアシートが食い違う原因になります。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_tearsheets_and_performance_stats"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="The price cut every number here reads")
print(f"{prices.num_rows:,} rows x {prices.num_columns} columns, "
      f"{daily.to_pandas()['symbol'].nunique()} names")
daily.to_pandas().head()

# %% [markdown]
# 銘柄ごとの日次リターンを取り、その日の平均を取ったものがベンチマークです。`lag` だけは
# ウィンドウ関数の逃げ道が要りますが、それ以外は動詞で書けます。

# %%
previous = sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")
returns_frame = (
    db.table("prices", snapshot="prices-v1")
    .with_columns(previous=previous)
    .with_columns(ret=col("adj_close") / col("previous") - 1)
    .filter(col("ret").is_not_null())
)
benchmark_table = (
    returns_frame.group_by("ts").agg(ret=col("ret").mean()).sort("ts").to_arrow()
)
db.create_table("benchmark_returns", benchmark_table.schema, time_column="ts")
db.append("benchmark_returns", benchmark_table, note="equal-weight universe")
print(f"{benchmark_table.num_rows:,} daily observations")
benchmark_table.to_pandas().tail(3)

# %% [markdown]
# 戦略の月次保有も同じクエリから出ます。等ウェイトのバスケットの日次リターンは、その日
# 持っていた銘柄のリターンの平均です。

# %%
monthly = (
    db.table("prices", snapshot="prices-v1")
    .with_columns(month=time_bucket("1mo", col("ts")))
    .group_by("symbol", "month")
    .agg(close=col("adj_close").last("ts"))
    .with_columns(
        lag_1=sql_expr("lag(close, 1)").over(partition_by="symbol", order_by="month"),
        lag_12=sql_expr("lag(close, 12)").over(partition_by="symbol", order_by="month"),
    )
    .with_columns(momentum=col("lag_1") / col("lag_12") - 1)
    .filter(col("momentum").is_not_null())
    .to_pandas()
)
picks = (
    monthly.sort_values("momentum", ascending=False)
    .groupby("month")
    .head(3)[["month", "symbol"]]
)
daily_returns = returns_frame.select(
    ts=col("ts"), symbol=col("symbol"), ret=col("ret")
).to_pandas()
daily_returns["month"] = (
    daily_returns["ts"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    .dt.tz_localize("UTC")
)
held = daily_returns.merge(
    picks.assign(month=lambda frame: frame["month"] + pd.offsets.MonthBegin(1)),
    on=["month", "symbol"],
)
strategy_table = pa.Table.from_pandas(
    held.groupby("ts", as_index=False)["ret"].mean().sort_values("ts"),
    preserve_index=False,
)
db.create_table("strategy_returns", strategy_table.schema, time_column="ts")
db.append("strategy_returns", strategy_table, note="12-1 momentum, top three, monthly")
db.snapshot("returns-v1", tables=["strategy_returns", "benchmark_returns"])
print(f"{strategy_table.num_rows:,} daily observations while invested")
strategy_table.to_pandas().tail(3)

# %% [markdown]
# ## 2. 系列を開く
#
# `quant.returns` は、時刻列とリターン列を持つテーブル（または遅延フレーム）を受け取り、
# `ReturnSeries` を返します。この時点では何も計算していません。系列はピン留めされたクエリで、
# 以下の統計量はそれぞれエンジンで走ります。
#
# `annualization` は1年の期間数です。定数は代表的な場合をカバーしますし、それ以外は単なる
# 数字です。時間足の暗号資産なら `24 * 365`、株式のザラ場の5分足なら `78` です。

# %%
strategy = quant.returns(db, "strategy_returns", snapshot="returns-v1",
                         annualization=quant.DAILY)
benchmark = quant.returns(db, "benchmark_returns", snapshot="returns-v1",
                          annualization=quant.DAILY)
print(strategy)
print(f"pinned: {strategy.provenance.pin.is_pinned}")
print(f"digest: {strategy.provenance.digest[:16]}")
print(f"\nconstants: DAILY={quant.DAILY} WEEKLY={quant.WEEKLY} "
      f"MONTHLY={quant.MONTHLY} YEARLY={quant.YEARLY}")

# %% [markdown]
# ## 3. 見出しの統計量
#
# 1行の SQL が、`pyfolio` が印字する `perf_stats` の表を丸ごと作ります。ベンチマークを渡すと、
# 2つの系列を時刻で結合してアルファとベータを足すので、重なっている日だけが寄与します。

# %%
stats = pd.DataFrame(
    {
        "strategy": strategy.stats(),
        "with benchmark": strategy.stats(benchmark=benchmark),
        "benchmark": benchmark.stats(),
    }
)
stats.round(4)

# %% [markdown]
# このうち3つは、単独よりも合わせて読む価値があります。`stability` は累積対数リターンを
# 時間に回帰した決定係数なので、シャープが高くて stability が低ければ、それは数週間で稼いだ
# 戦略です。`tail_ratio` はリターンの95パーセンタイルと5パーセンタイルの比なので、1を下回れば
# 損失側の裾のほうが厚いことになります。`daily_value_at_risk` は2シグマの日次損失で、悪い日
# ではなく、ふつうの日についての言明です。

# %%
alpha = strategy.stats(benchmark=benchmark)
print(f"annualized alpha over the universe  {alpha['alpha']:+.2%}")
print(f"beta to the universe                {alpha['beta']:.2f}")
print(f"stability                           {alpha['stability']:.2f}")
print(f"tail ratio                          {alpha['tail_ratio']:.2f}")

# %% [markdown]
# ## 4. ドローダウンを「出来事」として見る
#
# 最大ドローダウン1つの数字は、リスク委員会が必ず尋ねること、つまりどれだけ続いたかを隠します。
# `drawdown_table` は水面下の系列を、`pyfolio` と同じやり方で重ならない出来事に区切り、それぞれ
# の山、谷、回復を返します。

# %%
episodes = pd.DataFrame(strategy.drawdown_table(top=5))
episodes["net_drawdown"] = episodes["net_drawdown"].round(4)
episodes

# %% [markdown]
# 回復日のない出来事は、データが終わるまで系列が抜け出せなかったものです。その継続期間は
# ゼロではありません。不明です。

# %%
underwater = strategy.underwater().to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(underwater["ts"], underwater["drawdown"], 0, alpha=0.6, color="#e45756")
ax.set_title("Drawdown")
ax.set_xlabel("Date")
ax.set_ylabel("Below the running peak")
fig.tight_layout()

# %% [markdown]
# ## 5. ローリング統計
#
# ローリングの窓は、1つの数字では答えられない問いに答えます。最初から最後まで同じ戦略だったのか。
# `rolling_beta` はスライド窓の共分散集約を避け、エンジンの `ts_cov` を使います。DataFusion は
# 組み込みの共分散から値を取り除けないからです。

# %%
window = 126
rolling = (
    strategy.rolling_sharpe(window).to_pandas()
    .merge(strategy.rolling_volatility(window).to_pandas(), on="ts")
    .merge(strategy.rolling_beta(benchmark, window).to_pandas(), on="ts")
    .dropna()
)
print(f"{len(rolling):,} complete {window}-day windows")
rolling.tail(3).round(3)

# %%
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(rolling["ts"], rolling["rolling_sharpe"], linewidth=1.4)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title(f"Rolling {window}-day Sharpe")
axes[0].set_ylabel("Sharpe")
axes[1].plot(rolling["ts"], rolling["rolling_beta"], linewidth=1.4, color="#f58518")
axes[1].axhline(1, color="black", linewidth=0.8, linestyle="--")
axes[1].set_title(f"Rolling {window}-day beta to the universe")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Beta")
fig.tight_layout()

# %% [markdown]
# ## 6. ティアシート
#
# `quant.tearsheet` は標準のページを描きます。見出しの統計量、累積リターン、ドローダウン、
# ローリング・シャープ、ドローダウン表、そして来歴のヘッダーです。チャートは外部リクエストの
# ないインライン SVG です。*読む* ためにプロットのライブラリをインストールしなければならない
# レポートは、レポートではないからです。

# %%
html = quant.tearsheet(
    strategy,
    path="data/cache/momentum-tearsheet.html",
    benchmark=benchmark,
    title="12-1 momentum, top three",
    rolling_window=window,
    top_drawdowns=5,
)
print(f"wrote {len(html):,} bytes of self-contained HTML")
payload = quant.report_payload(strategy, benchmark=benchmark)
print(f"headline rows {len(payload['headline'])}, charts {len(payload['charts'])}")
print([chart["id"] for chart in payload["charts"]])

# %% [markdown]
# ## 7. バックテストの実行からティアシートへ
#
# 実行はフォークに `bt_equity` を書きます。これはリターン系列ではありません。*水準* の系列です。
# `quant.from_levels` がその橋渡しで、最初のバーをゼロリターンと呼ばずに落とします。どの
# カーブの先頭にも偽のフラットなバーを置くのは、複利で効いてくる嘘だからです。

# %%
frame = daily.to_pandas()
frame = frame[
    (frame["symbol"].isin(["AAPL", "MSFT", "JPM"]))
    & (frame["ts"] >= pd.Timestamp("2024-01-01", tz="UTC"))
]
market = cu.make_equity_market(pa.Table.from_pandas(frame, preserve_index=False))
run_db = h5i_db.Database(cu.fresh_db("06_tearsheet_run"), create=True)
for name in ("instruments", "book_deltas", "trades"):
    table = market[name]
    run_db.create_table(name, table.schema, time_column="ts_init")
    run_db.append(name, table)
run_db.snapshot("tape-v1", tables=["instruments", "book_deltas", "trades"])

book = market["book_deltas"].to_pandas()
book["session"] = book["ts_init"].dt.floor("s").dt.tz_localize("UTC")
first = book.groupby("instrument_id")["ts_init"].min()
signals = backtest.signal_table(
    [
        {
            "ts": stamp + dt.timedelta(microseconds=1),
            "instrument_id": symbol,
            "side": "buy",
            "quantity": 100.0,
            "tag": "buy-and-hold",
        }
        for symbol, stamp in first.items()
    ]
)
backtest.create_signal_table(run_db)
run_db.append("signals", signals)
report = backtest.run(
    run_db,
    "buy-and-hold",
    starting_cash=200_000.0,
    signals="signals",
    snapshot="tape-v1",
    fee_kind="proportional",
    fee_rate=0.0005,
    equity_interval_nanos=86_400_000_000_000,
)
fork = run_db.fork(report["fork"])
run_series = quant.from_levels(fork, "bt_equity", level="equity", annualization=quant.DAILY)
run_stats = run_series.stats()
print(f"{report['fills']} fills, {len(fork.read('bt_equity')):,} equity samples")
pd.Series(run_stats).to_frame("buy and hold").round(4)

# %% [markdown]
# ## 8. 引用できる数字
#
# `quant.verify` は計算を再実行し、両方を確認します。来歴のダイジェストと、計算し直した値です。
# ピン留めされていない系列は、合格せず *検証不能* として報告されます。「最新」を2回読んで
# 一致したことが証明するのは、その数秒のあいだ何も変わらなかったことだけだからです。

# %%
verified = quant.verify(strategy, rerun=lambda: quant.returns(
    db, "strategy_returns", snapshot="returns-v1", annualization=quant.DAILY))
print(f"verified {verified['verified']}, pinned {verified['pinned']}")

unpinned = quant.returns(db, "strategy_returns", annualization=quant.DAILY)
try:
    quant.verify(unpinned)
except quant.VerificationError as error:
    print(f"\nunpinned refused: {str(error)[:110]}...")
relaxed = quant.verify(unpinned, strict=False)
print(f"non-strict report: verified={relaxed['verified']} reason={relaxed['reason']!r}")

# %% [markdown]
# この拒否に意味を与えているのが来歴のヘッダーです。ピン、パラメータ、SQL を記録するので、
# 同じピンから作り直したティアシートは、似た数字ではありません。同じ数字を再現します。

# %%
provenance = strategy.provenance
print(f"kind        {provenance.kind}")
print(f"digest      {provenance.digest}")
print(f"pin         {provenance.pin}")
print(f"parameters  {provenance.parameters}")
print(f"warnings    {list(provenance.warnings()) or 'none'}")

# %% [markdown]
# ## まとめ
#
# - リターン系列はすべてのパフォーマンス統計の入力です。テーブルとして保存すれば、2つの
#   レポートがそれについて食い違うことはありません。
# - `ReturnSeries.stats()` は `empyrical` と一致し、ベンチマークを渡すと、重なる日についてだけ
#   アルファとベータが加わります。
# - ドローダウンは継続期間を持つ出来事であって、最悪の1つの数字ではありません。
# - `from_levels` は実行の `bt_equity` を同じオブジェクトに変えるので、バックテストとリサーチの
#   系列を同じコードで分析できます。
# - `quant.tearsheet` は来歴のヘッダー付きの自己完結したページを描きます。
# - `quant.verify` はピン留めされていない計算の検証を拒みます。その拒否こそが機能です。

# %%
fork.close()
run_db.close()
db.close()
