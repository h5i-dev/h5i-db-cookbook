# %% [markdown]
# # スイープ、検証、そして修正が自分のアルファに何をしたか
#
# リサーチの終わりには、3つの問いが順番にやってきます。パラメータのグリッドは何と言ったか。
# その数字を他人が再現できるか。そして来月ベンダーがデータを改訂したら、数字は変わるのか。
#
# 1つ目はふつうの作業です。2つ目と3つ目は、バージョン管理されたストアがただの保存の話でなく
# なる場所です。`quant.verify` はピン留めされていない計算に保証を与えませんし、
# `quant.restatement_impact` は同じ計算を2つの読み取り点で走らせて「改訂は答えを動かしたか」に
# 答えます。Parquet ファイルの並んだディレクトリには問えない問いです。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | スイープ | 1つの計算をパラメータのグリッド上で走らせること |
# | フォーク | 試行が書き込む、コピーオンライトのデータベースの枝 |
# | 来歴（プロビナンス） | その数字を生んだピン、パラメータ、SQL |
# | ダイジェスト | 来歴のハッシュ。2つの結果を厳密に比べられる |
# | 修正（リステートメント） | すでに使ったデータをベンダーが訂正すること |
# | as-of 読み取り | 過去のある時点の状態としてデータベースを読むこと |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import json

import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

# %% [markdown]
# ## 1. データと、そのバージョン
#
# 30銘柄の日次価格です。大事なのはバージョン履歴で、どのコミットもあとから読めます。この
# レシピの最後のセクションが成り立つのはそのためです。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_sweeps_verification_restatements"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="vendor load, first delivery")
db.snapshot("prices-v1", tables=["prices"], note="What the research below read")
first_version = db.versions("prices")[-1]["sequence"]
print(f"prices is at version {first_version}")
print(f"{prices.num_rows:,} rows")
daily.to_pandas().head(3)


# %% [markdown]
# ## 2. スイープは試行ごとに1つのフォーク
#
# `quant.sweep` はパラメータの組み合わせごとに関数を1回呼び、その試行専用のフォークで開いた
# データベースを渡します。関数が書いたものはフォークの中だけに入るので、試行どうしも、元データも
# 汚染されませんし、失敗した試行が後片付けを残すこともありません。
#
# 関数はメトリクスを返し、スイープはそれを、生んだパラメータと並べてフォークに書き込みます。

# %%
def score(fork, params) -> dict:
    """Build a factor panel inside this trial's fork and score it."""
    pinned = fork.table("prices", snapshot="prices-v1")
    factor = (
        pinned.with_columns(
            past=sql_expr(f"lag(adj_close, {params['lookback']})").over(
                partition_by="symbol", order_by="ts"
            )
        )
        .with_columns(value=col("adj_close") / col("past") - 1)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("value"))
    )
    price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
    panel = quant.build_panel(
        fork,
        factor,
        price_frame,
        periods=(params["horizon"],),
        quantiles=params["quantiles"],
    )
    decay = panel.ic_decay().to_pandas().iloc[0]
    spread = panel.spread().to_pandas()[f"spread_{params['horizon']}"]
    return {
        "mean_ic": float(decay["mean_ic"]),
        "ic_t_stat": float(decay["t_stat"]),
        "spread_sharpe": float(spread.mean() / spread.std() * np.sqrt(252)),
        "observations": int(decay["n"]),
    }


swept = quant.sweep(
    db,
    {"lookback": [63, 126, 252], "quantiles": [3, 5], "horizon": [21]},
    score,
    prefix="momentum",
    note="momentum lookback and bucket count",
)
print(swept)
compared = swept.to_pandas()
board = pd.concat(
    [
        pd.json_normalize(compared["_params"].map(json.loads)),
        compared[["mean_ic", "ic_t_stat", "spread_sharpe", "observations"]],
    ],
    axis=1,
).sort_values("spread_sharpe", ascending=False)
board.round(4)

# %% [markdown]
# `compare()` は全フォークを一度に読む1つのクエリです。フォークはベースのセグメントを共有する
# ので、100試行の比較にかかるのは、1つ読むぶんに各試行が実際に書いたぶんを足した程度です。
# パラメータは `_params` の JSON 文字列として運ばれるので、メトリクスの列がどうであれ試行の
# 同一性は保たれます。

# %%
best = swept.best("spread_sharpe")
print(f"forks created  {len(swept.forks)}")
print(f"best trial     lookback={best['lookback']:.0f} quantiles={best['quantiles']:.0f}")
print(f"               spread Sharpe {best['spread_sharpe']:.3f}, IC t-stat {best['ic_t_stat']:.2f}")
print(f"comparison table columns: {swept.compare().to_arrow().column_names}")

# %% [markdown]
# 最初の失敗で止まるスイープは、すでに走った試行を無駄にします。`keep_going` は失敗を記録して
# 続けます。夜間バッチで欲しい挙動であり、失敗がデータの誤りを意味するときには持ってはいけない
# 挙動です。

# %%
def brittle(fork, params) -> dict:
    if params["lookback"] > 1_000:
        raise ValueError("no history that long")
    return {"lookback": float(params["lookback"])}


partial = quant.sweep(
    db, {"lookback": [63, 5_000]}, brittle, prefix="brittle", keep_going=True
)
print(f"{len(partial)} trials succeeded, {len(partial.failures)} failed")
print(partial.failures)

# %% [markdown]
# ## 3. 検証はチェックマークではありません。拒否です
#
# `quant.verify` は計算を再実行して2つを確認します。来歴のダイジェストが変わっていないことと、
# 計算し直した数字が一致することです。ピン留めされていない計算は、合格せず検証不能として
# 報告されます。「最新」に対する2回の実行が一致しても、そのあいだ何も変わらなかったこと以上は
# 証明しないからです。

# %%
def build_panel(pin: dict):
    pinned = db.table("prices", **pin)
    factor = (
        pinned.with_columns(
            past=sql_expr("lag(adj_close, 126)").over(partition_by="symbol", order_by="ts")
        )
        .with_columns(value=col("adj_close") / col("past") - 1)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("value"))
    )
    price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
    return quant.build_panel(db, factor, price_frame, periods=(21,), quantiles=5, **pin)


panel = build_panel({"snapshot": "prices-v1"})
report = quant.verify(panel, rerun=lambda: build_panel({"snapshot": "prices-v1"}))
print(f"verified {report['verified']}  pinned {report['pinned']}  digest {report['digest'][:16]}")

unpinned = build_panel({})
relaxed = quant.verify(unpinned, strict=False)
print(f"unpinned: verified={relaxed['verified']} reason={relaxed['reason']!r}")

# %% [markdown]
# ## 4. ベンダーが訂正を出す
#
# 訂正が届きます。ある銘柄の、ある1営業日が誤っていました。修正はふつうのプレビュー可能な
# ミューテーションなので、古いバージョンは読めるまま残り、新しいほうが「最新」の意味になります。

# %%
target_symbol, target_day = "GE", pd.Timestamp("2024-03-14", tz="UTC").date()
frame = daily.to_pandas()
# The session stamp carries a time, so a whole day is a half-open range rather
# than an equality test.
window = frame[frame["ts"].dt.date == target_day].copy()
print("as delivered:")
print(window[window["symbol"] == target_symbol][["ts", "symbol", "close", "adj_close"]].to_string(index=False))

corrected = window.copy()
mask = corrected["symbol"] == target_symbol
corrected.loc[mask, ["close", "adj_close"]] = (
    corrected.loc[mask, ["close", "adj_close"]] * 0.82
).values
day_start = int(pd.Timestamp(target_day, tz="UTC").value // 1_000)
day_end = day_start + 24 * 60 * 60 * 1_000_000
plan = db.plan_replace_range(
    "prices",
    day_start,
    day_end,
    data=pa.Table.from_pandas(
        corrected.sort_values("symbol"), schema=prices.schema, preserve_index=False
    ),
    note="vendor restatement: GE close was 18% too high",
)
print(f"\nrows affected {plan.summary['rows_affected']}, "
      f"rows after {plan.summary['rows_after']:,}")
plan.apply()
print(f"prices is now at version {db.versions('prices')[-1]['sequence']}")

# %% [markdown]
# ## 5. 修正は答えを変えたか
#
# `restatement_impact` は1つの計算を2つの読み取り点で走らせ、メトリクスごとの差を報告します。
# バージョン管理されたストアが存在する理由そのものの問いです。「自分のアルファはいくつか」では
# なく、「改訂はそれを動かしたか、どれだけ動かしたか」です。

# %%
def headline(built) -> dict:
    decay = built.ic_decay().to_pandas().iloc[0]
    return {
        "mean_ic": float(decay["mean_ic"]),
        "ic_t_stat": float(decay["t_stat"]),
        "observations": float(decay["n"]),
    }


impact = quant.restatement_impact(
    build_panel,
    db,
    before={"version": first_version},
    after={},
    metric=headline,
)
print(f"changed: {impact['changed']}")
pd.DataFrame(impact["metrics"]).T.round(6)

# %% [markdown]
# 30銘柄のうち1銘柄の、2000営業日のうち1日の終値を1つ訂正しただけで、平均情報係数は小数第5位、
# t 値はおよそ100分の1だけ動きました。小さく、しかしゼロではありません。そこが要点です。
# 大きさを仮定せず測ったのです。セクター全体に及んだ修正や、銘柄を削除する生存バイアスの
# 訂正では、こうは読めません。
#
# 毎回走らせる理由は、費用がクエリ1本で済むからであり、代わりの道が、本番との差分から感度を
# 知ることだからです。

# %%
sensitive = quant.restatement_impact(
    build_panel,
    db,
    before={"version": first_version},
    after={},
    metric=lambda built: {
        "top_bucket_21d": float(
            built.quantile_returns().to_pandas().iloc[-1]["mean_21"]
        )
    },
)
pd.DataFrame(sensitive["metrics"]).T.round(8)

# %% [markdown]
# ## 6. 後片付け
#
# スイープのフォークは安価ですが無料ではなく、リサーチ用のデータベースには溜まっていきます。
# `drop()` はスイープのフォークを、比較用テーブルごと削除します。

# %%
print(f"forks before {len([name for name in db.fork_names()])}")
dropped = partial.drop()
print(f"dropped {dropped} brittle-sweep forks")
print(f"forks after  {len([name for name in db.fork_names()])}")

# %% [markdown]
# ## まとめ
#
# - `quant.sweep` は試行ごとにフォークを与えるので、試行どうしが汚染し合うことはなく、
#   `compare()` が1つのクエリで全部を読み戻します。
# - `keep_going` は夜間のグリッドには正しく、失敗がデータの破損を意味するときには誤りです。
# - `quant.verify` は来歴のダイジェスト *と* 計算し直した値の両方を確認し、ピン留めされて
#   いないものの検証は拒みます。
# - 修正はふつうのバージョン付きコミットなので、古い答えは上書きされずに読めるまま残ります。
# - `restatement_impact` は「ベンダーがデータを改訂した」をメトリクスごとの数字に変えます。
#   その事実が行動につながる唯一の形です。
# - リサーチが終わったらスイープのフォークは削除してください。比較用テーブルも一緒に消えます。

# %%
db.close()
