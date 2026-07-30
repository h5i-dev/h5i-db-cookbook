# %% [markdown]
# # 実際の Polymarket の板で因果的にシグナルをリプレイする
#
# 合成データで確かめられるのは、配管がつながっていることだけです。戦略が機能するかどうかは分かり
# ません。そこで見つかる構造は、生成器が置いた構造だからです。
#
# 試されるのは実際の板の上です。生成器が見逃してくれる設計ミスを、実データは容赦なく罰します。
# 売買するバー自身から計算した特徴量、シグナルに漏れ込んだ決着ラベル、板が一度も出していない価格
# での約定。どれも、負ける規則を勝つバックテストに変えてしまいます。
#
# このレシピでは、範囲を限った
# [Kaggle の Polymarket サンプル](https://www.kaggle.com/datasets/marvingozo/polymarket-tick-level-orderbook-dataset)
# の上に、意図して控えめなマイクロストラクチャ戦略を組み立てます。実務として学ぶところは実験の
# 設計です。
#
# - 特徴量は、その分足が閉じたあとにはじめて観測可能になる
# - 最終的な決着ラベルは除外する
# - 注文は特徴量バーの価格ではなく、実際に記録された板にぶつかる
# - 手数料、レイテンシ、スリッページは感応度の軸として扱う
# - どの実行もピン留めされ、それぞれのフォーク上でクエリできる

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | リプレイ | 記録されたマーケットデータを、元の順序どおりにエンジンへ流し直すこと |
# | 因果的な特徴量 | 自分のタイムスタンプ時点で観測できるデータだけから計算した特徴量 |
# | 観測可能性 | その値が実際に知り得るようになった時点。出来事が起きた時点とは違う |
# | 決着ラベル | 最終的な結果。戦略が決して見てはいけないもの |
# | インバランス | ベスト気配における、アスク数量に対するビッド数量の比 |
# | Zスコア | ある値がローリング平均から標準偏差いくつ分離れているか |
# | 実行フォーク | 実行が結果を書き込む、隔離されたデータベースの枝 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

CACHE = Path("data/cache/kaggle-polymarket")
sample = cu.load_kaggle_sample(CACHE, depth_levels=10)
print(sample.question)
print(pd.Series(sample.audit, name="value").to_frame())

# %% [markdown]
# ## 因果的な特徴量を作る
#
# 分 *t* の `depth_imbalance` には、ローダが *t + 1分* の時刻を押します。ベースラインは直前の120
# 観測だけを使います。ローリングの2つのモーメントはどちらも、現在のZスコアを計算する前にシフト
# してあります。絶対値で2シグマを超えた最初のイベントで1つ建玉し、2時間の固定ホライズンで決済します。
#
# これは執行のチュートリアルであって、掘り当てた売買ルールではありません。1つのマーケットの1日ぶんは、
# モデル選択の証拠としてはあまりに乏しい量です。

# %%
features = sample.features.to_pandas()
depth = features["depth_imbalance"]
features["history_mean"] = depth.rolling(120, min_periods=60).mean().shift(1)
features["history_std"] = depth.rolling(120, min_periods=60).std().shift(1)
features["depth_z"] = (
    (depth - features["history_mean"]) / features["history_std"].replace(0, np.nan)
)

book_times = sample.book_deltas.column("ts_init")
book_start = pd.Timestamp(book_times[0].as_py())
book_end = pd.Timestamp(book_times[-1].as_py())
eligible = features[
    features["ts_init"].between(
        book_start, book_end - pd.Timedelta(hours=2), inclusive="both"
    )
    & features["depth_z"].abs().ge(2.0)
]
assert not eligible.empty
entry = eligible.iloc[0]
exit_time = entry["ts_init"] + pd.Timedelta(hours=2)
direction = "buy" if entry["depth_z"] > 0 else "sell"
close_direction = "sell" if direction == "buy" else "buy"
print(
    f"{direction=} at {entry['ts_init']} z={entry['depth_z']:.2f}; "
    f"close at {exit_time}"
)

# %% [markdown]
# シグナルは時刻のついた意図です。執行価格は持ちません。シグナルを生んだのと同じバーの終値で約定
# させてしまう、よくある誤りを防げます。

# %%
quantity = 20.0
signals = backtest.signal_table(
    [
        {
            "ts": entry["ts_init"].to_pydatetime(),
            "instrument_id": sample.market_id,
            "side": direction,
            "quantity": quantity,
            "tag": "depth-z-entry",
        },
        {
            "ts": exit_time.to_pydatetime(),
            "instrument_id": sample.market_id,
            "side": close_direction,
            "quantity": quantity,
            "tag": "time-exit",
        },
    ]
)
signals.to_pandas()

# %% [markdown]
# ## 実行の前にデータをピン留めする

# %%
db = h5i_db.Database(cu.fresh_db("05_kaggle_polymarket_replay"), create=True)
for name, table in {
    "instruments": sample.instruments,
    "book_deltas": sample.book_deltas,
    "trades": sample.trades,
    "features_1m": sample.features,
}.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="bounded CC BY-NC Kaggle Polymarket sample")
db.snapshot(
    "approved-kaggle-cut",
    tables=["instruments", "book_deltas", "trades", "features_1m"],
    note="Real snapshot replay inputs; resolution label excluded",
)
backtest.create_signal_table(db)
db.append("signals", signals, note="causal depth-imbalance experiment")

# %% [markdown]
# ## 執行の前提はモデルリスクとして扱う
#
# スナップショットのフィードでは正確なキューの消化を再現できないので、この実験では成行注文を使い、
# キューポジションによる約定は有効にしません。変えるのは、Polymarket のカーブ状の手数料率、
# レイテンシ、不利なスリッページです。ベースラインを単独で示さないのは意図的です。

# %%
scenarios = {
    "frictionless": {
        "fee_rate": 0.0,
        "latency_nanos": 0,
        "slippage_ticks": 0,
    },
    "base": {
        "fee_rate": 0.02,
        "latency_nanos": 50_000_000,
        "slippage_ticks": 0,
    },
    "stressed": {
        "fee_rate": 0.035,
        "latency_nanos": 250_000_000,
        "slippage_ticks": 2,
    },
}
reports = {}
inspections = {}
for name, assumptions in scenarios.items():
    config = backtest.BacktestConfig(
        run_id=f"kaggle-{name}",
        portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
        data=backtest.DataConfig(
            signals="signals",
            snapshot="approved-kaggle-cut",
            minimum_coverage=0.75,
        ),
        execution=backtest.ExecutionConfig(
            fee_kind="prediction_market",
            fee_rate=assumptions["fee_rate"],
            latency_nanos=assumptions["latency_nanos"],
            slippage_ticks=assumptions["slippage_ticks"],
        ),
        risk=backtest.RiskConfig(
            max_order_quantity=quantity,
            max_abs_position=quantity,
            max_open_orders=1,
        ),
        output=backtest.OutputConfig(
            equity_interval_nanos=60_000_000_000,
        ),
        metadata={
            "dataset": cu.KAGGLE_POLYMARKET_DATASET,
            "scenario": name,
            "signal": "causal depth imbalance",
        },
    )
    inspections[name] = backtest.inspect(db, config)
    inspections[name].raise_for_errors()
    reports[name] = backtest.execute(db, config)

summary = pd.DataFrame(reports).T[
    ["fills", "commissions", "realized_pnl", "final_cash", "digest", "fork"]
]
summary

# %% [markdown]
# プリフライトは、このソースをティック差分の L2 ではなくスナップショットの L2 と判定します。この
# 忠実度の表明は結果の一部であり、ソースが支えられないキューポジションの主張を、ノートブックが
# そっと行ってしまうのを防ぎます。

# %%
{
    name: {
        "fidelity": inspection.fidelity.value,
        "warnings": [issue.message for issue in inspection.warnings],
    }
    for name, inspection in inspections.items()
}

# %% [markdown]
# 妥当な比較とは、どのシナリオでもタグのついた同じ2つの注文が完全に執行されることです。1つの注文が
# 板の複数段を食えば約定は複数件になるので、約定件数を取引回数として数えるのは見落としやすい分析の
# 誤りです。ダイジェストが変わるのは、執行設定も実行の同一性の一部だからです。

# %%
assert summary["fills"].ge(2).all()
assert summary["digest"].nunique() == len(summary)

fill_frames = []
for scenario, report in reports.items():
    run_db = db.fork(report["fork"])
    scenario_fills = run_db.read("bt_fills").to_pandas()
    filled_by_tag = scenario_fills.groupby("tag")["quantity"].sum()
    assert set(filled_by_tag.index) == {"depth-z-entry", "time-exit"}
    assert filled_by_tag.eq(quantity).all()
    scenario_fills["scenario"] = scenario
    fill_frames.append(scenario_fills)
    run_db.close()

# verify() reruns the config into a scratch fork and tears it down afterwards,
# so read the run's own fork before calling it.
assert reports["base"].verify()["verified"]

fills = pd.concat(fill_frames, ignore_index=True)
fills[["scenario", "ts", "side", "price", "quantity", "commission", "tag"]]

# %% [markdown]
# 最も保守的なシナリオのエクイティを描きます。エンジンは保持した板イベントをすべて消費しますが、
# 出力の頻度は1分です。

# %%
stressed_db = db.fork(reports["stressed"]["fork"])
equity = stressed_db.read("bt_equity").to_pandas()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(equity["ts"], equity["equity"], linewidth=1.6)
ax.set(
    title="Stressed execution scenario",
    xlabel="simulated time",
    ylabel="portfolio value",
)
fig.tight_layout()

# %% [markdown]
# ## 本番向けチェックリスト
#
# - 多数のマーケットと重ならない日付で繰り返す。この1回の取引はパイプラインのテストであって、
#   統計的な証拠ではない。
# - 外部ファイルのハッシュも h5i-db のスナップショットと並べてピン留めする。
# - 手数料率は取引所とマーケットの日付に合わせて較正する。
# - キューポジションの主張を有効にする前に、定期スナップショットではなくティック差分を使う。
# - 損益だけでなく、拒否された注文、カバレッジ、回転率、感応度も報告する。
# - 元データセットの非商用ライセンスを守る。

# %%
stressed_db.close()
db.close()
