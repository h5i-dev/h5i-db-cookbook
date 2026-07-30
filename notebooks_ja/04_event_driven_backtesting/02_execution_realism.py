# %% [markdown]
# # 執行の前提をストレステストする
#
# どんなバックテストにも執行の前提が入っていて、その多くは書き留められないままです。頼んだ価格で
# 約定する。判断した瞬間に注文が届く。自分の指値はキューの先頭にいる。どれも成り立ちませんし、
# どれも結果を良く見せます。
#
# 実務での作法は、テープを固定したうえで前提を1つずつ変え、結論がどれだけ動くかを見ることです。
# 手数料モデル次第で符号が変わる戦略は、戦略ではありません。ありうる設定のすべてで生き残ったなら、
# さらに手をかける価値があります。
#
# 1つの約定の前提が結論を決めてしまうとき、そのバックテストは脆いということです。このレシピでは、
# 1組のシグナルを手数料、不利なスリッページ、レイテンシ、キューポジションに通し、結果として出てくる
# 約定と現金を比べます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 手数料（fee） | 取引所が1取引ごとに明示的に課す費用 |
# | スリッページ | 想定した価格と実際に得た価格との差 |
# | レイテンシ | 判断から取引所に届くまでの遅延 |
# | キューポジション | 同じ価格帯の待ち行列で自分の注文が何番目か。約定するかどうかを左右する |
# | メイカー／テイカー | メイカーは指値を置いて待ち、テイカーはスプレッドを越えて即座に約定する |
# | インプリメンテーション・ショートフォール | 判断をポジションに変えるまでの総費用。約定しなかった分も含む |
# | 感応度分析 | 前提を1つだけ変えて再実行し、それが結論を動かしていたかを見ること |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %% [markdown]
# 入力のテープには 240 件の L2 スナップショットと 240 件のプリントが入っています。価格はトレンドを
# 描きながら振動するので、遅れて届いた注文が見る板は測れるほど違ってきます。
#
# | テーブル | 主な列 | 意味 |
# |---|---|---|
# | `instruments` | `instrument_id`、`tick_size` | 取引所側の制約 |
# | `book_deltas` | `ts_init`、`side`、`price`、`size` | アトミックな L2 スナップショット |
# | `trades` | `ts_init`、`price`、`size`、`aggressor` | キューを消費するプリント |

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

fixture = cu.make_backtest_fixture(steps=240)
for name, table in fixture.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")
fixture["trades"].to_pandas().head()

# %% [markdown]
# テープの保存とピン留めは一度だけ行います。以下のシナリオはすべてこのスナップショットだけを読むので、
# 差が出るとすればデータではなく執行設定のせいです。

# %%
db = h5i_db.Database(cu.fresh_db("04_execution_realism"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="execution sensitivity fixture")
db.snapshot(
    "execution-input-v1",
    tables=["instruments", "book_deltas", "trades"],
    note="Common tape for execution sensitivity",
)

# %% [markdown]
# アクティブな戦略は成行の買いと売りを交互に出します。成行の意図に絞ることで、手数料、スリッページ、
# レイテンシを、指値が約定するかどうかの不確実性から切り離せます。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts` | `timestamp[ns]` | 発注時刻 |
# | `side` | `string` | 売買の方向 |
# | `quantity` | `float64` | 要求する数量 |
# | `kind` | `string` | この実験では成行注文 |
# | `tag` | `string` | シナリオに依存しない注文ラベル |

# %%
base = dt.datetime(2026, 6, 1, 14, 0, 0)
active_rows = []
for index, second in enumerate((20, 50, 80, 110, 140, 170)):
    active_rows.append(
        {
            "ts": base + dt.timedelta(seconds=second),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy" if index % 2 == 0 else "sell",
            "quantity": 75.0,
            "tag": f"active-{index + 1}",
        }
    )
active_signals = backtest.signal_table(active_rows)
print(f"{active_signals.num_rows:,} rows x {active_signals.num_columns} columns")
active_signals.to_pandas()

# %% [markdown]
# アクティブな意図の保存も一度だけです。各実行は一意な ID を受け取るので、出力フォークもそれぞれ
# 別になります。

# %%
backtest.create_signal_table(db, "active_signals")
db.append("active_signals", active_signals, note="active execution experiment")

# %% [markdown]
# 摩擦の組み合わせ表は、前提を一度に1つだけ変えます。予測市場の手数料は取引所に合わせたカーブを
# 使います。比例手数料はメイカー／テイカーのレートを使います。スリッページとキューポジションは、
# 現在の API では別々のモードになっています。

# %%
scenarios = {
    "frictionless": {},
    "prediction_fee": {"fee_rate": 0.03},
    "proportional_fee": {
        "fee_kind": "proportional",
        "fee_rate": 0.001,
        "maker_rebate": -0.0001,
    },
    "20_tick_slippage": {"slippage_ticks": 20},
    "3_second_latency": {"latency_nanos": 3_000_000_000},
}

rows = []
for name, config in scenarios.items():
    report = backtest.run(
        db,
        f"active-{name}",
        starting_cash=10_000.0,
        signals="active_signals",
        snapshot="execution-input-v1",
        equity_interval_nanos=5_000_000_000,
        **config,
    )
    rows.append(
        {
            "scenario": name,
            "fills": report["fills"],
            "final_cash": report["final_cash"],
            "realized_pnl": report["realized_pnl"],
            "commissions": report["commissions"],
        }
    )
active_comparison = pd.DataFrame(rows).set_index("scenario")
active_comparison

# %% [markdown]
# 各シナリオをインプリメンテーション・ショートフォールの視点に変換します。摩擦なしの実行は対照群で
# あって、その執行が実現できるという主張ではありません。

# %%
control_cash = active_comparison.loc["frictionless", "final_cash"]
active_comparison["cash_shortfall_vs_control"] = (
    control_cash - active_comparison["final_cash"]
)
active_comparison.sort_values("cash_shortfall_vs_control", ascending=False)

# %% [markdown]
# パッシブな意図には別の実験が要ります。指値は発注時点で表示されていたビッドです。キューを考慮する
# マッチングはこれを表示数量の後ろに置くので、届くにはその後に売り仕掛けのプリントが必要になります。

# %%
book = fixture["book_deltas"].to_pandas()
arrival = book[(book["event_index"] == 30) & (book["side"] == "buy")].iloc[0]
passive_signals = backtest.signal_table(
    [
        {
            "ts": arrival["ts_init"].to_pydatetime(),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy",
            "quantity": 20.0,
            "kind": "limit",
            "limit_price": float(arrival["price"]),
            "time_in_force": "gtc",
            "tag": "passive-entry",
        }
    ]
)
print(f"{passive_signals.num_rows:,} rows x {passive_signals.num_columns} columns")
passive_signals.to_pandas()

# %% [markdown]
# ふつうの板マッチングを、保守的なキューモードと楽観的なキューモードと比べます。楽観的モードが変える
# のは、仕掛けたサイドが不明な約定だけです。このテープはすべての約定に仕掛けサイドが入っているので、
# 2つのキュー実行は一致するはずです。

# %%
backtest.create_signal_table(db, "passive_signals")
db.append("passive_signals", passive_signals, note="passive queue experiment")

queue_rows = []
for name, config in {
    "book_only": {},
    "queue_conservative": {"queue_position": True},
    "queue_optimistic": {
        "queue_position": True,
        "optimistic_queue": True,
    },
}.items():
    report = backtest.run(
        db,
        f"passive-{name}",
        starting_cash=10_000.0,
        signals="passive_signals",
        snapshot="execution-input-v1",
        **config,
    )
    run_db = db.fork(report["fork"])
    fills = run_db.read("bt_fills").to_pandas()
    run_db.close()
    queue_rows.append(
        {
            "scenario": name,
            "fills": report["fills"],
            "fill_time": None if fills.empty else fills.iloc[0]["ts"],
            "fill_price": None if fills.empty else fills.iloc[0]["price"],
            "is_taker": None if fills.empty else fills.iloc[0]["is_taker"],
        }
    )
queue_comparison = pd.DataFrame(queue_rows).set_index("scenario")
queue_comparison

# %% [markdown]
# 生の現金ではなく、現金のショートフォールを描きます。ポートフォリオ価値の軸を切り詰めることなく、
# 小さな執行費用が見えるようになります。

# %%
fig, ax = plt.subplots(figsize=(9, 4))
active_comparison["cash_shortfall_vs_control"].plot.bar(ax=ax, color="#4472C4")
ax.set_title("Execution-model cash shortfall")
ax.set_xlabel("Scenario")
ax.set_ylabel("Cash shortfall versus frictionless")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - ピン留めしたテープに対して、執行の前提は一度に1つだけ変える。
# - インプリメンテーション・ショートフォールは、はっきり名前をつけた対照群との比較で報告する。
# - 能動的な摩擦を調べるなら成行注文を、キューを調べるなら指値注文を使う。
# - キューポジションには、仕掛けサイドの分類が信頼できるプリントが要る。
# - ありうるモデル間の感応度は、誤差範囲ではなくモデルリスクとして扱う。

# %%
db.close()
