# %% [markdown]
# # 手元のデータが執行について言えること、言えないこと
#
# 予測市場の調査は、たいてい定期的な板のスナップショットの上で行われます。公開 API が返すのがそれ
# だからです。スナップショットで支えられる執行の主張とそうでない主張があり、その違いは程度の問題では
# ありません。キューポジションによる約定にはスナップショット間のすべての差分が必要で、5分グリッドから
# どれだけ丁寧に扱っても復元できません。このレシピでは、スナップショットが「支えられる」マイクロ
# ストラクチャのシグナルを計算し、支えられない主張をプリフライトが拒否する様子を見て、実際に取引できる
# 数量に値段をつけます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | スナップショットのデータ | グリッド上で標本化された板。公開 API が返すのはこれ |
# | 差分（delta）のデータ | 板の1つ1つの変化。スナップショットからは復元できない |
# | 板の厚み（depth） | 各価格帯に置かれている数量。つまり実際に取引できる数量 |
# | マイクロプライス | 両サイドの数量で加重したミッド。価格の行き先をよりよく表す |
# | インバランス | ベスト気配における、アスク数量に対するビッド数量の比 |
# | キューポジション | 同じ価格帯の待ち行列で自分の注文が何番目か。差分のデータが要る |
# | プリフライト | データが支えられない主張を、実行が始まる前に拒否する検査 |
# | 約定率（fill ratio） | 意図した数量のうち、実際に約定した割合 |
# | コスト予算 | 戦略が利益を出す前に越えなければならない費用の総額 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest

db = h5i_db.Database(cu.fresh_db("05_execution_fidelity_and_depth"), create=True)
FEE_RATE = 0.07

# %% [markdown]
# ## パネル
#
# 定期的な全板スナップショットで、各サイド1段ずつ、板の厚みはセッションを通じて変わります。この
# レシピが頼りにするのは `size` 列です。どれだけ取引できるかを決め、マイクロプライスの意味を決めるのが
# この列だからです。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | 記録側への到着時刻。リプレイの順序 |
# | `instrument_id` | `string` | マーケット |
# | `outcome` | `uint16` | 0 = YES、1 = NO |
# | `action` | `string` | `snapshot`。実フィードには `delta` やギャップもある |
# | `side` | `string` | `buy` がビッド、`sell` がアスク |
# | `price` | `float64` | 0.001 刻みのグリッド上の確率 |
# | `size` | `float64` | その価格帯に表示されている契約数 |

# %%
panel = cu.make_prediction_markets(n_markets=60, steps=32, seed=11)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel load")
db.snapshot("panel-v1", tables=list(panel), note="execution study input")
print(f"{panel['book_deltas'].num_rows:,} book rows across {panel['resolutions'].num_rows} markets")
panel["book_deltas"].to_pandas().head()

# %% [markdown]
# ## スナップショットが支えるシグナル
#
# マイクロプライスは、各サイドを「反対」サイドの数量で加重するので、置かれている流動性が少ないほうへ
# 傾きます。インバランスは同じ情報を、値域の決まった尺度で表したものです。どちらも1つのスナップ
# ショットから計算できます。だからこそ、スナップショットのデータに対して誠実な選択になります。

# %%
expiry_ns = int(db.sql("SELECT max(expiration_ns) AS e FROM instruments").to_pandas()["e"][0])
tops = db.sql(
    f"""
    SELECT ts_init AS ts, instrument_id,
           max(CASE WHEN side = 'buy'  THEN price END) AS bid,
           max(CASE WHEN side = 'sell' THEN price END) AS ask,
           max(CASE WHEN side = 'buy'  THEN size  END) AS bid_size,
           max(CASE WHEN side = 'sell' THEN size  END) AS ask_size
    FROM h5i('book_deltas', 'panel-v1')
    WHERE outcome = 0 AND ts_init <= to_timestamp_nanos({expiry_ns})
    GROUP BY ts_init, instrument_id
    ORDER BY ts_init, instrument_id
    """
).to_pandas()
tops["mid"] = (tops.bid + tops.ask) / 2
tops["microprice"] = (tops.bid * tops.ask_size + tops.ask * tops.bid_size) / (
    tops.bid_size + tops.ask_size
)
tops["imbalance"] = (tops.bid_size - tops.ask_size) / (tops.bid_size + tops.ask_size)
tops["lean_bp"] = (tops.microprice - tops.mid) * 10_000
tops["spread_bp"] = (tops.ask - tops.bid) * 10_000
print(f"{len(tops):,} snapshots")
print(tops[["bid", "ask", "mid", "microprice", "imbalance", "lean_bp", "spread_bp"]].describe().round(3).to_string())

# %% [markdown]
# スプレッドは両端で広がります。これは実際のイベント板が持つ形です。3セントの契約が両サイドとも
# 1ティック幅で提示されることはありえません。したがって、絶対的な確率で測るシグナルはどれも、価格
# 水準によって変わる費用と競うことになります。

# %%
tops["level"] = tops.mid.round(1).clip(0.1, 0.9)
by_level = tops.groupby("level").agg(
    snapshots=("mid", "size"),
    spread_bp=("spread_bp", "mean"),
    abs_lean_bp=("lean_bp", lambda s: s.abs().mean()),
    depth=("bid_size", "mean"),
)
print(by_level.round(1).to_string())

# %% [markdown]
# ## プリフライトが許さない主張
#
# `queue_position=True` は、板に載った注文がキューのどこにいたかをモデル化するようエンジンに求めます。
# それにはスナップショット間の順序づけられた差分が必要です。このパネルにあるのは定期スナップショット
# なので、`backtest.inspect` はもっともらしい数値を返す代わりに、その設定を拒否します。

# %%
submit_at = sorted({value.as_py() for value in panel["book_deltas"].column("ts_init")})[8]
signals = backtest.signal_table(
    [
        {
            "ts": submit_at + dt.timedelta(microseconds=1),
            "instrument_id": "EVENT-0030",
            "outcome": 0,
            "side": "buy",
            "quantity": 25.0,
            "tag": "probe",
        }
    ]
)
backtest.create_signal_table(db, "signals")
db.append("signals", signals)


def config(run_id: str, **execution) -> backtest.BacktestConfig:
    return backtest.BacktestConfig(
        run_id=run_id,
        data=backtest.DataConfig(signals="signals", snapshot="panel-v1"),
        portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
        execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE, **execution),
        metadata={"study": "execution-fidelity"},
    )


queue_claim = backtest.inspect(db, config("queue-probe", queue_position=True))
print(f"ok: {queue_claim.ok}")
print(f"fidelity: {queue_claim.fidelity}")
for issue in queue_claim.errors:
    print(f"  error   {issue.code}: {issue.message}")
for issue in queue_claim.warnings:
    print(f"  warning {issue.code}: {issue.message}")

# %% [markdown]
# `execute` も同じく拒否するので、`inspect` を呼ばずにこの検査を飛ばす、という抜け道はありません。
# ゲートが倒れるべき方向として、これが役に立つ向きです。

# %%
try:
    backtest.execute(db, config("queue-probe", queue_position=True))
except ValueError as error:
    print(f"execute refused: {error}")

# %% [markdown]
# ## このデータが支えるもの
#
# キューの主張を外せば設定は通り、検査は実際に持っている忠実度を報告します。`snapshot_only` の警告
# つきの `snapshot_l2` は正直なラベルです。既知の板に対する成行注文なら問題なく、キューの機構は
# 扱えません。

# %%
accepted = backtest.inspect(db, config("baseline"))
print(f"ok: {accepted.ok}  fidelity: {accepted.fidelity}")
for issue in accepted.warnings:
    print(f"  warning {issue.code}: {issue.message}")
baseline = backtest.execute(db, config("baseline"))
print(f"\nfilled at: {baseline.fills.to_pandas().price.iloc[0]:.4f}")
print(f"explain:   {baseline.explain()['fidelity']}, {baseline.explain()['status_counts']}")

# %% [markdown]
# ## 制約になるのは数量
#
# 各サイド1段ということは、表示されている厚みが板のすべてだということです。最上段より大きな注文は
# スナップショットからは約定できませんし、エンジンは流動性をでっちあげずにそう告げます。判断の瞬間に
# 実際にあった厚みに対して注文数量をスイープしてみるのは、2分でできて、架空のバックテストを1つ救う
# 検査です。

# %%
target = tops[(tops.ts == submit_at) & (tops.instrument_id == "EVENT-0030")].iloc[0]
print(f"EVENT-0030 at {submit_at}: ask {target.ask:.3f} for {target.ask_size:.1f} contracts")

rows = []
for quantity in (10.0, 25.0, 100.0, 250.0, 600.0):
    name = f"size-{int(quantity)}"
    table = backtest.signal_table(
        [
            {
                "ts": submit_at + dt.timedelta(microseconds=1),
                "instrument_id": "EVENT-0030",
                "outcome": 0,
                "side": "buy",
                "quantity": quantity,
                "tag": name,
            }
        ]
    )
    db.create_table(f"signals_{name}", table.schema, time_column="ts")
    db.append(f"signals_{name}", table)
    result = backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=name,
            data=backtest.DataConfig(signals=f"signals_{name}", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=1_000_000.0),
            execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE),
        ),
    )
    fills = result.fills.to_pandas()
    orders = result.orders.to_pandas()
    rows.append(
        {
            "requested": quantity,
            "filled": float(orders.filled.sum()),
            "status": orders.status.iloc[0],
            "avg_price": float((fills.price * fills.quantity).sum() / fills.quantity.sum())
            if len(fills)
            else float("nan"),
            "sweeps": result.explain()["orders_sweeping_multiple_levels"],
        }
    )
depth = pd.DataFrame(rows)
depth["fill_ratio"] = depth.filled / depth.requested
print(depth.round(3).to_string(index=False))

# %% [markdown]
# ## スリッページは別のシナリオとして
#
# `slippage_ticks` は大ざっぱな道具です。約定を決まったティック数だけずらし、キューモードより優先
# されるので、両方を組み合わせると存在しない取引所を記述することになります。エンジンは構築の時点で
# その組み合わせを拒否します。だからこれらは組み合わせ表ではなく、別々のシナリオとして実行します。

# %%
try:
    backtest.ExecutionConfig(queue_position=True, slippage_ticks=1)
except ValueError as error:
    print(f"rejected at construction: {error}")

scenarios = []
for ticks in (0, 2, 5, 10):
    name = f"slip-{ticks}"
    result = backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=name,
            data=backtest.DataConfig(signals="signals", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(
                fee_kind="kalshi", fee_rate=FEE_RATE, slippage_ticks=ticks or None
            ),
        ),
    )
    fills = result.fills.to_pandas()
    scenarios.append(
        {
            "slippage_ticks": ticks,
            "fill_price": float(fills.price.iloc[0]),
            "cost_bp_vs_ask": (float(fills.price.iloc[0]) - target.ask) * 10_000,
            "fees": float(result.summary()["commissions"]),
        }
    )
print(pd.DataFrame(scenarios).round(4).to_string(index=False))

# %% [markdown]
# ## 費用を1つの物差しに載せる
#
# スプレッド、スリッページ、手数料はどれも確率の単位なので、そのまま比較できます。この価格水準で
# 25契約の注文なら、単独で最も大きい成分は手数料です。このセクションで繰り返し出てくる教訓がこれ
# です。イベント契約において、手数料カーブは丸め誤差ではありません。

# %%
half_spread_bp = (target.ask - target.mid) * 10_000
fee_bp = FEE_RATE * target.ask * (1 - target.ask) * 10_000
slip_bp = scenarios[1]["cost_bp_vs_ask"]
budget = pd.DataFrame(
    [
        {"component": "half spread (mid to ask)", "cost_bp": half_spread_bp},
        {"component": "slippage, 2 ticks", "cost_bp": slip_bp},
        {"component": "fee at this price level", "cost_bp": fee_bp},
    ]
)
budget.loc[len(budget)] = {"component": "total", "cost_bp": budget.cost_bp.sum()}
print(budget.round(1).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(by_level.index, by_level.spread_bp, marker="o", color="#2c7fb8", label="spread")
axes[0].plot(
    by_level.index,
    [FEE_RATE * p * (1 - p) * 10_000 for p in by_level.index],
    marker="s",
    color="#c0392b",
    label="fee",
)
axes[0].set_title("Cost components by price level")
axes[0].set_xlabel("YES mid")
axes[0].set_ylabel("basis points of probability")
axes[0].legend(fontsize=8)
axes[1].bar(depth.requested.astype(str), depth.fill_ratio, color="#2c7fb8")
axes[1].axhline(1.0, color="black", lw=0.8)
axes[1].set_title("Fill ratio against one level of displayed depth")
axes[1].set_xlabel("contracts requested")
axes[1].set_ylabel("fraction filled")
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - マイクロプライスとインバランスは1つのスナップショットから計算できるので、スナップショットの
#   データが実際に支えられるマイクロストラクチャのシグナルはこの2つである。
# - キューポジションの主張は支えられない。`backtest.inspect` は `unsupported_queue_claim` を報告し、
#   `execute` は拒否する。この限界は文書に書かれているだけでなく、強制されている。
# - 注文数量を表示されている厚みと突き合わせる。各サイド1段なら、大口注文は悪い価格で約定するのでは
#   なく取り消される。確認しないバックテストは、なかった流動性をそっと仮定してしまう。
# - `slippage_ticks` と `queue_position` は設計上、両立しない。組み合わせ表ではなくシナリオとして
#   実行する。
# - ここで働いた h5i-db の機能。データが何を支えるかを名指しする `ReplayFidelity`、支えられない主張を
#   拒否するプリフライト、そしてすべてのシナリオの背後にある1つのピン留めされたスナップショット。
#   変わるのは執行の前提だけになる。

# %%
db.close()
