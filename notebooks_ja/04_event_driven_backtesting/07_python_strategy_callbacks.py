# %% [markdown]
# # 経路依存の Python 戦略
#
# 戦略のなかには、注文意図のテーブルとしては書けないものがあります。直前のポジションが決済されたと
# 確認できてはじめて建てる規則や、約定してから30秒待って動く規則は、いま見ている行ではなく、すでに
# 起きたことに依存しています。
#
# これが経路依存という言葉の意味で、戦略がイベントをまたいで状態を持つ必要があります。代償として、
# 戦略が見るイベントごとに Python へ渡る処理が入ります。このレシピの最後で、より安い境界と、それを
# 選ぶべき場面を整理するのはそのためです。
#
# signals とコマンドのテーブルは最も速い戦略の境界です。リプレイが最後まで Rust の中で完結するから
# です。とはいえ、状態や、約定を受けた判断や、タイマーが本当に必要な戦略もあります。このレシピでは、
# オプトインの Python コールバック面を使いつつ、戦略コードにエンジン内部の借用への直接アクセスは
# 渡しません。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | コールバック | 何かが起きたときにエンジンから呼ばれる、自分で書いた Python 関数 |
# | 経路依存 | いまのデータだけでなく、すでに起きたことに依存する判断 |
# | 状態を持つ（stateful） | 戦略がイベントをまたいで変数を持ち越すこと |
# | タイマー | データではなく、未来の時刻に予約して呼ばれるコールバック |
# | 戦略の同一性 | 戦略に付ける安定した名前。再実行しても実行どうしを比較できる |
# | 決定性 | 同じ入力から毎回同じ出力が出ること。再実行がこれを検証する |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

INSTRUMENT_ID = "RATE-CUT-YES"
MARKET_CUT = "callback-market-cut"
SECOND = 1_000_000_000

fixture = cu.make_backtest_fixture(steps=150, instrument_id=INSTRUMENT_ID)
db = h5i_db.Database(cu.fresh_db("07_python_strategy_callbacks"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="deterministic callback fixture")
db.snapshot(
    MARKET_CUT,
    tables=["instruments", "book_deltas", "trades"],
    note="Approved market-data cut for callback examples",
)

# %% [markdown]
# ## 効果を明示する状態機械を書く
#
# コールバックの入力はふつうの辞書です。コールバックは `None`、1つのアクションのマッピング、または
# マッピングの反復可能オブジェクトを返します。エンジンはそれらのアクションをコールバックのあとに
# 適用し、因果的なイベント順序を保ちます。この戦略は次のように動きます。
#
# 1. 最初の観測可能なマーケットイベントから、エントリのタイマーを予約する
# 2. そのタイマーが発火したら成行の買いを出す
# 3. エントリの約定が確認できてはじめて、エグジットを予約する
# 4. エグジットのタイマーから、建玉を減らすだけの売りを出す

# %%
class TimedRoundTrip(backtest.EventStrategy):
    def __init__(self):
        self.entry_scheduled = False
        self.exit_scheduled = False
        self.fill_log = []

    def on_event(self, context, event):
        assert context["now"] == event["ts_init"]
        if not self.entry_scheduled:
            self.entry_scheduled = True
            return {
                "action": "timer",
                "name": "enter",
                "ts": context["now"] + 20 * SECOND,
            }
        return None

    def on_timer(self, context, event):
        if event["name"] == "enter":
            return {
                "action": "submit",
                "client_order_id": "entry",
                "instrument_id": INSTRUMENT_ID,
                "side": "buy",
                "quantity": 25.0,
                "tag": "timed-entry",
            }
        if event["name"] == "exit":
            return {
                "action": "submit",
                "client_order_id": "exit",
                "instrument_id": INSTRUMENT_ID,
                "side": "sell",
                "quantity": 25.0,
                "reduce_only": True,
                "tag": "fill-driven-exit",
            }
        raise ValueError(f"unexpected timer {event['name']!r}")

    def on_fill(self, context, event):
        self.fill_log.append(event)
        if event["tag"] == "timed-entry" and not self.exit_scheduled:
            self.exit_scheduled = True
            return {
                "action": "timer",
                "name": "exit",
                "ts": event["ts"] + 60 * SECOND,
            }
        return None


strategy = TimedRoundTrip()

# %% [markdown]
# `strategy_id` は実行とともに保存されます。パッケージ化された調査コードなら、`run_strategy` が
# クラスのソースから導出できます。ノートブックや本番では、明示的にバージョンを与えるほうがよいことが
# 多いです。コードレビューがリリースやコミットに紐づけられるからです。

# %%
result = backtest.run_strategy(
    db,
    "timed-round-trip",
    strategy,
    strategy_id="cookbook.TimedRoundTrip:v1",
    starting_cash=10_000.0,
    data=backtest.DataConfig(
        snapshot=MARKET_CUT,
        minimum_coverage=0.95,
    ),
    execution=backtest.ExecutionConfig(
        fee_kind="prediction_market",
        fee_rate=0.02,
        latency_nanos=1_000_000,
    ),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=25.0,
        max_open_orders=2,
    ),
    output=backtest.OutputConfig(equity_interval_nanos=5 * SECOND),
    metadata={"purpose": "callback and timer contract demonstration"},
)
result

# %% [markdown]
# 約定のコールバックは、GIL の下で同じ戦略オブジェクトの上を走りました。正本はあくまで保存された
# テーブルです。ローカルな状態は判断や診断には役立ちますが、監査記録にはなりません。

# %%
orders = result.orders.to_pandas()
fills = result.fills.to_pandas()
assert result["fills"] == 2
assert [fill["tag"] for fill in strategy.fill_log] == [
    "timed-entry",
    "fill-driven-exit",
]
assert fills["tag"].tolist() == ["timed-entry", "fill-driven-exit"]
assert result.positions.to_pandas()["quantity"].abs().sum() < 1e-9
orders[["order_id", "side", "quantity", "filled", "status", "tag"]]

# %% [markdown]
# 同じ戦略実装を渡す限り、コールバックを使った実行も再現できます。検証は隔離された再実行を作り、
# 指標と正本の出力テーブルをすべて突き合わせてから、一時フォークを削除します。

# %%
verification = result.verify(strategy=TimedRoundTrip())
assert verification["verified"]
verification

# %% [markdown]
# ## どの戦略境界を選ぶか
#
# | 境界 | 向いている用途 | 性能 | ライフサイクル |
# |---|---|---|---|
# | signals | ベクトル化した建玉・決済と目標ポジション | ネイティブのホットループ | 発注 |
# | commands | 気配提示と、あらかじめ決まった執行スケジュール | ネイティブのホットループ | 発注／訂正／取消 |
# | Python コールバック | 状態機械、約定への反応、タイマー | コールバックごとに GIL を1回またぐ | 全体 |
#
# 戦略を表現できる最も単純な境界を選んでください。コールバックの柔軟性には価値がありますが、意識的な
# 選択であるべきで、すべてのバックテストが払う偶然の税になってはいけません。

# %%
db.close()
