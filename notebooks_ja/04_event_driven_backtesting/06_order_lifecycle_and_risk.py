# %% [markdown]
# # 注文のライフサイクルと口座リスク
#
# たいていのバックテストは、注文を1つのイベントとして扱います。送れば約定する、というわけです。
# 実際の注文には一生があります。板に載って待ち、市場が動けば値段を訂正され、そのたびにキューの優先
# 順位を失い、古くなればキャンセルされ、限度を破りそうならその場で拒否されます。
#
# 約定だけをモデル化すると、実際にお金のかかる2つのことが隠れます。値段の訂正はキューの最後尾へ
# 自分を送ります。そして、ノートブックの中にあるリスク限度は、リスク限度として働きません。
#
# 本番のバックテストには、時刻のついた建玉だけでは足りません。気配は訂正され、古い注文はキャンセル
# され、口座の限度は危険な意図をシミュレートされた取引所に届く前に拒否しなければなりません。この
# レシピでは、安定したクライアント注文 ID を使ってそのライフサイクル全体を動かし、そこから残る監査
# 証跡を確かめます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 注文のライフサイクル | 発注、訂正、取消、約定、失効。注文が送られたあとにすることすべて |
# | 訂正（amend） | 板に載っている注文を出し直さずに、価格や数量を変えること |
# | クライアント注文 ID | 戦略側で付ける安定した識別子。注文の履歴全体がこれでつながる |
# | プリフライト | 危険な意図や対応できない意図を、取引所に届く前に拒否する検査 |
# | 口座の限度 | エンジンが自前で守る、エクスポージャ・注文数量・現金の上限 |
# | 監査証跡 | 何が要求され、何が拒否され、それはなぜかの記録 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

INSTRUMENT_ID = "RATE-CUT-YES"
MARKET_CUT = "lifecycle-market-cut"
SECOND = 1_000_000_000

fixture = cu.make_backtest_fixture(steps=120, instrument_id=INSTRUMENT_ID)
db = h5i_db.Database(cu.fresh_db("06_order_lifecycle_and_risk"), create=True)
for name, table in fixture.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="deterministic lifecycle fixture")
db.snapshot(
    MARKET_CUT,
    tables=["instruments", "book_deltas", "trades"],
    note="Approved market-data cut for lifecycle examples",
)

# %% [markdown]
# ## ライフサイクルを宣言する
#
# `client_order_id` はエンジンではなく戦略のものです。後続の行はこの安定した名前で、`submit` が
# 作った注文そのものを指します。取消の行には ID だけあれば足りるので、発注時のフィールドは保存
# スキーマ上 nullable になっています。各アクションに必要なフィールドはビルダが検証します。

# %%
base = dt.datetime(2026, 6, 1, 14, 0, 0)
lifecycle = backtest.command_table(
    [
        {
            "ts": base + dt.timedelta(seconds=10),
            "action": "submit",
            "client_order_id": "yes-quote-001",
            "instrument_id": INSTRUMENT_ID,
            "side": "buy",
            "quantity": 20.0,
            "kind": "limit",
            "limit_price": 0.25,
            "tag": "passive-quote",
        },
        {
            "ts": base + dt.timedelta(seconds=30),
            "action": "amend",
            "client_order_id": "yes-quote-001",
            "quantity": 10.0,
            "limit_price": 0.26,
        },
        {
            "ts": base + dt.timedelta(seconds=60),
            "action": "cancel",
            "client_order_id": "yes-quote-001",
        },
    ]
)
backtest.create_command_table(db, "lifecycle_commands")
db.append(
    "lifecycle_commands",
    lifecycle,
    note="submit, reprice/resize, then cancel one quote",
)
lifecycle.to_pandas()

# %% [markdown]
# 型のついた設定が、結果を左右する前提をすべて捕まえます。プリフライトは、重いリプレイが始まる前に、
# マーケットのピン、スキーマ、カバレッジ、そのフィードが支えられる最も強い忠実度を確認します。

# %%
lifecycle_config = backtest.BacktestConfig(
    run_id="lifecycle",
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    data=backtest.DataConfig(
        commands="lifecycle_commands",
        snapshot=MARKET_CUT,
    ),
    execution=backtest.ExecutionConfig(
        fee_kind="prediction_market",
        fee_rate=0.02,
        latency_nanos=2_000_000,
    ),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=50.0,
        max_open_orders=4,
    ),
    output=backtest.OutputConfig(equity_interval_nanos=5 * SECOND),
    metadata={"research_ticket": "PM-142", "owner": "market-making"},
)
inspection = backtest.inspect(db, lifecycle_config)
inspection.to_dict()

# %%
inspection.raise_for_errors()
lifecycle_result = backtest.execute(db, lifecycle_config)
orders = lifecycle_result.orders.to_pandas()
orders[
    [
        "order_id",
        "side",
        "limit_price",
        "quantity",
        "filled",
        "status",
        "reject_reason",
        "tag",
    ]
]

# %% [markdown]
# この気配は意図して市場から離して置いてあるので、期待される結果は取消であって、約定ではありません。
# `explain()` は何も起きなかったことを検分できるようにし、`verify()` は保存された設定で再実行して、正本となる
# 出力テーブルをすべて突き合わせます。

# %%
assert lifecycle_result["fills"] == 0
assert orders["status"].tolist() == ["cancelled"]
explanation = lifecycle_result.explain()
verification = lifecycle_result.verify()
assert verification["verified"]
explanation

# %% [markdown]
# ## リスクが取引所での執行より先に拒否することを示す
#
# リスク管理はノートブック側のフィルタではなく、エンジンにもともと備わった制約です。過大な成行注文は
# 拒否として記録され、レイテンシにもマッチングにも入らず、その理由が `bt_orders` に残ります。

# %%
risk_commands = backtest.command_table(
    [
        {
            "ts": base + dt.timedelta(seconds=20),
            "action": "submit",
            "client_order_id": "oversized-entry",
            "instrument_id": INSTRUMENT_ID,
            "side": "buy",
            "quantity": 100.0,
            "tag": "must-reject",
        }
    ]
)
backtest.create_command_table(db, "risk_commands")
db.append("risk_commands", risk_commands, note="risk rejection demonstration")

risk_config = backtest.BacktestConfig(
    run_id="risk-rejection",
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    data=backtest.DataConfig(commands="risk_commands", snapshot=MARKET_CUT),
    risk=backtest.RiskConfig(
        max_order_quantity=25.0,
        max_abs_position=50.0,
        max_open_orders=4,
    ),
)
risk_result = backtest.execute(db, risk_config)
risk_order = risk_result.orders.to_pylist()[0]
assert risk_result["fills"] == 0
assert risk_order["status"] == "rejected"
assert "max_order_quantity" in risk_order["reject_reason"]
risk_result.explain()

# %% [markdown]
# ## まとめ
#
# - 安定したクライアント ID があると、訂正と取消の流れがエンジン側の ID から独立する。
# - 訂正は取引所と同じキューの規則に従う。値段を変えたり数量を増やしたりすると優先順位を失う。
# - ポジション限度は、約定済みのエクスポージャだけでなく、生きている注文もすべて含む。
# - 拒否の理由は保存されクエリできるので、約定ゼロの実行でも原因を追える。
# - 型のついた設定、プリフライト、意味的な検証の3つで、再現できる運用上の契約になる。

# %%
db.close()
