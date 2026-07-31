# %% [markdown]
# # 執行アルゴリズムと、終われなかったことのコスト
#
# レシピ 01/02 は VWAP と TWAP をベンチマークとして計測しました。今回はそこへ向けて売買します。
# 注文は2万株の買いで、最良気配には数百株しか出ていません。執行デスクが実際に置かれている状況
# そのものです。必要な流動性は、必要だと判断した瞬間には存在しません。
#
# このレシピが防ごうとしている間違いは、執行を平均価格だけで測ることです。アルゴリズムは
# 売買しなければどんなベンチマークにも勝てますし、買えなかった株数は、自分抜きで動いた相場に
# おける「持っていないポジション」です。Perold のインプリメンテーション・ショートフォールは
# その両方を数えます。そしてリプレイはその数字を直接出します。約定は板が認めたもので、
# 未約定の残りも同じテーブルの中にあります。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 親注文 | デスクが売買を依頼された全数量 |
# | スライス | 親注文を執行するために出す子注文1本 |
# | 到着価格 | 判断した瞬間の仲値。ごまかしの効かないベンチマーク |
# | TWAP | スライスを時間で均等に配分するスケジュール |
# | VWAP | 予想出来高に比例してスライスを配分するスケジュール |
# | POV | 出来高参加率。プリントの一定割合を売買する方式 |
# | インプリメンテーション・ショートフォール | 到着価格に対するコスト。買えなかった株数も含む |
# | 機会コスト | 未約定の残りを最後に買っていたら払ったはずの金額 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import backtest
from h5i_db.quant import costs
import cookbook_utils as cu

SYMBOL = "ACME"
PARENT = 20_000.0
SLICES = 40

# %% [markdown]
# ## 1. 2営業日、そのうち売買できるのは1日だけ
#
# テープには2日分が入っています。1日目は履歴で、VWAP スケジュールに必要な出来高プロファイルの
# 推定に使います。2日目が、注文を執行する日です。
#
# この分割は事務処理ではありません。売買している当日の出来高に合わせた VWAP スケジュールは
# スケジュールではなく、答えを知っている予測です。バックテストではあらゆるベンチマークに勝ち、
# 本番では何にも勝ちません。

# %%
quotes = cu.make_quotes(symbols=[SYMBOL], days=2, quotes_per_day=2_500, seed=17)
tape = cu.make_equity_tape(quotes, action="set", print_every=2, print_size=300.0)
for name, table in tape.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")

prints = tape["trades"].to_pandas()
prints["session"] = prints["ts_init"].dt.date
history, live = sorted(prints["session"].unique())
print(f"\nprofile session {history}, execution session {live}")
tape["trades"].to_pandas().head(3)

# %%
db = h5i_db.Database(cu.fresh_db("04_execution_algorithms"), create=True)
for name, table in tape.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="two-session tape")
db.snapshot("tape-v1", tables=list(tape), note="Market data every algorithm reads")

book = tape["book_deltas"].to_pandas()
touch = (
    book[book["action"].isin(["set", "snapshot"])]
    .pivot_table(index="ts_init", columns="side", values="price", aggfunc="last")
    .ffill()
    .dropna()
)
touch["mid"] = (touch["buy"] + touch["sell"]) / 2
session_touch = touch[touch.index.date == live]
ARRIVAL = session_touch.index[0]
arrival_mid = float(session_touch["mid"].iloc[0])
final_mid = float(session_touch["mid"].iloc[-1])
print(f"arrival {ARRIVAL}  mid {arrival_mid:.4f}")
print(f"close   {session_touch.index[-1]}  mid {final_mid:.4f}")
print(f"the market moved {10_000 * (final_mid / arrival_mid - 1):.1f} bp while the order was worked")

# %% [markdown]
# ## 2. スケジュール3つと、反応1つ
#
# TWAP と VWAP は *スケジュール* です。計画全体が場が始まる前に決まっているので、どちらも
# ふつうのシグナルテーブルになります。POV はまだプリントされていない出来高への *反応* なので、
# コールバックにする必要があります。アルゴリズムを表現できるいちばん単純な境界を選ぶのが
# 原則で、POV は単純な境界では足りなくなる場所です。

# %%
grid = pd.date_range(
    session_touch.index[0], session_touch.index[-1], periods=SLICES + 1, inclusive="left"
)
volume_profile = (
    prints[prints["session"] == history]
    .assign(bucket=lambda frame: pd.cut(
        frame["ts_init"],
        bins=pd.date_range(
            frame["ts_init"].min(), frame["ts_init"].max(), periods=SLICES + 1
        ),
        labels=False,
        include_lowest=True,
    ))
    .groupby("bucket")["size"]
    .sum()
    .reindex(range(SLICES), fill_value=0.0)
)
weights = volume_profile / volume_profile.sum()
schedule = pd.DataFrame(
    {
        "slice_ts": grid,
        "twap": PARENT / SLICES,
        "vwap": (weights.to_numpy() * PARENT).round(),
    }
)
print(f"TWAP slice {schedule['twap'].iloc[0]:,.0f} shares")
print(f"VWAP slices {schedule['vwap'].min():,.0f} to {schedule['vwap'].max():,.0f} shares")
schedule.head()

# %%
backtest.create_signal_table(db)
for algorithm in ("twap", "vwap"):
    rows = [
        {
            "ts": row.slice_ts.floor("us").to_pydatetime() + dt.timedelta(microseconds=1),
            "instrument_id": SYMBOL,
            "side": "buy",
            "quantity": float(getattr(row, algorithm)),
            "kind": "market",
            "tag": algorithm,
        }
        for row in schedule.itertuples()
        if getattr(row, algorithm) > 0
    ]
    table = backtest.signal_table(rows)
    db.create_table(f"signals_{algorithm}", table.schema, time_column="ts")
    db.append(f"signals_{algorithm}", table, note=f"{algorithm} schedule")

immediate = backtest.signal_table(
    [
        {
            "ts": ARRIVAL.floor("us").to_pydatetime() + dt.timedelta(microseconds=1),
            "instrument_id": SYMBOL,
            "side": "buy",
            "quantity": PARENT,
            "kind": "market",
            "tag": "immediate",
        }
    ]
)
db.create_table("signals_immediate", immediate.schema, time_column="ts")
db.append("signals_immediate", immediate, note="the whole parent, at once")
db.snapshot("plans-v1", tables=["signals_twap", "signals_vwap", "signals_immediate"])
db.tables()


# %%
class ParticipateOfVolume(backtest.EventStrategy):
    """Take a fixed share of every print until the parent is done."""

    def __init__(self, parent=PARENT, participation=0.25, start=None):
        self.remaining = parent
        self.participation = participation
        self.start = start
        self.sent = 0

    def on_event(self, context, event):
        if event.get("kind") != "trade" or self.remaining <= 0:
            return None
        if self.start is not None and event["ts_init"] < self.start:
            return None
        quantity = min(self.remaining, round(event["size"] * self.participation))
        if quantity <= 0:
            return None
        self.sent += 1
        return {
            "action": "submit",
            "client_order_id": f"pov-{self.sent}",
            "instrument_id": SYMBOL,
            "side": "buy",
            "quantity": float(quantity),
            "tag": "pov",
        }

    def on_fill(self, context, event):
        self.remaining -= event["quantity"]
        return None


# %% [markdown]
# ## 3. 4つとも実行する
#
# 同じ板、同じ手数料、同じピン。違うのは、いつ株を求めたかだけです。

# %%
CASH = 10_000_000.0
COMMON = dict(
    execution=backtest.ExecutionConfig(fee_kind="proportional", fee_rate=0.0002),
    risk=backtest.RiskConfig(max_order_quantity=PARENT),
    output=backtest.OutputConfig(equity_interval_nanos=300_000_000_000),
)


def run_schedule(run_id, signals):
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=run_id,
            portfolio=backtest.PortfolioConfig(starting_cash=CASH),
            data=backtest.DataConfig(signals=signals, snapshot="tape-v1"),
            **COMMON,
        ),
    )


def run_reaction(run_id, strategy):
    return backtest.run_strategy(
        db,
        run_id,
        strategy,
        strategy_id="cookbook.ParticipateOfVolume:v1",
        starting_cash=CASH,
        data=backtest.DataConfig(snapshot="tape-v1"),
        **COMMON,
    )


results = {
    "immediate": run_schedule("algo-immediate", "signals_immediate"),
    "twap": run_schedule("algo-twap", "signals_twap"),
    "vwap": run_schedule("algo-vwap", "signals_vwap"),
    "pov": run_reaction("algo-pov", ParticipateOfVolume(start=int(ARRIVAL.value) + 1_000)),
}
for label, result in results.items():
    print(f"{label:<10} orders {result['orders']:>4}  fills {result['fills']:>4}")

# %% [markdown]
# ## 4. インプリメンテーション・ショートフォールの両半分
#
# `quant.costs` は約定した側を測ります。各約定を到着時の仲値と比べ、数量で加重するので、
# デスクが1株あたり実際に払った金額になります。どのコストモデルも報告しないもう半分が残りで、
# ここでは引けの仲値で値段をつけます。終わらせるために払わなければならなかった価格です。

# %%
def shortfall(result) -> dict:
    fills = result.fills.to_pandas()
    filled = float(fills["quantity"].sum())
    samples = [
        costs.SlippageSample(
            direction=1,
            fill_price=float(row.price),
            reference_price=arrival_mid,
            quantity=float(row.quantity),
        )
        for row in fills.itertuples()
    ]
    execution_cost = costs.implementation_shortfall(samples) * filled if samples else 0.0
    unfilled = PARENT - filled
    opportunity_cost = unfilled * (final_mid - arrival_mid)
    return {
        "filled": filled,
        "completion": filled / PARENT,
        "avg price": float((fills["price"] * fills["quantity"]).sum() / filled),
        "execution cost ($)": execution_cost,
        "opportunity cost ($)": opportunity_cost,
        "total shortfall ($)": execution_cost + opportunity_cost,
        "shortfall (bp of parent)": 10_000
        * (execution_cost + opportunity_cost)
        / (PARENT * arrival_mid),
    }


report = pd.DataFrame({label: shortfall(result) for label, result in results.items()}).T
report.round(3)

# %% [markdown]
# コストは、到着時の仲値より高く払ったときに正になります。ですから負の値は、買い手に有利に
# 動いた1日を意味します。
#
# 2つのコスト列は互いに突き合わせて読みます。親注文を一度に出すと表示されていた数量を食べて
# 止まるので、執行コストはごくわずかで、ショートフォールのほぼ全部が手に入らなかったポジション
# です。最後まで終わったのは POV だけなので、機会コストはゼロで、コストのすべてを払った価格が
# 背負っています。
#
# この日は仲値が下がっていき、買い手は遅いほど得をします。順位もそう言っています。それがこの
# 表の発見です。1日だけの順位が測っているのは、アルゴリズムよりドリフトです。執行の研究が
# 数百本の親注文でショートフォールを平均するのはまさにこのためで、1日だけの比較は、会計の
# やり方のデモとして読むべきで、アルゴリズムの優劣の証拠として読むべきではありません。
#
# 一般化できるのは表の形のほうです。VWAP に対する平均価格だけを引用する執行レポートは反証
# できません。平均価格がいちばん悪かったアルゴリズムが、唯一終わったアルゴリズムかもしれない
# からです。

# %%
assert report.loc["immediate", "completion"] < 0.2, "the fixture stopped being illiquid"
assert report.loc["pov", "completion"] > report.loc["immediate", "completion"]
assert abs(report.loc["pov", "opportunity cost ($)"]) < 1e-6, "a finished order has no remainder"
print(f"lowest total shortfall  {report['total shortfall ($)'].idxmin()}")
print(f"highest total shortfall {report['total shortfall ($)'].idxmax()}")
print(f"completion range        {report['completion'].min():.1%} to {report['completion'].max():.1%}")

# %% [markdown]
# ## 5. 参加カーブ
#
# 執行の形こそがアルゴリズムです。累積株数を描くと、スケジュールと反応の違いがはっきりします。
# TWAP は作り方からして直線、VWAP は前日の忙しい時間帯へ曲がり、POV は相場が動いたときにだけ
# 進みます。

# %%
fig, ax = plt.subplots(figsize=(9, 4.5))
for label, result in results.items():
    fills = result.fills.to_pandas().sort_values("ts")
    if fills.empty:
        continue
    ax.step(fills["ts"], fills["quantity"].cumsum(), where="post", linewidth=1.6, label=label)
ax.axhline(PARENT, color="black", linestyle="--", linewidth=0.8, label="parent order")
ax.set_title("Shares executed through the session")
ax.set_xlabel("Simulated time")
ax.set_ylabel("Cumulative shares")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. このフィクスチャで分からないこと
#
# ここでの板は外生的です。こちらが買っているからといって広がりませんし、表示数量を丸ごと取った
# スライスがあっても、次の気配では何事もなかったように戻ります。ですからこれらの実行は
# *タイミング* のリスクと *厚み* の制約は正直に値付けし、マーケットインパクトはゼロとして
# 値付けします。
#
# これはこのエンジンに固有の話ではありません。記録データに対するあらゆるバックテストの性質です。自分が
# そこにいなかったら板がどうなっていたかを、リプレイは知りようがありません。取れる対応は2つ。
# 自分の約定からインパクトモデルを推定して課金するか（レシピ 04/11）、仮定が擁護できる程度に
# スライスを小さく保ち、そう書くかです。

# %%
displayed_asks = (
    book[(book["side"] == "sell") & (book["action"].isin(["set", "snapshot"]))]
    [["ts_init", "size"]]
    .sort_values("ts_init")
)

consumed = []
for label, result in results.items():
    fills = result.fills.to_pandas()
    if fills.empty:
        continue
    merged = pd.merge_asof(
        fills[["ts", "quantity"]].sort_values("ts"),
        displayed_asks,
        left_on="ts",
        right_on="ts_init",
    )
    consumed.append(
        {
            "algorithm": label,
            "median slice / displayed": (merged["quantity"] / merged["size"]).median(),
            "max slice / displayed": (merged["quantity"] / merged["size"]).max(),
        }
    )
pd.DataFrame(consumed).round(3)

# %% [markdown]
# ## まとめ
#
# - スケジュールはデータ、反応はコードです。TWAP と VWAP はシグナルテーブルで、POV はまだ
#   起きていないプリントに依存するのでコールバックが要ります。
# - VWAP のプロファイルは、売買する当日とは別の営業日から取らなければなりません。さもないと
#   スケジュールは答えの入った予測になります。
# - インプリメンテーション・ショートフォールには2つの半分があります。約定した側だけを報告
#   すると、売買しないアルゴリズムが得をします。
# - 成行注文は表示数量を取り、残りはキャンセルされます。完了率は仮定ではなく結果です。
# - `quant.costs.implementation_shortfall` は数量で加重するので、小さな約定をいくら並べても、
#   コストを決めた1本の大きな約定を打ち消せません。
# - 記録データに対するリプレイは、タイミングと厚みには値をつけ、インパクトはゼロと置きます。
#   そう書くか、推定するかしてください。

# %%
db.close()
