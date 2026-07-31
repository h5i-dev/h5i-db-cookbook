# %% [markdown]
# # マーケットメイク：在庫、レイテンシ、そして轢かれること
#
# このセクションの他のレシピは流動性を取る側です。マーケットメイカーは出す側です。仕事の
# 性質がまるで違います。予測はしません。エッジは自分が提示する2つの価格の差であり、リスクは
# 買ったばかりのものを持っている間に相場が動くことです。
#
# そのためマーケットメイクは、ベクトル化バックテストではまったく表現できない唯一の戦略です。
# 損益は、置いた指値のどれが、どの順番で叩かれたか、そして叩かれそうなものをどれだけ速く
# 引っ込められたかで決まります。どれもイベントループの性質です。
#
# このレシピでは、Python コールバックとして気配提示の戦略を作り、生き残れるかを決める3つを
# 測ります。在庫のコントロール、狙った約定と轢かれた約定の内訳、そして往復50ミリ秒の遅延が
# キャンセル・アンド・リプレイスに何をするかです。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | マーケットメイカー | ビッドとオファーの両方を出し、その差で稼ぐ参加者 |
# | 在庫（インベントリ） | 片側を約定させられた結果として抱えるポジション |
# | スキュー | 在庫を減らす側が有利になるように、両方の気配をずらすこと |
# | メイカー約定 | 置いてあった自分の指値が、相手のスプレッド越えで叩かれた約定 |
# | テイカー約定 | 意図の有無にかかわらず、自分の注文がスプレッドを越えた約定 |
# | 逆選択 | 価格が自分に不利に動く直前に、まさにそのタイミングで約定させられること |
# | マークアウト | 約定から一定時間後の仲値。逆選択の大きさを測るのに使う |
# | アメンド | キャンセルして出し直さずに、生きている注文の価格をその場で変えること |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

SYMBOL = "ACME"
TICK = 0.01
MILLISECOND = 1_000_000

# %% [markdown]
# ## 1. 気配提示の戦略が実際に読めるテープ
#
# コールバックの契約がデータの形を決めます。戦略には板 *デルタ* の価格・サイド・数量が
# 渡されますが、板 *スナップショット* は各サイドの価格帯の本数しか渡されません。最良気配が
# どこかを知る必要のある戦略には、したがってデルタのフィードが必要です。
# `cu.make_equity_tape(action="set")` がそれを作ります。銘柄ごとの最初のスナップショットに続き、
# 気配が動くたびに `delete` と `set` の組が並びます。
#
# プリントは気配とは別に生成せず、気配から導きます。ですから約定は必ず、板が
# 表示していた価格で起きます。キューを考慮した約定はそこに依存します。ビッドでのプリントは
# 表示されていたビッド数量を消費し、その後ろに並んだ注文が前へ進みます。

# %%
quotes = cu.make_quotes(symbols=[SYMBOL], days=1, quotes_per_day=3_000, seed=11)
tape = cu.make_equity_tape(
    quotes, action="set", print_every=2, print_size=400.0, tick_size=TICK
)
for name, table in tape.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")
tape["book_deltas"].to_pandas().head(6)

# %% [markdown]
# 気配はまとまって届きます。これが後で効いてきます。更新間隔の中央値こそ、キャンセルが
# 勝たなければならない相手だからです。

# %%
gaps = quotes.to_pandas()["ts"].diff().dt.total_seconds().dropna()
print(f"median gap between quotes   {gaps.median():.2f}s")
print(f"share of gaps under 50ms    {(gaps < 0.05).mean():.1%}")
print(f"share of gaps under 1s      {(gaps < 1).mean():.1%}")

db = h5i_db.Database(cu.fresh_db("04_market_making_inventory"), create=True)
for name, table in tape.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="synthetic delta tape")
db.snapshot("tape-v1", tables=list(tape), note="The book every quoting run reads")
db.tables()

# %% [markdown]
# ## 2. 戦略
#
# 判断は4つあり、どれも気配提示の戦略がつまずく場所です。
#
# **デルタから最良気配を追う。** どのデルタを見たかを知っているのは戦略自身だけなので、
# 最良ビッドと最良オファーは戦略が自分で保持します。
#
# **絶対に越えない。** 古くなった板の片側から計算した気配は、反対側を越えてしまいます。
# ビッドをオファー未満に、オファーをビッド超に留める処理は、細かい改良ではありません。
# これがないと「マーケットメイカー」は、稼いでいるつもりのスプレッドを払うことになります。
#
# **在庫でスキューさせる。** 在庫がロングならビッドを遠ざけ、オファーは最良気配に置きます。
# 次の約定が、ポジションをフラットにする側になりやすくなります。
#
# **リクオートは出し直しか、アメンドか。** 両方を書いてあります。遅延ゼロでは違いが見えず、
# 50ミリ秒では決定的になるからです。

# %%
class InventoryMaker(backtest.EventStrategy):
    def __init__(self, quote_size=100.0, skew_shares=300.0, requote="replace"):
        self.quote_size = quote_size
        self.skew_shares = skew_shares
        self.requote = requote
        self.bid = None
        self.ask = None
        self.live = {}
        self.want = {}
        self.submitted = 0

    def on_event(self, context, event):
        if event.get("kind") != "book_delta" or event.get("action") != "set":
            return None
        if event["side"] == "buy":
            self.bid = event["price"]
        else:
            self.ask = event["price"]
        if self.bid is None or self.ask is None:
            return None

        inventory = sum(position["quantity"] for position in context["positions"])
        step = round(inventory / self.skew_shares) if self.skew_shares else 0
        target = {
            "buy": round(min(self.bid - max(step, 0) * TICK, self.ask - TICK), 4),
            "sell": round(max(self.ask - min(step, 0) * TICK, self.bid + TICK), 4),
        }

        commands = []
        for side, price in target.items():
            if self.want.get(side) == price:
                continue
            self.want[side] = price
            if side in self.live:
                if self.requote == "amend":
                    commands.append(
                        {
                            "action": "amend",
                            "client_order_id": self.live[side],
                            "limit_price": price,
                        }
                    )
                    continue
                commands.append({"action": "cancel", "client_order_id": self.live[side]})
            self.submitted += 1
            client_order_id = f"{side}-{self.submitted}"
            self.live[side] = client_order_id
            commands.append(
                {
                    "action": "submit",
                    "client_order_id": client_order_id,
                    "instrument_id": SYMBOL,
                    "side": side,
                    "quantity": self.quote_size,
                    "kind": "limit",
                    "limit_price": price,
                    "time_in_force": "gtc",
                    "tag": f"quote-{side}",
                }
            )
        return commands or None

    def on_fill(self, context, event):
        # A filled order is no longer live, and its price is no longer quoted.
        self.live.pop(event["side"], None)
        self.want.pop(event["side"], None)
        return None


# %% [markdown]
# ## 3. 1回の実行と、約定が語ること
#
# `queue_position=True` は L2 フィードの正直な読み方です。ある価格に並んだ注文は、そこに
# すでに表示されていた数量の後ろに付き、約定がそれを食い尽くしてから初めて約定します。
# 反対の仮定は、起きなかったメイカー約定を作り出します。

# %%
def quote_run(run_id, *, skew_shares=300.0, requote="replace", latency_nanos=None):
    strategy = InventoryMaker(skew_shares=skew_shares, requote=requote)
    result = backtest.run_strategy(
        db,
        run_id,
        strategy,
        strategy_id=f"cookbook.InventoryMaker:{requote}:{skew_shares:g}",
        starting_cash=100_000.0,
        data=backtest.DataConfig(snapshot="tape-v1"),
        execution=backtest.ExecutionConfig(
            fee_kind="proportional",
            fee_rate=0.0002,
            queue_position=True,
            latency_nanos=latency_nanos,
        ),
        risk=backtest.RiskConfig(
            max_order_quantity=500.0, max_abs_position=1_000.0, max_open_orders=4
        ),
        output=backtest.OutputConfig(equity_interval_nanos=60_000_000_000),
    )
    return strategy, result


strategy, base = quote_run("mm-base")
metrics = base["metrics"]
print(f"quotes sent      {strategy.submitted:,}")
print(f"fills            {base['fills']:,}")
print(f"  maker          {metrics['fills_maker']:,}")
print(f"  taker          {metrics['fills_taker']:,}")
print(f"cancelled unfilled {metrics['orders_cancelled_unfilled']:,}")
base.fills.to_pandas().head()

# %% [markdown]
# ここでのテイカー約定は判断の結果ではありません。戦略が引っ込める前に板が突き抜けた、
# 置きっぱなしの気配です。相場のほうから来ました。だから注文が越え、稼ぐはずのスプレッドを
# 払っています。*自分の* 気配のメイカー／テイカー内訳は、気配提示のバックテストが生む
# いちばん安い診断です。ベクトル化されたバックテストには、それを置く場所がありません。
#
# この実行は損をしますが、そうなるべきです。このテープはランダムウォークで、流動性を出した
# 見返りを払ってくれる情報を持った注文がありません。約定の3分の1で轢かれるメイカーには、
# 収入源が残っていないのです。

# %%
fills = base.fills.to_pandas()
split = fills.groupby(["side", "is_taker"]).agg(
    fills=("quantity", "size"), shares=("quantity", "sum"), avg_price=("price", "mean")
)
split.round(4)

# %% [markdown]
# ## 4. スキューは効いているのか
#
# 検証する主張は狭いものです。在庫に応じて気配をスキューさせれば、抱える在庫は減るはずで、
# その代金を約定数で払わされることもないはずだ、という主張です。在庫は再構成せず
# エクイティカーブのポジション評価額から読むので、答えは実行自身のテーブルから出てきます。

# %%
runs = {}
for label, skew in (("skew on", 300.0), ("skew off", 0.0)):
    runs[label] = quote_run(f"mm-{label.replace(' ', '-')}", skew_shares=skew)

rows = []
for label, (built, result) in runs.items():
    equity = result.equity.to_pandas()
    inventory = equity["position_value"]
    rows.append(
        {
            "run": label,
            "fills": result["fills"],
            "maker": result["metrics"]["fills_maker"],
            "ending equity": equity["equity"].iloc[-1],
            "max |inventory| ($)": inventory.abs().max(),
            "mean |inventory| ($)": inventory.abs().mean(),
        }
    )
pd.DataFrame(rows).round(2)

# %% [markdown]
# 約定数はほとんど変わらず、平均ポジションはおよそ半分になります。これがメイカーが実際に
# している取引です。スキューはアルファではありません。約定と約定のあいだにどれだけリスクを
# 抱えるかを選ぶ手段です。ピーク値のほうは弱い検証です。不利な連続が一度来れば、どんな気配
# 提示戦略もポジション上限に張り付きます。

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for label, (_, result) in runs.items():
    equity = result.equity.to_pandas()
    ax.plot(equity["ts"], equity["position_value"], linewidth=1.4, label=label)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Inventory carried through the session")
ax.set_xlabel("Simulated time")
ax.set_ylabel("Position value ($)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 5. 50ミリ秒
#
# レイテンシはふつう、入る価格へのコストとして語られます。気配提示の戦略にとっては、もっと
# 悪いことが起きます。まだ届いていないキャンセルは注文を生かしたままにするので、出し直した
# 気配は相場ではなく、会場の未約定注文数の上限にぶつかります。ここではその役を
# `max_open_orders=4` が担っています。

# %%
rows = []
for requote in ("replace", "amend"):
    for latency in (None, 50 * MILLISECOND):
        built, result = quote_run(
            f"mm-{requote}-{0 if latency is None else 50}ms",
            requote=requote,
            latency_nanos=latency,
        )
        equity = result.equity.to_pandas()
        rows.append(
            {
                "requote": requote,
                "latency": "0ms" if latency is None else "50ms",
                "fills": result["fills"],
                "maker": result["metrics"]["fills_maker"],
                "rejected for risk": result["metrics"]["orders_rejected_risk"],
                "ending equity": equity["equity"].iloc[-1],
            }
        )
latency_table = pd.DataFrame(rows)
latency_table.round(2)

# %% [markdown]
# 往復がゼロでなくなった瞬間、キャンセル・アンド・リプレイスは気配の大半をリスク拒否で失い、
# その場でのアメンドはひとつも失いません。これはチューニングの細部ではありません。実運用の
# 気配提示エンジンがアメンドを使う理由であり、バーの価格で約定させるバックテストからは
# まったく見えません。
#
# エクイティの列を、遅いほうが良いという主張として読まないでください。このメイカーはこの
# テープで損をしているので、いちばん気配を出さない実行がいちばん損を減らします。レイテンシは
# 商売を取り上げることで損益を改善したのであって、商売を良くしたわけではありません。

# %%
assert latency_table.loc[
    (latency_table["requote"] == "amend") & (latency_table["latency"] == "50ms"),
    "rejected for risk",
].iloc[0] == 0
worst = latency_table.loc[
    (latency_table["requote"] == "replace") & (latency_table["latency"] == "50ms"),
    "rejected for risk",
].iloc[0]
print(f"cancel-and-replace at 50ms lost {worst:,} orders to the open-order limit")

# %% [markdown]
# ## 6. 約定のあとで分かるコスト
#
# メイカーが提示するスプレッドは、稼ぐスプレッドではありません。商売を決める数字はマークアウト、
# つまり各約定のあとで仲値がどこへ行ったかです。買った直後に仲値が下がっていく約定は、
# 誰かが喜んで渡してくれた約定です。

# %%
book = tape["book_deltas"].to_pandas()
touch = (
    book[book["action"].isin(["set", "snapshot"])]
    .pivot_table(index="ts_init", columns="side", values="price", aggfunc="last")
    .ffill()
    .dropna()
)
touch["mid"] = (touch["buy"] + touch["sell"]) / 2
mids = touch.reset_index()[["ts_init", "mid"]].sort_values("ts_init")

maker_fills = fills[~fills["is_taker"]][["ts", "side", "quantity", "price"]].copy()
maker_fills = maker_fills.sort_values("ts")
marked = pd.merge_asof(maker_fills, mids, left_on="ts", right_on="ts_init")
for horizon in (10, 60):
    future = mids.copy()
    future["ts_init"] = future["ts_init"] - pd.Timedelta(seconds=horizon)
    marked = pd.merge_asof(
        marked, future.rename(columns={"mid": f"mid_{horizon}s"}),
        left_on="ts", right_on="ts_init", direction="forward", suffixes=("", f"_{horizon}"),
    )
    signed = marked["side"].map({"buy": 1.0, "sell": -1.0})
    marked[f"markout_{horizon}s"] = signed * (marked[f"mid_{horizon}s"] - marked["mid"])

summary = marked.groupby("side")[["markout_10s", "markout_60s"]].mean()
print(f"{len(marked):,} maker fills marked out")
summary.round(4)

# %% [markdown]
# この数字は1株あたりのドルとして、正なら「あとから見て良かった約定」という符号で読みます。
# 提示スプレッドが1ティックのとき、同じくらいの大きさのマークアウトは、スプレッドをそのまま
# 返していることを意味します。
#
# ここでは両サイドが対称ではありません。このテープは作り方からして情報を持った注文を含まない
# ので、この非対称が表すのは、誰が相手だったかよりも、この戦略がたまたまいつ気配を出していたかを
# 表しています。合成した板の正直な限界です。移せるのは仕組みのほうです。記録された実データの
# テープでは、サイド別のマークアウトこそが、提示スプレッドが収入かどうかを語る数字になります。

# %%
fig, ax = plt.subplots(figsize=(9, 4))
for side, group in marked.groupby("side"):
    ax.hist(group["markout_60s"].dropna(), bins=40, alpha=0.6, label=f"{side} fills")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("60-second markout of maker fills")
ax.set_xlabel("Signed mid move after the fill ($ per share)")
ax.set_ylabel("Fills")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - 気配提示の戦略にはデルタのフィードが要ります。コールバックにはデルタの価格が渡され、
#   スナップショットは価格帯の本数しか渡されません。
# - 気配は反対側の最良気配に対してクランプしてください。古い側から作った気配は越えてしまい、
#   自分の約定のメイカー／テイカー内訳がそれを教えてくれます。
# - 在庫スキューはリターンを足しません。リスクをポジションから約定率へ移すだけで、それこそが
#   メイカーが行うべき取引です。
# - レイテンシが攻撃するのは約定ではなく、キャンセルです。往復があると出し直した気配が未約定注文の
#   上限に積み上がるので、実運用の気配提示エンジンはその場でアメンドします。
# - マークアウトは逆選択に値段をつけます。提示スプレッドが収入なのか補助金なのかを語る数字です。
# - ここでの実行はすべて、ピン留めしたスナップショット上のコールバック戦略なので、どの結果も
#   自分のフォークから再現できます。

# %%
db.close()
