# %% [markdown]
# # フェイバリット・ロングショット・バイアスを取引する
#
# イベント契約で最も古くから記録されている歪みは、ロングショットが割高でフェイバリットが割安になる
# ことです。レシピ 05/02 ではこれをキャリブレーションの差として測りました。ここではポジションに変え、
# 手数料カーブに耐えるかを問います。まず価格帯ごとの、決着まで持ち切ったリターンを見ます。次に同じ
# バイアスの反対側を取る2つのエンジン実行を行います。最後に勝負を決める部分、つまりどの価格水準なら
# エッジがそこで取引する費用を上回るのかを調べます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | フェイバリット・ロングショット・バイアス | 安い契約が系統的に割高で、高い契約が割安になる歪み |
# | ロングショット／フェイバリット | 0.50 をはるかに下回る価格の契約と、はるかに上回る価格の契約 |
# | アスク | 売り手が受け入れる最安値。買い手が実際に払う価格 |
# | 決着まで持ち切る | 途中で手仕舞わず、決済されるまで契約を保有すること |
# | キャリブレーションの差 | 提示確率から実測頻度を引いた値。1契約あたりで表す |
# | 資本利益率 | 利益を投じた資金で割った値。だから価格水準が効いてくる |
# | 手数料カーブ | `p*(1-p)` に比例する手数料。資本に対する割合では `rate * (1-p)` |
# | しきい値のスイープ | 区切り値を変えて再実行すること。平らなら発見、尖っていれば当てはめたパラメータ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest

db = h5i_db.Database(cu.fresh_db("05_favorite_longshot_bias"), create=True)
FEE_RATE = 0.07
QUANTITY = 20.0

# %% [markdown]
# ## パネル
#
# `book_deltas` の1行は、1つのアトミックな板イベントの1つの価格帯です。このレシピが必要とするのは、
# 買い手が払う価格である YES のアスクと、決着だけです。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | 記録側への到着時刻。リプレイの順序 |
# | `instrument_id` | `string` | マーケット |
# | `outcome` | `uint16` | 0 = YES、1 = NO |
# | `side` | `string` | `buy` がビッド、`sell` がアスク |
# | `price` | `float64` | 0.001 刻みのグリッド上の確率 |
# | `size` | `float64` | その価格帯の契約数 |

# %%
panel = cu.make_prediction_markets(n_markets=180, steps=32, seed=11)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel load")
db.snapshot("panel-v1", tables=list(panel), note="favorite-longshot study")
print(f"{panel['book_deltas'].num_rows:,} book rows, {panel['resolutions'].num_rows} markets")
panel["book_deltas"].to_pandas().head()

# %% [markdown]
# ## 価格帯ごとの、決着まで持ち切ったリターン
#
# 判断の瞬間を固定し、そこでの YES のアスクを取って決着まで持ちます。ペイオフは YES が勝てば 1、
# そうでなければ 0 なので、資本利益率は `(payoff - ask) / ask` です。アスクでバケットに分けるのが
# 古典的な見せ方です。どのバケットも、同じくらい起こりそうに見えた契約のポートフォリオになります。

# %%
stamps = sorted({value.as_py() for value in panel["book_deltas"].column("ts_init")})
decision = stamps[8]
decision_ns = int(pd.Timestamp(decision, tz="UTC").value)
quotes = db.sql(
    f"""
    SELECT instrument_id, outcome,
           max(CASE WHEN side = 'buy'  THEN price END) AS bid,
           max(CASE WHEN side = 'sell' THEN price END) AS ask
    FROM h5i('book_deltas', 'panel-v1')
    WHERE ts_init = to_timestamp_nanos({decision_ns})
    GROUP BY instrument_id, outcome
    """
).to_pandas()
truth = cu.market_truth(panel).to_pandas()

yes = quotes[quotes.outcome == 0].merge(truth, on="instrument_id", validate="one_to_one")
yes["payoff"] = yes.yes_won.astype(float)
yes["gross_return"] = (yes.payoff - yes.ask) / yes.ask
yes["fee_per_contract"] = FEE_RATE * yes.ask * (1.0 - yes.ask)
yes["net_return"] = (yes.payoff - yes.ask - yes.fee_per_contract) / yes.ask
print(f"decision instant {decision}, {len(yes)} markets")

EDGES = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.88, 1.0]
yes["bucket"] = pd.cut(yes.ask, EDGES)
buckets = (
    yes.groupby("bucket", observed=True)
    .agg(
        n=("ask", "size"),
        mean_ask=("ask", "mean"),
        hit_rate=("payoff", "mean"),
        gross=("gross_return", "mean"),
        net=("net_return", "mean"),
    )
    .assign(edge_pp=lambda f: (f.hit_rate - f.mean_ask) * 100)
)
print(buckets.round(3).to_string())

# %% [markdown]
# 安いバケットを買うと損をし、高いバケットを買うと儲かり、符号は真ん中のどこかで反転します。
# `edge_pp` 列は、レシピ 05/02 と同じキャリブレーションの差を1契約あたりで表したものです。`gross` は
# それを価格で割った値なので、15セントの契約では小さな差が大きな損失率になります。

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
centres = buckets.mean_ask
width = 0.05
ax.bar(centres - width / 2, buckets.gross * 100, width=width, label="gross", color="#2c7fb8")
ax.bar(centres + width / 2, buckets.net * 100, width=width, label="net of fees", color="#c0392b")
ax.axhline(0.0, color="black", lw=0.8)
ax.set_title("Hold-to-resolution return on a long YES position, by price")
ax.set_xlabel("YES ask at the decision instant")
ax.set_ylabel("return on capital, %")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## そこで取引する費用
#
# 手数料は `rate * q * p * (1-p)` なので、フェイバリット側のエッジがある場所がちょうど最も安くなり
# ます。「資本に対する割合」で見ると `rate * (1-p)` となり、契約が高くなるほど単調に下がります。
# 2つの効果は同じ向きを指しています。このバイアスのフェイバリット側が、手の届く側なのです。

# %%
levels = pd.DataFrame({"price": [0.10, 0.25, 0.50, 0.75, 0.90]})
levels["fee_per_contract"] = FEE_RATE * levels.price * (1 - levels.price)
levels["fee_pct_of_capital"] = levels.fee_per_contract / levels.price * 100
print(levels.round(4).to_string(index=False))

# %% [markdown]
# ## エンジンで両側を取る
#
# 同じピンの上での2回の実行です。一方はフェイバリットを買い、もう一方はロングショットを買い、ほかは
# 何も変えません。シグナルには判断に使った気配の1マイクロ秒あとの時刻を押すので、どの注文も選んだ
# ときのアスクで約定します。

# %%
FAVORITE, LONGSHOT = 0.75, 0.30
submit = decision + dt.timedelta(microseconds=1)


def signals_for(members: pd.DataFrame, tag: str) -> pa.Table:
    rows = [
        {
            "ts": submit,
            "instrument_id": row.instrument_id,
            "outcome": 0,
            "side": "buy",
            "quantity": QUANTITY,
            "tag": tag,
        }
        for row in members.itertuples()
    ]
    return backtest.signal_table(rows).sort_by([("ts", "ascending")])


legs = {
    "favorites": yes[yes.ask >= FAVORITE],
    "longshots": yes[yes.ask <= LONGSHOT],
}
for name, members in legs.items():
    table = signals_for(members, name)
    db.create_table(f"signals_{name}", table.schema, time_column="ts")
    db.append(f"signals_{name}", table)
    print(f"{name:10} {len(members):>3} markets, mean ask {members.ask.mean():.3f}")

# %%
def run(name: str, fee_kind: str | None) -> backtest.BacktestResult:
    execution = (
        backtest.ExecutionConfig(fee_kind=fee_kind, fee_rate=FEE_RATE)
        if fee_kind
        else backtest.ExecutionConfig()
    )
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=f"flb-{name}-{fee_kind or 'gross'}",
            data=backtest.DataConfig(signals=f"signals_{name}", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=execution,
            metadata={"study": "favorite-longshot", "leg": name},
        ),
    )


def account(result: backtest.BacktestResult) -> dict[str, float]:
    """Positions are held to resolution, so the result is in settlement_pnl."""
    positions = result.positions.to_pandas()
    fills = result.fills.to_pandas()
    settled = float(positions.settlement_pnl.fillna(0.0).sum())
    fees = float(result.summary()["commissions"])
    capital = float((fills.price * fills.quantity).sum())
    # realized_pnl already nets out commissions, so total = realized + settled.
    # These entries are held to resolution, so realized is exactly -fees and the
    # two agree; a rule that closes positions would make them differ.
    net = float(result.summary()["realized_pnl"]) + settled
    return {
        "fills": len(fills),
        "capital": capital,
        "settled": settled,
        "fees": fees,
        "net": net,
        "net_return_pct": net / capital * 100 if capital else float("nan"),
    }


report = {}
for name in legs:
    for fee_kind in (None, "kalshi"):
        label = f"{name} / {'gross' if fee_kind is None else 'net'}"
        report[label] = account(run(name, fee_kind))
print(pd.DataFrame(report).T.round(2).to_string())

# %% [markdown]
# フェイバリット側のレグはグロスで正、手数料を引いても正のままです。ロングショット側のレグはグロスで
# 負、手数料を引くとさらに悪化します。手数料の取り分がどれだけ非対称かが、上の表の要点です。ロング
# ショット側のレグは「投じた資本に対して」大きな手数料を払うので、間違えたときの費用が高くつきます。

# %% [markdown]
# ## エッジは実際どこにあるのか
#
# しきい値を1つ決めるのは選択であり、バックテストが道を誤るのは選択のところです。スイープすれば、
# その発見が紙一重なのか平らな台地なのかが分かります。ここにあるのは1つのパネルの1つの判断瞬間なので、
# 形のほうを結果として扱い、水準はノイズとみなしてください。レシピ 05/05 で、その区別に数値を与えます。

# %%
sweep = []
for threshold in np.arange(0.40, 0.92, 0.04):
    members = yes[yes.ask >= threshold]
    if len(members) < 5:
        continue
    sweep.append(
        {
            "threshold": round(float(threshold), 2),
            "markets": len(members),
            "mean_ask": members.ask.mean(),
            "gross_pct": members.gross_return.mean() * 100,
            "net_pct": members.net_return.mean() * 100,
        }
    )
sweep = pd.DataFrame(sweep)
print(sweep.round(2).to_string(index=False))

# %% [markdown]
# ## ボラティリティは p(1-p) に比例する
#
# 同じ算術から出てくる実務上の帰結です。0.05 の契約は 0.50 の契約ほど大きくは動けません。下がゼロで
# 抑えられているからです。これを無視したポジションサイジングは、板の真ん中でリスクを取りすぎ、両端で
# 取らなさすぎることになります。
#
# スキャンは `expiration_ns` で止めます。パネルは結果が観測可能になったあとも気配を出し続け、その
# スナップショットは 1.00 や 0.00 の近くに張り付きます。含めてしまうと、決着による跳ねを価格変動として
# 採点することになります。データの中で最大の「変動」でありながら、どのボラティリティ推定にも属さない
# ものです。

# %%
expiry_ns = int(db.sql("SELECT max(expiration_ns) AS e FROM instruments").to_pandas()["e"][0])
paths = db.sql(
    f"""
    SELECT instrument_id, ts_init,
           (max(CASE WHEN side = 'buy'  THEN price END)
          + max(CASE WHEN side = 'sell' THEN price END)) / 2 AS mid
    FROM h5i('book_deltas', 'panel-v1')
    WHERE outcome = 0 AND ts_init <= to_timestamp_nanos({expiry_ns})
    GROUP BY instrument_id, ts_init
    ORDER BY instrument_id, ts_init
    """
).to_pandas()
paths["step_move"] = paths.groupby("instrument_id").mid.diff()
paths["level"] = paths.groupby("instrument_id").mid.transform("mean").round(1).clip(0.1, 0.9)
vol = (
    paths.dropna(subset=["step_move"])
    .groupby("level")
    .agg(observations=("step_move", "size"), step_vol_bp=("step_move", lambda s: s.std() * 10_000))
    .assign(sqrt_p1mp=lambda f: np.sqrt(f.index * (1 - f.index)))
)
vol["vol_over_sqrt"] = vol.step_vol_bp / vol.sqrt_p1mp
print(vol.round(1).to_string())
print("\nthe last column is roughly flat, which is what p*(1-p) scaling looks like")

# %% [markdown]
# ## まとめ
#
# - フェイバリット・ロングショット・バイアスは符号つきのキャリブレーションの差として現れ、ポジション
#   に翻訳しても生き残る。このパネルでは、フェイバリットのロングは手数料後も儲かり、ロングショットの
#   ロングは損をした。
# - 価格で割ること。3ポイントの差は、15セントの契約なら20%のリターン、85セントの契約なら3%の
#   リターンになる。そして資本に対する手数料の割合は `rate * (1-p)` で、逆向きに動く。
# - 信じる前にしきい値をスイープすること。平らな台地は発見であり、尖った山は当てはめたパラメータである。
# - 価格のボラティリティは `sqrt(p*(1-p))` に比例する。定数ではなくこれを基準にサイジングする。
# - ここで働いた h5i-db の機能。4回の実行の背後にある1つの名前付きスナップショットのおかげで、
#   グロスとネット、フェイバリットとロングショットの比較は、調べている項目だけが違う形になった。
#   そして `bt_fills` が、意図した資本ではなく実際に投じた資本を教えてくれた。

# %%
db.close()
