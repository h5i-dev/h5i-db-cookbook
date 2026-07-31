# %% [markdown]
# # 短期リバーサルと、必要だったエッジ
#
# 1年の時間軸では、勝った銘柄が勝ち続けます。数日の時間軸では、教科書はそれを返すと言います。
# 同業に比べて大きく下げた銘柄は跳ね返る。値動きの一部が、何かを知っている人からではなく、
# 売買せざるをえない人によるものだからです。
#
# 2018年以降の大型株30銘柄では、その効果はありません。以下で試すどの形成期間でも、情報係数は
# ゼロと区別できません。これは大型株のリバーサルについて20年ほど報告されてきた内容と一致します。
# 効果は小型株に残っていて、しかも薄れてきました。
#
# そこでこのレシピは、もっと役に立つことをします。戦略の回転率を正直に測り、現実的なコストを
# 超えるために必要だったエッジを計算します。観測したリターンより先に見るべき必要グロスリターンという
# この数字が、探し続ける価値があるかを教えてくれます。そして短期の研究がいちばんよく飛ばす
# 計算でもあります。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 平均回帰（リバーサル） | 短期の値動きが続くほうではなく、戻るほうに賭けること |
# | クロスセクショナル | 自分の過去と比べず、その日の他の銘柄と比べること |
# | ドルニュートラル | ロングとショートを同額にして、相場全体の方向を打ち消すこと |
# | Zスコア | 平均から標準偏差いくつ分離れているかを表す値 |
# | ノートレード・バンド | 目標の変化が売買代金に見合わないうちはポジションを動かさないこと |
# | 損益分岐コスト | 純リターンがちょうどゼロになる売買コスト |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

FORMATION = 5
ANNUALIZE = float(np.sqrt(252))

# %% [markdown]
# ## 1. シグナル
#
# 各銘柄の5日リターンを、その日のユニバース平均で引いて符号を反転させます。相対的にいちばん
# 悪かった銘柄が、いちばん大きな正のスコアになります。平均を引く操作が、相場のタイミングを
# 当てる賭けを相対的な賭けに変えている部分で、`cs_demean` がエンジン側でそれを行います。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("alpha_mean_reversion"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="Everything here reads this cut")

pinned = db.table("prices", snapshot="prices-v1")
previous = sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")
formation = sql_expr(f"lag(adj_close, {FORMATION})").over(
    partition_by="symbol", order_by="ts"
)
scored = (
    pinned.with_columns(previous=previous, formation=formation)
    .with_columns(
        ret=col("adj_close") / col("previous") - 1,
        window_return=col("adj_close") / col("formation") - 1,
    )
    .filter(col("ret").is_not_null() & col("window_return").is_not_null())
    .with_columns(relative=col("window_return").cs_demean(partition_by="ts"))
    .with_columns(score=col("relative") * -1.0)
)
signal_frame = scored.select(
    ts=col("ts"), asset=col("symbol"), ret=col("ret"), factor=col("score")
)
panel_frame = signal_frame.sort(["ts", "asset"]).to_pandas()
print(f"{len(panel_frame):,} symbol-days")
panel_frame.head(3).set_index("ts").round(4)

# %% [markdown]
# ## 2. そもそも存在するのか
#
# レシピ 06/02 と同じパネルの仕組みで、生のファクターを採点します。もっともらしい形成期間を
# 一度にすべて試します。1つの期間だけ調べて報告するのは、t 値1の数字が戦略に化ける典型的な
# やり方です。

# %%
price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))


def reversal_panel(window: int):
    lag = sql_expr(f"lag(adj_close, {window})").over(partition_by="symbol", order_by="ts")
    frame = (
        pinned.with_columns(previous=previous, formation=lag)
        .with_columns(
            ret=col("adj_close") / col("previous") - 1,
            window_return=col("adj_close") / col("formation") - 1,
        )
        .filter(col("ret").is_not_null() & col("window_return").is_not_null())
        .with_columns(relative=col("window_return").cs_demean(partition_by="ts"))
        .with_columns(score=col("relative") * -1.0)
        .select(ts=col("ts"), asset=col("symbol"), factor=col("score"))
    )
    return quant.build_panel(db, frame, price_frame, periods=(1,), quantiles=5)


scan = []
for window in (1, 2, 3, 5, 10, 21):
    scanned = reversal_panel(window)
    row = scanned.ic_decay().to_pandas().iloc[0]
    buckets = scanned.quantile_returns().to_pandas()
    scan.append(
        {
            "formation": window,
            "mean IC": float(row["mean_ic"]),
            "t-stat": float(row["t_stat"]),
            "q5 - q1 (bp)": float(
                (buckets.iloc[-1]["mean_1"] - buckets.iloc[0]["mean_1"]) * 10_000
            ),
        }
    )
pd.DataFrame(scan).round(4)

# %% [markdown]
# t 値1を超えるものはひとつもなく、上位マイナス下位のスプレッドは隣り合う期間で符号が変わります。
# これが「効果がない」の見え方です。きれいなゼロは出ません。互いに一致しない小さな数字が並ぶだけです。
#
# そこで以降は、売買すべきかを問うのをやめて、何が成り立っていれば良かったのかを問います。

# %%
panel = reversal_panel(FORMATION)
quantiles = panel.quantile_returns().to_pandas()
top = quantiles.iloc[-1]["mean_1"]
bottom = quantiles.iloc[0]["mean_1"]
print(f"top bucket, one day     {top:+.5f} ({top * 10_000:+.1f} bp)")
print(f"bottom bucket, one day  {bottom:+.5f} ({bottom * 10_000:+.1f} bp)")
print(f"spread                  {(top - bottom) * 10_000:+.1f} bp per day")

# %% [markdown]
# ## 3. ドルニュートラルなポートフォリオ
#
# ウェイトはスコアに比例させ、もう一度平均を引いてドルニュートラルにし、グロスエクスポージャーが
# 1になるよう正規化します。ポジションは終値で決めて翌日保有します。

# %%
scores = panel_frame.pivot(index="ts", columns="asset", values="factor")
returns = panel_frame.pivot(index="ts", columns="asset", values="ret")
centred = scores.sub(scores.mean(axis=1), axis=0)
weights = centred.div(centred.abs().sum(axis=1), axis=0).fillna(0.0)

held = weights.shift(1).fillna(0.0)
gross = (held * returns).sum(axis=1)
turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
print(f"gross exposure        {held.abs().sum(axis=1).mean():.2f}x")
print(f"net exposure          {held.sum(axis=1).mean():+.4f}x")
print(f"daily turnover        {turnover.mean():.2f} of book")
print(f"gross Sharpe          {gross.mean() / gross.std() * ANNUALIZE:.2f}")
print(f"gross annual return   {gross.mean() * 252:.2%}")

# %% [markdown]
# 回転率が1に近いということは、毎日ポートフォリオを入れ替えているということです。問題はこの
# 数字ひとつに表れています。日次回転率100%なら、10ベーシスポイントのコストは年率でおよそ25%です。

# %%
costs = pd.DataFrame(
    [
        {
            "cost (bp)": bps,
            "annual cost": turnover.mean() * 252 * bps / 10_000,
            "net annual return": (gross - turnover * bps / 10_000).mean() * 252,
            "net Sharpe": (gross - turnover * bps / 10_000).mean()
            / (gross - turnover * bps / 10_000).std()
            * ANNUALIZE,
        }
        for bps in (0.0, 1.0, 2.0, 5.0, 10.0)
    ]
)
costs.round(4)

# %% [markdown]
# ## 4. 勝負を決める数字
#
# 毎日ほとんどを入れ替える戦略で役に立つ統計量は、シャープレシオより必要グロスリターン
# です。回転率×コストで決まります。この線より下では、話がどれだけ良くても戦略は損をします。
# そしてこの線はシグナルにいっさい依存しません。
#
# 比較するコストのほうも当て推量ではありません。レシピ 04/11 は実行の約定そのものからコストを
# 推定しますし、レシピ 04/08 は数ベーシスポイントのスプレッド仮定が、もっと遅い戦略に何をするかを
# 示しています。

# %%
required = pd.DataFrame(
    [
        {
            "cost (bp)": bps,
            "required gross (bp/day)": turnover.mean() * bps,
            "required gross (%/yr)": turnover.mean() * bps / 10_000 * 252,
            "observed gross (bp/day)": gross.mean() * 10_000,
        }
        for bps in (1.0, 2.0, 5.0, 10.0)
    ]
)
required.round(3)

# %%
print(f"observed gross return      {gross.mean() * 10_000:+.2f} bp per day")
print(f"turnover per day           {turnover.mean():.2f}")
print(f"required at 5bp            {turnover.mean() * 5.0:.2f} bp per day")
print(f"shortfall                  {turnover.mean() * 5.0 - gross.mean() * 10_000:.2f} bp per day")
print("\nA signal has to clear that line before its Sharpe means anything.")

# %% [markdown]
# ## 5. 回転率を買い下げる
#
# 必要リターンの線を動かすレバーは回転率で、動かす標準的な方法がノートレード・バンドです。
# 目標が売買代金に見合うだけ動かないかぎり、ポジションはそのままにします。理想のポートフォリオ
# からのトラッキングエラーを払うかわりに、コストのハードルも下げます。

# %%
def banded(threshold: float) -> dict:
    current = np.zeros(weights.shape[1])
    rows = []
    for _ts, target in weights.iterrows():
        move = target.to_numpy() - current
        traded = np.where(np.abs(move) > threshold, target.to_numpy(), current)
        rows.append(traded)
        current = traded
    positions = pd.DataFrame(rows, index=weights.index, columns=weights.columns)
    turned = (positions - positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    raw = (positions.shift(1).fillna(0.0) * returns).sum(axis=1)
    net = raw - turned * 5.0 / 10_000
    return {
        "band": threshold,
        "daily turnover": float(turned.mean()),
        "required at 5bp (bp/day)": float(turned.mean() * 5.0),
        "observed gross (bp/day)": float(raw.mean() * 10_000),
        "net Sharpe at 5bp": float(net.mean() / net.std() * ANNUALIZE),
    }


bands = pd.DataFrame([banded(threshold) for threshold in (0.0, 0.002, 0.005, 0.01, 0.02)])
bands.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
axes[0].plot(bands["band"], bands["daily turnover"], marker="o", linewidth=1.6)
axes[0].set_title("Turnover against the no-trade band")
axes[0].set_xlabel("Band (weight)")
axes[0].set_ylabel("Daily turnover")
axes[1].plot(bands["band"], bands["required at 5bp (bp/day)"], marker="o",
             linewidth=1.6, label="required at 5bp")
axes[1].plot(bands["band"], bands["observed gross (bp/day)"], marker="o",
             linewidth=1.6, label="observed gross")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("The hurdle, and what the signal delivered")
axes[1].set_xlabel("Band (weight)")
axes[1].set_ylabel("Basis points per day")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# バンドを広げるとハードルは下がり、ポートフォリオはシグナルが求めたものから離れていきます。
# 効いているシグナルなら、2本の線の差がいちばん大きいところが運用点になります。ここでは
# 2本の線は交わりません。それが答えです。

# %%
best = bands.loc[bands["net Sharpe at 5bp"].idxmax()]
print(f"best band              {best['band']:.3f}")
print(f"turnover there         {best['daily turnover']:.2f} per day "
      f"(from {bands.iloc[0]['daily turnover']:.2f})")
print(f"hurdle there           {best['required at 5bp (bp/day)']:.2f} bp/day "
      f"(from {bands.iloc[0]['required at 5bp (bp/day)']:.2f})")
print(f"observed gross there   {best['observed gross (bp/day)']:+.2f} bp/day")
print(f"still short by         "
      f"{best['required at 5bp (bp/day)'] - best['observed gross (bp/day)']:.2f} bp/day")

# %% [markdown]
# ## 6. まだ足りていないもの
#
# ここまでは日次終値のシミュレーションなので、ポートフォリオ全体を終値の1つの価格で売買できる
# という前提です。毎日ほとんどを入れ替える戦略では、その前提はシグナルより大きな仕事をしています。
#
# ここから先はレシピ 04/08 です。このウェイトを目標ポジションに変え、板に対してリプレイして、
# 何が約定したかをエンジンに決めさせます。上で計算した損益分岐の数字は、その実行に手間をかける
# 価値があるかを事前に教えてくれます。この戦略がどれだけの執行品質を買えるのか、という形で。

# %%
print(f"a book of {len(returns.columns)} names turning over "
      f"{bands.iloc[0]['daily turnover']:.0%} a day trades "
      f"{bands.iloc[0]['daily turnover'] * 252:.0f}x its capital a year")
print(f"at the banded operating point that falls to "
      f"{best['daily turnover'] * 252:.0f}x")

# %% [markdown]
# ## まとめ
#
# - 1つを信じる前に、もっともらしい形成期間をすべて走査してください。互いに食い違う小さな
#   数字が並ぶとき、それはシグナルの不在です。
# - 大型株の短期リバーサルはこのユニバースでは検出できません。20年ぶんの研究とも整合します。
#   それを報告することが結果です。
# - エンジン側でクロスセクショナルに平均を引くこと（`cs_demean`）が、方向の賭けを
#   相対の賭けにしています。
# - 速い戦略で決定的な数字は必要グロスリターン、つまり回転率×コストで、シグナルには依存しません。
# - ノートレード・バンドは回転率を下げてハードルを下げ、そのぶんシグナルが求めた
#   ポートフォリオからは離れます。
# - 日次終値のバックテストは、全体を1つの価格で売買できると仮定します。毎日入れ替える戦略なら、
#   それは板に対して検証しなければなりません。

# %%
db.close()
