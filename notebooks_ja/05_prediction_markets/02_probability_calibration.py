# %% [markdown]
# # この市場の確率は当てになるのか
#
# 予測市場は確率を提示するので、ほかのどんな予測とも同じように採点できます。このレシピでは信頼性
# 曲線を描き、Brier スコアを reliability、resolution、uncertainty に分解し、ミスキャリブレーションが
# どちら向きかを読み取ります。方法論として手間をかけるのは、正解を予測から遠ざけることです。決着は
# 専用のテーブルに置き、結果が「観測可能」になった瞬間で日付をつけ、ポイントインタイムの読み出しに
# よって、予測がそれを見られなかったことを証明します。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | キャリブレーション | 提示された確率が、実際に観測される頻度と一致しているか |
# | 信頼性曲線 | 提示確率に対して実測頻度をプロットしたもの。対角線が完全 |
# | Brier スコア | 確率予測の平均二乗誤差。小さいほど良い |
# | reliability / resolution / uncertainty | Brier スコアが分解される3つの成分 |
# | 対数損失 | 別の評価指標。自信を持った誤りを Brier よりはるかに強く罰する |
# | 基準率 | その出来事が全体として起きる頻度。どんな予測もまず超えるべき水準 |
# | 観測可能性 | 結果が知り得るようになった時点。resolutions の日付はこれで付く |
# | ポイントインタイム | 予測が正解を見られなかったことを示す読み出し |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cookbook_utils as cu
import h5i_db

db = h5i_db.Database(cu.fresh_db("05_probability_calibration"), create=True)

# %% [markdown]
# ## パネルと正解
#
# 240 のバイナリマーケットが、セッションを通じて気配を出し、そのあと決着します。ここで効いてくる
# テーブルは2つで、混同してはいけません。
#
# `book_deltas` が予測を持ちます。任意の瞬間の YES のミッドが、市場の確率です。`resolutions` が結果を
# 持ち、1マーケットにつき1行です。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | 結果が**観測可能**になった時刻。出来事が起きた時刻ではない |
# | `instrument_id` | `string` | マーケット |
# | `winner_outcome` | `uint16` | 0 = YES の勝ち、1 = NO の勝ち |
#
# 最初の行にあるこの区別が、規律のすべてです。選挙は開票の夜に決まりますが、確定の宣言は数時間後に
# なります。早いほうの瞬間で決済すると、誰も受け取れなかった利益を計上してしまいます。

# %%
panel = cu.make_prediction_markets(n_markets=240, steps=48, seed=11)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel load")
db.snapshot("panel-v1", tables=list(panel), note="calibration study input")
resolutions = panel["resolutions"]
print(f"resolutions: {resolutions.num_rows:,} rows x {resolutions.num_columns} columns")
resolutions.to_pandas().head()

# %% [markdown]
# ## 判断時点で固定した予測
#
# 採点には1マーケットにつき1つの予測が要るので、判断の瞬間を決め、そこでの YES のミッドを読みます。
# その瞬間より後は、予測にとっては範囲外、採点にとっては範囲内です。

# %%
stamps = sorted({value.as_py() for value in panel["book_deltas"].column("ts_init")})
decision = stamps[len(stamps) // 2]
decision_ns = int(pd.Timestamp(decision, tz="UTC").value)
print(f"session runs {stamps[0]} .. {stamps[-1]}")
print(f"decision instant: {decision} ({decision_ns} ns)")

forecasts = db.sql(
    f"""
    SELECT instrument_id,
           (max(CASE WHEN side = 'buy' THEN price END)
          + max(CASE WHEN side = 'sell' THEN price END)) / 2 AS p_yes
    FROM h5i('book_deltas', 'panel-v1')
    WHERE outcome = 0 AND ts_init = to_timestamp_nanos({decision_ns})
    GROUP BY instrument_id
    ORDER BY instrument_id
    """
).to_pandas()
print(f"\n{len(forecasts)} forecasts")
print(f"quoted probability spans {forecasts.p_yes.min():.3f} .. {forecasts.p_yes.max():.3f}")

# %% [markdown]
# ## 予測が覗けなかったことの証明
#
# `resolutions` は観測可能になった瞬間で時刻インデックスされているので、判断時刻で区切った読み出しは
# 何も返しません。これは分析者が忘れずに守る約束事ではなく、読み出しそのものの性質です。`time_end`
# は時刻列の生の単位で、これらのテーブルではナノ秒です。

# %%
knowable = db.read("resolutions", time_end=decision_ns)
print(f"resolutions knowable at the decision instant: {knowable.num_rows}")
print(f"resolutions available now:                    {db.read('resolutions').num_rows}")
assert knowable.num_rows == 0

# %% [markdown]
# ## 正解を、一度だけ結合する
#
# ここではじめてラベルを入れます。`yes_won` は実現した結果を 0/1 の変数にしたもので、以下のすべての
# 採点ルールがこれを使います。

# %%
truth = cu.market_truth(panel).to_pandas()
scored = forecasts.merge(truth, on="instrument_id", validate="one_to_one")
scored["y"] = scored.yes_won.astype(float)
print(f"{len(scored)} scored markets, base rate {scored.y.mean():.3f}")
scored.head()

# %% [markdown]
# ## 信頼性曲線
#
# 提示確率でバケットに分け、提示確率の平均と実測頻度を比べます。キャリブレーションが取れた市場は
# 対角線上に乗ります。差には符号があり、対角線より下ならそのバケットは割高だったということです。

# %%
EDGES = [0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 1.0]
scored["bucket"] = pd.cut(scored.p_yes, EDGES)
reliability = (
    scored.groupby("bucket", observed=True)
    .agg(n=("y", "size"), mean_quote=("p_yes", "mean"), realized=("y", "mean"))
    .assign(gap_pp=lambda f: (f.realized - f.mean_quote) * 100)
)
print(reliability.round(3).to_string())

# %% [markdown]
# 安いバケットは提示確率を下回り、高いバケットは上回ります。これがフェイバリット・ロングショット・
# バイアスです。ロングショットは割高で、フェイバリットは割安になります。レシピ 05/03 では、これを
# 取引に変えます。

# %%
low = reliability.head(3)
high = reliability.tail(3)
print(f"three cheapest buckets, mean gap: {np.average(low.gap_pp, weights=low.n):+.2f} pp")
print(f"three richest  buckets, mean gap: {np.average(high.gap_pp, weights=high.n):+.2f} pp")

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
ax.scatter(reliability.mean_quote, reliability.realized, s=reliability.n * 3,
           color="#c0392b", zorder=3, label="realized frequency")
for row in reliability.itertuples():
    ax.annotate(f"n={row.n}", (row.mean_quote, row.realized),
                textcoords="offset points", xytext=(6, -10), fontsize=7)
ax.set_title("Reliability curve, 240 binary markets")
ax.set_xlabel("quoted probability at the decision instant")
ax.set_ylabel("realized YES frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 採点ルール
#
# Brier スコアは確率予測の平均二乗誤差です。Murphy の分解は、これを別々の問いに答える3つの成分に
# 分けます。reliability はバケットが対角線からどれだけ外れているか、resolution はそもそも予測が結果を
# どれだけ区別しているか、uncertainty は基準率が持つ、減らしようのない分散です。
#
# `Brier = reliability - resolution + uncertainty` なので、reliability は小さいほど良く、resolution
# は大きいほど良い指標です。常に基準率を提示する予測は、reliability が完璧で resolution がゼロに
# なります。役には立ちませんが、正直ではあります。

# %%
def brier_decomposition(p: pd.Series, y: pd.Series, edges: list[float]) -> dict[str, float]:
    """Murphy's three-term decomposition of the Brier score."""
    frame = pd.DataFrame({"p": p, "y": y})
    frame["bin"] = pd.cut(frame.p, edges)
    n = len(frame)
    base = frame.y.mean()
    groups = frame.groupby("bin", observed=True)
    counts = groups.size()
    reliability = float((counts * (groups.p.mean() - groups.y.mean()) ** 2).sum() / n)
    resolution = float((counts * (groups.y.mean() - base) ** 2).sum() / n)
    uncertainty = float(base * (1.0 - base))
    return {
        "brier": float(((frame.p - frame.y) ** 2).mean()),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "identity": reliability - resolution + uncertainty,
    }


def log_loss(p: pd.Series, y: pd.Series, floor: float = 1e-6) -> float:
    clipped = p.clip(floor, 1 - floor)
    return float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())


parts = brier_decomposition(scored.p_yes, scored.y, EDGES)
for key, value in parts.items():
    print(f"  {key:12} {value:.4f}")
print(f"  {'log loss':12} {log_loss(scored.p_yes, scored.y):.4f}")
assert abs(parts["brier"] - parts["identity"]) < 0.02

# %% [markdown]
# ## ベンチマーク。スコアだけでは何も意味しないから
#
# 0.18 という値は、比べるまでは良くも悪くもありません。2つの参照予測がこれを挟みます。基準率は情報を
# まったく持ちません。市場価格を再キャリブレーションしたものは、損失のうちどれだけがミス
# キャリブレーションのせいで、どれだけが本当の不確実性だったのかを示します。
#
# 再キャリブレーションは、このパネルを作るときに入れた圧縮を逆に戻すものです。実データなら
# アウトオブサンプルで推定するところですが、ここでは閉じた形が使えるので会計が正確になります。

# %%
BIAS = 0.12
scored["p_recal"] = (0.5 + (scored.p_yes - 0.5) / (1.0 - BIAS)).clip(0.01, 0.99)
rows = []
for label, forecast in (
    ("market quote", scored.p_yes),
    ("base rate", pd.Series(scored.y.mean(), index=scored.index)),
    ("recalibrated", scored.p_recal),
):
    part = brier_decomposition(forecast, scored.y, EDGES)
    rows.append(
        {
            "forecast": label,
            "brier": part["brier"],
            "reliability": part["reliability"],
            "resolution": part["resolution"],
            "log_loss": log_loss(forecast, scored.y),
        }
    )
print(pd.DataFrame(rows).set_index("forecast").round(4).to_string())

# %% [markdown]
# 基準率は reliability の誤差がゼロで、resolution もゼロです。マーケットどうしを区別することが一度も
# ありません。市場の提示価格は本物の resolution を稼ぎ、その代わり reliability のペナルティを負います。
# 再キャリブレーションはそのペナルティを約4分の1に減らしつつ、resolution を保ちます。順位付けは
# うまいのに、スケールが狂っている予測の特徴です。
#
# reliability 項が絶対値としてどれだけ小さいかに注目してください。バケットの5〜9ポイントのずれも、
# 二乗すれば数千分の1にしかならないので、キャリブレーションの悪い市場でも自分の Brier スコアはほとんど
# 動きません。取引に使える発見は、この項の大きさよりも信頼性曲線の「符号の構造」のほうにあります。

# %% [markdown]
# ## セッションが進んでもキャリブレーションは良くならない
#
# 仮定せずに確かめる価値のあるところです。同じマーケットを複数の瞬間で採点すれば、満期が近づくにつれて
# 市場が学習するかどうかが分かります。このパネルでは学習しません。ミスプライスが持続するように作って
# あるからです。実データなら、resolution 項が上がっていく姿を期待したいところです。

# %%
horizon = []
for index in (4, 16, 28, 40):
    at = stamps[index]
    at_ns = int(pd.Timestamp(at, tz="UTC").value)
    frame = db.sql(
        f"""
        SELECT instrument_id,
               (max(CASE WHEN side = 'buy' THEN price END)
              + max(CASE WHEN side = 'sell' THEN price END)) / 2 AS p_yes
        FROM h5i('book_deltas', 'panel-v1')
        WHERE outcome = 0 AND ts_init = to_timestamp_nanos({at_ns})
        GROUP BY instrument_id
        """
    ).to_pandas().merge(truth, on="instrument_id")
    frame["y"] = frame.yes_won.astype(float)
    part = brier_decomposition(frame.p_yes, frame.y, EDGES)
    horizon.append(
        {
            "instant": str(at.time()),
            "steps_to_expiry": len(stamps) - 1 - index,
            "brier": part["brier"],
            "reliability": part["reliability"],
            "resolution": part["resolution"],
        }
    )
print(pd.DataFrame(horizon).set_index("instant").round(4).to_string())

# %% [markdown]
# ## まとめ
#
# - 提示された確率は予測であり、予測として採点すべきである。Brier の分解は、そのスコアになった「理由」
#   を教えてくれる。対角線から外れていること（reliability）と、そもそも結果を区別できていないこと
#   （resolution）は、別の失敗である。
# - ベンチマークと比べて採点する。基準率が役立たずの側の境界を定め、再キャリブレーションした予測が、
#   損失のうちどれだけが無知によるもので、どれだけがスケールの問題だったかを示す。
# - `resolutions` は、出来事の時刻ではなく結果が観測可能になった瞬間で日付がつく。レシピ 05/05 で
#   決済を制御するのは、この列である。
# - ここで働いた h5i-db の機能。ラベルが専用の時刻インデックス付きテーブルにあるので、
#   `db.read(..., time_end=decision_ns)` が0行を返すことが、先読みがないことの構造的な証明になる。
#   約束ではない。名前付きスナップショットのおかげで、上のすべての数値が1つのピンから再導出できる。

# %%
db.close()
