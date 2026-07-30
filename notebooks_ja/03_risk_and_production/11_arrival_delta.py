# %% [markdown]
# # arrival-delta: 昨夜のバックテストのうち、実際に持ちこたえたのはどれか
#
# リサーチのエージェントが一晩で40本のバックテストを回します。パラメータのスイープでも、
# for ループを持った新人でも同じです。朝には40個のシャープが並びます。
#
# まともにレビューするには、1本ずつを判断の瞬間のデータに対して導き直す必要があります。誰も
# それを40回はやらないので、実際には上位が昇格し、残りは削除されます。それこそが、リークを
# 昇格させる選別手順です。
#
# `arrival_delta` はそのレビューを1つの数字に変えます。同じクエリを2回、現在の先頭に対してと
# 判断時点の読み取り点に対して走らせ、その差を報告するのです。
#
# 指標のうち動いた部分が、注文を出したはずの時点にはまだ届いていなかったデータに依存していた
# 部分です。本番で蒸発するアルファのことです。
#
# このレシピでは実際に1つ測ったうえで、この検査があなたを誤らせる2つの筋道に同じだけの分量を
# 割きます。*空虚な*結果と、構造上見えないイベント時刻のリークです。誤って信頼している診断は、
# 診断がないより悪いものです。

# %% [markdown]
# ## ここで使う用語
#
# | 用語              | 意味 |
# | --------------- | --- |
# | `arrival_delta` | 同じクエリをヘッドと過去の意思決定時点で実行し、その差を報告する |
# | 意思決定時点          | その取引が置かれたはずの瞬間。使ってよいデータはそこまで |
# | 到着（arrival）の軸   | 意思決定後に届いた行や訂正された行から生じる先読み |
# | イベント時刻の軸        | 1つのスナップショット内で窓が前へはみ出すことから生じる先読み |
# | 先読みバイアス         | その時点では手に入らなかった情報を使ってしまうこと |
# | 選択（selection）   | 多数の実行から最良を選ぶこと。最も漏れた実行が昇格しやすい |
# | エンバーゴ           | 意思決定時刻とデータ時刻の間に空ける間隔。境界をまたぐ漏れを防ぐ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_leakage"), create=True)
rng = np.random.default_rng(11)

# %% [markdown]
# ## 1. データ
#
# `cu.make_daily_prices` が返すのは日足 OHLCV パネルです。合成60銘柄×500セッションで、1行が
# 1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、20:00 UTC |
# | `symbol` | `string` | 銘柄コード、`STK000` 〜 `STK059` |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `volume` | `int64` | 出来高（株数） |

# %%
symbols = [f"STK{i:03d}" for i in range(60)]
panel = cu.make_daily_prices(symbols=symbols, days=500)
print(f"{panel.num_rows:,} rows x {panel.num_columns} columns, {len(symbols)} symbols")
panel.to_pandas().head()

# %% [markdown]
# ## 2. 到着の履歴を持つパネル
#
# この練習が成り立つのは、データベースが*いつ何が届いたか*の記録であるときだけです。いまそれが
# 何を言っているか、だけでは足りません。
#
# そこでベンダーの最初の公表を読み込み、そのあと訂正がそれぞれのコミットとして遅れて着地する
# ようにします。履歴を残せば、ベンダーのフィードは実際にこういう姿になります。

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])

first_publication = (
    panel.select(["ts", "symbol", "close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
)
db.append("prices", first_publication, note="vendor delivery, first publication")

pdf = first_publication.to_pandas()
print(f"{len(pdf):,} rows, {pdf.ts.min().date()} .. {pdf.ts.max().date()}, version 1")

# %% [markdown]
# ## 3. 訂正が届く
#
# 言い直しが3件、それぞれ別のコミットです。桁の打ち間違い、見落とした分割、そして遅れて来た
# 調整。
#
# 範囲の置き換えは窓全体を書き直すので、置き換えるフレームはその日の無関係な行もそのまま通す
# 必要があります。下のヘルパーが窓を読み、当て、窓全体を返します。

# %%
def restate(day: pd.Timestamp, symbol: str, factor: float, note: str) -> int:
    """Replace one trading day, correcting a single symbol's close."""
    start_us = day.value // 1_000
    end_us = (day + pd.Timedelta(days=1)).value // 1_000
    window = db.read("prices", time_start=start_us, time_end=end_us).to_pandas()
    window.loc[window.symbol == symbol, "close"] *= factor
    fixed = pa.Table.from_pandas(
        window.sort_values(["ts", "symbol"])[["ts", "symbol", "close"]],
        schema=schema,
        preserve_index=False,
    )
    plan = db.plan_replace_range("prices", start_us, end_us, data=fixed, note=note)
    return plan.apply()["sequence"]


days = pd.DatetimeIndex(sorted(pdf.ts.unique()))
# Corrections land against days inside the sample, the way real ones do.
for day, sym, factor, why in [
    (days[-40], "STK003", 0.5, "missed 2:1 split"),
    (days[-25], "STK011", 0.97, "fat-finger print, vendor corrected"),
    (days[-12], "STK007", 1.02, "late dividend adjustment"),
]:
    seq = restate(day, sym, factor, why)
    print(f"v{seq}: {why} on {day.date()} ({sym})")

for v in db.versions("prices"):
    print(f"  v{v['sequence']:>2} {v['op']:<13} {v.get('note', '')}")

# %% [markdown]
# ## 4. 指標を、1つのクエリとして1度だけ書く
#
# `arrival_delta` が走らせ直すのは*クエリ*なので、戦略の全体が1つのクエリで表せる必要があり
# ます。この制約は機能です。データベースに渡せる指標は、過去のものを含めてどの読み取り点にも
# 渡せる指標です。
#
# 以下は素朴なクロスセクションのモメンタムの本です。21日モメンタムを日ごとに平均で引き、確信度
# で重み付けし、1日持ちます。
#
# シグナルはあえて `prev`、つまり獲得するリターンの*前*の終値を使うので、戦略そのものに先読みは
# ありません。これから測るものはすべて、データが動いたことから来ます。コードのごまかしでは
# ありません。

# %%
# arrival_delta() takes a SQL string (it re-runs the same text at two read
# points), so this study stays a string end to end.
SHARPE_SQL = """
WITH d AS (
  SELECT ts, symbol, close,
         lag(close, 1)  OVER (PARTITION BY symbol ORDER BY ts) AS prev,
         lag(close, 22) OVER (PARTITION BY symbol ORDER BY ts) AS prev22
  FROM prices
),
sig AS (
  SELECT ts,
         close / prev - 1.0   AS r,
         prev / prev22 - 1.0  AS mom
  FROM d
  WHERE prev IS NOT NULL AND prev22 IS NOT NULL
),
w AS (
  SELECT ts, r, mom - avg(mom) OVER (PARTITION BY ts) AS wt
  FROM sig
),
daily AS (
  SELECT ts, sum(wt * r) / nullif(sum(abs(wt)), 0) AS pnl
  FROM w GROUP BY ts
)
SELECT avg(pnl) / nullif(stddev(pnl), 0) * sqrt(252.0) AS sharpe FROM daily
"""

head_sharpe = db.sql(SHARPE_SQL).to_pandas().sharpe.iloc[0]
print("head Sharpe:", round(head_sharpe, 3))

# The data is a synthetic random walk with a common factor, so this number is
# noise around zero and its *level* means nothing. Everything below is about
# how much it moves when the read point changes, which is a property of the
# data's history rather than of the strategy.

# %% [markdown]
# ## 5. 検査そのもの
#
# 判断時点はバージョン1です。研究が行われたときの世界の状態で、どの訂正もまだ届いていません。
# それ以降はすべて後知恵です。

# %%
report = db.arrival_delta(SHARPE_SQL, version=1)

print(f"changed          : {report['changed']}")
print(f"vacuous          : {report['vacuous']}")
metric = report["columns"][0]
print(f"head             : {metric['head']:.3f}   (what the backtest reported)")
print(f"as-of v1         : {metric['asof']:.3f}   (what it could actually have known)")
print(f"delta            : {metric['delta']:+.3f} Sharpe")
print("\nwithheld commits:")
for w in report["withheld_versions"]:
    print(f"  {w['table']}: v{w['asof_version']} -> v{w['head_version']}")

# %% [markdown]
# 指標自身の単位で、*絶対値*の差を読んでください。この本のシャープは、知りえたことと今わかって
# いることのあいだで、およそ0.5ポイント動きました。
#
# レポートは `delta_pct` も持っていますが、ゼロ近くに座る比率の指標ではこのパーセンテージが
# 爆発します。-0.36 から -0.85 へのシャープは「-133%」で、算術的には正しく、役には立ちません。
# パーセンテージは意味のある基準値を持つ指標のためのものです。それ以外には生の差を使ってくだ
# さい。
#
# これは戦略が壊れているという判定ではありません。言い直しはデータへの真っ当な改善ですし、
# 訂正後のシャープのほうが正直な数字です。
#
# 差が言っているのは、*結果のこれだけの部分は判断時点では知りえなかった*ということです。だから
# その日に始まったライブの実行は、この本を売買していなかったでしょう。これは掛け目として使い、
# また40本のうち3本を見る時間しかないときの並べ替えのキーとして使ってください。

# %%
print("\n".join(report["notes"]))

# %% [markdown]
# ## 6. スイープ全体のトリアージ
#
# これが実際のワークフローです。一晩のスイープから出てきたすべての変種に同じ検査をかけ、
# 露出の大きい順に並べます。シャープ順では並べません。

# %%
def with_lookback(n: int) -> str:
    """Same book, different momentum window. Asserted so a silent no-op in the
    substitution cannot quietly turn the sweep into three copies of one query."""
    sql = SHARPE_SQL.replace("lag(close, 22)", f"lag(close, {n})").replace("prev22", f"prev{n}")
    assert (n == 22) or (sql != SHARPE_SQL), f"lookback substitution failed for {n}"
    return sql


VARIANTS = {f"mom_{n}d": with_lookback(n) for n in (22, 66, 5)}

rows = []
for name, sql in VARIANTS.items():
    rep = db.arrival_delta(sql, version=1)
    c = rep["columns"][0]
    rows.append(
        {
            "variant": name,
            "reported": c["head"],
            "knowable": c["asof"],
            "delta": c["delta"],
            "vacuous": rep["vacuous"],
        }
    )

triage = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False)
print(triage.round(3).to_string(index=False))

# %% [markdown]
# ゲートはそこから直に導けます。後知恵があってはじめて残る「優位性」を持つものは、何も昇格
# させない。閾値はハウスのパラメータです。規律とは、それが存在し、計算で決まっていることです。目分量で
# 決めてはいけません。

# %%
TOLERATED_SHARPE_MOVE = 0.15
promoted = triage[triage.delta.abs() <= TOLERATED_SHARPE_MOVE]
print(
    f"promoted {len(promoted)} of {len(triage)} variants "
    f"at |delta| <= {TOLERATED_SHARPE_MOVE} Sharpe"
)
if promoted.empty:
    print("  (none survive: a single missed split moves every variant in this book)")

# %% [markdown]
# ## 7. 失敗の形その1: 何も意味しないゼロ
#
# この診断が助けになるか、それとも安心させて眠らせるかを決めるのがここです。
#
# `arrival_delta` は2つの読み取り点を比べます。両方が同じバージョンに解決されると、同一のデータ
# を比べることになり、差はゼロに*強制*されます。一括で読み込んだだけのデータベースはすべてこう
# なりますし、それが普通のコールドスタートです。
#
# 何も測っていないのに、何も意味しないゼロは、健康診断の合格通知とまったく同じ見た目をします。
#
# レポートはそれを `vacuous` で伝えます。数字を読む前に、そのフィールドを確認してください。

# %%
bulk = h5i_db.Database(cu.fresh_db("prod_leakage_bulk"), create=True)
bulk.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])
bulk.append("prices", first_publication, note="ten years, one commit")

bulk_report = bulk.arrival_delta(SHARPE_SQL, version=1)
print(f"changed          : {bulk_report['changed']}   <- looks clean")
print(f"vacuous          : {bulk_report['vacuous']}   <- but it checked nothing")
print(f"withheld commits : {len(bulk_report['withheld_versions'])}")
print()
print(bulk_report["notes"][0])
bulk.close()

# %% [markdown]
# ## 8. 失敗の形その2: この検査に見えないリーク
#
# `arrival_delta` が測るのは*到着*の軸です。いまは存在するが、当時はまだ公表されていなかった行。
#
# もう一方の軸には見えていません。ずっとテーブルにあった行を、読むべきではない瞬間に読んで
# しまうケースです。
#
# 典型は、同じ日の終値を使って同じ日の終値で売買するシグナルです。以下では `mom` を `prev` では
# なく `close` から作っているので、戦略は同時に獲得する情報で売買することになります。シャープは
# ばかげた値になり、リークは全面的で、それでも検査はほとんど何も報告しません。

# %%
LEAKY_SQL = SHARPE_SQL.replace("prev / prev22 - 1.0  AS mom", "close / prev22 - 1.0 AS mom")
assert LEAKY_SQL != SHARPE_SQL, "the leaky-signal substitution did not apply"

leaky_report = db.arrival_delta(LEAKY_SQL, version=1)
lc = leaky_report["columns"][0]
inflation = lc["head"] - metric["head"]
print(f"leaky Sharpe     : {lc['head']:.2f}  (honest signal: {metric['head']:.2f})")
print(f"inflation from the leak : {inflation:+.2f} Sharpe")
print(f"arrival delta           : {lc['delta']:+.2f} Sharpe")
print(f"vacuous                 : {leaky_report['vacuous']}")

# %% [markdown]
# 検査が何を伝え、何を伝えなかったかに注意してください。
#
# このクエリについては本物の到着の差を報告します。言い直しは、リークしているシグナルのほうも
# 動かすからです。ですがその差が測っているのは言い直しであって、覗き見ではありません。自分の
# バーを読むシグナルとそうでないシグナルを、レポートの何かが区別するわけではありません。それは
# 測っている対象ではないのです。
#
# 差が大きく出ようと小さく出ようと、それは先読みについての判定ではありません。レポート自身の
# `notes` が、毎回そう書いています。
#
# イベント時刻の軸には別の道具が要ります。走査そのものに当てる打ち切りです。
#
# SQL なら、その境界を手で書けます。ウォークフォワード評価の各ステップがまさにこれです。T で
# 判断し、T 以前のタイムスタンプを持つ行だけを見る。
#
# 下の検査があえて行数になっているのは意図的です。境界をまたいでシャープを比べると、打ち切りと、
# それが生む短い標本を混同してしまいます。それも数字で嘘をつく1つの形です。

# %%
decision = days[-60]
cutoff = decision.strftime("%Y-%m-%dT%H:%M:%SZ")

n_rows = lambda frame: frame.select(count_star().alias("n")).to_pandas().n.iloc[0]
visible = n_rows(db.table("prices").filter(col("ts") <= cutoff))
total = n_rows(db.table("prices"))
print(f"decision date {decision.date()}: {visible:,} of {total:,} rows readable")
print(f"future rows hidden: {total - visible:,}")

# %% [markdown]
# その境界を手で書くことの厄介さは、毎回、すべての変種のすべてのサブクエリで、永久に覚えて
# おかなければならないことです。しかもリークしたバックテストは良いバックテストに見えるので、
# 忘れた日が昇進する日になります。

# %% [markdown]
# CLI はその境界を、覚えておくものから構造的なものに変えます。セッションが固定され、その中の
# *どの*クエリもその瞬間より先には届きません。明示的に求めるクエリでさえもです。
#
# ```bash
# h5i-db query prod_leakage.db "<the same SQL>" \
#   --decision-time 2026-05-02T00:00:00Z \
#   --embargo 1d
# ```
#
# 両方使ってください。打ち切りがイベント時刻のリークを起こさせず、`arrival_delta` はどんな
# 打ち切りでも防げない到着のリークを測ります。後者は、あなたのあとに世界が何かを知ったことに
# よって起きるからです。

# %% [markdown]
# ## まとめ
#
# - **`arrival_delta` は後知恵に値段を付けます。** クエリ1つ、読み取り点2つ、そしてその差が、
#   結果のうち判断時点では知りえなかった量です。指標自身の単位で読み、スイープはシャープでは
#   なくこれで並べてください。`delta_pct` は基準値がゼロ近くの指標で爆発しますし、シャープは
#   まさにそういう指標です。
# - **数字より先に `vacuous` を読んでください。** コミットが1件のデータベースでは検査が同一の
#   データを比べることになり、そのゼロは証拠ではなく算術です。一括で読み込んだばかりのストアの
#   普通の状態がこれです。
# - **見えるのは到着の軸だけです。** 同一バーの先読みはこの検査には見えません。覗き見している
#   シグナルについて報告される差は、覗き見ではなく言い直しを測っています。だからその差がどんな
#   値でも、シグナルの先読みが否定されたことにはなりません。レポートの `notes` が毎回そう書いて
#   いますし、それを信じてください。
# - **言い直された分割1件が、本まるごとを上回ることがあります。** 訂正された銘柄1つが、ここでは
#   すべての変種を 0.4 シャープ以上動かしました。コーポレートアクションは到着の履歴における
#   丸め誤差ではありません。たいていの場合、それが話のすべてです。
# - **2つの道具は代替ではなく補完です。** `--decision-time` はイベント時刻のリークを構造的に
#   防ぎます。`arrival_delta` は到着のリークを測ります。後者はデータが変わることで起きるので、
#   そもそも防げません。測ることしかできません。
# - **働いている h5i-db の機能:** コミットごとの書き換え不能なバージョニング（問いに答えられる
#   ようにする到着の履歴）、O(1) のタイムトラベル（2回の実行がどちらも安い）、そしてプレビュー
#   できる `plan_replace_range`（言い直しが静かな上書きにならず、監査できるものになる）。

# %%
db.close()
