# %% [markdown]
# # リーク検査: 昨夜のバックテストのうち、本当に持ちこたえたのはどれか
#
# リサーチのエージェント、あるいはパラメータスイープ、あるいは for ループを回した新人が、一晩で
# 40本のバックテストを走らせます。朝には40個の Sharpe が並びます。まともにレビューするなら、
# その1つずつを、判断の瞬間にデータがどうだったかに対して導出し直す必要があります。誰もやり
# ません。だから実際には上位が昇格し、残りは削除されます。それこそが、漏れを昇格
# させる選抜手続きにほかなりません。
#
# `arrival_delta` は、このレビューを1つの数値に変えます。同じクエリを2回――現在の先頭に対してと、
# 判断時点の読み取り点に対して――走らせ、その差を報告するだけです。難しい話ではありません。動いた部分が、その売買が置かれた
# はずの時点にはまだ届いていなかったデータに依存していた部分です。つまり、本番で蒸発するアルファ
# です。
#
# このレシピではまず本物を1つ測り、そのあと同じだけの分量を、この検査が誤解を招く2つの形に
# 割きます。*空虚な*結果と、構造上見えないイベント時刻の漏れです。誤って信頼している診断は、
# 診断がないより悪いからです。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_leakage"), create=True)
rng = np.random.default_rng(11)

# %% [markdown]
# ## 1. 到着の履歴を持つパネル
#
# この演習が成り立つかどうかは、データベースが「いま何と言っているか」だけでなく「*いつ届いたか*」の
# 記録になっているかにかかっています。そこでベンダーの最初の公表をまず読み込み、訂正はあとから
# それぞれのコミットとして着地させます。履歴を残しておけば、ベンダーのフィードは実際こう見えます。

# %%
symbols = [f"STK{i:03d}" for i in range(60)]
panel = cu.make_daily_prices(symbols=symbols, days=500)

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
# ## 2. 訂正が届く
#
# 訂正は3件、それぞれ別のコミットです。入力ミス、取りこぼした分割、遅れて届いた調整。範囲の
# 置き換えは窓全体を書き直すので、置き換えるフレームにはその日の無関係な行もそのまま含める必要が
# あります。以下のヘルパーが、窓を読み、パッチを当て、窓全体を返します。

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
# ## 3. 指標を、単一のクエリとして1回だけ書く
#
# `arrival_delta` が再実行するのは*クエリ*なので、戦略全体を1つのクエリで表現できなければ
# なりません。この制約は利点です。データベースに渡せる指標は、過去のものも含めて、どの読み取り点にも
# 渡せる指標だからです。
#
# 以下は素直なクロスセクショナル・モメンタムの帳簿です。21日モメンタムを日ごとに平均除去し、
# 確信度で加重して、1日保有します。シグナルはあえて `prev`、つまり獲得するリターンの*前の*終値を
# 使うので、戦略そのものに先読みはありません。これから測るものはすべて、データが動いたことから
# 来ます。コードのズルではありません。

# %%
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
# ## 4. 検査そのもの
#
# 判断時点はバージョン1、つまり訂正が1つも届いていない、リサーチを行った時点の世界の状態です。
# それより後はすべて後知恵です。

# %%
report = db.arrival_delta(SHARPE_SQL, version=1)

print(f"changed          : {report['changed']}")
print(f"vacuous          : {report['vacuous']}")
col = report["columns"][0]
print(f"head             : {col['head']:.3f}   (what the backtest reported)")
print(f"as-of v1         : {col['asof']:.3f}   (what it could actually have known)")
print(f"delta            : {col['delta']:+.3f} Sharpe")
print("\nwithheld commits:")
for w in report["withheld_versions"]:
    print(f"  {w['table']}: v{w['asof_version']} -> v{w['head_version']}")

# %% [markdown]
# 差は*絶対値*で、指標自身の単位で読んでください。この帳簿の Sharpe は、当時知りえたものと
# いま知っていることのあいだで0.5ポイントほど動きました。レポートには `delta_pct` も入って
# いますが、ゼロ近辺に座る比率指標ではこのパーセントが爆発します。Sharpe が −0.36 から −0.85 へ
# 動いたことを「−133%」と言うのは、算術的には正しくても役に立ちません。パーセントは意味のある
# 基準値を持つ指標のためのものです。それ以外は生の差で読みます。
#
# これは戦略が壊れているという判定ではありません。訂正はデータの正真正銘の改善で、訂正後の
# Sharpe のほうが誠実です。この差が言っているのは、*結果のうちこれだけは判断時点で知りえな
# かった*ということです。だからその日に始まったライブ運用は、この帳簿を売買しなかったでしょう。
# 掛け目として使い、40本のうち3本を見る時間しかないときの並べ替えのキーとして使ってください。

# %%
print("\n".join(report["notes"]))

# %% [markdown]
# ## 5. スイープ全体のトリアージ
#
# こちらが実際の使い方です。一晩のスイープから出てきた全バリアントに同じ検査をかけ、Sharpe では
# なく後知恵への依存度で並べます。

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
# そこからゲートが直に導かれます。報告された優位性が後知恵でしか生き残らないものは、昇格させない。
# 閾値は各社のパラメータです。肝心なのは、それが存在し、計算されていることです。目分量で決めない。

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
# ## 6. 失敗の形1: 何も意味しないゼロ
#
# この診断が助けになるか、それとも安心させて眠らせるかを分けるのがここです。`arrival_delta` は
# 2つの読み取り点を比べます。両方が同じバージョンに解決されるなら――一括取り込みで読み込んだ
# データベース、つまり通常のコールドスタートがそうです――同一のデータどうしを比べることになり、
# 差は*必然的に*ゼロになります。何も測っていません。それでも、何も意味しないゼロは健康診断の
# 合格とまったく同じ顔をしています。
#
# レポートはそれを `vacuous` で伝えます。数値を読む前に、このフィールドを確認してください。

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
# ## 7. 失敗の形2: この検査に見えない漏れ
#
# `arrival_delta` が測るのは*到着*の軸です。いまは存在するけれど、当時はまだ公表されていなかった
# 行のことです。もう一方の軸には盲目です。テーブルにずっとあったけれど、読むべきでない瞬間に
# 読んでしまった行のことです。
#
# 典型例は、その日の終値を使ってその日の終値で売買するシグナルです。以下では `mom` を `prev` では
# なく `close` から作り、同時に獲得する情報で戦略が売買するようにします。Sharpe はばかげた値に
# なり、漏れは全面的で、そして検査はほぼ何も報告しません。

# %%
LEAKY_SQL = SHARPE_SQL.replace("prev / prev22 - 1.0  AS mom", "close / prev22 - 1.0 AS mom")
assert LEAKY_SQL != SHARPE_SQL, "the leaky-signal substitution did not apply"

leaky_report = db.arrival_delta(LEAKY_SQL, version=1)
lc = leaky_report["columns"][0]
inflation = lc["head"] - col["head"]
print(f"leaky Sharpe     : {lc['head']:.2f}  (honest signal: {col['head']:.2f})")
print(f"inflation from the leak : {inflation:+.2f} Sharpe")
print(f"arrival delta           : {lc['delta']:+.2f} Sharpe")
print(f"vacuous                 : {leaky_report['vacuous']}")

# %% [markdown]
# この検査が何を伝え、何を伝えなかったかに注目してください。このクエリについて本物の到着差は
# 報告します。訂正が漏れのあるシグナルのほうも動かすからです。ただしその差が測っているのは訂正で
# あって、覗き見ではありません。自分の足を読むシグナルとそうでないシグナルを、レポートの何かが
# 区別してくれるわけではありません。それはこの検査が測っている対象ではないからです。差が大きく
# 出ようと小さく出ようと、それは先読みについての判定にはなりません。レポート自身の `notes` が
# 毎回そう書いています。
#
# イベント時刻の軸には別の道具が要ります。スキャンそのものに当てるカットオフです。
#
#
# SQL なら、その境界を手で書けます。ウォークフォワード評価の各ステップはまさにこれです。時刻Tで
# 判断し、Tまでのタイムスタンプを持つ行だけを見る。以下の検査をあえて Sharpe ではなく行数に
# したのは、境界をまたいで Sharpe を比べると、カットオフの効果とそれが生む短いサンプルの
# 効果が混ざるからです。それはそれで数字による嘘の一種になります。

# %%
decision = days[-60]
cutoff = decision.strftime("%Y-%m-%dT%H:%M:%SZ")
bound = f"WHERE ts <= TIMESTAMP '{cutoff}'"
visible = db.sql(f"SELECT count(*) AS n FROM prices {bound}").to_pandas().n.iloc[0]
total = db.sql("SELECT count(*) AS n FROM prices").to_pandas().n.iloc[0]
print(f"decision date {decision.date()}: {visible:,} of {total:,} rows readable")
print(f"future rows hidden: {total - visible:,}")

# %% [markdown]
# この境界を手で書くことの厄介さは、毎回、どのバリアントのどのサブクエリでも、永遠に覚えて
# おかなければならない点にあります。しかも漏れたバックテストは出来のよいバックテストに見えます。
# 忘れた日にかぎって昇進するわけです。

# %% [markdown]
# CLI は、その境界を「覚えておくもの」から構造的なものへ変えます。セッションが固定され、その
# 中の*どの*クエリも、明示的に求めるものも含めて、その瞬間より先には手を伸ばせません。
#
# ```bash
# h5i-db query prod_leakage.db "<the same SQL>" \
#   --decision-time 2026-05-02T00:00:00Z \
#   --embargo 1d
# ```
#
# 両方を使ってください。イベント時刻の漏れを起こさないためのカットオフと、どんなカットオフでも
# 防げない到着の漏れを測るための `arrival_delta` です。到着の漏れは、あなたが知ったあとで世界が
# 何かを知ったせいで起きるものだからです。

# %% [markdown]
# ## まとめ
#
# - **`arrival_delta` は後知恵に値段を付けます。** 1つのクエリ、2つの読み取り点。その差が、
#   結果のうち判断時点で知りえなかった量です。指標自身の単位で読み、スイープの並べ替えには
#   Sharpe ではなくこの差を使ってください。`delta_pct` は基準値がゼロ近くにある指標――Sharpe が
#   まさにそうです――で爆発します。
# - **数値より先に `vacuous` を読んでください。** コミットが1つだけのデータベースでは同一の
#   データどうしが比較され、そのゼロは証拠ではなく算術です。一括読み込みしたばかりのストアでは
#   これが通常の状態です。
# - **見えるのは到着の軸だけです。** 同じ足での先読みには盲目で、覗き見しているシグナルについて
#   報告される差は訂正を測っているだけです。だからその差がいくつであろうと、先読みの疑いは
#   晴れません。レポートの `notes` が毎回そう書いています。信じてください。
# - **訂正された分割1件が、帳簿1つぶんより重いこともあります。** ここでは1銘柄の訂正だけで、
#   どのバリアントも Sharpe が0.4以上動きました。コーポレートアクションは到着の履歴における
#   端数の誤差ではありません。たいてい話の全部です。
# - **2つの道具は代替ではなく補完です。** `--decision-time` はイベント時刻の漏れを構造的に
#   防ぎます。`arrival_delta` は到着の漏れを測ります。後者はデータが変わることで起きるので、
#   そもそも防ぎようがなく、定量化するしかありません。
# - **働いている h5i-db の機能:** コミットごとの書き換え不能なバージョン管理（この問いを答え
#   られるものにしている到着の履歴）、O(1) のタイムトラベル（2回の実行がどちらも安い）、そして
#   プレビューできる `plan_replace_range`（訂正が静かに上書きされず監査可能になる）。

# %%
db.close()
