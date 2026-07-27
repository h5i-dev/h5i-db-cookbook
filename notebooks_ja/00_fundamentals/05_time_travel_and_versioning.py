# %% [markdown]
# # タイムトラベルとバージョニング: そのバックテストはどのバージョンを見たのか
#
# h5i-db のテーブルへの書き込みは、どれもアトミックなコミットで、書き換え不能なバージョンを
# 1つ生みます。`append` も `write` も `delete` も `restore` もそうです。古いバージョンが
# 書き換えられることはなく、読むのはログの再生ではなく O(1) のマニフェスト参照です。
#
# クオンツの仕事では、これが2つの文の差になります。「バックテストは価格ファイルの*どこかの*
# 状態で走った」か、「バックテストはバージョン7、コミット時刻 14:02:31 UTC で走り、誰でも
# それをそのまま読み直せる」か。
#
# このレシピで進めるのは次の4つです。
#
# 1. `versions()` の中身を読む
# 2. 3つの方法でタイムトラベルする。バージョン番号、コミットの実時刻（`as_of`）、名前付き
#    スナップショット。Python でも SQL でも
# 3. ベンダーによる訂正を、監査用のノート付きの `write()` として受け取り、2つのバージョンを
#    クエリ1つで差分する
# 4. `restore()` を、履歴を消すのではなく*足す*ロールバックとして使う

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_timetravel"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_daily_prices` の日足 OHLCV パネルです。10銘柄×120セッションで、1行が1銘柄
# 1セッションぶん。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、20:00 UTC |
# | `symbol` | `string` | 銘柄コード、`STK000` 〜 `STK009` |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `volume` | `int64` | 出来高（株数） |

# %%
prices = cu.make_daily_prices(symbols=[f"STK{i:03d}" for i in range(10)], days=120)
print(f"{prices.num_rows:,} rows x {prices.num_columns} columns")
prices.to_pandas().head()

# %% [markdown]
# これを40日ずつ3回に分けて読み込みます。現実の価格ファイルが育つのと同じ、1回1配信の
# かたちです。`note=` はコミットに人間が読めるラベルを付けます。`versions()` に出てきますし、
# これ以上安い監査証跡は作れません。

# %%
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])

n = len(prices)
db.append("prices", prices.slice(0, n // 3), note="vendor delivery: days 1-40")
db.append("prices", prices.slice(n // 3, n // 3), note="vendor delivery: days 41-80")
db.append("prices", prices.slice(2 * (n // 3)), note="vendor delivery: days 81-120")

# %% [markdown]
# ## 2. `versions()` の中身
#
# 1件が1コミットです。
#
# - `sequence` は `read` や `h5i()` に渡すバージョン番号です
# - `op` は何が起きたかで、`create`、`append`、`write`、`delete_range`、`replace_range`、
#   `restore`、`compact` のいずれかです
# - `committed_at_ns` はコミットの実時刻です
# - `rows`、`bytes`、`segments` は差分ではなく、*そのバージョン時点の*テーブルを表します

# %%
hist = pd.DataFrame(db.versions("prices"))
hist["committed_at"] = pd.to_datetime(hist["committed_at_ns"], utc=True)
hist[["sequence", "op", "committed_at", "rows", "segments", "note"]]

# %% [markdown]
# ## 3. 3つのタイムトラベル
#
# - **バージョン番号で。** 厳密で、実行時のメタデータに残すならこれです。
# - **コミット時刻で**（`as_of`、RFC 3339）。「朝9時の時点で何を知っていたか」に答えます。
#   その瞬間以前でいちばん新しいコミットに解決されます。
# - **SQL の中で**、`h5i()` テーブル関数を使って。どちらの形も、6節のスナップショット名も
#   受け取ります。だから過去と現在の状態を1つのクエリで出会わせられます。

# %%
v1_rows = len(db.read("prices", version=1))

# A wall-clock instant just after tranche 2 was committed:
t_after_2 = pd.Timestamp(
    db.versions("prices")[2]["committed_at_ns"] + 1, unit="ns", tz="UTC"
).isoformat()
asof_rows = len(db.read("prices", as_of=t_after_2))

print(f"version 1          : {v1_rows} rows")
print(f"as_of {t_after_2}: {asof_rows} rows")

db.sql(
    f"""
    SELECT 'version 1' AS read_point, count(*) AS rows, max(ts) AS last_day FROM h5i('prices', 1)
    UNION ALL
    SELECT 'as-of tranche 2',          count(*),        max(ts)             FROM h5i('prices', '{t_after_2}')
    UNION ALL
    SELECT 'latest',                    count(*),        max(ts)             FROM prices
    """
).to_pandas()

# %% [markdown]
# ## 4. 訂正と、それが壊さなかったバージョン
#
# まずはおもちゃの「バックテスト」から。銘柄ごとの日次リターン平均を年率化したもので、
# ライブのテーブルからそのまま計算します。結果と並べて、そのときの先頭バージョンも記録して
# おきます。このレシピ全体が主張しているのは、この習慣です。

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")


def backtest_mean_return(version=None) -> pd.DataFrame:
    """The same study, parameterized by read point rather than by SQL text."""
    return (
        db.table("prices", version=version)
        .with_columns(ret=col("close") / PREV_CLOSE - 1)
        .filter(col("ret").is_not_null())
        .group_by("symbol")
        .agg(ann_mean_ret=col("ret").mean() * 252)
        .sort("symbol")
        .to_pandas()
    )


v_pre = db.versions("prices")[-1]["sequence"]
result_original = backtest_mean_return()
print(f"backtest ran against version {v_pre}")
result_original.head(3)

# %% [markdown]
# ここでベンダーが2日目の終値を訂正します。引け間際のおかしなプリントを +0.25% 直した、
# という想定です。定石は、訂正後のパネル全体をノート付きで `write()` することです。`write`
# はテーブルの中身を*新しいバージョンとして*置き換えるので、訂正前のテーブルは整数1つ隣に
# 残ります。

# %%
df = prices.to_pandas()
day2 = df["ts"].unique()[1]
df.loc[df["ts"] == day2, "close"] *= 1.0025
corrected = pa.Table.from_pandas(df, schema=prices.schema, preserve_index=False)

commit = db.write("prices", corrected, note="vendor restatement: day-2 closes +25bp")
v_post = commit["sequence"]
print(f"restatement committed as version {v_post}")

# %% [markdown]
# では、何がどう変わったのか。2つのバージョンを1つの文でジョインします。エクスポートの
# 工程も、2つ目のデータベースも要りません。ピン留めした `db.table(...)` の読み取りを2つ
# 並べれば両方の状態が同じクエリに入り、ジョインがそれを `l` と `r` に名付けます。

# %%
new, old = db.table("prices", version=v_post), db.table("prices", version=v_pre)

(
    new.join(old, on=["ts", "symbol"])
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        close_old=col("close", relation="r"),
        close_new=col("close", relation="l"),
        delta=col("close", relation="l") - col("close", relation="r"),
    )
    .filter(col("close_new") != col("close_old"))
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# 訂正はバックテストをわずかに、そして静かに動かします。CSV を上書きしていたなら、それは
# 取り返しがつかない動き方でもありました。ここでは*ピン留めした*バージョンで走らせ直せば、
# 元の数字がそのまま再現します。

# %%
result_after = backtest_mean_return()
result_pinned = backtest_mean_return(version=v_pre)

drifted = (result_after["ann_mean_ret"] - result_original["ann_mean_ret"]).abs().max()
pinned = (result_pinned["ann_mean_ret"] - result_original["ann_mean_ret"]).abs().max()
print(f"max drift, re-run on live head : {drifted:.2e}")
print(f"max drift, re-run on version {v_pre} : {pinned:.2e}  (bit-for-bit)")
assert pinned == 0.0

# %% [markdown]
# ## 5. `restore()`: 履歴を足すロールバック
#
# さきほどの訂正が、実は違う日に当たっていたとしましょう。`restore(version)` は古いバージョン
# を新しい先頭にします。ただし*新しいコミットとして*です。
#
# 消えるものはありません。まずい書き込みもログに残り、誰の仕業かも追えますし、差分も取れ
# ます。「先週の火曜、価格ファイルを触ったのは誰か」と聞かれたときに欲しいのは、まさにこれ
# です。

# %%
db.restore("prices", v_pre)

hist = pd.DataFrame(db.versions("prices"))
hist[["sequence", "op", "rows", "note"]]

# %%
# The head is byte-identical to the pre-restatement version:
assert db.read("prices").equals(db.read("prices", version=v_pre))
print("head content == version", v_pre)

# %% [markdown]
# ## 6. 名前付きスナップショット: 名前のある読み取り点
#
# バージョン番号は正確ですが、匿名です。`snapshot(name)` は、選んだテーブルの現在のバージョン
# を名前で固定します。その名前は実行台帳にもメールにもコンプライアンスの報告書にも書けます
# し、SQL からそのまま引けます。

# %%
db.snapshot("model-run-2026-07-21", tables=["prices"], note="momentum study, run 42")

(
    db.table("prices", snapshot="model-run-2026-07-21")
    .select(
        rows=count_star(),
        first_day=col("ts").min(),
        last_day=col("ts").max(),
    )
    .to_pandas()
)

# %% [markdown]
# ## まとめ
#
# - `versions()` が監査証跡です。コミットごとに連番、操作、実時刻があり、`note=` を書く規律
#   があれば理由も残ります。その規律のコストはキーワード引数1つです。
# - 読み取り点は3つ、考え方は1つ。厳密さなら `version=`、「時刻Tに何を知っていたか」なら
#   `as_of=`、人間が口にするものにはスナップショット名。3つとも `db.read` でも SQL の
#   `h5i()` でも使えます。
# - 訂正は `write()` とノートです。訂正と原本が共存し、ジョイン1つで何が変わったかが正確に
#   出ます。
# - `restore()` は、ロールバックの対象そのものの記録を壊さずに先頭を巻き戻します。
# - 全部を支える習慣はこれです。**リサーチの結果の隣にバージョン番号を書き残す。** ピン留め
#   して走らせ直せばビット単位で再現します。レシピ `03_risk_and_production/02` がこのパターン
#   を本格的に展開します。

# %%
db.close()
