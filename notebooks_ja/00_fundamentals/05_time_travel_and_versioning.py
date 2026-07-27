# %% [markdown]
# # タイムトラベルとバージョン管理: そのバックテストはどのバージョンを見たのか
#
# h5i-db のテーブルへの書き込みは、append でも write でも delete でも restore でも、
# すべてアトミックなコミットとして新しい書き換え不能バージョンを作ります。古い
# バージョンが書き換えられることはなく、読み出しはログの再生ではなくマニフェスト参照
# 1回の O(1) です。クオンツの仕事にとってこれは、「バックテストは価格ファイルの*どこかの*
# 状態で走った」と「バックテストは14:02:31 UTC にコミットされたバージョン7で走った。誰でも
# 同じものを読み直せる」の違いになります。このレシピはバージョン管理の全体を扱います。
#
# 1. `versions()` の中身
# 2. タイムトラベルの3つの入口 — バージョン番号、コミットの実時刻（`as_of`）、名前付き
#    スナップショット。Python からも SQL からも使えます
# 3. ベンダーによる訂正を、監査用の注記を添えた `write()` として実施し、2つのバージョンの
#    差分を SQL で取る
# 4. 履歴を消さずに*足す*ロールバックとしての `restore()`

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_timetravel"), create=True)

# %% [markdown]
# ## 1. 日次価格パネルを3回に分けて読み込む
#
# 10銘柄・120日ぶんの日次 OHLCV パネルを、40日ずつ3回に分けて append します。実際の価格
# ファイルが1納品ずつ育っていくのと同じ形です。`note=` はコミットに人が読めるラベルを
# 付けます。`versions()` に現れるので、これが世界でいちばん安い監査証跡になります。

# %%
prices = cu.make_daily_prices(symbols=[f"STK{i:03d}" for i in range(10)], days=120)
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])

n = len(prices)
db.append("prices", prices.slice(0, n // 3), note="vendor delivery: days 1-40")
db.append("prices", prices.slice(n // 3, n // 3), note="vendor delivery: days 41-80")
db.append("prices", prices.slice(2 * (n // 3)), note="vendor delivery: days 81-120")

# %% [markdown]
# ## 2. `versions()` の中身
#
# 1エントリが1コミットです。`sequence` は `read` や `h5i()` に渡すバージョン番号、`op` は
# 何が起きたか（`create`、`append`、`write`、`delete_range`、`replace_range`、`restore`、
# `compact`）、`committed_at_ns` はコミットの実時刻です。`rows`／`bytes`／`segments` が
# 表すのは*そのバージョン時点の*テーブルであって、差分ではありません。

# %%
hist = pd.DataFrame(db.versions("prices"))
hist["committed_at"] = pd.to_datetime(hist["committed_at_ns"], utc=True)
hist[["sequence", "op", "committed_at", "rows", "segments", "note"]]

# %% [markdown]
# ## 3. タイムトラベルの3つの入口
#
# - **バージョン番号で。** 正確なので、実行メタデータに残すならこれです。
# - **コミット時刻で**（`as_of`、RFC 3339）。「朝9時の時点で我々は何を知っていたか」を
#   問う入口で、その時刻以前でいちばん新しいコミットに解決されます。
# - **SQL で。** `h5i()` テーブル関数はどちらの形も受け取ります（スナップショット名も。
#   セクション6を参照）。だから古い状態と新しい状態が1つのクエリの中で出会えます。

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
# まず簡単な「バックテスト」として、銘柄別の日次平均リターンを年率換算し、そのまま
# ライブのテーブルから計算します。結果と一緒に先頭バージョンも記録しておきます。この
# レシピが主張したいのは、要するにこの習慣です。

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
# 続いてベンダーが2日目の終値を訂正してきたとします（引けのオークションで誤った約定が
# 出て、+0.25% で修正された、といった話です）。定石は、訂正済みのパネル全体を注記付きで
# `write()` することです。`write` はテーブルの中身を*新しいバージョンとして*置き換える
# ので、訂正前のテーブルは整数1つ隣にそのまま残ります。

# %%
df = prices.to_pandas()
day2 = df["ts"].unique()[1]
df.loc[df["ts"] == day2, "close"] *= 1.0025
corrected = pa.Table.from_pandas(df, schema=prices.schema, preserve_index=False)

commit = db.write("prices", corrected, note="vendor restatement: day-2 closes +25bp")
v_post = commit["sequence"]
print(f"restatement committed as version {v_post}")

# %% [markdown]
# では、何がどう変わったのか。2つのバージョンを1文でジョインします。エクスポートも2つ目の
# データベースも要りません。読み取り点を固定した `db.table(...)` を2つ並べれば両方の状態が
# 同じクエリに入り、ジョインはそれぞれを `l` と `r` という別名で扱います。

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
# 訂正はバックテストの結果を変えます。わずかに、静かに、そして CSV を上書きしていたなら
# 取り返しがつかない形で。ここでは*固定した*バージョンに対して走らせ直せば、元の数字が
# そのまま再現します。

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
# 訂正を適用する日を間違えていた、と後から分かったとします。`restore(version)` は古い
# バージョンを新しい先頭にします。ただし*新しいコミットとして*です。何も消えません。
# まずい write もログに残り、誰の仕業か辿れて差分も取れます。「先週の火曜、価格ファイルを
# 触ったのは誰だ」と聞かれたときに欲しいのは、まさにこれでしょう。

# %%
db.restore("prices", v_pre)

hist = pd.DataFrame(db.versions("prices"))
hist[["sequence", "op", "rows", "note"]]

# %%
# The head is byte-identical to the pre-restatement version:
assert db.read("prices").equals(db.read("prices", version=v_pre))
print("head content == version", v_pre)

# %% [markdown]
# ## 6. 名前付きスナップショット: 名前の付いた読み取り点
#
# バージョン番号は正確ですが、名無しです。`snapshot(name)` は選んだテーブルの現在の
# バージョンを、実行台帳にもメールにもコンプライアンス報告書にも書ける名前で固定します。
# SQL から直接クエリすることもできます。

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
# - `versions()` が監査証跡です。どのコミットにもシーケンス番号、操作、実時刻があり、
#   `note=` を面倒がらなければ理由も残ります。キーワード引数1つぶんの手間です。
# - 読み取り点は3つ、考え方は1つ。正確さが要るなら `version=`、「時刻Tに我々は何を
#   知っていたか」なら `as_of=`、人が口にする必要があるものにはスナップショット名。
#   3つとも `db.read` でも SQL の `h5i()` でも使えます。
# - 訂正は `write()` と注記で行います。訂正と原本が同居し、SQL のジョイン1つで何が
#   変わったかがはっきりします。
# - `restore()` は、巻き戻された当のものの記録を消さずに先頭を巻き戻します。
# - これらすべての元が取れる習慣は1つ、**リサーチ結果の隣にバージョン番号を書き残す**
#   ことです。固定したバージョンで走らせ直せば、結果はビット単位で再現します。詳しい
#   型は `03_risk_and_production/02` にあります。

# %%
db.close()
