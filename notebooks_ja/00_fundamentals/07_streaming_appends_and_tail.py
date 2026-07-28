# %% [markdown]
# # ストリーミング追記と `tail()`: バージョン管理ストアの上のライブフィード
#
# 追記だけで育った h5i-db のテーブルは、そのままメッセージログとしても使えます。`append`
# 1回が1コミット、コミットは厳密に順序づけられ、最後に処理したバージョンを覚えている読み手
# は、それ以降に増えた行だけを正確に取ってこられます。タイムスタンプをカーソル代わりにする
# 当て推量も、取りこぼしも二重取得もありません。
#
# これで「日中のティックはリサーチ用データベースに1日遅れで着く」が、「リサーチ用データ
# ベース*こそ*がフィードの消費者だ」に変わります。
#
# このレシピで進めるのは次の5つです。
#
# 1. ライブのティックフィードを、チャンクごとに追記する書き手として再現する
# 2. SQL の `tail('trades', after_version, poll_ms)` で消費し、`LIMIT` が任意ではない理由を
#    確かめる
# 3. 本番でよく使われる、もっと単純な高水位マークのポーリングを見る
# 4. 1分足を**逐次的に**維持する。新しいチャンクが触ったバケットだけを計算し直す
# 5. 追記だけの連鎖をわざと壊して、エラーを読む

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, time_bucket, vwap
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_streaming"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の、2銘柄・1営業日ぶんのティックデータです。1行が1約定です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード、`AAPL` か `MSFT` |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=15_000, seed=7)
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# この1日を6つのチャンクに切り、ライブで届いてくるかのように再生します。
#
# `tail` は**追記だけのバージョン連鎖**の上で動きます。範囲の中に `write`、削除、置換、復元
# が1つでも入ると逐次差分が壊れます。5節で実演します。だからストリーミング用のテーブルは、
# 作りからして追記専用にしておくべきです。生のフィードはたいていそうなっています。
#
# ここでは書き手と読み手を1つのプロセスで順番に動かします。これはデータベースの制約では
# ありません。別プロセスのハンドル同士は楽観的並行制御で協調できます（マルチライタのレシピを
# 参照）。単にこのノートブックを決定的に保つためです。

# %%
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

N_CHUNKS = 6
n = len(trades)
step = n // N_CHUNKS
chunks = [trades.slice(i * step, step if i < N_CHUNKS - 1 else n - (N_CHUNKS - 1) * step)
          for i in range(N_CHUNKS)]
print(f"{n:,} trades -> {N_CHUNKS} feed chunks of ~{step:,} rows")

# %% [markdown]
# ## 2. 書き手が追記し、読み手が tail する
#
# 読み手のカーソルはタイムスタンプではなく、**バージョン番号**です。
# `tail('trades', after_version, poll_ms)` は `after_version` より後にコミットされた行を
# すべて流し、`poll_ms` ごとに新しいコミットを見に行きます。
#
# つまりこれは*境界のない*ソースです。`LIMIT` がなければクエリは終わりません。次のコミットを
# ただ待ち続けます。そこで読み手はまず `versions()` を覗いて実際に何行あるかを知り、ちょうど
# その数を `LIMIT` に置いて tail します。
#
# 同時に1分足も逐次的に維持しますが、更新するのは新しいバッチが触ったバケットだけです。
# 影響を受ける最初のバケットは、そのバッチのいちばん早い行が入る分です。それより前はすでに
# 閉じているので、二度と計算し直しません。生のマイクロ秒を取る `read(time_start=...)` が、
# 読み直しをテーブルの末尾だけに絞ります。

# %%
def consume_new(last_version: int, rows_seen: int) -> pd.DataFrame | None:
    """Fetch exactly the rows committed after `last_version`, or None."""
    head = db.versions("trades")[-1]
    available = head["rows"] - rows_seen
    if available == 0:
        return None
    return db.sql(
        f"SELECT * FROM tail('trades', {last_version}, 25) LIMIT {available}"
    ).to_pandas()


def recompute_bars(bars: dict, batch: pd.DataFrame) -> int:
    """Recompute 1m bars from the first bucket `batch` touches, in place."""
    first_bucket = batch["ts"].min().floor("1min")
    part = db.read("trades", time_start=int(first_bucket.value // 1000)).to_pandas()
    grp = part.groupby([part["ts"].dt.floor("1min").rename("bar"), "symbol"])
    fresh = grp.apply(
        lambda g: pd.Series({
            "high": g["price"].max(),
            "low": g["price"].min(),
            "volume": g["size"].sum(),
            "vwap": (g["price"] * g["size"]).sum() / g["size"].sum(),
        }),
        include_groups=False,
    )
    bars.update(fresh.to_dict("index"))
    return len(fresh)


bars: dict = {}
last_version, rows_seen = db.versions("trades")[-1]["sequence"], 0

for i, chunk in enumerate(chunks):
    db.append("trades", chunk)                      # -- the writer

    batch = consume_new(last_version, rows_seen)    # -- the reader
    last_version = db.versions("trades")[-1]["sequence"]
    rows_seen += len(batch)
    touched = recompute_bars(bars, batch)
    print(f"chunk {i}: consumed {len(batch):>5,} rows "
          f"({batch['ts'].min():%H:%M:%S} -> {batch['ts'].max():%H:%M:%S}), "
          f"recomputed {touched:>3} of {len(bars)} bars")

assert rows_seen == n
print(f"\nreader consumed all {rows_seen:,} rows; cursor at version {last_version}")

# %% [markdown]
# 逐次で作ったバーは、ゼロから作り直したものと一致しなければなりません。テーブル全体に対する
# `time_bucket` の集約1つが、その正解です。

# %%
full = (
    db.table("trades")
    .group_by(time_bucket("1m", col("ts")).alias("bar"), "symbol")
    .agg(
        high=col("price").max(),
        low=col("price").min(),
        volume=col("size").sum(),
        vwap=vwap(col("price"), col("size")),
    )
    .sort(["bar", "symbol"])
    .to_pandas()
)

inc = (pd.DataFrame.from_dict(bars, orient="index")
       .rename_axis(["bar", "symbol"]).reset_index().sort_values(["bar", "symbol"])
       .astype({"volume": "int64"}))  # the pd.Series aggregation upcast volume to float
pd.testing.assert_frame_equal(
    full.reset_index(drop=True), inc.reset_index(drop=True), check_like=True
)
print(f"{len(full)} incremental bars match the full recompute exactly")

# %% [markdown]
# ## 3. `tail` は境界がないので、LIMIT を守る
#
# コミット済みより多くの行を要求すると、`tail` はフィードの消費者として正しい振る舞いを
# します。つまり次のコミットを*待ちます*。ノートブックの中では、永久に固まるということです。
#
# `tail` には必ず、上のように「あると分かっている行数」に合わせた `LIMIT` と、保険の
# `timeout=` を添えてください。ここでは1行だけ多く要求し、1.5秒の期限を切ります。

# %%
try:
    db.sql(
        f"SELECT * FROM tail('trades', 0, 25) LIMIT {n + 1}",
        timeout=1.5,
    )
except h5i_db.TimeoutError as e:
    print(f"{type(e).__name__}: code={e.code!r} retryable={e.retryable}")
    print("tail kept polling for a commit that never came - the timeout, not")
    print("the LIMIT, ended the query.")

# %% [markdown]
# ストリーミングの注意点をもう1つ。`tail` は境界のないストリームを作るので、集約やソートの
# ようなパイプラインを止める演算子を、その上に直接は載せられません。先に行を消費してから
# クライアント側で集約します。上の逐次バーのループがまさにそれをやっています。
#
# ## 4. 高水位マークのパターン（`tail` は要らない）
#
# 本番のポーラーは、ブロックするストリームを必要としないことがほとんどです。決まった間隔で
# 先頭バージョンと自分のカーソルを比べ、新しい行だけを範囲読みします。
#
# `read(time_start=...)` は時刻列でプルーニングするので、「高水位マーク以降の全部」を読み
# 直すコストは、新しいデータの量に比例します。テーブルがどれだけ育っても変わりません。
# まずはこのパターンを取ってください。`tail` が効くのは、プッシュ型のブロッキング配送が
# 欲しいときです。

# %%
cursor_version, hwm_us = 2, None  # pretend we last processed version 2

head = db.versions("trades")[-1]["sequence"]
if head > cursor_version:
    hwm_us = int(pa.compute.max(db.read("trades", version=cursor_version)["ts"]).value)
    new_rows = db.read("trades", time_start=hwm_us + 1)
    print(f"cursor v{cursor_version} -> head v{head}: {len(new_rows):,} new rows "
          f"(matches tail-based count: {len(new_rows) == n - 2 * step})")

# %% [markdown]
# ## 5. 追記だけの連鎖を壊す
#
# 追記以外のコミット、ここでは1分ぶんの `delete_range` を適用すると、そのバージョンをまたぐ
# `tail` は逐次差分を計算できなくなります。エラーは大きく、具体的です。そこが肝心なところ
# で、静かな欠落こそがフィードの消費者を殺します。

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
db.plan_delete_range("trades", lo, lo + 60_000_000, note="ops correction").apply()

try:
    db.sql("SELECT * FROM tail('trades', 0, 25) LIMIT 10", timeout=5)
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: code={e.code!r}")
    print(str(e))

# %% [markdown]
# メッセージは問題のバージョンを名指しし、逃げ道も示します。変更後の状態をフルスキャンする
# か、そのバージョンを含まない範囲を tail するかです。
#
# tail できる状態を保ちたいテーブルは、レシピ06の
# `set_policy(direct_delete=False, direct_write=False, ...)` で守り、訂正は下流の*クリーン済み*
# テーブルへ回してください。
#
# ## バッチサイズの目安
#
# コミット1件ごとにマニフェストと最低1つの Parquet セグメントが書かれます。だから見積もる
# 単位は、秒あたりのコミット数になります。
#
# - **ティックはコミット単位にまとめます。** 1秒ぶん、取引所のメッセージブロック単位、
#   あるいは1,000〜10,000行ぶん。1行1コミットは病的です。小さなセグメントが何千とできて、
#   すべてのスキャンが遅くなります。
# - 読み手はコミット単位でアトミックに見るので、バッチサイズはそのまま配送レイテンシの下限
#   になります。書き手が1秒バッチ、読み手が `poll_ms=25` なら、端から端でおよそ1秒です。
# - 小さなストリーミング追記が1日続いたら、`db.compact("trades")` で小さなセグメントを併合
#   します（レシピ08）。圧縮も他と同じバージョンですが、それをまたぐ追記だけの `tail` 連鎖は
#   壊れます。圧縮はセッションの切れ目でかけ、ストリームの最中は避けてください。
#
# ## まとめ
#
# - 追記だけのテーブルはメッセージログとしても使えます。読み手のカーソルは**バージョン番号**
#   で、`tail('t', after_version, poll_ms)` がそれ以降の行を順序どおり、ちょうど1回ずつ届け
#   ます。
# - `tail` は設計上、境界がありません。`LIMIT` は必ず `versions()` から決め、保険の
#   `timeout=` を置き、集約は行を消費したあとでやります。
# - 高水位マークのパターン、つまり `versions()` の差分と `read(time_start=hwm)` の組み合わせ
#   が、本番での単純な既定解です。安く済むのは時刻列のプルーニングのおかげです。
# - 逐次的なバーの維持も同じ考えから出てきます。新しいバッチが触ったバケットだけを計算し
#   直す。ここではそれが全体の集計と一致することを確認しました。
# - `tail` は追記だけの連鎖を要求し、範囲に変更が入ると静かにではなく大きな音を立てて失敗
#   します。ストリーミング用のテーブルは、ポリシーで追記専用に保ちましょう。

# %%
db.close()
