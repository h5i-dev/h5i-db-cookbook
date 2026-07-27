# %% [markdown]
# # ストリーミング append と `tail()`: バージョン管理ストアの上のライブフィード
#
# 追記のみの履歴を持つ h5i-db のテーブルは、そのままメッセージログとしても使えます。
# append 1回が1コミット、コミットは厳密に順序づけられ、最後に処理したバージョンを覚えて
# いる読み手は、それ以降に追加された行だけを正確に取れます。タイムスタンプをカーソル代わりに
# する当てずっぽうも、行の取りこぼしや二重取りもありません。これで「日中のティックは
# 1日遅れでリサーチ用データベースに届く」が「リサーチ用データベース*こそが*フィードの
# 消費者だ」に変わります。このレシピでは次を行います。
#
# 1. ライブのティックフィードを、チャンク単位で append する書き手として模擬する
# 2. SQL の `tail('trades', after_version, poll_ms)` で消費し、`LIMIT` が任意ではない
#    理由を見る
# 3. 本番で使われる、もっと単純な高水位マーク方式のポーリングを示す
# 4. 1分足を**インクリメンタルに**維持し、最新チャンクが触れたバケットだけを再計算する
# 5. 追記のみの連鎖をわざと壊して、エラーを読む

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, time_bucket, vwap
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_streaming"), create=True)

# %% [markdown]
# ## 1. 1営業日を、フィードのチャンクに刻む
#
# `tail` が動く前提は**追記のみのバージョン連鎖**です。対象のバージョン範囲に write、
# delete、replace、restore が1つでも入ると、増分の差分は計算できなくなります（セクション5で
# 実演します）。だからストリーミング用のテーブルは、作りからして追記のみにしておくべきです。
# 生フィードはたいていそうなっています。
#
# ここでは書き手と読み手を1つのプロセスの中で交互に動かします。データベース側の制約では
# ありません（別プロセスのハンドル同士は楽観的並行制御で協調します。マルチライタのレシピを
# 参照）。このノートブックを決定的にするための都合です。

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT"], days=1, trades_per_day=15_000, seed=7)
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

N_CHUNKS = 6
n = len(trades)
step = n // N_CHUNKS
chunks = [trades.slice(i * step, step if i < N_CHUNKS - 1 else n - (N_CHUNKS - 1) * step)
          for i in range(N_CHUNKS)]
print(f"{n:,} trades -> {N_CHUNKS} feed chunks of ~{step:,} rows")

# %% [markdown]
# ## 2. 書き手が append し、読み手が tail する
#
# 読み手のカーソルはタイムスタンプではなく**バージョン番号**です。
# `tail('trades', after_version, poll_ms)` は `after_version` より後にコミットされた行を
# すべて流し、`poll_ms` ごとに新しいコミットを見にいきます。つまりこれは*終わりのない*
# ソースです。`LIMIT` を付けなければクエリは完了せず、次のコミットをただ待ち続けます。
# そこで読み手は先に `versions()` を覗いて実際に取れる行数を知り、ちょうどその `LIMIT` を
# 付けて tail します。
#
# 同時に、1分足も行が届くたびにインクリメンタルに維持します。ただし新しいバッチが触れた
# バケットだけです。影響を受ける最初のバケットは、バッチ内でいちばん早い行を含む分です。
# それより前はすべて確定済みで、二度と再計算しません。`read(time_start=...)`（生の
# マイクロ秒）が、読み直しをテーブルの末尾部分だけに絞ります。

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
# インクリメンタルなバーは、ゼロから集計したものと一致しなければなりません。テーブル全体に
# `time_bucket` の集約を1回かけたものが正解です。

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
# ## 3. `tail` は終わりがない — LIMIT を軽く見ない
#
# コミットされている数より多くの行を要求すると、`tail` はフィード消費者として正しい
# 振る舞いをします。つまり次のコミットを*待ちます*。ノートブックではそれが永久停止に
# なるので、`tail` には必ず、取れると分かっている量に合わせた `LIMIT` と、保険の
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
# ストリーミングにはもう1つ注意点があります。`tail` が作るのは終わりのないストリームなので、
# パイプラインを塞ぐ演算子（集約やソート）を直接その上に載せられません。先に行を消費して、
# クライアント側で集約します。上のインクリメンタルなバー生成のループがまさにそれです。
#
# ## 4. 高水位マーク方式（`tail` なし）
#
# 本番のポーラーの多くは、ブロックするストリームを必要としません。決まった間隔で先頭
# バージョンと自分のカーソルを比べ、新しい行だけを範囲読み出しします。`read(time_start=...)`
# は時刻列でプルーニングするので、「高水位マークより後を全部読み直す」コストは、テーブルの
# 大きさに関係なく、新しいデータの量だけに比例します。まず手を伸ばすべきはこちらで、`tail` が
# 効くのはプッシュ型でブロックする配送が欲しいときです。

# %%
cursor_version, hwm_us = 2, None  # pretend we last processed version 2

head = db.versions("trades")[-1]["sequence"]
if head > cursor_version:
    hwm_us = int(pa.compute.max(db.read("trades", version=cursor_version)["ts"]).value)
    new_rows = db.read("trades", time_start=hwm_us + 1)
    print(f"cursor v{cursor_version} -> head v{head}: {len(new_rows):,} new rows "
          f"(matches tail-based count: {len(new_rows) == n - 2 * step})")

# %% [markdown]
# ## 5. 追記のみの連鎖を壊す
#
# append 以外のコミットを1つ入れるだけで（ここでは1分ぶんの `delete_range`）、そのバージョンを
# またぐ `tail` は増分の差分を計算できなくなります。エラーは大きな声で、しかも具体的です。
# そこが肝心なところで、フィード消費者を殺すのは静かな欠損だからです。

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
db.plan_delete_range("trades", lo, lo + 60_000_000, note="ops correction").apply()

try:
    db.sql("SELECT * FROM tail('trades', 0, 25) LIMIT 10", timeout=5)
except h5i_db.H5iError as e:
    print(f"{type(e).__name__}: code={e.code!r}")
    print(str(e))

# %% [markdown]
# メッセージは問題のバージョンを名指しし、代替手段も示します。変更後の状態を全走査するか、
# そのバージョンを含まない範囲を tail するかです。tail し続けたいテーブルは
# `set_policy(direct_delete=False, direct_write=False, ...)` で守り（レシピ06）、訂正は
# 下流の*クリーン済み*テーブルに回してください。
#
# ## バッチサイズの目安
#
# コミットのたびにマニフェストと最低1つの Parquet セグメントが書かれます。だから予算を
# 組む対象は1秒あたりのコミット数です。1コミットあたりの行数ではありません。
#
# - **ティックはまとめてコミットします**（1秒ぶん、venue のメッセージブロック単位、
#   1,000〜10,000行）。1行1コミットは病的で、小さなセグメントが数千個できてすべての
#   スキャンが遅くなります。
# - 読み手はコミット単位で原子的に見るので、バッチサイズはそのまま配送レイテンシの下限に
#   なります。書き手1秒バッチと読み手 `poll_ms=25` なら、端から端まで約1秒です。
# - 小さなストリーミング append を1日重ねたあとは、`db.compact("trades")` が小さな
#   セグメントを併合します（レシピ08）。コンパクションも他と同じバージョンですが、これを
#   またぐと追記のみの `tail` 連鎖は切れます。だから compact はストリームの途中ではなく、
#   セッションの切れ目でかけてください。
#
# ## まとめ
#
# - 追記のみのテーブルはメッセージログを兼ねます。読み手のカーソルは**バージョン番号**で、
#   `tail('t', after_version, poll_ms)` がそれ以降の行を順序どおり、ちょうど1回ずつ配送します。
# - `tail` は設計上終わりがありません。`LIMIT` は必ず `versions()` から採寸し、保険の
#   `timeout=` も置きます。集約はストリームの中では行わず、消費したあとに回します。
# - 高水位マーク方式（`versions()` の差分と `read(time_start=hwm)`）のほうが、本番の
#   既定としては単純です。時刻列のプルーニングが効くので安く済みます。
# - インクリメンタルなバー維持も同じ考えから出てきます。新しいバッチが触れたバケットだけを
#   再計算し、全体集計と一致することを確かめました。
# - `tail` は追記のみの連鎖を要求します。範囲に変更が混ざれば、黙って続けずエラーで止まります。
#   ストリーミング用テーブルは、ポリシーで追記のみに保ちましょう。

# %%
db.close()
