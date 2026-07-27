# %% [markdown]
# # メンテナンス: verify、compact、vacuum
#
# バージョン管理付きのデータベースは、変わった取引をしています。データを上書きしないので、
# マニフェストもセグメントも履歴も溜まっていきます。それが売りです。同時に、メンテナンスの
# 問いが3つ、正直な答えを要求してきます。
#
# 1. **データは壊れていないか。** `verify()` はマニフェストのチェックサム連鎖をたどります。
#    `verify(deep=True)` は保存された全バイトを取り直して照合します。
# 2. **なぜクエリが遅くなったのか。** ストリーミングの取り込みは小さな Parquet セグメントを
#    大量に残します。`compact()` がそれを1つに併合します。これも新しいバージョンです。
# 3. **ディスクはどこへ消えて、何が戻ってくるのか。** `vacuum()` が回収するのは*参照されて
#    いない*オブジェクトだけです。古いバージョンがどうなるのかは、仮定せずに確かめます。

# %%
import time
from pathlib import Path

import pandas as pd

import h5i_db
from h5i_db import count_star
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_maintenance"), create=True)


def du_mb(path: str) -> float:
    """Directory size in MiB - the du -s of this recipe."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20


def bench(fn, repeat: int = 3) -> float:
    """Min-of-N wall time: the standard cheap timing harness."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の1週間ぶんのティックデータで、1行が1約定です。これをわざと雑に扱います。
# バッチ化していない書き手がやるように、150回の小さなコミットに分けて追記します。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7)
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# ## 2. バッチ化しない書き手を再現する: 150個の小さなコミット
#
# コミット1件ごとにマニフェストと最低1つのセグメントが書かれます。約2,000行ずつ150回
# append すれば、テーブルには小さな Parquet ファイルが150個残ります。1つ1つが、後続の
# すべてのクエリにとってファイルのオープンであり、フッタのパースであり、マージの一段です。

# %%
db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])

N = 150
n = len(trades)
step = n // N
for i in range(N):
    db.append("trades", trades.slice(i * step, step if i < N - 1 else n - (N - 1) * step))

head = db.versions("trades")[-1]
print(f"{head['rows']:,} rows across {head['segments']} segments, "
      f"{head['bytes'] / 2**20:.1f} MiB of live data, db dir = {du_mb(db.path):.1f} MiB")

# %% [markdown]
# ## 3. `verify()`: チェックサムの連鎖
#
# 浅い `verify` はテーブルのマニフェスト連鎖を確認します。各マニフェストは1つ前のマニフェスト
# と、自分が参照するすべてのセグメントにコミットしているので、ファイルの途中切れ、コミットの
# 欠落、マニフェストの改竄はどれも連鎖を壊します。読むのはメタデータだけ（`bytes_checked: 0`）
# なので、開くたびに走らせても平気なくらい安上がりです。
#
# `deep=True` を付けると、さらに全セグメントのバイトを読み直してチェックサムを取り直します。
# こちらは定期的なビット腐敗の監査で、値段もそれなりです。

# %%
print("shallow:", db.verify("trades"))
print("deep   :", db.verify("trades", deep=True))

# %% [markdown]
# deep な verify が返す空の `problems` リストは、そのまま誰にでも渡せる証明書です。ディスク
# 上のバイトは、すべてのコミットが署名したバイトそのものだ、ということです。
#
# ## 4. `compact()`: セグメントの散らかりを片付ける
#
# 150セグメントのテーブルに対して代表的な集約の時間を測り、圧縮して、もう一度測ります。
# 圧縮は*同じ行*を1つの手頃なサイズのセグメントに書き直し、その結果を新しいバージョンとして
# コミットします。履歴には触れないので、取り込みが散らかしたと思ったらいつでも走らせて
# 構いません。

# %%
QUERY = """
    SELECT time_bucket('5m', ts) AS bar, symbol,
           vwap(price, size) AS vwap, sum(size) AS volume
    FROM trades GROUP BY bar, symbol ORDER BY bar, symbol
"""

t_before = bench(lambda: db.sql(QUERY).to_arrow())

commit = db.compact("trades", note="merge 150 streaming appends")
print({k: commit[k] for k in ("sequence", "op", "segments_total", "segments_added")})

t_after = bench(lambda: db.sql(QUERY).to_arrow())
print(f"\n5m-bar rollup: {t_before * 1e3:.1f} ms on 150 segments "
      f"-> {t_after * 1e3:.1f} ms on 1 segment  ({t_before / t_after:.1f}x)")

# %% [markdown]
# 正直な観察を2つ。
#
# ここでの高速化は本物ですが、控えめです。セグメントあたりのオーバーヘッドはセグメント数に
# 比例するので、1ティック1コミットを1日続けた場合、つまり数万セグメントのほうが、150個より
# はるかに痛みます。
#
# そしてディレクトリは*大きくなりました*。古い150セグメントはバージョン1〜150から参照され
# たままですし、圧縮後のテーブルは全行のコピーをもう1つ抱えます。バージョン履歴は安い買い物
# で、圧縮はそれを壊しません。

# %%
print(f"db dir after compact: {du_mb(db.path):.1f} MiB")
v75 = db.read("trades", version=75)
print(f"version 75 still readable: {len(v75):,} rows (head has {head['rows']:,})")

# %% [markdown]
# ## 5. `vacuum()`: 何を回収し、何を回収しないのか
#
# 自然に湧く不安は、vacuum が履歴とディスクを交換してしまうのではないか、というものです。
# 仮定せずに確かめましょう。圧縮の直後に、いちばん強い設定である `grace_seconds=0` で、
# ドライラン（`apply=False`）をかけます。

# %%
db.vacuum("trades", grace_seconds=0, apply=False)

# %% [markdown]
# **候補はゼロです。** あの151個のセグメントはどれも、どこかのバージョンのマニフェストから
# 参照されています。vacuum が集めるのは*参照されていない*オブジェクトだけです。破棄された
# 変更プランのステージングファイルや、中断された取り込みの残骸がそれにあたります。
#
# このビルドでは、vacuum がバージョン履歴を刈ることはありません。どれだけ強く vacuum を
# かけても、すべてのバージョンは読めるままです。保持期間はポリシーの決定であり、vacuum は
# それを暗黙に決めることを拒みます。監査で問われうるものについては、これが正しい既定です。
#
# では、vacuum が存在する理由であるゴミのほうを作ってみましょう。置換プランをステージすると
# 修復後のセグメントがただちにストレージへ書かれるので、それを破棄します。ステージされた
# セグメントは、これでどこからも参照されなくなりました。

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
hi = int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").value // 1000)
window = db.read("trades", time_start=lo, time_end=hi)

plan = db.plan_replace_range("trades", lo, hi, data=window, note="staged then abandoned")
plan.discard()

print("dry run, default grace (1h):", db.vacuum("trades", apply=False))
print("dry run, grace_seconds=0   :", db.vacuum("trades", grace_seconds=0, apply=False))

# %% [markdown]
# 既定の1時間の猶予期間は、この孤児を隠します。vacuum は若いファイルには触れません。いま
# 参照されていないように見えるファイルが、別のプロセスがあと数秒で公開するコミットの一部
# かもしれないからです。
#
# 破棄されたプランのセグメントを候補として見せるのは `grace_seconds=0` だけで、ここでは他に
# 書き手がいないので安全です。実際に回収してみましょう。

# %%
size_before = du_mb(db.path)
result = db.vacuum("trades", grace_seconds=0, apply=True)
print(result)
print(f"db dir: {size_before:.2f} MiB -> {du_mb(db.path):.2f} MiB")

# %%
# The load-bearing claim, checked: vacuum deleted the orphan and nothing else.
versions = db.versions("trades")
for v in versions:  # every version's manifest still resolves and reads
    db.read("trades", version=v["sequence"], limit=5)
assert db.vacuum("trades", grace_seconds=0, apply=False)["candidates"] == []
print(f"all {len(versions)} versions still readable after vacuum; "
      f"deep verify clean: {db.verify('trades', deep=True)['problems'] == []}")

# %% [markdown]
# ## 6. スナップショット: 名前とチェックサムの付いたピン
#
# ここでは vacuum が履歴を食べない以上、スナップショットは防御のために必須ではありません。
# メンテナンス上の役割は別のところにあります。
#
# `snapshot(name)` は、選んだテーブルの先頭バージョンを*マニフェストのチェックサムごと*、
# 永続的な名前で記録します。これで「EOD の断面」のように人間が口にできる読み取り点と、
# あとから照合し直せる完全性のアンカーが手に入ります。
#
# 保留中の変更プランも同じ扱いを受けます。ステージされたセグメントは、適用・破棄・失効の
# いずれかまで vacuum から守られます。TTL は7日です。

# %%
snap = db.snapshot("eod-2026-06-05", tables=["trades"], note="post-compact EOD cut")
entry = next(iter(snap["entries"].values()))
print(f"snapshot {snap['name']!r} pins {entry['table_name']} @ v{entry['sequence']}")
print(f"manifest checksum: {entry['manifest_checksum'][:16]}…")

db.table("trades", snapshot="eod-2026-06-05").select(count_star().alias("rows")).to_pandas()

# %% [markdown]
# ## 回るメンテナンスの周期
#
# - **開くたび、あるいは1時間ごと:** 浅い `verify`。メタデータだけなので実質ただです。
# - **取り込みのセッションごと:** 小さなコミットが積み重なったテーブルに `compact`。レシピ07
#   のバッチサイズの目安を守れば、その頻度は下がります。
# - **毎日、業務時間外に:** `vacuum(apply=False)` で候補一覧を確認し、既定の猶予期間のまま
#   `apply=True`。書き手が動いているかもしれない時間帯に `grace_seconds=0` は使わないこと。
# - **毎週、あるいは証明を出す前に:** `verify(deep=True)` と、守る必要のあるテーブルの名前
#   付きスナップショット。
#
# ## まとめ
#
# - `verify()` はマニフェストのチェックサム連鎖を安く証明します。`deep=True` はそれを完全な
#   ビット腐敗の監査に変え、空の `problems` リストが受領証になります。
# - 小さなコミットはストレージだけでなく*クエリ*への課税です。`compact()` は150セグメントを
#   1つに併合しました。ごく普通の新しいバージョンとしてで、測れるだけの高速化があり、履歴は
#   1つも失われていません。
# - `vacuum()` が集めるのは参照されていないオブジェクトだけです。破棄されたプランのステージ
#   ングや、中断された書き込みの残骸がそれです。`grace_seconds=0, apply=True` でさえ、
#   すべての過去バージョンが読めるままであることを確認しました。履歴の刈り取りは、このビルド
#   の vacuum の仕事ではありません。
# - 猶予期間は同時に書く相手に対するクラッシュ安全のためです。ゼロにするのは、書き手が自分
#   だけのメンテナンス時間帯に限ります。
# - スナップショットは名前とチェックサムの付いた読み取り点です。EOD の断面や証明のために
#   引用できる成果物であって、vacuum の回避策ではありません。

# %%
db.close()
