# %% [markdown]
# # メンテナンス: verify、compact、vacuum でストアを健全に保つ
#
# バージョン管理データベースは、ちょっと変わった取引をします。データを上書きしないので、
# マニフェストもセグメントも履歴も溜まっていきます。それが売りです。ただしその結果、
# 3つの問いに正直に答える必要が出てきます。
#
# 1. **データは無事か。** `verify()` はマニフェストのチェックサム連鎖をたどります。
#    `verify(deep=True)` なら保存されている全バイトを取り直して照合します。
# 2. **なぜクエリが遅くなったのか。** ストリーミング取り込みは小さな Parquet セグメントを
#    大量に残します。`compact()` がそれを1つに併合します。もちろん新しいバージョンとして。
# 3. **ディスクはどこへ消えて、どれだけ戻るのか。** `vacuum()` が回収するのは*参照されて
#    いない*オブジェクトだけです。では古いバージョンはどうなるのか。仮定せずに、
#    実際に試して確かめます。
#
# データセットはわざと雑に扱います。1週間ぶんのティックを、バッチ化していないフィード
# ライタがやりそうなとおり、150個の細切れコミットで append します。

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
# ## 1. バッチ化していない書き手を模擬する: 150個の細切れコミット
#
# コミットのたびにマニフェストと最低1つのセグメントが書かれます。約2,000行の append を
# 150回やると、テーブルには150個の小さな Parquet ファイルが残ります。以後のクエリは
# そのすべてについて、ファイルを開き、フッタを解析し、併合の工程を通ります。

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=5, trades_per_day=20_000, seed=7)
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
# ## 2. `verify()`: チェックサムの連鎖
#
# 浅い `verify` はテーブルのマニフェスト連鎖を検査します。各マニフェストは1つ前の
# マニフェストと、自分が参照するすべてのセグメントに対してコミットしているので、ファイルが
# 途中で切れていても、コミットが失われていても、マニフェストが改竄されていても連鎖が
# 壊れます。読むのはメタデータだけ（`bytes_checked: 0`）なので、開くたびに走らせても
# 気にならない安さです。`deep=True` はさらに、全セグメントのバイトを読み直してチェック
# サムを取り直します。定期的なビット腐敗の監査で、値段もそれなりです。

# %%
print("shallow:", db.verify("trades"))
print("deep   :", db.verify("trades", deep=True))

# %% [markdown]
# deep verify で `problems` が空なら、それは誰にでも差し出せる証明になります。ディスク上の
# バイトは、どのコミットも署名したとおりのバイトです。
#
# ## 3. `compact()`: 散らかったセグメントを畳む
#
# 代表的な集約を150セグメントのテーブルに対して計り、compact して、もう一度計ります。
# コンパクションは*同じ行*を適切なサイズのセグメント1つに書き直し、その結果を新しい
# バージョンとしてコミットします。履歴には触れないので、取り込みが残骸を残したらいつでも
# 安全にかけられます。

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
# 正直な観察を2つ。第一に、ここでの速度差は本物ですが控えめです。セグメントあたりの
# オーバーヘッドはセグメント数に比例するので、ティックごとにコミットした1日ぶん（数万
# セグメント）なら、150個どころの痛みではありません。第二に、ディレクトリは*大きく*
# なりました。150個の古いセグメントはバージョン1〜150から参照されたままで、compact 後の
# テーブルは全行のコピーをもう1つ抱えています。バージョン履歴はストレージ上のお買い得品で、
# コンパクションはそれを壊しません。

# %%
print(f"db dir after compact: {du_mb(db.path):.1f} MiB")
v75 = db.read("trades", version=75)
print(f"version 75 still readable: {len(v75):,} rows (head has {head['rows']:,})")

# %% [markdown]
# ## 4. `vacuum()`: 何を回収し、何を回収しないか
#
# 自然に湧く不安は、vacuum が履歴とディスクを引き換えにするのではないか、というものです。
# 仮定せずに試しましょう。compact 直後に、`grace_seconds=0` というもっとも攻撃的な設定で
# ドライラン（`apply=False`）します。

# %%
db.vacuum("trades", grace_seconds=0, apply=False)

# %% [markdown]
# **候補はゼロです。** 151個のセグメントはどれかのバージョンのマニフェストから参照されて
# いて、vacuum が集めるのは*参照されていない*オブジェクトだけだからです。破棄された変更
# プランのステージングファイルや、中断された取り込みの残骸がそれにあたります。このビルドの
# vacuum はバージョン履歴を刈りません。どれだけ攻撃的に vacuum しても、すべてのバージョンは
# 読めるままです。保持期間は暗黙に決めるのを拒む方針で、監査で問われうるものについては
# これが正しい既定値でしょう。
#
# では、vacuum が本来相手にするゴミを作ってみます。replace プランをステージングし
# （この時点で修復済みセグメントがストレージに書かれます）、そのまま破棄します。
# ステージングされたセグメントは、もう何からも参照されていません。

# %%
lo = int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000)
hi = int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").value // 1000)
window = db.read("trades", time_start=lo, time_end=hi)

plan = db.plan_replace_range("trades", lo, hi, data=window, note="staged then abandoned")
plan.discard()

print("dry run, default grace (1h):", db.vacuum("trades", apply=False))
print("dry run, grace_seconds=0   :", db.vacuum("trades", grace_seconds=0, apply=False))

# %% [markdown]
# 既定の1時間の猶予期間が、この孤児を隠します。vacuum は若いファイルに触れません。
# *いま*参照されていないように見えるファイルが、別のプロセスが数秒後に公開するコミットの
# 一部かもしれないからです。`grace_seconds=0` にして初めて、破棄したプランのセグメントが
# 候補として現れます（ここでは他に書き手がいないので安全です）。実際に回収しましょう。

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
# ## 5. スナップショット: 名前とチェックサムの付いた固定点
#
# ここでは vacuum が履歴を食べない以上、スナップショットは防御のために要るものではあり
# ません。メンテナンス上の役どころが違います。`snapshot(name)` は、選んだテーブルの先頭
# バージョン*とマニフェストのチェックサム*を、消えない名前の下に記録します。人が引用できる
# 読み取り点（「EOD カット」など）であり、あとから照合できる完全性の錨でもあります。保留中の
# 変更プランも同じ扱いを受け、ステージングされたセグメントは適用・破棄・期限切れ（7日の
# TTL）まで vacuum から保護されます。

# %%
snap = db.snapshot("eod-2026-06-05", tables=["trades"], note="post-compact EOD cut")
entry = next(iter(snap["entries"].values()))
print(f"snapshot {snap['name']!r} pins {entry['table_name']} @ v{entry['sequence']}")
print(f"manifest checksum: {entry['manifest_checksum'][:16]}…")

db.table("trades", snapshot="eod-2026-06-05").select(count_star().alias("rows")).to_pandas()

# %% [markdown]
# ## うまく回るメンテナンスの周期
#
# - **開くたび／毎時**: 浅い `verify`。メタデータだけなので実質タダです。
# - **取り込みセッションごと**: 小さなコミットで流れ込んだテーブルを `compact` します
#   （レシピ07のバッチサイズの目安を守れば、その必要は減ります）。
# - **毎日、業務時間外**: `vacuum(apply=False)` で候補一覧を確認してから、既定の猶予期間の
#   まま `apply=True`。書き手が動いている可能性があるあいだは、`grace_seconds=0` を絶対に
#   渡さないでください。
# - **毎週／証明を出す前**: `verify(deep=True)` と、守る必要のあるテーブルの名前付き
#   スナップショット。
#
# ## まとめ
#
# - `verify()` はマニフェストのチェックサム連鎖を安く証明します。`deep=True` にすると
#   ビット腐敗の全面監査になり、空の `problems` リストが受領証になります。
# - 小さなコミットはストレージだけでなく*クエリ*への課税です。`compact()` は150個の
#   セグメントを1つに併合しました。ただの新しいバージョンとして、測れるだけの高速化つき、
#   履歴の損失ゼロで。
# - `vacuum()` が集めるのは参照されていないオブジェクトだけです。破棄されたプランの
#   ステージングや、中断された書き込みの残骸がそれにあたります。`grace_seconds=0,
#   apply=True` でさえ過去のバージョンをすべて残すことを確認しました。履歴の刈り取りは、
#   このビルドの vacuum がやる仕事ではありません。
# - 猶予期間は並行する書き手のためのクラッシュ安全策です。ゼロにしてよいのは、書き手が
#   自分だけのメンテナンス時間帯だけです。
# - スナップショットは名前とチェックサムの付いた読み取り点です。EOD カットや証明のために
#   引用できる成果物であって、vacuum の回避策ではありません。

# %%
db.close()
