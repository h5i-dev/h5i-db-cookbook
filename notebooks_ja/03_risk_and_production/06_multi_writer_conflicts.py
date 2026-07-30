# %% [markdown]
# # 複数ライターの協調: 楽観ロック、競合、リトライ
#
# h5i-db のデータベースはディレクトリで、2つのプロセスが同時に開くのを止めるものはありません。
# フィードハンドラと訂正のジョブ、あるいは同僚2人のノートブック。
#
# 同時書き込みへの答えが*楽観的並行制御*です。どのコミットも `expected_version` を持てて、
# 読んだあとにテーブルの先頭が動いていれば、黙って混ざるかわりに明示的な `ConflictError` で
# 拒否されます。
#
# チームにとってはこれが正しい取引です。ポジションや評価のテーブルでの更新の取りこぼしは、
# 静かな P&L の誤りです。`ConflictError` はリトライです。
#
# このレシピで進めるのは次の3つです。
#
# 1. 同じパスに対する2つの `Database` ハンドルで、2人のライターを再現する
# 2. チャンクに切ったフィードの取り込みを、3スレッドで競わせる
# 3. 同じ競合の仕掛けに、plan/apply の変更の流れからぶつかる

# %% [markdown]
# ## ここで使う用語
#
# | 用語                 | 意味 |
# | ------------------ | --- |
# | 楽観的並行制御            | 書き手はロックを取らず、ヘッドが動いていればコミットが拒否される |
# | `expected_version` | そのコミットが拡張するつもりでいるバージョン |
# | `ConflictError`    | そうでなかったときに投げられる拒否。更新の消失ではなくリトライすべき事象 |
# | 更新の消失              | 2つの書き手が混ざり、一方が他方の行を黙って上書きしてしまうこと |
# | リトライ               | 新しいヘッドを読み直し、それに対して書き込みをやり直すこと |
# | plan / apply       | 削除や置換をプランとして積んでからコミットする流れ。衝突はここでも起きる |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import threading

import pyarrow as pa

import h5i_db
import cookbook_utils as cu

path = cu.fresh_db("prod_writers")
writer_a = h5i_db.Database(path, create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の、3銘柄・1セッションぶんのティックデータが「フィード」です。1行が1約定
# です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |
#
# 以下の節ではこれを時刻的に連続したチャンクに切り分けます。テーブルは `ts` 順なので、
# スライスがそのまま時間の窓になります。

# %%
feed = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, trades_per_day=8_000, seed=7)
print(f"feed: {feed.num_rows:,} rows x {feed.num_columns} columns")
feed.to_pandas().head()

# %%
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
writer_a.create_table("trades", schema, time_column="ts", sort_key=["ts", "symbol"])

# %% [markdown]
# ## 2. 1つのデータベースに2つのハンドル
#
# `writer_b` は*同じ*ディレクトリを開きます。2つ目のプロセスだと思ってください。
#
# 片方のハンドルで打ったコミットは、もう片方からすぐ見えます。どちらのハンドルもディスク上の
# 同じマニフェストを読みますし、ハンドルごとにテーブルの先頭をキャッシュして古くなることも
# ありません。

# %%
writer_b = h5i_db.Database(path)  # no create: the db already exists

writer_a.append("trades", feed.slice(0, 3_000), note="A: chunk 0")
head_a = writer_a.versions("trades")[-1]["sequence"]
head_b = writer_b.versions("trades")[-1]["sequence"]
print(f"head seen by A: v{head_a}, by B: v{head_b}")

# %% [markdown]
# ## 3. 古い `expected_version` は大きな音を立てて落ちる
#
# B は先頭を v1 で読み、append しようとします。その前に A が別のチャンクをコミットします。
#
# B の `append(..., expected_version=1)` は、もう存在しない先頭に対する compare-and-swap に
# なっているので、h5i-db はこれを拒否します。
#
# エラーは機械可読です。振り分け用の `.code`、リトライして安全だと教える `.retryable`、
# 運用者向けの `.hint`。

# %%
head_b_saw = writer_b.versions("trades")[-1]["sequence"]  # B reads: v1

writer_a.append("trades", feed.slice(3_000, 1_500), note="A: chunk 1")  # head -> v2

try:
    writer_b.append("trades", feed.slice(4_500, 1_500), expected_version=head_b_saw)
except h5i_db.ConflictError as e:
    print(f"code      = {e.code}")
    print(f"retryable = {e.retryable}")
    print(f"hint      = {e.hint}")

# %% [markdown]
# 何も書かれていません。B の行はテーブルに入っていませんし、バージョンの連鎖も無傷です。
#
# これを「最後に書いた人が勝つ」ストレージ、たとえば素の Parquet ディレクトリや CSV の置き場と
# 比べてみてください。そこでは B の書き込みが A のものを上書きするか混ざるかして、突き合わせの
# 時まで誰も気づきません。

# %%
print(f"rows in table: {len(writer_b.read('trades')):,}  (B's 1,500 rows were NOT committed)")

# %% [markdown]
# ## 4. リトライのパターン
#
# `retryable=True` なので、直し方は機械的です。先頭を読み直し、それに対して append をやり直し、
# 数回試して駄目なら諦める。
#
# 楽観的並行制御のどのストアに対しても書くことになる、いつもの CAS リトライのループです。

# %%
def append_with_retry(handle, table, data, note=None, max_attempts=5):
    """Append with optimistic locking; retry on conflict."""
    for attempt in range(1, max_attempts + 1):
        head = handle.versions(table)[-1]["sequence"]
        try:
            commit = handle.append(table, data, expected_version=head, note=note)
            return commit, attempt
        except h5i_db.ConflictError:
            continue  # head moved between read and commit - re-read and retry
    raise RuntimeError(f"gave up after {max_attempts} attempts")


commit, attempts = append_with_retry(writer_b, "trades", feed.slice(4_500, 1_500), note="B: chunk 2 (retried)")
print(f"B committed v{commit['sequence']} on attempt {attempts}; rows_total={commit['rows_total']:,}")

# %% [markdown]
# ## 5. 1つのフィードを3スレッドで奪い合う
#
# ここからが負荷試験です。フィードの残りを時刻的に連続した9つのチャンクに分け、3つのライター
# スレッドが、それぞれ**自分の** `Database` ハンドルを持って取り込みを競います。
#
# 協調は*すべて* CAS を通して行われます。各スレッドは次のチャンクがどれかをコミット済みの先頭
# シーケンスから導き、`expected_version=head` を付けて append します。
#
# 2つのスレッドが同じチャンクを選んだら、着地するコミットはちょうど1つです。負けたほうは
# `ConflictError` を受け取り、先頭を読み直し、次のチャンクへ進みます。ロックもキューもありま
# せん。バージョンの連鎖がキューです。

# %%
N_CHUNKS = 9
base_rows = 6_000  # rows already committed in sections 1-3
late_rows = 1_500  # held back for section 5
chunk_rows = (len(feed) - base_rows - late_rows) // N_CHUNKS
chunks = [feed.slice(base_rows + i * chunk_rows, chunk_rows) for i in range(N_CHUNKS)]
thread_rows = sum(len(c) for c in chunks)
print(f"{N_CHUNKS} chunks x {chunk_rows:,} rows for the race")

base_seq = writer_a.versions("trades")[-1]["sequence"]
rows_before = len(writer_a.read("trades"))
stats = {}


def feed_worker(name: str) -> None:
    handle = h5i_db.Database(path)  # per-thread handle, like a separate process
    wins = conflicts = 0
    try:
        while True:
            head = handle.versions("trades")[-1]["sequence"]
            next_chunk = head - base_seq  # chunk index is derived from the committed head
            if next_chunk >= N_CHUNKS:
                break
            try:
                handle.append("trades", chunks[next_chunk], expected_version=head,
                              note=f"{name}: chunk {next_chunk}")
                wins += 1
            except h5i_db.ConflictError:
                conflicts += 1  # another writer landed this chunk first - retry
    finally:
        handle.close()
    stats[name] = {"commits": wins, "conflicts": conflicts}


threads = [threading.Thread(target=feed_worker, args=(f"writer-{i}",)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

stats

# %% [markdown]
# どのスレッドがどのチャンクを取るかは実行ごとに変わります。*結果*のほうは決定的です。どの
# チャンクもちょうど1度、順番どおりにコミットされ、バージョンの連鎖は線形です。以下では行が
# 失われたり重複したりしていないことを確認します。

# %%
rows_after = len(writer_a.read("trades"))
expected = rows_before + thread_rows
assert rows_after == expected, f"lost updates! {rows_after} != {expected}"

seqs = [v["sequence"] for v in writer_a.versions("trades")]
assert seqs == list(range(len(seqs))), "version chain is not linear"

total_commits = sum(s["commits"] for s in stats.values())
total_conflicts = sum(s["conflicts"] for s in stats.values())
print(f"rows: {rows_before:,} -> {rows_after:,} (all {N_CHUNKS} chunks landed, none lost)")
print(f"commits: {total_commits}, conflicts absorbed by retries: {total_conflicts}")
print(f"version chain: v0..v{seqs[-1]}, strictly linear")

# %%
[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in writer_a.versions("trades")[-4:]
]

# %% [markdown]
# ## 6. plan/apply も同じ壁にぶつかる
#
# プレビューできる変更の流れ、`plan_delete_range` から確認して `apply` までも CAS で守られて
# います。プランは特定の基底バージョンに対して作られ、そのあと先頭が動いていれば `apply()` は
# 拒否します。
#
# ここでは A が疑わしいプリントの窓を削除しようとしますが、A が適用する前に B が新しいデータを
# コミットします。フィードと運用のジョブがテーブルを共有するとき、まさに捕まえたい競合です。
#
# 範囲の引数は生の**マイクロ秒**、`ts` 列の単位で、終端は排他的です。

# %%
ts0 = feed["ts"][0].value  # raw us since epoch
bad_lo, bad_hi = ts0, ts0 + 60_000_000  # first minute of the day

plan = writer_a.plan_delete_range("trades", bad_lo, bad_hi, note="drop suspect open prints")
print("planned:", plan.summary["rows_affected"], "rows to delete",
      f"({plan.summary['rows_before']:,} -> {plan.summary['rows_after']:,})")

# B lands one more chunk while A's plan sits unapplied:
writer_b.append("trades", feed.slice(base_rows + thread_rows, late_rows), note="B: late chunk")

try:
    plan.apply()
except h5i_db.ConflictError as e:
    print(f"\napply failed - code={e.code}, retryable={e.retryable}")
    print(f"hint = {e.hint}")

# %% [markdown]
# 古くなったプランは死にましたが、立て直しは安上がりです。2つ目のプランは*新しい*先頭に対して
# 作られるので、そのプレビューには B の遅れたチャンクも反映されます。
#
# それが CAS の狙いです。足元で変わったテーブルを闇雲に書き換えるかわりに、現在の事実で判断を
# やり直すわけです。

# %%
plan.discard()  # drop the stale plan
plan2 = writer_a.plan_delete_range("trades", bad_lo, bad_hi, note="drop suspect open prints (re-planned)")
commit = plan2.apply()
print(f"re-planned and applied as v{commit['sequence']} ({commit['op']}), "
      f"rows_total={commit['rows_total']:,}")

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in writer_a.versions("trades")[-3:]
]

# %% [markdown]
# ## まとめ
#
# - 1つのパスに対する複数の `Database` ハンドルは一級の使い方です。片方のハンドルからのコミット
#   は他方からすぐ見えますし、誰が書こうとバージョンの連鎖は線形のまま保たれます。
# - `expected_version` は `append` を compare-and-swap に変えます。古い書き込みは、`.code`、
#   `.retryable`、`.hint` を持つ `ConflictError` を上げます。静かな更新の取りこぼしではなく、
#   明示的でリトライ可能な失敗です。
# - リトライのループは5行です。先頭を読み直し、append し直し、試行回数に上限を置く。CAS だけで
#   協調した3つのスレッドが、チャンクに切ったフィードを1行も失わずに取り込みました。
# - `plan.apply()` も同じ仕掛けで守られています。プランは基底バージョンに束縛され、先頭が動けば
#   立て直しが強制されるので、適用の時点でプレビューが古いということが起こりません。
# - 共有するリサーチ／本番のテーブルでは、明示的な競合が「最後に書いた人が勝つ」に勝ります。
#   失敗の形が、突き合わせの破綻ではなくリトライになるからです。

# %%
writer_a.close()
writer_b.close()
