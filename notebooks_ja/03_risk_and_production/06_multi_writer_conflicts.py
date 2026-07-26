# %% [markdown]
# # 複数書き手の協調: 楽観ロック、衝突、リトライ
#
# h5i-db のデータベースはディレクトリなので、2つのプロセスが同時に開くのを止めるものは何も
# ありません。フィードハンドラと訂正ジョブでも、同僚2人のノートブックでも同じことです。並行する
# 書き手に対する h5i-db の答えが*楽観的並行制御*です。どのコミットも `expected_version` を持てて、
# 読んだあとにテーブルの先頭が動いていれば、黙って割り込む代わりに明示的な `ConflictError` で
# 拒否されます。チームにとって、これが正しい取引です。ポジションや評価のテーブルで更新が失われれば
# 静かな損益の誤りになりますが、`ConflictError` ならリトライで済みます。
#
# このレシピでは、同じパスに対する2つの `Database` ハンドルで書き手2つを模擬し、次に3スレッドが
# 刻んだフィードの取り込みを競い、最後に plan/apply の流れで同じ衝突の仕掛けに突き当たります。

# %%
import threading

import pyarrow as pa

import h5i_db
import cookbook_utils as cu

path = cu.fresh_db("prod_writers")
writer_a = h5i_db.Database(path, create=True)

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

# One day of ticks - our "feed". Sections below carve it into
# time-contiguous chunks (the table is sorted by ts, so slices are windows).
feed = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=1, trades_per_day=8_000, seed=7)
print(f"feed: {len(feed):,} rows")

# %% [markdown]
# ## 1. 1つのデータベースに2つのハンドル
#
# `writer_b` は*同じ*ディレクトリを開きます。2つ目のプロセスだと思ってください。一方のハンドルで
# 打ったコミットは、もう一方からすぐ見えます。どちらのハンドルもディスク上の同じマニフェストを
# 読むので、テーブルの先頭がハンドルごとにキャッシュされて古くなることはありません。

# %%
writer_b = h5i_db.Database(path)  # no create: the db already exists

writer_a.append("trades", feed.slice(0, 3_000), note="A: chunk 0")
head_a = writer_a.versions("trades")[-1]["sequence"]
head_b = writer_b.versions("trades")[-1]["sequence"]
print(f"head seen by A: v{head_a}, by B: v{head_b}")

# %% [markdown]
# ## 2. 古い `expected_version` は大きな声で失敗する
#
# B が先頭（v1）を読み、append しようとします。ところがその前に A が別のチャンクをコミットします。
# B の `append(..., expected_version=1)` は、もはや存在しない先頭に対する compare-and-swap に
# なっていて、h5i-db はこれを拒みます。エラーが機械可読である点に注目してください。振り分けの
# ための `.code`、リトライして安全だと伝える `.retryable`、運用者向けの `.hint` です。

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
# 何も書かれていません。B の行はテーブルに入っておらず、バージョン連鎖も無傷です。「最後に書いた
# 人が勝つ」タイプのストレージ（素の Parquet ディレクトリ、CSV の置き場）と比べてみてください。
# あちらでは B の書き込みが A のものを潰すか、混ざり込むかしたうえで、突き合わせをするまで誰も
# 気づきません。

# %%
print(f"rows in table: {len(writer_b.read('trades')):,}  (B's 1,500 rows were NOT committed)")

# %% [markdown]
# ## 3. リトライのパターン
#
# `retryable=True` なので、対処は機械的です。先頭を読み直し、それに対して append をやり直し、
# 数回で諦める。楽観的並行制御を持つどのストアに対しても書くのと同じ CAS リトライのループです。

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
# ## 4. 1つのフィードを3スレッドが奪い合う
#
# ここからが負荷試験です。フィードの残りを時間的に連続した9つのチャンクに分け、それぞれが
# **自分の** `Database` ハンドルを持つ3つの書き手スレッドが取り込みを競います。協調は*完全に*
# CAS だけで行います。各スレッドはコミット済みの先頭シーケンスから「次はどのチャンクか」を導き、
# `expected_version=head` を付けて append します。2つのスレッドが同じチャンクを選んだときは、
# ちょうど1つのコミットだけが着地します。負けたほうは `ConflictError` を受け取り、先頭を読み直し、
# 次のチャンクへ進みます。ロックもキューもありません。バージョン連鎖がキューです。

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
# どのスレッドがどのチャンクを取るかは実行ごとに変わります。ところが*結果*は決定的です。どの
# チャンクもちょうど1回、順序どおりにコミットされ、バージョン連鎖は一直線です。行の欠落も重複も
# ないことを確認します。

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
# ## 5. plan/apply も同じ壁に当たる
#
# プレビューできる変更の流れ（`plan_delete_range` → 検分 → `apply`）も CAS で守られています。
# プランは特定の土台バージョンに対して作られ、`apply()` は先頭が動いていれば拒否します。ここでは
# A が疑わしい約定の窓を削除しようとしますが、A が適用する前に B が新しいデータをコミットします。
# フィードと運用ジョブがテーブルを共有するとき、まさに捕まえてほしい競合です。
#
# 範囲の引数は生の**マイクロ秒**（`ts` 列の単位）で、終端の境界は含みません。

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
# 古いプランは死にました。ただしプランを立て直すのは安く済みますし、2つ目のプランは*新しい*先頭に
# 対して作られるので、そのプレビューには B の遅れて届いたチャンクも反映されます。CAS の狙いは
# そこにあります。足元で変わったテーブルを闇雲に書き換えず、いまの事実に基づいて判断し直すのです。

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
# - 1つのパスに対する複数の `Database` ハンドルは一級の使い方です。一方のハンドルのコミットは
#   すぐ他方から見えますし、誰が書こうとバージョン連鎖は一直線のままです。
# - `expected_version` は `append` を compare-and-swap に変えます。古い書き込みは `.code`、
#   `.retryable`、`.hint` を持つ `ConflictError` を上げます。更新が静かに失われる代わりに、
#   明示的でリトライ可能な失敗が返るわけです。
# - リトライのループは5行です。先頭を読み直し、append をやり直し、試行回数に上限を置く。CAS だけで
#   協調した3つの競合スレッドが、刻んだフィードを1行も落とさずに取り込みました。
# - `plan.apply()` も同じ仕掛けで守られています。プランは土台バージョンに束縛され、先頭が動けば
#   立て直しを迫られます。適用の時点でプレビューが古い、ということが起こりえません。
# - 共有のリサーチ／本番テーブルでは、明示的な衝突のほうが「最後に書いた人が勝つ」より優れて
#   います。失敗の形が、突き合わせの破綻ではなくリトライになるからです。

# %%
writer_a.close()
writer_b.close()
