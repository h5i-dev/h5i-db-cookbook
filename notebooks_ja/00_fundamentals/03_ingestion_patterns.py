# %% [markdown]
# # 取り込みのパターン: 5つの入力元を1つのテーブルへ
#
# 現場のデータが1か所から来ることはまずありません。ティックフィードは Arrow バッチを
# 渡してきますし、リサーチ用のノートブックは pandas か polars の中にあり、ベンダーは
# Parquet を置いていき、古い業務プロセスはいまだに CSV をメールで送ってきます。h5i-db の
# 取り込み口は意図的に小さく、`append`（フィードを伸ばす）と `write`（中身を差し替える）
# の2つだけで、どちらも Arrow の形をしたものなら受け取ります。このレシピでは、連続する
# 5営業日ぶんを5種類の入力元から1つの `trades` テーブルに取り込み、そのうえで取り込みの
# 運用面を扱います。`write` と `append` の違い、`expected_version` による楽観ロック、
# そしてコミットをまとめてから `compact` する理由です。

# %%
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_ingestion"), create=True)

SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", SCHEMA, time_column="ts", sort_key=["ts", "symbol"])

# %% [markdown]
# 11セッションぶんの連続したテープを、1日ずつのバッチに切り分けます。こうすると各
# 「納品」が時刻順に届きます。`append` はどのバッチもテーブルの保存済み最大タイム
# スタンプ以降から始まることを求めるためです（フィードとしての意味論）。ベンダーの
# 受け渡し場所の代わりに、`data/dbs` の下のステージング用ディレクトリを使います。

# %%
tape = cu.make_trades(days=11, trades_per_day=4_000, start="2026-06-01", seed=7)

dates = tape["ts"].to_pandas().dt.date
sessions = sorted(dates.unique())
by_day = {d: tape.filter(pa.array((dates == d).to_numpy())) for d in sessions}
print(f"{len(sessions)} sessions, {len(tape):,} trades:", sessions[0], "→", sessions[-1])

staging = Path("data/dbs/00_ingestion_staging")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

# %% [markdown]
# ## 1. pyarrow Table から
#
# 変換を挟まないネイティブの経路です。`append` は必ず**コミット辞書**を返します。中身は
# 新しいバージョン番号（`sequence`）と、コミット後の総行数・総セグメント数です。ローダーの
# ログに残しておきましょう。納品とバージョンを結びつける受領証になります。

# %%
commit = db.append("trades", by_day[sessions[0]], note="day 1: arrow feed")
commit

# %% [markdown]
# ## 2. pandas DataFrame から
#
# 重い仕事は `pa.Table.from_pandas` がやってくれますが、`schema=` は必ず渡してください。
# pandas のバージョンによっては datetime がナノ秒（テーブル側はマイクロ秒）のまま往復し、
# 厳格な append がその不一致を拒みます。目的のスキーマを渡せば、変換のついでにキャストが
# かかります。

# %%
df = by_day[sessions[1]].to_pandas()  # pretend this came from research code
commit = db.append(
    "trades",
    pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
    note="day 2: pandas",
)
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 3. polars DataFrame から
#
# Polars は Arrow をそのまま話しますが、1つだけ癖があります。`to_arrow()` が出すのは
# `large_string` の列で、厳格な append はこれをスキーマ不一致として拒みます。
# `.cast(SCHEMA)` はメタデータ層の安い修正なので、polars から h5i-db へ渡す境界では
# 習慣にしてしまうのがよいでしょう。

# %%
pldf = pl.from_arrow(by_day[sessions[2]])  # pretend this came from a polars pipeline
commit = db.append("trades", pldf.to_arrow().cast(SCHEMA), note="day 3: polars")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 4. Parquet ファイルから
#
# Parquet は型をそのまま保つので、ベンダーが置いていった Parquet は読んで append する
# だけです。（行の順序が怪しいベンダーなら、append の前にソートしてください。どちらに
# しても厳格な append が教えてくれます。）

# %%
pq.write_table(by_day[sessions[3]], staging / "vendor_day4.parquet")

commit = db.append("trades", pq.read_table(staging / "vendor_day4.parquet"), note="day 4: parquet drop")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 5. CSV から
#
# CSV は値を保ちますが型を落とします。素直に読むと、`ts` はパーサーが推測した何かに
# なって返ってきます。`ConvertOptions(column_types=...)` で時刻列をパース時点で
# `timestamp[us, tz=UTC]` に固定すれば、スキーマがぴたりと一致します。

# %%
pacsv.write_csv(by_day[sessions[4]], staging / "legacy_day5.csv")

from_csv = pacsv.read_csv(
    staging / "legacy_day5.csv",
    convert_options=pacsv.ConvertOptions(column_types={"ts": pa.timestamp("us", tz="UTC")}),
)
assert from_csv.schema.equals(by_day[sessions[4]].schema)
commit = db.append("trades", from_csv, note="day 5: legacy csv")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## `write` と `append` の違い
#
# - **`append`** はフィードを伸ばします。厳密に時刻順で、既存の行には触れません。テープの
#   ように振る舞うものはこちらです。
# - **`write`** はテーブルの中身を渡したデータで置き換えます。ただし*新しいバージョン*
#   としてで、履歴はすべて残ります。ユニバースの構成銘柄、シンボルマッピング、リスク
#   リミットのように、丸ごと言い直される参照データに使います。
#
# どちらも破壊的ではありません。古いバージョンは `db.read(..., version=n)` と、SQL の
# `h5i('table', n)` からいつでも読めます。

# %%
universe_schema = pa.schema(
    [pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False), pa.field("symbol", pa.string())]
)
db.create_table("universe", universe_schema, time_column="ts")

asof = pa.scalar(pd.Timestamp("2026-06-01", tz="UTC"), type=pa.timestamp("us", tz="UTC"))
db.write(
    "universe",
    pa.table({"ts": pa.array([asof] * 3), "symbol": pa.array(["AAPL", "MSFT", "NVDA"])}),
    note="June universe",
)
db.write(
    "universe",
    pa.table({"ts": pa.array([asof] * 4), "symbol": pa.array(["AAPL", "MSFT", "NVDA", "AVGO"])}),
    note="June universe, AVGO added",
)
print("head :", db.read("universe")["symbol"].to_pylist())
print("v1   :", db.read("universe", version=1)["symbol"].to_pylist())
[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("universe")]

# %% [markdown]
# ## `expected_version` による楽観ロック
#
# 2つのローダーが1つのテーブルを共有していると、「いつでも好きに append する」やり方は
# 納品を静かに混ぜてしまいます。`append(..., expected_version=n)` は compare-and-swap
# です。テーブルの先頭がまだバージョン `n` のときだけコミットが着地し、そうでなければ
# `ConflictError` が上がります。`retryable=True` と、復旧手順を書いたヒントが付いている
# はずです。リトライは機械的で、先頭を読み直してもう一度 append するだけです。

# %%
day6 = by_day[sessions[5]]
try:
    db.append("trades", day6, expected_version=1)  # stale: head is already at v5
except h5i_db.ConflictError as e:
    print(f"code      {e.code}")
    print(f"retryable {e.retryable}")
    print(f"hint      {e.hint}")

# retry pattern: re-read the head version, then re-append against it
head = db.versions("trades")[-1]["sequence"]
commit = db.append("trades", day6, expected_version=head, note="day 6: CAS append")
print(f"\nretried against v{head} -> committed v{commit['sequence']}")

# %% [markdown]
# ## バッチ化とコンパクション
#
# コミットのたびにマニフェストが1つと、少なくとも1つのセグメントが書かれます。だから
# コミットは*まとまり*で打ってください（1日ぶん、1時間ぶん、数千行ぶん）。1行ずつは
# 禁物です。ただ、1日サイズのコミットでも小さなセグメントは溜まっていき、クエリの
# プランニングはそのすべてに触れます。下の「日次ループ → compact」のパターンが通常の
# リズムです。平日は小さな append コミットを重ね、`compact` でセグメントを1つにまとめます。
# コンパクション自体もただのコミットで、データは同じ、セグメントは減り、履歴は丸ごと
# 残ります。

# %%
for d in sessions[6:]:
    commit = db.append("trades", by_day[d], note=f"daily load {d}")
    print(f"v{commit['sequence']}: +{len(by_day[d]):>6,} rows "
          f"-> {commit['segments_total']:>2} segments total")

# %%
before = db.versions("trades")[-1]
commit = db.compact("trades")
print(f"compacted: {before['segments']} segments -> {commit['segments_total']}, "
      f"rows unchanged: {commit['rows_total']:,}")

[
    {k: v[k] for k in ("sequence", "op", "rows", "segments", "note") if k in v}
    for v in db.versions("trades")
]

# %%
# One tape, five formats, eleven commits - and SQL sees a single clean table.
db.sql(
    """
    SELECT time_bucket('1d', ts) AS session, count(*) AS trades,
           round(sum(price * size) / 1e6, 1) AS notional_mm
    FROM trades GROUP BY session ORDER BY session
    """
).to_pandas()

# %% [markdown]
# ## まとめ
#
# - Arrow の形をしたものはそのまま append できます。境界でつまずくのは、pandas の
#   ナノ秒（`from_pandas(schema=...)`）、polars の `large_string`（`.cast(schema)`）、
#   そして型を忘れる CSV（`ConvertOptions`）の3つです。
# - `append` はフィードを伸ばす操作（厳密に時刻順）、`write` は中身を新しいバージョンとして
#   言い直す操作。どちらも履歴を壊しません。
# - コミット辞書（`sequence`、`rows_total`、`segments_total`）は取り込みの受領証です。
#   ログに残しましょう。
# - `expected_version` は append を compare-and-swap に変えます。`ConflictError`
#   （リトライ可）が出たら、先頭を読み直してもう一度 append します。
# - コミットはまとめて打ち、小さな append が続いたあとに `compact` します。コンパクションも
#   ただのバージョンで、履歴に触れずセグメントだけを併合します。

# %%
db.close()
