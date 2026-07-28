# %% [markdown]
# # 取り込みのパターン: 5つの入口から1つのテーブルへ
#
# データが1か所から届くデスクはありません。ティックフィードは Arrow のバッチを寄こし、
# リサーチのノートブックは pandas や polars に住み、ベンダーは Parquet を置いていき、
# どこかの古いプロセスはいまだに CSV をメールで送ってきます。
#
# h5i-db の取り込み面は意図的に小さくしてあります。`append` がフィードを伸ばし、`write` が
# 中身を置き換える。どちらも Arrow の形をしたものなら何でも受け取ります。
#
# このレシピで進めるのは次の4つです。
#
# 1. 連続する5セッションを5つの別々のソースから1つの `trades` テーブルに取り込む
# 2. `write` と `append` の意味論を対比する
# 3. 楽観ロックで、同時に走るローダを安全にする
# 4. コミットをまとめ、残ったセグメントを圧縮する

# %%
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import h5i_db
from h5i_db import col, count_star, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_ingestion"), create=True)

# %% [markdown]
# ## データ
#
# `cu.make_trades` が返す、連続11セッションぶんのティックデータです。1行が1約定です。
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
tape = cu.make_trades(days=11, trades_per_day=4_000, start="2026-06-01", seed=7)
print(f"{tape.num_rows:,} rows x {tape.num_columns} columns")
tape.to_pandas().head()

# %% [markdown]
# これを日ごとのバッチに切り分け、「1回の配信」が時刻順に届くようにします。`append` は
# どのバッチも、テーブルに保存済みの最大タイムスタンプ以降から始まることを要求します。
# フィードの意味論とは、実務上はそういうことです。ベンダーの受け渡し場所の代わりに、
# `data/dbs` の下にステージング用のディレクトリを1つ用意します。

# %%
dates = tape["ts"].to_pandas().dt.date
sessions = sorted(dates.unique())
by_day = {d: tape.filter(pa.array((dates == d).to_numpy())) for d in sessions}
print(f"{len(sessions)} sessions, {len(tape):,} trades:", sessions[0], "→", sessions[-1])

staging = Path("data/dbs/00_ingestion_staging")
if staging.exists():
    shutil.rmtree(staging)
staging.mkdir(parents=True)

# %% [markdown]
# 行き先は1つのテーブルです。以下のソースはすべてここに着地します。

# %%
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
# ## 1. pyarrow の Table から
#
# 変換の要らないネイティブの経路です。`append` はどれも**コミット辞書**を返します。新しい
# バージョン番号（`sequence`）と、コミット後の行数・セグメント数が入っています。ローダでは
# これをログに残してください。配信とバージョンを結びつける受領証です。

# %%
commit = db.append("trades", by_day[sessions[0]], note="day 1: arrow feed")
commit

# %% [markdown]
# ## 2. pandas の DataFrame から
#
# 重い仕事は `pa.Table.from_pandas` がやってくれます。それに加えて `schema=` を必ず渡して
# ください。pandas のバージョンによっては、日時がテーブルのマイクロ秒ではなくナノ秒として
# 往復してしまい、厳格な append がその不一致を拒否します。目的のスキーマを渡しておけば、
# 変換のついでにキャストされます。

# %%
df = by_day[sessions[1]].to_pandas()  # pretend this came from research code
commit = db.append(
    "trades",
    pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False),
    note="day 2: pandas",
)
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 3. polars の DataFrame から
#
# polars は Arrow をそのまま話しますが、1つだけ癖があります。`to_arrow()` が吐くのは
# `large_string` の列で、厳格な append はこれをスキーマ不一致として弾きます。`.cast(SCHEMA)`
# ならメタデータ層で片付くので、コストはかかりません。polars から h5i-db へ渡す境界では、
# これを習慣にしてください。

# %%
pldf = pl.from_arrow(by_day[sessions[2]])  # pretend this came from a polars pipeline
commit = db.append("trades", pldf.to_arrow().cast(SCHEMA), note="day 3: polars")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 4. Parquet ファイルから
#
# Parquet は型をそのまま保つので、ベンダーが置いていった Parquet は読んで append するだけ
# です。行の並び順が怪しいベンダーなら、append の前にソートしておきましょう。どちらにせよ
# 厳格な append が教えてくれます。

# %%
pq.write_table(by_day[sessions[3]], staging / "vendor_day4.parquet")

commit = db.append("trades", pq.read_table(staging / "vendor_day4.parquet"), note="day 4: parquet drop")
{k: commit[k] for k in ("sequence", "rows_total", "segments_total")}

# %% [markdown]
# ## 5. CSV から
#
# CSV は値を保ちますが型を落とします。素直に読むと `ts` はパーサが推測した何かとして返って
# きます。`ConvertOptions(column_types=...)` でパース時に時刻列を `timestamp[us, tz=UTC]` に
# 固定すれば、スキーマは厳密に一致します。

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
# ## `write` と `append`
#
# - **`append`** はフィードを伸ばします。厳密に時刻順で、既存の行には触れません。テープの
#   ように振る舞うものはすべてこちらです。
# - **`write`** はテーブルの中身を、渡したデータで置き換えます。ただし*新しいバージョン*と
#   してで、履歴はすべて残ります。まるごと言い直される参照データ、たとえばユニバースの構成、
#   銘柄マッピング、リスクリミットはこちらです。
#
# どちらも破壊的ではありません。古いバージョンは Python の `db.read(..., version=n)` からも、
# SQL の `h5i('table', n)` からも読めるままです。

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
# 1つのテーブルを2つのローダが共有していると、「いつでも好きなものを append する」は配信を
# 静かに混ぜ込みます。`append(..., expected_version=n)` はコミットを compare-and-swap に
# 変えます。テーブルの先頭がまだバージョン `n` のときだけ着地するのです。そうでなければ
# `ConflictError` が返り、`retryable=True` と復旧手順を書いたヒントが付いてきます。リトライ
# 自体は機械的です。先頭を読み直して、もう一度 append するだけです。

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
# ## バッチ化と圧縮
#
# コミット1件ごとにマニフェストと最低1つのセグメントが書かれます。だから*バッチ*でコミット
# してください。1日ぶん、1時間ぶん、数千行ぶん。1行ずつは絶対にやめましょう。
#
# 1日単位のコミットでも小さなセグメントは溜まりますし、クエリの計画はその全部に触ります。
# 下のループが普通のリズムです。平日のあいだは小さな append コミットを重ね、最後に `compact`
# を1回かけてセグメントをまとめます。圧縮もそれ自体がただのコミットで、同じデータを少ない
# セグメントで持ち、履歴はすべて残ります。

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
# One tape, five formats, eleven commits - and a query sees a single clean table.
(
    db.table("trades")
    .group_by(time_bucket("1d", col("ts")).alias("session"))
    .agg(
        trades=count_star(),
        notional_mm=((col("price") * col("size")).sum() / 1e6).round(1),
    )
    .sort("session")
    .to_pandas()
)

# %% [markdown]
# ## まとめ
#
# - Arrow の形をしたものはそのまま append できます。境界で引っかかるのは pandas のナノ秒
#   （`from_pandas(schema=...)`）、polars の `large_string`（`.cast(schema)`）、CSV の型忘れ
#   （`ConvertOptions`）の3つです。
# - `append` は厳密な時刻順でフィードを伸ばし、`write` は中身をまるごと新しいバージョンとして
#   言い直します。どちらも履歴を壊しません。
# - コミット辞書（`sequence`、`rows_total`、`segments_total`）が取り込みの受領証です。ログに
#   残してください。
# - `expected_version` は append を compare-and-swap に変えます。リトライ可能な
#   `ConflictError` が出たら、先頭を読み直して append し直します。
# - コミットはまとめ、小さな append が続いたあとは `compact` をかけます。圧縮は履歴に触れず
#   セグメントを併合するだけの、ごく普通のバージョンです。

# %%
db.close()
