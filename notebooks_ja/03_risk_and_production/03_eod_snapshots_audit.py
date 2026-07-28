# %% [markdown]
# # EOD スナップショットと、規制当局が実際に求める監査証跡
#
# 18か月後に飛んでくる質問は「いまの価格はいくらか」ではありません。*「2026-06-02 の営業終了
# 時点で、あなたのシステムは何を知っていたか」*です。答えにバックアップテープの復元が含まれる
# なら、その会議はもう負けています。
#
# h5i-db では日次の断面が名前付きスナップショットです。テーブルのマニフェストをチェックサム
# つきで固定したもので、作成は O(1)、SQL からは `h5i('trades', 'eod-2026-06-02')` として永久に
# 指せます。
#
# このレシピで進めるのは次の4つです。
#
# 1. その日のフィードを追記し、データ品質を検査し、スナップショットを取り、記録する小さな
#    EOD パイプラインを組む
# 2. 規制当局の質問に3通りで答える
# 3. 2つの EOD 断面を SQL で差分する
# 4. `verify(deep=True)` による完全性の証明と、`vacuum` による保持で締める

# %%
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, count_star, sql_expr

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_eod"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の3セッションぶんのティックデータで、1行が1約定です。
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
# 本番のローダがそうするように、1セッションずつ流し込みます。

# %%
all_trades = cu.make_trades(days=3, trades_per_day=20_000).to_pandas()
print(f"{len(all_trades):,} rows x {all_trades.shape[1]} columns, "
      f"{all_trades['ts'].dt.date.nunique()} sessions")
all_trades.head()

# %% [markdown]
# ## 2. テーブル: フィードと、自前のスナップショット記録
#
# 先に API について正直に書いておきます。Python API にあるのはスナップショットの*作成*だけで、
# `db.snapshot(name, tables=[...], note=...)` です。一覧表示は CLI の機能で、
# `h5i-db snapshot list` になります。
#
# 本番のパイプラインはカタログをデータの隣で引きたいので、自前の `snapshot_log` テーブルを
# 持ちます。EOD の実行ごとに、`snapshot()` が返す名前・固定されたシーケンス・マニフェストの
# チェックサムを1行追記します。
#
# この記録自体もバージョン管理された追記専用で、監査人がログに求めるのはまさにその性質です。

# %%
trade_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])

log_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("snapshot_name", pa.string()),
        pa.field("table_name", pa.string()),
        pa.field("pinned_sequence", pa.int64()),
        pa.field("manifest_checksum", pa.string()),
        pa.field("note", pa.string()),
    ]
)
db.create_table("snapshot_log", log_schema, time_column="ts")
db.tables()

# %% [markdown]
# ## 3. EOD のパイプライン: 追記、検査、スナップショット、記録
#
# 本番の3日ぶんを再現します。各日はまず、その日のフィードをノート付きでアトミックに `append`
# し、続いて行数・価格の正値性・ユニバースの充足を見る最小限のデータ品質ゲートを通します。
# ゲートを通ったときだけ、名前付きスナップショットと記録の1行が続きます。
#
# スナップショットの辞書は、テーブルごとに固定されたバージョンとマニフェストのチェックサムを
# 返します。その名前が正確に何を指すのかの暗号学的な証拠です。

# %%
all_trades["day"] = all_trades["ts"].dt.date

def eod_checks(day: str) -> dict:
    row = (
        db.table("trades")
        .filter(col("ts") >= f"{day}T00:00:00Z", col("ts") < f"{day}T23:59:59Z")
        .select(
            rows=count_star(),
            symbols=col("symbol").n_unique(),
            px_min=col("price").min(),
            px_max=col("price").max(),
        )
        .to_pandas()
        .iloc[0]
    )
    return {
        "rows": int(row["rows"]),
        "symbols_ok": int(row["symbols"]) == 3,
        "prices_ok": bool(row["px_min"] > 0),
        "nonempty": int(row["rows"]) > 0,
    }

last_snap = None
for day, chunk in all_trades.groupby("day", sort=True):
    data = pa.Table.from_pandas(
        chunk.drop(columns="day").sort_values(["ts", "symbol"]), schema=trade_schema,
        preserve_index=False,
    )
    commit = db.append("trades", data, note=f"{day} feed")
    checks = eod_checks(str(day))
    assert all(v for k, v in checks.items() if k != "rows"), f"EOD gate failed: {checks}"

    last_snap = db.snapshot(f"eod-{day}", tables=["trades"], note=f"EOD risk cut {day}")
    entry = next(iter(last_snap["entries"].values()))
    db.append("snapshot_log", pa.table({
        "ts": pa.array([pd.Timestamp(last_snap["created_at_ns"], unit="ns", tz="UTC")],
                       type=pa.timestamp("us", tz="UTC")),
        "snapshot_name": [last_snap["name"]],
        "table_name": [entry["table_name"]],
        "pinned_sequence": [entry["sequence"]],
        "manifest_checksum": [entry["manifest_checksum"]],
        "note": [last_snap["note"]],
    }))
    print(f"{day}: appended {data.num_rows:,} rows as v{commit['sequence']}, "
          f"checks {checks}, snapshot 'eod-{day}'")

# %% [markdown]
# スナップショットの辞書そのものです。名前、作成時刻、そしてテーブルごとの固定とチェックサム。
# これが記録に落ちます。

# %%
last_snap

# %%
(
    db.table("snapshot_log")
    .select("ts", "snapshot_name", "pinned_sequence",
            checksum=sql_expr("substr(manifest_checksum, 1, 12)"))
    .sort("ts")
    .to_pandas()
)

# %% [markdown]
# ## 4. 「2026-06-02 に何を知っていたか」への3通りの答え
#
# 同じ読み取り点は3通りで指せます。
#
# - **スナップショット名**で。EOD の断面という業務上の意味を持ちます。
# - **バージョン番号**で。記録の `pinned_sequence` から取ります。
# - **as-of のコミット時刻**で。T より前にコミットされていたもの、という意味です。
#
# 3つとも再生ではなく O(1) のマニフェスト参照です。

# %%
by_name = (
    db.table("trades", snapshot="eod-2026-06-02")
    .select(
        rows=count_star(),
        last_trade=col("ts").max(),
        vwap_all=((col("price") * col("size")).sum() / col("size").sum()).round(2),
    )
    .to_pandas()
)
by_name

# %%
# By version number: the log row tells us which sequence the name pins.
pinned = (
    db.table("snapshot_log")
    .filter(col("snapshot_name") == "eod-2026-06-02")
    .select("pinned_sequence")
    .to_pandas()["pinned_sequence"]
    .iloc[0]
)
by_version = db.read("trades", version=int(pinned))

# By commit wall-clock time: the append's committed_at_ns from versions().
v2 = [v for v in db.versions("trades") if v["sequence"] == pinned][0]
as_of = pd.Timestamp(v2["committed_at_ns"], unit="ns", tz="UTC").isoformat()
by_time = db.read("trades", as_of=as_of)

print(f"by snapshot name: {int(by_name['rows'].iloc[0]):,} rows")
print(f"by version {pinned}:     {by_version.num_rows:,} rows")
print(f"by as_of {as_of}: {by_time.num_rows:,} rows")
assert by_version.num_rows == by_time.num_rows == int(by_name["rows"].iloc[0])

# %% [markdown]
# ## 5. 2つの EOD 断面を SQL 1文で差分する
#
# スナップショットは関係なので、「2日と3日で何が変わったか」は ETL ジョブではなくジョインに
# なります。銘柄ごとに、増えた行数と最後のプリントがどこへ動いたかが出ます。

# %%
def per_symbol(snapshot: str):
    return (
        db.table("trades", snapshot=snapshot)
        .group_by("symbol")
        .agg(n=count_star(), last_px=col("price").last("ts"))
    )


d3, d2 = per_symbol("eod-2026-06-03"), per_symbol("eod-2026-06-02")
n3, n2 = col("n", relation="l"), col("n", relation="r")
px3, px2 = col("last_px", relation="l"), col("last_px", relation="r")

(
    d3.join(d2, on="symbol")
    .select(
        symbol=col("symbol", relation="l"),
        trades_added=n3 - n2,
        close_jun02=px2,
        close_jun03=px3,
        px_chg=(px3 / px2 - 1).round(4),
    )
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# ## 6. 完全性の証明
#
# `verify(deep=True)` はすべてのマニフェストとセグメントを読み直し、先頭から起点までのチェック
# サム連鎖を確認します。
#
# `snapshot_log` に記録したマニフェストのチェックサムと合わせれば、そのまま手渡せる証明に
# なります。記録に名前のある EOD 断面はチェックサムの一致するマニフェストに解決され、その下の
# すべてのバイトが検証されます。

# %%
report = db.verify("trades", deep=True)
report

# %%
assert not report["problems"], f"integrity check failed: {report['problems']}"
print(f"checked {report['manifests_checked']} manifests, "
      f"{report['segments_checked']} segments, "
      f"{report['bytes_checked']:,} bytes - no problems")

# %% [markdown]
# ## 7. 保持: vacuum は履歴とスナップショットを尊重する
#
# `vacuum` は何からも参照されていないストレージを回収します。ここではすべてのセグメントが、
# バージョンの連鎖からも3つの EOD スナップショットからも、なお到達可能です。
#
# だから `grace_seconds=0` という強い設定で走らせても何も消えませんし、いちばん古い EOD 断面も
# そのあと読めるままです。スナップショットが保持ポリシーそのものです。残すべきものを固定し、
# あとは vacuum する。

# %%
print("dry run:", db.vacuum(apply=False))
print("applied:", db.vacuum(grace_seconds=0, apply=True))
print("oldest EOD cut still readable:",
      f"{len(db.read('trades', snapshot='eod-2026-06-01')):,} rows")

# %% [markdown]
# ## まとめ
#
# - EOD の断面は `db.snapshot(name, tables=[...], note=...)` です。O(1) で、データはコピーされ
#   ず、SQL からは `h5i('table', 'name')` として永久に指せます。
# - 「X日時点で知られていたこと」には3つの等価な綴りがあります。スナップショット名、バージョン
#   番号、`as_of` のコミット時刻。どれも再生なしで解決します。
# - Python API はまだスナップショットを一覧できません。CLI はできます。追記専用の
#   `snapshot_log` テーブルがその隙間を埋め、断面ごとの固定シーケンスとマニフェストのチェック
#   サムを持つ*引ける、バージョン管理された*カタログを与えます。
# - `verify(deep=True)` と記録したチェックサムが「信じてください」を証明に変えますし、
#   スナップショットが固定しているものに `vacuum` は手を出せません。

# %%
db.close()
