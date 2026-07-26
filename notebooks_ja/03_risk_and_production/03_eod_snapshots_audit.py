# %% [markdown]
# # EOD スナップショットと、規制当局が実際に求める監査証跡
#
# 18か月後に飛んでくる質問は、「いまの価格はいくらか」ではありません。*「2026年6月2日の営業
# 終了時点で、御社のシステムは何を把握していましたか」*です。その答えにバックアップテープの
# 復旧が出てくるなら、その打ち合わせはもう負けています。h5i-db では、日次の締めは名前付きの
# スナップショットです。テーブルのマニフェストをチェックサム付きで固定するもので、作成は
# O(1)、以後は SQL から `h5i('trades', 'eod-2026-06-02')` として永久に参照できます。
#
# このレシピでは小さな EOD パイプラインを組み――その日のフィードを append し、データ品質を
# 検査し、スナップショットを取り、ログに残す――規制当局の問いに3通りで答え、2つの EOD 締めを
# SQL で差分にし、最後に完全性の証明（`verify(deep=True)`）と保持（`vacuum`）で締めます。

# %%
import pandas as pd
import pyarrow as pa
import h5i_db

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_eod"), create=True)

# %% [markdown]
# ## 1. テーブル: フィードと、自前のスナップショットログ
#
# API について正直な注意を先に。Python API には現在、スナップショットの*作成*しかありません
# （`db.snapshot(name, tables=[...], note=...)`）。一覧表示は CLI の機能です
# （`h5i-db snapshot list`）。本番のパイプラインはカタログをデータの隣でクエリしたいので、
# `snapshot_log` テーブルを自前で持ちます。EOD 実行のたびに1行、名前と、`snapshot()` が返す
# 固定シーケンスとマニフェストのチェックサムを追記します。このログ自体もバージョン管理された
# 追記のみのテーブルで、監査担当がログに求めるのはまさにこの性質です。

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
# ## 2. EOD パイプライン: append、検査、スナップショット、ログ
#
# 本番を模した3日ぶんです。各日、その日のフィードをアトミックに `append` し（注記付き）、
# 最小限のデータ品質ゲート（行数、価格が正であること、ユニバースの網羅）を通し、通過した場合
# *だけ*名前付きスナップショットとログ行を作ります。スナップショットの辞書はテーブルごとに、
# 固定バージョンとマニフェストのチェックサムを返します。その名前が何を指すかについての、暗号的な
# 証拠です。

# %%
all_trades = cu.make_trades(days=3, trades_per_day=20_000).to_pandas()
all_trades["day"] = all_trades["ts"].dt.date

def eod_checks(day: str) -> dict:
    row = db.sql(
        f"""
        SELECT count(*) AS rows, count(DISTINCT symbol) AS symbols,
               min(price) AS px_min, max(price) AS px_max
        FROM trades
        WHERE ts >= '{day}T00:00:00Z' AND ts < '{day}T23:59:59Z'
        """
    ).to_pandas().iloc[0]
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
# スナップショットの辞書そのものです。名前、作成時刻、テーブルごとの固定とチェックサム。これが
# ログに入ります。

# %%
last_snap

# %%
db.sql(
    """
    SELECT ts, snapshot_name, pinned_sequence, substr(manifest_checksum, 1, 12) AS checksum
    FROM snapshot_log ORDER BY ts
    """
).to_pandas()

# %% [markdown]
# ## 3. 「6月2日に何を把握していたか」に答える3つの道
#
# 同じ読み取り点を、**スナップショット名**で（業務上の意味は EOD の締め）、**バージョン番号**で
# （ログの `pinned_sequence` から）、そして**as-of のコミット時刻**で（実時刻。「時刻Tより前に
# コミットされていたもの」）参照できます。3つとも O(1) のマニフェスト参照で、再生ではありません。

# %%
by_name = db.sql(
    """
    SELECT count(*) AS rows, max(ts) AS last_trade,
           round(sum(price * size) / sum(size), 2) AS vwap_all
    FROM h5i('trades', 'eod-2026-06-02')
    """
).to_pandas()
by_name

# %%
# By version number: the log row tells us which sequence the name pins.
pinned = db.sql(
    "SELECT pinned_sequence FROM snapshot_log WHERE snapshot_name = 'eod-2026-06-02'"
).to_pandas()["pinned_sequence"].iloc[0]
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
# ## 4. 2つの EOD 締めを SQL 1文で差分にする
#
# スナップショットは関係なので、「2日と3日のあいだで何が変わったか」は ETL ジョブではなく
# ジョインです。銘柄ごとに、増えた行数と、最終約定がどこへ動いたかを出します。

# %%
db.sql(
    """
    SELECT d3.symbol,
           d3.n - d2.n                      AS trades_added,
           d2.last_px                       AS close_jun02,
           d3.last_px                       AS close_jun03,
           round(d3.last_px / d2.last_px - 1, 4) AS px_chg
    FROM (SELECT symbol, count(*) AS n, last_value(price ORDER BY ts) AS last_px
          FROM h5i('trades', 'eod-2026-06-03') GROUP BY symbol) d3
    JOIN (SELECT symbol, count(*) AS n, last_value(price ORDER BY ts) AS last_px
          FROM h5i('trades', 'eod-2026-06-02') GROUP BY symbol) d2
    USING (symbol)
    ORDER BY symbol
    """
).to_pandas()

# %% [markdown]
# ## 5. 完全性の証明
#
# `verify(deep=True)` はマニフェストとセグメントをすべて読み直し、先頭から起点までチェックサムの
# 連鎖を検査します。`snapshot_log` に記録したマニフェストのチェックサムと組み合わせれば、
# そのまま差し出せる証明になります。ログに書かれた EOD の締めはチェックサムの一致する
# マニフェストに解決され、その下の全バイトが検証を通る、というわけです。

# %%
report = db.verify("trades", deep=True)
report

# %%
assert not report["problems"], f"integrity check failed: {report['problems']}"
print(f"checked {report['manifests_checked']} manifests, "
      f"{report['segments_checked']} segments, "
      f"{report['bytes_checked']:,} bytes - no problems")

# %% [markdown]
# ## 6. 保持: vacuum は履歴とスナップショットを尊重する
#
# `vacuum` が回収するのは、何からも参照されていないストレージだけです。ここではすべての
# セグメントが、バージョン連鎖と3つの EOD スナップショットを通じて到達可能なので、`grace_seconds=0`
# という攻撃的な実行でも何も削除されず、いちばん古い EOD の締めもそのあと読めるままです。
# スナップショットが保持ポリシーそのものです。残すべきものを固定し、残りを vacuum してください。

# %%
print("dry run:", db.vacuum(apply=False))
print("applied:", db.vacuum(grace_seconds=0, apply=True))
print("oldest EOD cut still readable:",
      f"{len(db.read('trades', snapshot='eod-2026-06-01')):,} rows")

# %% [markdown]
# ## まとめ
#
# - EOD の締めは `db.snapshot(name, tables=[...], note=...)` です。O(1)、データのコピーなし、
#   以後は SQL から `h5i('table', 'name')` として永久に参照できます。
# - 「日付Xの時点で把握していたこと」には等価な綴りが3つあります。スナップショット名、バージョン
#   番号、`as_of` のコミット時刻。どれも再生なしで解決します。
# - Python API はまだスナップショットを一覧できません（CLI はできます）。追記のみの
#   `snapshot_log` テーブルがその穴を埋め、締めごとの固定シーケンスとマニフェストのチェックサムを
#   持つ*クエリ可能でバージョン管理された*カタログになります。
# - `verify(deep=True)` と記録済みのチェックサムが、「信じてください」を証明に変えます。
#   スナップショットが固定しているものには、`vacuum` は手を出せません。

# %%
db.close()
