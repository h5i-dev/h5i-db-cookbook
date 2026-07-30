# %% [markdown]
# # ベンダーのミラーから正規テーブルへ
#
# 予測市場の調査はたいてい、すでにディスクにあるベンダーの Parquet ディレクトリから始まります。
# 時間単位の全フィードアーカイブか、日ごとのチャネルファイルです。このレシピでは、そのディレクトリを
# リプレイが読むテーブルに変えます。そして分量の大半を、結果が信頼できるかどうかを決める4つの点に
# 使います。ある行がどの結果に属するのか、取り込みを2回走らせたら何が起きるのか、対象期間のうち実際に
# どれだけ取れたのか、そして認識できない行を取り込み側がどう扱うのか、です。
#
# ここでは何も取得しません。ダウンロードは、認証情報やレート制限と一緒にスクリプトに置くものです。
# 再現できなければならない部分が `h5i_db.venues` です。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | ベンダーのミラー | すでにディスクにあるベンダーファイルのディレクトリ。多くの調査はここから始まる |
# | 正規テーブル | ベンダーのレイアウトが何であれ、リプレイが読む正規化された形 |
# | マーケット仕様 | マーケットの定義。結果、呼値の刻み、満期を含む |
# | レイアウト | どのベンダー列が何を意味するか。ハードコードせずデータとして宣言する |
# | コンテンツアドレス | ファイルのハッシュを鍵にすること。2回取り込んでも重複せず再生になる |
# | カバレッジ | 対象としていた期間のうち、取り込みが実際に取れた割合 |
# | 隔離（quarantine） | 認識できない行を、捨てずに取り込み側がどう扱うか |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import cookbook_utils as cu
import h5i_db
from h5i_db import venues

db = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp"), create=True)

# %% [markdown]
# ## 代わりのミラー
#
# `cu.write_polymarket_archive` は、合成パネルを、公開されている時間単位アーカイブが出す形に非正規化
# します。このレシピをオフラインで走らせるために置いてあります。ご自分のマシンでは、同じコードを実際の
# ミラーに向けて、このセルを消してください。
#
# アーカイブの1行が1イベントです。板の状態は価格帯をネストして持ち、増分の変化やプリントは
# `price`／`size`／`side` をフラットに持ちます。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `event_type` | `string` | `book`、`price_change`、`last_trade_price` など |
# | `timestamp` | `int64` | イベント時刻。単位は**ミリ秒** |
# | `market` | `string` | condition id |
# | `asset_id` | `string` | 結果ごとのトークン。これが結合キーになる |
# | `bids` / `asks` | `list<struct<price, size>>` | 板の状態の各価格帯 |
# | `price` / `size` / `side` | `float64` / `float64` / `string` | 1つの価格帯、または1件のプリント |

# %%
panel = cu.make_prediction_markets(n_markets=40, steps=24, seed=11)
mirror = Path("data/cache/onramp-mirror")
files = cu.write_polymarket_archive(panel, mirror)
print(f"{len(files)} hourly files under {mirror}")
sample = pq.read_table(files[0])
print(f"{sample.num_rows:,} rows x {sample.num_columns} columns in {files[0].name}")
sample.to_pandas().head()

# %% [markdown]
# ## ステップ1: マーケットとは何かを決める
#
# マーケット仕様は、取引できる1つの出来事の同一性です。結果、それぞれに対応するベンダーのトークン、
# 取引が止まる時刻、結果が知り得るようになった時刻からなります。結果の順序を間違えると、すべての約定が
# 反対のサイドに帰属してしまいます。ここは細かくこだわるべき手順です。
#
# `polymarket_markets_from_json` は、公開のマーケットエンドポイントが返すペイロードを読みます。
# 扱いにくい部分も含めてです。リスト型のフィールドは JSON エンコードされた文字列として届き、決着は
# 決済後の価格と closed フラグの組で表されます。

# %%
payloads = cu.polymarket_market_payloads(panel)
print(json.dumps(payloads[0], indent=2)[:420], "...")

specs = venues.polymarket_markets_from_json(payloads)
print(f"\n{len(specs)} markets parsed")
first = specs[0]
print(f"  {first.instrument_id}: outcomes {first.outcome_labels}")
print(f"  tokens              {first.tokens}")
print(f"  token -> outcome    {first.tokens[1]} is outcome {first.outcome_of_token(first.tokens[1])}")
print(f"  resolved            {first.is_resolved}, winner {first.winner_outcome}")

# %% [markdown]
# `outcome_labels` と `tokens` は位置で対応します。それぞれのインデックス `i` が同じ結果を指します。
# 契約はこれだけであり、仕様はこれを壊すあらゆる方法を、曖昧さを解決せずに拒否します。

# %%
for description, build in (
    ("one outcome", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes",))),
    ("tokens and outcomes disagree", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes", "No"), tokens=("a",))),
    ("resolved with no resolution time", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes", "No"), winner_outcome=0)),
):
    try:
        build()
    except ValueError as error:
        print(f"{description:32} refused: {str(error).split(';')[0][:78]}")

# A token two markets both claim would make every row keyed by it ambiguous.
try:
    venues.token_index([
        venues.MarketSpec(instrument_id="a", venue="v",
                          outcome_labels=("Yes", "No"), tokens=("t1", "t2")),
        venues.MarketSpec(instrument_id="b", venue="v",
                          outcome_labels=("Yes", "No"), tokens=("t1", "t3")),
    ])
except ValueError as error:
    print(f"{'token claimed twice':32} refused: {error}")

# %% [markdown]
# ## ステップ2: 定義を書き込む
#
# `instruments` には結果ごとに1行、`resolutions` には決着したマーケットごとに1行入ります。日付は
# 出来事が起きた時刻ではなく、結果が**観測可能**になった瞬間で付けます。あとで決済を制御するのが
# この列です。

# %%
report = venues.write_markets(db, specs, note="market definitions")
print(report)
print(f"\ninstruments rows: {report.tables['instruments'].rows}")
print(f"resolutions rows: {report.tables['resolutions'].rows}")
db.sql("SELECT * FROM instruments ORDER BY instrument_id, outcome LIMIT 4").to_pandas()

# %% [markdown]
# ## ステップ3: アーカイブを取り込む
#
# 行は、指定したマーケットのトークンに絞り込まれるので、全フィードの1時間ぶんでも、必要なマーケットの
# ぶんしかコストがかかりません。対象期間はエポックナノ秒の半開区間 `[start, end)` で、読む範囲を
# 決めます。

# %%
expiry = int(db.sql("SELECT max(expiration_ns) AS e FROM instruments").to_pandas()["e"][0])
start = int(pd.Timestamp("2026-05-01T12:00:00Z").value)
ingest = venues.ingest_archive(
    db,
    files=venues.discover(mirror),
    markets=specs,
    layout=venues.PMXT_LAYOUT,
    window=(start, expiry + 1),
    note="mirror backfill",
)
print(ingest)
print("rows by table: ", {name: write.rows for name, write in ingest.tables.items()})
print(f"files read:     {len(ingest.sources)}")
print(f"coverage:      {ingest.coverage:.4f}")

# %% [markdown]
# `book` イベントはまとめられたスナップショットになり、プリントは約定になりました。outcome 列が何を
# 示しているかに注目してください。各イベントは1つの銘柄の1つの結果を記述しており、これがエンジンが
# 破ることを拒む不変条件です。

# %%
grouping = db.sql(
    """
    SELECT count(*) AS events,
           min(rows_per_event) AS min_rows, max(rows_per_event) AS max_rows,
           sum(CASE WHEN outcomes > 1 THEN 1 ELSE 0 END) AS events_mixing_outcomes,
           sum(CASE WHEN terminators <> 1 THEN 1 ELSE 0 END) AS badly_terminated
    FROM (
        SELECT event_index, count(*) AS rows_per_event,
               count(DISTINCT outcome) AS outcomes,
               sum(CASE WHEN is_last THEN 1 ELSE 0 END) AS terminators
        FROM book_deltas GROUP BY event_index
    )
    """
).to_pandas()
print(grouping.to_string(index=False))
db.sql(
    "SELECT * FROM book_deltas ORDER BY ts_init, event_index, is_last LIMIT 4"
).to_pandas()

# %% [markdown]
# ## もう一度走らせる
#
# どのコミットも、そこに含まれる行のハッシュを鍵にしているので、同じ入力からは同じ鍵ができ、h5i-db が
# それを認識します。中断したバックフィルは安全に再開できますし、同じ1時間を配る2つのソースは、2つでは
# なく1つのコミットに収束します。

# %%
before = db.sql("SELECT count(*) AS n FROM book_deltas").to_pandas()["n"][0]
versions_before = len(db.versions("book_deltas"))
again = venues.ingest_archive(
    db,
    files=venues.discover(mirror),
    markets=specs,
    layout=venues.PMXT_LAYOUT,
    window=(start, expiry + 1),
)
after = db.sql("SELECT count(*) AS n FROM book_deltas").to_pandas()["n"][0]
print(f"first  pass replayed: {ingest.replayed}")
print(f"second pass replayed: {again.replayed}")
print(f"rows    {before:,} -> {after:,}")
print(f"versions {versions_before} -> {len(db.versions('book_deltas'))}")
assert after == before

# %% [markdown]
# ## カバレッジは仮定せず、事実として報告する
#
# `requested_window` と `loaded_window` は別々のままにしてあります。1週間ぶんを求めて6時間しか取れて
# いないというのは、ふつうに起きうる発見です。大事なのは、結果そのものから気づけることです。3手先の
# 奇妙なバックテストで気づくのでは遅すぎます。
#
# 期間を指定しなかったときの `coverage` は意図して `None` になります。範囲を区切っていない要求に対する
# 比率には、意味がないからです。

# %%
week = start + 7 * 24 * 3_600 * 1_000_000_000
probe = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_probe"), create=True)
short = venues.ingest_archive(
    probe, files=venues.discover(mirror), markets=specs,
    layout=venues.PMXT_LAYOUT, window=(start, week),
)
print(f"asked for {(week - start) / 3.6e12:.1f} hours")
print(f"loaded    {(short.loaded_window[1] - short.loaded_window[0]) / 3.6e12:.1f} hours")
print(f"coverage  {short.coverage:.4f}")

unbounded = venues.ingest_archive(
    probe, files=venues.discover(mirror), markets=specs, layout=venues.PMXT_LAYOUT
)
print(f"coverage with no window requested: {unbounded.coverage}")
probe.close()

# %% [markdown]
# ## 取り込み側が推測しないもの
#
# 寛容なローダなら隠してしまう2つの失敗です。レイアウトが必要とする列を欠いたファイルは、ファイル名と
# 欠けている列を記録したうえでスキップされます。データには現れるのにレイアウトにないイベント型は、
# 件数を数えます。効いていたはずの更新を黙って落とすことが、板がひそかに狂っていく道筋だからです。

# %%
broken = Path("data/cache/onramp-broken")
broken.mkdir(parents=True, exist_ok=True)
import pyarrow as pa

pq.write_table(pa.table({"nonsense": pa.array([1, 2, 3])}), broken / "bad.parquet")
# A layout that does not know about prints leaves them unrecognised.
book_only = venues.ArchiveLayout(
    name="book-only",
    timestamp_unit="ms",
    instrument_column="market",
    snapshot_events=("book",),
)
strict = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_strict"), create=True)
picky = venues.ingest_archive(
    strict,
    files=[broken / "bad.parquet", *venues.discover(mirror)],
    markets=specs,
    layout=book_only,
)
for item in picky.skipped:
    print(json.dumps(item)[:150])
strict.close()

# %% [markdown]
# ## ベンダーの方言はデータである
#
# `ArchiveLayout` は、列名、イベントの語彙、タイムスタンプの単位、価格帯の形を持ちます。同梱の2つの
# レイアウトはこの型のリテラルなので、3つめのベンダーは新しいモジュールではなく、もう1つのリテラルに
# なります。ここでは、新しいコードなしで自社形式のフィードを取り込んでいます。

# %%
print("PMXT_LAYOUT:")
for field in ("timestamp_column", "timestamp_unit", "token_column", "snapshot_events",
              "delta_events", "trade_events"):
    print(f"  {field:18} {getattr(venues.PMXT_LAYOUT, field)!r}")
print("\nTELONEX_LAYOUT differs only in these:")
for field in ("timestamp_unit", "event_type_column", "snapshot_events"):
    print(f"  {field:18} {getattr(venues.TELONEX_LAYOUT, field)!r}")

# %%
house = Path("data/cache/onramp-house")
house.mkdir(parents=True, exist_ok=True)
level = pa.struct([("px", pa.float64()), ("qty", pa.float64())])
token = specs[0].tokens[0]
pq.write_table(
    pa.table({
        "channel": pa.array(["depth"], pa.string()),
        "recv_ns": pa.array([start], pa.int64()),
        "token": pa.array([token], pa.string()),
        "buys": pa.array([[{"px": 0.31, "qty": 40.0}]], pa.list_(level)),
        "sells": pa.array([[{"px": 0.33, "qty": 35.0}]], pa.list_(level)),
    }),
    house / "day.parquet",
)
house_layout = venues.ArchiveLayout(
    name="house-feed",
    timestamp_column="recv_ns",
    timestamp_unit="ns",
    token_column="token",
    event_type_column="channel",
    snapshot_events=("depth",),
    levels=venues.LevelLayout(
        style="nested", bids_column="buys", asks_column="sells",
        price_field="px", size_field="qty",
    ),
)
other = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_house"), create=True)
house_report = venues.ingest_archive(
    other, files=[house / "day.parquet"], markets=specs, layout=house_layout
)
print(f"vendor: {house_report.vendor}, rows: {house_report.rows}")
print(other.sql("SELECT outcome, side, price, size FROM book_deltas").to_pandas().to_string(index=False))
other.close()

# %% [markdown]
# ## 同じ3ステップをシェルから
#
# マーケットの定義は JSON ファイルとして渡します。1つのマーケットは十数個のフィールド
# からなり、フラグの羅列では読めもしませんしバージョン管理もできないからです。`--min-coverage` は、
# 足りない読み込みを黙って通す代わりに非ゼロで終了します。定時のバックフィルで使えるのはこのおかげ
# です。

# %%
spec_path = Path("data/cache/onramp-specs.json")
spec_path.write_text(json.dumps(payloads), encoding="utf-8")
print(f"""
python -m h5i_db.venues markets  market.db {spec_path}
python -m h5i_db.venues ingest   market.db {spec_path} --root {mirror} \\
    --start-ns {start} --end-ns {expiry + 1} --min-coverage 0.95
python -m h5i_db.venues inspect  market.db
""".strip())

# %%
from h5i_db.venues.__main__ import main as venues_cli

cli_db = cu.fresh_db("05_vendor_data_onramp_cli")
assert venues_cli(["markets", cli_db, str(spec_path)]) == 0
assert venues_cli(["ingest", cli_db, str(spec_path), "--root", str(mirror)]) == 0
assert venues_cli(["inspect", cli_db]) == 0
# The gate fires when the window is wider than the data.
code = venues_cli([
    "ingest", cli_db, str(spec_path), "--root", str(mirror),
    "--start-ns", str(start), "--end-ns", str(week), "--min-coverage", "0.95",
])
print(f"\nexit code when coverage falls short: {code}")

# %% [markdown]
# ## これでテーブルはリプレイできる
#
# それがこの手順の目的です。スナップショットが取り込んだ内容をピン留めし、実行はほかの h5i-db の
# データとまったく同じ経路でそれを読みます。

# %%
db.snapshot("mirror-v1", tables=["instruments", "book_deltas", "trades", "resolutions"],
            note="ingested from the vendor mirror")
from h5i_db import backtest

# One order, timed a microsecond after a quote instant so it transacts at the
# price it was decided from. Recipe 05/07 builds real strategies on this data.
quotes = db.sql(
    f"""
    SELECT instrument_id, ts_init,
           max(CASE WHEN side = 'sell' THEN price END) AS ask
    FROM h5i('book_deltas', 'mirror-v1')
    WHERE outcome = 0
    GROUP BY instrument_id, ts_init
    ORDER BY ts_init, instrument_id
    LIMIT 200
    """
).to_pandas()
decision = quotes.iloc[len(quotes) // 2]
backtest.create_signal_table(db, "signals")
db.append("signals", backtest.signal_table([{
    "ts": decision.ts_init.to_pydatetime() + pd.Timedelta(microseconds=1).to_pytimedelta(),
    "instrument_id": decision.instrument_id, "outcome": 0,
    "side": "buy", "quantity": 10.0, "tag": "onramp",
}]))

config = backtest.BacktestConfig(
    run_id="onramp-probe",
    data=backtest.DataConfig(signals="signals", snapshot="mirror-v1"),
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=0.07),
)
inspection = backtest.inspect(db, config)
print(f"replay fidelity: {inspection.fidelity}")
print(f"config accepted: {inspection.ok}")
for name, stats in sorted(inspection.tables.items()):
    print(f"  {name:12} {stats['row_count']:>7,} rows")

result = backtest.execute(db, config)
fill = result.fills.to_pandas()
print(f"\nfilled {fill.quantity.iloc[0]:.0f} at {fill.price.iloc[0]:.3f}, "
      f"decision ask was {decision.ask:.3f}")
assert fill.price.iloc[0] == decision.ask

# %% [markdown]
# ## まとめ
#
# - 取り込みは3ステップである。マーケットのペイロードを解析し、`instruments` と `resolutions` を
#   書き、アーカイブを `book_deltas` と `trades` に正規化する。レシピ 05/07 は、ここで残った
#   スナップショットから始まる。
# - 結果の同一性は位置で決まり、それを壊すあらゆる方法が拒否される。2つのマーケットが主張するトークン、
#   観測可能時刻のない決着、結果リストと合わないトークンリストは、すべてエラーになる。これらの誤りが
#   黙って通ってしまうと、あとから取り返しがつかないからである。
# - 取り込みの再実行は再生になる。コミットは正規化された行のハッシュを鍵にするので、再開した
#   バックフィルは何も足さず、1時間ぶんの2つのソースは収束する。
# - `coverage` は要求した期間と読み込んだ期間を比べた値で、何も要求しなければ `None` になる。未知の
#   イベント型と使えないファイルは `report.skipped` に数えられ、黙って捨てられることはない。
# - ベンダーの方言は `ArchiveLayout` のリテラルなので、3つめのベンダーに新しいコードは要らない。
#   このレシピでは、それを示すために自社形式を取り込んだ。
# - ここで働いた h5i-db の機能。コンテンツアドレス方式の冪等キーが再取り込みを再生に変え、名前付き
#   スナップショットが取り込んだ内容をピン留めし、`versions()` が2回目の実行では何も書かれていない
#   ことを示した。

# %%
db.close()
