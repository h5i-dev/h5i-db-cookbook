# %% [markdown]
# # 再現できるバックテストを運用する
#
# バックテストは過去についての主張であり、その価値は裏づけとなる証拠の価値と同じです。半年後、
# レビューする人はもう一度走らせて同じ数字を得たいと思います。
#
# やっかいなのは、調査用データベースが取り込みを続けることです。遅れてプリントが届き、ベンダーが
# 数値を訂正し、同じ実行が前より大きなテーブルを読むようになります。コードをピン留めするだけでは
# 足りません。足元でデータが動いたからです。
#
# 再現性は乱数シードの話ではなく、運用の性質です。このレシピでは、マーケットデータの断面をピン留め
# し、遅れて届いたデータを追記し、ピン留めした実行が変わらないことを示し、レビューに要る実行
# マニフェストを確認します。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 再現性 | 数か月後に再実行しても同じバイト列を読み、同じ数字が出ること |
# | ピン留め | 実行の入力を特定のバージョンやスナップショットに固定すること |
# | 遅れて届くデータ | ある期間の行が、その期間をすでに読んだあとで到着すること |
# | カバレッジ・ゲート | ピン留めした窓に、実行が前提としたデータが実際に入っているかの検査 |
# | 実行マニフェスト | その実行が何を読み何を出したかの記録。レビューが読むのはこれ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %% [markdown]
# 240秒ぶんのテープを1本作り、承認済みの180秒の断面と、遅れて届く末尾に分けます。実行が認定された
# あとも取り込みを続ける調査用データベースを模したものです。
#
# | 入力 | 行数 | 役割 |
# |---|---:|---|
# | `instruments` | 2 | 取引所と契約の安定したメタデータ |
# | 初期の `book_deltas` | 360 | 承認済みの L2 断面 |
# | 初期の `trades` | 180 | 承認済みのプリント断面 |
# | 遅れて届く末尾 | 板360行 + 約定60件 | あとから続く取り込み |

# %%
import datetime as dt

import pandas as pd
import pyarrow.compute as pc

import h5i_db
from h5i_db import backtest
import cookbook_utils as cu

full = cu.make_backtest_fixture(steps=240)
base = dt.datetime(2026, 6, 1, 14, 0, 0)
cutoff = base + dt.timedelta(seconds=181)

initial_book = full["book_deltas"].filter(
    pc.less(full["book_deltas"]["ts_init"], cutoff)
)
late_book = full["book_deltas"].filter(
    pc.greater_equal(full["book_deltas"]["ts_init"], cutoff)
)
initial_trades = full["trades"].filter(pc.less(full["trades"]["ts_init"], cutoff))
late_trades = full["trades"].filter(
    pc.greater_equal(full["trades"]["ts_init"], cutoff)
)

print(f"initial book: {initial_book.num_rows:,} rows")
print(f"late book: {late_book.num_rows:,} rows")
initial_book.to_pandas().tail()

# %% [markdown]
# 承認済みの断面を作って読み込みます。初期バッチと後続バッチでテーブルのスキーマが同じなので、
# append が1本の論理的な履歴を保てます。

# %%
db = h5i_db.Database(
    cu.fresh_db("04_reproducible_backtest_operations"),
    create=True,
)
db.create_table(
    "instruments",
    full["instruments"].schema,
    time_column="ts_init",
)
db.create_table("book_deltas", initial_book.schema, time_column="ts_init")
db.create_table("trades", initial_trades.schema, time_column="ts_init")
db.append("instruments", full["instruments"], note="reference data")
db.append("book_deltas", initial_book, note="approved 180-second book cut")
db.append("trades", initial_trades, note="approved 180-second trade cut")
db.snapshot(
    "approved-cut",
    tables=["instruments", "book_deltas", "trades"],
    note="Input approved by research controls",
)
db.versions("book_deltas")

# %% [markdown]
# 戦略は承認済みの断面の中で建玉を作り、遅れて届く末尾で決済します。どの実行でも同じ signals
# テーブルを使うので、変わるのはマーケットデータの読み出し地点だけです。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts` | `timestamp[ns]` | 注文意図が到着する時刻 |
# | `instrument_id` | `string` | 契約の識別子 |
# | `side` | `string` | 買いで建て、売りで決済 |
# | `quantity` | `float64` | 要求する数量 |
# | `tag` | `string` | 安定した監査用ラベル |

# %%
signals = backtest.signal_table(
    [
        {
            "ts": base + dt.timedelta(seconds=30),
            "instrument_id": "RATE-CUT-YES",
            "side": "buy",
            "quantity": 50.0,
            "tag": "open-approved",
        },
        {
            "ts": base + dt.timedelta(seconds=210),
            "instrument_id": "RATE-CUT-YES",
            "side": "sell",
            "quantity": 50.0,
            "tag": "close-late",
        },
    ]
)
print(f"{signals.num_rows:,} rows x {signals.num_columns} columns")
signals.to_pandas()

# %% [markdown]
# 戦略の意図は、マーケットのスナップショットを作ったあとに保存します。実行フォークは戦略テーブルを
# 別にピン留めするので、スナップショットはマーケットデータの承認地点として純粋なまま残ります。

# %%
backtest.create_signal_table(db)
db.append("signals", signals, note="approved strategy intent")

# %% [markdown]
# 遅れたデータを取り込む前に、承認済みの断面に対して実行します。ピン留めしたデータでリプレイが
# 終わるため、到達するのは建玉のシグナルだけです。

# %%
first = backtest.run(
    db,
    "approved-before-late-data",
    starting_cash=10_000.0,
    signals="signals",
    snapshot="approved-cut",
    equity_interval_nanos=10_000_000_000,
)
first

# %% [markdown]
# 遅れて届いた末尾を、ふつうの取り込みとして追記します。既存のバージョンと名前付きスナップショットは
# そのまま読めますし、スナップショットにデータが複製されることもありません。

# %%
db.append("book_deltas", late_book, note="late-arriving final minute")
db.append("trades", late_trades, note="late-arriving final minute")
print(db.versions("book_deltas")[-1])

# %% [markdown]
# 承認済みスナップショットで再実行し、あわせて最新のデータでも1回実行します。ピン留めした結果は
# 変わらないはずです。最新での実行は決済シグナルまで到達できるので、調査の入力としては別物です。

# %%
pinned_again = backtest.run(
    db,
    "approved-after-late-data",
    starting_cash=10_000.0,
    signals="signals",
    snapshot="approved-cut",
    equity_interval_nanos=10_000_000_000,
)
latest = backtest.run(
    db,
    "latest-after-late-data",
    starting_cash=10_000.0,
    signals="signals",
    equity_interval_nanos=10_000_000_000,
)

comparison = pd.DataFrame(
    [
        {"run": "pinned before append", **first},
        {"run": "pinned after append", **pinned_again},
        {"run": "latest after append", **latest},
    ]
).set_index("run")
comparison[
    [
        "fills",
        "orders",
        "records_processed",
        "final_cash",
        "realized_pnl",
    ]
]

# %% [markdown]
# レビューする人が気にする証拠を、そのまま表明します。ピン留めした実行どうしは、金額もイベント数も
# 一致します。最新での実行は、意図して違う結果になります。

# %%
stable_fields = (
    "fills",
    "orders",
    "records_processed",
    "final_cash",
    "realized_pnl",
    "commissions",
)
assert all(first[field] == pinned_again[field] for field in stable_fields)
assert latest["records_processed"] > first["records_processed"]
assert latest["fills"] > first["fills"]
print("Pinned result survived subsequent ingestion unchanged.")

# %% [markdown]
# カバレッジ・ゲートは、不完全なデータをその場での失敗に変えます。ここでは、承認済みの断面が、
# 遅れて届く末尾まで伸びる窓を満たせません。

# %%
try:
    backtest.run(
        db,
        "coverage-must-fail",
        starting_cash=10_000.0,
        signals="signals",
        snapshot="approved-cut",
        window=(
            base + dt.timedelta(seconds=1),
            base + dt.timedelta(seconds=240),
        ),
        minimum_coverage=0.95,
    )
except h5i_db.InvalidInputError as error:
    print(f"Rejected as intended: {error}")
else:
    raise AssertionError("the incomplete approved cut passed its coverage gate")

# %% [markdown]
# 実行フォークにはそれぞれ、1行のマニフェストと詳細な結果テーブルが入っています。マニフェスト、
# ソースのスナップショット名、戦略のバージョン、設定の4つは、レビュー用の資料に残してください。
# 約定テーブルが執行についての正本であることは変わりません。

# %%
audit_rows = []
for label, report in (
    ("pinned-before", first),
    ("pinned-after", pinned_again),
    ("latest", latest),
):
    run_db = db.fork(report["fork"])
    manifest = run_db.read("bt_run").to_pandas().iloc[0].to_dict()
    manifest["label"] = label
    manifest["fork"] = report["fork"]
    audit_rows.append(manifest)
    run_db.close()
audit = pd.DataFrame(audit_rows).set_index("label")
audit[
    [
        "run_id",
        "config_digest",
        "records_processed",
        "final_cash",
        "realized_pnl",
        "fork",
    ]
]

# %% [markdown]
# ## まとめ
#
# - バックテストの結果を受け入れる前に、マーケットデータをピン留めする。
# - 戦略の意図は、過去データの断面とは別にバージョン管理する。
# - 名前付きスナップショットでの再実行は、あとから追記があっても安定している。
# - カバレッジ・ゲートは、それらしい指標が出てしまう前に、切り詰められた窓を弾く。
# - 実行フォーク、マニフェスト、ダイジェスト、約定は、1つのレビュー可能な成果物として保存する。

# %%
db.close()
