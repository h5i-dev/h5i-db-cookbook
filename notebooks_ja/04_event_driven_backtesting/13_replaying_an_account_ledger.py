# %% [markdown]
# # 口座の約定履歴をリプレイする
#
# このセクションの他のレシピは、戦略が何をしただろうかを問います。今回はバックテスターに
# 問える中でいちばん厳しい問いです。ある口座が *実際に* 行った売買が公開の台帳にあります。
# エンジンは、当時の板に対してそのポートフォリオを再現できるでしょうか。
#
# 答えはたいてい「いいえ」で、それは失敗というより発見です。約定を強制するシミュレータは、
# どんな台帳でも構造上そのまま再現してしまい、何も検証していません。ですから台帳は **注文意図**
# にコンパイルします。口座が得た価格の指値、IOC、売りには reduce-only を付けたうえで、
# 記録された板がそれを拒否できるようにします。
#
# 拒否した場所こそ、役に立つ出力です。それは自分のマーケットデータについての事実（手元の板は
# 彼らが売買した板ではない）か、台帳についての事実（その約定は報告どおりには起きていない）の
# どちらかです。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 台帳（レジャー） | ある口座が執行した売買の公開記録 |
# | 注文意図 | 会場が約定の可否を決める前の、送信された時点の注文 |
# | IOC（即時執行・残数量取消） | いま取れるだけ約定させ、残りは取り消す条件 |
# | reduce-only | ポジションを減らすことだけができ、反対側を新規に建てられない注文 |
# | 突き合わせ（リコンサイル） | 報告された内容とリプレイの結果を、市場ごとに比べること |
# | 約定率 | 台帳の数量のうち、リプレイが再現できた割合 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, venues
import cookbook_utils as cu

TICK = 0.001

# %% [markdown]
# ## 1. 口座が売買した相手のマーケットデータ
#
# 二値のイベント契約のパネルで、板、プリント、決済結果が入っています。これが当時の記録で、
# 以下の台帳は、その中で自分はこうしたと誰かが主張している内容です。
#
# このフィクスチャはすべての市場を同じ瞬間に更新するので、取り込む前に市場ごとに1ミリ秒ずつ
# ずらします。実際のフィードも12の市場を同一ナノ秒で提示したりはしませんし、このずらしは
# 答えを左右してしまう曖昧さを取り除きます。同時刻イベントの束に解放された注文は、マージが
# 最初に到達した市場についてだけ *新しい* 板に出会い、残りの市場については前の板に出会うのです。

# %%
raw = cu.make_prediction_markets(n_markets=12, steps=24, seed=5)
offsets = {
    name: pd.Timedelta(milliseconds=index)
    for index, name in enumerate(sorted(set(raw["instruments"].column("instrument_id").to_pylist())))
}


def stagger(table):
    """Give every market its own instant, as a real feed would."""
    frame = table.to_pandas()
    if "instrument_id" not in frame.columns:
        return table
    frame["ts_init"] = frame["ts_init"] + frame["instrument_id"].map(offsets)
    if "ts_event" in frame.columns:
        frame["ts_event"] = frame["ts_init"]
    frame = frame.sort_values("ts_init", kind="stable")
    return pa.Table.from_pandas(frame, schema=table.schema, preserve_index=False)


panel = {name: stagger(table) for name, table in raw.items()}
db = h5i_db.Database(cu.fresh_db("04_replaying_an_account_ledger"), create=True)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel fixture, one instant per market")
db.snapshot("books-v1", tables=list(panel), note="The book the ledger is checked against")
for name, table in panel.items():
    print(f"{name}: {table.num_rows:,} rows x {table.num_columns} columns")

specs = venues.polymarket_markets_from_json(cu.polymarket_market_payloads(raw))
print(f"\n{len(specs)} market specs, tokens resolved positionally to outcomes")
panel["book_deltas"].to_pandas().head(4)

# %% [markdown]
# ## 2. リプレイできない部分も含む台帳
#
# 本物の台帳はダウンロードしてくるものです。ここでは、突き合わせを既知の答えと照合できるように
# 台帳を構成します。3種類の行を入れ、比較はリプレイできない2種類をちょうど見つけるはずです。
#
# | 種類 | 内容 | 板がすべきこと |
# | --- | --- | --- |
# | 最良気配 | 表示数量より小さい、オファーでの買い | 約定させる |
# | 最良気配より有利 | 最良ビッドより下で報告された買い | 拒否する。そこには何も置かれていない |
# | 大きすぎ | 表示数量より大きい、オファーでの買い | 一部だけ約定させる |

# %%
book = panel["book_deltas"].to_pandas()
quotes = (
    book.pivot_table(
        index=["ts_init", "instrument_id", "outcome"],
        columns="side",
        values=["price", "size"],
        aggfunc="last",
    )
    .dropna()
    .reset_index()
)
quotes.columns = ["ts_init", "instrument_id", "outcome", "bid", "ask", "bid_size", "ask_size"]
quotes = quotes[quotes["outcome"] == 0].sort_values("ts_init")
print(f"{len(quotes):,} usable two-sided quotes")
quotes.head(3)

# %%
rows, kinds = [], []
for index, quote in enumerate(quotes.iloc[40:100:6].itertuples()):
    kind = ("at the touch", "better than the touch", "oversized")[index % 3]
    # Prices stay on the market's 0.001 grid; an off-grid order is refused
    # before it ever meets the book.
    price = (
        round(float(quote.bid) - TICK, 3)
        if kind == "better than the touch"
        else float(quote.ask)
    )
    quantity = float(quote.ask_size) * (3.0 if kind == "oversized" else 0.4)
    rows.append(
        venues.LedgerRow(
            ts_ns=int(pd.Timestamp(quote.ts_init).value),
            instrument_id=quote.instrument_id,
            outcome=0,
            side="buy",
            quantity=round(quantity, 2),
            price=price,
            trade_id=f"0x{index:04d}",
        )
    )
    kinds.append(kind)

ledger = venues.ledger_table(rows).to_pandas().assign(kind=kinds)
print(f"{len(rows)} ledger rows over {ledger['instrument_id'].nunique()} markets")
ledger.head(6)

# %% [markdown]
# `LedgerRow` は、ポジションを黙って反転させかねないものを拒否します。`(0, 1)` の外の価格は
# 確率ではありませんし、正でない数量は約定ではありませんし、buy でも sell でもないサイドは
# 売買ではありません。

# %%
for description, build in (
    ("price of 1.4", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="buy", quantity=10.0, price=1.4)),
    ("quantity of 0", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="buy", quantity=0.0, price=0.5)),
    ("side 'long'", lambda: venues.LedgerRow(
        ts_ns=0, instrument_id="m", outcome=0, side="long", quantity=10.0, price=0.5)),
):
    try:
        build()
    except ValueError as error:
        print(f"{description:16} refused: {str(error).split(';')[0][:70]}")

# %% [markdown]
# ## 3. 台帳を注文意図にコンパイルする
#
# `commands_from_ledger` は各行を、台帳が報告する価格の指値、IOC、主張された瞬間の1マイクロ秒
# あとのタイムスタンプに変えます。売りは reduce-only になるので、台帳に現れていないショートを
# リプレイが作り出すことはできません。
#
# ベンダーの辞書形式もそのまま使えます。生の行とマーケット仕様を渡せば、トークンが位置で
# アウトカムに解決されます。ここで型付きの行を使っているのは、以下の突き合わせがどのみち
# それを必要とするからです。

# %%
commands = venues.commands_from_ledger(rows, specs)
backtest.create_command_table(db, "commands")
db.append("commands", commands, note="one account's published trades, as intent")
frame = commands.to_pandas()
print(f"{commands.num_rows} commands")
frame[["ts", "action", "instrument_id", "side", "quantity", "kind", "limit_price",
       "time_in_force", "reduce_only"]].head(4)

# %% [markdown]
# ## 4. リプレイ
#
# `DataConfig(commands=...)` がシグナルテーブルをコマンドの流れに置き換えます。それ以外は、
# ピン留めした板に対するふつうの実行です。

# %%
result = backtest.execute(
    db,
    backtest.BacktestConfig(
        run_id="ledger-replay",
        portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
        data=backtest.DataConfig(commands="commands", snapshot="books-v1"),
        execution=backtest.ExecutionConfig(fee_kind="prediction_market", fee_rate=0.02),
        output=backtest.OutputConfig(equity_interval_nanos=900_000_000_000),
    ),
)
print(f"orders {result['orders']}  fills {result['fills']}")
result.orders.to_pandas()[["order_id", "instrument_id", "side", "quantity", "filled", "status"]].head(6)

# %% [markdown]
# ## 5. 市場ごとの突き合わせ
#
# `compare_to_ledger` は合否を1つ返すのではなく、リプレイと台帳が食い違った場所を報告します。
# 板が *どこで* 拒否したかが発見だからです。

# %%
comparison = venues.compare_to_ledger(result, rows)
print(f"ledger rows        {comparison['ledger_rows']}")
print(f"replay fills       {comparison['replay_fills']}")
print(f"ledger quantity    {comparison['ledger_quantity']:,.2f}")
print(f"replay quantity    {comparison['replay_quantity']:,.2f}")
print(f"fill ratio         {comparison['fill_ratio']:.1%}")
print(f"markets reproduced {comparison['markets_reproduced']} of {len(comparison['markets'])}")
pd.DataFrame(comparison["markets"]).round(3).head(8)

# %% [markdown]
# リプレイできなかった行は、できないように作った行です。行ごとの結果を、それがどの種類の行
# だったかに結び直すと、突き合わせは数字ではなくデータについての言明になります。

# %%
submitted = result.orders.to_pandas().sort_values("order_id").reset_index(drop=True)
assert (submitted["quantity"].round(2).to_numpy() == ledger["quantity"].round(2).to_numpy()).all(), (
    "orders and ledger rows are no longer in the same order"
)
ledger["replayed"] = submitted["filled"].to_numpy()
by_kind = ledger.groupby("kind").agg(
    rows=("quantity", "size"),
    ledger_quantity=("quantity", "sum"),
    replayed_quantity=("replayed", "sum"),
)
by_kind["ratio"] = by_kind["replayed_quantity"] / by_kind["ledger_quantity"]
by_kind.round(3)

# %% [markdown]
# 3つの行は、3つの別の会話として読みます。「最良気配」は再現できたので、問題ありません。
# 「最良気配より有利」はまったく約定しませんでした。実データなら、見えていない流動性か、
# 見ていない会場か、仲値で報告された記録のいずれかです。「大きすぎ」は一部だけ約定しました。
# 表示されていた板ではその売買を支えられなかったということで、自分のアーカイブの厚みについての
# 言明になります。

# %%
assert by_kind.loc["at the touch", "ratio"] > 0.99
assert by_kind.loc["better than the touch", "ratio"] < 0.01
assert by_kind.loc["oversized", "ratio"] < 1.0
print("the reconciliation found exactly the rows that were built to fail")

# %% [markdown]
# ## 6. 約定しなかった注文を読む価値
#
# 実行自身の説明には、約定を生まなかった注文の件数が入っています。台帳が本物で、誰も失敗を
# 仕込んでいないときに見るべき数字です。

# %%
explanation = result.explain()
print(f"orders without fills {explanation.get('orders_without_fills')}")
unfilled = result.orders.to_pandas()
unfilled = unfilled[unfilled["filled"] == 0]
print(f"{len(unfilled)} of {result['orders']} orders never filled")
unfilled[["instrument_id", "side", "quantity", "limit_price", "status"]].head(5)

# %% [markdown]
# ## まとめ
#
# - 台帳は *注文意図* としてリプレイします。約定を強制することはしません。強制すれば構造上
#   台帳を再現してしまい、何も検証できません。
# - `commands_from_ledger` が指値・IOC・reduce-only を選ぶのは意図的です。どれも、リプレイが
#   ポジションを作り出す経路を1つずつ塞いでいます。
# - `compare_to_ledger` が市場ごとの不足を報告するのは、どの市場がどれだけ食い違ったかこそが
#   役に立つ出力だからです。
# - リプレイできない行は証拠です。見えない流動性、抜けている会場、仲値での報告、あるいは
#   その売買を支える厚みのないアーカイブ。
# - コマンドの流れはふつうのピン留めされたテーブルなので、突き合わせもこのセクションの他の
#   実行と同じだけ再現できます。

# %%
db.close()
