# %% [markdown]
# # 自分をだまさずに戦略空間を探索する
#
# 1回だけ走らせたバックテストは測定です。400回走らせて1つだけ報告したバックテストは選択で、
# スライドに載る数字は400回引いた中の最大値です。このレシピにあるものはすべて、その2つを
# 分けておくために存在します。
#
# ここには2種類の探索があり、形が違います。戦略パラメータは「何を売買したか」を変えるので、
# 候補ごとに別のシグナルテーブルと別の実行になります。執行パラメータは売買のコストだけを
# 変えるので、`backtest.study` で振れます。ホールドアウトを取っておく方法も、複数の期間で
# 候補を採点する方法も、study が知っています。
#
# レシピ 05/07 は予測市場でこれを行います。今回は株式で行い、探索が広いときにだけ効いてくる
# 部分を足します。ランダムサーチ、同じ実行を無料にする試行台帳、そして残った実行を1つに
# まとめるレポートです。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | グリッドサーチ | 挙げたパラメータの組み合わせをすべて試すこと |
# | ランダムサーチ | 与えた範囲からパラメータの組をランダムに引くこと |
# | ウォークフォワード | 時間順の複数の学習／ホールドアウト分割で候補を採点すること |
# | ホールドアウト | すでに選ばれた候補を採点するために、一度だけ使うデータ |
# | 試行台帳 | 採点した設定を、何を読み何をしたかで一意に記録したもの |
# | バスケットレポート | 保存済みテーブルから多数の実行を1つの文書にまとめたもの |
# | インサンプル | パラメータを選ぶのに使ったのと同じデータで測ること |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt
import itertools

import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import backtest, quant
import cookbook_utils as cu

CAPITAL = 250_000.0
LOT = 100.0

# %% [markdown]
# ## 1. マーケットデータは一度だけ
#
# 2022年以降の大型株10銘柄と、レシピ 04/08 とまったく同じやり方でバーから合成した板です。
# 以下のどの候補もこの1つのスナップショットを読むので、2つの違うテープの比較にはなりません。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES[:10], start="2020-01-01", end="2026-07-01")
frame = daily.to_pandas()
frame = frame[frame["ts"] >= pd.Timestamp("2022-01-01", tz="UTC")]
market = cu.make_equity_market(
    pa.Table.from_pandas(frame, preserve_index=False), spread_bps=4.0
)

db = h5i_db.Database(cu.fresh_db("04_searching_a_strategy_space"), create=True)
for name in ("instruments", "book_deltas", "trades"):
    table = market[name]
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="4bp synthetic book")
db.snapshot("tape-v1", tables=["instruments", "book_deltas", "trades"], note="One tape for every trial")

closes = frame.pivot(index="ts", columns="symbol", values="close").sort_index()
sessions = closes.index
book = market["book_deltas"].to_pandas()
book["session"] = book["ts_init"].dt.floor("s").dt.tz_localize("UTC")
quote_at = {(row.session, row.instrument_id): row.ts_init for row in book.itertuples()}
print(f"{len(sessions):,} sessions, {closes.shape[1]} names")
closes.tail(3)

# %% [markdown]
# ## 2. 候補ごとに1つのシグナルテーブル
#
# ルールは移動平均のクロスオーバーです。短期平均が長期平均を上回っているあいだ固定ロットを
# 持ち、それ以外はフラット。パラメータは2つの窓の長さです。
#
# これは `backtest.study` では振れません。そしてその拒否は意図的です。`data.signals` は
# *何を売買したか* を特定するので、それを振る study は、1つの戦略の設定を比べていると言い
# ながら別々の戦略を比べることになります。ですから候補ごとに自分のテーブルを持たせ、その
# テーブルは他と同じくバージョン管理されたデータになります。

# %%
def crossover_signals(fast: int, slow: int) -> pa.Table:
    """Target one lot per name while the fast average leads the slow one."""
    averages = {
        "fast": closes.rolling(fast).mean(),
        "slow": closes.rolling(slow).mean(),
    }
    holding = (averages["fast"] > averages["slow"]).astype(float) * LOT
    # The last session has no session after it to execute in, so it cannot
    # carry a target. The one before it is set flat, which makes final cash the
    # whole answer instead of cash plus an open position to mark.
    holding = holding.iloc[slow:-1]
    holding.iloc[-1] = 0.0

    stamps, targets, names = [], [], []
    for session, row in holding.iterrows():
        index = sessions.get_loc(session)
        execution = sessions[index + 1]
        for symbol, target in row.items():
            stamps.append(quote_at[(execution, symbol)] + dt.timedelta(microseconds=1))
            targets.append(float(target))
            names.append(symbol)
    order = sorted(range(len(stamps)), key=lambda i: (stamps[i], names[i]))
    return backtest.target_positions(
        [stamps[i] for i in order],
        [targets[i] for i in order],
        instrument_id=[names[i] for i in order],
        tag=f"ma-{fast}-{slow}",
    )


CANDIDATES = [
    (fast, slow)
    for fast, slow in itertools.product((10, 20, 50), (50, 100, 200))
    if fast < slow
]
for fast, slow in CANDIDATES:
    table = crossover_signals(fast, slow)
    name = f"signals_{fast}_{slow}"
    db.create_table(name, table.schema, time_column="ts")
    db.append(name, table, note=f"MA {fast}/{slow}")
db.snapshot("plans-v1", tables=[f"signals_{f}_{s}" for f, s in CANDIDATES])
print(f"{len(CANDIDATES)} candidates, one signals table each")
db.table(f"signals_{CANDIDATES[0][0]}_{CANDIDATES[0][1]}").head(3).to_pandas()

# %% [markdown]
# ## 3. インサンプルのリーダーボードと、それが結果でない理由
#
# 全期間で候補を走らせれば順位が出ます。規律を一切かける前に探索が生む順位であり、あとの
# 数字と比べる相手を作るためにここに載せます。

# %%
def config(run_id: str, signals: str, **execution) -> backtest.BacktestConfig:
    return backtest.BacktestConfig(
        run_id=run_id,
        portfolio=backtest.PortfolioConfig(starting_cash=CAPITAL),
        data=backtest.DataConfig(signals=signals, snapshot="tape-v1"),
        execution=backtest.ExecutionConfig(
            fee_kind="proportional", fee_rate=execution.get("fee_rate", 0.0005)
        ),
        output=backtest.OutputConfig(equity_interval_nanos=86_400_000_000_000),
    )


leaderboard = []
runs = {}
for fast, slow in CANDIDATES:
    label = f"ma-{fast}-{slow}"
    result = backtest.execute(db, config(label, f"signals_{fast}_{slow}"))
    runs[label] = result
    leaderboard.append(
        {
            "candidate": label,
            "fills": result["fills"],
            "commissions": result["commissions"],
            "net": result["final_cash"] - CAPITAL,
        }
    )
board = pd.DataFrame(leaderboard).sort_values("net", ascending=False)
board.round(2)

# %% [markdown]
# ## 4. 同じ試行は無料で、2回とは数えない
#
# ピン留めされた宣言的な設定はどれも、リプレイに影響するすべてをハッシュしたダイジェストを
# 持つ1つの試行です。同じものをもう一度出すと、実行せずに記録済みの結果を返し、試行回数は
# 増えません。これが効く理由は2つあります。再試行されたジョブが実効的な試行回数を黙って
# 増やせないこと、そして設定を回すエージェントが同じ問いに二度払わずに済むことです。

# %%
before = backtest.trial_count(db)
again = backtest.execute(db, config("ma-20-100", "signals_20_100"))
after = backtest.trial_count(db)
print(f"cached          {again['cached']}")
print(f"trial count     {before} -> {after}")
print(f"fork reused     {again['fork'] == runs['ma-20-100']['fork']}")
print(f"same final cash {again['final_cash'] == runs['ma-20-100']['final_cash']}")
assert again["cached"] and after == before
assert again["fork"] == runs["ma-20-100"]["fork"]

# %% [markdown]
# ## 5. 執行コストを正直に振る
#
# 先頭のルールに残る問いは、そのエッジが売買コストより大きいかどうかです。これは study の
# 仕事です。シグナルは固定で、動くのは `execution` だけだからです。
#
# 働くのは3つの部品です。`RandomSearch` はグリッドを列挙するかわりに範囲から引きます。空間が
# 広く、ほとんどの軸が効かないときに分がある選び方です。`WalkForward` は複数の学習／
# ホールドアウト分割で採点するので、たまたま良かった1期間だけでは候補を運べません。`TopK` は
# ホールドアウトを第2段階にします。候補は学習期間で順位づけされ、上位数件だけがアウトオブ
# サンプルに進みます。

# %%
best_label = board.iloc[0]["candidate"]
best_fast, best_slow = best_label.split("-")[1:]
edges = [sessions[i] for i in (0, len(sessions) // 3, 2 * len(sessions) // 3, len(sessions) - 1)]
folds = backtest.WalkForward.of(
    *[
        backtest.ValidationWindows(
            train=(edges[i].tz_localize(None), edges[i + 1].tz_localize(None)),
            holdout=(edges[i + 1].tz_localize(None), edges[i + 2].tz_localize(None)),
        )
        for i in range(len(edges) - 2)
    ]
)
study = backtest.study(
    db,
    study_id="cost-tolerance",
    base=config(f"study-{best_label}", f"signals_{best_fast}_{best_slow}"),
    parameters={"execution.fee_rate": backtest.Range(0.0, 0.004)},
    search=backtest.RandomSearch(trials=8, seed=5),
    validation=folds,
    selection=backtest.TopK(k=3, metric="final_cash"),
)
ranked = pd.DataFrame(study.ranked())
columns = [c for c in ("trial", "parameters", "train_median_final_cash",
                       "holdout_median_final_cash") if c in ranked.columns]
print(f"trials {len(study.trials)}, reached the holdout {len(study.selected)}")
ranked[columns].round(2)

# %% [markdown]
# 自分の study で確認すべき性質は、この表が示しているものです。ほとんどの候補にはホールド
# アウトの列がありません。全候補が触ったホールドアウトは、名前の良い2つ目の学習データです。

# %%
with_holdout = sum(
    1 for row in study.trials if any(key.startswith("fold0_holdout_") for key in row)
)
print(f"{with_holdout} of {len(study.trials)} trials ever saw the holdout")
assert with_holdout == len(study.selected) <= 3

# %% [markdown]
# 3つ目の探索の形が `TPESearch` です。これまでの結果から次の点を提案するので逐次実行になり、
# オプションの `optuna` が必要です。呼び出し方はそれ以外同じです。
#
# ```python
# search=backtest.TPESearch(trials=40, seed=7)
# ```
#
# どの探索でも、重複した引きは引き直さずそのまま残します。落とすと、デフレーテッド・シャープ
# の補正が依存する試行回数を、こっそり変えてしまうからです。

# %%
try:
    import optuna  # noqa: F401

    available = True
except ImportError:
    available = False
print(f"optuna installed: {available}")
print(f"trials recorded in this database so far: {backtest.trial_count(db)}")

# %% [markdown]
# ## 6. 探索全体を1つの文書に
#
# フォークが20個あっても比較にはなりません。`quant.basket_payload` は再シミュレーションなしに、
# 保存済みのテーブルから1つのレポートを組み立てます。バスケット全体のポートフォリオパネル、
# 読める大きさのあいだは実行ごとのパネル、そして描画を断ったものの記録です。

# %%
basket = {label: result for label, result in list(runs.items())[:6]}
report = quant.basket_payload(
    db,
    basket,
    basket_id="ma-crossover search",
    panels=quant.PORTFOLIO_PANELS + ("equity",),
    snapshot="tape-v1",
)
print(f"runs      {report.totals['runs']}")
print(f"net       {report.totals['net']:,.2f}")
print(f"drawdown  {report.totals.get('max_drawdown', 0):.4f}")
print(f"panels    {', '.join(report.drawn)}")
if report.skipped:
    print(f"not drawn {[item['panel'] for item in report.skipped]}")
path = report.to_html("data/cache/ma-search-basket.html")
print(f"\nwrote {len(path):,} bytes of self-contained HTML")

# %% [markdown]
# ## 7. 探索が結論から差し引くもの
#
# 候補9個とコストの引き8回は小さな探索ですが、それでも探索です。補正は結果と一緒に置くべき
# ものです。レシピ 05/05 はまさにこの状況でデフレーテッド・シャープとバックテスト過剰適合
# 確率を計算しますし、レシピ 06/03 は同じ考えを株式パネルのパージド交差検証に適用します。
#
# このレシピから持ち出す価値がある数字は試行回数です。正直な補正はどれもそれを必要とし、
# それを知っているのは台帳だけだからです。

# %%
print(f"strategy candidates run   {len(CANDIDATES)}")
print(f"execution trials run      {len(study.trials)}")
print(f"total trials in ledger    {backtest.trial_count(db)}")
print(f"best in-sample candidate  {best_label}")

# %% [markdown]
# ## まとめ
#
# - 戦略パラメータと執行パラメータには別々の仕組みが要ります。戦略候補ごとに1つのシグナル
#   テーブル、残りは `backtest.study` です。
# - 空間が広いときは `RandomSearch` がグリッドに勝ちます。`TPESearch` は `optuna` が必要で、
#   構造上逐次実行です。
# - `WalkForward` は1期間だけで候補を運ぶことを止め、`TopK` はホールドアウトを最終候補の
#   ために取っておきます。ほとんどの試行にホールドアウトの列がないことを確認してください。
# - 試行台帳は再送された設定を無料にし、試行回数を正直に保ちます。過剰適合の補正はどれも
#   その関数です。
# - `quant.basket_payload` は再シミュレーションなしに保存済みテーブルから実行を比較し、
#   描かなかったものは黙って間引かずに記録します。
# - アウトオブサンプルの何かが見るまで、リーダーボードは結果ではありません。

# %%
db.close()
