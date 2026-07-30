# %% [markdown]
# # 予測市場のバックテストが嘘をつく2つの経路
#
# 1つめは決済です。マーケットが決着する前に止まったリプレイは、まだポジションを抱えています。それを
# 最終的な勝者で評価すると、その期間に取引していた誰も受け取れなかったお金を計上することになります。
# 2つめは選択です。イベント契約は数が少なく、40のマーケットをまたぐしきい値のスイープには当てはめの
# 自由度がありあまるほどあるので、勝ったしきい値は最良というより最も運が良かっただけ、ということが
# よくあります。このレシピでは1つめを実演し、2つめに数値を与えます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | 決済リスク | 未決済のポジションを最終的な勝者で評価すること。誰も受け取れなかった利益になる |
# | 観測可能性のゲート | 結果が実際に知り得るようになるまで決済を認めないこと |
# | 選択バイアス | スイープの勝者が、最良の候補というより最も運の良かった候補であること |
# | ホールドアウト | 最後まで手をつけずに取っておくデータ。使えるのは一度きり |
# | オーバーフィット | 手元の標本のノイズを説明するだけの規則を見つけてしまうこと |
# | PBO | バックテスト過学習確率。0.5 を超えるなら選び方がランダムより悪い |
# | デフレーテッド・シャープ | 試した候補の数だけ下方修正したシャープレシオ |
# | 最小トラックレコード長 | シャープレシオがゼロと区別できるまでに必要な運用期間 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cookbook_utils as cu
import h5i_db
from h5i_db import backtest, quant

db = h5i_db.Database(cu.fresh_db("05_settlement_and_selection_risk"), create=True)
FEE_RATE = 0.07
QUANTITY = 20.0

# %% [markdown]
# ## パネル
#
# 取引は満期まで続き、結果はその45分後に観測可能になり、パネルはその後も決着済みの板の気配を出し
# 続けます。2つの瞬間を記録するのが `instruments` テーブルで、決済を制御するのは2つめのほうです。
#
# | 列 | 型 | 意味 |
# |---|---|---|
# | `ts_init` | `timestamp[ns]` | 定義が記録された時刻 |
# | `instrument_id` | `string` | マーケット |
# | `outcome` / `outcome_label` | `uint16` / `string` | 0 = YES、1 = NO |
# | `tick_size` / `lot_size` | `float64` | 気配と数量のグリッド |
# | `expiration_ns` | `int64` | 取引が止まる時刻 |
# | `settlement_observable_ns` | `int64` | 結果が**知り得るようになった**時刻 |

# %%
panel = cu.make_prediction_markets(n_markets=90, steps=28, seed=11)
for name, table in panel.items():
    db.create_table(name, table.schema, time_column="ts_init")
    db.append(name, table, note="panel load")
db.snapshot("panel-v1", tables=list(panel), note="settlement study input")
instruments = panel["instruments"]
print(f"instruments: {instruments.num_rows} rows x {instruments.num_columns} columns")
instruments.to_pandas().head(4)

# %%
stamps = sorted({value.as_py() for value in panel["book_deltas"].column("ts_init")})
timing = db.sql(
    """
    SELECT max(expiration_ns) AS expiry, max(settlement_observable_ns) AS observable
    FROM instruments
    """
).to_pandas().iloc[0]
expiry = pd.Timestamp(int(timing.expiry), unit="ns")
observable = pd.Timestamp(int(timing.observable), unit="ns")
print(f"trading ends       {expiry}")
print(f"result observable  {observable}")
print(f"data ends          {stamps[-1]}")

# %% [markdown]
# ## 1つのポジション、2つの窓
#
# 同じ建玉を2回リプレイします。短いほうの実行は取引セッションの内側で止まり、完全な実行は観測可能に
# なる瞬間を越えて進みます。ほかは何も変えていないので、動くのは決済の判断だけです。

# %%
decision = stamps[6]
truth = cu.market_truth(panel).to_pandas()
quotes = db.sql(
    f"""
    SELECT instrument_id,
           max(CASE WHEN side = 'sell' THEN price END) AS ask
    FROM h5i('book_deltas', 'panel-v1')
    WHERE outcome = 0 AND ts_init = to_timestamp_nanos({int(pd.Timestamp(decision, tz='UTC').value)})
    GROUP BY instrument_id
    """
).to_pandas().merge(truth, on="instrument_id")
favorites = quotes[quotes.ask >= 0.70].sort_values("instrument_id")
signals = backtest.signal_table(
    [
        {
            "ts": decision + dt.timedelta(microseconds=1),
            "instrument_id": row.instrument_id,
            "outcome": 0,
            "side": "buy",
            "quantity": QUANTITY,
            "tag": "favorite",
        }
        for row in favorites.itertuples()
    ]
).sort_by([("ts", "ascending")])
backtest.create_signal_table(db, "signals")
db.append("signals", signals)
print(f"{len(favorites)} favorites at mean ask {favorites.ask.mean():.3f}")


def run(run_id: str, window=None) -> backtest.BacktestResult:
    return backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=run_id,
            data=backtest.DataConfig(signals="signals", snapshot="panel-v1", window=window),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE),
            metadata={"study": "settlement"},
        ),
    )


short = run("short-window", window=(stamps[0], stamps[20]))
full = run("full-window")

# %% [markdown]
# 短いほうの実行は、すべてのポジションを未決済のまま残し、マーケットごとに理由を述べます。この
# メッセージこそが機能です。未決済のポジションは、黙って評価されるのではなく報告されます。

# %%
for label, result in (("short", short), ("full", full)):
    manifest = result.run.to_pandas().iloc[0]
    positions = result.positions.to_pandas()
    attributed = positions.settlement_pnl.notna().sum()
    print(f"{label:6} settlement_applied={bool(manifest.settlement_applied)}  "
          f"simulated_through={pd.Timestamp(int(manifest.simulated_through_ns), unit='ns')}")
    print(f"       positions={len(positions)}  with settlement attribution={attributed}")

# On the unsettled run BOTH attribution columns are null. The engine does not
# offer a settled number and does not quietly substitute the mark either; the
# mark-to-market value stays where it belongs, on the equity curve.
short_positions = short.positions.to_pandas()
print(f"\nshort run: settlement_pnl all null = {short_positions.settlement_pnl.isna().all()}, "
      f"market_exit_pnl all null = {short_positions.market_exit_pnl.isna().all()}")
short_equity = short.equity.to_pandas()
print(f"short run unrealized P&L at the last sample: {short_equity.unrealized_pnl.iloc[-1]:,.2f}")
warning = short.run.to_pandas().warnings.iloc[0]
print(f"\nfirst warning from the short run:\n  {str(warning).split(';')[0]}")

# %% [markdown]
# 完全な実行では両方の数値が残るので、調整の跡をたどれます。`market_exit_pnl` は決着前の最後の気配で
# 評価したポジション、`settlement_pnl` は決着時にそれがどうなったかです。差が、最後まで持ち切ることに
# まるごと依存している部分にあたります。

# %%
full_positions = full.positions.to_pandas()
adjustment = full_positions.settlement_pnl.sum() - full_positions.market_exit_pnl.sum()
print(f"marked to the last quote: {full_positions.market_exit_pnl.sum():>9,.2f}")
print(f"settled at resolution:    {full_positions.settlement_pnl.sum():>9,.2f}")
print(f"settlement adjustment:    {adjustment:>+9,.2f}")
print(f"\nshare of the result that only exists if you hold: {adjustment / full_positions.settlement_pnl.sum():.0%}")

# %% [markdown]
# ## このパネルを時間で分割できない理由
#
# アウトオブサンプル検証と聞いてまず思いつくのは、標本を日付で半分に切ることです。`backtest.study`
# は `ValidationWindows` でそれを直接支えているので、ここで何が出るのか、そしてなぜその答えが役に
# 立たないのかを見るために、一度走らせておく価値があります。
#
# このパネルではすべてのマーケットが同じ瞬間に決着します。したがって時間分割は、建玉をすべて片方の
# 半分に、決済をすべてもう片方に置いてしまいます。学習側は決済に到達できず、ホールドアウト側は一度も
# 取引しません。

# %%
mid_point = stamps[len(stamps) // 2]
study = backtest.study(
    db,
    study_id="time-split",
    base=backtest.BacktestConfig(
        run_id="time-split",
        data=backtest.DataConfig(signals="signals", snapshot="panel-v1"),
        portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
        execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE),
    ),
    parameters={"execution.fee_rate": [0.0, 0.07]},
    validation=backtest.ValidationWindows(
        train=(stamps[0], mid_point), holdout=(mid_point, stamps[-1])
    ),
)
board = pd.DataFrame(study.leaderboard("holdout_final_cash"))
print(board[["trial", "parameters", "train_fills", "holdout_fills",
             "train_final_cash", "holdout_final_cash"]].to_string(index=False))
print(f"\nattention state: {study.attention_state}  unseen warnings: {study.warning_badge}")
print("the attention state is FAILED_WARNED because the train phase could not")
print("settle: the study surfaced the defect rather than leaving it in a column")

# %% [markdown]
# ホールドアウト列がどの試行でも同じなのは、ホールドアウト側が何も執行しなかったからです。片側で
# 取引が起きない分割は検証ではありませんし、その上で計算した統計量には何の意味もありません。
#
# クロスセクションの発見なら、分割すべき軸はクロスセクションです。互いに重ならない2つのマーケット群を、
# 同じやり方で取引し、どちらもふつうに決済させます。

# %%
half = len(favorites) // 2
folds = {
    "fold-a": favorites.iloc[:half],
    "fold-b": favorites.iloc[half:],
}
fold_report = []
for name, members in folds.items():
    table = backtest.signal_table(
        [
            {
                "ts": decision + dt.timedelta(microseconds=1),
                "instrument_id": row.instrument_id,
                "outcome": 0,
                "side": "buy",
                "quantity": QUANTITY,
                "tag": name,
            }
            for row in members.itertuples()
        ]
    ).sort_by([("ts", "ascending")])
    db.create_table(f"signals_{name}", table.schema, time_column="ts")
    db.append(f"signals_{name}", table)
    result = backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=name,
            data=backtest.DataConfig(signals=f"signals_{name}", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE),
        ),
    )
    positions = result.positions.to_pandas()
    fills = result.fills.to_pandas()
    capital = float((fills.price * fills.quantity).sum())
    net = float(result.summary()["realized_pnl"] + positions.settlement_pnl.fillna(0.0).sum())
    fold_report.append(
        {
            "fold": name,
            "markets": len(members),
            "capital": round(capital, 2),
            "net": round(net, 2),
            "net_pct": round(net / capital * 100, 2) if capital else float("nan"),
        }
    )
print(pd.DataFrame(fold_report).to_string(index=False))
print("\nagreeing signs across disjoint market sets is weak evidence, but it is")
print("evidence; a time split on this panel was not even that.")

# %% [markdown]
# ## しきい値グリッド上の選択リスク
#
# ここからが、レシピ 05/03 のフェイバリットの発見が本物かどうかを決める部分です。しきい値ごとに専用の
# signals テーブルと専用の実行を与えるので、それぞれが独自のエクイティカーブを持つ本物の試行になり
# ます。90のマーケットに対して11試行は、かなりの自由度です。問うべきは、勝者の成績のうちどれだけが
# エッジによるものか、その自由度によるものか、です。

# %%
thresholds = np.round(np.arange(0.50, 0.94, 0.04), 2)
curves = {}
for threshold in thresholds:
    members = quotes[quotes.ask >= threshold]
    if len(members) < 5:
        continue
    name = f"th{int(threshold * 100)}"
    table = backtest.signal_table(
        [
            {
                "ts": decision + dt.timedelta(microseconds=1),
                "instrument_id": row.instrument_id,
                "outcome": 0,
                "side": "buy",
                "quantity": QUANTITY,
                "tag": name,
            }
            for row in members.itertuples()
        ]
    ).sort_by([("ts", "ascending")])
    db.create_table(f"signals_{name}", table.schema, time_column="ts")
    db.append(f"signals_{name}", table)
    result = backtest.execute(
        db,
        backtest.BacktestConfig(
            run_id=name,
            data=backtest.DataConfig(signals=f"signals_{name}", snapshot="panel-v1"),
            portfolio=backtest.PortfolioConfig(starting_cash=100_000.0),
            execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=FEE_RATE),
            output=backtest.OutputConfig(equity_interval_nanos=15 * 60 * 1_000_000_000),
            metadata={"study": "threshold-grid", "threshold": float(threshold)},
        ),
    )
    equity = result.equity.to_pandas()
    positions = result.positions.to_pandas()
    curves[name] = {
        "threshold": float(threshold),
        "markets": len(members),
        "equity": equity.set_index("ts").equity if "equity" in equity else None,
        "net": float(result.summary()["realized_pnl"] + positions.settlement_pnl.fillna(0.0).sum()),
    }
print(f"{len(curves)} trials")
print(pd.DataFrame([{k: v[k] for k in ("threshold", "markets", "net")} for v in curves.values()]).to_string(index=False))

# %% [markdown]
# ## バックテスト過学習確率
#
# 組み合わせ的に対称な交差検証は、標本をブロックに切り、半分と半分のあらゆる分割を作り、インサンプル
# の勝者をアウトオブサンプルで参照し、それが下位半分に落ちる頻度を数えます。PBO が 0.5 に近いなら、
# その選択には情報がまったく乗っていなかったということです。
#
# 行列は `(観測数, 試行数)` です。しきい値ごとに1列、エクイティの標本ごとに1行になります。

# %%
equity_columns = {}
for name, payload in curves.items():
    series = payload["equity"]
    if series is None or len(series) < 12:
        continue
    equity_columns[name] = series.reset_index(drop=True)
matrix = pd.DataFrame(equity_columns).dropna()
returns = matrix.diff().dropna()
print(f"returns matrix: {returns.shape[0]} observations x {returns.shape[1]} trials")

pbo = quant.probability_of_backtest_overfitting(returns.to_numpy(), partitions=8)
print(f"\nPBO: {pbo.pbo:.3f}")
print(f"splits evaluated: {pbo.splits}")

# %% [markdown]
# ## 勝者のシャープレシオを割り引く
#
# 11のしきい値を試して見つけたシャープレシオは、1回目で見つけたシャープレシオと同じ統計量では
# ありません。`quant.deflated_sharpe` は探索の規模で割り引き、真のシャープレシオがベンチマークを
# 上回る確率として `probability` を報告します。0.95 を下回れば、その結果は同じ回数のコイン投げの
# 最良値と区別がつきません。試行のシャープレシオの分散が未知のとき、この関数はリターン自身の標本
# 分散で代用します。ゼロと仮定せず、保守的に振る舞う選択です。

# %%
best = max(curves.values(), key=lambda payload: payload["net"])
best_name = next(name for name, payload in curves.items() if payload is best)
best_returns = returns[best_name] / matrix[best_name].iloc[0]
deflated = quant.deflated_sharpe(best_returns.tolist(), trials=len(curves))
print(f"winning threshold: {best['threshold']:.2f} over {best['markets']} markets, net {best['net']:,.2f}")
print(f"observed Sharpe:   {deflated.sharpe:.3f}")
print(f"benchmark:         {deflated.benchmark:.3f}")
print(f"P(true Sharpe > benchmark): {deflated.probability:.3f}")
print(f"trials declared:   {deflated.trials} ({deflated.trials_source}), "
      f"{deflated.observations} observations")
print(f"skew {deflated.skew:+.2f}, excess kurtosis {deflated.kurtosis:.2f}")

# %%
track = quant.minimum_track_record_length(best_returns.tolist(), benchmark=0.0)
print(f"minimum track record length for significance: {track:.0f} observations")
print(f"we have: {len(best_returns)}")

# %% [markdown]
# シャープレシオより先に、歪度と尖度を読んでください。決着まで持ち切るブックのエクイティカーブは、
# 平らなまま進んで最後に跳ねます。つまりリターンの分布は、ノイズに囲まれた1つの大きな外れ値です。
# シャープレシオは正規分布にずっと近いものを前提にしていますし、割り引きの補正も同じモーメントに
# 頼っています。
#
# ですから誠実な読み方は「エッジは偽物だ」ではありません。ジャンプ過程の31標本に対してシャープレシオ
# という要約が向いていない、ということです。そしてデフレーテッド版は、心地よい数値を報告する代わりに
# そう告げています。この形のペイオフには、レシピ 05/03 の価格帯ごとの持ち切りリターンのほうが良い
# 推定量です。ここで用意した仕掛けが本領を発揮するのは、カーブがカーブらしく見えるだけ取引の頻度が
# 上がってからです。

# %% [markdown]
# ## 2つを合わせて読む
#
# どちらの数値も、単独では判定になりません。PBO が低く、割り引いてもシャープレシオが生き残るなら、
# 先へ進める価値のある発見です。PBO が 0.5 近くなら、勝者の表向きの数値が何であれ、順位付けはノイズ
# だったということです。そして必要なトラックレコード長が標本より長いなら、誠実な答えは「まだ判定
# できない」であり、これは正当な研究の結論であり、小さなパネルにふさわしい答えでもあります。

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
frame = pd.DataFrame([{k: v[k] for k in ("threshold", "net")} for v in curves.values()])
axes[0].bar(frame.threshold.astype(str), frame.net, color="#2c7fb8")
axes[0].axhline(0.0, color="black", lw=0.8)
axes[0].set_title("Net result by entry threshold")
axes[0].set_xlabel("minimum YES ask")
axes[0].set_ylabel("net, currency units")
axes[0].tick_params(axis="x", labelrotation=45)
for name in matrix:
    axes[1].plot(matrix.index, matrix[name] - matrix[name].iloc[0], lw=1, alpha=0.7)
axes[1].set_title(f"Equity paths, {len(matrix.columns)} trials (PBO {pbo.pbo:.2f})")
axes[1].set_xlabel("equity sample")
axes[1].set_ylabel("change from start")
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - 決済を制御するのは観測可能性であって、出来事の発生ではない。セッションの内側で止まったリプレイは
#   ポジションを未決済のまま残し、1件ずつ名指しする。受け取れなかった利益を計上したりはしない。
# - 決済まで進んだ実行では `market_exit_pnl` と `settlement_pnl` の両方が残るので、最後まで持ち切る
#   ことに依存する部分が、仮定ではなく明示的な数値になる。
# - しきい値のスイープは探索であり、探索は最良の結果を膨らませる。PBO は順位付けがアウトオブサンプル
#   で保たれたかを問い、デフレーテッド・シャープは勝者の数値が探索の規模に耐えるかを問う。
# - パネルが小さいときに提示すべき数字が、最小トラックレコード長である。「まだ分からない」を、あと
#   どれだけデータが要るかという具体的な量に変えてくれる。
# - ここで働いた h5i-db の機能。試行ごとに1つのフォークを与えることで11のしきい値が互いを汚さず、
#   `bt_equity` が試行ごとに比較できるカーブを与え、`backtest.study` が試行ごとに学習フェーズと
#   ホールドアウトフェーズを走らせ、観測可能性のゲートが分析者の記憶に頼らずエンジンの中にある。

# %%
db.close()
