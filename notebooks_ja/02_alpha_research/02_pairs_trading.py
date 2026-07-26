# %% [markdown]
# # ペアトレード: バージョンで固定したデータの背骨の上で
#
# 実際の価格に対する共和分ペア戦略です。Engle-Granger 検定で候補ペアを走査し、ローリングの
# ヘッジ比率を作り、スプレッドの z スコアを h5i-db のウィンドウ関数で*SQL のまま*計算し、
# シグナルを1期ずらしてコスト込みでバックテストします。h5i-db ならではのひねりはこうです。
# 価格を2回のコミットで読み込むので、最後にパイプライン全体を `h5i('prices', v_early)`――
# 過去の研究が見ていたはずのテーブルそのもの――に対して走らせ直せます。結果が固定される先は、
# ベンダーファイルの今日の中身ではなく、データのバージョンだと示せるわけです。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_pairs"), create=True)

# %% [markdown]
# ## 1. 価格を2回のコミットで読み込む
#
# モメンタムのレシピと同じ30銘柄の日次キャッシュです。あえて2025-01-01 で読み込みを分けます。
# 1回目の `append` は、仮に2024年の研究が使ったであろうテーブル。2回目が現在まで持ってきます。
# `append` はどちらも1回のアトミックなコミットで、`versions()` が両方を永久に保持します。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])

prices = (
    daily.select(["ts", "symbol", "adj_close"])
    .sort_by([("ts", "ascending"), ("symbol", "ascending")])
    .cast(schema)
)
cutoff = pd.Timestamp("2025-01-01", tz="UTC")
mask = pa.compute.less(prices["ts"], pa.scalar(cutoff, type=pa.timestamp("us", tz="UTC")))

c1 = db.append("prices", prices.filter(mask), note="history through 2024")
v_early = c1["sequence"]
c2 = db.append("prices", prices.filter(pa.compute.invert(mask)), note="2025 onwards")
print(f"v{v_early}: {c1['rows_total']:,} rows   v{c2['sequence']} (head): {c2['rows_total']:,} rows")

# %% [markdown]
# ## 2. 共和分の走査
#
# ユニバースから経済的に筋の通った候補を3つ選びます。KO/PEP、GS/JPM、CVX/XOM です。素直な
# SQL スキャンで対数調整終値を取り出し、それぞれに Engle-Granger 検定（`statsmodels.coint`）を
# かけます。相関しているからといって共和分しているとは限りません。p 値が検定しているのは
# *スプレッド*が定常かどうかで、平均回帰が本当に必要としているのはそちらです。

# %%
from statsmodels.tsa.stattools import coint

CANDIDATES = [("KO", "PEP"), ("GS", "JPM"), ("CVX", "XOM")]

logpx = (
    db.sql(
        """
        SELECT ts, symbol, ln(adj_close) AS log_px
        FROM prices
        WHERE symbol IN ('KO','PEP','GS','JPM','CVX','XOM')
        ORDER BY ts, symbol
        """
    )
    .to_pandas()
    .pivot(index="ts", columns="symbol", values="log_px")
)

scan = pd.DataFrame(
    [
        {
            "pair": f"{a}/{b}",
            "corr": logpx[a].diff().corr(logpx[b].diff()),
            "coint_p": coint(logpx[a], logpx[b])[1],
        }
        for a, b in CANDIDATES
    ]
).set_index("pair")
scan.round(4)

# %% [markdown]
# このサンプルで5%の水準をくぐるのは CVX/XOM だけです。教科書的なペアである KO/PEP は
# 落ちます（リターン相関は高いのですが、スプレッドがトレンドします）。走査の結果としては
# よくある話で、腹に入れておく価値があります。「見るからにペアらしい」組み合わせの多くは
# 定常性の検定に落ちるのです。ここでは CVX/XOM を売買します。
#
# ## 3. ローリングのヘッジ比率とスプレッド
#
# ヘッジ比率は、log CVX に対する log XOM の252日ローリング OLS ベータです（ローリングの
# 共分散/分散、同じものをベクトル化しただけです）。スプレッド `log(XOM) - beta*log(CVX)` と
# そのベータを、それ自体の h5i テーブルに入れます。こうするとシグナルの工程を SQL で書け、
# スプレッド系列自体も、それを生んだ価格と並んでバージョン管理されます。

# %%
lx, ly = logpx["CVX"], logpx["XOM"]
beta = ly.rolling(252).cov(lx) / lx.rolling(252).var()
spread = ly - beta * lx

spread_df = (
    pd.DataFrame({"ts": spread.index, "spread": spread.values, "beta": beta.values})
    .dropna()
    .reset_index(drop=True)
)
spread_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("spread", pa.float64()),
        pa.field("beta", pa.float64()),
    ]
)
db.create_table("spread", spread_schema, time_column="ts", sort_key=["ts"])
db.append(
    "spread",
    pa.Table.from_pandas(spread_df, preserve_index=False).cast(spread_schema),
    note="XOM vs CVX, rolling 252d OLS hedge",
)
len(spread_df)

# %% [markdown]
# ## 4. z スコアを SQL で
#
# 売買シグナルは、60日窓に対するスプレッドの z スコアです。平均には h5i-db の糖衣構文
# `rolling_avg(x, ts, n)` を、ばらつきには標準的な
# `stddev(...) OVER (... ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)` を使います。1クエリで
# 整列済みストレージの上を流れ、pandas の `rolling()` は状態を持つバックテスト本体まで出番が
# ありません。

# %%
zs = db.sql(
    """
    SELECT ts, spread, beta,
           (spread - rolling_avg(spread, ts, 60))
             / stddev(spread) OVER (ORDER BY ts ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
             AS z
    FROM spread
    ORDER BY ts
    """
).to_pandas()
zs = zs.set_index("ts")
zs.tail(3).round(4)

# %% [markdown]
# ## 5. バックテスト: |z| > 2 で入り、ゼロ交差で出る
#
# エントリーとイグジットの規則は状態を持つので、小さな明示的ループを回します（2千行なので
# 一瞬です）。守るべき点は3つあります。
#
# - ポジションは**前日の z** で決めます（`shift(1)`）。先読みはしません。
# - 日次損益には**1期前の**ヘッジ比率を使います。`pos[t-1] * (dlog XOM - beta[t-1] * dlog CVX)`
#   で、実際に持っていた帳簿のリターンです。ベータが黙って勝手にリバランスしたスプレッドの
#   リターンではありません。
# - コストは想定元本に対して片脚10bps。ペアの往復は約 `2 * (1 + |beta|) * 10` bps かかります。

# %%
z_lag = zs["z"].shift(1)
pos = np.zeros(len(zs))
p = 0.0
for i, z in enumerate(z_lag.fillna(0.0).to_numpy()):
    if p == 0.0:
        if z > 2:
            p = -1.0  # spread rich: short XOM, long beta*CVX
        elif z < -2:
            p = 1.0
    elif (p < 0 and z <= 0) or (p > 0 and z >= 0):
        p = 0.0
    pos[i] = p
pos = pd.Series(pos, index=zs.index, name="pos")

dlx = lx.diff().reindex(zs.index)
dly = ly.diff().reindex(zs.index)
beta_lag = zs["beta"].shift(1)
pair_ret = dly - beta_lag * dlx           # per unit of XOM leg
pnl_gross = pos.shift(1) * pair_ret
traded = pos.diff().abs().fillna(0.0) * (1 + beta_lag.abs())
pnl_net = (pnl_gross - 10 / 1e4 * traded).dropna()

equity = pnl_net.cumsum()
sharpe = pnl_net.mean() / pnl_net.std() * np.sqrt(252)
max_dd = (equity - equity.cummax()).min()
n_trades = int((pos.diff().abs() > 0).sum())
print(
    f"round turns: {n_trades // 2}   time in market: {(pos != 0).mean():.0%}\n"
    f"Sharpe (net): {sharpe:.2f}   total P&L: {equity.iloc[-1]:+.1%} (log, unit notional)\n"
    f"max drawdown: {max_dd:.1%}"
)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(zs.index, zs["z"], lw=0.7, color="tab:blue")
axes[0].axhline(2, color="tab:red", lw=0.7, ls="--")
axes[0].axhline(-2, color="tab:red", lw=0.7, ls="--")
axes[0].axhline(0, color="gray", lw=0.6)
entries = pos[(pos.diff() != 0) & (pos != 0)].index
axes[0].plot(entries, zs.loc[entries, "z"], "v", color="tab:red", ms=5, label="entry")
axes[0].set_ylabel("spread z-score")
axes[0].set_title("CVX/XOM spread z-score (60d) and net P&L")
axes[0].legend(loc="upper left")
axes[1].plot(equity.index, equity, lw=1.2, color="tab:green")
axes[1].set_ylabel("cum. P&L (log points)")
axes[1].set_xlabel("date")
fig.tight_layout()

# %% [markdown]
# Sharpe はわずかにプラス、ただしドローダウンは深い。広がっていくスプレッドを平均に戻るまで
# 持ち続ける、それがまさにペアの帳簿が痛む形です。ペアは1つ、p 値はぎりぎり、ストップロスも
# なし。これは手法のデモであって戦略ではありません。実際の帳簿なら多数のペアに分散し、乖離
# リスクに上限をかけるでしょう。
#
# ## 6. シグナルを保存し、実行を固定する
#
# ポジションと z スコアを `signals` テーブルに入れ、スナップショット1つで `prices`、`spread`、
# `signals` をまとめて実行 ID の下に固定します。

# %%
sig_df = zs.reset_index()[["ts", "z"]].assign(pos=pos.values).dropna()
sig_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("z", pa.float64()),
        pa.field("pos", pa.float64()),
    ]
)
db.create_table("signals", sig_schema, time_column="ts", sort_key=["ts"])
db.append("signals", pa.Table.from_pandas(sig_df, preserve_index=False).cast(sig_schema))
db.snapshot("pairs-run-001", tables=["prices", "spread", "signals"], note="CVX/XOM |z|>2")
db.sql("SELECT count(*) AS rows, min(ts) AS first, max(ts) AS last FROM h5i('signals','pairs-run-001')").to_pandas()

# %% [markdown]
# ## 7. 過去のデータバージョンでパイプラインを走らせ直す
#
# 研究全体を、*`prices` のどのバージョンを読むか*でパラメータ化します。SQL のタイムトラベルの
# おかげで、変更は文字列1つです。`FROM prices` が `FROM h5i('prices', v_early)` になるだけ。
# 2025年より前のバージョンで走らせれば、2024年の研究が見つけたはずのものが再現します。同じ
# バージョンで2回走らせればビット単位で同一です。ベンダーが足元で歴史を訂正してきたときに、
# 効いてくる性質です。

# %%
def run_pipeline(prices_rel: str) -> dict:
    """Coint p-value + net Sharpe for CVX/XOM from any prices relation."""
    px = (
        db.sql(
            f"""
            SELECT ts, symbol, ln(adj_close) AS log_px
            FROM {prices_rel}
            WHERE symbol IN ('CVX','XOM') ORDER BY ts, symbol
            """
        )
        .to_pandas()
        .pivot(index="ts", columns="symbol", values="log_px")
    )
    lx_, ly_ = px["CVX"], px["XOM"]
    b = ly_.rolling(252).cov(lx_) / lx_.rolling(252).var()
    s = ly_ - b * lx_
    z = ((s - s.rolling(60).mean()) / s.rolling(60).std()).shift(1)
    q = np.zeros(len(z))
    p_ = 0.0
    for i, v in enumerate(z.fillna(0.0).to_numpy()):
        if p_ == 0.0:
            if v > 2:
                p_ = -1.0
            elif v < -2:
                p_ = 1.0
        elif (p_ < 0 and v <= 0) or (p_ > 0 and v >= 0):
            p_ = 0.0
        q[i] = p_
    q = pd.Series(q, index=z.index)
    ret = q.shift(1) * (ly_.diff() - b.shift(1) * lx_.diff())
    net_ = (ret - 10 / 1e4 * q.diff().abs().fillna(0) * (1 + b.shift(1).abs())).dropna()
    return {
        "rows": len(px),
        "last_day": px.index[-1].date().isoformat(),
        "coint_p": round(coint(lx_, ly_)[1], 4),
        "sharpe_net": round(net_.mean() / net_.std() * np.sqrt(252), 3),
    }


runs = pd.DataFrame(
    {
        "head (today)": run_pipeline("prices"),
        f"h5i('prices', {v_early})": run_pipeline(f"h5i('prices', {v_early})"),
        f"h5i('prices', {v_early}) again": run_pipeline(f"h5i('prices', {v_early})"),
    }
).T
assert runs.iloc[1].equals(runs.iloc[2]), "same version must reproduce exactly"
runs

# %% [markdown]
# 2025年より前の実行は違うサンプルを見ます（p 値も Sharpe も変わります）。そして固定した
# バージョンで走らせ直せば、完全に再現します。「この数字はどのデータから出たのか」に、
# バージョンの整数という正確な答えが返ります。
#
# ## まとめ
#
# - 実データに対する Engle-Granger は謙虚にさせてくれます。筋の通った候補3つのうち、5%で
#   共和分していたのは CVX/XOM だけでした。相関は共和分ではありません。
# - z スコアは完全に SQL の中で走りました。糖衣構文の `rolling_avg` と、保存済み `spread`
#   テーブルへの標準的な `stddev ... OVER ROWS` のウィンドウです。
# - バックテストの誠実さは、1期前の z、損益に使う1期前のヘッジ比率、脚ごとのコスト。結果
#   （そこそこの Sharpe、見苦しいドローダウン）は、ペア1本の帳簿がまさにこう見えるという
#   ことです。
# - 意味のある単位でコミットを打っておくと、あとで効きます。`h5i('prices', v)` はどの研究も
#   当時のバイトそのものに対して走らせ直せますし、`db.snapshot(...)` は3つのテーブルの状態を
#   まとめて1つの名前で呼べるようにします。

# %%
db.close()
