# %% [markdown]
# # ティックから実現ボラティリティへ: シグネチャプロット、ジャンプ、オーバーナイトリスク
#
# 実現分散――日中リターンの二乗和――は標準的なノンパラメトリックのボラ推定量で、うまく計算
# できるかどうかはほとんど*サンプリング*の問題です。速く取りすぎればビッド・アスクの跳ね返りが
# 推定値を膨らませ、遅すぎれば情報を捨てることになります。ティックが h5i-db に時刻順で保存されて
# いれば、同じデータを別の頻度でバケットし直すのは `time_bucket` のクエリ1つです。だから古典的な
# 診断（シグネチャプロット、bipower variation、オーバーナイトの分解）がどれも数行で書けます。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_rv"), create=True)

# %% [markdown]
# ## 1. ジャンプが分かっているティックデータ
#
# 1銘柄について、密度の高い合成約定を5セッションぶん（1日10万件）用意します。生成器は約定を
# ミッドの両側に出すので、本物のビッド・アスクの跳ね返りが入ります。シグネチャプロットが
# あぶり出すマイクロストラクチャノイズそのものです。あわせて4日目のセッション途中に、
# **一度きりの +1.5% のジャンプ**（以降の価格すべての水準シフト）を仕込みます。ジャンプ検出の
# 節に、見つけるべき本物を用意しておくためです。

# %%
trades = cu.make_trades(symbols=["SPX"], days=5, trades_per_day=100_000, seed=42)

jump_ts = pd.Timestamp("2026-06-04 17:00:00", tz="UTC")
tdf = trades.to_pandas()
tdf.loc[tdf["ts"] >= jump_ts, "price"] *= 1.015  # permanent level shift

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
    ]
)
db.create_table("trades", schema, time_column="ts", sort_key=["ts"])
db.append(
    "trades",
    pa.Table.from_pandas(tdf[["ts", "symbol", "price", "size"]], preserve_index=False).cast(schema),
    note="5 sessions, +1.5% jump injected 06-04 17:00",
)
print(f"{len(tdf):,} trades over {tdf.ts.dt.date.nunique()} sessions")

# %% [markdown]
# ## 2. 5分足からの日次 RV
#
# 定石はこうです。5分にバケットし、バケットごとに最終約定価格を取り、セッション内の対数
# リターンを二乗して合計する。`time_bucket('5m', ts)` と `last_value(price ORDER BY ts)` が
# 1回のストリーミングでバケット化を済ませ、日ごとの差分と合計は pandas で2行です。年率ボラは
# `sqrt(RV_day * 252)` になります。

# %%
bars5 = db.sql(
    """
    SELECT time_bucket('5m', ts) AS bar,
           last_value(price ORDER BY ts) AS close,
           count(*) AS n_trades
    FROM trades
    GROUP BY bar
    ORDER BY bar
    """
).to_pandas()
bars5["day"] = bars5["bar"].dt.date
bars5["r"] = np.log(bars5["close"]).groupby(bars5["day"]).diff()  # within-session only

rv_daily = bars5.groupby("day")["r"].apply(lambda r: (r**2).sum()).rename("rv")
pd.DataFrame({"rv": rv_daily, "ann_vol_%": np.sqrt(rv_daily * 252) * 100}).round(5)

# %% [markdown]
# ## 3. シグネチャプロット
#
# サンプリング間隔を1秒から30分まで変えて RV を計算し直します。どれも*同じ*SQL クエリで、
# バケット幅だけが違います。摩擦のない世界なら RV は頻度に対して平坦なはずです。ところが
# ビッド・アスクの跳ね返りがあると、観測されるリターンは「真のリターン＋iid ノイズ」になり、
# ノイズの分散は*観測1件ごとに*払うことになります。だから間隔を詰めるほど RV は膨れ上がります。
# カーブが平らになる肘の位置が、信用できる最高の頻度です。流動性の高い銘柄では数分あたりに
# 座り、「5分 RV」が業界の既定値になっているのはそのためです。

# %%
INTERVALS_S = [1, 5, 15, 30, 60, 120, 300, 600, 900, 1800]

sig_rows = []
for sec in INTERVALS_S:
    b = db.sql(
        f"""
        SELECT time_bucket('{sec}sec', ts) AS bar,
               last_value(price ORDER BY ts) AS close
        FROM trades
        GROUP BY bar
        ORDER BY bar
        """
    ).to_pandas()
    b["day"] = b["bar"].dt.date
    r = np.log(b["close"]).groupby(b["day"]).diff()
    rv = (r**2).groupby(b["day"]).sum()
    sig_rows.append({"interval_s": sec, "ann_vol": float(np.sqrt(rv.mean() * 252))})
sig = pd.DataFrame(sig_rows)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(sig["interval_s"], sig["ann_vol"] * 100, "o-", lw=1.4)
ax.set_xscale("log")
ax.set_xticks(INTERVALS_S)
ax.set_xticklabels(["1s", "5s", "15s", "30s", "1m", "2m", "5m", "10m", "15m", "30m"])
ax.set_title("Volatility signature plot: RV vs sampling interval (5-day mean)")
ax.set_xlabel("sampling interval")
ax.set_ylabel("annualized RV vol (%)")
fig.tight_layout()

# %% [markdown]
# 教科書どおりの形です。1秒での推定値は安定値のおよそ2倍あります。その頻度では、観測される
# 「リターン」のかなりの部分がボラティリティではなくスプレッドだということです。ここでは1分
# あたりでカーブが平らになります。ノイズ源の多い実際のティックデータでは肘はもう少し外側に
# 座るのが普通で、だからこその5分という慣習です。
#
# ## 4. ジャンプ: bipower variation と RV
#
# RV が収束するのは*全体の*二次変分、つまり連続部分の分散**プラス**ジャンプの二乗です。
# Barndorff-Nielsen と Shephard の bipower variation
# $BV = \frac{\pi}{2}\sum |r_i||r_{i-1}|$ はジャンプに頑健です（1つの巨大なリターンが二乗
# される代わりに、有限の隣接値と掛けられるため）。したがって `max(RV - BV, 0)` がジャンプの
# 寄与を切り出します。1.5% のジャンプを仕込んだ4日目が光るはずで、他の日は RV ≈ BV に
# なるはずです。

# %%
def bipower(r: pd.Series) -> float:
    a = r.abs().to_numpy()
    return float(np.pi / 2 * np.nansum(a[1:] * a[:-1]))


bv_daily = bars5.groupby("day")["r"].apply(bipower).rename("bv")
jumps = pd.DataFrame({"rv": rv_daily, "bv": bv_daily})
jumps["jump_var"] = (jumps["rv"] - jumps["bv"]).clip(lower=0)
jumps["jump_share_%"] = 100 * jumps["jump_var"] / jumps["rv"]
jumps.round(5)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(jumps))
ax.bar(x - 0.2, jumps["rv"] * 1e4, width=0.4, label="RV")
ax.bar(x + 0.2, jumps["bv"] * 1e4, width=0.4, label="bipower variation")
ax.set_xticks(x)
ax.set_xticklabels([str(d) for d in jumps.index], rotation=20)
ax.set_title("RV vs bipower variation by day (jump injected on 06-04)")
ax.set_xlabel("session")
ax.set_ylabel("daily variance (bps²  ×10⁴)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ジャンプを仕込んだ日は RV が BV をはっきり上回り、その差は仕込んだ
# `ln(1.015)² ≈ 2.2e-4` に近い値です。ふつうの日はノイズの範囲で一致します。
#
# ## 5. 実際の SPY データで、オーバーナイトと日中を分ける
#
# 終値から終値までの分散は、オーバーナイトのギャップ（`open/前日終値`）と日中の部分
# （`close/open`）に分かれます。キャッシュ済みの30分足 SPY を使い、SQL 1クエリでセッションの
# 始値と終値を取ります。`time_bucket('1d', ts, 'America/New_York')` を使うので、バケットは
# UTC の日ではなく取引所の日で切れます。

# %%
spy = cu.fetch_intraday(["SPY"], period="30d", interval="30m")
sschema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("spy_30m", sschema, time_column="ts", sort_key=["ts"])
db.append("spy_30m", spy.select(["ts", "symbol", "open", "close"]).cast(sschema))

days = db.sql(
    """
    SELECT time_bucket('1d', ts, 'America/New_York') AS day,
           first_value(open  ORDER BY ts) AS day_open,
           last_value(close ORDER BY ts)  AS day_close
    FROM spy_30m
    GROUP BY day
    ORDER BY day
    """
).to_pandas()

r_on = np.log(days["day_open"] / days["day_close"].shift(1)).dropna()
r_id = np.log(days["day_close"] / days["day_open"]).iloc[1:]
var_on, var_id = r_on.var(), r_id.var()
print(
    f"{len(days)} sessions of SPY\n"
    f"overnight var share: {var_on / (var_on + var_id):.0%}   "
    f"intraday var share: {var_id / (var_on + var_id):.0%}\n"
    f"ann vol - overnight: {np.sqrt(var_on * 252):.1%}, intraday: {np.sqrt(var_id * 252):.1%}"
)

# %% [markdown]
# SPY の全分散のうち、無視できない部分が市場の閉まっているあいだに積み上がります。日中 RV が
# 決して見ないリスクで、終値ベースで評価するもの（VaR、オプション）にとっては効いてきます。
# セッションが約30しかないので、この分割は推定値というより例示です。本番なら何年ぶんもの足で
# 走らせるところで、それは同じクエリを大きなテーブルにかけるだけです。
#
# ## まとめ
#
# - ティックのテーブル1つと、幅を10通り変えた `time_bucket` で、シグネチャプロットが完成します。
#   バケットし直しはクエリであって、データパイプラインではありません。（注意: 1分未満の幅は
#   単位を綴ります。`'30s'` ではなく `'30sec'` です。）
# - マイクロストラクチャノイズは合成データでも見えます。ビッド・アスクの跳ね返りが1秒 RV を
#   安定値のおよそ2倍にしました。5分サンプリングが慣習的に安全な肘です。
# - RV から bipower variation を引くと、仕込んだジャンプの日をきれいに検出し、その大きさも
#   復元できました。連続部分のボラ予測で欲しい、ジャンプに頑健な分母が BV です。
# - オーバーナイトの分散は本物のリスクで（SPY 全体のうち目に見える割合を占めます）、
#   日中 RV は構造上それを除外します。RV を終値ベースのボラと比べる前に、分解してください。

# %%
db.close()
