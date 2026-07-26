# %% [markdown]
# # VWAP、TWAP、そして執行ベンチマーク
#
# 執行デスクの生死を分けるのはベンチマークの算術です。区間 VWAP、TWAP、到着時価格、
# ベーシスポイントのスリッページ。その中身はどれも、約定テープに対するバケット単位の加重
# 集約と、気配ストリームへの時点参照1回です。h5i-db の `time_bucket`、`vwap()`（集約関数
# *であり*ウィンドウ関数でもあります）、`asof_join` は、まさにこの形のクエリのために
# あります。
#
# ここでは日次と30分区間の VWAP を計算し、出来高の偏りで両者が離れる場面を TWAP と対比
# させます。そのうえで、シミュレートした親注文（AAPL 5万株を1時間かけて執行）を、区間 VWAP と
# 到着時ミッドに対してベンチマークします。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import matplotlib.pyplot as plt

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_vwap"), create=True)

# %% [markdown]
# ## 1. マイクロストラクチャとして整合したテープ
#
# 約定を気配に対してベンチマークすることに意味があるのは、両者が*同じ*市場を描いていると
# きだけです。そこで、印字されるテープを気配ストリームから導きます。各約定は、その時点の
# 気配のオファーを取るかビッドを叩き（10%はミッドで執行）、取引所のわずかな遅延を挟みます。
# （クックブックの約定生成器と気配生成器は始値しか共有しません。バーのレシピには十分ですが、
# 気配との相対を測るベンチマークには使えません。）

# %%
quotes = cu.make_quotes(symbols=["AAPL", "MSFT", "NVDA"], days=3, seed=11)

qp = quotes.to_pandas()
rng = np.random.default_rng(17)
is_trade = rng.random(len(qp)) < 0.35
tr = qp.loc[is_trade, ["ts", "symbol", "bid", "ask"]].reset_index(drop=True)
n = len(tr)
buy = rng.random(n) < 0.5
at_mid = rng.random(n) < 0.10
px = np.where(buy, tr["ask"], tr["bid"])
px = np.where(at_mid, (tr["bid"] + tr["ask"]) / 2, px)
tr["price"] = np.round(px, 4)
tr["size"] = np.maximum(1, rng.lognormal(4.0, 1.2, n) // 100 * 100).astype("int64")
tr["side"] = np.where(buy, "B", "S")
tr["ts"] = (tr["ts"] + pd.to_timedelta(rng.uniform(0.2, 5.0, n), unit="ms")).dt.floor("us")
tr = tr.sort_values(["ts", "symbol"])[["ts", "symbol", "price", "size", "side"]]

ts_field = pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False)
trade_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64()), pa.field("side", pa.string())]
)
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", pa.Table.from_pandas(tr, preserve_index=False).cast(trade_schema))

quote_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("bid", pa.float64()),
     pa.field("ask", pa.float64()), pa.field("bid_size", pa.int64()),
     pa.field("ask_size", pa.int64())]
)
db.create_table("quotes", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes.cast(quote_schema))

print(f"{len(tr):,} trades derived from {len(qp):,} quotes")

# %% [markdown]
# ## 2. 日次 VWAP と区間 VWAP
#
# `vwap(price, size)` はネイティブの集約関数なので、日次 VWAP は1行です。
# `time_bucket('1d', ts, 'America/New_York')` でニューヨークの深夜にセッションを切れば、
# 数字は夏時間に対して安全になります。同じ文の幅を `'30m'` にすれば、執行スケジューラが
# 狙う区間 VWAP が出ます。

# %%
day_vwap = db.sql(
    """
    SELECT time_bucket('1d', ts, 'America/New_York') AS session, symbol,
           vwap(price, size) AS day_vwap,
           sum(size)         AS volume
    FROM trades GROUP BY 1, 2 ORDER BY 1, 2
    """
).to_pandas()
day_vwap

# %%
ivwap = db.sql(
    """
    SELECT time_bucket('30m', ts) AS interval_start, symbol,
           vwap(price, size) AS interval_vwap,
           sum(size)         AS volume
    FROM trades GROUP BY 1, 2 ORDER BY 1, 2
    """
).to_pandas()
ivwap.head(6)

# %% [markdown]
# ## 3. TWAP と VWAP
#
# TWAP はどの分にも同じ重みを置き、VWAP は出来高で分を重み付けします。U字型の出来高
# プロファイルでは VWAP の重みが寄り付きと引けに集まるので、両端付近の価格が日中の
# 真ん中と違えば、2つのベンチマークは離れます。ここでは1分足の終値から TWAP を組み立て
# （バーを CTE にして、終値は `last_value(... ORDER BY ts)`）、セッションごとの乖離を
# ベーシスポイントで測ります。

# %%
twap = db.sql(
    """
    WITH bars_1m AS (
        SELECT time_bucket('1m', ts) AS bar, symbol,
               last_value(price ORDER BY ts) AS close
        FROM trades GROUP BY 1, 2
    )
    SELECT time_bucket('1d', bar, 'America/New_York') AS session, symbol,
           avg(close) AS twap
    FROM bars_1m GROUP BY 1, 2 ORDER BY 1, 2
    """
).to_pandas()

bench = day_vwap.merge(twap, on=["session", "symbol"])
bench["vwap_minus_twap_bps"] = (bench["day_vwap"] / bench["twap"] - 1) * 1e4
bench[["session", "symbol", "day_vwap", "twap", "vwap_minus_twap_bps"]]

# %% [markdown]
# どちらに振れるかは、寄り付きと引けの厚い出来高がその日の平均価格の上で印字されたか下で
# 印字されたか、それだけで決まります。数 bps の差です。累積で見ると仕組みが目に見えます。
# `vwap()` は*ウィンドウ*関数としても働くので、累積 VWAP に手作りの `sum(pv)/sum(v)` は
# 要りません。

# %%
run = db.sql(
    """
    SELECT ts, price,
           vwap(price, size) OVER (ORDER BY ts) AS running_vwap
    FROM trades
    WHERE symbol = 'AAPL'
      AND ts >= '2026-06-02T00:00:00Z' AND ts < '2026-06-03T00:00:00Z'
    ORDER BY ts
    """
).to_pandas()
closes_1m = db.sql(
    """
    SELECT time_bucket('1m', ts) AS bar, last_value(price ORDER BY ts) AS close
    FROM trades
    WHERE symbol = 'AAPL'
      AND ts >= '2026-06-02T00:00:00Z' AND ts < '2026-06-03T00:00:00Z'
    GROUP BY 1 ORDER BY 1
    """
).to_pandas()
closes_1m["running_twap"] = closes_1m["close"].expanding().mean()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(run["ts"], run["price"], lw=0.3, color="0.7", label="trades")
ax.plot(run["ts"], run["running_vwap"], lw=1.4, label="running VWAP")
ax.plot(closes_1m["bar"], closes_1m["running_twap"], lw=1.4, ls="--", label="running TWAP")
ax.set_title("AAPL 2026-06-02: running VWAP vs running TWAP")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 4. 親注文をベンチマークする
#
# デスクは2026-06-02の14:00〜15:00 UTC（ニューヨーク10:00〜11:00）に AAPL を5万株買います。
# アグレッシブな（オファーを取る）フローに対しておおむね一定の参加率で執行する POV 型です。
# テープから約定を再現して `fills` テーブルに保存し、到着時価格は定石どおりに記録します。
# つまり注文受領時点のその場の気配ミッドで、1行の `asof_join` で取ってきます。

# %%
t0, t1 = pd.Timestamp("2026-06-02 14:00", tz="UTC"), pd.Timestamp("2026-06-02 15:00", tz="UTC")
win = tr[(tr["symbol"] == "AAPL") & (tr["ts"] >= t0) & (tr["ts"] < t1)]
liftable = win[win["side"] == "B"]  # prints where an aggressive buyer paid the offer

target = 50_000
pov = target / liftable["size"].sum()
fills = liftable[["ts", "symbol", "price"]].copy()
fills["size"] = np.maximum(1, (liftable["size"] * pov).round()).astype("int64")
fills.iloc[-1, fills.columns.get_loc("size")] += target - fills["size"].sum()

fill_schema = pa.schema(
    [ts_field, pa.field("symbol", pa.string()), pa.field("price", pa.float64()),
     pa.field("size", pa.int64())]
)
db.create_table("fills", fill_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("fills", pa.Table.from_pandas(fills, preserve_index=False).cast(fill_schema))
print(f"{len(fills):,} fills, {fills['size'].sum():,} shares, "
      f"~{pov:.0%} of aggressive buy volume")

# %% [markdown]
# `asof_join` は保存済みのテーブル名を取るので、まず注文受領時刻の前後の気配の窓を、それ
# 自体の小さなテーブルに切り出します（デスクはあとの TCA レビューのために、まさにこうした
# 窓を保存します）。そこに1行の親注文をジョインします。裏取りとして、この asof 参照が
# 気配テーブル全体に対する「14:00 以前で最新の気配」という時点クエリと一致することも
# 確かめます。

# %%
qwin = db.sql(
    """
    SELECT * FROM quotes
    WHERE symbol = 'AAPL'
      AND ts >= '2026-06-02T13:50:00Z' AND ts < '2026-06-02T14:10:00Z'
    ORDER BY ts
    """
).to_arrow()
db.create_table("quotes_arrival", quote_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_arrival", qwin.cast(quote_schema))

parent = pa.table(
    {"ts": pa.array([t0], type=pa.timestamp("us", tz="UTC")), "symbol": ["AAPL"]}
).cast(pa.schema([ts_field, pa.field("symbol", pa.string())]))
db.create_table("parent", parent.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("parent", parent)

arrival = db.sql(
    """
    SELECT ts, symbol, ts_right AS quote_ts, (bid + ask) / 2 AS arrival_mid
    FROM asof_join('parent', 'quotes_arrival', 'ts', 'ts', 'symbol')
    """
).to_pandas()

direct = db.sql(
    """
    SELECT (bid + ask) / 2 AS mid FROM quotes
    WHERE symbol = 'AAPL' AND ts <= '2026-06-02T14:00:00Z'
    ORDER BY ts DESC LIMIT 1
    """
).to_pandas()["mid"][0]
assert np.isclose(arrival["arrival_mid"][0], direct)
arrival

# %% [markdown]
# `asof_join` は 14:00:00 以前で最後の気配を拾いました（どれだけ古かったかは `quote_ts` に
# 出ています。右側の列名が衝突すると `_right` が付きます）。ここからが採点表です。約定 VWAP を
# 区間 VWAP と到着時価格に対して比べます。

# %%
fill_vwap = db.sql("SELECT vwap(price, size) AS v FROM fills").to_pandas()["v"][0]
mkt_vwap = db.sql(
    """
    SELECT vwap(price, size) AS v FROM trades
    WHERE symbol = 'AAPL'
      AND ts >= '2026-06-02T14:00:00Z' AND ts < '2026-06-02T15:00:00Z'
    """
).to_pandas()["v"][0]
arrival_mid = arrival["arrival_mid"][0]

print(f"arrival mid          : {arrival_mid:.4f}")
print(f"interval VWAP (14-15): {mkt_vwap:.4f}")
print(f"fill VWAP            : {fill_vwap:.4f}")
print(f"slippage vs interval VWAP: {(fill_vwap / mkt_vwap - 1) * 1e4:+.2f} bps")
print(f"slippage vs arrival mid  : {(fill_vwap / arrival_mid - 1) * 1e4:+.2f} bps")

# %%
fp = fills
mkt = db.sql(
    """
    SELECT ts, price, vwap(price, size) OVER (ORDER BY ts) AS running_vwap
    FROM trades
    WHERE symbol = 'AAPL'
      AND ts >= '2026-06-02T14:00:00Z' AND ts < '2026-06-02T15:00:00Z'
    ORDER BY ts
    """
).to_pandas()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mkt["ts"], mkt["price"], lw=0.3, color="0.7", label="market trades")
ax.plot(mkt["ts"], mkt["running_vwap"], lw=1.4, label="running interval VWAP")
ax.scatter(fp["ts"], fp["price"], s=6, color="tab:red", label="our fills", zorder=3)
ax.axhline(arrival_mid, ls=":", color="tab:green", label="arrival mid")
ax.axhline(fill_vwap, ls="--", color="tab:red", label="fill VWAP")
ax.set_title("50k AAPL buy, 14:00-15:00 UTC: fills vs benchmarks")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

# %% [markdown]
# オファーを払うコストは、区間 VWAP に対しておおよそハーフスプレッドぶんです。マーケッタブルな
# 注文を並べる POV スケジュールなら、まさにこう出るはずです。一方、到着時価格に対する
# スリッページには、その1時間の価格ドリフトも上乗せされます（インプリメンテーション
# ショートフォール）。
#
# ## 5. `vwap` と kdb 流の `wavg` は同じ統計量
#
# q から移ってきた人向けに。`w wavg x` は重みが先に来る書き方で、h5i-db は両方の綴りを
# 用意しています。

# %%
eq = db.sql(
    "SELECT vwap(price, size) AS vwap, wavg(size, price) AS wavg FROM fills"
).to_pandas()
assert np.isclose(eq["vwap"][0], eq["wavg"][0])
eq

# %% [markdown]
# ## まとめ
#
# - `vwap(price, size)` は集約関数としても（`time_bucket` と組んだ日次・区間 VWAP）、
#   ウィンドウ関数としても（累積 VWAP）働きます。手作りの `sum(pv)/sum(v)` は要りません。
#   kdb 流の綴りは `wavg(w, x)` です。
# - 1分足の終値から作る TWAP は CTE 1つぶんの距離にあります。VWAP と TWAP の乖離は
#   セッションごとの bps として落ちてきて、その正体はU字型の出来高です。
# - 到着時価格は、保存した気配の窓に対する1行の `asof_join` です。明示的な時点クエリとの
#   照合も安く済むので、お金が動くベンチマークでは習慣にする価値があります。
# - ベンチマークが意味を持つのは、テープが整合しているときだけです。気配との相対で出した
#   数字を信じる前に、約定を気配ストリームから導くか、少なくとも突き合わせておきましょう。

# %%
db.close()
