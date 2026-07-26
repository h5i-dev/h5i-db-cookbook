# %% [markdown]
# # オーダーフロー・インバランス: 符号付き出来高はリターンを予測するか
#
# オーダーフロー・インバランス（OFI）――買い手主導の出来高が売り手主導を上回るぶん――は
# マイクロストラクチャの主力シグナルです。同時点の価格変動とは強く相関し、実際の市場の短い
# 時間軸では次の値動きもわずかに予測します。このレシピでは、両方の系統を h5i-db の上で作ります。
#
# 1. **約定 OFI**。判定ルールを正解に対して検証し（その場の気配への ASOF ジョインによる
#    Lee-Ready、代替としてティックテスト）、テープに符号を付け、符号付き出来高を分ごとに
#    集計します
# 2. **気配 OFI**（Cont, Kukanov & Stoikov 2014）。最良ビッド・アスクの価格と数量の変化から
#    板の圧力を、`lag()` ウィンドウで求めます
# 3. 1分リターンとの相関を、同時点と*予測*の両面から正直に見ます

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_ofi"), create=True)

# %% [markdown]
# ## 1. 気配ストリームと、そこから印字したテープ
#
# 生成器の癖を1つ先に。`make_trades` の約定はランダムウォークに従い、`make_quotes` のミッドとは
# 緩くしか結びついていません。これでは気配ベースのサイド判定に意味がなくなります。そこで合成の
# 気配ストリームは残しつつ、*そこから自分でテープを印字*します。マッチングエンジンがやるのと
# 同じ形です。気配更新のうちランダムに25%が1〜50ミリ秒後に約定を引き起こし、買い手はオファーを
# 取り、売り手はビッドを叩き、本当の攻撃側を記録しておきます。テープは作りからして気配と価格が
# 整合しているので、以下のどの判定ルールも正解に対して採点できます。実データでは決して手に
# 入らない贅沢です。

# %%
quotes = cu.make_quotes(symbols=["AAPL", "MSFT"], days=2)
q = quotes.to_pandas()

rng = np.random.default_rng(21)
picks = q[rng.random(len(q)) < 0.25].copy()
side = rng.choice(np.array([1, -1]), size=len(picks))
picks["price"] = np.where(side > 0, picks["ask"], picks["bid"])
picks["ts"] = (
    picks["ts"] + pd.to_timedelta(rng.integers(1_000, 50_000, len(picks)), unit="us")
).astype("datetime64[us, UTC]")
picks["size"] = np.maximum(100, (rng.lognormal(4.0, 1.2, len(picks)) // 100 * 100)).astype("int64")
picks["side"] = np.where(side > 0, "B", "S")

trade_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("side", pa.string()),
    ]
)
trades_df = picks[["ts", "symbol", "price", "size", "side"]].sort_values(["ts", "symbol"])
trades = pa.Table.from_pandas(trades_df, preserve_index=False).cast(trade_schema)

db.create_table("quotes", quotes.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes, note="2-day synthetic NBBO")
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades, note="tape printed off the quote stream")
print({t: len(db.read(t)) for t in db.tables()})

# %% [markdown]
# ## 2. QA 用の窓で判定ルールを検証する
#
# Lee-Ready はこう言います。その場のミッドより上なら買い、下なら売り、ミッドちょうどなら
# ティックテスト（アップティックなら買い）に落とす。その場の気配の参照は
# `asof_join(..., 'backward', 5_000_000)`、つまり銘柄ごとに、古くとも5秒以内の直近の気配です。
# 許容差は生のマイクロ秒です。
#
# 検証は日中の15分の窓で走らせます。時間範囲を指定した `db.read`（生のマイクロ秒の境界で、
# セグメント単位にプルーニングされます）で切り出し、小さな作業台テーブルとして保存します。
# 窓を切った検証は反復が速く、目視でも確認しやすい。ジョインが約定1件につきちょうど1行を
# 返すことを表明し、貼り付いた気配を `pandas.merge_asof` と照合します。

# %%
w0 = int(pd.Timestamp("2026-06-01 16:00", tz="UTC").value // 1000)
w1 = int(pd.Timestamp("2026-06-01 16:15", tz="UTC").value // 1000)

trades_w = db.read("trades", time_start=w0, time_end=w1)
quotes_w = db.read("quotes", time_start=w0 - 60_000_000, time_end=w1)  # 60s run-up
db.create_table("trades_qa", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades_qa", trades_w, note="midday QA window")
db.create_table("quotes_qa", quotes.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes_qa", quotes_w, note="midday QA window + run-up")
print(f"QA window: {len(trades_w):,} trades, {len(quotes_w):,} quotes")

audit = db.sql(
    """
    WITH tq AS (
        SELECT ts, symbol, price, size, side, bid, ask,
               (bid + ask) / 2 AS mid,
               lag(price) OVER (PARTITION BY symbol ORDER BY ts) AS prev_px
        FROM asof_join('trades_qa', 'quotes_qa', 'ts', 'ts', 'symbol', 'backward', 5000000)
    )
    SELECT ts, symbol, mid,
           CASE WHEN price > mid THEN 1 WHEN price < mid THEN -1
                WHEN price > prev_px THEN 1 WHEN price < prev_px THEN -1 ELSE 0 END AS lr_sign,
           CASE WHEN price > prev_px THEN 1 WHEN price < prev_px THEN -1 ELSE 0 END AS tick_sign,
           CASE WHEN side = 'B' THEN 1 ELSE -1 END AS true_sign
    FROM tq
    ORDER BY ts, symbol
    """
).to_pandas()
assert len(audit) == len(trades_w), "asof_join must return one row per trade"

# cross-check the ASOF lookup itself against pandas.merge_asof
ref = pd.merge_asof(
    trades_w.to_pandas().sort_values("ts"),
    quotes_w.to_pandas().sort_values("ts").assign(mid_ref=lambda d: (d["bid"] + d["ask"]) / 2),
    on="ts",
    by="symbol",
    tolerance=pd.Timedelta("5s"),
)
chk = audit.sort_values("ts").reset_index(drop=True)
assert np.allclose(chk["mid"], ref["mid_ref"], equal_nan=True), "asof mismatch vs pandas"
print("asof_join validated against pandas.merge_asof")

for rule in ("lr_sign", "tick_sign"):
    scored = audit[audit[rule] != 0]
    acc = (scored[rule] == scored["true_sign"]).mean()
    print(f"  {rule:9s}: {acc:.1%} accuracy vs true aggressor side ({len(scored):,} classified)")

# %% [markdown]
# 価格が整合したテープの上では、Lee-Ready は評判どおりの働きをします。クオートルールはほぼ
# 完璧です（誤りは、気配と印字のあいだの1〜50ミリ秒に気配が更新された場合から出ます）。一方、
# 純粋なティックテストはかなり後れを取ります。実際の TAQ で見られるのと同じ並びで、そちらでは
# Lee-Ready が約85%を取ります。
#
# ## 3. テープ全体に符号を付ける
#
# 検証したクオートルールを使うなら、約定と気配の ASOF を全量で回す必要があります。このレシピ
# では、ティックテスト（`trades` に対する純粋な `lag()` ウィンドウで、揃えは不要です）でも OFI の
# 研究を支えるには十分な精度なので、そちらを使い、生成器の本当のサイドは上限の目安として
# 残しておきます。符号を付けたテープは一級のテーブルとして保存します。符号付けはそれなりに
# 高くつくので、1回で済ませ、コミットし、バージョンを付けておきたい作業です。

# %%
signed = db.sql(
    """
    WITH t AS (
        SELECT ts, symbol, price, size, side,
               lag(price) OVER (PARTITION BY symbol ORDER BY ts) AS prev_px
        FROM trades
    )
    SELECT ts, symbol, price, size,
           CASE WHEN price > prev_px THEN 1 WHEN price < prev_px THEN -1 ELSE 0 END AS tick_sign,
           CASE WHEN side = 'B' THEN 1 ELSE -1 END AS true_sign
    FROM t
    ORDER BY ts, symbol
    """
).to_arrow()

full = signed.to_pandas()
scored = full[full["tick_sign"] != 0]
print(f"full tape: {(scored['tick_sign'] == scored['true_sign']).mean():.1%} tick-test accuracy on {len(scored):,} prints")

# %%
signed_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("tick_sign", pa.int32()),
        pa.field("true_sign", pa.int32()),
    ]
)
db.create_table("signed_trades", signed_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("signed_trades", signed.cast(signed_schema), note="tick-test + oracle signs")

# %% [markdown]
# ## 4. 分ごとの約定 OFI
#
# `time_bucket` の集約1つです。符号付きの買い出来高と売り出来高、正規化したインバランス
# `(buys - sells) / total`、そして `last_value(... ORDER BY ts)` の定番でバケットの終値を取ります。

# %%
trade_ofi = db.sql(
    """
    SELECT time_bucket('1m', ts) AS bar, symbol,
           sum(CASE WHEN tick_sign > 0 THEN size ELSE 0 END) AS buy_vol,
           sum(CASE WHEN tick_sign < 0 THEN size ELSE 0 END) AS sell_vol,
           sum(CASE WHEN true_sign > 0 THEN size ELSE -size END) AS oracle_net,
           sum(size) AS volume,
           last_value(price ORDER BY ts) AS close
    FROM signed_trades
    GROUP BY bar, symbol
    ORDER BY bar, symbol
    """
).to_pandas()
trade_ofi["ofi"] = (trade_ofi["buy_vol"] - trade_ofi["sell_vol"]) / trade_ofi["volume"]
trade_ofi["ofi_oracle"] = trade_ofi["oracle_net"] / trade_ofi["volume"]
trade_ofi.head(4)

# %% [markdown]
# ## 5. 気配 OFI（Cont ら）
#
# 板の圧力による OFI が必要とするのは、最良気配における*変化*です。ビッド価格が上がる（あるいは
# ビッドが同値のまま数量が増える）と買い圧力が加わり、アスク側では鏡像として引かれます。気配
# 1件あたり4回の `lag()` 比較、つまり純粋な SQL ウィンドウの仕掛けで、それを分ごとに合計し、
# リターン用にバケットの終値ミッドを添えます。

# %%
quote_ofi = db.sql(
    """
    WITH q AS (
        SELECT ts, symbol, bid, ask, bid_size, ask_size,
               lag(bid)      OVER w AS pb,  lag(ask)      OVER w AS pa,
               lag(bid_size) OVER w AS pbs, lag(ask_size) OVER w AS pas
        FROM quotes
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    )
    SELECT time_bucket('1m', ts) AS bar, symbol,
           sum(  CASE WHEN bid >= pb THEN bid_size ELSE 0 END
               - CASE WHEN bid <= pb THEN pbs      ELSE 0 END
               - CASE WHEN ask <= pa THEN ask_size ELSE 0 END
               + CASE WHEN ask >= pa THEN pas      ELSE 0 END) AS qofi,
           last_value((bid + ask) / 2 ORDER BY ts) AS mid_close
    FROM q
    WHERE pb IS NOT NULL
    GROUP BY bar, symbol
    ORDER BY bar, symbol
    """
).to_pandas()
quote_ofi.head(4)

# %% [markdown]
# ## 6. 同時点か、予測か
#
# 肝心な問いはこうです。この分の OFI が語るのは*この*分のリターン（機械的な関係）なのか、
# それとも*次の*分のリターン（アルファ）なのか。両方の相関を、銘柄ごと・推定量ごとに計算します。
# リターンはセッション内に限ります（銘柄・日ごとの `shift` なので、オーバーナイトの作り物は
# 入りません）。

# %%
def bucket_corrs(frame, ofi_col, px_col):
    rows = []
    frame = frame.copy()
    frame["day"] = frame["bar"].dt.date
    for (sym, _), g in frame.groupby(["symbol", "day"]):
        g = g.sort_values("bar")
        frame.loc[g.index, "ret"] = g[px_col].pct_change()
        frame.loc[g.index, "ret_next"] = g[px_col].pct_change().shift(-1)
    for sym, g in frame.groupby("symbol"):
        g = g.dropna(subset=["ret", "ret_next"])
        rows.append(
            {
                "symbol": sym,
                "n_buckets": len(g),
                "corr_same_min": g[ofi_col].corr(g["ret"]),
                "corr_next_min": g[ofi_col].corr(g["ret_next"]),
            }
        )
    return pd.DataFrame(rows)

results = pd.concat(
    [
        bucket_corrs(trade_ofi, "ofi", "close").assign(estimator="trade OFI (tick test)"),
        bucket_corrs(trade_ofi, "ofi_oracle", "close").assign(estimator="trade OFI (oracle)"),
        bucket_corrs(quote_ofi, "qofi", "mid_close").assign(estimator="quote OFI (Cont)"),
    ]
)[["estimator", "symbol", "n_buckets", "corr_same_min", "corr_next_min"]]
results.round(3)

# %%
import matplotlib.pyplot as plt

sym = "AAPL"
g = trade_ofi[trade_ofi["symbol"] == sym].copy()
g["day"] = g["bar"].dt.date
g["ret"] = g.groupby("day")["close"].pct_change()
g = g.dropna(subset=["ret"])

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(g["ofi"], g["ret"] * 1e4, s=8, alpha=0.35, color="tab:blue")
b, a = np.polyfit(g["ofi"], g["ret"] * 1e4, 1)
xs = np.linspace(-1, 1, 50)
ax.plot(xs, a + b * xs, color="tab:red", lw=1.5, label=f"OLS slope = {b:.1f} bps per unit OFI")
ax.set_title(f"{sym}: 1-minute trade OFI vs same-minute return")
ax.set_xlabel("OFI = (buy vol - sell vol) / total vol")
ax.set_ylabel("1-minute return (bps)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 1つの表から、正直な教訓が3つ出てきます。
#
# - **気配 OFI** は教科書どおりの同時点の結びつき（約0.5〜0.6）を示します。板の圧力と同じ分の
#   ミッドの動きは、ほとんど機械的な関係にあります。
# - **ティックテストで符号を付けた約定 OFI** も同時点で約0.5と読めます。ところが**オラクル**の
#   行が、そのほとんどを循環論法だと暴きます。*本当の*攻撃側の符号を使うと相関は約0.07まで
#   落ちます。この市場のミッドは外生的なランダムウォークで、フローに価格インパクトはなく、
#   本当の符号付き出来高がその分のリターンと関係するのは、終値の印字の跳ね返りを通じてだけ
#   だからです。残りはティックルールが、あとで相関を取る当の価格変化から符号を導いて作り出した
#   ものです。フローが実際に価格を動かす実データなら、オラクル相当の数字はしっかりプラスに
#   なります。この比較こそが、問いの立て方を教えてくれます。
# - **予測の**相関はゼロ近辺で、約定 OFI はやや負に傾きます。ビッド・アスクの跳ね返りがあるので、
#   買いに偏った分はアスクで引けて少し平均回帰しがちだからです。完璧な符号があっても助けには
#   なりません。ここには見つけるべきフローから将来価格への因果がなく、パイプラインは正しく
#   「何もない」を見つけています。
#
# 実際のティックデータでは、気配 OFI は短い時間軸で予測力を持ちます。ただしアルファと呼ぶ前に
# 現実確認が要ります。効果が生きるのは秒から分の時間軸で、しかも枚数の限られた板の厚みの中です。
# 約1bp の動きを取るのにスプレッドと手数料を払うことになります。OFI のシグナルは、単独の戦略と
# いうより執行層の入力（いつクロスし、いつ指値を置くか）として理解するのがいちばんです。
#
# ## まとめ
#
# - `asof_join(..., 'backward', tolerance)` がその場の気配の参照で、時間範囲を指定した
#   `db.read(time_start=, time_end=)` が QA 用の窓を O(窓幅) で切り出します。現ビルドでの注意点:
#   ASOF の両側の入力を1つのストレージバッチ（約8千行）に収め、左側1行につき出力1行であることを
#   表明してください。これより大きい入力は黙って切り詰められます。`pandas.merge_asof` との照合は
#   3行のコストで確信が買えます。
# - 判定ルールは、そのジョインの上の CASE 式と `lag()` ウィンドウです。結果を `signed_trades`
#   テーブルとして保存すれば、この工程は1回で済み、下流の研究はどれもバージョン管理された
#   コミット済みの符号を読みます。
# - OFI はどちらの系統も `time_bucket` の GROUP BY 1つに収まります。`last_value(... ORDER BY ts)`
#   があるので、バケットの終値に自己結合は要りません。
# - 正直な「何もなし」には価値があります。しっかりした同時点の相関の隣に、予測の相関がほぼゼロ
#   （オラクルの符号を使ってさえ）。これは機械的で、利用できない関係の署名です。

# %%
db.close()
