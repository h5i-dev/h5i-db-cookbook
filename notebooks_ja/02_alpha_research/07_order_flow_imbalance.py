# %% [markdown]
# # オーダーフロー・インバランス: 符号付き出来高はリターンを予測するか
#
# オーダーフロー・インバランスは、買い主導の出来高が売り主導をどれだけ上回るかで、マイクロ
# ストラクチャの主力シグナルです。同時点の価格変動と強く相関し、実際の市場の短い horizon では
# 次の変動をわずかに予測します。
#
# このレシピでは両方の系統を作ります。
#
# 1. **約定 OFI**: 署名ルールを正解と突き合わせて検証します。その時点の気配への ASOF ジョイン
#    による Lee-Ready と、代替としてのティックテストです。そのうえでテープに符号を付け、
#    符号付き出来高を分ごとに集計します。
# 2. **気配 OFI**（Cont, Kukanov, Stoikov 2014）: 最良ビッド・アスクの価格と数量の変化から出る
#    板の圧力を、`lag()` のウィンドウで計算します。
# 3. 1分リターンとの、同時点の相関と*予測的な*相関を正直に見ます。

# %% [markdown]
# ## ここで使う用語
#
# | 用語                   | 意味 |
# | -------------------- | --- |
# | オーダーフロー・インバランス       | ある区間の買い仕掛け出来高から売り仕掛け出来高を引いた値 |
# | 買い仕掛け                | 買い方がスプレッドを越えて約定を成立させたこと。フィードは教えないので推定する |
# | 符号付き出来高              | その符号を持たせた出来高 |
# | Lee-Ready 法          | 標準の符号付けルール。約定価格を直前ミッドと比べる |
# | ティックテスト              | 代替ルール。直前の約定価格と比べる |
# | 気配 OFI               | 最良ビッド・アスクの価格と数量の変化から測る板の圧力 |
# | 同時点（contemporaneous） | リターンと同じ区間で測ること。予測ではなく説明にとどまる |
# | 予測的（predictive）      | リターンの直前の区間で測ること。役に立つのはこちらだけ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_ofi"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_quotes` の2セッションぶんのベストビッド・ベストオファーです。最良気配が動くたびに
# 1行です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 気配の時刻、昇順 |
# | `symbol` | `string` | 銘柄コード、`AAPL` か `MSFT` |
# | `bid`、`ask` | `float64` | 最良のビッドとオファー |
# | `bid_size`、`ask_size` | `int64` | 各サイドの表示数量 |

# %%
quotes = cu.make_quotes(symbols=["AAPL", "MSFT"], days=2)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# 生成器の癖が1つ、以降の作りを決めます。`make_trades` のプリントは `make_quotes` のミッドと
# 緩くしか結びつかないランダムウォークで、これでは気配を使った約定の署名が意味を持ちません。
#
# そこで合成の気配ストリームはそのまま使い、そこから*自分たちでテープをプリント*します。
# マッチングエンジンがやるのと同じです。気配更新の25%が1〜50ms 後に約定を引き起こし、買い手は
# オファーを取り、売り手はビッドを叩き、本当のアグレッサー側を記録します。
#
# テープは構造上、気配と価格が整合します。だから以下のどの署名ルールも正解と突き合わせて採点
# できます。実データでは決して手に入らない贅沢です。プリントしたテープの列は `ts`、`symbol`、
# `price`、`size`、`side` です。

# %%
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
print(f"{len(trades_df):,} trades printed off {len(q):,} quotes")
trades_df.head()

# %%
db.create_table("quotes", quotes.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("quotes", quotes, note="2-day synthetic NBBO")
db.create_table("trades", trade_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", trades, note="tape printed off the quote stream")
print({t: len(db.read(t)) for t in db.tables()})

# %% [markdown]
# ## 2. QA 窓で署名ルールを検証する
#
# Lee-Ready はこう言います。その時点のミッドより上なら買い、下なら売り、ミッドちょうどなら
# ティックテストに落とし、アップティックなら買い。
#
# その時点の気配の参照は `asof_join(..., 'backward', 5_000_000)` です。銘柄ごとに、古くても
# 5秒以内の最新の気配を取り、許容差は生のマイクロ秒です。
#
# 検証は日中の15分の窓で行います。時間範囲を指定した `db.read` で切り出し、生のマイクロ秒の
# 境界を渡してセグメント単位でプルーニングし、小さな作業台テーブルとして保存します。窓を
# 切った検証は回すのが速く、目でも追えます。
#
# ジョインが約定1件につき1行を返すことを assert し、貼り付いた気配を `pandas.merge_asof` と
# 突き合わせます。

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

def sign_of(x, prev):
    """+1 uptick, -1 downtick, 0 unchanged - the tick test."""
    return when(x > prev).then(lit(1)).when(x < prev).then(lit(-1)).otherwise(lit(0))


PREV_PX = sql_expr("lag(price)").over(partition_by="symbol", order_by="ts")
TRUE_SIGN = when(col("side") == "B").then(lit(1)).otherwise(lit(-1))

audit = (
    db.table("trades_qa")
    .join_asof(db.table("quotes_qa"), on="ts", by="symbol", tolerance=5_000_000)
    .select("ts", "symbol", "price", "size", "side",
            mid=(col("bid") + col("ask")) / 2)
    .with_columns(prev_px=PREV_PX)
    .select(
        "ts", "symbol", "mid",
        lr_sign=when(col("price") > col("mid")).then(lit(1))
        .when(col("price") < col("mid")).then(lit(-1))
        .when(col("price") > col("prev_px")).then(lit(1))
        .when(col("price") < col("prev_px")).then(lit(-1))
        .otherwise(lit(0)),
        tick_sign=sign_of(col("price"), col("prev_px")),
        true_sign=TRUE_SIGN,
    )
    .sort(["ts", "symbol"])
    .to_pandas()
)
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
# 価格の整合したテープの上では、Lee-Ready は評判どおりの働きをします。クオートルールはほぼ
# 完璧で、誤りは気配とプリントのあいだの1〜50ms に気配が更新された場合から来ます。純粋な
# ティックテストはかなり後ろを走ります。実際の TAQ で Lee-Ready がおよそ85%を取るのと同じ
# 序列です。
#
# ## 3. テープ全体に符号を付ける
#
# 検証したクオートルールを本番規模で使うには、約定と気配の ASOF が要ります。このレシピでは
# ティックテストで OFI の研究を担うのに十分な精度がありますし、`trades` に対する純粋な
# `lag()` のウィンドウなので揃え作業も不要です。生成器の本当のサイドは、オラクルの上限として
# 残しておきます。
#
# 符号を付けたテープは一級のテーブルとして保存します。署名は十分に高くつく処理なので、1度
# やって、コミットして、バージョン管理したいところです。

# %%
signed = (
    db.table("trades")
    .with_columns(prev_px=PREV_PX)
    .select(
        "ts", "symbol", "price", "size",
        tick_sign=sign_of(col("price"), col("prev_px")),
        true_sign=TRUE_SIGN,
    )
    .sort(["ts", "symbol"])
    .to_arrow()
)

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
# `time_bucket` の集約1つで、符号付きの買い・売り出来高、正規化したインバランス
# `(buys - sells) / total`、そして `last_value(... ORDER BY ts)` の定型によるバケットの終値が
# 出ます。

# %%
BAR = time_bucket("1m", col("ts"))
size_when = lambda flag: when(flag).then(col("size")).otherwise(lit(0)).sum()

trade_ofi = (
    db.table("signed_trades")
    .group_by(BAR.alias("bar"), "symbol")
    .agg(
        buy_vol=size_when(col("tick_sign") > 0),
        sell_vol=size_when(col("tick_sign") < 0),
        oracle_net=when(col("true_sign") > 0).then(col("size")).otherwise(-col("size")).sum(),
        volume=col("size").sum(),
        close=col("price").last("ts"),
    )
    .sort(["bar", "symbol"])
    .to_pandas()
)
trade_ofi["ofi"] = (trade_ofi["buy_vol"] - trade_ofi["sell_vol"]) / trade_ofi["volume"]
trade_ofi["ofi_oracle"] = trade_ofi["oracle_net"] / trade_ofi["volume"]
trade_ofi.head(4)

# %% [markdown]
# ## 5. 気配 OFI（Cont ら）
#
# 板の圧力から出す OFI には、最良における*変化*が要ります。ビッドの価格が上がる、あるいは
# ビッドが変わらないまま数量が増えると、買い圧力が加わります。アスク側はその鏡像で引き算に
# なります。
#
# 気配1件につき `lag()` の比較が4つ、純粋な SQL のウィンドウ機構です。それを分ごとに合計し、
# リターン用にバケットの終値ミッドを添えます。

# %%
def prev(name: str):
    return sql_expr(f"lag({name})").over(partition_by="symbol", order_by="ts")


# Cont et al.: size added at the bid counts positive, size pulled counts
# negative, mirrored on the ask. Four lagged comparisons per quote.
keep = lambda flag, x: when(flag).then(x).otherwise(lit(0))

quote_ofi = (
    db.table("quotes")
    .with_columns(pb=prev("bid"), pa=prev("ask"), pbs=prev("bid_size"), pas=prev("ask_size"))
    .filter(col("pb").is_not_null())
    .group_by(BAR.alias("bar"), "symbol")
    .agg(
        qofi=(
            keep(col("bid") >= col("pb"), col("bid_size"))
            - keep(col("bid") <= col("pb"), col("pbs"))
            - keep(col("ask") <= col("pa"), col("ask_size"))
            + keep(col("ask") >= col("pa"), col("pas"))
        ).sum(),
        mid_close=((col("bid") + col("ask")) / 2).last("ts"),
    )
    .sort(["bar", "symbol"])
    .to_pandas()
)
quote_ofi.head(4)

# %% [markdown]
# ## 6. 同時点と予測的
#
# 肝心な問いはこれです。今分の OFI が語るのは*今分*のリターン、つまり機械的な関係でしょうか。
# それとも*次の分*のリターン、つまりアルファでしょうか。
#
# 推定量ごと、銘柄ごとに両方の相関を計算します。リターンはセッション内だけで、`shift` は銘柄と
# 日の中で取るので、オーバーナイトの副作用は入りません。

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
# 1つの表に正直な教訓が3つ入っています。
#
# - **気配 OFI** は教科書どおりの同時点の関係、およそ 0.5〜0.6 を示します。板の圧力と同分の
#   ミッドの動きは、ほぼ機械的に結びついています。
# - **ティックテストの符号による約定 OFI** も同時点でおよそ 0.5 を示します。**オラクル**の行が
#   その大半を循環だと暴きます。*本当の*アグレッサー側を使うと相関はおよそ 0.07 まで落ちます。
#   この市場ではミッドが外生のランダムウォークで、フローに価格インパクトはなく、本当の符号付き
#   出来高がその分のリターンと結びつくのは、終値のプリントの跳ね返りを通じてだけです。残りは
#   ティックルールが、そのあと相関を取る当の価格変化から符号を導いて作り出したものです。実際の
#   データではフローが価格を動かすので、オラクル相当の数字はしっかりプラスになります。この
#   比較こそが、問いの立て方を教えてくれます。
# - **予測的**な相関はゼロ近くで、約定 OFI ではマイナスに傾きます。ビッド・アスクの跳ね返りに
#   より、買いに偏った分はアスクで終わりやすく、少し平均回帰します。完璧な符号でも助けには
#   なりません。ここには見つけるべきフローから将来価格への因果がなく、パイプラインは正しく
#   何も見つけません。
#
# 実際のティックデータでは、気配 OFI は短い horizon の予測力を確かに持ちます。それでもアルファ
# と呼ぶ前に現実確認が要ります。効果は秒から分の horizon で、数量の限られた板の中に住み、
# 1bp の動きを取るのにスプレッドと手数料を払うことになります。OFI のシグナルは、いつ叩き、
# いつ置くかという執行層の入力と考えるのがいちばんで、単独の戦略ではありません。
#
# ## まとめ
#
# - `asof_join(..., 'backward', tolerance)` がその時点の気配の参照で、時間範囲つきの
#   `db.read(time_start=, time_end=)` が QA の窓を窓の大きさに比例するコストで切り出します。
#   現行ビルドの注意点が1つ。ASOF の両入力を1つのストレージバッチ、およそ8,000行の中に収め、
#   左の行数と出力の行数が一致することを assert してください。それより大きな入力は黙って
#   切り捨てられます。`pandas.merge_asof` との突き合わせは3行で確信が買えます。
# - 署名ルールは、そのジョインの上の CASE 式と `lag()` のウィンドウです。結果を
#   `signed_trades` テーブルとして保存すれば、処理は1度だけ走り、下流の研究はどれも
#   バージョン管理されコミットされた符号を読みます。
# - どちらの系統の OFI も、`time_bucket` の GROUP BY 1つずつに畳めます。バケットの終値は
#   `last_value(... ORDER BY ts)` が自己結合なしで返します。
# - 正直なヌルには意味があります。同時点の相関がしっかりしていて、オラクルの符号を使ってさえ
#   予測的な相関がゼロ近くなら、それは利用できる関係ではなく機械的な関係の署名です。

# %%
db.close()
