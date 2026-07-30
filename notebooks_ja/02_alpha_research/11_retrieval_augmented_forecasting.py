# %% [markdown]
# # 検索拡張予測: 過去のアナログを知識ベースにする
#
# 「相場がこう見えた直近20回、そのあと何が起きたか」。アナログ予測はこの業界で
# 最も古い発想の1つです。検索拡張生成（RAG）は同じ発想を新しい部品で組んだもので、
# クエリを符号化し、過去事例の知識ベースを検索し、返ってきたもので予測を条件づけます。
# TS-RAG（[arXiv:2503.07649](https://arxiv.org/abs/2503.07649)）はこれを時系列基盤
# モデルに適用し、ファインチューニングなしで精度が上がったと報告しています。
#
# RAG は3つの部品でできています。*エンコーダ*は窓をベクトルに変換します。*知識ベース*は
# そのベクトルを、後に続いた結果と並べて保持します。*リトリーバ*はクエリベクトルの
# 最近傍を探します。
#
# 時系列では知識ベース自体が時系列であり、そこから話がややこしくなります。近傍が使えるのは、
# その結果が問い合わせの時点ですでに判明していた場合だけです。ここを間違えると、リトリーバは
# 答えそのものを差し出してきます。パイプライン全体を h5i-db で組み、そのうえで漏れを実測
# します。結果は情報係数 1.00 に相当する漏れでした。
#
# 1. 20日ぶんのリターン窓を固定長ベクトルに符号化する
# 2. h5i-db に普通の列として格納する。時刻は*結果が判明した時点*で打つ
# 3. DataFrame ビルダ経由の `array_distance` で検索する
# 4. アナログとその後の推移を描く
# 5. 漏れに値段をつける。検索ルール4通り、まったく違うバックテスト4つ
# 6. 知識ベースをバージョンに固定する。イベント時刻は2つある時計の片方でしかないため

# %% [markdown]
# ## ここで使う用語
#
# | 用語               | 意味 |
# | ---------------- | --- |
# | RAG              | 検索拡張生成。知識ベースを検索し、返ってきたものを条件にして予測する |
# | エンコーダ            | データの窓を固定長のベクトルに変換する段 |
# | 埋め込み（embedding）  | そのベクトル。ここでは普通のリスト列として保存する |
# | 知識ベース            | 保存したベクトル群。それぞれの隣にその後の結果が置かれている |
# | リトリーバ            | クエリベクトルの最近傍を見つける検索 |
# | `array_distance` | 2つのベクトルの距離を測る SQL 関数 |
# | リーク（leakage）     | 結果がまだ知り得なかった近傍。リトリーバが答えを渡してしまう |
# | IC（情報係数）         | 予測と、その後に実現したリターンとの相関 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

LOOKBACK, HORIZON, TOP_K = 20, 5, 25

db = h5i_db.Database(cu.fresh_db("alpha_rag_forecasting"), create=True)

# %% [markdown]
# ## 1. コーパス
#
# `cu.fetch_daily` はキャッシュ済みの S&P 30銘柄サンプルを、1銘柄1営業日1行で返します。
# 期間は8年半です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、昇順 |
# | `symbol` | `string` | ティッカー |
# | `open`、`high`、`low`、`close` | `float64` | その日の価格 |
# | `adj_close` | `float64` | 分割・配当調整済み終値 |
# | `volume` | `int64` | 出来高 |
#
# `adj_close` だけを残してコーパステーブルに落とします。このレシピの他はすべてここから
# 導出されるので、正しさが要求されるのはこの1つだけです。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01").to_pandas()
print(f"{len(daily):,} rows x {daily.shape[1]} columns, {daily['symbol'].nunique()} symbols")
daily.head()

# %%
prices = daily[["ts", "symbol", "adj_close"]].sort_values(["ts", "symbol"]).reset_index(drop=True)
price_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", price_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    pa.Table.from_pandas(prices, preserve_index=False).cast(price_schema),
    note="30-name daily adjusted closes",
)

returns = (
    db.table("prices")
    .select(
        "ts",
        "symbol",
        logret=(col("adj_close") / sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts")).log(),
    )
    .sort("ts")
    .to_pandas()
    .dropna()
    .reset_index(drop=True)
)
print(f"{len(returns):,} daily log returns")
returns.head()

# %% [markdown]
# ## 2. エンコーダ
#
# TS-RAG は各窓を学習済みの時系列基盤モデルで符号化し、最終トークンの埋め込みを取ります。
# 1セグメントあたり768次元です。ここでは代わりに20日ぶんの対数リターン窓を z 化するので、
# ベクトルは20次元、中身はただの四則演算です。
#
# 理由は2つあります。GPU なしでレシピが動くこと。そしてより重要なのは、検索結果が*読める*
# ことです。2つの窓が近傍になるのは、水準とボラティリティを取り除いたあとのリターンの形が
# 一致するときであり、チャートを見て納得も反論もできます。
#
# 本質はインタフェースで、そこはどちらでも同じです。エンコーダとは窓を固定長の float32
# ベクトルに写す関数にすぎません。ここを Chronos や MOMENT に差し替えても、下流で変わるのは
# 次元数だけです。
#
# z 化は窓のボラティリティを意図的に捨てます。静かなドリフトと荒々しいドリフトが、形さえ
# 同じなら互いに引き当たるようにするためです。とはいえボラティリティはノイズではありません。
# ベクトルと一緒に返して予測時に戻します。ボラ40のアナログから借りてきた5日の値動きは、
# ボラ12の相場では意味が変わるからです。

# %%
def encode(window: np.ndarray) -> tuple[np.ndarray, float]:
    """Z-score a return window. Returns the shape vector and the scale it dropped."""
    sd = float(window.std())
    vec = (window - window.mean()) / sd if sd > 0 else np.zeros_like(window)
    return vec, sd

# %% [markdown]
# ## 3. 知識ベースと、それを誠実にする1つの列
#
# 1行が1つの（銘柄, 窓）です。エンコーダの出力が `emb`、結果が `fwd_ret`、つまり窓の
# あとに続いた5日間の対数リターンです。
#
# このレシピ全体を支える設計判断は時刻列です。素直に選ぶなら窓の最終日、つまり*コンテキスト*が
# 揃った瞬間でしょう。これが間違いです。この行が使えるようになるのは*ラベル*が揃ってから、
# 5営業日あとです。ですから時刻列は `knowable_at`、すなわちフォワードリターンが出そろった
# 営業日にします。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `knowable_at` | `timestamp[us, tz=UTC]` | `fwd_ret` が観測可能になった営業日 |
# | `symbol` | `string` | 窓の出どころの銘柄 |
# | `window_end` | `timestamp[us, tz=UTC]` | 20日コンテキストの最終営業日 |
# | `emb` | `fixed_size_list<float>[20]` | 符号化したコンテキスト窓 |
# | `window_sd` | `float64` | エンコーダが z 化で落とした日次ボラティリティ |
# | `fwd_ret` | `float64` | `window_end` 以降5日間の対数リターン |
#
# `emb` はごく普通の Arrow の列です。h5i-db は他の列とまったく同じ不変 Parquet セグメントに
# 書き込み、`knowable_at` に対する同じマニフェスト枝刈りがそのまま効きます。コーパスと同期を
# 取り続けるべき別のベクトルストアは存在しません。そこが要点です。

# %%
records = []
for symbol, group in returns.groupby("symbol", sort=True):
    group = group.sort_values("ts")
    r = group["logret"].to_numpy()
    ts = group["ts"].to_numpy()
    for t in range(LOOKBACK - 1, len(r) - HORIZON):
        vec, sd = encode(r[t - LOOKBACK + 1 : t + 1])
        records.append(
            (
                ts[t + HORIZON],                        # knowable_at
                symbol,
                ts[t],                                  # window_end
                vec,
                sd,
                float(r[t + 1 : t + 1 + HORIZON].sum()),
            )
        )

kb = (
    pd.DataFrame(records, columns=["knowable_at", "symbol", "window_end", "emb", "window_sd", "fwd_ret"])
    .sort_values(["knowable_at", "symbol"])
    .reset_index(drop=True)
)
print(f"{len(kb):,} knowledge-base rows, {LOOKBACK}-dim embeddings")
kb.drop(columns="emb").head()

# %% [markdown]
# Arrow が求めるのは、平坦な float32 バッファ1本と固定リスト長です。ベクトルの束が
# メモリ上ですでに取っている形そのものです。

# %%
kb_schema = pa.schema(
    [
        pa.field("knowable_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("window_end", pa.timestamp("us", tz="UTC")),
        pa.field("emb", pa.list_(pa.float32(), LOOKBACK)),
        pa.field("window_sd", pa.float64()),
        pa.field("fwd_ret", pa.float64()),
    ]
)

def kb_arrow(frame: pd.DataFrame) -> pa.Table:
    flat = pa.array(np.concatenate(frame["emb"].to_numpy()).astype("float32"), type=pa.float32())
    return pa.table(
        {
            "knowable_at": pa.array(frame["knowable_at"], type=pa.timestamp("us", tz="UTC")),
            "symbol": pa.array(frame["symbol"]),
            "window_end": pa.array(frame["window_end"], type=pa.timestamp("us", tz="UTC")),
            "emb": pa.FixedSizeListArray.from_arrays(flat, LOOKBACK),
            "window_sd": pa.array(frame["window_sd"], type=pa.float64()),
            "fwd_ret": pa.array(frame["fwd_ret"], type=pa.float64()),
        },
        schema=kb_schema,
    )

db.create_table("analogs", kb_schema, time_column="knowable_at", sort_key=["knowable_at", "symbol"])

# Two appends, split by knowable_at, so the KB has a version history to pin in
# section 6. In production this is the nightly job.
cutoff = pd.Timestamp("2024-01-01", tz="UTC")
early, late = kb[kb["knowable_at"] < cutoff], kb[kb["knowable_at"] >= cutoff]
db.append("analogs", kb_arrow(early), note="KB through 2023")
db.append("analogs", kb_arrow(late), note="KB 2024 onward")

for v in db.versions("analogs"):
    print(f"  v{v['sequence']}  {v['op']:<7} rows={v['rows']:>7,}  {v.get('note', '')}")

# %% [markdown]
# ## 4. 検索
#
# DataFusion には `array_distance` があります。リスト列とリテラルベクトルとの L2 距離です。
# TS-RAG が FAISS の下で回している `IndexFlatL2` と同じ指標を、常駐行列の上ではなくスキャンの
# 中で評価します。
#
# 1本のクエリで3つのことが起きます。`knowable_at` の述語が知識ベースを判断時点で観測可能
# だった行に絞り、そのついでにセグメントを枝刈りします。残ったものに `array_distance` が
# 点数をつけます。`row_number` のウィンドウが、1銘柄1暦年あたり最大1つの窓だけを残します。
#
# 最後のルールは飾りではありません。同じ銘柄の連続する窓は20本中19本のリターンを共有するので、
# 重複を除かない上位25件はしばしば1つの相場局面を25回数えたものになります。そうなると近傍平均は
# 分散を均すどころか、そのまま引き継ぎます。

# %%
def retrieve(qvec, as_of=None, exclude_symbol=None, k=TOP_K, version=None):
    """Top-k analogs of `qvec`, optionally bounded to what was knowable at `as_of`."""
    literal = "[" + ", ".join(f"{v:.6f}" for v in qvec) + "]"
    q = db.table("analogs", version=version)
    predicates = []
    if as_of is not None:
        predicates.append(col("knowable_at") <= as_of)
    if exclude_symbol is not None:
        predicates.append(col("symbol") != exclude_symbol)
    if predicates:
        q = q.filter(*predicates)
    return (
        q.with_columns(d=sql_expr(f"array_distance(emb, {literal})"))
        .with_columns(
            rank=sql_expr("row_number()").over(
                partition_by=["symbol", sql_expr("date_part('year', knowable_at)")], order_by="d"
            )
        )
        .filter(col("rank") <= 1)
        .sort("d")
        .limit(k)
        .select("knowable_at", "symbol", "window_end", "d", "window_sd", "fwd_ret")
    )

def predict(neighbours: pd.DataFrame, query_sd: float) -> float:
    """Distance-weighted mean of the neighbours' forward moves, rescaled to the query's vol."""
    in_sigmas = neighbours["fwd_ret"] / neighbours["window_sd"]
    return float(query_sd * np.average(in_sigmas, weights=1.0 / (neighbours["d"] + 1e-6)))

# %% [markdown]
# クエリは AAPL の直近の完全な20日窓のうち、答え合わせ用に実現した5営業日をまだ残している
# ものを使います。

# %%
aapl = returns[returns["symbol"] == "AAPL"].sort_values("ts").reset_index(drop=True)
aapl_r = aapl["logret"].to_numpy()

q_idx = len(aapl) - 200
q_date = aapl["ts"].iloc[q_idx]
q_vec, q_sd = encode(aapl_r[q_idx - LOOKBACK + 1 : q_idx + 1])
realized = float(aapl_r[q_idx + 1 : q_idx + 1 + HORIZON].sum())

frame = retrieve(q_vec, as_of=q_date.isoformat())
print(frame.sql()[:420], "...\n")
neighbours = frame.to_pandas()
print(f"query: AAPL window ending {q_date.date()}, KB bounded to knowable_at <= {q_date.date()}")
neighbours.head(8)

# %% [markdown]
# この結果で最も近いものでも距離はおよそ2.5です。ベクトルが z 化された20次元空間での話なので、
# アナログとしてはかなり緩いということになります。日付もティッカーもサンプル全体に散っており、
# 30銘柄でクロスセクションのアナログ検索を誠実にやるとこうなります。
#
# 検索が返すのは*ポインタ*であって経路ではありません。各アナログが実際にどう動いたかの復元は、
# `symbol` と `window_end` を鍵にしたコーパスへの引き当てです。
#
# どの経路もエンコーダ自身の出力空間で描きます。コンテキストのリターンから平均を引き、窓の
# ボラティリティで割り、それを累積したものです。これはリトリーバが距離を最小化した対象の
# ベクトルそのものです。ホライズン側は同じシグマ単位のままで平均は引きません。将来のドリフトは
# まさに予測しようとしている当のものだからです。

# %%
import matplotlib.pyplot as plt

by_symbol = {s: g.sort_values("ts").reset_index(drop=True) for s, g in returns.groupby("symbol")}

def analog_path(symbol: str, window_end: pd.Timestamp, sd: float) -> np.ndarray:
    """Encoded context then realized horizon, in window sigmas, zeroed at the window end."""
    g = by_symbol[symbol]
    t = int(g.index[g["ts"] == window_end][0])
    r = g["logret"].to_numpy()
    context = (r[t - LOOKBACK + 1 : t + 1] - r[t - LOOKBACK + 1 : t + 1].mean()) / sd
    horizon = r[t + 1 : t + 1 + HORIZON] / sd
    return np.concatenate([[0.0], np.cumsum(context), np.cumsum(horizon)])

forecast = predict(neighbours, q_sd)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(-LOOKBACK, HORIZON + 1)
for _, nb in neighbours.iterrows():
    ax.plot(
        x, analog_path(nb["symbol"], nb["window_end"], nb["window_sd"]),
        color="tab:gray", alpha=0.35, lw=0.9,
    )
ax.plot(x, analog_path("AAPL", q_date, q_sd), color="tab:blue", lw=2.5, label="query window (AAPL)")
ax.plot([0, HORIZON], [0, forecast / q_sd], color="tab:orange", lw=2.5, marker="o",
        label=f"analog forecast ({forecast:+.3f} raw)")
ax.plot([0, HORIZON], [0, realized / q_sd], color="tab:red", lw=2.5, ls="--", marker="o",
        label=f"realized ({realized:+.3f} raw)")
ax.axvline(0, color="black", lw=0.8)
ax.set_title(f"{TOP_K} retrieved analogs and their continuations")
ax.set_xlabel("sessions relative to window end")
ax.set_ylabel("cumulative log return (window sigmas)")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
print(f"query window vol {q_sd:.4f}/day, forecast {forecast:+.4f}, realized {realized:+.4f}")

# %% [markdown]
# 灰色の扇形が検索されてきた事例で、クエリ日で揃えてあります。0より左ではクエリ窓にぴったり
# 沿っており、これはリトリーバが仕事をしている証拠です。0より右ではおよそマイナス6シグマから
# プラス4シグマまで広がります。これが誠実な答えです。似て見える過去は、その後どこへでも
# 行きました。
#
# 1回の抽出では何も証明できません。予測は扇形の加重中心であり、その良し悪しは多数のクエリに
# わたって見るしかありません。
#
# ## 5. 漏れに値段をつける
#
# 同じ149個のクエリ日、同じエンコーダ、同じ上位k件に対して、検索ルールを4通り。変えるのは
# フィルタだけです。
#
# **A. 知識ベース全体。** フィルタなし。素朴なベクトルストアの構成です。
#
# **B. クエリ銘柄を除外。** 問題は自己検索だという見立てで、たいていの人が最初に書く手当てです。
#
# **C. `knowable_at <= t`。** 結果がすでに出ていた行だけに知識ベースを絞ります。
#
# **D. 両方。** C に銘柄除外を足し、時計を直したあとでもこの手当てに意味が残るかを見ます。

# %%
query_idx = list(range(len(aapl) - 750, len(aapl) - HORIZON, 5))

def backtest(leakfree: bool, exclude_self: bool) -> dict:
    preds, actuals = [], []
    for i in query_idx:
        d = aapl["ts"].iloc[i]
        vec, sd = encode(aapl_r[i - LOOKBACK + 1 : i + 1])
        nb = retrieve(
            vec,
            as_of=d.isoformat() if leakfree else None,
            exclude_symbol="AAPL" if exclude_self else None,
        ).to_pandas()
        preds.append(predict(nb, sd))
        actuals.append(float(aapl_r[i + 1 : i + 1 + HORIZON].sum()))
    preds, actuals = np.array(preds), np.array(actuals)
    return {
        "IC": np.corrcoef(preds, actuals)[0, 1],
        "hit_rate": float((np.sign(preds) == np.sign(actuals)).mean()),
    }

variants = {
    "A whole KB": backtest(leakfree=False, exclude_self=False),
    "B exclude own symbol": backtest(leakfree=False, exclude_self=True),
    "C knowable_at <= t": backtest(leakfree=True, exclude_self=False),
    "D both": backtest(leakfree=True, exclude_self=True),
}
noise = 1.0 / np.sqrt(len(query_idx))
print(f"{len(query_idx)} query dates, IC standard error under the null ~{noise:.3f}")
pd.DataFrame(variants).T.round(3)

# %%
names = list(variants)
ics = [variants[n]["IC"] for n in names]
colors = ["tab:red", "tab:orange", "tab:blue", "tab:blue"]

fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4))
for ax, keep, title in (
    (ax_all, slice(0, 4), "All four rules"),
    (ax_zoom, slice(1, 4), "Without A, rescaled"),
):
    ax.bar(names[keep], ics[keep], color=colors[keep])
    ax.axhspan(-1.96 * noise, 1.96 * noise, color="gray", alpha=0.3, label="95% band under the null")
    for x, v in zip(names[keep], ics[keep]):
        ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("retrieval rule")
    ax.tick_params(axis="x", labelrotation=15)
ax_all.set_ylabel("corr(prediction, realized 5d return)")
ax_zoom.legend(fontsize=9)
fig.suptitle("Information coefficient by retrieval rule")
fig.tight_layout()

# %% [markdown]
# **A は 1.00 です。**「強い」でも「怪しい」でもなく、ちょうど1。クエリ窓そのものが知識ベースの
# 1行として入っており、自分自身との距離はゼロ、距離の逆数で重みづけすればほぼ全部の重みが
# そこに乗ります。リトリーバは答えを引いてきているだけです。
#
# **危ないのは B です。** クエリ銘柄を除くと完全一致が消え、IC は 0.17 まで落ちます。正で、
# 標準誤差の2倍をわずかに超え、社内レビューを通ってしまう類いの数字です。それでも中身は
# 純然たる漏れです。他の29銘柄の窓のうちクエリ日*より後*に終わるものは、AAPL がこれから
# 通過する当の営業日にわたるフォワードリターンを抱えています。マーケットファクターが別の
# 扉から未来を運んでくるわけです。
#
# **C はノイズ帯の内側**で、これがこの実験の誠実な読み方です。30銘柄に対する20次元の形状照合は
# 5日リターンを予測しません。このレシピの仕事はそれを測ることであって打ち負かすことではなく、
# A と C の差はアルファではなく配管の誤りでした。
#
# **D は診断を裏づけます。** `knowable_at` が知識ベースを縛ったあとでは、クエリ自身の銘柄を
# 除いても何も変わりません。銘柄除外はタイムスタンプの問題の代理でしかなく、代理は思いついた
# ケースしか直しません。
#
# 一般化するとこうです。ベクトルインデックスは時間について何の意見も持ちません。それを時系列に
# 貼り付ければ、どの検索も先読みバグの予備軍になります。埋め込みを時系列データベースに置き、
# 時刻列を*ラベルが判明した時点*に取ると、正しいクエリのほうが自然な書き方になります。
#
# ## 6. もう1つの時計
#
# `knowable_at` が扱うのはイベント時刻、事実が真になった瞬間です。*到着*時刻、つまり行が
# 自分のストアに着いた瞬間については何も言いません。2つが食い違うのは、知識ベースの行が
# **導出物**だからです。調整済み終値は分割や配当のあとで訂正され、誤ったプリントも修正される
# ので、今日のコーパスから知識ベースを組み直しても、去年配っていた行は再現しません。
#
# バージョン固定がその第2の時計です。追記1回ごとにバージョンが立ち、固定は `knowable_at`
# フィルタを置き換えるのではなく、組み合わさります。

# %%
at_head = retrieve(q_vec, as_of=q_date.isoformat()).to_pandas()
at_v1 = retrieve(q_vec, as_of=q_date.isoformat(), version=1).to_pandas()

for label, version, got in (("head", db.versions("analogs")[-1], at_head), ("v1", db.versions("analogs")[1], at_v1)):
    print(
        f"KB at {label:<4} {version['rows']:>7,} rows -> {len(got)} neighbours, "
        f"newest knowable_at {got['knowable_at'].max().date()}"
    )

# %% [markdown]
# どちらのクエリも判断時点は同じです。固定のほうが厳しいところでは固定が勝ちます。バージョン1は
# 2024年の追記より前の知識ベースなので、`knowable_at` の上限が2025年まで許していても、使える
# 最も新しいアナログは2023年のものになります。
#
# 追記だけの知識ベースなら差はここまでで、固定が買っているのは正しさではなく再現性です。
# それが正しさを買い始めるのは、コーパスが訂正された瞬間からです。そのときは行数だけでなく
# 古い行の*中身*まで変わるからです。訂正を監査可能なバージョンとして扱う話はレシピ
# [01/05](../01_market_data_engineering/05_corporate_actions.ipynb)、固定を再現可能な
# 実行に変える話はレシピ
# [03/02](../03_risk_and_production/02_reproducible_backtests.ipynb) にあります。
#
# ## 総当たりの先へ
#
# `array_distance` は全件走査で、TS-RAG が使う `IndexFlatL2` も全件走査なので、指標の面では
# 同条件です。定数倍の面では同条件ではありません。FAISS は走査を常駐行列に対する行列積として
# 回しますが、DataFusion は汎用の Arrow カーネルを行ごとに評価します。
#
# 6万3千行・20次元のここでは検索1回が数十ミリ秒で、しかもその前に `knowable_at` の述語が
# セグメントを枝刈りします。数百万セグメント・768次元になれば算術がボトルネックになり、近似
# インデックスが欲しくなります。その移行を生き延びる分担は、すでにここで組んだものです。
# h5i-db がコーパスとラベルと2つの時計を持ち、距離を計算する何かに絞り込み済みの候補集合を
# 渡します。

# %% [markdown]
# ## まとめ
#
# - 埋め込みはただの列です。`pa.list_(pa.float32(), n)` が1行あたり固定長ベクトルを保持し、
#   `array_distance` がフィルタと同じスキャンの中で厳密な L2 上位k件を返します。別途同期を
#   取るインデックスはありません。
# - 知識ベースの行には、コンテキストが終わった時刻ではなく**ラベルが判明した時刻**を打ちます。
#   これで漏れ対策がただの時刻述語になり、しかもマニフェスト枝刈りが速度まで面倒を見ます。
# - 時系列の検索は放っておくと漏れます。素朴な知識ベースは IC 1.00 を出し、素直な手当てを
#   しても、もっともらしく発表もできてしまう完全な偽物の 0.17 が残りました。
# - イベント時刻と到着時刻は別の時計です。前者は `knowable_at`、後者は
#   `db.table(..., version=)` が受け持ち、訂正が起きる場面では両方が要ります。
# - 平均を取る前に近傍の重複を除きます。1銘柄の重なり合う窓は互いにほぼ同じベクトルなので、
#   重複を除かない上位k件は1つの局面に偽の自信をつけたものになります。

# %%
db.close()
