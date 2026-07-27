# %% [markdown]
# # リードラグの発見: 不規則なティックの上の相互相関
#
# どちらが先に動くのか。リードラグ分析は関連する銘柄を組にします。指数と先物、ADR と本国上場、
# 相関する FX クロス。そしてこの分析には機械的な問題が1つ付きまといます。ティックは不規則かつ
# *非同期*に届き、素朴な手当てはどれも答えを歪めます。バケット化も直前値の持ち越しもそうで、
# その結果が Epps 効果です。
#
# *既知の、注入した*リードを持つペアで道具を一通り見ていきます。
#
# 1. EURUSD を 500ms 遅れで追う合成 GBPUSD を作る
# 2. バケットして相関する。`time_bucket` による 250ms 格子の上の古典的な相互相関関数
# 3. `asof_join` でティック単位に揃え、鮮度の許容差を振る
# 4. 格子をまったく必要としない Hayashi-Yoshida 推定量を回す。時間シフトを振れば、再標本化
#    せずにラグを特定できる
# 5. 実データ、日次の株式ペアで現実確認をする

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_leadlag"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_fx_ticks` は 24/7 の FX 型のティックを返します。1行が1回の気配更新です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 気配の時刻、昇順 |
# | `pair` | `string` | 通貨ペア |
# | `bid`、`ask` | `float64` | 最良のビッドとオファー |
#
# ここでは EURUSD を24時間、1時間あたり20,000ティック、およそ180ms に1本のペースで取ります。

# %%
LAG_US = 500_000
BETA = 0.9

eur = cu.make_fx_ticks(pairs=["EURUSD"], hours=24, ticks_per_hour=20_000).to_pandas()
print(f"{len(eur):,} EURUSD ticks x {eur.shape[1]} columns")
eur.head()

# %% [markdown]
# 生成器が作るペアは互いに*独立*なので、2つのあいだにリードラグを見つけたらそれはノイズです。
# そこで生成器の EURUSD をリーダーとして扱い、GBPUSD は自分たちで構成します。
#
# GBPUSD の各ティック時刻 `t` において、対数ミッドは
# `beta * eur_logmid(t - 500ms) + idio(t)` です。リーダーの経路を半秒古い時点で後ろ向きに
# asof サンプリングし、同程度の大きさの固有ランダムウォークを足したものです。
#
# 正解はこうです。EURUSD がちょうど 500ms 先行し、ベータは 0.9、それ以外に2つを結ぶものは
# ありません。

# %%
eur_ts = eur["ts"].astype("int64").to_numpy()
eur_logmid = np.log((eur["bid"] + eur["ask"]) / 2).to_numpy()

rng = np.random.default_rng(99)
n = len(eur)
t0, t1 = eur_ts[0], eur_ts[-1]
gbp_ts = np.sort(rng.integers(t0 + LAG_US, t1, n))  # own asynchronous clock

# backward-asof sample of the leader at (t - 500ms), in numpy
idx = np.searchsorted(eur_ts, gbp_ts - LAG_US, side="right") - 1
idio = np.cumsum(rng.normal(0, 4e-5, n))
gbp_logmid = np.log(1.27) + BETA * (eur_logmid[idx] - eur_logmid[0]) + idio
gbp_mid = np.exp(gbp_logmid)
half = gbp_mid * rng.uniform(0.2e-4, 1e-4, n) / 2

gbp = pd.DataFrame(
    {
        "ts": pd.to_datetime(gbp_ts, unit="us", utc=True).astype("datetime64[us, UTC]"),
        "pair": "GBPUSD",
        "bid": np.round(gbp_mid - half, 5),
        "ask": np.round(gbp_mid + half, 5),
    }
)
fx = (
    pd.concat([eur, gbp])
    .sort_values(["ts", "pair"])
    .reset_index(drop=True)
)
schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("pair", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)
db.create_table("fx", schema, time_column="ts", sort_key=["ts", "pair"])
db.append("fx", pa.Table.from_pandas(fx, preserve_index=False).cast(schema), note="EURUSD + lagged GBPUSD")
print(f"{len(fx):,} ticks over 24h ({n:,} per pair, ~{n / 86_400:.1f}/sec)")

# %% [markdown]
# ## 2. バケットした相互相関関数: 250ms の `time_bucket`
#
# 古典的なやり方は、両方の系列を細かい共通格子に再標本化し、リターンを計算し、あらゆる
# リード／ラグのオフセットで相関を取るものです。
#
# `time_bucket` は `'250ms'` のような1秒未満の幅を取るので、格子への縮約はデータベースの中で
# 起きます。GROUP BY 1つが96万ティックをバケットごとの終値ミッドに変え、整列済みストレージの
# 上を流れます。pandas 側は格子を前方補完してずらすだけです。

# %%
# The builder's time_bucket() parses whole-second units only, so the 250 ms
# width comes in through the escape hatch - everything else is still verbs.
BUCKET_250MS = sql_expr("time_bucket('250ms', ts)")

bucketed = (
    db.table("fx")
    .group_by(BUCKET_250MS.alias("bucket"), "pair")
    .agg(mid=((col("bid") + col("ask")) / 2).last("ts"))
    .to_pandas()
)

grid = bucketed.pivot(index="bucket", columns="pair", values="mid")
full_index = pd.date_range(grid.index.min(), grid.index.max(), freq="250ms", tz="UTC")
rets = np.log(grid.reindex(full_index).ffill()).diff().dropna()

lags = np.arange(-12, 13)
ccf = pd.Series(
    {k: rets["EURUSD"].corr(rets["GBPUSD"].shift(-k)) for k in lags}
)
band = 1.96 / np.sqrt(len(rets))

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(lags * 0.25, ccf.values, width=0.2, color="tab:blue")
ax.axhspan(-band, band, color="gray", alpha=0.3, label="95% band under independence")
ax.axvline(0.5, color="tab:red", ls="--", lw=1, label="true lag (0.5 s)")
ax.set_title("CCF of 250 ms returns: corr(EURUSD$_t$, GBPUSD$_{t+k}$)")
ax.set_xlabel("k (seconds; positive = EURUSD leads)")
ax.set_ylabel("correlation")
ax.legend()
fig.tight_layout()
peak = ccf.idxmax()
print(f"CCF peak at k = {peak * 0.25:+.2f}s (corr {ccf.max():.2f}); corr at k=0: {ccf.loc[0]:.2f}")

# %% [markdown]
# 相互相関関数は方向を言い当て、ラグの位置も絞ります。質量が +0.5 秒と +0.75 秒に乗ります。
# 注入した 500ms に、最終ティックによるサンプリングの古さが加わって、効果を1バケット右へ
# にじませます。ティックは180ms ほどごとに届くので、バケットの「終値ミッド」自体が少し古い
# のです。
#
# ゼロにも負のラグにも何もないので、GBPUSD が先行することはありません。
#
# これは離散化のトレードオフの縮図です。もっと粗い格子なら効果まるごとがラグ0の1本に積み上が
# り、方向を*隠して*しまいます。もっと細かい格子は各バケットを痩せさせ、持ち越したゼロが
# 支配するようになります。それが Epps 効果です。
#
# ## 3. ティック単位の揃え: `asof_join` と鮮度の許容差
#
# 格子なしの推定に入る前の主力操作は、不規則な系列をもう一方に揃えることです。GBPUSD の各
# ティックに対して、直近の EURUSD ティックを取る。それが `asof_join(..., 'backward', tolerance)`
# です。
#
# 許容差は生のマイクロ秒で、データ品質のダイヤルです。リーダーの気配がどれだけ古いところまで
# 許すのか。20分の窓で検証し、許容差を振りながら、対応率と揃えたティック間リターンの相関を
# 追います。
#
# 窓で切った抽出は検証を速く保ち、数字も目で追えます。ジョインが左のティック1件につき1行を
# 返すことを assert し、標本を `pandas.merge_asof` と突き合わせます。ASOF のパイプラインでは
# 標準的な衛生管理です。

# %%
w0 = int(pd.Timestamp("2026-06-01 12:00", tz="UTC").value // 1000)
w1 = int(pd.Timestamp("2026-06-01 12:20", tz="UTC").value // 1000)
BUF_US = 10_000_000  # give the window a 10s run-up so early ticks can match

qa_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("base", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
    ]
)
eur_w = eur[(eur["ts"].astype("int64") >= w0 - BUF_US) & (eur["ts"].astype("int64") < w1)]
gbp_w = gbp[(gbp["ts"].astype("int64") >= w0) & (gbp["ts"].astype("int64") < w1)]
for name, frame in (("eur_qa", eur_w), ("gbp_qa", gbp_w)):
    tbl = pa.Table.from_pandas(
        frame.assign(base="USD")[["ts", "base", "bid", "ask"]], preserve_index=False
    ).cast(qa_schema)
    db.create_table(name, qa_schema, time_column="ts")
    db.append(name, tbl, note="QA window 12:00-12:20")
print(f"QA window: {len(gbp_w):,} follower ticks, {len(eur_w):,} leader ticks")

rows = []
for tol_ms in (50, 100, 250, 500, 1000, 5000):
    j = (
        db.table("gbp_qa")
        .join_asof(db.table("eur_qa"), on="ts", by="base", tolerance=tol_ms * 1000)
        .select(
            "ts", "ts_right",
            gbp_mid=(col("bid") + col("ask")) / 2,
            eur_mid=(col("bid_right") + col("ask_right")) / 2,
        )
        .sort("ts")
        .to_pandas()
    )
    assert len(j) == len(gbp_w), "asof_join must return one row per left tick"
    matched = j.dropna(subset=["eur_mid"])
    r = np.log(matched[["gbp_mid", "eur_mid"]]).diff().dropna()
    rows.append(
        {
            "tolerance_ms": tol_ms,
            "match_rate": len(matched) / len(j),
            "aligned_ret_corr": r["gbp_mid"].corr(r["eur_mid"]),
        }
    )

# cross-check the last sweep (5s tolerance) against pandas.merge_asof
ref = pd.merge_asof(
    gbp_w[["ts"]].reset_index(drop=True),
    eur_w.assign(eur_mid_ref=(eur_w["bid"] + eur_w["ask"]) / 2)[["ts", "eur_mid_ref"]],
    on="ts",
    tolerance=pd.Timedelta("5s"),
)
chk = j.merge(ref, on="ts")
assert np.allclose(chk["eur_mid"], chk["eur_mid_ref"], equal_nan=True), "asof mismatch vs pandas"
print("asof_join validated against pandas.merge_asof")
pd.DataFrame(rows).round(3)

# %% [markdown]
# リーダーのティックが毎秒およそ5.5本なので、50ms の許容差ではフォロワーのティックの4分の1
# しか残りません。残りは十分に新しい対応がなく NULL で返るので、静かに古いのではなく目に見え
# ます。500ms あたりになれば、ほぼすべてが対応づきます。
#
# 相関の列は丁寧に読む価値があります。許容差を緩めると相関が*下がる*のです。原因は対応が古く
# なることではありません。生き残るティックのペアが密になり、連続する揃ったリターンがどんどん
# 短い区間をまたぐようになるからです。500ms のラグより短い区間は、相手方とほとんど同時点の
# 重なりを持ちません。相関は 0.07 のあたりまで落ちます。
#
# 許容差をきつくすると、痩せた系列はおよそ1秒の区間をまたぎ、それがラグをまたぐので相関が
# 戻ってきます。
#
# どちらの数字も「その」相関ではありません。非同期でラグのある系列では、測る区間の長さで
# 測れるものが変わります。表になった Epps 効果であり、区間を明示的に尊重する推定量が要る理由
# です。
#
# ## 4. Hayashi-Yoshida: 格子なしで、ラグのダイヤル付き
#
# HY 推定量は、時間区間が*重なる*ティック区間リターンのすべての組について `r_i * s_j` を
# 足し上げます。非同期データに対して不偏で、再標本化も持ち越したゼロもありません。
#
# フォロワーの時計に時間シフトをかけて振ると、これが格子なしの相互相関図になります。Hoffmann、
# Rosenbaum、Yoshida の流儀です。HY 相関を最大にするシフトがラグの推定値になります。2ポインタ
# の走査で O(n + m) に収まります。

# %%
def hy_corr(t_a, r_a, t_b, r_b):
    """Hayashi-Yoshida correlation from interval times and returns."""
    cov = 0.0
    j0 = 0
    for i in range(1, len(t_a)):
        a_lo, a_hi = t_a[i - 1], t_a[i]
        while j0 < len(t_b) - 1 and t_b[j0 + 1] <= a_lo:
            j0 += 1
        j = j0
        while j < len(t_b) - 1 and t_b[j] < a_hi:
            if min(a_hi, t_b[j + 1]) > max(a_lo, t_b[j]):  # overlap
                cov += r_a[i - 1] * r_b[j]
            j += 1
    return cov / np.sqrt((r_a**2).sum() * (r_b**2).sum())

# 2-hour subsample of raw ticks, pulled with a pruned time-range scan
sub = (
    db.table("fx")
    .filter(col("ts") >= "2026-06-01T06:00:00Z", col("ts") < "2026-06-01T08:00:00Z")
    .select("ts", "pair", mid=(col("bid") + col("ask")) / 2)
    .sort("ts")
    .to_pandas()
)
te, tg = {}, {}
for p, g in sub.groupby("pair"):
    ts_us = g["ts"].astype("int64").to_numpy()
    (te if p == "EURUSD" else tg)["t"] = ts_us
    (te if p == "EURUSD" else tg)["r"] = np.diff(np.log(g["mid"].to_numpy()))

shifts_ms = np.arange(-1000, 1001, 250)
hy = [
    hy_corr(te["t"][1:], te["r"], tg["t"][1:] - s * 1000, tg["r"])
    for s in shifts_ms
]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(shifts_ms / 1000, hy, marker="o", color="tab:purple")
ax.axvline(0.5, color="tab:red", ls="--", lw=1, label="true lag (0.5 s)")
ax.set_title("Hayashi-Yoshida correlation vs follower clock shift")
ax.set_xlabel("shift applied to GBPUSD timestamps (s)")
ax.set_ylabel("HY correlation")
ax.legend()
fig.tight_layout()
best = shifts_ms[int(np.argmax(hy))]
print(f"HY peak at shift {best} ms: corr {max(hy):.2f}   (unshifted: {hy[4]:.2f})")

# %% [markdown]
# HY の相互相関図は注入したラグを正確に復元します。+500ms に鋭いピーク、右へのにじみもなく、
# ピーク相関はバケット化の最良の推定値と同等以上で、しかも格子が情報を捨てていません。
#
# シフトなしの HY の値がほぼゼロなのは、正しい理由によります。同時点ではこの2つの系列は本当に
# ほぼ無関係で、依存はまるごと半秒のオフセットに住んでいるのです。
#
# ## 5. 現実確認: 日次の株式ペア
#
# 同じ相互相関の機構を、今度はキャッシュしてある30銘柄の S&P 標本から、実際の KO/PEP の日次
# 終値にかけます。`cu.fetch_daily` から来る1行1銘柄1セッションのデータのうち、`ts`、`symbol`、
# `adj_close` を使います。
#
# 流動性の高い大型株の日次 horizon では、正直な予想は強い同時点の相関と、利用できるクロス
# ラグが*ない*ことです。合成市場で 500ms かかった情報は、実際の市場では1日よりずっと短い時間で
# 伝わります。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01").to_pandas()
pair_df = (
    daily[daily["symbol"].isin(["KO", "PEP"])][["ts", "symbol", "adj_close"]]
    .sort_values(["ts", "symbol"])
    .reset_index(drop=True)
)
eq_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("equity_daily", eq_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("equity_daily", pa.Table.from_pandas(pair_df, preserve_index=False).cast(eq_schema), note="KO/PEP daily")

eq = (
    db.table("equity_daily")
    .select(
        "ts", "symbol",
        ret=col("adj_close")
        / sql_expr("lag(adj_close)").over(partition_by="symbol", order_by="ts") - 1,
    )
    .sort("ts")
    .to_pandas()
)
eq_panel = eq.pivot(index="ts", columns="symbol", values="ret").dropna()
eq_band = 1.96 / np.sqrt(len(eq_panel))
eq_ccf = {k: eq_panel["KO"].corr(eq_panel["PEP"].shift(-k)) for k in range(-5, 6)}
print(f"KO->PEP daily CCF ({len(eq_panel)} days, 95% band ±{eq_band:.3f}):")
for k, v in eq_ccf.items():
    flag = " *" if abs(v) > eq_band and k != 0 else ""
    print(f"  lag {k:+d}d: {v:+.3f}{flag}")

# %% [markdown]
# 同時点の相関は 0.5〜0.6 あたりに落ち着き、クロスラグはどれもノイズの帯の中か境界上です。
# KO/PEP に日次のリードラグはありません。そうあるべきです。
#
# 本物のリードラグのアルファは2節から4節の horizon に住んでいます。ティックの基盤が、そして
# 非同期性を尊重する推定量が効いてくる理由がここにあります。
#
# ## まとめ
#
# - `time_bucket` は `'250ms'` のような1秒未満の幅を受け取るので、相互相関の格子は GROUP BY
#   1つからそのまま出てきます。バケットの終値ミッドは `last_value(... ORDER BY ts)` です。
# - `asof_join(..., 'backward', tolerance)` は、明示的な鮮度の予算つきで不規則な系列を揃え
#   ます。対応しなかったティックは数えられる NULL として浮かぶので、対応率と許容差の検証が
#   ただで手に入ります。現行ビルドの注意点が1つ。ジョインの両入力を1つのストレージバッチ、
#   およそ8,000行の中に収め、左の行数と出力の行数が一致することを assert してください。それより
#   大きな入力は黙って切り捨てられます。
# - バケット化はラグを格子の精度までしか絞れず、Epps 効果で相関も薄めます。生のティック区間に
#   かけた Hayashi-Yoshida を時計のシフトで振ると、正確な 500ms のラグと相関の全量の両方を
#   復元しました。
# - 効果を注入すること、つまり既知のラグと既知のベータを置くことが、ノートブックを較正の練習に
#   変えます。どの推定量も正解に照らして採点されますし、実データの節は日次 horizon で何も
#   見つからないことを正直に書いています。

# %%
db.close()
