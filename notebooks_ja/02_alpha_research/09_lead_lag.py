# %% [markdown]
# # リードラグの発見: 不規則なティックに対する相互相関
#
# 先に動くのはどちらか。関連する銘柄どうし（指数と先物、ADR と本国上場、相関する FX クロス）の
# リードラグ分析には、機械的な問題が1つ付きまといます。ティックは不規則に、しかも*非同期に*
# 届き、素朴な対処――バケット分け、直近観測値の前方持ち越し――はどれも答えを歪めます（Epps 効果）。
# ここでは、*既知のリードを仕込んだ*ペアを題材に、道具立てを順に見ていきます。
#
# 1. EURUSD に500ミリ秒遅れて追随する合成 GBPUSD を作る
# 2. バケットして相関を取る。`time_bucket` による250ミリ秒格子上の古典的な CCF です
# 3. `asof_join` によるティックレベルの揃えと、陳腐化許容差のスイープ
# 4. Hayashi-Yoshida 推定量。格子をまったく必要とせず、時刻シフトでスイープすればリサンプル
#    なしにラグを特定できます
# 5. 実データによる現実確認（日次の株式ペア）

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_leadlag"), create=True)

# %% [markdown]
# ## 1. 既知の500ミリ秒のリードを仕込む
#
# `make_fx_ticks` が生成するペアは*独立*なので、そこにリードラグを見つけたらそれはノイズです。
# そこで EURUSD をリーダーとして使い、GBPUSD は自分たちで構築します。各 GBPUSD のティック時刻
# `t` において、対数ミッドを `beta * eur_logmid(t - 500ms) + idio(t)` とします。リーダーの経路を
# 0.5秒古い状態で backward asof 標本し、同程度の大きさの個別ランダムウォークを足したものです。
# 正解はこうです。EURUSD がちょうど500ミリ秒先行し、ベータは0.9、それ以外に2つをつなぐものは
# ありません。

# %%
LAG_US = 500_000
BETA = 0.9

eur = cu.make_fx_ticks(pairs=["EURUSD"], hours=24, ticks_per_hour=20_000).to_pandas()
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
# ## 2. バケットした CCF: 250ミリ秒の `time_bucket`
#
# 古典的なやり方はこうです。両方の系列を細かい共通格子にリサンプルし、リターンを計算し、
# リード／ラグのオフセットごとに相関を取る。h5i-db の `time_bucket` は秒未満の幅（`'250ms'`）を
# 取れるので、格子への縮約はデータベースの中で起きます。GROUP BY 1つで96万件のティックが
# バケットごとの終値ミッドになり、整列済みストレージの上をストリーミングで流れます。pandas 側は
# 格子を前方補完してずらすだけです。

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
# CCF は方向を的中させ、ラグも絞り込みます。重みは +0.5秒 と +0.75秒 に乗ります。仕込んだ
# 500ミリ秒に、最終ティック標本の陳腐化（ティックは約180ミリ秒ごとに届くので、バケットの
# 「終値ミッド」自体がわずかに古い）が加わって、効果が1バケットぶん右へにじんだ形です。ゼロには
# 何もなく、負のラグにも何もありません。GBPUSD が先行することは一度もない。離散化のトレードオフの
# 縮図です。格子を粗くすれば効果はラグゼロの1本に積み上がって方向が*隠れ*、細かくすればバケットが
# 薄くなって持ち越されたゼロが支配します（Epps 効果）。
#
# ## 3. ティックレベルの揃え: `asof_join` と陳腐化の許容差
#
# 格子なしで何かを推定する前に、主力になる操作は、不規則な系列をもう一方に揃えることです。
# GBPUSD の各ティックに対して直近の EURUSD ティックを取る、つまり
# `asof_join(..., 'backward', tolerance)` です。許容差（生のマイクロ秒）はデータ品質のつまみに
# なります。リーダーの気配がどれだけ古くても受け入れるか、という問いです。20分の窓で検証し、
# 許容差をスイープしながら一致率と、揃えたティック間リターンの相関を追います。
#
# 窓を切って抜き出せば検証は速く、数字も検分しやすくなります。ジョインが左側のティック1件に
# つき1行を返したことを表明し、標本を `pandas.merge_asof` と照合します。ASOF のパイプライン
# なら標準的な衛生管理です。

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
# リーダーのティックは毎秒約5.5件なので、50ミリ秒の許容差ではフォロワーのティックの4分の1しか
# 残りません（残りは十分に新しい相手がなく NULL で返ります。黙って古いまま使われるのではなく、
# 目に見える形で）。500ミリ秒あたりになれば、ほぼすべてが一致します。相関の列は注意深く読むと
# 報われます。許容差を緩めるほど相関が*下がる*のです。一致が古くなるからではありません。生き残る
# ティックのペアが密になり、連続する揃えられたリターンがどんどん短い区間をまたぐようになるからで、
# 500ミリ秒のラグより短い区間は、相手側の同時点とほとんど重なりません（相関 → 0.07）。許容差が
# きついときは、間引かれた系列が約1秒の区間をまたぎ、ラグをまたぐので相関が戻ってきます。
# どちらの数字も「その」相関ではありません。非同期でラグのある系列では、測る区間の長さが測れる
# ものを決めます。表になった Epps 効果であり、区間を明示的に尊重する推定量が要る理由でもあります。
#
# ## 4. Hayashi-Yoshida: 格子なしで、ラグのつまみ付き
#
# HY 推定量は、時間区間が*重なる*ティック区間リターンのペアすべてについて `r_i * s_j` を合計
# します。非同期データに対して不偏で、リサンプルも持ち越しのゼロも不要です。フォロワーの時計に
# 時刻シフト `delta` をかけてスイープすれば、格子なしの相互相関図になります
# （Hoffmann-Rosenbaum-Yoshida）。HY の相関を最大化するシフトがラグの推定値です。2ポインタで
# 1回なめれば O(n + m) で済みます。

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
# HY の相互相関図は、仕込んだラグを正確に復元します。+500ミリ秒に鋭いピークが立ち、右への
# にじみはなく、ピークの相関はバケット版の最良値と同等以上です。しかも格子が情報を捨てていま
# せん。シフトなしの HY 値が約0なのは、正しい理由からです。同時点で見れば、この2つの系列は
# 本当にほとんど無関係で、依存は丸ごと0.5秒のオフセットに住んでいます。
#
# ## 5. 現実確認: 日次の株式ペア
#
# 同じ CCF の仕掛けを、実際の KO/PEP の日次終値に当てます（キャッシュ済みの30銘柄 S&P
# サンプルから）。流動性の高い大型株の日次時間軸なら、誠実な期待は、強い同時点の相関と、
# 利用できるクロスラグは*ない*ことです。合成市場で500ミリ秒かかった情報の伝達は、実際の市場では
# 1日よりはるかに短い時間で終わります。

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
# 同時点の相関は約0.5〜0.6、クロスラグはどれもノイズの帯の上か内側です。KO/PEP に日次の
# リードラグはありません。あるべき姿です。本物のリードラグのアルファはセクション2〜4の時間軸に
# 住んでいて、だからこそティックの基盤（と非同期性を尊重する推定量）が効いてきます。
#
# ## まとめ
#
# - `time_bucket` は秒未満の幅（`'250ms'`）を受け取るので、CCF の格子は GROUP BY 1つから
#   そのまま出てきます。バケットごとの終値ミッドは `last_value(... ORDER BY ts)` です。
# - `asof_join(..., 'backward', tolerance)` は不規則な系列を、陳腐化の予算を明示したうえで
#   揃えます。一致しなかったティックは数えられる NULL として現れるので、一致率と許容差の検証が
#   ただで手に入ります。（現ビルドでの注意点: ジョインの両側の入力を1つのストレージバッチ
#   （約8千行）に収め、左側1行につき出力1行であることを表明してください。これより大きい入力は
#   黙って切り詰められます。）
# - バケット分けはラグを格子の精度までしか特定できず、しかも相関を薄めます（Epps）。生の
#   ティック区間に対する Hayashi-Yoshida を時計シフトでスイープすると、正確な500ミリ秒のラグも
#   相関の全量も取り戻せました。
# - 効果を仕込むこと（既知のラグ、既知のベータ）が、このノートブックを較正の演習に変えます。
#   どの推定量も正解に対して採点され、実データの節は日次時間軸で何も見つからないことを正直に
#   述べています。

# %%
db.close()
