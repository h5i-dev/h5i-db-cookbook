# %% [markdown]
# # オプション: インプライドボラティリティのサーフェスをバージョン管理された評価として
#
# ボラティリティデスクのサーフェスは1つのオブジェクトではありません。評価の*系列*です。EOD の
# スナップショット、日中の再評価、訂正。
#
# チェーンのスナップショットをコミットとして保存すれば、その系列全体が手に入り、どれにも O(1)
# でアクセスできます。SQL の `h5i('chain', v)` は「T 時点で評価されていたサーフェス」であり、
# リスクも P&L 分解もモデル検証も、ずっとそれを尋ね続けます。
#
# テーブルの設計には真似する価値のある細部があります。評価時刻の `ts` も `expiry` も、
# **どちらも** `timestamp[us, UTC]` の列です。テナーは常に `expiry - ts` として*導出*され、
# 保存されません。だから時計が進んでも古くなりようがありません。
#
# このレシピで進めるのは次の4つです。
#
# 1. SPX 型のチェーンのスナップショットを5日ぶん、1日1コミットで保存する
# 2. ATM のターム構造と、25デルタのリスクリバーサルとバタフライを SQL で取り出す
# 3. サーフェスを描く
# 4. バージョニングで「昨日の引けにデスクが見ていたのは何か」に答える

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_options"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_option_chain` が返すのは日次のオプションチェーンのスナップショットです。1行が
# 1評価日1契約です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 評価時刻、セッションの引け |
# | `underlier` | `string` | 原資産。ここでは `SPX` |
# | `expiry` | `timestamp[us, tz=UTC]` | 契約の満期 |
# | `strike` | `float64` | 権利行使価格 |
# | `cp` | `string` | `C` はコール、`P` はプット |
# | `iv` | `float64` | インプライドボラティリティ |
# | `mid` | `float64` | オプションのミッド価格 |
# | `delta` | `float64` | ブラック・ショールズのデルタ |

# %%
chain = cu.make_option_chain(snapshots=5)
print(f"{chain.num_rows:,} rows x {chain.num_columns} columns, "
      f"{len(set(chain['ts'].to_pylist()))} mark dates")
chain.to_pandas().head()

# %% [markdown]
# ## 2. スナップショット1日がコミット1件
#
# `sort_key` は必須の時刻列から始まり、そのあと各スナップショットを満期と権利行使価格で並べ
# ます。サーフェスのクエリにとって自然な走査順です。
#
# 日ごとに追記していくことで、EOD の評価がそれぞれ自分のバージョンになり、`note` が監査証跡に
# なります。

# %%
db.create_table("chain", chain.schema, time_column="ts", sort_key=["ts", "expiry", "strike"])

chain_df = chain.to_pandas()
for day_ts, day_chain in chain_df.groupby("ts"):
    db.append(
        "chain",
        pa.Table.from_pandas(day_chain, schema=chain.schema, preserve_index=False),
        note=f"EOD marks {day_ts.date()}",
    )

[
    {k: v[k] for k in ("sequence", "op", "rows", "note") if k in v}
    for v in db.versions("chain")
]

# %% [markdown]
# ## 3. プット・コール・パリティからスポットを復元する
#
# チェーンにスポットの列はありませんし、要りません。
#
# この生成器のように金利がゼロなら、パリティは `C - P = S - K` を与えるので、どの権利行使価格
# でも `S = K + C - P` です。チェーン全体で平均すれば、評価日ごとのフォワードのスポットが気配の
# 丸め誤差の範囲で復元できます。以下では権利行使価格をマネーネスの軸に載せ替えるのに使います。

# %%
# Put-call parity as a self-join: the calls frame joined to the puts frame on
# (ts, expiry, strike). Holding each side in a variable is what makes the
# self-join readable.
calls = db.table("chain").filter(col("cp") == "C")
puts = db.table("chain").filter(col("cp") == "P")

SPOT = (
    calls.join(puts, on=["ts", "expiry", "strike"])
    .group_by(col("ts", relation="l").alias("ts"))
    .agg(
        spot=(
            col("strike", relation="l") + col("mid", relation="l") - col("mid", relation="r")
        ).mean()
    )
)

spot = SPOT.sort("ts").to_pandas()
spot

# %% [markdown]
# ## 4. 評価日ごとの ATM ターム構造
#
# `(ts, expiry)` の組ごとに、コールの権利行使価格をスポットからの距離で順位づけ、いちばん近い
# ものを残します。パリティのスポットの CTE に対するウィンドウ関数です。テナーは2つのタイム
# スタンプ列から pandas で導出します。

# %%
# The parity spot frame is reused here rather than re-written: join it back
# to the call chain, then rank strikes by moneyness distance.
atm = (
    calls.join(SPOT, on="ts")
    .select(
        ts=col("ts", relation="l"),
        expiry=col("expiry", relation="l"),
        atm_iv=col("iv", relation="l"),
        spot=col("spot", relation="r"),
        moneyness_gap=(col("strike", relation="l") / col("spot", relation="r") - 1).abs(),
    )
    .with_columns(
        rn=sql_expr("row_number()").over(
            partition_by=["ts", "expiry"], order_by="moneyness_gap"
        )
    )
    .filter(col("rn") == 1)
    .select("ts", "expiry", "atm_iv", "spot")
    .sort(["ts", "expiry"])
    .to_pandas()
)
atm["tenor_d"] = (atm["expiry"] - atm["ts"]).dt.days
atm.head(7)

# %%
fig, ax = plt.subplots(figsize=(9, 4))
dates = sorted(atm["ts"].unique())
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(dates)))
for d, c in zip(dates, colors):
    g = atm[atm["ts"] == d]
    ax.plot(g["tenor_d"], 100 * g["atm_iv"], marker="o", ms=4, lw=1.2,
            color=c, label=str(pd.Timestamp(d).date()))
ax.set_xscale("log")
ax.set_xticks([7, 14, 30, 60, 91, 182, 365])
ax.set_xticklabels(["7d", "14d", "30d", "60d", "91d", "6m", "1y"])
ax.set_title("ATM implied-vol term structure, by mark date")
ax.set_xlabel("tenor")
ax.set_ylabel("ATM IV (%)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 5. 25デルタのリスクリバーサルとバタフライ
#
# チェーンがデルタを持っているので、標準的なスキューの要約はウィンドウの順位づけ3つで済みます。
# +0.25 にいちばん近いコール、−0.25 にいちばん近いプット、そして ATM 用に 0.50 にいちばん近い
# コール。
#
# `RR = σ_25C − σ_25P` がスキューの向きを、`BF = ½(σ_25C + σ_25P) − σ_ATM` がウィングの凸性を
# 与えます。
#
# 短いテナーではデルタ空間の権利行使価格の格子が粗いので、「25Δ にいちばん近い」が 25Δ から
# そこそこ離れることがあります。実際の上場チェーンでも同じです。短期側はそれを踏まえて読んで
# ください。

# %%
# Three ranked CTEs joined pairwise: .join() is binary and aliases each side
# l/r, so a three-way join would nest into unreadable qualifiers. SQL wins.
rr_bf = db.sql(
    """
    WITH c25 AS (SELECT ts, expiry, iv,
                        row_number() OVER (PARTITION BY ts, expiry ORDER BY abs(delta - 0.25)) AS rn
                 FROM chain WHERE cp = 'C'),
         p25 AS (SELECT ts, expiry, iv,
                        row_number() OVER (PARTITION BY ts, expiry ORDER BY abs(delta + 0.25)) AS rn
                 FROM chain WHERE cp = 'P'),
         atm AS (SELECT ts, expiry, iv,
                        row_number() OVER (PARTITION BY ts, expiry ORDER BY abs(delta - 0.50)) AS rn
                 FROM chain WHERE cp = 'C')
    SELECT c.ts, c.expiry,
           c.iv - p.iv                  AS rr25,
           0.5 * (c.iv + p.iv) - a.iv   AS bf25,
           a.iv                         AS atm_iv
    FROM c25 c
    JOIN p25 p ON c.ts = p.ts AND c.expiry = p.expiry
    JOIN atm a ON c.ts = a.ts AND c.expiry = a.expiry
    WHERE c.rn = 1 AND p.rn = 1 AND a.rn = 1
    ORDER BY c.ts, c.expiry
    """
).to_pandas()
rr_bf["tenor_d"] = (rr_bf["expiry"] - rr_bf["ts"]).dt.days

latest_ts = rr_bf["ts"].max()
rr_bf[rr_bf["ts"] == latest_ts][["tenor_d", "atm_iv", "rr25", "bf25"]].round(4)

# %% [markdown]
# ## 6. サーフェス、マネーネス × テナー
#
# 1評価日、コールのみ。1つのスナップショットの中ではどの満期も同じ権利行使価格の格子を共有する
# ので、ピボットするときれいな長方形のサーフェスになりますし、パリティのスポットが権利行使価格を
# マネーネスに変換します。
#
# 描画は `pcolormesh` に知覚的に均等な連続カラーマップを合わせます。IV は大きさなので、色相は
# 1つ、明から暗へ。

# %%
day_spot = float(spot.loc[spot["ts"] == latest_ts, "spot"].iloc[0])
surf = chain_df[(chain_df["ts"] == latest_ts) & (chain_df["cp"] == "C")].copy()
surf["tenor_d"] = (surf["expiry"] - surf["ts"]).dt.days
surf["moneyness"] = surf["strike"] / day_spot
grid = surf.pivot_table(index="tenor_d", columns="moneyness", values="iv")

fig, ax = plt.subplots(figsize=(9, 4.5))
pcm = ax.pcolormesh(grid.columns, grid.index, 100 * grid.values,
                    cmap="viridis", shading="nearest")
ax.set_yscale("log")
ax.set_yticks([7, 14, 30, 60, 91, 182, 365])
ax.set_yticklabels(["7d", "14d", "30d", "60d", "91d", "6m", "1y"])
ax.set_title(f"Implied-vol surface, marks of {pd.Timestamp(latest_ts).date()}")
ax.set_xlabel("moneyness (K / S)")
ax.set_ylabel("tenor")
fig.colorbar(pcm, ax=ax, label="IV (%)")
fig.tight_layout()

# %% [markdown]
# ## 7. 1週間のスマイルの推移
#
# 5つのスナップショットそれぞれの30日スマイルを重ねます。
#
# チェーンは毎日、新しい30日満期を建て直します。だから*導出した*テナーで選びます。`expiry` を
# テナーのラベルではなくタイムスタンプとして持っていることの、もう1つの見返りです。ラベルは
# ずれていきます。

# %%
smiles = chain_df[chain_df["cp"] == "C"].copy()
smiles["tenor_d"] = (smiles["expiry"] - smiles["ts"]).dt.days
smiles = smiles[smiles["tenor_d"] == 30].merge(spot, on="ts")
smiles["moneyness"] = smiles["strike"] / smiles["spot"]

fig, ax = plt.subplots(figsize=(9, 4))
for d, c in zip(dates, colors):
    g = smiles[smiles["ts"] == d].sort_values("moneyness")
    ax.plot(g["moneyness"], 100 * g["iv"], lw=1.2, color=c, label=str(pd.Timestamp(d).date()))
ax.set_title("30-day smile, five consecutive mark dates")
ax.set_xlabel("moneyness (K / S)")
ax.set_ylabel("IV (%)")
ax.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# ## 8. as-of のクエリ: T 時点で評価されていたサーフェス
#
# スナップショットがそれぞれコミットなので、「2日目の引けにデスクが見ていたサーフェス」は
# フィルタではありません。*バージョン*です。
#
# `h5i('chain', 2)` は2日目のコミット直後のテーブルをそのまま返しますし、3日目から5日目の評価は
# その視界に存在しません。
#
# 実時刻版の `read(as_of=...)` は同じ問いにコミット時刻で答えます。監査人が渡してくるのは
# そちらです。

# %%
db.sql(
    """
    SELECT 'as of v2 (day-2 close)' AS view, count(*) AS rows, max(ts) AS latest_mark
    FROM h5i('chain', 2)
    UNION ALL
    SELECT 'head', count(*), max(ts) FROM chain
    """
).to_pandas()

# %%
v3_commit_iso = pd.Timestamp(db.versions("chain")[3]["committed_at_ns"], unit="ns", tz="UTC").isoformat()
as_of_v3 = db.read("chain", as_of=v3_commit_iso)
print(f"read(as_of='{v3_commit_iso}') -> {as_of_v3.num_rows} rows,",
      f"latest mark {pd.Timestamp(pa.compute.max(as_of_v3['ts']).as_py()).date()}")

# %% [markdown]
# ## 9. 日中の再評価も、ただのバージョンの追加
#
# 5日目の後半、デスクはリスクオフの再評価としてボラティリティを6%引き上げ、引けの30分後に新しい
# 評価をコミットします。
#
# 再評価のタイムスタンプが EOD のものより後なので、append の意味論がこれを許します。これで5日目
# の評価が2世代、先頭の中で共存します。デスクの2つの定番の問いは、それぞれ1行になります。
#
# - *現在の*サーフェスは契約ごとの最新の評価、つまりウィンドウによる重複排除です。
# - *引け時点で評価された*サーフェスは `h5i('chain', 5)` で、永久に凍結されています。

# %%
day5 = chain_df[chain_df["ts"] == latest_ts].copy()
remark = day5.assign(ts=day5["ts"] + pd.Timedelta(minutes=30), iv=(day5["iv"] * 1.06).round(4))
db.append("chain", pa.Table.from_pandas(remark, schema=chain.schema, preserve_index=False),
          note="intraday re-mark day 5 (+6% vol)")

day5_us = int(pd.Timestamp(latest_ts).value // 1000)
db.sql(
    f"""
    WITH latest AS (
        SELECT iv, row_number() OVER (PARTITION BY expiry, strike, cp ORDER BY ts DESC) AS rn
        FROM chain
        WHERE ts >= to_timestamp_micros({day5_us})
    )
    SELECT 'current (post re-mark)' AS view, avg(iv) AS avg_iv
    FROM latest WHERE rn = 1
    UNION ALL
    SELECT 'as marked at day-5 close', avg(iv)
    FROM h5i('chain', 5) WHERE ts >= to_timestamp_micros({day5_us})
    """
).to_pandas().round(4)

# %% [markdown]
# ## まとめ
#
# - `expiry` は、評価時刻の `ts` の隣に `timestamp[us, UTC]` の列としてモデル化してください。
#   テナーもマネーネスも「今日どの上場満期がおよそ30日か」も、すべて導出になり、古くなりません。
# - チェーンのスナップショット1つにつきコミット1件にすると、`versions()` がデスクの評価履歴に
#   なり、`h5i('chain', v)` と `read(as_of=...)` が「T 時点で評価されたサーフェス」の即答に
#   なります。バイテンポラルの記帳テーブルは要りません。
# - サーフェスの分析はウィンドウ関数が SQL で片付けます。ATM にはスポットからの距離の順位、
#   リスクリバーサルとバタフライには 25Δ からの距離の順位、契約ごとの最新評価には
#   `row_number ... ORDER BY ts DESC` の重複排除です。
# - プット・コール・パリティがチェーン自身からスポットを復元するので、サーフェスのクエリから
#   スポットのフィードへの結合が1つ消えます。
# - 日中の再評価はごく普通の append です。引けの評価はそのバージョンで凍結されたまま、先頭は
#   最新の世代を映します。

# %%
db.close()
