# %% [markdown]
# # オプション: インプライドボラティリティ曲面をバージョン付きの評価として持つ
#
# ボラティリティデスクの曲面は1つのオブジェクトではなく、評価の*連なり*です。EOD の
# スナップショット、日中の再評価、訂正。チェーンのスナップショットを h5i-db のコミットとして
# 保存すれば、その連なり全体が手に入り、どの1つにも O(1) でアクセスできます。SQL の
# `h5i('chain', v)` は「時刻Tに評価された曲面」であり、リスク、損益分解、モデル検証が繰り返し
# 尋ねてくるのはまさにこの問いです。
#
# テーブルの構成には真似する価値のある細部があります。マーク時刻の `ts` と `expiry` の
# **両方**を `timestamp[us, UTC]` 列にすることです。残存期間は常に*導出*し（`expiry - ts`）、
# 保存しません。だから時計が進んでも古くなりようがありません。
#
# ここでは SPX 型のチェーンのスナップショットを5営業日ぶん保存し、ATM のターム構造と25デルタの
# リスクリバーサル／バタフライを SQL で取り出し、曲面を描き、最後にバージョン管理を使って
# 「昨日の引けにデスクが見ていたものは何か」に答えます。

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_options"), create=True)

chain = cu.make_option_chain(snapshots=5)  # ts, underlier, expiry, strike, cp, iv, mid, delta
chain.schema

# %% [markdown]
# ## 1. スナップショット1日＝コミット1回
#
# `sort_key` は時刻列から始め（必須です）、そのあと各スナップショットを満期と行使価格で
# 並べます。曲面のクエリにとって自然なスキャン順です。1日ずつ append すれば、EOD の評価が
# それぞれ独立したバージョンになり、`note` がそのまま監査証跡になります。

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
# ## 2. プットコールパリティから原資産価格を復元する
#
# チェーンに原資産価格の列はありません。そして必要もありません。金利ゼロなら（この生成器が
# そうです）パリティから `C - P = S - K`、つまりどの行使価格でも `S = K + C - P` です。
# チェーン全体で平均を取れば、マーク日ごとのフォワード／原資産価格が、気配の丸め誤差の範囲で
# 復元できます。以下ではこれを再利用して、行使価格をマネーネスの軸に載せます。

# %%
spot = db.sql(
    """
    SELECT c.ts, avg(c.strike + c.mid - p.mid) AS spot
    FROM chain c
    JOIN chain p ON c.ts = p.ts AND c.expiry = p.expiry AND c.strike = p.strike
    WHERE c.cp = 'C' AND p.cp = 'P'
    GROUP BY c.ts
    ORDER BY c.ts
    """
).to_pandas()
spot

# %% [markdown]
# ## 3. マーク日ごとの ATM ターム構造
#
# `(ts, expiry)` の組ごとに、コールの行使価格を原資産価格からの距離で順位付けし、いちばん近い
# ものを残します。パリティで求めた原資産価格の CTE に対するウィンドウ関数です。残存期間は2つの
# タイムスタンプ列から pandas で導出します。

# %%
atm = db.sql(
    """
    WITH spot AS (
        SELECT c.ts, avg(c.strike + c.mid - p.mid) AS spot
        FROM chain c
        JOIN chain p ON c.ts = p.ts AND c.expiry = p.expiry AND c.strike = p.strike
        WHERE c.cp = 'C' AND p.cp = 'P'
        GROUP BY c.ts
    ),
    ranked AS (
        SELECT c.ts, c.expiry, c.iv, s.spot,
               row_number() OVER (PARTITION BY c.ts, c.expiry
                                  ORDER BY abs(c.strike / s.spot - 1)) AS rn
        FROM chain c
        JOIN spot s ON c.ts = s.ts
        WHERE c.cp = 'C'
    )
    SELECT ts, expiry, iv AS atm_iv, spot
    FROM ranked WHERE rn = 1
    ORDER BY ts, expiry
    """
).to_pandas()
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
# ## 4. 25デルタのリスクリバーサルとバタフライ
#
# チェーンがデルタを保存しているので、標準的なスキューの要約はウィンドウの順位付け3つで済みます。
# +0.25 にいちばん近いコール、−0.25 にいちばん近いプット、0.50 にいちばん近いコール（ATM）です。
# `RR = σ_25C − σ_25P` がスキューの向き、`BF = ½(σ_25C + σ_25P) − σ_ATM` がウィングの凸性です。
# 残存の短い側では、行使価格の格子がデルタ空間で粗くなります。実際の上場チェーンと同じく
# 「25Δ にいちばん近い」が 25Δ からかなり離れうるので、短期側はそのつもりで読んでください。

# %%
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
# ## 5. 曲面: マネーネス × 残存期間
#
# マーク日1つ、コールのみです。1つのスナップショットの中ではどの満期も同じ行使価格の格子を
# 共有するので、ピボットすればきれいな長方形の曲面になります。行使価格からマネーネスへの変換には
# パリティで求めた原資産価格を使います。描画は `pcolormesh` に知覚的に一様な連続カラーマップで。
# IV は大きさの量なので、色相は1つ、明から暗へ取ります。

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
# ## 6. 1週間のスマイルの推移
#
# 5つのスナップショットそれぞれの30日スマイルを重ね描きします。チェーンは毎日、新しい30日満期を
# 上場し直すので、*導出した*残存期間で選びます。残存期間のラベルを保存せずに `expiry` を
# タイムスタンプのまま持っておくことの、もう1つの見返りです。

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
# ## 7. as-of のクエリ: 時刻Tに評価された曲面
#
# 各スナップショットがコミットなので、「2日目の引けにデスクが見ていた曲面」はフィルタではなく
# *バージョン*です。`h5i('chain', 2)` は2日目のコミット直後の状態そのままのテーブルを返し、
# 3〜5日目の評価はその視界に存在しません。実時刻版の `read(as_of=...)` は同じ問いにコミット
# 時刻で答えます。監査担当が差し出してくるのはこちらでしょう。

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
# ## 8. 日中の再評価も、ただのバージョンが増えるだけ
#
# 5日目の遅い時間、デスクがリスクオフの再評価としてボラを6%引き上げ、引けの30分後に新しい評価を
# コミットします。append の意味論はこれを許します（再評価のタイムスタンプは EOD より後だから
# です）。こうして先頭には5日目の評価が2世代共存します。デスクが日常的に抱える2つの問いは、
# それぞれ1行になります。
#
# - *現在の*曲面 → 契約ごとの最新の評価（ウィンドウによる重複排除）
# - *引け時点の評価* → `h5i('chain', 5)`。永久に凍結されています

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
# - `expiry` は、マーク時刻の `ts` の隣に `timestamp[us, UTC]` の列としてモデル化してください。
#   残存期間もマネーネスも「今日どの上場満期が約30日か」も、すべて導出されるので古くなりません。
# - チェーンのスナップショット1つにつき1コミットにすれば、`versions()` がデスクの評価履歴に、
#   `h5i('chain', v)` と `read(as_of=...)` が「時刻Tに評価された曲面」の即時クエリになります。
#   バイテンポラルの管理テーブルは要りません。
# - 曲面の分析はウィンドウ関数が SQL の中で片付けます。ATM には原資産価格に最も近い順位、
#   リスクリバーサルとバタフライには 25Δ に最も近い順位、契約ごとの最新評価には
#   `row_number ... ORDER BY ts DESC` による重複排除です。
# - プットコールパリティがチェーン自体から原資産価格を復元します。曲面のクエリのたびに、原資産の
#   フィードへのジョインを1つ減らせます。
# - 日中の再評価はただの append です。引けの評価は自分のバージョンで凍結されたまま、先頭は最新の
#   世代を映します。

# %%
db.close()
