# %% [markdown]
# # ティックデータのクリーニング: 見つけて、直し方をプレビューして、証跡を残す
#
# ベンダーの生ティックファイルは汚れて届きます。10倍ずれた入力ミス、フィード障害による
# ゼロ価格、パケット再送による重複ブロック、時間外のゴミ。危ないのは検出のほうではなく
# *修正*のほうです。ティックをその場で UPDATE するスクリプトを使うと、何がいつ変わったのか、
# どう戻せばいいのかを説明できなくなります。h5i-db の答えが plan/apply の流れです。変更は
# プレビューできるプラン（行数、変更前後のサンプル）としてステージングされ、注記付きの
# アトミックなコミットとして適用され、`restore()` で戻せます。このレシピでは2日ぶんの約定を
# 汚し、欠陥の種類ごとに SQL で見つけ、削除プランと置換プランで修復し、最後にテーブルを
# 締めて、レビューを経ない直接の変更をポリシー違反にします。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, sql_expr, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_cleaning"), create=True)

# %% [markdown]
# ## 1. 汚いベンダー納品を作る
#
# きれいな合成ティックを2日ぶん用意してから、固定シードで4種類の欠陥を仕込みます。15分の
# 窓の中に10倍の入力ミスとゼロ価格、同じ窓の中にまったく同一の約定ブロック（パケット再送）、
# そして22:00 UTC の時間外の約定のかたまり（立会は20:00に終わります）です。

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=2, trades_per_day=20_000, seed=7).to_pandas()
rng = np.random.default_rng(99)

DAY1 = pd.Timestamp("2026-06-01", tz="UTC")
W0 = DAY1 + pd.Timedelta(hours=15)            # polluted window: 15:00-15:15 UTC
W1 = DAY1 + pd.Timedelta(minutes=15) + pd.Timedelta(hours=15)

in_win = trades.index[(trades["ts"] >= W0) & (trades["ts"] < W1)].to_numpy()
ff = rng.choice(in_win, 25, replace=False)                                # fat fingers
zr = rng.choice(np.setdiff1d(in_win, ff), 6, replace=False)               # zero prices
dp = rng.choice(np.setdiff1d(in_win, np.concatenate([ff, zr])), 30, replace=False)

trades.loc[ff, "price"] = (trades.loc[ff, "price"] * 10).round(2)
trades.loc[zr, "price"] = 0.0
dup_block = trades.loc[dp].copy()                                         # replayed packet

day1_last = trades[trades["ts"] < DAY1 + pd.Timedelta(hours=20)].groupby("symbol")["price"].last()
ooh_syms = rng.choice(["AAPL", "MSFT", "NVDA"], 40)
after_hours = pd.DataFrame(
    {
        "ts": DAY1 + pd.Timedelta(hours=22, minutes=5)
        + pd.to_timedelta(np.sort(rng.integers(0, 45 * 60, 40)), unit="s"),
        "symbol": ooh_syms,
        "price": (day1_last.loc[ooh_syms].to_numpy() * (1 + rng.normal(0, 1e-3, 40))).round(2),
        "size": (rng.lognormal(4, 1, 40) // 100 * 100 + 100).astype("int64"),
        "exchange": rng.choice(["NYSE", "ARCA"], 40),
        "side": rng.choice(["B", "S"], 40),
    }
)

dirty = (
    pd.concat([trades, dup_block, after_hours])
    .sort_values(["ts", "symbol"], kind="stable")
    .reset_index(drop=True)
)
print(f"{len(dirty):,} rows including {len(dup_block)} duplicates and {len(after_hours)} after-hours prints")

# %% [markdown]
# 1日ずつ、ベンダーファイル1つにつき1コミットで取り込みます。コミットの境界*こそが*出所の
# 境界であり、注記がどのバージョンがどのファイル由来かを語ります。

# %%
TRADE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)
db.create_table("trades", TRADE_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
for day, label in [("2026-06-01", "day 1"), ("2026-06-02", "day 2")]:
    chunk = dirty[dirty["ts"].dt.date == pd.Timestamp(day).date()]
    db.append("trades", pa.Table.from_pandas(chunk, preserve_index=False).cast(TRADE_SCHEMA),
              note=f"vendor file {day}")
V_RAW = db.versions("trades")[-1]["sequence"]  # the as-delivered state, for auditing later
print("as-delivered head version:", V_RAW)

# %% [markdown]
# ## 2. 検出は、欠陥の種類ごとに1クエリ
#
# ゼロ価格はただの述語です。入力ミスは、頑健な局所基準――`approx_percentile_cont` に
# よる5分間の銘柄別中央値――と比べると浮き上がります。外れ値自身に引きずられる平均では、
# こうはいきません。

# %%
(
    db.table("trades")
    .filter(col("price") <= 0)
    .group_by("symbol")
    .agg(zero_price_prints=count_star())
    .to_pandas()
)

# %%
W5 = time_bucket("5m", col("ts"))

ref = (
    db.table("trades")
    .filter(col("price") > 0)
    .group_by(W5.alias("w"), "symbol")
    .agg(med=sql_expr("approx_percentile_cont(price, 0.5)"))
)

px, ref_med = col("price", relation="l"), col("med", relation="r")

fat = (
    db.table("trades")
    .with_columns(w=W5)
    .join(ref, on=["w", "symbol"])
    .filter(px > 3 * ref_med)
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        price=px,
        local_median=ref_med.round(2),
        ratio=(px / ref_med).round(1),
    )
    .sort("ts")
    .to_pandas()
)
print(f"{len(fat)} suspected fat fingers, ratios ~{fat['ratio'].min()}-{fat['ratio'].max()}x")
fat.head(5)

# %% [markdown]
# 再送されたパケットは*完全に*重複した行として現れます。時間外のゴミは、13:30〜20:00 UTC の
# 立会に対する `EXTRACT(hour ...)` の述語です。

# %%
(
    db.table("trades")
    .group_by("ts", "symbol", "price", "size", "exchange", "side")
    .count("n")
    .filter(col("n") > 1)  # the HAVING, one level up
    .select(dup_groups=count_star(), excess_rows=(col("n") - 1).sum())
    .to_pandas()
)

# %%
HOUR = sql_expr("EXTRACT(hour FROM ts)")
AFTER_HOURS = (HOUR >= 20) | (HOUR < 13)

(
    db.table("trades")
    .filter(AFTER_HOURS)
    .select(first_print=col("ts").min(), last_print=col("ts").max(), n=count_star())
    .to_pandas()
)

# %% [markdown]
# ## 3. 修正1 — 時間外のブロックを、プレビュー付きで削除する
#
# `plan_delete_range` は、テーブルに触れないまま変更をステージングします。範囲は時刻列の
# 単位（ここではマイクロ秒）で表した生の `int64` の窓です。要約と `before_sample` が、何が
# 消えることになるのかを正確に見せます。*何かが変わる前に、被害範囲を確認してください*。

# %%
OOH0 = int(pd.Timestamp("2026-06-01 22:00", tz="UTC").value // 1_000)
OOH1 = int(pd.Timestamp("2026-06-01 23:00", tz="UTC").value // 1_000)

plan = db.plan_delete_range("trades", OOH0, OOH1, note="drop after-hours junk 2026-06-01")
plan.summary

# %%
plan.before_sample.to_pandas().head(4)

# %%
plan.apply()

# %% [markdown]
# ## 4. 修正2 — 汚れた窓を、修復したデータで置き換える
#
# 15:00〜15:15 の窓については、削除ではなく*訂正された*約定が欲しいところです。入力ミスは
# 10で割り戻し、ゼロ価格は落とし、重複はまとめます。窓を取り出して pandas で修復し、修復
# 済みの行で `plan_replace_range` をステージングします。変更前後のサンプルがあるので、訂正を
# 1行ずつレビューできます。

# %%
W0_US, W1_US = int(W0.value // 1_000), int(W1.value // 1_000)

win = (
    db.table("trades")
    .filter(col("ts") >= W0.isoformat(), col("ts") < W1.isoformat())
    .sort(["ts", "symbol"])
    .to_pandas()
)

med = win[win["price"] > 0].groupby("symbol")["price"].median()
ratio = win["price"] / win["symbol"].map(med)
win.loc[ratio > 3, "price"] = (win.loc[ratio > 3, "price"] / 10).round(2)
repaired = (
    win[win["price"] > 0]
    .drop_duplicates()
    .sort_values(["ts", "symbol"], kind="stable")
)
print(f"window: {len(win)} rows as delivered -> {len(repaired)} repaired")

plan = db.plan_replace_range(
    "trades", W0_US, W1_US,
    data=pa.Table.from_pandas(repaired, preserve_index=False).cast(TRADE_SCHEMA),
    note="repair 15:00-15:15 window: /10 fat fingers, drop zeros+dups",
)
plan.summary

# %%
pd.concat(
    [
        plan.before_sample.to_pandas().head(3).assign(view="before"),
        plan.after_sample.to_pandas().head(3).assign(view="after"),
    ]
)

# %%
plan.apply()
V_CLEAN = db.versions("trades")[-1]["sequence"]

# All three detection queries now come back empty:
(
    db.table("trades")
    .select(
        zeros=when(col("price") <= 0).then(lit(1)).otherwise(lit(0)).sum(),
        after_hours=when(AFTER_HOURS).then(lit(1)).otherwise(lit(0)).sum(),
    )
    .to_pandas()
)

# %% [markdown]
# ## 5. 監査証跡: 訂正はすべてバージョンになる
#
# `versions()` は変更履歴のように読めます。どのベンダーファイルがいつ届き、何が削除され、
# 何が置き換えられ、それぞれに注記が付いています。納品時のバージョンも変わらず参照できるので、
# 「結局どこを変えたのか」は、同じテーブルの2つのバージョン間の SQL ジョインで済みます。
# 発掘作業は要りません。

# %%
[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("trades")]

# %%
now_px, was_px = col("price", relation="l"), col("price", relation="r")

(
    db.table("trades")  # head: repaired
    .join(db.table("trades", version=V_RAW), on=["ts", "symbol", "size", "exchange", "side"])
    .filter(now_px != was_px)
    .select(
        prints_repriced=count_star(),
        max_correction_ratio=(was_px / now_px).max().round(1),
    )
    .to_pandas()
)

# %%
import matplotlib.pyplot as plt

def aapl_window(version=None):
    return (
        db.table("trades", version=version)
        .filter(
            col("symbol") == "AAPL",
            col("ts") >= W0.isoformat(),
            col("ts") < W1.isoformat(),
        )
        .select("ts", "price")
        .to_pandas()
    )


raw_win, clean_win = aapl_window(version=V_RAW), aapl_window()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(raw_win["ts"], raw_win["price"], "x", ms=4, color="crimson", alpha=0.6,
        label=f"as delivered (v{V_RAW})")
ax.plot(clean_win["ts"], clean_win["price"], ".", ms=3, color="tab:blue",
        label=f"repaired (v{V_CLEAN})")
ax.set_yscale("log")
ax.set_title("AAPL, polluted window: as-delivered vs repaired")
ax.set_xlabel("time")
ax.set_ylabel("price (USD, log)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. 消しすぎた? `restore()` が取り消しボタン
#
# 勇み足の2周目が、実は正当なボラティリティだった窓を削除してしまったとします。削除もただの
# バージョンなので、巻き戻しは `restore()` です。これは先頭を*前へ*進めて、古い中身を持つ
# 新しいバージョンにします。失敗そのものも含めて、何も失われません。

# %%
bad0 = int(pd.Timestamp("2026-06-02 14:00", tz="UTC").value // 1_000)
bad1 = int(pd.Timestamp("2026-06-02 14:10", tz="UTC").value // 1_000)
oops = db.plan_delete_range("trades", bad0, bad1, note="overzealous: deleting real volatility")
oops.apply()
n_rows = lambda: db.table("trades").select(count_star().alias("n")).to_pandas().n[0]
print("rows after over-delete:", n_rows())

db.restore("trades", V_CLEAN)
print("rows after restore:   ", n_rows())
[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("trades")[-3:]]

# %% [markdown]
# ## 7. テーブルを締める: プランだけを通す
#
# 共有のリサーチ用データベースでは、レビューを経ない破壊的な書き込みを*誰にも*――新人にも、
# cron ジョブにも、LLM エージェントにも――走らせたくないでしょう。`set_policy` は直接の変更
# 経路にゲートをかけます。plan/apply の流れ（プレビュー可能で注記付きのアトミックなコミットを
# 強制します）はそのまま動きます。ゲートされた呼び出しは、機械可読なコードと復旧のヒントを
# 添えて `PolicyError` を上げます。

# %%
print("policy before:", db.policy())
db.set_policy(direct_write=False, direct_delete=False)

try:
    db.write("trades", db.read("trades"))
except h5i_db.PolicyError as e:
    print(f"\nblocked: code={e.code}\nhint: {e.hint}")

# %% [markdown]
# 保留中のプランも一級のオブジェクトです。`list_plans()` に現れて同僚がレビューでき、プロセスを
# 再起動しても生き残り、適用されないまま7日たつと期限切れになります（生きているプランが参照する
# セグメントは vacuum が回収しません）。

# %%
pending = db.plan_delete_range("trades", bad0, bad1, note="for review: is 14:00-14:10 real?")
print([(p.plan_id, p.raw["note"]) for p in db.list_plans("trades")])
pending.discard()
print("after discard:", db.list_plans("trades"))

# %% [markdown]
# ## まとめ
#
# - 欠陥の種類はどれも SQL 1クエリで見つかります。ゼロと立会時間には述語、入力ミスには
#   `approx_percentile_cont` の中央値、再送パケットには GROUP BY と HAVING です。
# - `plan_delete_range` と `plan_replace_range` は、訂正を*起きる前に*レビュー可能に
#   します。要約と変更前後のサンプルがあり、そのあとアトミックで注記付きのコミットとして
#   着地します。範囲は時刻列の生のマイクロ秒です。
# - バージョンの連鎖が監査証跡です。納品時のデータはクエリできるまま残り（`h5i('trades', v)`）、
#   差分は SQL のジョインで取れ、`restore()` は消しすぎを、それが起きた記録を消さずに
#   取り消します。
# - `set_policy(direct_write=False, direct_delete=False)` は「本番テーブルを直接いじらないで
#   ください」というお願いを `PolicyError` に変えます。

# %%
db.close()
