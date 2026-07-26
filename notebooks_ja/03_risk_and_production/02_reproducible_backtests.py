# %% [markdown]
# # 再現できるバックテスト: コードだけでなくデータも固定する
#
# どのクオンツチームにも覚えのある事故でしょう。3月のバックテストが7月に再現できない。コードは
# git にあり、パラメータは実行ログにある。ところが足元で*データ*が動いていたのです。ベンダーが
# 歴史を訂正し（分割の修正、配当の訂正、生存バイアスの手当て）、「同じ」バックテストが違う
# Sharpe を出す。git はこれをコードについて解決しました。h5i-db はデータについて解決します。
# 書き込みはすべて書き換え不能なバージョンになり、名前付きスナップショットが、その実行が
# 消費したバイトそのものを固定します。しかも O(1) です。スナップショットはコピーではなく
# マニフェストの固定だからです。
#
# このレシピでは、まず失敗を具体的に見せ（訂正 → 素朴な再実行 → 違う Sharpe）、次に対処を
# 見せ（スナップショットで固定した再実行 → **ビット単位で同一**、表明つき）、最後に最小限の
# 実行台帳のパターンで締めます。

# %%
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_repro"), create=True)

# %% [markdown]
# ## 1. 価格パネルを読み込む（データバージョン1）
#
# 合成20銘柄、約3年ぶんの日次終値です。`sort_key=["ts", "symbol"]` なので、入力はキー全体で
# 整列している必要があります。日次パネルはタイムスタンプごとに20銘柄あるので、`ts` だけの順序
# では足りません。

# %%
symbols = [f"STK{i:03d}" for i in range(20)]
panel = cu.make_daily_prices(symbols=symbols, days=750)

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("prices", schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    panel.select(["ts", "symbol", "close"]).sort_by([("ts", "ascending"), ("symbol", "ascending")]),
    note="vendor delivery v1",
)

# %% [markdown]
# ## 2. 読み取り点を引数に取るバックテスト
#
# 再現性を可能にする設計上の判断はただ1つ、バックテストが `FROM prices` を決め打ちしないこと
# です。受け取るのは*関係*で、ライブのテーブルでも、固定したスナップショット
# `h5i('prices', 'bt-run-001')` でも、バージョン番号でも、as-of のタイムスタンプでも構いません。
# そして下流のすべてが、その読み取り点とパラメータの純粋な関数になります。
#
# 戦略そのものはあえて素っ気ないものです（126日モメンタム、月次リバランス、上位5銘柄を等
# ウェイト）。主役は配管であって、アルファではありません。

# %%
def run_backtest(read_point: str, lookback: int = 126, top_n: int = 5) -> dict:
    """Momentum backtest against any h5i-db relation. Deterministic."""
    px = db.sql(
        f"SELECT ts, symbol, close FROM {read_point} ORDER BY ts, symbol"
    ).to_pandas()
    wide = px.pivot(index="ts", columns="symbol", values="close").sort_index()

    mom = wide.pct_change(lookback)
    month_keys = wide.index.tz_convert(None).to_period("M")
    month_ends = wide.groupby(month_keys).tail(1).index
    weights = pd.DataFrame(0.0, index=wide.index, columns=wide.columns)
    for i, d in enumerate(month_ends[:-1]):
        if mom.loc[d].isna().all():
            continue  # still inside the lookback warm-up
        winners = mom.loc[d].nlargest(top_n).index
        in_month = (wide.index > d) & (wide.index <= month_ends[i + 1])
        weights.loc[in_month, winners] = 1.0 / top_n

    daily_ret = (weights * wide.pct_change()).sum(axis=1)
    curve = (1.0 + daily_ret).cumprod()
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    digest = hashlib.sha256(np.ascontiguousarray(curve.to_numpy()).tobytes()).hexdigest()
    return {"curve": curve, "sharpe": sharpe, "sha256": digest}

# %% [markdown]
# ## 3. 読み取り点を固定してから走らせる
#
# 研究の実行前に、名前付きスナップショットを取ります。この瞬間の `prices` のマニフェストを
# 凍結するもので、データはコピーされず、以後どんな書き込みもこの名前が指す先には触れられません。

# %%
db.snapshot("bt-run-001", tables=["prices"], note="data pin for momentum study 001")

run1 = run_backtest("h5i('prices', 'bt-run-001')")
print(f"run 1  sharpe={run1['sharpe']:.6f}  curve sha256={run1['sha256'][:16]}...")

# %% [markdown]
# ## 4. ベンダーが歴史を訂正する
#
# 3か月後、ベンダーが「訂正済み」ファイルを送ってきます。20銘柄のうち10銘柄について、配当調整の
# 履歴を改訂したものです。調整幅は履歴の先頭で25%、訂正の締め日に向かってゼロへ減衰します。
# 実際の調整済み価格の訂正はこういう形をしていて、水準だけでなく締め日より前の*リターン*を
# すべて静かに変えます。読み込みは誠実なやり方で行います。`write()` がテーブルの中身を*新しい
# バージョン*として注記付きで置き換え、バージョン1はバイト単位でそのまま残ります。

# %%
df = panel.select(["ts", "symbol", "close"]).to_pandas()
dates = np.sort(df["ts"].unique())
cutoff = dates[300]
factor = pd.Series(np.linspace(0.75, 1.0, 300), index=dates[:300])

restate_mask = df["symbol"].isin(symbols[:10]) & (df["ts"] < cutoff)
df.loc[restate_mask, "close"] = (
    df.loc[restate_mask, "close"] * df.loc[restate_mask, "ts"].map(factor)
).round(2)

restated = pa.Table.from_pandas(
    df.sort_values(["ts", "symbol"]), schema=schema, preserve_index=False
)
db.write("prices", restated, note="vendor restatement: dividend-adjustment fix, 10 names")

[{k: v.get(k) for k in ("sequence", "op", "rows", "note")} for v in db.versions("prices")]

# %% [markdown]
# ## 5. 素朴な再実行はずれる
#
# 「同じバックテスト」をライブのテーブルに対して走らせ直すと、訂正後のデータを読みます。同じ
# コード、同じパラメータ、違う Sharpe。再現不能の事故を2行に煮詰めた姿です。

# %%
run2 = run_backtest("prices")
print(f"run 1 (pinned data): sharpe={run1['sharpe']:.6f}")
print(f"run 2 (live table):  sharpe={run2['sharpe']:.6f}")
print(f"divergence: {abs(run2['sharpe'] - run1['sharpe']):.4f} Sharpe units")
assert run2["sha256"] != run1["sha256"], "restatement should have changed the results"

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(run1["curve"].index, run1["curve"], label=f"pinned v1 data (sharpe {run1['sharpe']:.2f})")
ax.plot(run2["curve"].index, run2["curve"], label=f"restated data (sharpe {run2['sharpe']:.2f})",
        ls="--")
ax.axvline(pd.Timestamp(cutoff), color="0.6", lw=0.8, ls=":")
ax.annotate(" restatement cutoff", xy=(pd.Timestamp(cutoff), ax.get_ylim()[1]),
            fontsize=8, va="top", color="0.4")
ax.set_title("Same code, same params - different data version, different backtest")
ax.set_xlabel("date")
ax.set_ylabel("equity (start = 1)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. 固定した再実行はビット単位で同一 — 表明つき
#
# 今度は訂正前に取ったスナップショットに対して走らせ直します。「近い」でも「許容範囲内」でも
# ありません。エクイティカーブの生の float64 バイトの SHA-256 が一致します。入力が同じ書き換え
# 不能なセグメントだからです。

# %%
run3 = run_backtest("h5i('prices', 'bt-run-001')")

assert run3["sha256"] == run1["sha256"], "pinned re-run must be bit-identical"
assert np.array_equal(run3["curve"].to_numpy(), run1["curve"].to_numpy())
print(f"re-run on 'bt-run-001': sharpe={run3['sharpe']:.6f}")
print(f"curve sha256 identical: {run3['sha256'][:32]}... == {run1['sha256'][:32]}...")
print("bit-identical: PASS")

# %% [markdown]
# ## 7. 最小限の実行台帳
#
# 固定を見つけられるようにします。`runs` テーブルに、実行 ID、読み取り点、パラメータ（JSON）、
# Sharpe、結果のダイジェストを記録します。台帳に載っている実行は、データベースのディレクトリを
# 持つ人なら誰でも正確に再現できます。行に書かれたスナップショット名*こそが*データ依存関係です。

# %%
runs_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("run_id", pa.string()),
        pa.field("read_point", pa.string()),
        pa.field("params", pa.string()),
        pa.field("sharpe", pa.float64()),
        pa.field("curve_sha256", pa.string()),
    ]
)
db.create_table("runs", runs_schema, time_column="ts")

params = json.dumps({"strategy": "momentum", "lookback": 126, "top_n": 5, "universe": 20})
for run_id, read_point, res in [
    ("bt-run-001", "h5i('prices', 'bt-run-001')", run1),
    ("bt-run-001-naive-rerun", "prices (live head)", run2),
    ("bt-run-001-verify", "h5i('prices', 'bt-run-001')", run3),
]:
    row = pa.table(
        {
            "ts": pa.array([pd.Timestamp.now(tz="UTC")], type=pa.timestamp("us", tz="UTC")),
            "run_id": [run_id],
            "read_point": [read_point],
            "params": [params],
            "sharpe": [res["sharpe"]],
            "curve_sha256": [res["sha256"]],
        }
    )
    db.append("runs", row, note=f"register {run_id}")

db.sql(
    """
    SELECT run_id, read_point, round(sharpe, 4) AS sharpe,
           substr(curve_sha256, 1, 12) AS digest
    FROM runs ORDER BY ts
    """
).to_pandas()

# %% [markdown]
# ## まとめ
#
# - バックテストはテーブル名ではなく**読み取り点**を受け取るべきです。探索にはライブの先頭を、
#   いずれ弁明する必要があるものには `h5i('prices', '<snapshot>')` を。
# - スナップショットはマニフェストの固定です。作成は O(1)、データのコピーはゼロ、以後の訂正の
#   影響も受けません。民間療法である「実行ごとに CSV を書き出す」と比べてみてください。あちらは
#   実行ごとにストレージを食い、元データからずれていき、SQL でクエリもできません。
# - 訂正は誠実なやり方で取り込めます。注記付きの `write()` が新しいバージョンを作り、古いほうは
#   永久に参照できます。データの系譜は口伝ではなく `versions()` です。
# - ビット単位で同一なら表明できます。結果のバイトをハッシュして、登録済みの実行がいまも再現
#   することを CI に検証させましょう。

# %%
db.close()
