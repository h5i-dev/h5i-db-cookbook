# %% [markdown]
# # 再現できるバックテスト: コードだけでなくデータも固定する
#
# どのクオンツチームもこの事故を経験しています。3月のバックテストが7月に再現できない。コード
# は git にあり、パラメータも実行ログにあるのに、*データ*が足元で動いていた。
#
# ベンダーが履歴を言い直したのです。分割の修正、配当の訂正、生存者バイアスのパッチ。そして
# 「同じ」バックテストが違うシャープを打ち出します。
#
# git はこれをコードについて解決しました。h5i-db はデータについて解決します。書き込みは
# すべて書き換え不能なバージョンで、名前付きスナップショットは実行が消費したバイト列そのもの
# を固定します。固定のコストは O(1) です。スナップショットはコピーではなくマニフェストへの
# 参照だからです。
#
# このレシピで進めるのは次の3つです。
#
# 1. 失敗を具体的に見せる。データを言い直し、素朴に走らせ直して違うシャープを出す
# 2. 直し方を当てる。スナップショットで固定して走らせ直し、**ビット単位で同一**の結果を
#    assert する
# 3. 最小限の実行台帳のパターンで締める

# %%
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db
from h5i_db import col, sql_expr

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_repro"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_daily_prices` が返すのは日足 OHLCV パネルです。合成20銘柄のおよそ3年ぶんで、1行が
# 1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 引け時刻、20:00 UTC |
# | `symbol` | `string` | 銘柄コード、`STK000` 〜 `STK019` |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `volume` | `int64` | 出来高（株数） |

# %%
symbols = [f"STK{i:03d}" for i in range(20)]
panel = cu.make_daily_prices(symbols=symbols, days=750)
print(f"{panel.num_rows:,} rows x {panel.num_columns} columns")
panel.to_pandas().head()

# %% [markdown]
# `ts`、`symbol`、`close` をデータのバージョン1として保存します。`sort_key=["ts", "symbol"]`
# なので、入力はキー全体で整列している必要があります。日次パネルは同じタイムスタンプを20銘柄で
# 共有しているので、`ts` だけの順序では足りません。

# %%
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
# 再現性を可能にする設計上の決定が1つあります。バックテストが読み取り点をハードコードしない
# ことです。
#
# 読み取り点は引数として受け取ります。ライブのテーブルでも、固定したスナップショットでも、
# バージョン番号でも、as-of のタイムスタンプでも構いません。そうすると下流はすべて、その
# 読み取り点とパラメータの純関数になります。`db.table(name, snapshot=..., version=...)` が
# 固定をキーワードにします。クエリ文字列に関係名を継ぎ足す必要はありません。
#
# 戦略そのものはあえて平凡です。126日モメンタム、月次リバランス、上位5銘柄の等ウェイト。
# 主題は配管のほうで、アルファではありません。

# %%
def run_backtest(snapshot=None, version=None, lookback: int = 126, top_n: int = 5) -> dict:
    """Momentum backtest against any read point of `prices`. Deterministic."""
    px = (
        db.table("prices", snapshot=snapshot, version=version)
        .select("ts", "symbol", "close")
        .sort(["ts", "symbol"])
        .to_pandas()
    )
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
# 研究の実行の前に、名前付きスナップショットを取ります。これがこの瞬間の `prices` のマニフェスト
# を凍結します。データはコピーされませんし、その後のどの書き込みも、その名前が指すものには
# 触れられません。

# %%
db.snapshot("bt-run-001", tables=["prices"], note="data pin for momentum study 001")

run1 = run_backtest(snapshot="bt-run-001")
print(f"run 1  sharpe={run1['sharpe']:.6f}  curve sha256={run1['sha256'][:16]}...")

# %% [markdown]
# ## 4. ベンダーが履歴を言い直す
#
# 3か月後、ベンダーが「訂正版」のファイルを送ってきます。20銘柄のうち10銘柄について、配当調整
# の履歴を改訂したものです。掛け目は履歴の先頭で25%、訂正の区切り日でゼロへと減衰します。
#
# 実際の調整後価格の言い直しはこういう形をしていて、水準だけでなく区切り日より前の*リターン*
# をすべて静かに変えます。
#
# 読み込みは正直なやり方でやります。`write()` がテーブルの中身をノート付きの*新しいバージョン*
# として置き換え、バージョン1はバイト単位でそのまま残ります。

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
# 「同じバックテスト」をライブのテーブルに対して走らせ直すと、言い直されたデータを読みます。
# 同じコード、同じパラメータ、違うシャープ。再現不能事故を2行に縮めたものです。

# %%
run2 = run_backtest()  # live head
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
# ## 6. 固定した再実行はビット単位で同一、assert 付き
#
# 今度は言い直しの前に取ったスナップショットに対して走らせ直します。
#
# 結果は「近い」でも「許容誤差の中」でもありません。エクイティカーブの生の float64 バイト列の
# SHA-256 が一致します。入力が同じ書き換え不能なセグメントだからです。

# %%
run3 = run_backtest(snapshot="bt-run-001")

assert run3["sha256"] == run1["sha256"], "pinned re-run must be bit-identical"
assert np.array_equal(run3["curve"].to_numpy(), run1["curve"].to_numpy())
print(f"re-run on 'bt-run-001': sharpe={run3['sharpe']:.6f}")
print(f"curve sha256 identical: {run3['sha256'][:32]}... == {run1['sha256'][:32]}...")
print("bit-identical: PASS")

# %% [markdown]
# ## 7. 最小限の実行台帳
#
# 固定を見つけられるようにします。`runs` テーブルに実行 ID、読み取り点、JSON のパラメータ、
# シャープ、結果のダイジェストを記録します。
#
# 台帳にある実行は、データベースのディレクトリを持つ人なら誰でも厳密に再現できます。その行の
# スナップショット名が*そのまま*データの依存関係だからです。

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
    ("bt-run-001", "snapshot bt-run-001", run1),
    ("bt-run-001-naive-rerun", "prices (live head)", run2),
    ("bt-run-001-verify", "snapshot bt-run-001", run3),
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

(
    db.table("runs")
    .select(
        "run_id", "read_point",
        sharpe=col("sharpe").round(4),
        digest=sql_expr("substr(curve_sha256, 1, 12)"),
    )
    .sort("ts")
    .to_pandas()
)

# %% [markdown]
# ## まとめ
#
# - バックテストが取るべきなのはテーブル名ではなく**読み取り点**です。探索にはライブの先頭を、
#   いつか説明する必要があるものには `h5i('prices', '<snapshot>')` を使ってください。
# - スナップショットはマニフェストの固定です。作成は O(1)、データのコピーはゼロ、あとからの
#   言い直しの影響も受けません。実行ごとに CSV を切り出す民間療法と比べてみてください。実行
#   ごとにストレージを食い、元データから乖離し、SQL でも引けません。
# - 言い直しは正直なやり方で読み込めます。ノート付きの `write()` が新しいバージョンを作り、
#   古いほうは永久に指せるまま残ります。データの系譜は口伝ではなく `versions()` に住みます。
# - ビット単位で同一なら assert できます。結果のバイト列をハッシュして、登録済みの実行がいまも
#   再現することを CI に確認させましょう。

# %%
db.close()
