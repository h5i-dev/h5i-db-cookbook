# %% [markdown]
# # コーポレートアクション: テープを失わずに株式分割を調整する
#
# 株式分割は歴史を書き換えます。効力発生日より前の価格はすべてスケールし直さなければならず、
# さもなければ分割をまたいで計算したリターンはすべてゴミです。
#
# ここで設計上の選択を迫られます。保存済みの系列を*言い直して*生のテープを失うのか、それとも
# 生のテープを正典として残し、読み取り時に調整するのか。
#
# h5i-db なら、どちらを選んでも失うものはありません。言い直しはノート付きの `write()`
# コミット1件で、言い直す前の系列はタイムトラベルで永久に読めますし、調整係数のパターンは
# 小さな SQL のジョインで済みます。
#
# このレシピでは、3回の分割をまたぐ AAPL と NVDA の実データで両方のパターンを組み立てます。
# AAPL 4:1（2020-08-31）、NVDA 4:1（2021-07-20）、NVDA 10:1（2024-06-10）です。

# %% [markdown]
# ## ここで使う用語
#
# | 用語          | 意味 |
# | ----------- | --- |
# | コーポレートアクション | 価格ではなく証券そのものに起きる変更 |
# | 株式分割        | 株数を倍にする措置。たとえば 4:1 で、価格は同じ倍率で割られる |
# | 調整後終値       | その後の分割と配当で調整した終値。時系列で比較できる |
# | 調整係数        | 生の価格を調整後の価格に変換する係数 |
# | リステートメント    | 保存された履歴の書き換え。ここではノート付きの `write()` コミット |
# | タイムトラベル     | 過去のバージョンの姿でテーブルを読むこと |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_corpactions"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.fetch_daily` は Yahoo Finance から実際の日足を取ってきて parquet にキャッシュします。
# だからこのレシピは決定的で、初回のあとはオフラインでも動きます。1行が1銘柄1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | セッションの日付 |
# | `symbol` | `string` | 銘柄コード |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `adj_close` | `float64` | 分割*と*配当で調整済みの終値 |
# | `volume` | `int64` | 出来高（株数） |

# %%
real = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
print(f"{real.num_rows:,} rows x {real.num_columns} columns, "
      f"{len(set(real['symbol'].to_pylist()))} symbols")
real.to_pandas().head()

# %% [markdown]
# このレシピ全体を決める癖が1つあります。Yahoo の `close` 列は**すでに分割調整済み**で、
# `adj_close` はその上に配当の調整を足しただけです。2018年の AAPL の close/adj_close 比が
# 1.07 前後なのは、純粋に配当のぶんです。
#
# 実際の取引所フィードなら、2020-08-28 の AAPL は \\$125 ではなく \\$499 前後でプリントされて
# いたはずです。そこで既知の分割スケジュールを使って*調整を戻し*、プリントされたとおりの
# テープを復元します。生のベンダーフィードが届けるのはそれですし、ここでの正直な出発点でも
# あります。

# %%
px = (
    real.to_pandas()
    .query("symbol in ['AAPL', 'NVDA']")[["ts", "symbol", "close", "adj_close", "volume"]]
    .sort_values(["ts", "symbol"])
    .reset_index(drop=True)
)

SPLITS = pd.DataFrame(
    {
        "ts": pd.to_datetime(["2020-08-31", "2021-07-20", "2024-06-10"], utc=True),
        "symbol": ["AAPL", "NVDA", "NVDA"],
        "ratio": [4.0, 4.0, 10.0],  # shares multiply by ratio, price divides
    }
)

# Cumulative product of all splits *after* each row's date = the factor that
# was later divided out of the price. Multiply it back to get the raw print.
px["future_factor"] = 1.0
for s in SPLITS.itertuples():
    mask = (px["symbol"] == s.symbol) & (px["ts"] < s.ts)
    px.loc[mask, "future_factor"] *= s.ratio
px["close_raw"] = (px["close"] * px["future_factor"]).round(2)

px[(px["symbol"] == "AAPL") & px["ts"].between("2020-08-27", "2020-09-02")]

# %% [markdown]
# 崖がここにあります。28日は \\$499、31日は \\$129 です。
#
# ## 2. 生のテープを正典のテーブルとして保存する
#
# 生の系列は取引所が実際にプリントしたものなので、これを書き換え不能な真実の源にすべきです。
# 調整済みの系列も、リターンも、ファクターも、下流のものはすべてここから導出でき、再現できます。

# %%
PRICE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
db.create_table("prices", PRICE_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])

raw_table = pa.Table.from_pandas(
    px[["ts", "symbol", "close_raw", "volume"]].rename(columns={"close_raw": "close"}),
    preserve_index=False,
).cast(PRICE_SCHEMA)
db.append("prices", raw_table, note="raw unadjusted tape, AAPL+NVDA 2018-2026")

# %% [markdown]
# ## 3. 定番のやらかし: 分割をまたぐ素朴なリターン
#
# 生の終値に `lag()` のウィンドウを当てて日次リターンを出し、記録に残る最悪の日を眺めます。
#
# そのうち3日は暴落ではありません。-75% や -90% の「リターン」のふりをした分割です。この系列を
# 食べたリスクモデルもモメンタムのシグナルもストップロスも、何も起きていない日に発火して
# いたはずです。

# %%
PREV_CLOSE = sql_expr("lag(close)").over(partition_by="symbol", order_by="ts")


def worst_days(version=None, n: int = 5):
    """The n most negative daily returns - re-run after the restatement."""
    return (
        db.table("prices", version=version)
        .with_columns(ret=col("close") / PREV_CLOSE - 1)
        .filter(col("ret").is_not_null())
        .select("ts", "symbol", ret_pct=(col("ret") * 100).round(1))
        .sort("ret_pct")
        .limit(n)
        .to_pandas()
    )


worst = worst_days()
worst

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for sym, g in px.groupby("symbol"):
    axes[0].plot(g["ts"], g["close_raw"], lw=0.9, label=sym)
for s in SPLITS.itertuples():
    axes[0].axvline(s.ts, color="crimson", ls="--", lw=0.8)
axes[0].set_yscale("log")
axes[0].set_title("Raw tape: split cliffs (dashed = effective dates)")
axes[0].set_xlabel("date")
axes[0].set_ylabel("close (USD, log)")
axes[0].legend()

nvda = px[px["symbol"] == "NVDA"]
axes[1].plot(nvda["ts"], nvda["close_raw"] / nvda["close_raw"].iloc[0], lw=0.9, label="naive (raw closes)")
axes[1].plot(nvda["ts"], nvda["close"] / nvda["close"].iloc[0], lw=0.9, label="split-adjusted")
axes[1].set_yscale("log")
axes[1].set_title("NVDA cumulative growth: naive vs adjusted")
axes[1].set_xlabel("date")
axes[1].set_ylabel("growth of $1 (log)")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# ## 4. パターンA: 読み取り時に調整係数を結合する
#
# `prices` は永久に生のまま置きます。分割スケジュールはそれ自体の小さなテーブルとして保存し、
# 調整済みの系列は SQL で導出します。
#
# ある行の調整係数は、効力発生日が*その行より後*の分割比率をすべて掛け合わせたものです。
# `exp(sum(ln ...))` がその積を集約に変えるので、ジョイン1つと GROUP BY 1つで仕事が終わります。
# 次の分割が発表されたら、取り込みは1行の append で済み、過去のデータには手を触れません。

# %%
SPLIT_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("ratio", pa.float64()),
    ]
)
db.create_table("splits", SPLIT_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append("splits", pa.Table.from_pandas(SPLITS, preserve_index=False).cast(SPLIT_SCHEMA),
          note="split schedule through 2024-06-10")

# A non-equi LEFT JOIN (`s.ts > p.ts`) whose whole condition would have to be
# a raw fragment anyway - one of the places a SQL string stays clearer than
# the builder.
adjusted = db.sql(
    """
    SELECT p.ts, p.symbol, p.close AS close_raw,
           round(p.close / coalesce(exp(sum(ln(s.ratio))), 1.0), 4) AS close_adj
    FROM prices p
    LEFT JOIN splits s
           ON s.symbol = p.symbol AND s.ts > p.ts
    GROUP BY p.ts, p.symbol, p.close
    ORDER BY p.ts, p.symbol
    """
).to_pandas()
adjusted[(adjusted["symbol"] == "AAPL") & adjusted["ts"].between("2020-08-27", "2020-09-02")]

# %% [markdown]
# 確認です。SQL で調整した系列は、ベンダーの分割調整済み `close` とセント単位で一致しなければ
# なりません。ずれの元は、生のテープを復元するときに当てたセント丸めだけです。

# %%
check = adjusted.merge(px[["ts", "symbol", "close"]], on=["ts", "symbol"])
max_err = (check["close_adj"] - check["close"]).abs().max()
print(f"max |SQL-adjusted - vendor adjusted| = ${max_err:.4f} across {len(check):,} rows")
assert max_err < 0.01

# %% [markdown]
# `splits` テーブルも他と同じフィードなので、append は時刻順に届く必要があります。2019年の
# 分割を後から埋め戻して2024年の行の後ろに append することは許されません。h5i-db は並び順を
# 黙って壊すかわりに拒否しますし、過去の訂正は `write()` か plan/apply の流れを通します。

# %%
stale_split = pa.table(
    {
        "ts": pa.array([pd.Timestamp("2019-06-01", tz="UTC")], pa.timestamp("us", tz="UTC")),
        "symbol": ["FAKE"],
        "ratio": [2.0],
    }
).cast(SPLIT_SCHEMA)
try:
    db.append("splits", stale_split)
except h5i_db.H5iError as e:
    print(f"rejected: code={e.code}\nhint: {e.hint}")

# %% [markdown]
# ## 5. パターンB: その場で言い直し、バージョニングで履歴を残す
#
# 調整済みの系列が*そのままテーブルであってほしい*デスクも多いでしょう。そうすれば利用者は
# ジョインの記述ゼロできれいな価格を手にできます。ふつうはその代償として生のテープが消えます。
#
# バージョン管理されたストレージなら消えません。`write()` は先頭を置き換えますが、過去の
# バージョンはすべて読めるまま残るので、言い直しは監査でき、巻き戻せる操作になります。

# %%
restated = px.copy()
restated["close"] = (restated["close_raw"] / restated["future_factor"]).round(4)
db.write(
    "prices",
    pa.Table.from_pandas(restated[["ts", "symbol", "close", "volume"]], preserve_index=False).cast(PRICE_SCHEMA),
    note="restated: split-adjusted (AAPL 4:1 2020-08-31, NVDA 4:1 2021-07-20, NVDA 10:1 2024-06-10)",
)

[{k: v[k] for k in ("sequence", "op", "rows", "note") if k in v} for v in db.versions("prices")]

# %% [markdown]
# バージョン履歴が*そのまま*監査証跡です。`h5i('prices', 1)` は言い直す前のテープを、ライブの
# テーブルと並べて引きます。以下は分割前後の AAPL で、2つのバージョンから逆算した係数も
# 付けています。

# %%
now_px = col("close", relation="l")
was_px = col("close", relation="r")

(
    db.table("prices")  # head: restated
    .join(db.table("prices", version=1), on=["ts", "symbol"])  # raw tape
    .filter(
        col("symbol", relation="l") == "AAPL",
        col("ts", relation="l").between("2020-08-26", "2020-09-03"),
    )
    .select(
        ts=col("ts", relation="l"),
        symbol=col("symbol", relation="l"),
        raw_close=was_px,
        adj_close=now_px,
        implied_factor=(was_px / now_px).round(2),
    )
    .sort("ts")
    .to_pandas()
)

# %% [markdown]
# 「言い直す前の価格系列はどうなっていたか」は、実時刻でも引けます。下流のジョブが読んだ
# *時刻*だけを記録していて、どのバージョンかを残していない場合に効きます。

# %%
raw_commit = db.versions("prices")[1]  # sequence 1 = the raw-tape append
as_of = pd.Timestamp(raw_commit["committed_at_ns"], unit="ns", tz="UTC").isoformat()
pre = db.read("prices", as_of=as_of).to_pandas()
# Compare against a Timestamp, not a string: pandas parses a string literal to
# datetime64[ns] and the unit mismatch with our [us] column matches nothing.
split_eve = pd.Timestamp("2020-08-28 20:00", tz="UTC")
aapl_pre = pre[(pre["symbol"] == "AAPL") & (pre["ts"] == split_eve)]
print(f"as of {as_of}:")
print(aapl_pre[["ts", "symbol", "close"]].to_string(index=False))

# %% [markdown]
# そして先頭を言い直したいま、分割の痕跡は消えました。系列の最悪の日は、コロナ期の急落や
# 決算という本物の相場の動きになっています。

# %%
worst_days()

# %% [markdown]
# ## パターンの選び方
#
# - **調整係数の結合**（パターンA）。生のテープが正典のまま残り、調整済みの価格は常に*現在の*
#   分割スケジュールと整合し、新しいアクションは1行の append です。代償は、利用者全員が
#   そのジョインを当てるか、あなたが保守する派生テーブルを読む必要があることです。
# - **その場で言い直す**（パターンB）。利用者はジョインなしできれいな価格を得られ、言い直しは
#   監査できるコミット1件で、`restore()` で巻き戻せて `h5i('prices', v)` で覗けます。代償は、
#   アクションの発表から言い直しまでのあいだ先頭が古いことと、言い直しのたびにテーブルを
#   書き直すことです。
# - 2つは組み合わせられます。正典として `prices` を生のまま置き、`splits` テーブルを添え、
#   利便性のために言い直した `prices_adj` テーブルを公開する。バージョニングは、どちらにも
#   監査証跡を与えます。

# %% [markdown]
# ## まとめ
#
# - ベンダーの「close」列はすでに調整済みのことが多いので、その上に何かを積む前に、自分の
#   フィードが何を届けているかを知ってください。ここでは分割スケジュールから本当の生のテープを
#   復元しました。
# - 分割をまたぐ素朴なリターンは -75%（AAPL 4:1）や -90%（NVDA 10:1）という痕跡を刻みます。
#   `lag()` のウィンドウクエリ1つで、それが飛び出してきます。
# - 調整係数のパターンは、追記専用の `splits` テーブルに対する `exp(sum(ln(ratio)))` という
#   ジョイン1つの導出です。
# - `write()` と `note` は言い直しを監査できるコミットに変えます。`h5i('prices', 1)` と
#   `read(as_of=...)` が言い直す前の眺めをすべて引けるようにするので、きれいなデータと生の
#   テープのどちらかを選ぶ必要はありません。

# %%
db.close()
