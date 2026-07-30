# %% [markdown]
# # プレビューできる変更: 削除キーを怖がらずに悪いティックを直す
#
# 共有のティックストアで行を消したり書き換えたりするのは、クオンツデスクの日常業務の中で
# いちばん怖い操作です。指が滑った `DELETE WHERE` は、どんなハードウェア故障よりも多くの
# リサーチ用データセットを壊してきました。
#
# h5i-db の答えが **plan/apply の流れ**です。変更はまず、機械可読なサマリと変更前後の行
# サンプルを持つプランとしてステージされます。それを確認したうえで適用します。適用は新しい
# バージョンになり、元のバージョンは無傷のまま残ります。
#
# このレシピで進めるのは次の4つです。
#
# 1. 定番のティックデータの病理を2つ、フィードの瞬断とスケーリングのバグを注入し、SQL で
#    見つける
# 2. 削除をステージし、消しすぎているのをプレビューで捕まえ、破棄して、狭く引き直してから
#    適用する
# 3. `replace_range` をステージして価格をその場で修復する
# 4. 変更ポリシーでデータベースを締め、破壊的な直接書き込みが `PolicyError` を投げるように
#    する
#
# 最後の1つが、共有データベースやエージェントが操作するデータベースにおける安全性の話に
# なります。

# %% [markdown]
# ## ここで使う用語
#
# | 用語           | 意味 |
# | ------------ | --- |
# | ティック         | マーケットデータフィード上の1イベント。ふつうは約定か気配の更新 |
# | plan / apply | 削除や置換をプランとして積み、行数とサンプルを検分してからコミットする |
# | 変更ポリシー       | 破壊的な直接書き込みを禁じるデータベース単位の設定 |
# | バージョン        | コミット後のテーブルの状態。イミュータブルでいつまでも読める |
# | `restore`    | 古いバージョンを前に進める形の巻き戻し。履歴は消えず増える |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr, time_bucket
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_mutations"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_trades` の2セッションぶんのティックデータで、1行が1約定です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | 約定時刻、昇順 |
# | `symbol` | `string` | 銘柄コード |
# | `price` | `float64` | 約定価格 |
# | `size` | `int64` | 約定株数 |
# | `exchange` | `string` | 報告した取引所 |
# | `side` | `string` | `B` は買い主導、`S` は売り主導 |

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=2, trades_per_day=20_000, seed=7)
print(f"{trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# ## 2. 病理を2つ注入する
#
# このきれいなフィードを、現実のフィードが壊れるのと同じやり方で汚します。
#
# - **フィードの瞬断。** 1日目の10秒間（15:00:00〜15:00:10 UTC）だけ、全銘柄のあらゆる
#   プリントが10倍で届きます。小数点がずれたバーストです。
# - **スケーリングのバグ。** 2日目の10分間（14:00〜14:10 UTC）だけ、1銘柄の価格がすべて
#   10倍になります。フィードハンドラの銘柄別の掛け算バグです。
#
# この2つは扱いが違います。瞬断の行はゴミなので削除します。スケーリングのバグの行は本物の
# 約定で、価格も復元できるので修復します。

# %%
df = trades.to_pandas()

glitch = (df["ts"] >= "2026-06-01 15:00:00+00:00") & (df["ts"] < "2026-06-01 15:00:10+00:00")
scaling = (
    (df["ts"] >= "2026-06-02 14:00:00+00:00")
    & (df["ts"] < "2026-06-02 14:10:00+00:00")
    & (df["symbol"] == "MSFT")
)
df.loc[glitch | scaling, "price"] *= 10.0
corrupted = pa.Table.from_pandas(df, schema=trades.schema, preserve_index=False)

db.create_table("trades", trades.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("trades", corrupted, note="raw feed capture, days 1-2")
print(f"{glitch.sum()} glitch prints, {scaling.sum()} scaled prints, {len(df):,} rows total")

# %% [markdown]
# ## 3. 被害を見つける
#
# 頑健なスクリーニングは、各プリントをその銘柄の日次中央値と比べます。`approx_percentile_cont`
# を使います。壊れた行は中央値のおよそ10倍のところに座り、一方で中央値そのものはほとんど
# 動きません。壊れた行の割合が小さいからです。外れ値の検出を平均ではなく中央値に錨づける
# のは、これが理由です。
#
# この検出器は2回、いまと修復後に走らせます。だから読み取り点の関数として一度だけ書いて
# おきます。`trades` のフレームは自分自身のジョインの両側に入ります。片方は日次中央値へ
# 集約され、もう片方は行レベルの左側になります。

# %%
DAY = time_bucket("1d", col("ts"))


def outliers(version=None):
    src = db.table("trades", version=version)
    med = src.group_by("symbol", DAY.alias("day")).agg(
        med_px=sql_expr("approx_percentile_cont(price, 0.5)")
    )
    return (
        src.with_columns(day=DAY)
        .join(med, on=["symbol", "day"])
        .filter(col("price", relation="l") > 3 * col("med_px", relation="r"))
        .select(
            ts=col("ts", relation="l"),
            symbol=col("symbol", relation="l"),
            price=col("price", relation="l"),
            x_median=col("price", relation="l") / col("med_px", relation="r"),
        )
        .sort("ts")
    )


flagged = outliers().to_pandas()
flagged.groupby([flagged["ts"].dt.floor("1D").rename("day"), "symbol"]).size()

# %% [markdown]
# 2つのかたまりが、注入したとおりに出ました。1日目の全銘柄バーストと、2日目の MSFT だけの
# 一続きです。次は、フラグの立った行からそれぞれの時間範囲を割り出します。
#
# **変更プランの範囲は生の int64 マイクロ秒**で、`ts` 列の単位に揃っています。そして半開区間
# `[start, end)` です。だから最後の悪いプリントの1µs 先まで足します。

# %%
burst = flagged[flagged["ts"] < "2026-06-02"]
bug = flagged[(flagged["ts"] >= "2026-06-02") & (flagged["symbol"] == "MSFT")]

burst_lo = int(burst["ts"].min().value // 1000)
burst_hi = int(burst["ts"].max().value // 1000) + 1
bug_lo = int(bug["ts"].min().value // 1000)
bug_hi = int(bug["ts"].max().value // 1000) + 1
print("burst window:", burst["ts"].min(), "->", burst["ts"].max())
print("bug   window:", bug["ts"].min(), "->", bug["ts"].max())

# %% [markdown]
# ## 4. 削除をステージし、消しすぎを未然に捕まえる
#
# 最初に思いつくのは「念のため瞬断の前後5分をまとめて吹き飛ばす」でしょう。ステージして、
# 信じる前にプランを見てください。プランはコミットではないので、この時点でテーブルは何も
# 変わっていません。
#
# ここでは範囲変更の性質が何より効いてきます。**範囲は時刻だけにかかり**、銘柄では絞られ
# ません。「安全のために広く」取った窓は、健全な AAPL と NVDA のプリントを黙って巻き込みます。
# プランのサマリは、被害が出る前にそれを見せてくれます。

# %%
sloppy = db.plan_delete_range(
    "trades",
    int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000),
    int(pd.Timestamp("2026-06-01 15:05:00", tz="UTC").value // 1000),
    note="delete glitch burst (wide window)",
)
sloppy.summary

# %% [markdown]
# `rows_affected` は悪いプリントの数のおよそ25倍です。つまりこの広い窓は、まともな約定を
# 何百件も壊します。`before_sample` には処刑予定の行が並びます。健全な \\$200 前後のプリント
# が、10倍のゴミに混じっています。
#
# 破棄して、検出器が出した狭い範囲で引き直しましょう。

# %%
sloppy.before_sample.to_pandas().head(6)

# %%
sloppy.discard()

tight = db.plan_delete_range("trades", burst_lo, burst_hi, note="delete 10s decimal-shift burst")
print("rows_affected:", tight.summary["rows_affected"], " (bad prints:", len(burst), ")")
commit = tight.apply()
print("applied as version", commit["sequence"], "op:", commit["op"])

# %% [markdown]
# `apply()` は先にテーブルの先頭を確認し直します。プランをステージしたあとに誰かがコミット
# していれば、古い土台の上で計算された変更を公開するかわりに `ConflictError` を投げます。
# `append(expected_version=...)` と同じ楽観的並行制御です。
#
# ## 5. `plan_replace_range` でその場で修復する
#
# MSFT のスケーリングのバグには、もう一方の扱いが要ります。約定は本物で、価格は10で割れば
# 戻ります。
#
# `plan_replace_range(start, end, data)` は、窓の中の*すべて*を渡したデータで入れ替えます。
# だから置き換えるデータには、修復した MSFT の行だけでなく、窓の中の AAPL と NVDA の行も
# そのまま入っている必要があります。

# %%
win = df[(df["ts"] >= pd.Timestamp(bug_lo, unit="us", tz="UTC"))
         & (df["ts"] < pd.Timestamp(bug_hi, unit="us", tz="UTC"))].copy()
win.loc[win["symbol"] == "MSFT", "price"] /= 10.0
repaired = pa.Table.from_pandas(win, schema=trades.schema, preserve_index=False)

fix = db.plan_replace_range("trades", bug_lo, bug_hi, data=repaired,
                            note="MSFT 10x scaling bug: prices /10")
fix.summary

# %%
# after_sample previews the post-mutation rows - MSFT back at sane levels:
fix.after_sample.to_pandas().query("symbol == 'MSFT'").head(4)

# %%
fix.apply()
assert len(outliers().collect()) == 0
print("detector re-run: 0 outliers remaining")

# %% [markdown]
# 訂正は1つ1つがバージョンです。訂正前の記録は整数1つ隣に残るので、「クリーニングで
# バックテストは変わったか」は考古学ではなく SQL のジョインになります。レシピ05がその比較を
# 扱っています。

# %%
pd.DataFrame(db.versions("trades"))[["sequence", "op", "rows", "note"]]

# %%
import matplotlib.pyplot as plt

def msft_window(version=None):
    return (
        db.table("trades", version=version)
        .filter(
            col("symbol") == "MSFT",
            col("ts") >= "2026-06-02T13:50:00Z",
            col("ts") < "2026-06-02T14:20:00Z",
        )
        .select("ts", "price")
        .sort("ts")
        .to_pandas()
    )


before, after = msft_window(version=1), msft_window()

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(before["ts"], before["price"], lw=0.8, color="tab:red", label="version 1 (raw capture)")
ax.plot(after["ts"], after["price"], lw=0.8, color="tab:blue", label="head (repaired)")
ax.set_yscale("log")
ax.set_title("MSFT prints around the scaling bug: raw capture vs repaired head")
ax.set_xlabel("time (UTC)")
ax.set_ylabel("price (log scale)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 6. 変更ポリシー: 安全な道を唯一の道にする
#
# 共有のデータベース、あるいは自動化されたエージェントが書き込むデータベースでは、破壊的な
# 直接操作は既定で切っておきたいものです。`set_policy` は操作の種類ごとに真偽値のゲートを
# 切り替え、ゲートされた直接呼び出しは**何にも触れないうちに** `PolicyError` を投げます。
#
# plan/apply の流れは、どのポリシーの下でも使えます。これは意図的です。plan/apply こそが
# 公認の経路だからです。意図と変更のあいだに、プレビューでき、レビューできる成果物を必ず
# 挟ませます。

# %%
print("default policy:", db.policy())
db.set_policy(direct_write=False, direct_delete=False, direct_restore=False)

try:
    db.write("trades", corrupted)
except h5i_db.PolicyError as e:
    print(f"\nwrite blocked -> {type(e).__name__}")
    print("  code:", e.code)
    print("  hint:", e.hint)

# %% [markdown]
# 制限の厳しいポリシーの下でも、プランは立てられます。保留中のプランは一級のオブジェクト
# です。`list_plans` は、何がどんなサマリでステージされているかを見せます。
# プランは7日の TTL で失効し、ステージされたセグメントは、適用・破棄・失効のいずれかまで
# `vacuum` から守られます。

# %%
pending = db.plan_delete_range(
    "trades",
    int(pd.Timestamp("2026-06-01 16:00:00", tz="UTC").value // 1000),
    int(pd.Timestamp("2026-06-01 16:00:30", tz="UTC").value // 1000),
    note="staged under restrictive policy",
)

for p in db.list_plans("trades"):
    ttl_days = (p.raw["expires_at_ns"] - p.raw["created_at_ns"]) / 86_400e9
    print(f"plan {p.plan_id[:8]}…  note={p.raw['note']!r}  "
          f"rows_affected={p.summary['rows_affected']}  ttl={ttl_days:.0f}d")

pending.discard()
db.set_policy(direct_write=True, direct_delete=True, direct_restore=True)

# %% [markdown]
# ## まとめ
#
# - **プランを立て、見て、それから適用する。** `plan_delete_range` と `plan_replace_range` は、
#   行数のサマリと変更前後のサンプルを付けて変更をステージします。広すぎる削除は、害を
#   なす前に `rows_affected` で自ら名乗り出ました。
# - 変更の範囲は生の int64 マイクロ秒、半開区間、そして時刻のみです。銘柄では絞られないので、
#   置き換えるデータには窓の中の無関係な行もそのまま通す必要があります。
# - `apply()` はテーブルの先頭に対して競合を確認します。適用されたプランはノート付きの新しい
#   バージョンになり、訂正前のデータは `h5i('trades', v)` 1つ隣に残ります。
# - `set_policy(direct_write=False, ...)` は「気をつけてください」を `PolicyError` に変えます。
#   共有ストアの、そして LLM エージェントに触らせるものすべての、正しい既定値です。

# %%
db.close()
