# %% [markdown]
# # プレビューできる変更: 削除キーを恐れずに悪いティックを直す
#
# 共有のティックストアで行を消したり書き換えたりする作業は、クオンツデスクの日常業務で
# いちばん怖いものです。指が滑った `DELETE WHERE` は、どんなハードウェア障害よりも多くの
# リサーチ用データセットを葬ってきました。h5i-db の答えが**plan/apply の流れ**です。変更は
# まず、機械可読な要約と変更前後の行サンプルを添えたプランとして*ステージング*され、
# レビューを経て、そこで初めて適用されます。しかも新しいバージョンとして適用され、古い
# バージョンはそのまま残ります。このレシピでは次を行います。
#
# 1. ティックデータの典型的な病理を2つ（フィードの一時的な異常と10倍のスケーリングバグ）
#    仕込み、SQL で見つける
# 2. 削除をステージングし、プレビューの段階で消しすぎに気づき、破棄して、範囲を狭めて
#    立て直し、適用する
# 3. 価格をその場で直す `replace_range` をステージングする
# 4. 変更ポリシーでデータベースを締め、直接の破壊的な書き込みが `PolicyError` を上げる
#    ようにする。共有データベースやエージェントが操作するデータベースにとっての安全策です

# %%
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_mutations"), create=True)

# %% [markdown]
# ## 1. 2日ぶんのティックに、2つの病理を仕込む
#
# 合成フィードを、現実のフィードが壊れるのと同じやり方で壊します。
#
# - **フィードの異常**: 1日目の10秒間（15:00:00〜15:00:10 UTC）、全銘柄のすべての約定が
#   10倍で届きます。小数点のずれが連続する状態です。
# - **スケーリングバグ**: 2日目の10分間（14:00〜14:10 UTC）、1銘柄（MSFT）だけ価格が
#   すべて10倍になります。ハンドラの銘柄別の係数バグです。
#
# 異常のほうの行はゴミなので削除します。スケーリングバグの行は実在の約定で、価格も
# 復元できるので修復します。

# %%
trades = cu.make_trades(symbols=["AAPL", "MSFT", "NVDA"], days=2, trades_per_day=20_000, seed=7)
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
# ## 2. 被害を SQL で見つける
#
# 丈夫な検出方法として、すべての約定をその銘柄の日次中央値
# （`approx_percentile_cont`）と比べます。壊れた行は中央値のおよそ10倍のところに座り、
# 壊れた行の割合が小さいので中央値自体はほとんど動きません。外れ値の検出で平均ではなく
# 中央値を軸にするのは、このためです。

# %%
flagged = db.sql(
    """
    WITH med AS (
        SELECT symbol, time_bucket('1d', ts) AS day,
               approx_percentile_cont(price, 0.5) AS med_px
        FROM trades GROUP BY symbol, day
    )
    SELECT t.ts, t.symbol, t.price, t.price / m.med_px AS x_median
    FROM trades t
    JOIN med m ON t.symbol = m.symbol AND time_bucket('1d', t.ts) = m.day
    WHERE t.price > 3 * m.med_px
    ORDER BY t.ts
    """
).to_pandas()

flagged.groupby([flagged["ts"].dt.floor("1D").rename("day"), "symbol"]).size()

# %% [markdown]
# 仕込んだとおり、クラスタが2つ見えます。1日目の全銘柄にわたる異常と、2日目の MSFT だけの
# 区間です。フラグの立った行から、それぞれの時間の境界を求めます。**変更プランの範囲は
# 生の int64 マイクロ秒**（`ts` 列の単位）で、`[start, end)` の半開区間です。だから最後の
# 悪い約定より 1µs 先まで取ります。

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
# ## 3. 削除をステージングし、消しすぎを起きる前に捕まえる
#
# 最初に思いつくのは「安全側に倒して、異常の前後5分をまとめて消す」でしょう。それを
# ステージングして、信じる前にプランを見ます。プランはコミットではありません。テーブルは
# まだ何も変わっていません。
#
# 範囲変更には重要な性質が1つあります。**範囲は時刻だけにかかり**、銘柄では絞りません。
# 「安全のために広く」取った窓は、健全な AAPL と NVDA の約定も黙って巻き込みます。
# プランの要約は、被害が出る*前に*それを見せてくれます。

# %%
sloppy = db.plan_delete_range(
    "trades",
    int(pd.Timestamp("2026-06-01 15:00:00", tz="UTC").value // 1000),
    int(pd.Timestamp("2026-06-01 15:05:00", tz="UTC").value // 1000),
    note="delete glitch burst (wide window)",
)
sloppy.summary

# %% [markdown]
# `rows_affected` は悪い約定の数のおよそ25倍です。広い窓なら、何百もの正常な約定が
# 消えていました。`before_sample` を見ると、消える運命だった行が並んでいます。10倍のゴミに
# 混じって、健全な \\$200 前後の約定が入っています。破棄して、検出器が示した狭い境界で
# 立て直しましょう。

# %%
sloppy.before_sample.to_pandas().head(6)

# %%
sloppy.discard()

tight = db.plan_delete_range("trades", burst_lo, burst_hi, note="delete 10s decimal-shift burst")
print("rows_affected:", tight.summary["rows_affected"], " (bad prints:", len(burst), ")")
commit = tight.apply()
print("applied as version", commit["sequence"], "op:", commit["op"])

# %% [markdown]
# `apply()` はまずテーブルの先頭を再確認します。プランをステージングしたあとに誰かが
# コミットしていれば、古い土台の上で計算された変更を公開する代わりに `ConflictError` を
# 上げます。`append(expected_version=...)` と同じ楽観的並行制御です。
#
# ## 4. `plan_replace_range` でその場で修復する
#
# MSFT のスケーリングバグは事情が違います。約定は実在し、価格も復元できます（`/10`）。
# `plan_replace_range(start, end, data)` は、その窓の中身を*丸ごと*渡したデータに差し替え
# ます。だから差し替えるデータには、窓の中の AAPL と NVDA の行をそのまま含めたうえで、
# 修復した MSFT の行を入れる必要があります。

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
assert len(db.sql(
    """
    WITH med AS (SELECT symbol, time_bucket('1d', ts) AS day,
                        approx_percentile_cont(price, 0.5) AS med_px
                 FROM trades GROUP BY symbol, day)
    SELECT t.ts FROM trades t
    JOIN med m ON t.symbol = m.symbol AND time_bucket('1d', t.ts) = m.day
    WHERE t.price > 3 * m.med_px
    """
)) == 0
print("detector re-run: 0 outliers remaining")

# %% [markdown]
# 訂正はすべてバージョンになります。未訂正の生データは整数1つ隣にあるので、「クリーニングで
# バックテストは変わったのか」は発掘作業ではなく SQL のジョインで済みます（レシピ05）。

# %%
pd.DataFrame(db.versions("trades"))[["sequence", "op", "rows", "note"]]

# %%
import matplotlib.pyplot as plt

before = db.sql(
    """
    SELECT ts, price FROM h5i('trades', 1)
    WHERE symbol = 'MSFT' AND ts >= '2026-06-02T13:50:00Z' AND ts < '2026-06-02T14:20:00Z'
    ORDER BY ts
    """
).to_pandas()
after = db.sql(
    """
    SELECT ts, price FROM trades
    WHERE symbol = 'MSFT' AND ts >= '2026-06-02T13:50:00Z' AND ts < '2026-06-02T14:20:00Z'
    ORDER BY ts
    """
).to_pandas()

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
# ## 5. 変更ポリシー: 安全な道を唯一の道にする
#
# 共有のデータベース、あるいは自動エージェントが書き込むデータベースでは、直接の破壊的な
# 操作は既定で切っておきたいものです。`set_policy` は操作の種類ごとに真偽値のゲートを
# 切り替えます。ゲートされた直接呼び出しは、**何にも触れないうちに** `PolicyError` を
# 上げます。plan/apply の流れは使えるままです。それが認められた経路だからで、意図と変更の
# あいだにプレビュー可能でレビュー可能な成果物を必ず1つ挟ませます。

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
# 制限ポリシーの下でも、プランを立てること自体は許されています。保留中のプランは一級の
# オブジェクトです。`list_plans` が、何がどんな要約でステージングされているかを見せます。
# プランは7日の TTL で期限切れになり、ステージングされたセグメントは、適用・破棄・期限切れの
# いずれかが起きるまで `vacuum` から保護されます。

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
# - **立てて、見て、それから適用する。** `plan_delete_range` と `plan_replace_range` は、
#   行数の要約と変更前後のサンプルを添えて変更をステージングします。広すぎた削除は、害を
#   なす前に `rows_affected` の形で自ら名乗り出ました。
# - 変更範囲は**生の int64 マイクロ秒、半開区間、時刻のみ**です。銘柄では絞らないので、
#   差し替えるデータには窓の中の無関係な行をそのまま通してやる必要があります。
# - `apply()` はテーブルの先頭に対して衝突を検査します。適用されたプランは注記付きの
#   新しいバージョンになり、未訂正のデータは `h5i('trades', v)` 1つ隣に残ります。
# - `set_policy(direct_write=False, ...)` は「気をつけてください」を `PolicyError` に
#   変えます。共有ストアと、LLM エージェントが触れる場所には、これが正しい既定値です。

# %%
db.close()
