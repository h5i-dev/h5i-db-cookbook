# %% [markdown]
# # ポイントインタイムのファンダメンタルズ: ASOF ジョインで先読みバイアスを断つ
#
# ファンダメンタルズにはタイムスタンプが2つあります。`period_end` はその数字が説明する会計上の
# 日付です。もう1つ、ふつう25〜55日あとに来る報告日が、市場が実際にその数字を知った時点です。
#
# データベースを間違ったほうで索引すると、どのバックテストもまだ存在しない数字で黙って
# 売買します。株式リサーチでいちばんよくある、そしていちばん見栄えのするバグです。
#
# このレシピで進めるのは次の4つです。
#
# 1. ファンダメンタルズを*報告時刻*で索引して保存する
# 2. `asof_join` で「直近に報告された EPS」を日足パネルに貼る
# 3. `period_end` で結合すると、素朴な決算シグナルがどれだけ水増しされるかを定量化する
# 4. 訂正が届いたあとも研究を再現できるよう、バージョンをピン留めする

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star, lit, time_bucket, when
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("mde_pit"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.make_fundamentals` は現実的な報告ラグつきの四半期 EPS を生成します。2つのタイムスタンプ
# こそが、このテーブルの肝です。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | **報告**時刻。数字が公になった瞬間 |
# | `period_end` | `timestamp[us, tz=UTC]` | その数字が説明する会計四半期の期末 |
# | `symbol` | `string` | 銘柄コード |
# | `eps` | `float64` | 報告された1株当たり利益 |
# | `revenue_m`、`book_value_m` | `float64` | 売上高と純資産、百万ドル |

# %%
SYMS = [f"STK{i:03d}" for i in range(30)]

funda = cu.make_fundamentals(symbols=SYMS, quarters=10).to_pandas()
print(f"{len(funda):,} reports x {funda.shape[1]} columns, "
      f"{funda['ts'].min():%Y-%m-%d} .. {funda['ts'].max():%Y-%m-%d}")
funda.head()

# %% [markdown]
# 価格の側は `cu.make_daily_prices` の日足 OHLCV パネルです。列は `ts`、`symbol`、`open`、
# `high`、`low`、`close`、`volume` で、1行が1銘柄1セッションです。

# %%
daily = cu.make_daily_prices(symbols=SYMS, days=700).to_pandas()
print(f"{len(daily):,} daily rows, {daily['ts'].min():%Y-%m-%d} .. {daily['ts'].max():%Y-%m-%d}")
daily.head()

# %% [markdown]
# 2つの生成器は独立なので、そのままでは EPS のシグナルは何も予測しません。先読みバイアスを
# *測れる*ようにするため、肝心な仕組みを1つ植え込みます。各報告の翌セッションに、株価が EPS
# サプライズの方向へ4%、恒久的にジャンプするのです。
#
# これが教科書どおりの決算反応であり、バイアスを見せる正直なやり方です。情報が価格に織り込ま
# れるのは*公表された*ときであって、四半期が終わったときではありません。

# %%
funda = funda.sort_values(["symbol", "ts"]).reset_index(drop=True)
funda["eps_growth"] = funda.groupby("symbol")["eps"].pct_change().round(4)

JUMP = 0.04
for r in funda.dropna(subset=["eps_growth"]).itertuples():
    direction = float(np.sign(r.eps_growth))
    if direction == 0.0:
        continue
    mask = (daily["symbol"] == r.symbol) & (daily["ts"] >= r.ts)
    daily.loc[mask, ["open", "high", "low", "close"]] *= 1 + JUMP * direction

print(f"{len(funda)} reports for {len(SYMS)} symbols, "
      f"{funda['ts'].min():%Y-%m-%d} .. {funda['ts'].max():%Y-%m-%d}")
print(f"{len(daily)} daily rows, {daily['ts'].min():%Y-%m-%d} .. {daily['ts'].max():%Y-%m-%d}")

# %% [markdown]
# 2つのタイムスタンプの隙間が、問題のすべてです。1銘柄について、各四半期の数字は灰色の帯の
# 長さだけ宙に浮いています。`period_end` で結合すると、バックテストは帯の*左端*で数字を
# 受け取ります。市場がそれを見たのは*右端*です。

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
f0 = funda[funda["symbol"] == "STK000"].reset_index(drop=True)
for i, r in f0.iterrows():
    axes[0].hlines(i, r["period_end"], r["ts"], color="grey", lw=4, alpha=0.6)
    axes[0].plot(r["period_end"], i, "o", color="tab:blue")
    axes[0].plot(r["ts"], i, "o", color="tab:red")
axes[0].set_title("STK000: period_end (blue) vs report date (red)")
axes[0].set_xlabel("date")
axes[0].set_ylabel("fiscal quarter #")

lags = (funda["ts"] - funda["period_end"]).dt.days
axes[1].hist(lags, bins=15, color="grey", edgecolor="white")
axes[1].set_title(f"Reporting lag, all reports (mean {lags.mean():.0f} days)")
axes[1].set_xlabel("days from period_end to report")
axes[1].set_ylabel("count")
fig.tight_layout()

# %% [markdown]
# ## 2. 正しい時計で両方のテーブルを保存する
#
# ファンダメンタルズのテーブルの `time_column` は**報告**時刻、つまり各行が公知になった瞬間
# です。`period_end` はふつうの列として同乗します。
#
# このスキーマの決定1つが、下流のすべての ASOF クエリを既定でポイントインタイムにします。

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
db.append(
    "prices",
    pa.Table.from_pandas(
        daily[["ts", "symbol", "close", "volume"]].sort_values(["ts", "symbol"]), preserve_index=False
    ).cast(PRICE_SCHEMA),
    note="daily closes with announcement reactions",
)

FUNDA_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),  # report time
        pa.field("period_end", pa.timestamp("us", tz="UTC")),
        pa.field("symbol", pa.string()),
        pa.field("eps", pa.float64()),
        pa.field("eps_growth", pa.float64()),
    ]
)
db.create_table("fundamentals", FUNDA_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "fundamentals",
    pa.Table.from_pandas(
        funda[["ts", "period_end", "symbol", "eps", "eps_growth"]].sort_values(["ts", "symbol"]),
        preserve_index=False,
    ).cast(FUNDA_SCHEMA),
    note="as-reported quarterly EPS, 10 quarters",
)

# %% [markdown]
# この研究は月次でリバランスするので、研究用のパネルは月末のクロスセクションです。
# `time_bucket('1mo', ...)` の集計1つを、それ自体のテーブルとして実体化します。
#
# 正典の日次ストアから、目的に合わせた小さなテーブルを導出するのが、このワークフローのふつうの
# 形です。こうするとジョインは、月末およそ990件に対して報告300件で走ります。日次パネル全体を
# 舐める必要はありません。
#
# 大きさが何であれ、ASOF ジョインのあとは左の行数と出力の行数が一致することを、下のように
# assert してください。静かなジョインのミスが、うるさいミスに変わります。

# %%
monthly = (
    db.table("prices")
    .group_by(time_bucket("1mo", col("ts")).alias("month"), "symbol")
    .agg(month_end=col("ts").max(), close=col("close").last("ts"))
    .select(col("month_end").alias("ts"), "symbol", "close")
    .sort(["ts", "symbol"])
    .to_arrow()
)

MONTHLY_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
    ]
)
db.create_table("prices_m", MONTHLY_SCHEMA, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices_m", monthly.cast(MONTHLY_SCHEMA), note="month-end closes derived from prices")
N_MONTHLY = monthly.num_rows
print(f"{N_MONTHLY} month-end observations")

# %% [markdown]
# ## 3. ポイントインタイムのジョイン
#
# `asof_join('prices_m', 'fundamentals', 'ts', 'ts', 'symbol')` は、各月末のバーに対して、
# *報告時刻*がそのバー以前で最新のファンダメンタルズ行を貼り付けます。銘柄ごとに、整列済み
# ストレージの上をストリーミングで、ウィンドウ関数の曲芸なしで。
#
# 右側でぶつかった列には `_right` が付くので、報告のタイムスタンプは `ts_right` として残ります。
# どの行も自分の「いつから知られていたか」を持ち歩くわけです。

# %%
pit = (
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), on="ts", by="symbol")
    .select("ts", "symbol", "close", "eps", "eps_growth", "period_end",
            known_since=col("ts_right"))
    .sort(["ts", "symbol"])
    .to_pandas()
)
assert len(pit) == N_MONTHLY, f"ASOF returned {len(pit)} rows for {N_MONTHLY} left rows"

# spot-check the ASOF result against an independent point query
probe = pit[pit["symbol"] == "STK005"].iloc[12]
point = (
    db.table("fundamentals")
    .filter(col("symbol") == "STK005", col("ts") <= probe["ts"].isoformat())
    .sort("ts", descending=True)
    .limit(1)
    .select("eps")
    .to_pandas()["eps"][0]
)
assert point == probe["eps"]

pit[pit["symbol"] == "STK000"].iloc[3:9]

# %% [markdown]
# `tolerance` は生のマイクロ秒で、古さの上限を決めます。120日を上限にすると、直近の報告が
# 1四半期半ほどより古い月末には、ゾンビの EPS ではなく NULL が入ります。上場廃止になった銘柄や
# 提出が遅れている銘柄を洗い出すのに使えます。

# %%
TOL_US = 120 * 86_400 * 1_000_000

(
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), on="ts", by="symbol", tolerance=TOL_US)
    .select(
        month_ends=count_star(),
        no_fresh_report=when(col("eps").is_null()).then(lit(1)).otherwise(lit(0)).sum(),
    )
    .to_pandas()
)

# %% [markdown]
# ## 4. 間違ったジョインと、その代償
#
# 定番の誤りが `period_end` での結合です。四半期が終わった晩に数字が分かっていたかのように
# 扱ってしまいます。`asof_join` は右側の時刻列を自由に取れるので、バグ版は引数1つぶんの距離に
# あります（ここでの報告ラグは `period_end` を報告順に単調のまま保つので、ジョインの要件を
# 満たします）。
#
# まず、漏れの機械的な大きさから。

# %%
ahead = (
    db.table("prices_m")
    .join_asof(db.table("fundamentals"), left_on="ts", right_on="period_end", by="symbol")
    .select("ts", "symbol", "close", eps_ahead=col("eps"), growth_ahead=col("eps_growth"))
    .sort(["ts", "symbol"])
    .to_pandas()
)
assert len(ahead) == N_MONTHLY

panel = pit.merge(ahead[["ts", "symbol", "eps_ahead", "growth_ahead"]], on=["ts", "symbol"])
both = panel.dropna(subset=["eps", "eps_ahead"])
leak = (both["eps"] != both["eps_ahead"]).mean()
print(f"{leak:.1%} of month-end observations see a *different* EPS under the period_end join")
print(f"on those, the join is early by up to the reporting lag (mean {lags.mean():.0f} days)")

# %% [markdown]
# 次に、シグナルとしての被害です。
#
# いちばん素朴な決算シグナル、四半期比 EPS 成長の符号を取り、*先の*21セッションのリターンと
# 結びつけます。そのリターンはクロスセクション平均からの差で測ります。共通ファクターと、その
# 実効的な観測数の少なさに比較が呑まれないようにするためです。
#
# ポイントインタイムのシグナルは何も知りません。そのシグナルが存在する頃には、決算のジャンプは
# すでに価格に入っているからです。先読みのシグナルは、覗き見したジャンプを「予測」します。

# %%
fut = daily.sort_values(["symbol", "ts"]).copy()
fut["fwd21"] = fut.groupby("symbol")["close"].transform(lambda s: s.shift(-21) / s - 1)
panel = panel.merge(fut[["ts", "symbol", "fwd21"]], on=["ts", "symbol"], how="left")
panel["fwd21_rel"] = panel["fwd21"] - panel.groupby("ts")["fwd21"].transform("mean")

rows = []
for label, field in [("point-in-time", "eps_growth"), ("lookahead", "growth_ahead")]:
    d = panel.dropna(subset=[field, "fwd21_rel"])
    d = d[d[field] != 0]
    sig = np.sign(d[field])
    up, dn = d.loc[sig > 0, "fwd21_rel"].mean(), d.loc[sig < 0, "fwd21_rel"].mean()
    rows.append(
        {
            "join": label,
            "corr(sign, fwd ret)": round(float(np.corrcoef(sig, d["fwd21_rel"])[0, 1]), 3),
            "fwd_ret_eps_up_pct": round(100 * up, 2),
            "fwd_ret_eps_down_pct": round(100 * dn, 2),
            "spread_pct": round(100 * (up - dn), 2),
        }
    )
bias = pd.DataFrame(rows).set_index("join")
bias

# %%
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(2)
ax.bar(x - 0.18, bias["fwd_ret_eps_up_pct"], 0.36, label="EPS growth > 0", color="tab:green")
ax.bar(x + 0.18, bias["fwd_ret_eps_down_pct"], 0.36, label="EPS growth < 0", color="tab:red")
ax.set_xticks(x, bias.index)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Market-relative forward 21-session return - the lookahead mirage")
ax.set_xlabel("how fundamentals were joined")
ax.set_ylabel("mean forward return (%)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 先読み版の「アルファ」の正体は、市場より先に見た決算反応がすべてです。同じデータ、同じ
# シグナル、タイムスタンプが1つ違うだけで、どこからともなくスプレッドが現れます。実データでも、
# 出来すぎたファンダメンタルズのバックテストはまさにこうして生まれます。
#
# ## 5. 訂正: 元の研究を再現する
#
# ファンダメンタルズは訂正されます。追記で索引した PIT テーブルでは、訂正は*それ自身の報告時刻
# を持つ新しい行*なので、履歴は編集されず、古い眺めも引けるままです。
#
# STK003 の直近の EPS が、監査調整で35%上方修正されたとしましょう。

# %%
v_study = db.versions("fundamentals")[-1]["sequence"]  # pin: the version our study used

last = funda[funda["symbol"] == "STK003"].iloc[-1]
restated = pa.table(
    {
        "ts": pa.array([funda["ts"].max() + pd.Timedelta(days=3)], pa.timestamp("us", tz="UTC")),
        "period_end": pa.array([last["period_end"]], pa.timestamp("us", tz="UTC")),
        "symbol": ["STK003"],
        "eps": [round(last["eps"] * 1.35, 2)],
        "eps_growth": [None],
    }
).cast(FUNDA_SCHEMA)
db.append("fundamentals", restated, note="STK003 EPS restated +35% (audit adjustment)")

def latest_report(label: str, version=None):
    return (
        db.table("fundamentals", version=version)
        .filter(col("symbol") == "STK003")
        .sort("ts", descending=True)
        .limit(1)
        .select(lit(label).alias("view"), "eps", known_since=col("ts"))
        .to_pandas()
    )


latest_report("head (post-restatement)")

# %%
latest_report(f"pinned v{v_study} (as studied)", version=v_study)

# %% [markdown]
# `v_study`、あるいは名前付きスナップショットを研究の実行と並べて記録しておけば、その研究は
# 永久にバイト単位で同じ入力に対して再実行できます。訂正はライブの利用のために着地しつつ、
# あなたの証跡は書き換わりません。
#
# 1つ注意点があります。`asof_join` はテーブルの*先頭*に対して働きます。ピン留めしたパネルを
# まるごと組み直すには、`db.read("fundamentals", version=v_study)` や SQL の `h5i(...)` で
# ピン留めしたバージョンを読み、それに対して結合してください。

# %% [markdown]
# ## まとめ
#
# - ファンダメンタルズは**報告時刻**で索引し、`period_end` はデータとして持たせます。この
#   スキーマの決定1つが、あらゆる `asof_join` を既定でポイントインタイムにします。
# - `asof_join(prices, fundamentals, 'ts', 'ts', 'symbol')` が PIT の仕掛けのすべてです。
#   銘柄ごとの直近既知値に、`ts_right` という組み込みの「いつから知られていたか」の来歴と、
#   古い数字を拒否する `tolerance` が付いてきます。
# - `period_end` での結合は、何もしないはずのシグナルに決算ジャンプぶんの太いスプレッドを
#   渡しました。つまり先読みバイアスを、主張ではなく実測として示せたわけです。
# - 訂正は編集ではなく追記です。`h5i('t', v)` や `read(version=)` でバージョンをピン留めすれば
#   元の研究がそのまま再現され、先頭は訂正後の眺めを提供し続けます。

# %%
db.close()
