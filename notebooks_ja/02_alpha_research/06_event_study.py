# %% [markdown]
# # イベントスタディ: ASOF で発表日を揃えて CAR を測る
#
# 古典的なイベントスタディのパイプライン――マーケットモデルによる異常リターン、発表前後の
# 累積異常リターン（CAR）――には、昔から面倒な工程が1つあります。発表は立会日に落ちてくれません。
# 決算は土曜に出るし、M&A は祝日に漏れるし、どの研究も「発表以降で最初の立会
# セッション」を必要とします。それはまさに `'forward'` 方向の ASOF ジョインであり、h5i-db なら
# 銘柄ごとに、陳腐化の許容差込みで、SQL 1呼び出しで済みます。
#
# 段取りはこうです。
#
# 1. 100銘柄の日次パネルを作り、100件のイベントのうち処置群の半分に*既知の*発表日ショック
#    （day-0 に +2%、その後10日のドリフト）を仕込む。研究が復元すべき正解を用意するためです
# 2. `prices` と `events` のテーブルを保存し、`asof_join(..., 'forward', tolerance)` で
#    イベントを立会セッションに揃える
# 3. マーケットモデルのベータを推定し、処置群と対照群の CAR[-10,+10] を信頼区間つきで計算する

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_events"), create=True)

# %% [markdown]
# ## 1. 効果が分かっているパネルを合成する
#
# `make_daily_prices` が返すのはファクター駆動のパネル（共通の市場ファクターと個別ノイズ）で、
# イベントは入っていません。そこで保存する*前に*、自分たちで pandas で足します。100件の
# イベントを銘柄1つにつき1件、ランダムな暦上のタイムスタンプに置きます。あえて週末も含めます。
# 処置群の半分については、効力を持つセッション以降の価格経路をスケールします。day-0 に +2% の
# 水準シフト、その後10セッションにわたって1日 +0.15% のドリフト（発表後ドリフトの戯画です）。
# 対照群には何もしないので、その CAR はゼロで平坦になるはずです。プラセボ検査が最初から
# 組み込まれている形です。

# %%
N_SYMBOLS, N_DAYS, N_EVENTS = 100, 650, 100
DAY0_SHOCK, DRIFT_PER_DAY, DRIFT_DAYS = 0.02, 0.0015, 10

symbols = [f"STK{i:03d}" for i in range(N_SYMBOLS)]
prices = cu.make_daily_prices(symbols=symbols, days=N_DAYS).to_pandas()

sess_us = np.sort(prices["ts"].astype("int64").unique())  # session closes, epoch us
rng = np.random.default_rng(7)

# One event per symbol: pick the *effective* session j, then draw the
# announcement uniformly between the previous close and that close - Monday
# sessions naturally pick up weekend announcements.
event_sym = rng.permutation(symbols)[:N_EVENTS]
event_j = rng.integers(150, len(sess_us) - 30, N_EVENTS)
ann_us = sess_us[event_j - 1] + (rng.random(N_EVENTS) * (sess_us[event_j] - sess_us[event_j - 1])).astype("int64")
treated = rng.random(N_EVENTS) < 0.5

ts_us_all = prices["ts"].astype("int64").to_numpy()
for sym, j, is_t in zip(event_sym, event_j, treated):
    if not is_t:
        continue
    k = np.arange(len(sess_us))
    factor = np.where(k < j, 1.0, (1 + DAY0_SHOCK) * (1 + DRIFT_PER_DAY) ** np.clip(k - j, 0, DRIFT_DAYS))
    mask = prices["symbol"].to_numpy() == sym
    idx = np.searchsorted(sess_us, ts_us_all[mask])
    for col in ("open", "high", "low", "close"):
        prices.loc[mask, col] = prices.loc[mask, col].to_numpy() * factor[idx]

print(f"{treated.sum()} treated / {(~treated).sum()} control events")
print("weekend announcements:", (pd.to_datetime(ann_us, unit="us", utc=True).dayofweek >= 5).sum())

# %% [markdown]
# ## 2. `prices`、`events`、そして取引カレンダーそのものを保存する
#
# 3つとも `ts` を時刻列とする h5i-db テーブルです。events テーブルが持つのは立会日ではなく
# *生の発表タイムスタンプ*です。その対応付けを解くのはデータベースの仕事であって、取り込み
# スクリプトの仕事ではありません。3つ目の `sessions` は取引カレンダーで、セッションの引けごとに
# 1行、手元にある実際のパネルから導きます（つまり祝日は、データがそう言っているものが祝日です）。
# カレンダーをデータとして実体化しておくからこそ、揃える作業が `BusinessDay` の計算をせずに
# ジョインで済みます。

# %%
price_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
db.create_table("prices", price_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    pa.Table.from_pandas(prices.sort_values(["ts", "symbol"]), preserve_index=False).cast(price_schema),
    note="synthetic panel with injected event shocks",
)

events = (
    pd.DataFrame(
        {
            "ts": pd.to_datetime(ann_us, unit="us", utc=True),
            "symbol": event_sym,
            "cal": "XNYS",  # which trading calendar governs this event
            "event_id": np.arange(N_EVENTS),
            "treated": treated,
        }
    )
    .sort_values("ts")
    .reset_index(drop=True)
)
event_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("cal", pa.string()),
        pa.field("event_id", pa.int64()),
        pa.field("treated", pa.bool_()),
    ]
)
db.create_table("events", event_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("events", pa.Table.from_pandas(events, preserve_index=False).cast(event_schema), note="100 announcements")

session_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("cal", pa.string()),
        pa.field("session_id", pa.int64()),
    ]
)
db.create_table("sessions", session_schema, time_column="ts")
db.append(
    "sessions",
    pa.table(
        {
            "ts": pa.array(sess_us, pa.timestamp("us", tz="UTC")),
            "cal": pa.array(["XNYS"] * len(sess_us)),
            "session_id": pa.array(np.arange(len(sess_us))),
        }
    ),
    note="trading calendar derived from the panel",
)
db.tables()

# %% [markdown]
# ## 3. forward の ASOF ジョインで、発表を立会セッションに揃える
#
# `asof_join(left, right, lts, rts, key, 'forward', tolerance)` は、各イベントを発表以降で
# *最初の*セッションに対応づけます。ここでのキーはカレンダー ID です（この研究ではカレンダーは
# 1つですが、グローバルな帳簿なら複数持つことになります）。許容差は生の時刻単位、つまり
# `timestamp[us, UTC]` 列に合わせたマイクロ秒で、7日という緩めの上限を置きます。カレンダーの
# 問題が、何週間も先に黙って対応づけられる代わりに NULL として浮かび上がります。右テーブルの
# 衝突した `ts` は `ts_right` として戻るので、発表と効力セッションが並んで見えます。しかも
# `asof_join(...)` はただの関係なので、その出力をそのまま `prices` に等値結合して day-0 の
# 終値を取れます。全部で1文です。
#
# 運用上の注意を1つ。100行の events テーブルは、6万5千行の価格パネルではなく650行の
# カレンダーに結合します。そうすればジョインは小さいままです。用途に絞った小さなテーブルを
# 導いて、それを結合してください。出力がイベント1件につき1行であることを表明し、セッションの
# 対応付けはカレンダーに対する numpy の `searchsorted` と照合します。この程度の安い表明が
# あれば、揃えの誤りでイベント窓が静かにずれる代わりに、大きな声の失敗になります。

# %%
aligned = db.sql(
    f"""
    SELECT e.event_id, e.symbol, e.treated,
           e.ts        AS announced,
           e.ts_right  AS effective_session,
           p.close     AS day0_close
    FROM asof_join('events', 'sessions', 'ts', 'ts', 'cal', 'forward', {7 * 86_400 * 1_000_000}) e
    JOIN prices p ON p.ts = e.ts_right AND p.symbol = e.symbol
    ORDER BY e.event_id
    """
).to_pandas()
assert len(aligned) == N_EVENTS, "one output row per event"
expected_us = sess_us[np.searchsorted(sess_us, ann_us)]  # independent check
got_us = aligned.sort_values("event_id")["effective_session"].astype("int64").to_numpy()
assert (got_us == expected_us).all(), "ASOF session mapping mismatch"
print("unmatched events:", aligned["effective_session"].isna().sum())
weekend = aligned[pd.to_datetime(aligned["announced"]).dt.dayofweek >= 5]
weekend.assign(
    ann_dow=pd.to_datetime(weekend["announced"]).dt.day_name(),
    eff_dow=pd.to_datetime(weekend["effective_session"]).dt.day_name(),
).head(6)

# %% [markdown]
# 週末の発表はどれも翌月曜のセッションに着地します。カレンダーの計算も `BusinessDay` の
# オフセットも要りません。ジョインが狙うのは価格テーブルに*実際に存在する*セッションなので、
# 祝日でもそのまま動きます。
#
# ## 4. リターンと市場ファクターを SQL で
#
# 銘柄ごとの単純リターンは `lag()` で、等ウェイトの市場リターンはセッションごとのウィンドウ
# 平均で。整列済みストレージの上で計算される1文です。

# %%
rets = db.sql(
    """
    WITH r AS (
        SELECT ts, symbol,
               close / lag(close) OVER (PARTITION BY symbol ORDER BY ts) - 1 AS ret
        FROM prices
    )
    SELECT ts, symbol, ret, avg(ret) OVER (PARTITION BY ts) AS mkt
    FROM r
    WHERE ret IS NOT NULL
    ORDER BY ts, symbol
    """
).to_pandas()
rets.head(3)

# %% [markdown]
# ## 5. マーケットモデルと CAR[-10,+10]
#
# イベントごとに、効力セッションから見て [-130, -11] 立会日でアルファとベータを推定し、
# [-10, +10] にわたって異常リターン `AR = r - (a + b*mkt)` を累積します。クロスセクションの
# 平均に、正規近似のバンド（`1.96 * sd / sqrt(n)`）を添えます。

# %%
panel = rets.pivot(index="ts", columns="symbol", values="ret")
mkt = rets.groupby("ts")["mkt"].first().loc[panel.index]
pos = {ts.value: i for i, ts in enumerate(panel.index)}  # ns epoch -> row

EST, EVT = (-130, -11), (-10, 10)
rel_days = np.arange(EVT[0], EVT[1] + 1)
cars = {}
for _, ev in aligned.iterrows():
    j = pos[ev["effective_session"].value]
    est = slice(j + EST[0], j + EST[1] + 1)
    r_est, m_est = panel[ev["symbol"]].iloc[est], mkt.iloc[est]
    beta = np.cov(r_est, m_est)[0, 1] / np.var(m_est, ddof=1)
    alpha = r_est.mean() - beta * m_est.mean()
    win = slice(j + EVT[0], j + EVT[1] + 1)
    ar = panel[ev["symbol"]].iloc[win].to_numpy() - (alpha + beta * mkt.iloc[win].to_numpy())
    cars[ev["event_id"]] = np.cumsum(ar)

car = pd.DataFrame(cars, index=rel_days).T
car["treated"] = aligned.set_index("event_id").loc[car.index, "treated"]

summary = {}
for grp, g in car.groupby("treated"):
    m = g[rel_days].mean()
    ci = 1.96 * g[rel_days].std() / np.sqrt(len(g))
    summary["treated" if grp else "control"] = (m, ci, len(g))

def group_diff_t(a, b):
    return (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))

t_end, c_end = car[car["treated"]][rel_days[-1]], car[~car["treated"]][rel_days[-1]]
t_d0 = car[car["treated"]][0].sub(car[car["treated"]][-1])
c_d0 = car[~car["treated"]][0].sub(car[~car["treated"]][-1])
print(
    f"mean CAR[+10]  treated {t_end.mean():+.2%}   control {c_end.mean():+.2%}"
    f"   diff t-stat {group_diff_t(t_end, c_end):.1f}"
)
print(
    f"mean day-0 AR  treated {t_d0.mean():+.2%}   control {c_d0.mean():+.2%}"
    f"   diff t-stat {group_diff_t(t_d0, c_d0):.1f}"
)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.5))
for name, color in (("treated", "tab:red"), ("control", "tab:blue")):
    m, ci, n = summary[name]
    ax.plot(rel_days, m * 100, color=color, label=f"{name} (n={n})")
    ax.fill_between(rel_days, (m - ci) * 100, (m + ci) * 100, color=color, alpha=0.15)
ax.axvline(0, color="k", lw=0.7, ls="--")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Average CAR around announcement (market model, 95% band)")
ax.set_xlabel("trading days relative to effective session")
ax.set_ylabel("CAR (%)")
ax.legend()
fig.tight_layout()

# %% [markdown]
# 研究は仕込んだものを復元します。処置群は day-0 に約 +2% 跳ね（1日の差は鋭く、有意性も高い
# です）、day +10 までに約 +3.5% へドリフトします。対照群の CAR はゼロ周りのバンドの中に
# とどまり、プラセボはきちんと振る舞います。ここで、CAR[+10] の t 値が day-0 のそれより
# ずっと弱いことに注目してください。約50件のイベントについて21日ぶんの個別ノイズを累積すると、
# 検出力は急速に削られます。実際のイベントスタディの生死をサンプルサイズが握るのは、まさに
# このためです。イベント前の CAR は両群とも平坦で、揃えの工程が先読みを持ち込んでいないことを
# 裏付けています。発表が価格に触れるのは、効力セッション以降だけです。
#
# ## まとめ
#
# - 実体化した取引カレンダーに対する `asof_join(..., 'forward', tolerance)` が、「発表以降で
#   最初のセッション」を表す正しい原始操作です。祝日に依存せず、許容差を超えた先は静かなゴミでは
#   なく NULL になります。許容差は `timestamp[us, UTC]` の時刻列に合わせた生のマイクロ秒です。
# - `asof_join(...)` は他の関係と同じように合成できます。forward の揃えを `prices` への等値
#   結合につないで day-0 の終値まで、1文で書けました。
# - `events` を一級の h5i-db テーブル（時刻列は生の発表時刻）として持つと、カレンダーの対応付けが
#   クエリの中に住みます。改訂された価格パネルに対して走らせ直せば、対応付けも自動で解き直されます。
# - リターンと等ウェイトの市場ファクターは SQL のウィンドウ1文から出ました。pandas に残るのは
#   イベントごとの OLS ループだけです。
# - 既知の効果を合成データに仕込み、対照群には手を触れないでおくと、このレシピはパイプライン自身を
#   検証するテストになります。

# %%
db.close()
