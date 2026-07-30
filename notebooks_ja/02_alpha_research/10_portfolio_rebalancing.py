# %% [markdown]
# # バージョン管理された保有でリバランスする
#
# ポートフォリオの帳簿は、バージョン管理されたデータセットの典型です。リバランスのたびに
# コミットが1つ、コミットにはノートが付き、「3月のリバランス後は何を持っていたか」「1月は
# どれだけ売買したか」といった問いは、表計算の発掘作業ではなくバージョンへのクエリになります。
#
# このレシピは月次のモメンタムポートフォリオを実データ、2018〜2026 年の S&P 30銘柄で走らせ、
# バージョンの連鎖を記録の正本として使います。
#
# 1. 実際の日足から `prices` テーブルを作り、12-1 モメンタムを SQL で計算する
# 2. リバランスのたびに `holdings` テーブルへ追記する。ノート付きのコミット1件ずつと、監査用の
#    スナップショット
# 3. `h5i('holdings', v)` を使い、ターンオーバーと売買明細を*バージョンの差分から*復元する
# 4. NAV をベンチマークと比べ、リバランス間のウェイトのずれを測り、帳簿を時点で復元する

# %% [markdown]
# ## ここで使う用語
#
# | 用語            | 意味 |
# | ------------- | --- |
# | ホールディングス      | ブックの中身。何をどのウェイトで持っているか |
# | リバランス         | 目標ウェイトを計算し直し、そこへ向けて定期的に売買すること |
# | 回転率（turnover） | 1回のリバランスでブックのどれだけを売買するか。取引費用を直接決める |
# | ウェイトのドリフト     | 売買ではなく価格変動によって、リバランスの間にウェイトがずれること |
# | NAV           | 純資産価値。ある時点でのブック全体の価値 |
# | ベンチマーク        | 戦略を比較する相手になるパッシブな代替案 |
# | バージョン差分       | コミット済みの2バージョンを比べて、何が変わったかを復元すること |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
from h5i_db import col, sql_expr
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_rebalance"), create=True)

# %% [markdown]
# ## 1. データ
#
# `cu.fetch_daily` が返すのは、流動性の高い30銘柄の 2018〜2026 年の実際の日足です。1行が1銘柄
# 1セッションです。
#
# | 列 | 型 | 意味 |
# | --- | --- | --- |
# | `ts` | `timestamp[us, tz=UTC]` | セッションの日付 |
# | `symbol` | `string` | 銘柄コード |
# | `open`、`high`、`low`、`close` | `float64` | セッションの価格 |
# | `adj_close` | `float64` | 分割と配当で調整済みの終値 |
# | `volume` | `int64` | 出来高（株数） |

# %%
raw = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01").to_pandas()
print(f"{len(raw):,} rows x {raw.shape[1]} columns, {raw['symbol'].nunique()} symbols")
raw.head()

# %% [markdown]
# 使うのは `ts`、`symbol`、`adj_close` です。12-1 モメンタムのシグナルは、直近1か月を除いた
# 過去12か月のリターン、つまり古典的な形成ルールです。SQL では銘柄ごとに `lag()` のウィンドウ
# を2つ、営業日の系列に対して `lag(adj_close, 21)` を `lag(adj_close, 252)` で割ります。

# %%
px_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("adj_close", pa.float64()),
    ]
)
db.create_table("prices", px_schema, time_column="ts", sort_key=["ts", "symbol"])
db.append(
    "prices",
    pa.Table.from_pandas(
        raw[["ts", "symbol", "adj_close"]].sort_values(["ts", "symbol"]), preserve_index=False
    ).cast(px_schema),
    note="30 S&P names, 2018-2026, split-adjusted",
)

def lag_px(n: int):
    return sql_expr(f"lag(adj_close, {n})").over(partition_by="symbol", order_by="ts")


mom_df = (
    db.table("prices")
    .select("ts", "symbol", "adj_close", mom=lag_px(21) / lag_px(252) - 1)
    .sort(["ts", "symbol"])
).to_pandas()
px = mom_df.pivot(index="ts", columns="symbol", values="adj_close")
mom = mom_df.pivot(index="ts", columns="symbol", values="mom")
print(f"{px.shape[0]} trading days x {px.shape[1]} symbols")

# %% [markdown]
# ## 2. リバランスのループ: 1回のリバランスにつき1コミット
#
# 2023年1月から2026年6月まで毎月、モメンタムで順位付けし、上位10銘柄を等ウェイトで持ち、
# そのセッションの終値で売買し、取引金額に 10bps を払います。
#
# リバランスのたびに、その日の帳簿、つまり保有銘柄ごとの株数・価格・ウェイトを `holdings` に
# 追記します。`note="rebalance YYYY-MM"` を持つアトミックなコミット1件です。
#
# 帳簿の*履歴*は追記専用です。任意のバージョンにおける*状態*は、そのバージョンの最新
# タイムスタンプにある行の集合です。
#
# 2025年12月のリバランスでは、名前付きスナップショットも切ります。年度末レビューが求めるであろう
# 監査の錨です。

# %%
hold_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("shares", pa.float64()),
        pa.field("price", pa.float64()),
        pa.field("weight", pa.float64()),
    ]
)
db.create_table("holdings", hold_schema, time_column="ts", sort_key=["ts", "symbol"])

month_ends = (
    px.loc["2023-01-01":"2026-06-30"].groupby([px.loc["2023-01-01":"2026-06-30"].index.year,
                                               px.loc["2023-01-01":"2026-06-30"].index.month])
    .apply(lambda g: g.index.max())
    .tolist()
)

TOP_N, COST_BPS = 10, 10
nav, shares = 10_000_000.0, pd.Series(dtype=float)
ledger = []  # (date, nav_pre, cost, turnover) per rebalance

for reb_ts in month_ends:
    p = px.loc[reb_ts]
    nav_pre = float((shares * p).sum()) if len(shares) else nav
    winners = mom.loc[reb_ts].dropna().nlargest(TOP_N).index
    # two-pass so costs are funded out of NAV before sizing
    target = pd.Series(nav_pre / TOP_N, index=winners) / p[winners]
    traded = (target.reindex(p.index, fill_value=0.0) - shares.reindex(p.index, fill_value=0.0)).abs()
    cost = float((traded * p).sum()) * COST_BPS / 1e4
    nav_post = nav_pre - cost
    shares = pd.Series(nav_post / TOP_N, index=winners) / p[winners]
    ledger.append((reb_ts, nav_pre, cost, float((traded * p).sum()) / nav_pre))

    ym = f"{reb_ts.year}-{reb_ts.month:02d}"
    book = pd.DataFrame(
        {
            "ts": reb_ts,
            "symbol": winners,
            "shares": shares[winners].to_numpy(),
            "price": p[winners].to_numpy(),
            "weight": (shares[winners] * p[winners]).to_numpy() / nav_post,
        }
    ).sort_values(["ts", "symbol"])
    db.append("holdings", pa.Table.from_pandas(book, preserve_index=False).cast(hold_schema),
              note=f"rebalance {ym}")
    if ym == "2025-12":
        db.snapshot("eoy-2025", tables=["holdings"], note="year-end audit anchor")

versions = db.versions("holdings")
print(f"{len(month_ends)} rebalances -> {len(versions)} versions; last notes:")
for v in versions[-3:]:
    print(f"  v{v['sequence']:>2}  {v['op']:<7} {v['rows']:>3} rows  note={v.get('note')!r}")

# %% [markdown]
# ## 3. バージョンの差分から売買明細とターンオーバーを出す
#
# ループの中で「売買」テーブルを書いた箇所はありませんし、そんなものは要りません。売買明細
# *そのもの*が、帳簿の連続するバージョンの差だからです。
#
# `h5i('holdings', v)` がどのバージョンも SQL に露出するので、バージョン `v` の最新の帳簿を
# バージョン `v-1` のものと FULL OUTER JOIN すれば、何を買って何を売ったかが、新規と撤退を
# 含めて正確に復元できます。

# %%
head = versions[-1]["sequence"]

def book_ts(v):
    """Timestamp of the latest book inside version v of `holdings`."""
    return db.table("holdings", version=v).select(m=col("ts").max()).to_pandas()["m"][0]

# Resolve each version's book timestamp once and inline it as a literal -
# the resolved read point then appears verbatim in the SQL, which is what
# you want in an audit artifact (and it saves re-running the subquery in
# every relation). px_now marks every leg (including exits) at the
# rebalance-date close.
def book_diff_sql(v_new, v_old, select):
    return f"""
    WITH cur AS (SELECT symbol, shares, price FROM h5i('holdings', {v_new})
                 WHERE ts = TIMESTAMP '{book_ts(v_new)}'),
         prev AS (SELECT symbol, shares, price FROM h5i('holdings', {v_old})
                  WHERE ts = TIMESTAMP '{book_ts(v_old)}'),
         px_now AS (SELECT symbol, adj_close FROM prices
                    WHERE ts = TIMESTAMP '{book_ts(v_new)}')
    {select}
    """

# A FULL OUTER JOIN with COALESCE across both sides, over three pinned
# relations: the builder's binary l/r aliasing would obscure this rather than
# clarify it, so the diff stays SQL.
trade_list = db.sql(
    book_diff_sql(
        head,
        head - 1,
        """
        SELECT COALESCE(cur.symbol, prev.symbol)               AS symbol,
               COALESCE(prev.shares, 0)                        AS shares_before,
               COALESCE(cur.shares, 0)                         AS shares_after,
               COALESCE(cur.shares, 0) - COALESCE(prev.shares, 0) AS trade_shares,
               px_now.adj_close                                AS price
        FROM cur FULL OUTER JOIN prev ON cur.symbol = prev.symbol
        JOIN px_now ON px_now.symbol = COALESCE(cur.symbol, prev.symbol)
        ORDER BY trade_shares
        """,
    )
).to_pandas()
assert len(trade_list) >= TOP_N, "diff must cover holds, entries and exits"
trade_list.round(1)

# %% [markdown]
# 同じ差分を集計すれば、リバランスごとの取引金額と両方向のターンオーバーが出ます。コミット
# されたバージョンだけから導かれた、監査に耐える数字です。これをループ自身の台帳と突き合わせ
# ます。

# %%
turnover_rows = []
for v in range(2, head + 1):
    r = db.sql(
        book_diff_sql(
            v,
            v - 1,
            """
            , j AS (SELECT COALESCE(cur.shares, 0) - COALESCE(prev.shares, 0) AS d,
                           px_now.adj_close AS p2
                    FROM cur FULL OUTER JOIN prev ON cur.symbol = prev.symbol
                    JOIN px_now ON px_now.symbol = COALESCE(cur.symbol, prev.symbol))
            SELECT sum(abs(d) * p2)                          AS traded,
                   (SELECT sum(shares * price) FROM cur)     AS book_value
            FROM j
            """,
        )
    ).to_pandas()
    turnover_rows.append(
        {"version": v, "traded": r["traded"][0], "turnover": r["traded"][0] / r["book_value"][0]}
    )
turn = pd.DataFrame(turnover_rows)
turn["date"] = [d for d, *_ in ledger[1:]]

ledger_turnover = np.array([t for *_, t in ledger[1:]])
assert np.allclose(turn["turnover"], ledger_turnover, atol=2e-3), "version diff vs ledger"
print(f"version-diff turnover matches the loop's ledger; mean two-way turnover "
      f"{turn['turnover'].mean():.1%} per month")

# %% [markdown]
# ## 4. NAV、ベンチマーク、リバランス間のずれ
#
# 日次の NAV は、保有株数を価格パネルで評価するだけです。ベンチマークは30銘柄の等ウェイト
# ポートフォリオで、毎日リバランスしコストは取らないので、厳しい基準になります。
#
# リバランスの合間に帳簿は等ウェイトの目標からずれていきます。L1 のずれ
# `0.5 * sum|w - w_target|` が、月次のリバランスが許容しているのこぎり波を見せます。

# %%
start_ts = month_ends[0]
days = px.index[px.index >= start_ts]
nav_daily, drift_daily = [], []
k = -1
for d in days:
    if k + 1 < len(month_ends) and d >= month_ends[k + 1]:
        k += 1
        reb_us = int(month_ends[k].value // 1000)
        book_shares = (
            db.read("holdings", time_start=reb_us, time_end=reb_us + 1)
            .to_pandas()
            .set_index("symbol")["shares"]
        )
    vals = book_shares * px.loc[d, book_shares.index]
    nav_daily.append(vals.sum())
    w = vals / vals.sum()
    drift_daily.append(0.5 * (w - 1 / TOP_N).abs().sum())

nav_s = pd.Series(nav_daily, index=days)
drift_s = pd.Series(drift_daily, index=days)
bench = (1 + px.loc[days].pct_change().mean(axis=1).fillna(0)).cumprod() * 10_000_000

def stats(series):
    r = series.pct_change().dropna()
    dd = (series / series.cummax() - 1).min()
    return {
        "ann. return": (series.iloc[-1] / series.iloc[0]) ** (252 / len(r)) - 1,
        "ann. vol": r.std() * np.sqrt(252),
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "max drawdown": dd,
    }

pd.DataFrame({"momentum top-10": stats(nav_s), "equal-weight 30": stats(bench)}).T.round(3)

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(nav_s.index, nav_s / 1e6, label="momentum top-10 (10 bps costs)", color="tab:blue")
axes[0].plot(bench.index, bench / 1e6, label="equal-weight 30 (no costs)", color="tab:gray", lw=1)
axes[0].set_title("NAV: monthly momentum portfolio vs equal-weight benchmark")
axes[0].set_ylabel("NAV ($M)")
axes[0].legend()
axes[1].plot(drift_s.index, drift_s * 100, color="tab:orange", lw=0.9)
axes[1].bar(turn["date"], turn["turnover"] * 100, width=8, color="tab:blue", alpha=0.6,
            label="two-way turnover at rebalance")
axes[1].set_title("Weight drift between rebalances (L1, %) and turnover")
axes[1].set_ylabel("%")
axes[1].set_xlabel("date")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# ## 5. 「帳簿はどうだったか」を3通りで復元する
#
# バージョニングが買ってくれる監査の物語です。
#
# - **データ時刻で。** ライブのテーブルに `WHERE ts = ...` をかければ、あるリバランス日に組んだ
#   帳簿が出ます。
# - **名前付きスナップショットで。** `h5i('holdings', 'eoy-2025')` は、2025年12月のリバランスを
#   コミットしたときに固定したものです。
# - **コミット時刻で。** `read(as_of=...)` は、あるコミットが先頭だった時点のテーブルをそのまま
#   再現します。このノートブックではコミットが数秒間隔で起きているので、`as_of` が指すのは
#   実行時刻です。カレンダー時刻ではありません。本番で各リバランスが実際の引け後にコミット
#   されるなら、
#   `as_of="2025-12-31T23:00:00Z"` は*そのまま*カレンダーの問いになります。
#
# 3つとも O(1) です。再生も復元も要りません。

# %%
# A scalar subquery has no verb, so resolve the latest book timestamp first
# and filter against it - one round trip more, and far easier to read.
eoy_book = db.table("holdings", snapshot="eoy-2025")
eoy_ts = eoy_book.select(m=col("ts").max()).to_pandas()["m"][0]

eoy = (
    eoy_book.filter(col("ts") == eoy_ts.isoformat())
    .select("symbol", "shares", "weight")
    .sort("weight", descending=True)
    .to_pandas()
)
print("book at snapshot 'eoy-2025' (December 2025 rebalance):")
print(eoy.round(4).to_string(index=False))

v_dec = next(v["sequence"] for v in versions if v.get("note") == "rebalance 2025-12")
as_of_iso = pd.Timestamp(
    next(v for v in versions if v["sequence"] == v_dec)["committed_at_ns"], unit="ns", tz="UTC"
).isoformat()
by_version = db.read("holdings", version=v_dec)
by_time = db.read("holdings", as_of=as_of_iso)
assert by_version.equals(by_time)
print(f"\nread(version={v_dec}) == read(as_of='{as_of_iso}'): {by_version.num_rows} rows")

# %% [markdown]
# ## まとめ
#
# - ノート付きのコミット1件を1回のリバランスに対応させるのが、保有テーブルの自然な粒度です。
#   `versions()` がそのままリバランスの記録になり、別の売買テーブルは要りません。売買明細も
#   ターンオーバーも `h5i('holdings', v)` と `h5i('holdings', v-1)` の FULL OUTER JOIN から
#   出てきますし、シミュレーション自身の台帳と厳密に一致しました。
# - `'eoy-2025'` のような名前付きスナップショットは、監査人に安定した取っ手を渡します。
#   バージョンのピン留めと `as_of` 読みが、「帳簿はどう見えていたか」に O(1) で答えます。
# - モメンタム対ベンチマークの結果は、モメンタムがいつもそうであるように期間依存です。ここでは
#   リターンで勝ち、シャープで負けています。主題は仕掛けのほうです。コスト、ターンオーバー
#   （上位10銘柄のモメンタムの本で月あたり両方向およそ40%）、ずれのいずれも、コミットされた
#   状態から測っています。期待を込めて計算し直したものではありません。
# - 12-1 モメンタムは SQL の `lag()` ウィンドウ2つでした。pandas に残るのはシミュレーションの
#   ループだけです。

# %%
db.close()
