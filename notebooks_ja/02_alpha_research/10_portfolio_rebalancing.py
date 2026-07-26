# %% [markdown]
# # バージョン管理された保有でポートフォリオをリバランスする
#
# ポートフォリオの帳簿は、バージョン管理データセットの典型です。リバランス1回が1コミット、
# どのコミットにも注記があり、「3月のリバランス後、何を持っていたか」「1月にどれだけ売買したか」
# といった問いは、表計算の発掘ではなくバージョンへのクエリになります。このレシピでは実データ
# （S&P の30銘柄、2018〜2026年）で月次モメンタムのポートフォリオを回し、h5i-db のバージョン連鎖を
# 正式な記録として使います。
#
# 1. 実際の日次終値から `prices` テーブルを作り、12-1モメンタムを SQL で計算する
# 2. リバランスのたびに1回追記する `holdings` テーブル（リバランス1回につき1コミット、注記
#    付き）と、監査用のスナップショット
# 3. 回転率と売買リストを、SQL の `h5i('holdings', v)` を使って*バージョン差分から*復元する
# 4. NAV とベンチマークの比較、リバランス間のウェイトのずれ、そして任意時点の帳簿の復元

# %%
import numpy as np
import pandas as pd
import pyarrow as pa

import h5i_db
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("alpha_rebalance"), create=True)

# %% [markdown]
# ## 1. 実際の価格を入れて、モメンタムを出す
#
# 流動性の高い30銘柄の分割調整済み終値です。12-1モメンタムのシグナル――直近1か月を除く過去
# 12か月のリターンという古典的な形成ルール――は、SQL のウィンドウ2つで済みます。立会日ベースの
# 系列に対する銘柄ごとの `lag(adj_close, 21)` を `lag(adj_close, 252)` で割るだけです。

# %%
raw = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01").to_pandas()
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

mom_df = db.sql(
    """
    SELECT ts, symbol, adj_close,
           lag(adj_close, 21)  OVER w / lag(adj_close, 252) OVER w - 1 AS mom
    FROM prices
    WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ORDER BY ts, symbol
    """
).to_pandas()
px = mom_df.pivot(index="ts", columns="symbol", values="adj_close")
mom = mom_df.pivot(index="ts", columns="symbol", values="mom")
print(f"{px.shape[0]} trading days x {px.shape[1]} symbols")

# %% [markdown]
# ## 2. リバランスのループ: 1回のリバランスにつき1コミット
#
# 2023年1月から2026年6月まで月次で、モメンタムでランク付けし、上位10銘柄を等ウェイトで保有し、
# その日の終値で売買し、売買代金に10bps を払います。リバランスのたびに、その日の帳簿（保有銘柄
# ごとの株数、価格、ウェイト）を `holdings` に追記します。`note="rebalance YYYY-MM"` を持つ
# アトミックなコミットです。帳簿の*履歴*は追記のみで、任意のバージョンにおける*状態*は「その
# バージョンで最新のタイムスタンプを持つ行」になります。2025年12月のリバランスでは名前付き
# スナップショットも切ります。年末レビューが求めるであろう監査の錨です。

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
# ## 3. バージョン差分から売買リストと回転率を出す
#
# ループの中で「trades」テーブルを書いた箇所はありません。そんなものは存在しなくてよいのです。
# 売買リスト*こそが*、帳簿の連続するバージョンの差だからです。`h5i('holdings', v)` があらゆる
# バージョンを SQL に露出するので、バージョン `v` の最新の帳簿と `v-1` の帳簿を FULL OUTER JOIN
# すれば、新規と手仕舞いも含めて、何を買い何を売ったかがそのまま復元されます。

# %%
head = versions[-1]["sequence"]

def book_ts(v):
    """Timestamp of the latest book inside version v of `holdings`."""
    return db.sql(f"SELECT max(ts) AS m FROM h5i('holdings', {v})").to_pandas()["m"][0]

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
# 同じ差分を集約すれば、リバランスごとの売買代金と往復の回転率が出ます。コミット済みの
# バージョンだけから導かれた監査水準の数字で、ループ自身の台帳と突き合わせて確認します。

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
# 日次 NAV は、保有株数を価格パネルで評価するだけです。ベンチマークは30銘柄の等ウェイト
# ポートフォリオ（日次リバランス、コストなし。厳しい基準です）。リバランスとリバランスのあいだ、
# 帳簿は等ウェイトの目標からずれていきます。L1 のずれ `0.5 * sum|w - w_target|` を見ると、月次
# リバランスが許容しているのこぎり波が現れます。

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
# ## 5. 「帳簿はどうだったか」を復元する3つの道
#
# バージョン管理が買ってくれる監査の物語です。
#
# - **データ時刻で**。ライブのテーブルに `WHERE ts = ...` を当てる（あるリバランス日に組んだ
#   帳簿）
# - **名前付きスナップショットで**。`h5i('holdings', 'eoy-2025')`。2025年12月のリバランスを
#   コミットしたときに固定したものです
# - **コミット時刻で**。`read(as_of=...)` は、あるコミットが先頭だった時点そのままのテーブルを
#   再現します。このノートブックではコミットが数秒差で起きているので、`as_of` は暦の時刻ではなく
#   実行時刻に対応します。各リバランスが実際の引け時刻にコミットされる本番なら、
#   `as_of="2025-12-31T23:00:00Z"` *こそが*暦の問いになります。いずれにせよ O(1) で、再生も
#   restore も要りません。

# %%
eoy = db.sql(
    """
    SELECT symbol, shares, weight FROM h5i('holdings', 'eoy-2025')
    WHERE ts = (SELECT max(ts) FROM h5i('holdings', 'eoy-2025'))
    ORDER BY weight DESC
    """
).to_pandas()
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
# - リバランス1回＝注記付きのコミット1回。これが保有テーブルにとって自然な粒度です。
#   `versions()` がリバランスのログになり、別途 trades テーブルは要りません。売買リストも回転率も
#   `h5i('holdings', v)` と `h5i('holdings', v-1)` の FULL OUTER JOIN から落ちてきて、
#   シミュレーション自身の台帳とぴたり一致しました。
# - 名前付きスナップショット（`'eoy-2025'`）は監査担当に安定した取っ手を渡します。バージョンの
#   固定と `as_of` 読み出しが、「帳簿はどうだったか」に O(1) で答えます。
# - モメンタムとベンチマークの優劣は期間に依存します。モメンタムは常にそうです（ここではリターンで
#   勝ち、Sharpe で負けます）。肝心なのは仕掛けのほうで、コストも回転率（上位10銘柄のモメンタム
#   帳簿で月あたり往復約40%）もずれも、コミット済みの状態から測っています。期待を込めて計算し
#   直したものではありません。
# - 12-1モメンタムは SQL のウィンドウ2つでした。pandas に残るのはシミュレーションのループだけです。

# %%
db.close()
