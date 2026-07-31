# %% [markdown]
# # ファクターパネル：IC、分位、回転率
#
# レシピ 02/05 はファクターを作り、pandas と scipy で採点しました。情報係数とは何かを学ぶには
# 正しいやり方で、ファクターライブラリを運用するやり方としては誤りです。そうしているデスクは
# たいてい、同じ順位相関の実装を3つ抱え、どれが正なのか合意がありません。
#
# `quant.build_panel` が共通の実装です。ファクターを先行リターンに結合し、分位を割り当て、
# `alphalens` の機能一式（IC、減衰、分位リターン、スプレッド、回転率、ファクターポートフォリオの
# アルファ）をエンジンへのクエリとして提供します。パネルはフレームではなく、ピン留めされた
# クエリなので、数字は元のデータバージョンを持ち歩き、10万行がデータベースの外に出ることは
# ありません。

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | ファクター | 銘柄ごと・日付ごとの数値で、リターンを予測すると想定されるもの |
# | 先行リターン | これから n バーぶんのリターン。ファクターを採点する相手 |
# | 情報係数（IC） | ファクターとその後のリターンの順位相関 |
# | 分位バケット | 日付ごとにファクター値で等件数に分けたグループ |
# | スプレッド | 上位バケットのリターンから下位バケットを引いたもの |
# | 回転率 | あるバケットの構成銘柄が、次の日付までにどれだけ入れ替わるか |
# | 順位の自己相関 | 銘柄のファクター順位が時間を通じてどれだけ安定しているか |
# | グループニュートラル | 同じセクター内の銘柄とだけ比べること |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
from h5i_db import col, quant, sql_expr
import cookbook_utils as cu

# %% [markdown]
# ## 1. 価格とファクター、どちらもロング形式
#
# 入力はどちらもロング形式 `(ts, asset, value)` です。ここではファクターが12-1モメンタム、
# 価格が調整後終値で、どちらも同じテーブルから作ります。

# %%
daily = cu.fetch_daily(cu.SP500_EXAMPLES, start="2018-01-01", end="2026-07-01")
db = h5i_db.Database(cu.fresh_db("06_factor_panels_and_reports"), create=True)
prices = daily.sort_by([("ts", "ascending"), ("symbol", "ascending")])
db.create_table("prices", prices.schema, time_column="ts", sort_key=["ts", "symbol"])
db.append("prices", prices, note="30 large caps, 2018-2026")
db.snapshot("prices-v1", tables=["prices"], note="The cut the panel is pinned to")
print(f"{prices.num_rows:,} rows, {daily.to_pandas()['symbol'].nunique()} names")
daily.to_pandas().head(3)

# %% [markdown]
# `build_panel` に渡した遅延フレームはそのまま使われます。つまり **ピンはフレーム側に**
# 必要です。`db.table(name, snapshot=...)` がそれを与えます。ピンのないフレームを `snapshot=`
# と一緒に渡すと、来歴ではピンを主張しながら実際には最新版を読むパネルができてしまいます。

# %%
pinned = db.table("prices", snapshot="prices-v1")
price_frame = pinned.select(ts=col("ts"), asset=col("symbol"), price=col("adj_close"))
factor_frame = (
    pinned.with_columns(
        month_ago=sql_expr("lag(adj_close, 21)").over(partition_by="symbol", order_by="ts"),
        year_ago=sql_expr("lag(adj_close, 252)").over(partition_by="symbol", order_by="ts"),
    )
    .with_columns(momentum=col("month_ago") / col("year_ago") - 1)
    .select(ts=col("ts"), asset=col("symbol"), factor=col("momentum"))
)
print(factor_frame.sql())

# %% [markdown]
# ## 2. パネル
#
# `periods` はバー単位の先行リターンの期間なので、日次データでは1日、1週間、1か月にあたります。
# `quantiles=5` は各日付を等件数の5バケットに分けます。`filter_zscore` は平均から標準偏差20を
# 超えて離れたファクター値を落とします。モデリング上の選択というより、データ不良のフィルタです。

# %%
panel = quant.build_panel(
    db,
    factor_frame,
    price_frame,
    periods=(1, 5, 21),
    quantiles=5,
    filter_zscore=20.0,
    max_loss=0.35,
)
print(panel)
panel.collect().to_pandas().head()

# %% [markdown]
# ## 3. パネルが捨てたもの
#
# `loss_report` は `alphalens` が印字し、多くの利用者が読み飛ばす会計です。行は先行リターンとの
# 結合（十分先の価格がない、ファクター値が有限でない）で失われ、分位付けでもう一度失われます。
# ファクターの多くが最後まで残らなかったとき、`max_loss` はパネルそのものを拒否します。自分の
# 値の生き残った40%で採点したファクターは、別のファクターだからです。

# %%
report = panel.loss_report()
print(f"factor rows          {report['initial']:,}")
print(f"after forward returns{report['after_forward_returns']:>8,}  "
      f"({report['forward_returns']:.2%} lost)")
print(f"after binning        {report['after_binning']:>8,}  ({report['binning']:.2%} lost)")
print(f"total loss           {report['total']:.2%} against a limit of {report['max_loss']:.0%}")

try:
    quant.build_panel(db, factor_frame, price_frame, periods=(1, 5, 21), max_loss=0.001)
except quant.MaxLossExceededError as error:
    print(f"\nrefused at a 0.1% limit: {str(error)[:120]}...")

# %% [markdown]
# ## 4. 情報係数
#
# ファクターと各先行リターンの、日付ごとの順位相関です。順位はエンジンの `cs_rank` を使い、
# pandas と同じ同順位の扱いに従うので、SQL の `percent_rank` ではなく、
# `scipy.stats.spearmanr` と一致します。

# %%
ic = panel.ic().to_pandas()
print(f"{len(ic):,} dates scored")
print(pd.DataFrame({"mean IC": panel.mean_ic().to_pandas().iloc[0]}).round(4).to_string())
ic.tail(3).set_index("ts").round(4)

# %% [markdown]
# `ic_decay` は同じ問いを期間横断で1クエリにしたものです。シグナルはどれだけ生き延びるのか。
# 平均 IC 0.02 がシグナルなのか丸め誤差なのかを語るのが `t_stat` の列です。

# %%
decay = panel.ic_decay().to_pandas()
decay.round(4)

# %% [markdown]
# 月次へのリサンプルは、ノイズの多い日次系列を眺められるものに変えます。弱いファクターを
# 強くはしません。弱いファクターを読めるようにするだけです。

# %%
monthly_ic = panel.mean_ic(by="1mo").to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(monthly_ic["bucket"], monthly_ic["ic_21"], width=20)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Monthly mean IC, 21-day horizon")
ax.set_xlabel("Month")
ax.set_ylabel("Information coefficient")
fig.tight_layout()

# %% [markdown]
# ## 5. 分位リターンとスプレッド
#
# IC は順位づけに情報があると言います。分位の表は、その情報が実際に売買できる分布の部分に
# あるかを言い、標準誤差は、2つのバケットの差がそもそも差なのかを言います。

# %%
quantiles = panel.quantile_returns().to_pandas()
quantiles.round(5)

# %% [markdown]
# `spread` は日付ごとの上位マイナス下位で、両者を合わせた標準誤差が付きます。パネルの中で
# いちばん売買可能な系列に近いものですが、それでもセクション04のコストはすべてグロスのままです。

# %%
spread = panel.spread().to_pandas()
summary = {
    f"spread_{period}": {
        "mean": spread[f"spread_{period}"].mean(),
        "t-stat": spread[f"spread_{period}"].mean()
        / spread[f"spread_{period}"].std()
        * len(spread) ** 0.5,
    }
    for period in (1, 5, 21)
}
pd.DataFrame(summary).round(4)

# %%
cumulative = panel.cumulative_returns(period=21).to_pandas()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(cumulative["ts"], cumulative["cumulative_return"], linewidth=1.6)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Factor-weighted portfolio, 21-day horizon")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative return")
fig.tight_layout()

# %% [markdown]
# `alpha_beta` はファクターポートフォリオを等ウェイトのユニバースに回帰し、期間ごとに1行返します。
# 「これは単なるマーケットへのエクスポージャーではないのか」という問いに、議論を待たず答えが
# 返ってきます。

# %%
pd.DataFrame(panel.alpha_beta()).round(4)

# %% [markdown]
# ## 6. 保有にかかるコスト
#
# 回転率と順位の自己相関は、IC がレシピ 04/11 との接触に耐えるかを決める2つの数字です。上位
# バケットが毎日まるごと入れ替わるファクターは、自分の代金を払うために途方もない IC を必要と
# します。

# %%
turnover = panel.turnover(period=1).to_pandas()
autocorrelation = panel.rank_autocorrelation(period=1).to_pandas()
print(f"mean daily turnover by bucket:")
print(turnover.groupby("factor_quantile")["turnover"].mean().round(4).to_string())
print(f"\nmean rank autocorrelation: {autocorrelation['autocorrelation'].mean():.4f}")
print(f"implied average holding period: "
      f"{1 / max(turnover['turnover'].mean(), 1e-9):.1f} days")

# %% [markdown]
# ## 7. セクターと、ニュートラル化が変えるもの
#
# 1つのセクターに偏るモメンタムファクターは、手間をかけたセクターの賭けです。`group` は銘柄から
# グループへの対応を受け取ります。`by_group` はセクターごとに採点し、`group_adjust` は採点前に
# セクター内で先行リターンの平均を引きます。これは「セクターの *中* でも効くのか」という問いです。

# %%
SECTORS = {
    "AAPL": "tech", "MSFT": "tech", "NVDA": "tech", "GOOGL": "tech", "META": "tech",
    "CSCO": "tech", "IBM": "tech", "V": "financials", "JPM": "financials",
    "BAC": "financials", "GS": "financials", "BRK-B": "financials",
    "XOM": "energy", "CVX": "energy", "UNH": "health", "JNJ": "health",
    "MRK": "health", "ABBV": "health", "PG": "staples", "PEP": "staples",
    "KO": "staples", "COST": "staples", "WMT": "staples", "AMZN": "discretionary",
    "HD": "discretionary", "MCD": "discretionary", "DIS": "discretionary",
    "CAT": "industrials", "GE": "industrials", "T": "telecom",
}
grouped = quant.build_panel(
    db,
    factor_frame,
    price_frame,
    periods=(1, 5, 21),
    quantiles=5,
    group=SECTORS,
    max_loss=0.35,
)
by_sector = grouped.mean_ic(by_group=True).to_pandas()
print(by_sector.round(4).to_string(index=False))

# %% [markdown]
# `telecom` は構成銘柄が1つで、1銘柄のグループには順位づけするクロスセクションがないので、
# IC はゼロにはならず、未定義になります。これが正しい答えであり、グループの統計量はグループの
# 大きさを引き継ぐという、役に立つ注意でもあります。

# %%
plain = grouped.mean_ic().to_pandas().iloc[0]
neutral = grouped.mean_ic(group_adjust=True).to_pandas().iloc[0]
pd.DataFrame({"raw": plain, "sector-neutral": neutral}).round(4)

# %% [markdown]
# ここでの答えは居心地が悪く、はっきり書く価値があります。このファクターの情報係数の大半は
# セクターの賭けです。セクター内で先行リターンの平均を引くと、1日 IC は半分以下になり、
# 1か月 IC は負に転じます。このユニバースでモメンタムを運用していたら、そのとき走っていた
# セクターをロングしていたことになります。

# %% [markdown]
# ## 8. レポート
#
# `quant.factor_report` は `alphalens` が印字するのと同じページを、来歴のヘッダーを付けた
# 自己完結の HTML ファイルとして描きます。

# %%
html = quant.factor_report(panel, path="data/cache/momentum-factor-report.html")
payload = quant.report_payload(panel)
print(f"wrote {len(html):,} bytes")
print(f"tables {[table['id'] for table in payload.get('tables', [])]}")
print(f"charts {[chart['id'] for chart in payload.get('charts', [])]}")

# %% [markdown]
# ## 9. パネルは結果ではありません。クエリです
#
# ここまでで、パネルを Python 側に実体化した箇所はありません。`frame` は遅延フレームとして
# 公開するので、後続のクエリがエンジンの中で絞り込み・結合・集約でき、`sql()` は実際に走る
# SQL をそのまま印字します。

# %%
recent = (
    panel.frame.filter(col("ts") >= "2026-01-01")
    .group_by("factor_quantile")
    .agg(days=col("fwd_21").count(), mean_fwd_21=col("fwd_21").mean())
    .sort("factor_quantile")
)
print(f"panel SQL is {len(panel.sql()):,} characters; nothing was collected to get here")
recent.to_pandas().round(5)

# %% [markdown]
# ## まとめ
#
# - `build_panel` はファクターと先行リターンの結合と分位付けを一度だけ行い、それ以外はすべて
#   その1つの定義に対するクエリです。
# - IC を読む前に `loss_report` を読んでください。生き残った行で採点したファクターは、自分が
#   作ったファクターではありません。
# - `ic_decay` は期間と t 値を1クエリで返します。「シグナルがある」と「数字がある」の違いは
#   そこにあります。
# - IC が自分の代金を払えるかは、回転率と順位の自己相関が決めます。
# - `group` はセクターの賭けを検証可能な主張に変え、`group_adjust` はセクターの中でも効くのかを
#   問います。
# - パネルはピン留めされたクエリです。読める大きさの集約が返るまで、エンジンからは何も出ません。

# %%
db.close()
