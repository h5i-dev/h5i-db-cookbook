# %% [markdown]
# # データ品質ゲート: ステージング、ポリシー、プレビューできる修復
#
# 壊れたベンダーファイルに気づく場所として最悪なのは、損益の会議の席です。それを防ぐのは、
# バージョン管理のひねりを加えた昔ながらの ETL の知恵です。**納品はすべてステージングテーブルに
# 着地させ、ゲートを走らせ、合格したものだけを本番へ昇格させる**。h5i-db は各工程を弁明可能に
# します。壊れた生の納品はステージングのバージョンとして記録に残り（恥ではなく証拠です）、
# 修正は変更前後のサンプル付きのプレビューできる plan/apply を通り、データベース層のポリシーが
# 直接の破壊的書き込みを*軽い気持ちでは実行できない*ものにします。人間にもパイプラインの
# エージェントにも等しく、です。

# %%
import numpy as np
import pandas as pd
import pyarrow as pa
import h5i_db

import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("prod_dq"), create=True)

# %% [markdown]
# ## 1. 本番の履歴と、ステージングテーブル
#
# 20銘柄のユニバースについて、約1年ぶんの日次終値が `prices_prod` にあります。ベンダーの次の
# ファイルは `vendor_staging` に着地します。スキーマは同じで、テーブルは別。検証を経ていない
# ものが本番に触れることは決してありません。

# %%
UNIVERSE = [f"STK{i:03d}" for i in range(20)]
panel = cu.make_daily_prices(symbols=UNIVERSE, days=250).to_pandas()

sessions = np.sort(panel["ts"].unique())
delivery_ts = sessions[-1]                      # today's file
history = panel[panel["ts"] < delivery_ts]
delivery_true = panel[panel["ts"] == delivery_ts]  # what the vendor SHOULD send

schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)
cols = ["ts", "symbol", "close", "volume"]
for t in ("prices_prod", "vendor_staging"):
    db.create_table(t, schema, time_column="ts", sort_key=["ts", "symbol"])

db.append(
    "prices_prod",
    pa.Table.from_pandas(history[cols].sort_values(["ts", "symbol"]), schema=schema,
                         preserve_index=False),
    note="production history backfill",
)
db.sql("SELECT count(*) AS rows, count(DISTINCT ts) AS sessions FROM prices_prod").to_pandas()

# %% [markdown]
# ## 2. まずデータベースを締める
#
# `set_policy` は、*直接の*変更経路にゲートをかけるデータベース全体のフラグを切り替えます。
# `direct_write` と `direct_delete` を切ると、言い直しも削除も plan/apply の流れ――プレビューされ、
# 注記が付き、衝突が検査される――を通るしかなくなります。一方、素の `append`（取り込みジョブが
# やってよい唯一のこと）は開いたままです。cron ジョブや LLM エージェントがパイプラインを回す
# ときに欲しいガードレールがこれです。破壊的な経路は、そもそも呼べません。（Python API では
# 削除はすでにプラン専用ですが、このフラグは CLI と将来の直接経路もカバーします。）

# %%
db.set_policy(direct_write=False, direct_delete=False)

# %% [markdown]
# ## 3. 今日の納品は4通りに壊れている
#
# ベンダーのファイルは、ユニバースの半分が欠け、負の価格が1つ、null が1つ、重複行が2つあります。
# これをそのままステージングに着地させます。生のファイルが書き換え不能なステージングバージョンに
# なるので、ベンダーに問い合わせを立てるときにそのまま見せられます。

# %%
corrupt = delivery_true[delivery_true["symbol"].isin(UNIVERSE[:10])].copy()
corrupt.loc[corrupt.index[2], "close"] = -1.0          # sign-flip fat finger
corrupt.loc[corrupt.index[5], "close"] = np.nan        # null print
corrupt = pd.concat([corrupt, corrupt.iloc[[7, 8]]])   # duplicated rows

db.append(
    "vendor_staging",
    pa.Table.from_pandas(corrupt[cols].sort_values(["ts", "symbol"]), schema=schema,
                         preserve_index=False),
    note=f"vendor delivery {pd.Timestamp(delivery_ts).date()} (raw, unvetted)",
)
corrupt[cols].head(8)

# %% [markdown]
# ## 4. ゲート: 5つの検査、すべてステージングに対する SQL
#
# 期待するユニバースに対する網羅性、null と価格の妥当性、重複の検出、そして本番の直近20立会日の
# 平均に対する行数の検査です（安いわりに効く「ベンダーが半分しか送っていないのでは」という警報
# です）。どの検査も*ステージングテーブルの現在のバージョン*を読むので、ゲートの結果はそのバージョンに
# 対して永久に再現できます。

# %%
def run_gate(day: str) -> pd.DataFrame:
    stat = db.sql(
        f"""
        SELECT count(*)                                 AS rows,
               count(DISTINCT symbol)                   AS symbols,
               sum(CASE WHEN close IS NULL THEN 1 ELSE 0 END) AS null_closes,
               min(close)                               AS px_min
        FROM vendor_staging
        WHERE ts >= '{day}T00:00:00Z' AND ts < '{day}T23:59:59Z'
        """
    ).to_pandas().iloc[0]
    trailing = db.sql(
        """
        SELECT avg(n) AS avg_rows FROM (
            SELECT ts, count(*) AS n FROM prices_prod GROUP BY ts ORDER BY ts DESC LIMIT 20
        )
        """
    ).to_pandas()["avg_rows"].iloc[0]

    rows, symbols = int(stat["rows"]), int(stat["symbols"])
    checks = [
        ("universe complete", symbols == len(UNIVERSE), f"{symbols}/{len(UNIVERSE)} symbols"),
        ("no null prices", int(stat["null_closes"]) == 0, f"{int(stat['null_closes'])} nulls"),
        ("prices positive", bool(stat["px_min"] > 0), f"min close {stat['px_min']}"),
        ("no duplicate rows", rows == symbols, f"{rows} rows / {symbols} distinct"),
        ("row count vs history", abs(rows - trailing) / trailing < 0.2,
         f"{rows} vs trailing avg {trailing:.0f}"),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])

day = str(pd.Timestamp(delivery_ts).date())
gate_raw = run_gate(day)
print(f"gate verdict: {'PROMOTE' if gate_raw['passed'].all() else 'REJECT - hold in staging'}")
gate_raw

# %% [markdown]
# ## 5. ポリシーが不可能にする近道
#
# つい取りたくなる修正は「訂正済みファイルでステージングを上書きする」でしょう。`direct_write` を
# 切ってあるので、その経路は `PolicyError` を上げます。エラーのヒントが、認められた流れを指します。
# 午前2時にレビューなしの `write()` で本番データを直す人は、もう出ません。

# %%
fixed = pa.Table.from_pandas(delivery_true[cols].sort_values(["ts", "symbol"]),
                             schema=schema, preserve_index=False)
try:
    db.write("vendor_staging", fixed)
except h5i_db.PolicyError as e:
    print(f"PolicyError  code={e.code}  retryable={e.retryable}")
    print(f"hint: {e.hint}")

# %% [markdown]
# ## 6. 認められたやり方での修復: 立てて、検分して、適用する
#
# `plan_replace_range` が、壊れた日の窓に対して訂正済みの納品をステージングします（範囲の境界は
# 時刻列の単位、つまり生の**マイクロ秒**です）。プランはコミットではありません。要約と変更前後の
# サンプルを持っていて、`apply()` がアトミックに公開する前に目で確かめられますし、変更チケットに
# 添付することもできます。適用時には衝突が検査されます。その間に誰かがステージングにコミットして
# いれば、上書きする代わりに失敗します。

# %%
day_start_us = int(pd.Timestamp(delivery_ts).value // 1000)
plan = db.plan_replace_range(
    "vendor_staging", day_start_us, day_start_us + 1,
    data=fixed, note=f"vendor re-delivery {day}: full universe, corrected prices",
)
plan.summary

# %%
print("BEFORE (broken rows in the affected window):")
print(plan.before_sample.to_pandas().head(6).to_string(index=False))
print("\nAFTER (corrected delivery):")
print(plan.after_sample.to_pandas().head(6).to_string(index=False))

# %%
result = plan.apply()
{k: result[k] for k in ("sequence", "op", "rows_total")}

# %% [markdown]
# ## 7. ゲートを走らせ直し、昇格させる
#
# ゲートが通ったので、納品日の行を本番へ append します。ゲートを通った納品と昇格を結びつける
# 注記を添えます。2つのテーブルのバージョン履歴を並べれば、話の全体が見えます。壊れた生ファイル、
# プレビューされた修正、ゲートを通った昇格です。

# %%
gate_fixed = run_gate(day)
assert gate_fixed["passed"].all(), "gate must pass after remediation"
print("gate verdict: PROMOTE")

staged = db.sql(
    f"SELECT ts, symbol, close, volume FROM vendor_staging "
    f"WHERE ts >= '{day}T00:00:00Z' ORDER BY ts, symbol"
).to_arrow()
db.append("prices_prod", staged.cast(schema), note=f"promoted gated delivery {day}")

[
    {k: v.get(k) for k in ("sequence", "op", "rows", "note")}
    for v in db.versions("vendor_staging") + db.versions("prices_prod")
    if v["op"] != "create"
]

# %% [markdown]
# ## 8. ゲートのレポートを、ダッシュボード風に

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis("off")
col_labels = ["raw delivery", "after remediation"]
cell_text, cell_colors = [], []
for i in range(len(gate_raw)):
    row_text, row_colors = [], []
    for g in (gate_raw, gate_fixed):
        ok = bool(g.loc[i, "passed"])
        row_text.append(("PASS - " if ok else "FAIL - ") + g.loc[i, "detail"])
        row_colors.append("#c8e6c9" if ok else "#ffcdd2")
    cell_text.append(row_text)
    cell_colors.append(row_colors)

tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
               rowLabels=gate_raw["check"].tolist(), colLabels=col_labels,
               loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.5)
ax.set_title(f"Data-quality gate - vendor delivery {day}", pad=18)
fig.tight_layout()

# %% [markdown]
# ## まとめ
#
# - ステージングして、ゲートを通して、昇格させる。パターン自体は昔からのものですが、h5i-db は
#   そこに領収書を足します。壊れた生ファイルも、プレビューされた修正も、昇格も、すべて*注記付きの
#   バージョン*です。インシデントレポートは `versions()` から自ずと書けます。
# - `set_policy(direct_write=False, direct_delete=False)` は「本番を直接いじらないでください」という
#   慣行を `PolicyError` に変えます。append は開いたままなので、取り込みは流れ続けます。
# - `plan_replace_range` は `summary` と変更前後のサンプルを備えたドライランを、そのあとアトミックで
#   衝突検査つきの `apply()` を提供します。プランの範囲の境界が生のマイクロ秒である点をお忘れなく。
# - 直近履歴との行数比較は、いちばん安くていちばんよく捕まえるゲートです。半端なファイルや二重
#   送信は、微妙な破損よりずっとよく起こります。

# %%
db.close()
