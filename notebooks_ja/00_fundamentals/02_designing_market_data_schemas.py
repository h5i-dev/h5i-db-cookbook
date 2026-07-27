# %% [markdown]
# # マーケットデータのスキーマ設計
#
# h5i-db のテーブルは、Arrow スキーマと時刻列の組で、実体はバージョン管理された
# マニフェストの下に置かれた、書き換え不能で時刻順の Parquet セグメントです。だから
# スキーマ設計だけは、あとから安く直せません。選んだ型がストレージサイズを決め、どの
# SQL 演算子がきれいに当たるかを決め、`time_column` と `sort_key` を通じてスキャンの
# プルーニング効率や ASOF ジョイン・バー集計の速度まで決めます。このレシピでは、株式
# デスクがまず用意する3つのテーブル（ティック、気配、日足）を設計しながら型選択の理由を
# 説明し、続いて*厳格な append 契約*を試します。h5i-db が入口で何を拒むのか、そして
# チームで共有するリサーチ用データベースにとって、なぜその厳しさが利点になるのかを見ます。

# %%
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_schemas"), create=True)

# %% [markdown]
# ## 1. ティックレベルの約定データ、型の選び方
#
# 列ごとに、判断の分かれ目を挙げます。
#
# - **`ts` — `timestamp[us, tz=UTC]`、NULL 不可。** マイクロ秒あれば統合フィードの
#   粒度をまかなえます。タイムゾーン付き UTC にしておくと、夏時間まわりのバグが
#   まとめて消えます（取引所時刻への変換は保存時ではなく、クエリ時に `time_bucket` の
#   タイムゾーン引数でやります）。NULL 不可にするのは、タイムスタンプのない約定は
#   データとして成立しないからです。この選択のおかげで、API の他の場所で生値を取る引数
#   （プランの範囲、ギャップ補完の刻み、ASOF の許容差）もすべてマイクロ秒で揃います。
# - **`price` — `float64`。** 古典的なトレードオフです。`decimal128` は誤差がなく、
#   決済台帳ならこちらが正解でしょう。ただ float64 でも、セント刻みの株価なら
#   $10^{13}$ 程度まで往復して値が変わりません。分析系の関数（`vwap`、`ewma`、
#   `stddev`、`corr`）が想定しているのも float64 で、サイズは半分で済みます。リサーチ層の
#   慣習は float64、decimal は帳簿と記録の系に取っておきましょう。
# - **`size` — `int64`。** float は避けます。株数は整数ですし、int64 の余裕があれば
#   端株からインデックスリバランスのクロスまで、1つの型でまかなえます。
# - **`symbol`、`exchange`、`side` — `utf8`。** カーディナリティの低い文字列は
#   Parquet セグメントの中で自動的に辞書エンコードされます。スキーマ層で素の `utf8` を
#   使ってもディスク上のコストはほとんど変わらず、スキーマは単純なまま保てます。

# %%
trades_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.int64()),
        pa.field("exchange", pa.string()),
        pa.field("side", pa.string()),
    ]
)

quotes_schema = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string()),
        pa.field("bid", pa.float64()),
        pa.field("ask", pa.float64()),
        pa.field("bid_size", pa.int64()),
        pa.field("ask_size", pa.int64()),
    ]
)

bars_1d_schema = pa.schema(
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

# %% [markdown]
# ## 2. `time_column` と `sort_key`
#
# `time_column="ts"` の宣言は飾りのメタデータではありません。h5i-db がすべてを
# その上に組み立てる契約です。
#
# - セグメントは **`ts` で整列**して保存され、各セグメントの時間範囲がマニフェストに
#   記録されます。だから `WHERE ts BETWEEN ...` の述語は、セグメントを開かないまま
#   丸ごと読み飛ばせます（マニフェストプルーニング）。
# - **ASOF ジョイン**と **`time_bucket` のバー集計**は、クエリのたびにティックを
#   並べ直すのではなく、その整列順の上をストリーミングで流れます。
# - 「保存済みのどれよりも後」がきちんと定義できるので、`append` はフィードとしての
#   意味論を強制できます（後述）。
#
# `sort_key=["ts", "symbol"]` は、タイムスタンプが同着になったとき*その内側*で使う
# 二次的な順序を足します。そしてより効くのが、同じ銘柄の行がひとかたまりに並ぶので、
# 銘柄別のスキャンが連続した領域だけを触れば済むようになる点です。決まりは1つ、ソート
# キーの先頭は時刻列でなければなりません。その次には、いちばんよく絞り込みに使う列
# （たいていは `symbol`）を置きます。

# %%
db.create_table("trades", trades_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("quotes", quotes_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("bars_1d", bars_1d_schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %%
trades = cu.make_trades(days=2, trades_per_day=5_000)
quotes = cu.make_quotes(days=2, quotes_per_day=8_000)
daily = cu.make_daily_prices(symbols=["AAPL", "MSFT", "NVDA"], days=250)

for name, data in [("trades", trades), ("quotes", quotes), ("bars_1d", daily)]:
    commit = db.append(name, data, note="initial load")
    print(f"{name:8s} v{commit['sequence']}: {commit['rows_total']:>7,} rows, "
          f"{commit['segments_total']} segment(s)")

db.schema("trades")

# %% [markdown]
# スキーマが働いていることを、短いクエリで確かめます。quotes テーブルから銘柄別の
# 気配スプレッドをそのままベーシスポイントで出します。素の `utf8` の symbol 列でも、
# 期待どおりにグループ化も絞り込みもできます。この層に特別な「カテゴリ型」は要りません。

# %%
mid = (col("ask") + col("bid")) / 2

(
    db.table("quotes")
    .group_by("symbol")
    .agg(
        quotes=count_star(),
        avg_spread_bps=(((col("ask") - col("bid")) / mid).mean() * 1e4).round(2),
        min_bid=col("bid").min().round(2),
        max_ask=col("ask").max().round(2),
    )
    .sort("symbol")
    .to_pandas()
)

# %% [markdown]
# ## 3. 厳格な append が拒むもの
#
# `append` はフィードとしての意味論を持ちます。バッチは宣言したスキーマと一致し、時刻列で
# 整列していて、テーブルの現在の最大タイムスタンプ以降から始まらなければなりません。
# それ以外は、何かが書き込まれる*前に*拒否されます。append が失敗しても、テーブルの
# 先頭は動きません。チームで共有するデータベース（あるいはエージェントが書き込む
# データベース）では、この厳しさこそが狙いです。壊れたベンダーファイルや順序の狂った
# バックフィルは、下流のリサーチを静かに汚染する代わりに、取り込みの時点で声を上げて
# 失敗します。
#
# h5i-db の例外はどれも、機械可読な `.code` と `.retryable` フラグを持ち、多くの場合は
# 推奨される直し方が `.hint` に入っています。
#
# **拒否1 — スキーマ不一致。** ベンダーが「気を利かせて」価格を float32 で送ってきて、
# `side` 列を落とした場合です。

# %%
bad_schema_batch = pa.table(
    {
        "ts": trades["ts"][:100],
        "symbol": trades["symbol"][:100],
        "price": trades["price"][:100].cast(pa.float32()),   # wrong width
        "size": trades["size"][:100],
        "exchange": trades["exchange"][:100],
        # 'side' column missing entirely
    }
)
try:
    db.append("trades", bad_schema_batch)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"message   {e}")
    print(f"hint      {e.hint or '(none - the message says it all)'}")

# %% [markdown]
# **拒否2 — 未整列のデータ。** 同じ行を並べ替えただけのものです。取引所別のファイルを
# 連結したあと、最後のソートを忘れるとこうなります。

# %%
import numpy as np

rng = np.random.default_rng(0)
shuffled = trades.take(rng.permutation(len(trades)))
try:
    db.append("trades", shuffled)
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %% [markdown]
# **拒否3 — 時間範囲の重複。** テーブルが既に持っているデータの再送（バッチの最小 `ts`
# が保存済みの最大値より前）も、同じ理由で拒まれます。`append` の意味は*フィードを
# 伸ばす*ことであって、*履歴の間に割り込む*ことではありません。意図した訂正は `write`
# か、プレビューできる `plan_replace_range` の流れを通します（レシピ05と06）。

# %%
try:
    db.append("trades", trades)  # the exact batch already ingested
except h5i_db.InvalidInputError as e:
    print(f"rejected  code={e.code}")
    print(f"hint      {e.hint}")

# %%
# The head never moved: still one data commit per table.
[{k: v[k] for k in ("sequence", "op", "rows") if k in v} for v in db.versions("trades")]

# %% [markdown]
# ## 4. スキーマの進化: 限界を先に知っておく
#
# セグメントが書き換え不能である以上、スキーマは意図的に変えにくく作られています。
# 安く済む進化を前提に設計し、そうでないものは避けてください。
#
# - **安い:** *末尾に NULL 可の列を足す*こと（古いセグメントは NULL として読まれます）と、
#   数値型を*広げる*こと（int32 → int64、float32 → float64）。
# - **高い:** 改名、型を狭める変更、並べ替え、列の削除。どれも新しいテーブルを作って
#   バックフィルする作業になります。
#
# 実務上の帰結は単純です。最初から広めに取っておくこと（`int64`、`float64`、
# `timestamp[us]`）。そして、いつか使う*かもしれない*列（約定の `condition` コードなど）は、
# 初日から NULL 可で入れておいても、ほとんど損をしません。

# %% [markdown]
# ## まとめ
#
# - テーブル ＝ Arrow スキーマ ＋ `time_column`。セグメントは時刻順の Parquet として
#   保存されるので、時刻列を宣言するだけでプルーニング、ストリーミングの `time_bucket`
#   集計、ソート不要の ASOF ジョインがついてきます。
# - クオンツの仕事によく合う既定値は、時刻に NULL 不可の `timestamp[us, tz=UTC]`、
#   リサーチ層の価格に `float64`、数量に `int64`、銘柄に素の `utf8`（どのみち Parquet が
#   辞書エンコードします）。
# - `sort_key=["ts", "symbol"]` の先頭は時刻列でなければなりません。二次キーのおかげで、
#   銘柄別のスキャンが連続した領域だけを読みます。
# - 厳格な append は、スキーマ不一致・未整列のバッチ・時間範囲の重複を、書き込みの*前に*
#   拒みます。エラーには `.code` と `.hint` が付き、失敗してもテーブルの先頭は動きません。
# - スキーマの進化は、末尾への NULL 可列の追加と数値型の拡張で行います。それ以外は作り
#   直しになるので、最初から広く取っておきましょう。

# %%
db.close()
