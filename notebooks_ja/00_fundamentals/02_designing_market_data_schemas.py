# %% [markdown]
# # マーケットデータのスキーマ設計
#
# h5i-db のテーブルは Arrow スキーマと時刻列の組で、実体は書き換え不能・時刻順の Parquet
# セグメントとバージョン付きマニフェストです。セグメントが書き換え不能だからこそ、スキーマ
# 設計は後から安く見直せない唯一の決定になります。
#
# 型の選び方は、ストレージのサイズと、どの SQL 演算子がきれいに効くかを決めます。
# `time_column` と `sort_key` の選び方は、スキャンのプルーニングの効き具合と、ASOF ジョインや
# バー集計の速さを決めます。
#
# このレシピで進めるのは次の5つです。
#
# 1. 株式デスクが最初に持つ3つのフィードを眺める
# 2. 型の選択を列ごとに根拠づける
# 3. `time_column` と `sort_key` を宣言し、それで何が買えるのかを確かめる
# 4. 厳格な append の契約、つまり h5i-db が入口で何を弾くのかを突く
# 5. 後から安いスキーマ変更と、作り直しになる変更を仕分ける

# %% [markdown]
# ## ここで使う用語
#
# | 用語        | 意味 |
# | --------- | --- |
# | 約定（trade） | 成立した取引。価格・数量・時刻を持つ |
# | 気配（quote） | 取引所が今出している売買の意思表示。ビッドとアスクで表す |
# | ビッド／アスク   | 買い手が払う最高値と、売り手が受け入れる最安値 |
# | OHLCV バー  | ティックを一定区間に集計したもの。始値・高値・安値・終値・出来高 |
# | Arrow     | h5i-db が話すインメモリ列指向フォーマット。pandas や polars から安く変換できる |
# | スキーマ      | テーブルの列名と型。作成時に固定される |
# | 時刻列       | h5i-db がソートと枝刈りに使う列。すべてのテーブルが1つ宣言する |
# | ソートキー     | セグメント内の物理的な行順。時刻列で始める必要がある |
# | セグメント     | ある時間範囲の行を持つ、イミュータブルな Parquet ファイル1つ |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %%
import pyarrow as pa

import h5i_db
from h5i_db import col, count_star
import cookbook_utils as cu

db = h5i_db.Database(cu.fresh_db("00_schemas"), create=True)

# %% [markdown]
# ## 1. データ
#
# フィードが3つ、テーブルも3つです。約定は `cu.make_trades` のティックデータで、1行が
# 1約定です。
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
trades = cu.make_trades(days=2, trades_per_day=5_000)
print(f"trades: {trades.num_rows:,} rows x {trades.num_columns} columns")
trades.to_pandas().head()

# %% [markdown]
# 気配はベストビッド・ベストオファーのスナップショットです。最良気配が動くたびに1行増え、
# `bid`／`ask` の価格と `bid_size`／`ask_size` の数量を持ちます。

# %%
quotes = cu.make_quotes(days=2, quotes_per_day=8_000)
print(f"quotes: {quotes.num_rows:,} rows x {quotes.num_columns} columns")
quotes.to_pandas().head()

# %% [markdown]
# 日足は同じ3銘柄を1セッション1行にまとめたもので、`ts` は 20:00 UTC の引けを指します。

# %%
daily = cu.make_daily_prices(symbols=["AAPL", "MSFT", "NVDA"], days=250)
print(f"daily:  {daily.num_rows:,} rows x {daily.num_columns} columns")
daily.to_pandas().head()

# %% [markdown]
# ## 2. ティック約定の型を選ぶ
#
# 効いてくる決定を列ごとに挙げます。
#
# - **`ts`: `timestamp[us, tz=UTC]`、NOT NULL。** マイクロ秒あれば統合フィードの粒度は足り
#   ます。タイムゾーン付き UTC にすると夏時間まわりのバグがまとめて消えます。取引所時刻へ
#   の変換は保存時ではなく、クエリ時に `time_bucket` のタイムゾーン引数でやります。NULL は
#   禁じてください。時刻のない約定はデータではありません。こうしておくと API の他の生値引数
#   （プランの範囲、ギャップ補完の刻み、ASOF の許容差）もすべてマイクロ秒で揃います。
# - **`price`: `float64`。** `decimal128` は厳密で、決済台帳ならそちらが正解です。リサーチ層
#   では float64 が勝ちます。セント刻みの株価なら $10^{13}$ あたりまで誤差なく往復しますし、
#   分析関数（`vwap`、`ewma`、`stddev`、`corr`）が期待する型でもあり、幅は半分です。10進数は
#   帳簿系のシステムに取っておきましょう。
# - **`size`: `int64`。** float は使いません。株数は整数です。int64 の余裕があれば、端株から
#   インデックスのリバランス・クロスまで1つの型で足ります。
# - **`symbol`、`exchange`、`side`: `utf8`。** カーディナリティの低い文字列は、Parquet が
#   セグメントの中で勝手に辞書エンコードします。スキーマ層で素の `utf8` にしておけばディスク
#   上のコストはほとんど変わらず、スキーマは単純なままです。

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
# ## 3. `time_column` と `sort_key`
#
# `time_column="ts"` の宣言はメタデータの飾りではありません。h5i-db が他のすべてを載せる
# 契約です。
#
# - セグメントは **`ts` 順で保存**され、各セグメントの時刻範囲はマニフェストに入ります。
#   だから `WHERE ts BETWEEN ...` はセグメントを開かずに丸ごと飛ばせます
# - **ASOF ジョイン**と **`time_bucket` の集計**は、その順序の上を流れます。クエリのたびに
#   ティックを並べ直したりしません
# - `append` がフィードの意味論を強制できるようになります。「保存済みのどれよりも後」が
#   well-defined になるからです
#
# `sort_key=["ts", "symbol"]` は、同じ時刻の中での二次的な順序を足します。それ以上に効くのが、
# 銘柄ごとのスキャンが連続した領域だけを触るように行がまとまることです。規則はこうです。
# ソートキーの先頭は時刻列、次はいちばんよく絞り込む列（ほぼ必ず `symbol`）。

# %%
db.create_table("trades", trades_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("quotes", quotes_schema, time_column="ts", sort_key=["ts", "symbol"])
db.create_table("bars_1d", bars_1d_schema, time_column="ts", sort_key=["ts", "symbol"])
db.tables()

# %%
for name, data in [("trades", trades), ("quotes", quotes), ("bars_1d", daily)]:
    commit = db.append(name, data, note="initial load")
    print(f"{name:8s} v{commit['sequence']}: {commit['rows_total']:>7,} rows, "
          f"{commit['segments_total']} segment(s)")

db.schema("trades")

# %% [markdown]
# スキーマが働いていることを確かめる軽いクエリを1つ。気配テーブルからそのまま出した、銘柄
# ごとのクオートスプレッド（ベーシスポイント）です。素の `utf8` の銘柄列でも、グループ化も
# 絞り込みも期待どおりに動きます。この層で「カテゴリ型」のようなものは要りません。

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
# ## 4. 厳格な append が弾くもの
#
# `append` はフィードの意味論を持ちます。バッチは宣言したスキーマと一致し、時刻列で整列し、
# 先頭がテーブルの現在の最大タイムスタンプ以降でなければなりません。それ以外は書き込みの
# *前に*弾かれるので、append が失敗してもテーブルの先頭は動きません。
#
# チームで共有するデータベース、あるいはエージェントが書き込むデータベースでは、この厳格さ
# こそが狙いです。壊れたベンダーファイルや順序の狂ったバックフィルが、下流のリサーチを
# 静かに汚染するかわりに、取り込みの時点で大きな音を立てて落ちます。
#
# h5i-db の例外はどれも、機械可読な `.code` と `.retryable` フラグ、そしてたいてい推奨対処を
# 書いた `.hint` を持ちます。
#
# **弾かれ方その1: スキーマ不一致。** ベンダーが「気を利かせて」価格を float32 で送ってきて、
# `side` 列を忘れた場合です。

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
# **弾かれ方その2: 未整列。** 同じ行をシャッフルしたものです。取引所ごとのファイルを結合した
# まま最後のソートを忘れる、という定番の結末です。

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
# **弾かれ方その3: 時間範囲の重複。** テーブルがすでに持っているデータを再送すると、つまり
# バッチの最小 `ts` が保存済みの最大値より前に来ると、同じ理由で拒否されます。`append` の
# 意味は*フィードを伸ばす*ことであって、*履歴に割り込む*ことではありません。意図的な修正は
# `write` か、プレビューできる `plan_replace_range` の流れを通します。レシピ05と06で扱います。

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
# ## 5. スキーマ進化: 限界を先に知っておく
#
# セグメントは書き換え不能なので、スキーマは意図的に変えにくくしてあります。安い進化を前提に
# 設計し、高い進化は避けてください。
#
# - **安い:** *末尾に NULL 許容列を足す*（古いセグメントは NULL として読まれます）、数値型を
#   *広げる*（int32 → int64、float32 → float64）。
# - **高い:** 改名、型を狭める、並べ替え、列の削除。どれも新しいテーブルを作ってバックフィル
#   する話になります。
#
# 実務上の帰結は、最初から広く取っておくことです。`int64`、`float64`、`timestamp[us]`。
# そして「いつか要るかもしれない」列、たとえば約定の `condition` コードのようなものは、初日
# から NULL 許容で入れておいてもコストはほとんどありません。

# %% [markdown]
# ## まとめ
#
# - テーブルは Arrow スキーマと `time_column` です。セグメントは時刻順の Parquet として保存
#   されるので、時刻列を宣言するだけでプルーニング、ストリーミングの `time_bucket` 集計、
#   ソート不要の ASOF ジョインが手に入ります。
# - クオンツ業務で無難な既定値は、時刻が NOT NULL の `timestamp[us, tz=UTC]`、リサーチ層の
#   価格が `float64`、数量が `int64`、銘柄が素の `utf8` です。
# - `sort_key=["ts", "symbol"]` の先頭は必ず時刻列にします。銘柄ごとのスキャンが連続した
#   データだけを触るようにするのは、二次キーの仕事です。
# - 厳格な append は、スキーマ不一致・未整列バッチ・時間範囲の重複を、何も書かないうちに
#   弾きます。エラーは `.code` と `.hint` を持ち、失敗してもテーブルの先頭は動きません。
# - スキーマの進化は末尾への NULL 許容列の追加と数値型の拡張で行います。それ以外は作り直し
#   なので、最初から広く取っておきましょう。

# %%
db.close()
