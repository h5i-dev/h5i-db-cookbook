# %% [markdown]
# # Kaggle Polymarket L2 のための本番データ契約
#
# 公開データセットは、そのままでは調査の入力になりません。記録した側がたまたま出力したとおりの
# タイムスタンプ、単位、重複行を抱えたファイルの山であり、しかもライセンスがやりたいことを許して
# いるとは限りません。
#
# データ契約は、その2つの間に置く層です。テーブルが何を保証するかを述べ、その保証を仮定せずに検査し、
# どのバイトを読んだのかを正確に記録します。これがないと、再現できない結果と間違った結果を区別
# できません。
#
# このレシピでは、
# [Marvingozo の Polymarket データセット](https://www.kaggle.com/datasets/marvingozo/polymarket-tick-level-orderbook-dataset)
# の範囲を限った一部を、正規化されバージョン管理された h5i-db のテーブルに変えます。目的はアルファの
# 主張ではありません。信頼できるバックテストに要る証拠の層をそろえることです。
#
# - 入力ファイルの正確なハッシュとライセンス
# - タイムスタンプの明示的な修復
# - アトミックな全板イベント
# - 正規化された YES 契約1本
# - 戦略を作る前に取る、名前付きの h5i-db スナップショット

# %% [markdown]
# ## ここで使う用語
#
# | 用語 | 意味 |
# | --- | --- |
# | データ契約 | テーブルが何を保証するかの表明。仮定せずに検査する |
# | L2 | 各サイドの複数の価格帯と、そこに置かれている数量 |
# | 板イベント | 板への1回のアトミックな更新。全部適用されるか、まったく適用されないか |
# | タイムスタンプの修復 | 単位・順序・タイムゾーンが誤って届いた時刻を直すこと |
# | ソースハッシュ | 各入力ファイルのチェックサム。取り込みを追跡できるよう記録する |
# | YES 契約 | 出来事が起きれば 1.00、起きなければ 0.00 を払うバイナリ契約 |
# | スナップショット | 戦略が存在する前に取る、名前とチェックサム付きのバージョン固定 |
#
# はじめて見る用語があれば、[GLOSSARY.ja.md](../../GLOSSARY.ja.md) にもう少し詳しい説明があります。ほかのレシピで使う用語もまとめてあります。

# %% [markdown]
# ## レシピが使う分だけダウンロードする
#
# 4つのファイルは圧縮状態で合計およそ 165 MB です。1〜2 GB ある日次ティックファイルは意図して
# ダウンロードしません。キャッシュが空なら、表示されたコマンドをリポジトリのルートで実行して
# ください。Kaggle の認証が必要です。
#
# このデータセットは **CC BY-NC 4.0** です。別途権利を取得しない限り、このレシピとその派生物は
# 非商用・学術用途に限られます。

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import h5i_db
import cookbook_utils as cu

CACHE = Path("data/cache/kaggle-polymarket")
missing = cu.kaggle_missing_files(CACHE)
if missing:
    commands = cu.kaggle_download_commands(CACHE)
    raise FileNotFoundError("Run these bounded downloads:\n" + "\n".join(commands))
print(f"dataset={cu.KAGGLE_POLYMARKET_DATASET}")
print(f"license={cu.KAGGLE_POLYMARKET_LICENSE}")

# %% [markdown]
# 入力をハッシュ化することも再現性の一部です。Kaggle のスラッグとファイル名は不変の識別子では
# ありません。公開者は新しいデータセットバージョンでファイルを差し替えられます。

# %%
manifest = cu.kaggle_source_manifest(CACHE)
manifest.to_pandas()[["remote_path", "bytes", "sha256", "license"]]

# %% [markdown]
# ## 実体化の前に述語を押し下げる
#
# ローダは、1日ぶんのスナップショット、特徴量バー、約定、決着済みマーケットのラベルを遅延評価で
# 突き合わせます。重なっているマーケットのうちプリントが最も多いものを選び、そのマーケットだけを
# 読みます。各サイド10段を保持します。
#
# 現在の特徴量アーティファクトは3月6日から11日までですが、データセットの README には21日ぶんの
# カバレッジと書かれています。そこでローダは、説明文を信じずに Parquet のメタデータと実際の値から
# 重なりを検証します。

# %%
sample = cu.load_kaggle_sample(CACHE, depth_levels=10)
print(sample.question)
print(f"market={sample.market_id}")
print(f"eventual resolution target={sample.target} (audit only, never a feature)")
pd.Series(sample.audit, name="value").to_frame()

# %% [markdown]
# ## 因果と板の契約を検証する
#
# このスナップショットファイルでは `timestamp_created_at` が `timestamp_received` より後になっている
# ので、2つのソース時刻を、そのままイベント時刻と到着時刻として読むわけにはいきません。ローダは
# イベント時刻を保ちつつ、保守的に `ts_init = max(created, received)` と置きます。修正はすべて
# 件数を数えます。
#
# データセットに同梱された `target` 列は、特徴量テーブルから意図して取り除いています。残せば、最終的な
# マーケットの決着が学習用の全行に漏れてしまいます。

# %%
book = sample.book_deltas.to_pandas()
features = sample.features.to_pandas()

assert "target" not in features.columns
assert (book["ts_init"] >= book["ts_event"]).all()
assert sample.audit["crossed_snapshots_dropped"] == 0
assert book.groupby("event_index")["is_last"].sum().eq(1).all()
assert book["price"].between(0.0, 1.0, inclusive="both").all()
assert (book["size"] > 0).all()
print(
    f"{book['event_index'].nunique():,} atomic snapshots, "
    f"{len(book):,} canonical levels, "
    f"{sample.audit['clock_adjustments']:,} clock repairs"
)

# %% [markdown]
# ベスト気配のチャートは、手早く目で見る契約テストです。クロス、上下逆転したサイド、途切れた
# タイムスタンプは、戦略のティアシートよりここでずっと見つけやすくなります。

# %%
bids = (
    book[book["side"] == "buy"]
    .groupby(["event_index", "ts_init"], as_index=False)["price"]
    .max()
    .rename(columns={"price": "bid"})
)
asks = (
    book[book["side"] == "sell"]
    .groupby(["event_index", "ts_init"], as_index=False)["price"]
    .min()
    .rename(columns={"price": "ask"})
)
tops = bids.merge(asks, on=["event_index", "ts_init"], validate="one_to_one")
assert (tops["bid"] < tops["ask"]).all()

fig, ax = plt.subplots(figsize=(10, 4))
ax.step(tops["ts_init"], tops["bid"], where="post", label="best bid")
ax.step(tops["ts_init"], tops["ask"], where="post", label="best ask")
ax.set(title="Polymarket YES top of book — bounded Kaggle sample", ylabel="price")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## 承認した調査用の断面をコミットする
#
# 生の Parquet は使い捨てのキャッシュファイルのままです。正規化されたテーブルと外部バイトの
# マニフェストが、バージョン管理された h5i-db の入力になります。決着ラベルは戦略の特徴量テーブルには
# 取り込みません。

# %%
db = h5i_db.Database(cu.fresh_db("04_kaggle_polymarket_contract"), create=True)
tables = {
    "instruments": (sample.instruments, "ts_init"),
    "book_deltas": (sample.book_deltas, "ts_init"),
    "trades": (sample.trades, "ts_init"),
    "features_1m": (sample.features, "ts_init"),
}
for name, (table, time_column) in tables.items():
    db.create_table(name, table.schema, time_column=time_column)
    db.append(name, table, note="bounded CC BY-NC Kaggle Polymarket sample")

db.snapshot(
    "kaggle-polymarket-2026-03-09",
    tables=list(tables),
    note="Validated one-market research cut; source hashes recorded above",
)
db.tables()

# %% [markdown]
# ## この断面で示せること、示せないこと
#
# これらの全板スナップショットは、保守的なスナップショットリプレイや、スプレッドと板の厚みの調査には
# 向いています。ティック単位の差分ストリームではないので、スナップショットの間で起きたキューの遷移を
# すべて明らかにするには足りません。この標本から得たキューポジションの結果を「正確」と呼ばないで
# ください。
#
# 正確なキューリプレイが要るなら、同じ契約を日次の `orderbook_YYYY-MM-DD.parquet` 1ファイルに、
# `market_id` の遅延フィルタ付きで当てはめてください。日次ファイル全体に `read_parquet` を呼んでは
# いけません。公開者の見積もりで、1日あたり約3億行、RAM 50 GB になります。

# %%
db.close()
