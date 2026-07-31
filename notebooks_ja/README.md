# h5i-db クックブック（日本語版）

[h5i-db](https://github.com/h5i-dev/h5i-db) — マーケットデータ向けの、組み込み型で
バージョン管理付きの時系列データベース — を、クオンツの実務に沿って学ぶレシピ集です。

各レシピは英語版（[`notebooks/`](../notebooks)）と**コードが1文字も違いません**。翻訳して
あるのは解説の文章だけなので、どちらを開いても同じ結果が出ますし、2つを並べて読むこともできます。

クオンツの予備知識は前提にしていません。各レシピは、それが解こうとしている実務上の問題から始まり、
続く「ここで使う用語」の表で、そのレシピに要る用語だけを説明します。すでにご存じならこのセルは
飛ばしてください。[GLOSSARY.ja.md](../GLOSSARY.ja.md) には全用語をまとめ、出発点ごとの読む順番も
書いてあります。

実データを使うレシピは初回実行時に Yahoo Finance からダウンロードして `data/cache/` に
保存するので、2回目以降はオフラインで再現できます。ティックレベルのデータが要るレシピは、
シードを固定した現実的な合成データ生成器（`cookbook_utils/`）を使います。

クエリは SQL 文字列ではなく、遅延 DataFrame ビルダ（`db.table(...)` と動詞）で書いています。
変数に持てるクエリのほうが、読むにも使い回すにも生成するにも楽だからです。ビルダは SQL に
コンパイルされ、`.sql()` で生成物がそのまま見えるので、隠れているものはありません。ビルダ自体は
レシピ09が扱い、レシピ04は SQL だけのツアーです。対応する動詞がないクエリ（`UNION`、`gapfill`、
`tail`、深い多段 CTE）はそのまま SQL で書いています。

## レシピ一覧

### 00 - 基礎

| レシピ | 学べること |
|---|---|
| [01 クイックスタート](00_fundamentals/01_quickstart.ipynb) | DB を作り、ティックを取り込み、最初のクエリとタイムトラベルまで5分 |
| [02 マーケットデータのスキーマ設計](00_fundamentals/02_designing_market_data_schemas.ipynb) | 約定・気配・バーのスキーマ、時刻列、ソートキー、Arrow の型 |
| [03 取り込みのパターン](00_fundamentals/03_ingestion_patterns.ipynb) | Parquet／CSV／pandas／polars から。append と write、バッチ化、衝突処理 |
| [04 クオンツのための SQL ツアー](00_fundamentals/04_sql_tour_for_quants.ipynb) | DataFusion SQL: ウィンドウ、CTE、`time_bucket`、`vwap`、`ewma`、ローリングの糖衣 |
| [05 タイムトラベルとバージョン管理](00_fundamentals/05_time_travel_and_versioning.ipynb) | `h5i('t', v)`、as-of 読み出し、`versions()`、`restore`、スナップショット |
| [06 プレビューできる変更](00_fundamentals/06_previewable_mutations.ipynb) | plan → 検分 → apply／discard。変更ポリシーによるゲート |
| [07 ストリーミング append と tail](00_fundamentals/07_streaming_appends_and_tail.ipynb) | ライブフィードの模擬、追記のみの tail、インクリメンタルな消費者 |
| [08 メンテナンス](00_fundamentals/08_maintenance.ipynb) | snapshot、compact、vacuum、verify でストアを健全に保つ |
| [09 DataFrame ビルダ](00_fundamentals/09_dataframe_builder.ipynb) | `db.table(...)` と動詞。クエリを Python の値として扱い、`.sql()` で SQL へ戻る |

### 01 - マーケットデータエンジニアリング

| レシピ | 学べること |
|---|---|
| [01 OHLCV バー](01_market_data_engineering/01_ohlcv_bars.ipynb) | `time_bucket` によるティック→バー集計、タイムゾーンを考慮したセッション |
| [02 VWAP と TWAP](01_market_data_engineering/02_vwap_twap_execution.ipynb) | 執行ベンチマーク、区間 VWAP、スリッページの計測 |
| [03 ギャップ補完とリサンプル](01_market_data_engineering/03_gapfill_resample.ipynb) | 流動性の低い銘柄に規則的な格子を: locf、interpolate、null |
| [04 ASOF ジョイン: 約定 ↔ 気配](01_market_data_engineering/04_asof_join_trades_quotes.ipynb) | 気配の貼り付け、サイド判定（Lee–Ready）、実効スプレッド |
| [05 コーポレートアクション](01_market_data_engineering/05_corporate_actions.ipynb) | 分割・配当の調整を、監査可能でバージョン付きの言い直しとして |
| [06 ポイントインタイムのファンダメンタルズ](01_market_data_engineering/06_point_in_time_fundamentals.ipynb) | 報告ラグ、as-reported のジョイン、先読みバイアスの回避 |
| [07 ティックデータのクリーニング](01_market_data_engineering/07_tick_data_cleaning.ipynb) | 外れ値検出、プレビューできる削除、監査可能な訂正 |
| [08 NBBO の統合](01_market_data_engineering/08_nbbo_consolidation.ipynb) | 複数取引所のフィードから統合最良気配へ |

### 02 - アルファリサーチ

| レシピ | 学べること |
|---|---|
| [01 クロスセクショナル・モメンタム](02_alpha_research/01_momentum_backtest.ipynb) | 実データでの古典的な12–1モメンタムを、SQL と pandas で端から端まで |
| [02 ペアトレード](02_alpha_research/02_pairs_trading.ipynb) | 共和分の走査、スプレッドの z スコア、バージョン付きの研究反復 |
| [03 EWMA ボラとボラターゲティング](02_alpha_research/03_ewma_vol_targeting.ipynb) | SQL の `ewma`、RiskMetrics のボラ、ボラ調整オーバーレイ |
| [04 実現ボラティリティ](02_alpha_research/04_realized_volatility.ipynb) | 日中足からの RV、サンプリング頻度、シグネチャプロット |
| [05 ファクターの構築](02_alpha_research/05_factor_construction.ipynb) | バリュー／クオリティ／モメンタム、ポイントインタイム、IC 分析 |
| [06 イベントスタディ](02_alpha_research/06_event_study.ipynb) | ASOF ジョインで揃えたイベント前後の異常リターン |
| [07 オーダーフロー・インバランス](02_alpha_research/07_order_flow_imbalance.ipynb) | 符号付き出来高、OFI の特徴量、短期の予測力 |
| [08 日中の季節性](02_alpha_research/08_intraday_seasonality.ipynb) | U字型の出来高・ボラ曲線、時刻効果 |
| [09 リードラグ分析](02_alpha_research/09_lead_lag.ipynb) | ASOF で揃えたクロスアセットのリードラグ |
| [10 ポートフォリオのリバランス](02_alpha_research/10_portfolio_rebalancing.ipynb) | バージョン付きの保有、回転率の管理、リバランスの監査証跡 |
| [11 検索拡張予測](02_alpha_research/11_retrieval_augmented_forecasting.ipynb) | 埋め込みを列として持つ、`array_distance` の上位k件、時系列 RAG が漏れる理由 |
| [12 トレンドフォロー](02_alpha_research/12_trend_following.ipynb) | 時系列モメンタム、オーバーレイによるボラターゲティング、回転率の予算、クライシスアルファの主張を検証 |
| [13 平均回帰](02_alpha_research/13_mean_reversion.ipynb) | そこに無い短期シグナルと、それを示す必要グロスリターン |

### 03 - リスクと本番運用

| レシピ | 学べること |
|---|---|
| [01 VaR と期待ショートフォール](03_risk_and_production/01_var_expected_shortfall.ipynb) | ヒストリカル／パラメトリック VaR、ES、例外のバックテスト |
| [02 再現できるバックテスト](03_risk_and_production/02_reproducible_backtests.ipynb) | 実行ごとのデータバージョン固定、ビット単位で同一の再実行 |
| [03 EOD スナップショットと監査](03_risk_and_production/03_eod_snapshots_audit.ipynb) | 規制対応の監査証跡としての名前付きスナップショット、バージョン差分 |
| [04 データ品質ゲート](03_risk_and_production/04_data_quality_gates.ipynb) | コミット前の検証、ポリシーで強制するレビュー、異常検知 |
| [05 ライブのペーパートレードループ](03_risk_and_production/05_live_paper_trading_loop.ipynb) | tail → シグナル → 注文まで、すべてバージョン付きで |
| [06 複数書き手のパターン](03_risk_and_production/06_multi_writer_conflicts.ipynb) | CAS の衝突、リトライ、チームで安全な並行リサーチ |
| [07 オプションと IV 曲面](03_risk_and_production/07_options_iv_surface.ipynb) | チェーンの保存、曲面のスナップショット、スマイルの推移 |
| [08 FX と暗号資産の24時間データ](03_risk_and_production/08_fx_crypto_24_7.ipynb) | セッションも切れ目もない世界でのバケット分けとローリング統計 |
| [09 債券のカーブ](03_risk_and_production/09_fixed_income_curves.ipynb) | ポイントインタイムのイールドカーブ、キャリーとロールダウン、カーブ履歴 |
| [10 性能チューニング](03_risk_and_production/10_performance_tuning.ipynb) | プルーニング、コンパクション、資源制限、バッチ化、そして速い理由 |
| [11 arrival-delta](03_risk_and_production/11_arrival_delta.ipynb) | 後知恵に値段を付ける: スイープ全体への `arrival_delta` と、2つの誤解の形 |

### 04 - イベント駆動バックテスト

| レシピ | 学べること |
|---|---|
| [01 はじめてのイベント駆動バックテスト](04_event_driven_backtesting/01_first_event_driven_run.ipynb) | 注文意図としてのシグナル、ピン留めしたリプレイ、実行フォーク上の約定・ポジション・エクイティ |
| [02 執行のリアリズム](04_event_driven_backtesting/02_execution_realism.ipynb) | 手数料、スリッページ、レイテンシ、キューポジション、そして実装ショートフォールの感応度 |
| [03 再現できる運用](04_event_driven_backtesting/03_reproducible_backtest_operations.ipynb) | 安定したスナップショット再実行、遅れて届くデータ、カバレッジ・ゲート、監査用マニフェスト |
| [04 Kaggle Polymarket のデータ契約](04_event_driven_backtesting/04_kaggle_polymarket_data_contract.ipynb) | 範囲を限ったダウンロード、ソースハッシュ、タイムスタンプの修復、正規 L2 スナップショット、能力の限界 |
| [05 Kaggle Polymarket のリプレイ](04_event_driven_backtesting/05_kaggle_polymarket_replay.ipynb) | 実際の板の上での因果的な特徴量、イベント駆動の約定、手数料・レイテンシ・スリッページの感応度 |
| [06 注文のライフサイクルとリスク](04_event_driven_backtesting/06_order_lifecycle_and_risk.ipynb) | 型のついたプリフライト、発注／訂正／取消、エンジン内蔵の口座制限、説明、意味的な検証 |
| [07 Python の戦略コールバック](04_event_driven_backtesting/07_python_strategy_callbacks.ipynb) | 状態を持つ戦略、タイマー、約定を受けた行動、安定した戦略の同一性、コールバック再実行の検証 |
| [08 バーからイベント駆動へ](04_event_driven_backtesting/08_equity_bars_to_event_driven.ipynb) | バーから合成した板、`target_positions`、ベクトル化とリプレイを1セント単位で突き合わせる |
| [09 マーケットメイクと在庫](04_event_driven_backtesting/09_market_making_inventory.ipynb) | デルタからの気配提示、在庫スキュー、轢かれること、キャンセルに効くレイテンシ、マークアウト |
| [10 執行アルゴリズム](04_event_driven_backtesting/10_execution_algorithms.ipynb) | 到着価格に対する TWAP・VWAP・POV と、終われなかったことの機会コスト |
| [11 約定からのコスト推定](04_event_driven_backtesting/11_calibrating_costs_from_fills.ipynb) | `quant.costs`: 実効スプレッド、平方根インパクトの推定、それを安いバックテストに課金する |
| [12 戦略空間の探索](04_event_driven_backtesting/12_searching_a_strategy_space.ipynb) | 候補ごとに1つのシグナルテーブル、`RandomSearch` とウォークフォワードと `TopK`、試行台帳、1つのバスケットレポート |
| [13 口座の約定履歴のリプレイ](04_event_driven_backtesting/13_replaying_an_account_ledger.ipynb) | 台帳を注文意図にコンパイルし、板が拒否した場所を市場ごとに突き合わせる |

### 05 - 予測市場

払い戻しに上限があり、手数料が `p*(1-p)` に比例し、標本が数千日ではなく数百マーケットになる、
バイナリ／カテゴリカルなイベント契約に特有のクオンツの進め方です。

| レシピ | 学べること |
|---|---|
| [01 バイナリのパリティと手数料カーブ](05_prediction_markets/01_binary_parity_and_fee_curves.ipynb) | YES+NO=1 のアービトラージ、価格水準ごとの二次的な手数料ハードル、算術と突き合わせた決済 |
| [02 確率のキャリブレーション](05_prediction_markets/02_probability_calibration.ipynb) | 信頼性曲線、Brier スコアの分解、対数損失、ベンチマーク予測、ラベル漏れがないことの point-in-time な証明 |
| [03 フェイバリット・ロングショット・バイアス](05_prediction_markets/03_favorite_longshot_bias.ipynb) | 価格帯ごとの持ち切りリターン、手数料後に両側を取引、しきい値の台地、`sqrt(p(1-p))` のボラ比例 |
| [04 執行の忠実度と板の厚み](05_prediction_markets/04_execution_fidelity_and_depth.ipynb) | スナップショットからのマイクロプライスとインバランス、キューの主張を拒むプリフライト、表示厚みに対する約定率、コスト予算 |
| [05 決済リスクと選択リスク](05_prediction_markets/05_settlement_and_selection_risk.ipynb) | 観測可能性で制御する決済、このパネルで時間分割が失敗する理由、PBO、デフレーテッド・シャープ、最小トラックレコード長 |
| [06 ベンダーデータの取り込み](05_prediction_markets/06_vendor_data_onramp.ipynb) | ベンダー Parquet から正規テーブルへ: マーケット仕様、データとしてのレイアウト、コンテンツアドレスの再取り込み、カバレッジ、CLI |
| [07 ループ全体を一度だけ](05_prediction_markets/07_end_to_end_workflow.ipynb) | 取り込みから意思決定まで: ピン、戦略パック、ショートリストのホールドアウトを使うウォークフォワード、バスケットレポート、Brier アドバンテージ、デフレーテッド・シャープ、検証 |
| [08 実データでのループ全体](05_prediction_markets/08_real_polymarket_end_to_end.ipynb) | ティックレベルの実 Polymarket データで一気通貫: 11の規則を試してすべて損。コスト予算がその理由を説明する |

### 06 - パフォーマンス分析

`h5i_db.quant` 層です。`alphalens` と `pyfolio` に対する、保守された答えであり、どちらにも
用意できなかった部分を持っています。どの統計量も、計算元のデータバージョンを持ち歩くエンジンへの
クエリで、ピン留めされていない結果は検証を拒まれます。

| レシピ | 学べること |
|---|---|
| [01 ティアシートとパフォーマンス統計](06_performance_analytics/01_tearsheets_and_performance_stats.ipynb) | empyrical と一致する `ReturnSeries` の統計、ドローダウンの出来事、ローリングベータ、実行からの `from_levels`、`verify` が拒む理由 |
| [02 ファクターパネルとレポート](06_performance_analytics/02_factor_panels_and_reports.ipynb) | `build_panel`: 欠損の会計、IC の減衰、分位リターン、回転率、セクターニュートラル、1つの HTML レポート |
| [03 リークのない交差検証](06_performance_analytics/03_cross_validation_without_leakage.ipynb) | パージド k 分割、エンバーゴ、CPCV の分布、ウォークフォワード、そして勝者への PBO とデフレーテッド・シャープ |
| [04 スイープ・検証・修正](06_performance_analytics/04_sweeps_verification_restatements.ipynb) | 試行ごとに1フォークの `quant.sweep`、来歴のダイジェスト、ベンダー改訂が答えに与えた変化 |

## 作り直し方

`.py`（jupytext の percent 形式）が原本で、`.ipynb` はそこから実行して生成します。

```bash
python scripts/build_notebooks.py --dir notebooks_ja            # 全レシピ
python scripts/build_notebooks.py --dir notebooks_ja 00_fundamentals/01_quickstart
```

英語版と日本語版はデータベースのディレクトリ（`data/dbs/`）を共有するので、ビルドは片方ずつ
実行してください。

環境構築の手順は英語版の [README](../README.md) を参照してください。
