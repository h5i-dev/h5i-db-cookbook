# h5i-db-cookbook

Practical, quant-focused recipes for [h5i-db](https://github.com/h5i-dev/h5i-db),
the embedded, versioned time-series database for market data workloads.

日本語版は [`notebooks_ja/`](notebooks_ja/README.md) にあります（コードは同一、解説のみ翻訳）.

## Setup

```bash
# 1. Build & install h5i-db (PyPI release pending)
git clone https://github.com/h5i-dev/h5i-db ../h5i-db
pip install maturin
maturin build -m ../h5i-db/crates/h5i-db-python/Cargo.toml   # do NOT pass --release
pip install ../h5i-db/target/wheels/h5i_db-*.whl

# 2. Cookbook dependencies
pip install -r requirements.txt

# 3. Run any recipe
python notebooks/00_fundamentals/01_quickstart.py
# ... or open the .ipynb next to it
```

Real-data recipes download from Yahoo Finance on first run and cache to
`data/cache/`, so everything is reproducible offline afterwards. Recipes that
need tick-level data use realistic synthetic generators (`cookbook_utils/`),
deterministic given a seed.

## Recipes

### 00 - Fundamentals

| Recipe | What you learn |
|---|---|
| [01 Quickstart](notebooks/00_fundamentals/01_quickstart.ipynb) | Create a DB, ingest ticks, first SQL, first time travel, in 5 minutes |
| [02 Designing market-data schemas](notebooks/00_fundamentals/02_designing_market_data_schemas.ipynb) | Trades/quotes/bars schemas, time columns, sort keys, Arrow types |
| [03 Ingestion patterns](notebooks/00_fundamentals/03_ingestion_patterns.ipynb) | Parquet/CSV/pandas/polars in; append vs write; batching; conflict handling |
| [04 SQL tour for quants](notebooks/00_fundamentals/04_sql_tour_for_quants.ipynb) | DataFusion SQL: windows, CTEs, `time_bucket`, `vwap`, `ewma`, rolling sugar |
| [05 Time travel & versioning](notebooks/00_fundamentals/05_time_travel_and_versioning.ipynb) | `h5i('t', v)`, as-of reads, `versions()`, `restore`, snapshots |
| [06 Previewable mutations](notebooks/00_fundamentals/06_previewable_mutations.ipynb) | plan → inspect → apply/discard; mutation policy gates |
| [07 Streaming appends & tail](notebooks/00_fundamentals/07_streaming_appends_and_tail.ipynb) | Live feed simulation; append-only tails; incremental consumers |
| [08 Maintenance](notebooks/00_fundamentals/08_maintenance.ipynb) | snapshot, compact, vacuum, verify: keeping a store healthy |

### 01 - Market data engineering

| Recipe | What you learn |
|---|---|
| [01 OHLCV bars](notebooks/01_market_data_engineering/01_ohlcv_bars.ipynb) | Tick→bar rollups with `time_bucket`, timezone-aware sessions |
| [02 VWAP & TWAP](notebooks/01_market_data_engineering/02_vwap_twap_execution.ipynb) | Execution benchmarks; interval VWAP; slippage measurement |
| [03 Gapfill & resample](notebooks/01_market_data_engineering/03_gapfill_resample.ipynb) | Regular grids for illiquid names: locf, interpolate, null |
| [04 ASOF joins: trades ↔ quotes](notebooks/01_market_data_engineering/04_asof_join_trades_quotes.ipynb) | Quote attachment, trade signing (Lee–Ready), effective spread |
| [05 Corporate actions](notebooks/01_market_data_engineering/05_corporate_actions.ipynb) | Split/dividend adjustments as audited, versioned restatements |
| [06 Point-in-time fundamentals](notebooks/01_market_data_engineering/06_point_in_time_fundamentals.ipynb) | Reporting lags, as-reported joins, avoiding lookahead bias |
| [07 Tick data cleaning](notebooks/01_market_data_engineering/07_tick_data_cleaning.ipynb) | Outlier detection; previewable deletes; auditable corrections |
| [08 NBBO consolidation](notebooks/01_market_data_engineering/08_nbbo_consolidation.ipynb) | Multi-venue feeds into a consolidated best bid/offer |

### 02 - Alpha research

| Recipe | What you learn |
|---|---|
| [01 Cross-sectional momentum](notebooks/02_alpha_research/01_momentum_backtest.ipynb) | Classic 12–1 momentum on real data, end to end in SQL + pandas |
| [02 Pairs trading](notebooks/02_alpha_research/02_pairs_trading.ipynb) | Cointegration scan, spread z-scores, versioned research iterations |
| [03 EWMA vol & vol targeting](notebooks/02_alpha_research/03_ewma_vol_targeting.ipynb) | `ewma` in SQL; RiskMetrics vol; a vol-targeted overlay |
| [04 Realized volatility](notebooks/02_alpha_research/04_realized_volatility.ipynb) | RV from intraday bars, sampling frequency, signature plots |
| [05 Factor construction](notebooks/02_alpha_research/05_factor_construction.ipynb) | Value/quality/momentum factors, point-in-time, IC analysis |
| [06 Event studies](notebooks/02_alpha_research/06_event_study.ipynb) | Abnormal returns around events with ASOF joins |
| [07 Order-flow imbalance](notebooks/02_alpha_research/07_order_flow_imbalance.ipynb) | Signed volume, OFI features, short-horizon predictability |
| [08 Intraday seasonality](notebooks/02_alpha_research/08_intraday_seasonality.ipynb) | U-shaped volume/vol curves; time-of-day effects |
| [09 Lead–lag analysis](notebooks/02_alpha_research/09_lead_lag.ipynb) | Cross-asset lead–lag with ASOF alignment |
| [10 Portfolio rebalancing](notebooks/02_alpha_research/10_portfolio_rebalancing.ipynb) | Versioned holdings, turnover control, rebalance audit trail |

### 03 - Risk & production

| Recipe | What you learn |
|---|---|
| [01 VaR & Expected Shortfall](notebooks/03_risk_and_production/01_var_expected_shortfall.ipynb) | Historical/parametric VaR, ES, backtesting exceptions |
| [02 Reproducible backtests](notebooks/03_risk_and_production/02_reproducible_backtests.ipynb) | Pinning data versions per run; re-running byte-identical research |
| [03 EOD snapshots & audit](notebooks/03_risk_and_production/03_eod_snapshots_audit.ipynb) | Named snapshots as regulatory audit trail; diffing versions |
| [04 Data-quality gates](notebooks/03_risk_and_production/04_data_quality_gates.ipynb) | Validation before commit; policy-enforced review; anomaly checks |
| [05 Live paper-trading loop](notebooks/03_risk_and_production/05_live_paper_trading_loop.ipynb) | Streaming tail → signal → orders, all versioned |
| [06 Multi-writer patterns](notebooks/03_risk_and_production/06_multi_writer_conflicts.ipynb) | CAS conflicts, retries, team-safe concurrent research |
| [07 Options & IV surfaces](notebooks/03_risk_and_production/07_options_iv_surface.ipynb) | Storing chains, surface snapshots, smile evolution |
| [08 FX & crypto 24/7 data](notebooks/03_risk_and_production/08_fx_crypto_24_7.ipynb) | No sessions, no gaps: bucketing and rolling stats around the clock |
| [09 Fixed-income curves](notebooks/03_risk_and_production/09_fixed_income_curves.ipynb) | Point-in-time yield curves, carry/rolldown, curve history |
| [10 Performance tuning](notebooks/03_risk_and_production/10_performance_tuning.ipynb) | Pruning, compaction, resource limits, batching, and why it's fast |
| [11 Leakage check](notebooks/03_risk_and_production/11_leakage_check.ipynb) | Pricing hindsight: `arrival_delta` across a sweep, and the two ways it misleads |

## Layout

```
cookbook_utils/     shared synthetic-data generators + cached Yahoo downloader
notebooks/          37 recipes × (.py source ⇄ executed .ipynb)
notebooks_ja/       the same 37 recipes, prose translated to Japanese
scripts/            build tooling (py → executed ipynb)
data/cache/         cached real market data (parquet)
data/dbs/           databases created by recipes (disposable)
```

## Rebuilding the notebooks

```bash
python scripts/build_notebooks.py                         # all recipes
python scripts/build_notebooks.py 00_fundamentals/01_quickstart
python scripts/build_notebooks.py --dir notebooks_ja      # Japanese recipes
```

Both trees run the same code against the same `data/dbs/` directories, so build
one at a time.

## License

Apache-2.0
