"""Shared helpers for the h5i-db cookbook.

Synthetic market-data generators plus a cached Yahoo Finance downloader, so
every recipe is reproducible offline after the first run.
"""

from .synthetic import (
    make_trades,
    make_quotes,
    make_trades_and_quotes,
    make_daily_prices,
    make_fx_ticks,
    make_option_chain,
    make_yield_curves,
    make_fundamentals,
)
from .real_data import fetch_daily, fetch_intraday, SP500_EXAMPLES
from .dbs import db_path, fresh_db
from .backtest_data import (
    BOOK_DELTAS_SCHEMA,
    INSTRUMENTS_SCHEMA,
    TRADES_SCHEMA,
    make_backtest_fixture,
)

__all__ = [
    "make_trades",
    "make_quotes",
    "make_trades_and_quotes",
    "make_daily_prices",
    "make_fx_ticks",
    "make_option_chain",
    "make_yield_curves",
    "make_fundamentals",
    "fetch_daily",
    "fetch_intraday",
    "SP500_EXAMPLES",
    "db_path",
    "fresh_db",
    "BOOK_DELTAS_SCHEMA",
    "INSTRUMENTS_SCHEMA",
    "TRADES_SCHEMA",
    "make_backtest_fixture",
]
