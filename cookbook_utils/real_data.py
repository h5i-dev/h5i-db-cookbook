"""Real market data via Yahoo Finance, cached to parquet.

First call hits the network; afterwards recipes are fully reproducible
offline from ``data/cache``.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"

# A liquid, sector-diverse sample used across the research recipes.
SP500_EXAMPLES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "JPM", "V", "XOM",
    "UNH", "JNJ", "PG", "HD", "COST", "MRK", "ABBV", "CVX", "PEP", "KO",
    "WMT", "BAC", "MCD", "CSCO", "DIS", "CAT", "GS", "IBM", "GE", "T",
]


def _cache_path(kind: str, key: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{kind}_{key}.parquet"


def fetch_daily(
    symbols: list[str],
    start: str = "2020-01-01",
    end: str = "2026-07-01",
    refresh: bool = False,
) -> pa.Table:
    """Daily OHLCV for `symbols` as a long-format Arrow table.

    Columns: ts (timestamp[us, UTC], session close), symbol, open, high, low,
    close, adj_close, volume. Sorted by ts.
    """
    key = f"{'-'.join(sorted(symbols))[:80]}_{start}_{end}"
    path = _cache_path("daily", key)
    if path.exists() and not refresh:
        return pq.read_table(path)

    import pandas as pd
    import yfinance as yf

    raw = yf.download(
        symbols, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker"
    )
    parts = []
    for sym in symbols:
        try:
            df = raw[sym].dropna(how="all").copy() if len(symbols) > 1 else raw.copy()
        except KeyError:
            continue
        if df.empty:
            continue
        df = df.rename(
            columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close",
                "Adj Close": "adj_close", "Volume": "volume",
            }
        )
        # Stamp at 20:00 UTC (US session close) so the time column is unambiguous.
        ts = pd.to_datetime(df.index).tz_localize("UTC") + pd.Timedelta(hours=20)
        out = pd.DataFrame(
            {
                "ts": ts,
                "symbol": sym,
                "open": df["open"].astype(float),
                "high": df["high"].astype(float),
                "low": df["low"].astype(float),
                "close": df["close"].astype(float),
                "adj_close": df["adj_close"].astype(float),
                "volume": df["volume"].fillna(0).astype("int64"),
            }
        ).dropna()
        parts.append(out)
    if not parts:
        raise RuntimeError("no data returned from Yahoo Finance")
    table = pa.Table.from_pandas(
        pd.concat(parts).sort_values("ts"), preserve_index=False
    ).cast(
        pa.schema(
            [
                ("ts", pa.timestamp("us", tz="UTC")),
                ("symbol", pa.string()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("adj_close", pa.float64()),
                ("volume", pa.int64()),
            ]
        )
    )
    pq.write_table(table, path)
    return table


def fetch_intraday(
    symbols: list[str],
    period: str = "30d",
    interval: str = "1h",
    refresh: bool = False,
) -> pa.Table:
    """Recent intraday bars (Yahoo limits: 1h → ~2 years, finer → days).

    Columns: ts, symbol, open, high, low, close, volume.
    """
    key = f"{'-'.join(sorted(symbols))[:80]}_{period}_{interval}"
    path = _cache_path("intraday", key)
    if path.exists() and not refresh:
        return pq.read_table(path)

    import pandas as pd
    import yfinance as yf

    parts = []
    for sym in symbols:
        df = yf.download(
            sym, period=period, interval=interval, auto_adjust=False, progress=False
        )
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        idx = pd.to_datetime(df.index)
        ts = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        parts.append(
            pd.DataFrame(
                {
                    "ts": ts,
                    "symbol": sym,
                    "open": df["Open"].astype(float).values,
                    "high": df["High"].astype(float).values,
                    "low": df["Low"].astype(float).values,
                    "close": df["Close"].astype(float).values,
                    "volume": df["Volume"].fillna(0).astype("int64").values,
                }
            ).dropna()
        )
    if not parts:
        raise RuntimeError("no intraday data returned from Yahoo Finance")
    table = pa.Table.from_pandas(
        pd.concat(parts).sort_values("ts"), preserve_index=False
    ).cast(
        pa.schema(
            [
                ("ts", pa.timestamp("us", tz="UTC")),
                ("symbol", pa.string()),
                ("open", pa.float64()),
                ("high", pa.float64()),
                ("low", pa.float64()),
                ("close", pa.float64()),
                ("volume", pa.int64()),
            ]
        )
    )
    pq.write_table(table, path)
    return table
