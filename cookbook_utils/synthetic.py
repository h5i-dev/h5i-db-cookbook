"""Synthetic market-data generators.

All generators are deterministic given a seed and return pyarrow Tables with
timezone-aware ``timestamp[us, tz=UTC]`` time columns (h5i-db's preferred time
type), sorted by time. Prices follow simple but realistic dynamics: geometric
Brownian motion with intraday U-shaped activity, bid-ask bounce for trades and
inventory-driven quote updates.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa

US = 1_000_000  # microseconds per second

# NYSE regular session, expressed in UTC (summer: 13:30-20:00 UTC).
SESSION_OPEN_S = 13 * 3600 + 30 * 60
SESSION_CLOSE_S = 20 * 3600


def _ts_array(epoch_us: np.ndarray) -> pa.Array:
    return pa.array(epoch_us.astype("int64"), type=pa.timestamp("us", tz="UTC"))


def _u_shaped_arrivals(n: int, day_start_us: int, rng: np.random.Generator) -> np.ndarray:
    """n event times (epoch us) inside one session with U-shaped intensity."""
    # Mixture: 40% near open, 20% near close, 40% uniform.
    frac = rng.random(n)
    u = rng.random(n)
    open_len = SESSION_CLOSE_S - SESSION_OPEN_S
    sec = np.where(
        frac < 0.4,
        SESSION_OPEN_S + np.abs(rng.normal(0, open_len * 0.12, n)),
        np.where(
            frac < 0.6,
            SESSION_CLOSE_S - np.abs(rng.normal(0, open_len * 0.10, n)),
            SESSION_OPEN_S + u * open_len,
        ),
    )
    sec = np.clip(sec, SESSION_OPEN_S, SESSION_CLOSE_S - 1e-3)
    return np.sort(day_start_us + (sec * US).astype("int64"))


def _sessions(start: str, days: int) -> list[int]:
    """Epoch-us midnight for each of `days` consecutive weekdays from `start`."""
    out, d = [], np.datetime64(start, "D")
    while len(out) < days:
        if np.is_busday(d):
            out.append(int(d.astype("datetime64[us]").astype("int64")))
        d += 1
    return out


def make_trades(
    symbols: list[str] | None = None,
    days: int = 5,
    trades_per_day: int = 20_000,
    start: str = "2026-06-01",
    seed: int = 7,
    base_prices: dict[str, float] | None = None,
) -> pa.Table:
    """Tick-level trades: ts, symbol, price, size, exchange, side."""
    symbols = symbols or ["AAPL", "MSFT", "NVDA"]
    rng = np.random.default_rng(seed)
    base = base_prices or {s: float(rng.uniform(40, 400)) for s in symbols}
    exchanges = np.array(["NYSE", "NASDAQ", "ARCA", "IEX", "BATS"])

    parts = []
    for sym in symbols:
        px = base[sym]
        vol = rng.uniform(0.10, 0.35)  # annualized
        for day_us in _sessions(start, days):
            n = int(trades_per_day * rng.uniform(0.7, 1.3))
            ts = _u_shaped_arrivals(n, day_us, rng)
            dt = np.diff(ts, prepend=ts[0]) / (US * 6.5 * 3600 * 252)
            ret = rng.normal(0, vol * np.sqrt(np.maximum(dt, 1e-12)))
            mid = px * np.exp(np.cumsum(ret))
            spread = px * rng.uniform(1e-4, 4e-4)
            side = rng.choice([-1, 1], n)  # bid-ask bounce
            price = np.round((mid + side * spread / 2) * 100) / 100
            size = np.maximum(1, (rng.lognormal(4.0, 1.2, n) // 100 * 100)).astype("int64")
            parts.append(
                pa.table(
                    {
                        "ts": _ts_array(ts),
                        "symbol": pa.array([sym] * n),
                        "price": pa.array(price, pa.float64()),
                        "size": pa.array(size, pa.int64()),
                        "exchange": pa.array(rng.choice(exchanges, n)),
                        "side": pa.array(np.where(side > 0, "B", "S")),
                    }
                )
            )
            px = float(mid[-1])
    out = pa.concat_tables(parts)
    return out.sort_by("ts")


def make_quotes(
    symbols: list[str] | None = None,
    days: int = 5,
    quotes_per_day: int = 60_000,
    start: str = "2026-06-01",
    seed: int = 11,
    base_prices: dict[str, float] | None = None,
) -> pa.Table:
    """NBBO-style quotes: ts, symbol, bid, ask, bid_size, ask_size."""
    symbols = symbols or ["AAPL", "MSFT", "NVDA"]
    rng = np.random.default_rng(seed)
    base = base_prices or {s: float(rng.uniform(40, 400)) for s in symbols}

    parts = []
    for sym in symbols:
        px = base[sym]
        vol = rng.uniform(0.10, 0.35)
        for day_us in _sessions(start, days):
            n = int(quotes_per_day * rng.uniform(0.7, 1.3))
            ts = _u_shaped_arrivals(n, day_us, rng)
            dt = np.diff(ts, prepend=ts[0]) / (US * 6.5 * 3600 * 252)
            mid = px * np.exp(np.cumsum(rng.normal(0, vol * np.sqrt(np.maximum(dt, 1e-12)))))
            half = px * rng.uniform(0.5e-4, 3e-4, n) / 2
            parts.append(
                pa.table(
                    {
                        "ts": _ts_array(ts),
                        "symbol": pa.array([sym] * n),
                        "bid": pa.array(np.round((mid - half) * 100) / 100, pa.float64()),
                        "ask": pa.array(np.round((mid + half) * 100) / 100, pa.float64()),
                        "bid_size": pa.array(
                            (rng.lognormal(5.5, 1.0, n) // 100 * 100 + 100).astype("int64")
                        ),
                        "ask_size": pa.array(
                            (rng.lognormal(5.5, 1.0, n) // 100 * 100 + 100).astype("int64")
                        ),
                    }
                )
            )
            px = float(mid[-1])
    return pa.concat_tables(parts).sort_by("ts")


def make_trades_and_quotes(**kw) -> tuple[pa.Table, pa.Table]:
    """Trades and quotes over the same sessions/symbols (shared base prices).

    Note: the two streams share base prices but follow *independent* random
    walks, so trade prices are not consistent with the prevailing quote mid.
    Recipes that need a microstructure-consistent tape (e.g. Lee-Ready
    signing) should derive trades from the quote stream instead.
    """
    symbols = kw.pop("symbols", None) or ["AAPL", "MSFT", "NVDA"]
    seed = kw.pop("seed", 7)
    rng = np.random.default_rng(seed)
    base = {s: float(rng.uniform(40, 400)) for s in symbols}
    trades = make_trades(symbols=symbols, seed=seed, base_prices=base, **kw)
    qkw = {k: v for k, v in kw.items() if k != "trades_per_day"}
    quotes = make_quotes(symbols=symbols, seed=seed + 1, base_prices=base, **qkw)
    return trades, quotes


def make_daily_prices(
    symbols: list[str] | None = None,
    days: int = 750,
    start: str = "2023-01-02",
    seed: int = 21,
) -> pa.Table:
    """Daily OHLCV panel: ts, symbol, open, high, low, close, volume."""
    symbols = symbols or [f"STK{i:03d}" for i in range(50)]
    rng = np.random.default_rng(seed)
    sess = _sessions(start, days)
    market = rng.normal(0.0003, 0.009, days)  # common factor

    parts = []
    for sym in symbols:
        beta = rng.uniform(0.5, 1.6)
        idio = rng.normal(0, rng.uniform(0.008, 0.02), days)
        drift = rng.normal(0.0002, 0.0004)
        ret = drift + beta * market + idio
        close = float(rng.uniform(10, 300)) * np.exp(np.cumsum(ret))
        o = close * (1 + rng.normal(0, 0.003, days))
        hi = np.maximum(o, close) * (1 + np.abs(rng.normal(0, 0.004, days)))
        lo = np.minimum(o, close) * (1 - np.abs(rng.normal(0, 0.004, days)))
        volume = (rng.lognormal(13, 0.5, days)).astype("int64")
        ts = np.array(sess, dtype="int64") + SESSION_CLOSE_S * US
        parts.append(
            pa.table(
                {
                    "ts": _ts_array(ts),
                    "symbol": pa.array([sym] * days),
                    "open": pa.array(np.round(o, 2)),
                    "high": pa.array(np.round(hi, 2)),
                    "low": pa.array(np.round(lo, 2)),
                    "close": pa.array(np.round(close, 2)),
                    "volume": pa.array(volume),
                }
            )
        )
    return pa.concat_tables(parts).sort_by("ts")


def make_fx_ticks(
    pairs: list[str] | None = None,
    hours: int = 72,
    ticks_per_hour: int = 3_000,
    start: str = "2026-06-01",
    seed: int = 33,
) -> pa.Table:
    """24/7 FX/crypto style ticks: ts, pair, bid, ask."""
    pairs = pairs or ["EURUSD", "USDJPY", "GBPUSD"]
    rng = np.random.default_rng(seed)
    base = {"EURUSD": 1.08, "USDJPY": 157.0, "GBPUSD": 1.27, "BTCUSD": 68_000.0}
    t0 = int(np.datetime64(start, "us").astype("int64"))

    parts = []
    for pair in pairs:
        px = base.get(pair, float(rng.uniform(0.5, 2.0)))
        n = hours * ticks_per_hour
        ts = np.sort(t0 + (rng.random(n) * hours * 3600 * US).astype("int64"))
        mid = px * np.exp(np.cumsum(rng.normal(0, 5e-5, n)))
        half = px * rng.uniform(0.2e-4, 1e-4, n) / 2
        digits = 3 if "JPY" in pair else 5
        parts.append(
            pa.table(
                {
                    "ts": _ts_array(ts),
                    "pair": pa.array([pair] * n),
                    "bid": pa.array(np.round(mid - half, digits)),
                    "ask": pa.array(np.round(mid + half, digits)),
                }
            )
        )
    return pa.concat_tables(parts).sort_by("ts")


def make_option_chain(
    underlier: str = "SPX",
    spot: float = 5_500.0,
    snapshots: int = 5,
    start: str = "2026-06-01",
    seed: int = 55,
) -> pa.Table:
    """Daily option-chain snapshots: ts, expiry, strike, cp, iv, mid, delta."""
    rng = np.random.default_rng(seed)
    sess = _sessions(start, snapshots)
    expiries_d = np.array([7, 14, 30, 60, 91, 182, 365])
    strikes_rel = np.arange(0.80, 1.21, 0.025)

    rows = {k: [] for k in ("ts", "underlier", "expiry", "strike", "cp", "iv", "mid", "delta")}
    px = spot
    for day_us in sess:
        px *= float(np.exp(rng.normal(0, 0.01)))
        base_vol = 0.15 + 0.05 * rng.random()
        for ed in expiries_d:
            t = ed / 365.0
            expiry_us = day_us + int(ed) * 86_400 * US
            for kr in strikes_rel:
                k = round(px * kr / 25) * 25
                m = np.log(k / px) / (base_vol * np.sqrt(t))
                iv = base_vol * (1 + 0.15 * m * m - 0.10 * m) + rng.normal(0, 0.002)
                iv = float(np.clip(iv, 0.05, 1.5))
                for cp in ("C", "P"):
                    from math import erf, exp, log, sqrt

                    d1 = (log(px / k) + 0.5 * iv * iv * t) / (iv * sqrt(t))
                    nd1 = 0.5 * (1 + erf(d1 / sqrt(2)))
                    delta = nd1 if cp == "C" else nd1 - 1
                    d2 = d1 - iv * sqrt(t)
                    nd2 = 0.5 * (1 + erf(d2 / sqrt(2)))
                    if cp == "C":
                        mid = px * nd1 - k * nd2
                    else:
                        mid = k * (1 - nd2) - px * (1 - nd1)
                    rows["ts"].append(day_us + SESSION_CLOSE_S * US)
                    rows["underlier"].append(underlier)
                    rows["expiry"].append(expiry_us)
                    rows["strike"].append(float(k))
                    rows["cp"].append(cp)
                    rows["iv"].append(round(iv, 4))
                    rows["mid"].append(round(max(mid, 0.05), 2))
                    rows["delta"].append(round(delta, 4))
    return pa.table(
        {
            "ts": _ts_array(np.array(rows["ts"], dtype="int64")),
            "underlier": pa.array(rows["underlier"]),
            "expiry": _ts_array(np.array(rows["expiry"], dtype="int64")),
            "strike": pa.array(rows["strike"], pa.float64()),
            "cp": pa.array(rows["cp"]),
            "iv": pa.array(rows["iv"], pa.float64()),
            "mid": pa.array(rows["mid"], pa.float64()),
            "delta": pa.array(rows["delta"], pa.float64()),
        }
    ).sort_by("ts")


def make_yield_curves(
    days: int = 250,
    start: str = "2025-07-01",
    seed: int = 77,
) -> pa.Table:
    """Daily par yield curves: ts, tenor_years, yield_pct (Nelson-Siegel)."""
    rng = np.random.default_rng(seed)
    tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    sess = _sessions(start, days)
    beta0, beta1, beta2, lam = 4.3, -0.8, -1.2, 1.8

    rows_ts, rows_tenor, rows_y = [], [], []
    for day_us in sess:
        beta0 += rng.normal(0, 0.03)
        beta1 += rng.normal(0, 0.04)
        beta2 += rng.normal(0, 0.05)
        x = tenors / lam
        f = (1 - np.exp(-x)) / x
        y = beta0 + beta1 * f + beta2 * (f - np.exp(-x))
        ts = day_us + 21 * 3600 * US
        rows_ts.extend([ts] * len(tenors))
        rows_tenor.extend(tenors.tolist())
        rows_y.extend(np.round(y, 4).tolist())
    return pa.table(
        {
            "ts": _ts_array(np.array(rows_ts, dtype="int64")),
            "tenor_years": pa.array(rows_tenor, pa.float64()),
            "yield_pct": pa.array(rows_y, pa.float64()),
        }
    )


def make_fundamentals(
    symbols: list[str] | None = None,
    quarters: int = 12,
    start: str = "2023-03-31",
    seed: int = 99,
) -> pa.Table:
    """Quarterly fundamentals with realistic reporting lags.

    ``ts`` is the *report* (public availability) timestamp; ``period_end`` the
    fiscal quarter end — the pair that makes point-in-time joins meaningful.
    """
    symbols = symbols or [f"STK{i:03d}" for i in range(50)]
    rng = np.random.default_rng(seed)
    q_ends = []
    d = np.datetime64(start, "D")
    for _ in range(quarters):
        q_ends.append(int(d.astype("datetime64[us]").astype("int64")))
        d = (d.astype("datetime64[M]") + 3).astype("datetime64[D]") - 1 + np.timedelta64(1, "D")
        # next quarter end: move to last day of the +3 month
        d = (d.astype("datetime64[M]") + 1).astype("datetime64[D]") - np.timedelta64(1, "D")

    rows = {k: [] for k in ("ts", "period_end", "symbol", "eps", "revenue_m", "book_value_m")}
    for sym in symbols:
        eps = float(rng.uniform(0.5, 4.0))
        rev = float(rng.uniform(500, 20_000))
        bv = rev * rng.uniform(0.5, 2.0)
        for pe in q_ends:
            lag_days = int(rng.integers(25, 55))  # reporting lag
            eps *= float(np.exp(rng.normal(0.01, 0.08)))
            rev *= float(np.exp(rng.normal(0.01, 0.04)))
            bv *= float(np.exp(rng.normal(0.008, 0.02)))
            rows["ts"].append(pe + lag_days * 86_400 * US + 21 * 3600 * US)
            rows["period_end"].append(pe)
            rows["symbol"].append(sym)
            rows["eps"].append(round(eps, 2))
            rows["revenue_m"].append(round(rev, 1))
            rows["book_value_m"].append(round(bv, 1))
    return pa.table(
        {
            "ts": _ts_array(np.array(rows["ts"], dtype="int64")),
            "period_end": _ts_array(np.array(rows["period_end"], dtype="int64")),
            "symbol": pa.array(rows["symbol"]),
            "eps": pa.array(rows["eps"], pa.float64()),
            "revenue_m": pa.array(rows["revenue_m"], pa.float64()),
            "book_value_m": pa.array(rows["book_value_m"], pa.float64()),
        }
    ).sort_by("ts")
