"""Kalshi's public REST API, cached to Parquet.

Kalshi publishes no historical order book, and third-party archives of one are
either paid or weeks behind. What the exchange does publish, to anyone, with no
account and no key, is enough to study execution honestly:

* **settled market definitions**, carrying the exchange's own `result`, the
  time trading stopped and the time settlement was recorded;
* **one-minute candles**, each carrying the closing *bid and ask* of that
  minute rather than only a traded price;
* **the trade tape**, every print with its size and which side crossed.

Fetching lives here rather than in a recipe for the usual reason: the network
call is the one line that cannot be tested offline, so it is kept in one place,
wrapped in a cache, and everything downstream is a pure function of files.
First call hits the API; afterwards a notebook re-runs offline from
``data/cache/kalshi``.

What this cannot give you is depth. A candle carries the touch and nothing
behind it, so a study built on this source can price a small order crossing the
spread and must refuse to price a large one. That limit belongs to the source
and is restated wherever it matters.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent.parent / "data" / "cache" / "kalshi"

#: Kalshi's published taker rate for its general fee schedule. Series with
#: their own schedule exist, so this is a default rather than a constant.
TAKER_FEE_RATE = 0.07

MARKETS_SCHEMA = pa.schema(
    [
        ("ticker", pa.string()),
        ("event_ticker", pa.string()),
        ("series_ticker", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("open_time", pa.timestamp("ns", tz="UTC")),
        ("close_time", pa.timestamp("ns", tz="UTC")),
        ("settlement_ts", pa.timestamp("ns", tz="UTC")),
        ("result", pa.string()),
        ("volume", pa.float64()),
        ("open_interest", pa.float64()),
        ("tick_size", pa.float64()),
    ]
)

CANDLES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("ticker", pa.string()),
        ("yes_bid_close", pa.float64()),
        ("yes_ask_close", pa.float64()),
        ("price_close", pa.float64()),
        ("volume", pa.float64()),
        ("open_interest", pa.float64()),
    ]
)

TRADES_SCHEMA = pa.schema(
    [
        ("ts", pa.timestamp("ns", tz="UTC")),
        ("ticker", pa.string()),
        ("yes_price", pa.float64()),
        ("count", pa.float64()),
        ("taker_side", pa.string()),
        ("trade_id", pa.string()),
    ]
)


def taker_fee(price: float, contracts: float, rate: float = TAKER_FEE_RATE) -> float:
    """Kalshi's published trading fee, in dollars, for one execution.

    ``rate * contracts * p * (1 - p)``, rounded **up** to the next cent. The
    quadratic is the variance of a coin flip, so the fee is largest on a
    contract the market thinks is a toss-up and vanishes at either end. Rounding
    up rather than to nearest is why a run of small orders pays more than one
    large one for the same contracts.
    """
    cents = rate * contracts * price * (1.0 - price) * 100.0
    # Round to nanocents before rounding up. Without it 0.07 * 100 * 0.25 lands
    # at 1.7500000000000002 in binary floating point and the ceiling charges a
    # cent nobody owes, exactly at the price where the fee matters most.
    return math.ceil(round(cents, 9)) / 100.0


def _request(path: str, params: Mapping[str, Any], attempts: int = 4) -> dict:
    """One GET against the public API, with backoff on rate limiting."""
    url = f"{KALSHI_API}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            # 429 is the documented rate limit; 5xx is the host having a
            # moment. Anything else (404 for a ticker that does not exist,
            # 400 for a malformed range) is a fact about the request and is
            # raised immediately rather than retried four times.
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            time.sleep(2.0**attempt)
    raise RuntimeError(f"unreachable: {url}")


def _require(payload: Mapping[str, Any], keys: Iterable[str], what: str) -> None:
    """Fail loudly, naming what actually arrived.

    A public API renames its fields without warning, and the failure mode that
    costs a day is a silently empty table rather than an exception.
    """
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(
            f"{what}: response is missing {missing}; it carried {sorted(payload)}"
        )


def _dollars(payload: Mapping[str, Any], base: str) -> float | None:
    """A price from either spelling the API uses.

    Kalshi is mid-migration from integer cents (`yes_bid`) to decimal strings
    (`yes_bid_dollars`). Reading both keeps a cache built today comparable with
    one built before the change, and neither is guessed at: a payload carrying
    neither spelling returns None and the caller drops the row.
    """
    if f"{base}_dollars" in payload:
        value = payload[f"{base}_dollars"]
        return None if value is None else float(value)
    if base in payload:
        value = payload[base]
        return None if value is None else float(value) / 100.0
    return None


def _number(payload: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if payload.get(name) is not None:
            return float(payload[name])
    return 0.0


def _cached(path: Path, schema: pa.Schema, build: Callable[[], list[dict]], refresh: bool) -> pa.Table:
    if path.exists() and not refresh:
        return pq.read_table(path)
    rows = build()
    table = pa.Table.from_pylist(rows, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return table


def _stamp(value: Any) -> dt.datetime | None:
    """An ISO-8601 instant from the API as an aware datetime."""
    if value in (None, ""):
        return None
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def fetch_markets(event_tickers: Sequence[str], refresh: bool = False) -> pa.Table:
    """Market definitions for whole events, one row per market.

    An *event* is one question ("the high temperature in New York on 1 August")
    and its *markets* are the mutually exclusive answers, each a separate
    contract with its own ticker. Fetching by event is what gives a study a
    ladder rather than an arbitrary basket.
    """

    def build() -> list[dict]:
        rows = []
        for event in event_tickers:
            payload = _request("/markets", {"event_ticker": event, "limit": 200})
            _require(payload, ["markets"], f"markets for {event}")
            for market in payload["markets"]:
                _require(market, ["ticker", "status"], "market definition")
                steps = market.get("price_ranges") or [{}]
                rows.append(
                    {
                        "ticker": market["ticker"],
                        "event_ticker": market.get("event_ticker"),
                        "series_ticker": event.split("-")[0],
                        "title": market.get("title"),
                        "subtitle": market.get("yes_sub_title") or market.get("subtitle"),
                        "open_time": _stamp(market.get("open_time")),
                        "close_time": _stamp(market.get("close_time")),
                        "settlement_ts": _stamp(market.get("settlement_ts")),
                        "result": market.get("result") or None,
                        "volume": _number(market, "volume_fp", "volume"),
                        "open_interest": _number(market, "open_interest_fp", "open_interest"),
                        "tick_size": float(steps[0].get("step", 0.01) or 0.01),
                    }
                )
        return rows

    key = "-".join(sorted(event_tickers))[:100]
    return _cached(CACHE / f"markets_{key}.parquet", MARKETS_SCHEMA, build, refresh)


def fetch_candles(
    markets: pa.Table,
    period_minutes: int = 1,
    refresh: bool = False,
) -> pa.Table:
    """One row per market per period, carrying that period's closing touch.

    Each market is fetched over its own open-to-close window and cached in its
    own file, so adding a market to a study costs one request rather than a
    refetch. Periods where the venue reports no book are returned as they came;
    dropping them is the caller's decision, not this function's.
    """
    frames = []
    for row in markets.to_pylist():
        ticker, series = row["ticker"], row["series_ticker"]
        start = int(row["open_time"].timestamp())
        end = int(row["close_time"].timestamp())

        def build(ticker=ticker, series=series, start=start, end=end) -> list[dict]:
            rows = []
            # The endpoint caps a response, so a long-lived market is walked in
            # windows rather than asked for in one go.
            span = period_minutes * 60 * 4_000
            for window_start in range(start, end, span):
                payload = _request(
                    f"/series/{series}/markets/{ticker}/candlesticks",
                    {
                        "start_ts": window_start,
                        "end_ts": min(window_start + span, end),
                        "period_interval": period_minutes,
                    },
                )
                _require(payload, ["candlesticks"], f"candles for {ticker}")
                for candle in payload["candlesticks"]:
                    bid = _dollars(candle.get("yes_bid") or {}, "close")
                    ask = _dollars(candle.get("yes_ask") or {}, "close")
                    rows.append(
                        {
                            "ts": dt.datetime.fromtimestamp(
                                int(candle["end_period_ts"]), dt.timezone.utc
                            ),
                            "ticker": ticker,
                            "yes_bid_close": bid,
                            "yes_ask_close": ask,
                            "price_close": _dollars(candle.get("price") or {}, "close"),
                            "volume": _number(candle, "volume_fp", "volume"),
                            "open_interest": _number(candle, "open_interest_fp", "open_interest"),
                        }
                    )
            return rows

        frames.append(
            _cached(
                CACHE / f"candles_{period_minutes}m_{ticker}.parquet",
                CANDLES_SCHEMA,
                build,
                refresh,
            )
        )
    return pa.concat_tables(frames).sort_by([("ts", "ascending")])


def fetch_trades(markets: pa.Table, refresh: bool = False) -> pa.Table:
    """Every print in each market, paged to the beginning of its life."""
    frames = []
    for row in markets.to_pylist():
        ticker = row["ticker"]

        def build(ticker=ticker) -> list[dict]:
            rows, cursor = [], None
            while True:
                params: dict[str, Any] = {"ticker": ticker, "limit": 1000}
                if cursor:
                    params["cursor"] = cursor
                payload = _request("/markets/trades", params)
                _require(payload, ["trades"], f"trades for {ticker}")
                for trade in payload["trades"]:
                    rows.append(
                        {
                            "ts": _stamp(trade["created_time"]),
                            "ticker": ticker,
                            "yes_price": _dollars(trade, "yes_price"),
                            "count": _number(trade, "count_fp", "count"),
                            "taker_side": trade.get("taker_side"),
                            "trade_id": trade.get("trade_id"),
                        }
                    )
                cursor = payload.get("cursor")
                if not cursor or not payload["trades"]:
                    return rows

        frames.append(_cached(CACHE / f"trades_{ticker}.parquet", TRADES_SCHEMA, build, refresh))
    return pa.concat_tables(frames).sort_by([("ts", "ascending")])
