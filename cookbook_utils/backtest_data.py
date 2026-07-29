"""Deterministic event-driven backtest fixtures used by the recipes."""

from __future__ import annotations

import datetime as dt
import math

import pyarrow as pa

BOOK_DELTAS_SCHEMA = pa.schema(
    [
        pa.field("ts_init", pa.timestamp("ns"), nullable=False),
        pa.field("ts_event", pa.timestamp("ns"), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("outcome", pa.uint16(), nullable=False),
        pa.field("action", pa.string(), nullable=False),
        pa.field("side", pa.string()),
        pa.field("price", pa.float64()),
        pa.field("size", pa.float64()),
        pa.field("event_index", pa.int64(), nullable=False),
        pa.field("is_last", pa.bool_(), nullable=False),
        pa.field("source_vendor", pa.string()),
    ]
)

TRADES_SCHEMA = pa.schema(
    [
        pa.field("ts_init", pa.timestamp("ns"), nullable=False),
        pa.field("ts_event", pa.timestamp("ns"), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("outcome", pa.uint16(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("aggressor", pa.string()),
        pa.field("trade_id", pa.string()),
        pa.field("source_vendor", pa.string()),
    ]
)

INSTRUMENTS_SCHEMA = pa.schema(
    [
        pa.field("ts_init", pa.timestamp("ns"), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("outcome", pa.uint16(), nullable=False),
        pa.field("outcome_label", pa.string(), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("expiration_ns", pa.int64()),
        pa.field("settlement_observable_ns", pa.int64()),
    ]
)


def make_backtest_fixture(
    *,
    steps: int = 180,
    instrument_id: str = "RATE-CUT-YES",
    start: str = "2026-06-01T14:00:00",
) -> dict[str, pa.Table]:
    """Build instruments, L2 snapshots and prints for one prediction market."""
    base = dt.datetime.fromisoformat(start)
    instruments = pa.table(
        {
            "ts_init": [base, base],
            "instrument_id": [instrument_id, instrument_id],
            "venue": ["example-prediction"] * 2,
            "kind": ["prediction_market"] * 2,
            "outcome": [0, 1],
            "outcome_label": ["YES", "NO"],
            "tick_size": [0.0001, 0.0001],
            "lot_size": [1.0, 1.0],
            "expiration_ns": [None, None],
            "settlement_observable_ns": [None, None],
        },
        schema=INSTRUMENTS_SCHEMA,
    )

    book: dict[str, list] = {name: [] for name in BOOK_DELTAS_SCHEMA.names}
    trades: dict[str, list] = {name: [] for name in TRADES_SCHEMA.names}
    for step in range(1, steps + 1):
        at = base + dt.timedelta(seconds=step)
        mid = 0.48 + 0.00035 * step + 0.018 * math.sin(step / 14)
        bid = round(mid - 0.005, 4)
        ask = round(mid + 0.005, 4)
        displayed = 80.0 + float((step % 5) * 10)

        for index, (side, price) in enumerate((("buy", bid), ("sell", ask))):
            book["ts_init"].append(at)
            book["ts_event"].append(at)
            book["instrument_id"].append(instrument_id)
            book["outcome"].append(0)
            book["action"].append("snapshot")
            book["side"].append(side)
            book["price"].append(price)
            book["size"].append(displayed)
            book["event_index"].append(step)
            book["is_last"].append(index == 1)
            book["source_vendor"].append("cookbook-sim")

        aggressor = "sell" if step % 3 == 0 else "buy"
        trades["ts_init"].append(at)
        trades["ts_event"].append(at)
        trades["instrument_id"].append(instrument_id)
        trades["outcome"].append(0)
        trades["price"].append(bid if aggressor == "sell" else ask)
        trades["size"].append(25.0 + float((step % 4) * 10))
        trades["aggressor"].append(aggressor)
        trades["trade_id"].append(f"T{step:05d}")
        trades["source_vendor"].append("cookbook-sim")

    return {
        "instruments": instruments,
        "book_deltas": pa.table(book, schema=BOOK_DELTAS_SCHEMA),
        "trades": pa.table(trades, schema=TRADES_SCHEMA),
    }
