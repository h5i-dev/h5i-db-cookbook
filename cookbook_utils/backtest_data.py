"""Deterministic event-driven backtest fixtures used by the recipes."""

from __future__ import annotations

import datetime as dt
import math
import random

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


RESOLUTIONS_SCHEMA = pa.schema(
    [
        pa.field("ts_init", pa.timestamp("ns"), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("winner_outcome", pa.uint16(), nullable=False),
    ]
)

_TICK = 0.001


def _epoch_nanos(moment: dt.datetime) -> int:
    """Wall-clock datetime to epoch nanoseconds, UTC."""
    return int(moment.replace(tzinfo=dt.timezone.utc).timestamp() * 1_000_000_000)


def _round_tick(value: float) -> float:
    """Snap to the 0.1-cent grid these venues quote on."""
    return round(round(value / _TICK) * _TICK, 4)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def make_prediction_markets(
    *,
    n_markets: int = 240,
    steps: int = 48,
    start: str = "2026-05-01T12:00:00",
    step_minutes: int = 15,
    seed: int = 11,
    longshot_bias: float = 0.12,
    basis_amplitude: float = 0.035,
    settlement_lag_minutes: int = 45,
    tail_steps: int = 4,
) -> dict[str, pa.Table]:
    """Build a panel of binary prediction markets with known ground truth.

    Every market carries a latent probability ``p_true``. Quotes are that
    probability *compressed toward 0.5* by ``longshot_bias``, which is the
    favorite-longshot bias as it appears in real event contracts: longshots
    trade above their true chance and favorites below it. The path mean-reverts
    to the quoted probability rather than the true one, so the mispricing is
    persistent instead of decaying over the session.

    Winners are assigned by systematic sampling over markets sorted by
    ``p_true``, so the realized frequency inside any price bucket tracks the
    injected bias instead of sampling noise. That is what makes calibration
    measurable at this panel size; on real data you would need far more markets.

    The NO book is the complement of the YES book plus a deterministic ``basis``
    in absolute probability, so YES+NO departs from 1.00 by roughly the same
    amount at every price level. A fee that scales with ``p*(1-p)`` therefore
    bites unevenly across the panel, which recipe 05/01 measures.

    Trading runs for ``steps`` instants and expiry lands on the last one. The
    result becomes observable ``settlement_lag_minutes`` later, and ``tail_steps``
    further snapshots quote the resolved market at ~1.00 / ~0.00, the way a
    recorder that keeps capturing past resolution would. A replay over the whole
    window therefore reaches settlement; a replay that stops inside the trading
    session does not, which recipe 05/05 uses deliberately.

    Returns the four canonical tables: ``instruments``, ``book_deltas``,
    ``trades`` and ``resolutions``, each sorted by ``ts_init``.
    """
    if not 0.0 <= longshot_bias < 1.0:
        raise ValueError("longshot_bias must be in [0, 1)")
    if steps < 2:
        raise ValueError("a market needs at least two trading instants")
    rng = random.Random(seed)
    base = dt.datetime.fromisoformat(start)
    step = dt.timedelta(minutes=step_minutes)
    expires_at = base + step * (steps - 1)
    observable_at = expires_at + dt.timedelta(minutes=settlement_lag_minutes)

    markets = []
    for index in range(n_markets):
        p_true = 0.04 + 0.92 * (index + 0.5) / n_markets
        markets.append(
            {
                "instrument_id": f"EVENT-{index:04d}",
                "p_true": p_true,
                "p_quote": 0.5 + (p_true - 0.5) * (1.0 - longshot_bias),
                "vol": 0.9 + 0.4 * rng.random(),
                "phase": rng.random() * math.tau,
            }
        )

    # Systematic sampling: across any contiguous block of sorted p_true, the
    # count of YES winners tracks that block's summed probability.
    cumulative = 0.0
    wins = 0
    for market in sorted(markets, key=lambda m: m["p_true"]):
        cumulative += market["p_true"]
        market["yes_wins"] = cumulative - wins >= 1.0 - 1e-12
        wins += int(market["yes_wins"])

    # Walk each market's price path first, then emit rows in time order so
    # event_index increases with the clock, as a recorder's sequence would.
    for market in markets:
        mid = market["p_quote"]
        path = []
        for index in range(steps):
            pull = 0.10 * (market["p_quote"] - mid)
            shock = (rng.random() - 0.5) * 0.045 * market["vol"]
            mid = _clip(mid + pull + shock * math.sqrt(mid * (1.0 - mid)), 0.01, 0.99)
            half_spread = _round_tick(0.004 + 0.010 * abs(mid - 0.5))
            basis = basis_amplitude * math.sin(market["phase"] + index / 5.0)
            no_mid = _clip(1.0 - mid + basis, 0.01, 0.99)
            # Each side of each book carries its own depth cycle, so top-of-book
            # imbalance varies instead of being a constant ratio. Without this a
            # microprice is just an affine function of the spread.
            def depth(offset: float, period: float) -> float:
                return _round_tick(
                    110.0 + 55.0 * math.sin(market["phase"] + offset + index / period)
                )

            path.append(
                {
                    "yes": (
                        _round_tick(_clip(mid - half_spread, _TICK, 1 - _TICK)),
                        _round_tick(_clip(mid + half_spread, _TICK, 1 - _TICK)),
                        depth(0.0, 3.0),
                        depth(1.7, 4.5),
                    ),
                    "no": (
                        _round_tick(_clip(no_mid - half_spread, _TICK, 1 - _TICK)),
                        _round_tick(_clip(no_mid + half_spread, _TICK, 1 - _TICK)),
                        depth(0.9, 5.0),
                        depth(2.6, 3.5),
                    ),
                    "aggressor": "buy" if shock >= 0 else "sell",
                    "trade_size": _round_tick(20.0 + 30.0 * rng.random()),
                }
            )
        market["path"] = path

    instruments: dict[str, list] = {name: [] for name in INSTRUMENTS_SCHEMA.names}
    book: dict[str, list] = {name: [] for name in BOOK_DELTAS_SCHEMA.names}
    trades: dict[str, list] = {name: [] for name in TRADES_SCHEMA.names}
    resolutions: dict[str, list] = {name: [] for name in RESOLUTIONS_SCHEMA.names}

    for market in markets:
        for outcome, label in ((0, "YES"), (1, "NO")):
            instruments["ts_init"].append(base)
            instruments["instrument_id"].append(market["instrument_id"])
            instruments["venue"].append("example-prediction")
            instruments["kind"].append("prediction_market")
            instruments["outcome"].append(outcome)
            instruments["outcome_label"].append(label)
            instruments["tick_size"].append(_TICK)
            instruments["lot_size"].append(1.0)
            instruments["expiration_ns"].append(_epoch_nanos(expires_at))
            instruments["settlement_observable_ns"].append(_epoch_nanos(observable_at))
        resolutions["ts_init"].append(observable_at)
        resolutions["instrument_id"].append(market["instrument_id"])
        resolutions["winner_outcome"].append(0 if market["yes_wins"] else 1)

    event_index = 0

    def _snapshot(at, instrument_id, outcome, bid, ask, bid_size, ask_size):
        """Emit one atomic book event: rows share an event_index, is_last ends it."""
        nonlocal event_index
        event_index += 1
        for position, (side, price, quantity) in enumerate(
            (("buy", bid, bid_size), ("sell", ask, ask_size))
        ):
            book["ts_init"].append(at)
            book["ts_event"].append(at)
            book["instrument_id"].append(instrument_id)
            book["outcome"].append(outcome)
            book["action"].append("snapshot")
            book["side"].append(side)
            book["price"].append(price)
            book["size"].append(quantity)
            book["event_index"].append(event_index)
            book["is_last"].append(position == 1)
            book["source_vendor"].append("cookbook-sim")

    for index in range(steps):
        at = base + step * index
        for market in markets:
            state = market["path"][index]
            yes_bid, yes_ask, yes_bid_depth, yes_ask_depth = state["yes"]
            no_bid, no_ask, no_bid_depth, no_ask_depth = state["no"]
            _snapshot(at, market["instrument_id"], 0, yes_bid, yes_ask, yes_bid_depth, yes_ask_depth)
            _snapshot(at, market["instrument_id"], 1, no_bid, no_ask, no_bid_depth, no_ask_depth)

            trades["ts_init"].append(at)
            trades["ts_event"].append(at)
            trades["instrument_id"].append(market["instrument_id"])
            trades["outcome"].append(0)
            trades["price"].append(yes_ask if state["aggressor"] == "buy" else yes_bid)
            trades["size"].append(state["trade_size"])
            trades["aggressor"].append(state["aggressor"])
            trades["trade_id"].append(f"{market['instrument_id']}-{index:04d}")
            trades["source_vendor"].append("cookbook-sim")

    # After the result is observable the book collapses onto the winner. These
    # rows are what let a full-window replay reach settlement.
    for tail in range(tail_steps):
        at = observable_at + step * tail
        for market in markets:
            winner = 0 if market["yes_wins"] else 1
            for outcome in (0, 1):
                won = outcome == winner
                _snapshot(
                    at,
                    market["instrument_id"],
                    outcome,
                    0.990 if won else 0.001,
                    0.999 if won else 0.010,
                    500.0,
                    500.0,
                )

    def _table(payload: dict[str, list], schema: pa.Schema) -> pa.Table:
        return pa.table(payload, schema=schema)

    return {
        "instruments": _table(instruments, INSTRUMENTS_SCHEMA).sort_by(
            [("ts_init", "ascending"), ("instrument_id", "ascending"), ("outcome", "ascending")]
        ),
        "book_deltas": _table(book, BOOK_DELTAS_SCHEMA),
        "trades": _table(trades, TRADES_SCHEMA),
        "resolutions": _table(resolutions, RESOLUTIONS_SCHEMA).sort_by(
            [("ts_init", "ascending"), ("instrument_id", "ascending")]
        ),
    }


def market_truth(tables: dict[str, pa.Table]) -> pa.Table:
    """One row per market: the resolved winner, for scoring after a run.

    Kept separate from the feature path on purpose. A strategy must never read
    this; it is the answer key, and joining it early is how look-ahead enters a
    prediction-market study.
    """
    resolutions = tables["resolutions"]
    return pa.table(
        {
            "instrument_id": resolutions.column("instrument_id"),
            "yes_won": pa.array(
                [value.as_py() == 0 for value in resolutions.column("winner_outcome")],
                pa.bool_(),
            ),
        }
    )
