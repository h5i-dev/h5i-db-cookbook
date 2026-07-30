"""Bounded, auditable loading for the Marvingozo Polymarket dataset.

The public dataset is large enough that a tutorial must never read a daily
tick file eagerly.  These helpers use lazy Parquet scans, select one market,
and convert one snapshot day into h5i-db's canonical replay tables.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from .backtest_data import BOOK_DELTAS_SCHEMA, INSTRUMENTS_SCHEMA, TRADES_SCHEMA

DATASET = "marvingozo/polymarket-tick-level-orderbook-dataset"
LICENSE = "CC BY-NC 4.0"
SAMPLE_DATE = dt.date(2026, 3, 9)
SAMPLE_FILES = (
    "features/ml_features_1m_v2.parquet",
    "labels/market_targets.parquet",
    "labels/trades.parquet",
    "snapshots/snapshots_2026-03-09.parquet",
)


@dataclass(frozen=True)
class KaggleSample:
    """One bounded market sample and the evidence needed to audit it."""

    market_id: str
    question: str
    target: int
    instruments: pa.Table
    book_deltas: pa.Table
    trades: pa.Table
    features: pa.Table
    audit: dict[str, Any]


def expected_paths(cache_dir: str | Path) -> dict[str, Path]:
    """Return local paths after Kaggle's per-file downloads are unzipped."""
    root = Path(cache_dir)
    return {remote: root / Path(remote).name for remote in SAMPLE_FILES}


def missing_files(cache_dir: str | Path) -> list[str]:
    """List remote dataset paths that are not present in ``cache_dir``."""
    paths = expected_paths(cache_dir)
    return [remote for remote, local in paths.items() if not local.is_file()]


def download_commands(cache_dir: str | Path = "data/cache/kaggle-polymarket") -> list[str]:
    """Build bounded Kaggle CLI commands; total compressed size is about 165 MB."""
    return [
        f"kaggle datasets download {DATASET} -f {remote} "
        f"-p {Path(cache_dir)} --unzip"
        for remote in SAMPLE_FILES
    ]


def source_manifest(cache_dir: str | Path) -> pa.Table:
    """Hash the bounded source files so a run can cite exact external bytes."""
    rows = {"dataset": [], "remote_path": [], "bytes": [], "sha256": [], "license": []}
    for remote, local in expected_paths(cache_dir).items():
        digest = hashlib.sha256()
        with local.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        rows["dataset"].append(DATASET)
        rows["remote_path"].append(remote)
        rows["bytes"].append(local.stat().st_size)
        rows["sha256"].append(digest.hexdigest())
        rows["license"].append(LICENSE)
    return pa.table(rows)


def _utc_from_millis(value: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value / 1_000, tz=dt.timezone.utc)


def _utc_from_seconds(value: int) -> dt.datetime:
    return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)


def _parse_end_date(value: str | None) -> int | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return int(parsed.timestamp() * 1_000_000_000)


def _select_market(
    snapshots: pl.LazyFrame,
    features: pl.LazyFrame,
    trades: pl.LazyFrame,
    targets: pl.LazyFrame,
    day_start: dt.datetime,
    day_end: dt.datetime,
) -> dict[str, Any]:
    snapshot_counts = snapshots.group_by("market_id").len().rename({"len": "snapshots"})
    feature_counts = (
        features.filter(pl.col("minute_bar").is_between(day_start, day_end, closed="left"))
        .group_by("market_id")
        .len()
        .rename({"len": "bars"})
    )
    trade_counts = (
        trades.filter(
            pl.col("timestamp").is_between(
                int(day_start.timestamp()),
                int(day_end.timestamp()),
                closed="left",
            )
        )
        .group_by("condition_id")
        .len()
        .rename({"len": "trades"})
    )
    candidates = (
        snapshot_counts.join(feature_counts, on="market_id", how="inner")
        .join(
            trade_counts,
            left_on="market_id",
            right_on="condition_id",
            how="inner",
        )
        .join(
            targets.select(
                "condition_id", "question", "target", "closed", "end_date", "volume"
            ),
            left_on="market_id",
            right_on="condition_id",
            how="inner",
        )
        .filter(pl.col("closed") & (pl.col("bars") >= 60))
        .sort(["trades", "snapshots", "market_id"], descending=[True, True, False])
        .head(1)
        .collect()
    )
    if candidates.is_empty():
        raise ValueError("no closed market overlaps snapshots, features, and trades")
    return candidates.to_dicts()[0]


def _normalize_snapshots(
    frame: pl.DataFrame,
    market_id: str,
    depth_levels: int,
) -> tuple[pa.Table, dict[str, int]]:
    columns: dict[str, list[Any]] = {name: [] for name in BOOK_DELTAS_SCHEMA.names}
    clock_adjustments = 0
    crossed_snapshots = 0

    for event_index, row in enumerate(frame.iter_rows(named=True), start=1):
        payload = json.loads(row["data"])
        if payload.get("side") != "YES":
            continue
        bids = sorted(
            ((float(price), float(size)) for price, size in payload["bids"]),
            reverse=True,
        )[:depth_levels]
        asks = sorted(
            (float(price), float(size)) for price, size in payload["asks"]
        )[:depth_levels]
        if not bids or not asks:
            continue
        if bids[0][0] >= asks[0][0]:
            crossed_snapshots += 1
            continue

        event_ms = int(row["timestamp_created_at"])
        received_ms = int(row["timestamp_received"])
        init_ms = max(event_ms, received_ms)
        clock_adjustments += int(init_ms != received_ms)
        ts_event = _utc_from_millis(event_ms)
        ts_init = _utc_from_millis(init_ms)
        levels = [("buy", *level) for level in bids] + [
            ("sell", *level) for level in asks
        ]

        for level_index, (side, price, size) in enumerate(levels):
            columns["ts_init"].append(ts_init)
            columns["ts_event"].append(ts_event)
            columns["instrument_id"].append(market_id)
            columns["outcome"].append(0)
            columns["action"].append("snapshot")
            columns["side"].append(side)
            columns["price"].append(price)
            columns["size"].append(size)
            columns["event_index"].append(event_index)
            columns["is_last"].append(level_index == len(levels) - 1)
            columns["source_vendor"].append("kaggle-marvingozo-polymarket-v3")

    return pa.table(columns, schema=BOOK_DELTAS_SCHEMA), {
        "clock_adjustments": clock_adjustments,
        "crossed_snapshots_dropped": crossed_snapshots,
    }


def _normalize_trades(frame: pl.DataFrame, market_id: str) -> pa.Table:
    columns: dict[str, list[Any]] = {name: [] for name in TRADES_SCHEMA.names}
    for index, row in enumerate(frame.iter_rows(named=True), start=1):
        price = float(row["price"])
        if str(row["outcome"]).lower() == "no":
            price = 1.0 - price
        timestamp = int(row["timestamp"])
        at = _utc_from_seconds(timestamp)
        columns["ts_init"].append(at)
        columns["ts_event"].append(at)
        columns["instrument_id"].append(market_id)
        columns["outcome"].append(0)
        columns["price"].append(price)
        columns["size"].append(float(row["size"]))
        # The publisher calls `side` a transaction side, not explicitly a
        # taker/aggressor side. Do not invent queue-depletion evidence.
        columns["aggressor"].append(None)
        columns["trade_id"].append(f"kaggle-{timestamp}-{index:06d}")
        columns["source_vendor"].append("kaggle-marvingozo-polymarket-v3")
    return pa.table(columns, schema=TRADES_SCHEMA)


def load_kaggle_sample(
    cache_dir: str | Path = "data/cache/kaggle-polymarket",
    *,
    depth_levels: int = 10,
    market_id: str | None = None,
) -> KaggleSample:
    """Load one real market without materializing the dataset-wide Parquets.

    Feature timestamps are shifted one minute to represent bar-close
    availability, and the bundled resolution target is deliberately omitted
    from the feature table.
    """
    if depth_levels < 1:
        raise ValueError("depth_levels must be positive")
    missing = missing_files(cache_dir)
    if missing:
        commands = "\n".join(download_commands(cache_dir))
        raise FileNotFoundError(
            f"missing Kaggle sample files: {missing}\nRun:\n{commands}"
        )

    paths = expected_paths(cache_dir)
    snapshots = pl.scan_parquet(paths[SAMPLE_FILES[3]])
    features = pl.scan_parquet(paths[SAMPLE_FILES[0]])
    targets = pl.scan_parquet(paths[SAMPLE_FILES[1]])
    trades = pl.scan_parquet(paths[SAMPLE_FILES[2]])
    day_start = dt.datetime.combine(
        SAMPLE_DATE, dt.time.min, tzinfo=dt.timezone.utc
    )
    day_end = day_start + dt.timedelta(days=1)

    selected = _select_market(
        snapshots, features, trades, targets, day_start, day_end
    )
    chosen = market_id or str(selected["market_id"])
    if market_id is not None:
        target_rows = (
            targets.filter(pl.col("condition_id") == chosen).head(1).collect()
        )
        if target_rows.is_empty():
            raise ValueError(f"market {chosen!r} has no target metadata")
        selected.update(target_rows.to_dicts()[0])

    snapshot_rows = (
        snapshots.filter(pl.col("market_id") == chosen)
        .sort(["timestamp_received", "timestamp_created_at"])
        .collect()
    )
    feature_rows = (
        features.filter(
            (pl.col("market_id") == chosen)
            & pl.col("minute_bar").is_between(day_start, day_end, closed="left")
        )
        .sort("minute_bar")
        .with_columns(
            (pl.col("minute_bar") + pl.duration(minutes=1))
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("ns"))
            .alias("ts_init"),
            pl.col("minute_bar")
            .dt.replace_time_zone(None)
            .cast(pl.Datetime("ns")),
        )
        .drop("target")
        .select("ts_init", pl.exclude("ts_init"))
        .collect()
    )
    trade_rows = (
        trades.filter(
            (pl.col("condition_id") == chosen)
            & pl.col("timestamp").is_between(
                int(day_start.timestamp()),
                int(day_end.timestamp()),
                closed="left",
            )
        )
        .sort("timestamp")
        .collect()
    )
    books, book_audit = _normalize_snapshots(snapshot_rows, chosen, depth_levels)
    normalized_trades = _normalize_trades(trade_rows, chosen)

    first_ts = books.column("ts_init")[0].as_py()
    end_ns = _parse_end_date(selected.get("end_date"))
    instruments = pa.table(
        {
            "ts_init": [first_ts, first_ts],
            "instrument_id": [chosen, chosen],
            "venue": ["polymarket", "polymarket"],
            "kind": ["prediction_market", "prediction_market"],
            "outcome": [0, 1],
            "outcome_label": ["YES", "NO"],
            "tick_size": [0.001, 0.001],
            "lot_size": [0.01, 0.01],
            "expiration_ns": [end_ns, end_ns],
            # The target file contains a result but no result-observation time.
            "settlement_observable_ns": [None, None],
        },
        schema=INSTRUMENTS_SCHEMA,
    )
    audit = {
        "dataset": DATASET,
        "license": LICENSE,
        "sample_date": SAMPLE_DATE.isoformat(),
        "source_snapshot_rows": snapshot_rows.height,
        "source_feature_rows": feature_rows.height,
        "source_trade_rows": trade_rows.height,
        "canonical_book_rows": books.num_rows,
        "canonical_trade_rows": normalized_trades.num_rows,
        "depth_levels": depth_levels,
        "trade_aggressor_known": False,
        **book_audit,
    }
    return KaggleSample(
        market_id=chosen,
        question=str(selected["question"]),
        target=int(selected["target"]),
        instruments=instruments,
        book_deltas=books,
        trades=normalized_trades,
        features=feature_rows.to_arrow(),
        audit=audit,
    )
