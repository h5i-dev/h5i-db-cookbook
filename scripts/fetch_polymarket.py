#!/usr/bin/env python
"""Download Polymarket market definitions and books into a local mirror.

The step before `h5i_db.venues`. That module deliberately does not fetch, so
this script is where credentials, retries, rate limits and caching live, and it
writes files in a layout `venues.ingest_archive` already reads:

    python scripts/fetch_polymarket.py markets --closed --limit 200 \\
        --out data/cache/pm-specs.json
    python scripts/fetch_polymarket.py books --specs data/cache/pm-specs.json \\
        --out data/cache/pm-mirror --max-markets 25

    python -m h5i_db.venues markets market.db data/cache/pm-specs.json
    python -m h5i_db.venues ingest  market.db data/cache/pm-specs.json \\
        --root data/cache/pm-mirror

Two design choices worth knowing.

**The transport is injected.** Everything except the socket call is a pure
function of a response, so pagination, retry, cache keys, shape validation and
Parquet output are tested offline against recorded fixtures
(`scripts/test_fetch_polymarket.py`). The live request is the only unverified
line, and it is one line.

**Responses are validated, not assumed.** A public API changes its field
spellings without warning. Every response passes through a check that names the
keys it actually received when they do not match, so a shape change is an
immediate, legible failure rather than a silently empty mirror.

Nothing here needs an API key. The endpoints used are public read-only.

What each endpoint can actually give you, checked against the live hosts rather
than inferred from documentation:

| command / function | covers | limit |
|---|---|---|
| `markets` | live or resolved definitions, tokens, outcomes, resolution | metadata only |
| `books` | current full depth for **live** markets | a resolved market 404s |
| `fetch_price_history` | historical mid points for any market | points, so no depth and no queue claims |

So a historical depth study needs an archive: the fetch path gives live books
and past prices, not past books. Recipe 05/08 uses a captured archive for
exactly that reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

GAMMA_URL = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com"

#: Sent so the operator of a public endpoint can identify the traffic. Being
#: identifiable is the minimum courtesy when reading someone else's API, and it
#: is also load-bearing: these hosts answer 403 to the default
#: `Python-urllib/3.x` agent, which reads exactly like a blocked network and is
#: not one. Verified 2026-07-30: same URL, 403 without this header and 200 with.
USER_AGENT = "h5i-db-cookbook/1.0 (+https://github.com/h5i-dev/h5i-db-cookbook)"

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class FetchError(RuntimeError):
    """A request failed, or a response did not look like what it should."""


# ---------------------------------------------------------------- transport


@dataclass
class HttpTransport:
    """urllib with a disk cache, polite pacing and bounded retries.

    The cache is keyed by the full URL, so re-running a fetch is free and an
    interrupted backfill resumes instead of restarting. `min_interval` paces
    requests; `attempts` bounds the retry on the statuses that mean "later".
    """

    cache_dir: Optional[Path] = None
    timeout: float = 30.0
    attempts: int = 4
    min_interval: float = 0.2
    # Separate from `min_interval` on purpose. Pacing is politeness between
    # successful requests; backoff is how long to wait after being told "later".
    # Deriving one from the other means a caller who disables pacing also
    # disables backoff, and then a rate limit becomes a hot loop.
    retry_delay: float = 0.5
    backoff: float = 1.5
    sleep: Callable[[float], None] = time.sleep
    _last_call: float = field(default=0.0, repr=False)
    requests_made: int = field(default=0, repr=False)
    cache_hits: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be at least 1")
        if self.retry_delay <= 0 or self.backoff < 1.0:
            raise ValueError(
                "retry_delay must be positive and backoff at least 1.0, or a "
                "rate limit becomes a hot loop"
            )

    def _cache_path(self, url: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
        return Path(self.cache_dir) / f"{digest}.json"

    def get_json(self, url: str, params: Optional[Mapping[str, Any]] = None) -> Any:
        full = url if not params else f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        cached = self._cache_path(full)
        if cached is not None and cached.exists():
            self.cache_hits += 1
            return json.loads(cached.read_text(encoding="utf-8"))

        delay = self.retry_delay
        last: Optional[Exception] = None
        for attempt in range(1, self.attempts + 1):
            gap = time.monotonic() - self._last_call
            if gap < self.min_interval:
                self.sleep(self.min_interval - gap)
            request = urllib.request.Request(full, headers={"User-Agent": USER_AGENT})
            try:
                self.requests_made += 1
                self._last_call = time.monotonic()
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                payload = json.loads(body)
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
                return payload
            except urllib.error.HTTPError as error:
                last = error
                if error.code not in RETRY_STATUSES or attempt == self.attempts:
                    raise FetchError(
                        f"{full} returned HTTP {error.code}"
                        + (
                            ". This host refuses requests from some networks; a "
                            "403 here usually means the network, not the URL."
                            if error.code == 403
                            else ""
                        )
                    ) from error
            except urllib.error.URLError as error:
                last = error
                if attempt == self.attempts:
                    raise FetchError(f"{full} unreachable: {error}") from error
            self.sleep(delay)
            delay *= self.backoff
        raise FetchError(f"{full} failed after {self.attempts} attempts: {last}")


# ------------------------------------------------------------- validation


def _require_list(payload: Any, url: str) -> list[Any]:
    """A list response, or an error naming what arrived instead."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in ("data", "markets", "results", "history"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise FetchError(
            f"{url}: expected a list or a wrapper containing one; got an object "
            f"with keys {sorted(payload)}"
        )
    raise FetchError(f"{url}: expected a list, got {type(payload).__name__}")


def _require_keys(record: Mapping[str, Any], wanted: Sequence[str], what: str) -> None:
    missing = [key for key in wanted if key not in record]
    if missing:
        raise FetchError(
            f"{what}: response is missing {missing}. Keys present: "
            f"{sorted(record)[:14]}. The endpoint's field names may have changed."
        )


# ------------------------------------------------------------------ markets


def market_pages(
    transport: Any,
    *,
    closed: Optional[bool] = None,
    page_size: int = 100,
    max_pages: int = 50,
    base_url: str = GAMMA_URL,
    extra: Optional[Mapping[str, Any]] = None,
) -> Iterator[list[dict]]:
    """Walk the Gamma markets endpoint, one page at a time.

    Stops on the first short or empty page, and at `max_pages` regardless. The
    cap is deliberate: an unbounded crawl of someone else's API is not a
    reasonable default, and a caller who wants more can say so.
    """
    if page_size < 1 or max_pages < 1:
        raise ValueError("page_size and max_pages must be positive")
    url = f"{base_url}/markets"
    for page in range(max_pages):
        params: dict[str, Any] = {"limit": page_size, "offset": page * page_size}
        if closed is not None:
            params["closed"] = str(bool(closed)).lower()
        if extra:
            params.update(extra)
        records = _require_list(transport.get_json(url, params), url)
        if not records:
            return
        yield [record for record in records if isinstance(record, Mapping)]
        if len(records) < page_size:
            return


def fetch_markets(
    transport: Any,
    *,
    closed: Optional[bool] = True,
    limit: int = 200,
    page_size: int = 100,
    max_pages: int = 50,
    require_tokens: bool = True,
    base_url: str = GAMMA_URL,
) -> list[dict]:
    """Market definitions, in the shape `venues.polymarket_markets_from_json` reads.

    `require_tokens` drops markets with no CLOB token ids, because a market whose
    outcomes cannot be keyed to a token is not ingestible and keeping it would
    only produce a confusing failure two steps later.
    """
    out: list[dict] = []
    for records in market_pages(
        transport,
        closed=closed,
        page_size=page_size,
        max_pages=max_pages,
        base_url=base_url,
    ):
        for record in records:
            _require_keys(record, ("outcomes",), "gamma /markets")
            if require_tokens and not record.get("clobTokenIds"):
                continue
            out.append(dict(record))
            if len(out) >= limit:
                return out
    return out


def token_ids(market: Mapping[str, Any]) -> list[str]:
    """The CLOB token ids of one market payload, in outcome order."""
    raw = market.get("clobTokenIds")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError as error:
            raise FetchError(
                f"clobTokenIds is a string that is not JSON: {raw[:60]!r}"
            ) from error
    if not isinstance(raw, (list, tuple)) or not raw:
        raise FetchError(f"market {market.get('condition_id')} has no usable tokens")
    return [str(item) for item in raw]


# -------------------------------------------------------------------- books


def fetch_book(transport: Any, token_id: str, *, base_url: str = CLOB_URL) -> dict:
    """The current book for one token.

    Live markets only. A resolved market has no book to serve and the endpoint
    answers 404, which is why a `--closed` market list is the wrong input for
    this command. Verified against the live endpoint 2026-07-30: every token of
    six closed markets returned 404, every token of six open markets returned a
    book.
    """
    url = f"{base_url}/book"
    try:
        payload = transport.get_json(url, {"token_id": token_id})
    except FetchError as error:
        if "HTTP 404" in str(error):
            raise FetchError(
                f"no book for token {token_id[:16]}...: the CLOB serves books for "
                "live markets only, so a resolved market returns 404. For history "
                "use fetch_price_history (points, no depth) or an archive."
            ) from error
        raise
    if not isinstance(payload, Mapping):
        raise FetchError(f"{url}: expected an object, got {type(payload).__name__}")
    _require_keys(payload, ("bids", "asks"), "clob /book")
    return dict(payload)


def fetch_price_history(
    transport: Any,
    token_id: str,
    *,
    interval: str = "1d",
    fidelity: int = 1,
    base_url: str = CLOB_URL,
) -> list[dict]:
    """Historical mid points for one token.

    The public history endpoint returns points, not books. That is a real
    capability limit and it is why the mirror this writes supports bar-style
    research and not queue-position claims: there is no depth in a point.
    """
    url = f"{base_url}/prices-history"
    payload = transport.get_json(
        url, {"market": token_id, "interval": interval, "fidelity": fidelity}
    )
    points = _require_list(payload, url)
    out = []
    for point in points:
        if not isinstance(point, Mapping):
            continue
        _require_keys(point, ("t", "p"), "clob /prices-history")
        out.append({"t": int(point["t"]), "p": float(point["p"])})
    return out


# ------------------------------------------------------------------ writing


def _level_rows(levels: Any) -> list[dict[str, float]]:
    """Book levels in either spelling into `{price, size}` dicts."""
    out: list[dict[str, float]] = []
    for level in levels or ():
        if isinstance(level, Mapping):
            price, size = level.get("price"), level.get("size")
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price, size = level[0], level[1]
        else:
            continue
        if price is None or size is None:
            continue
        out.append({"price": float(price), "size": float(size)})
    return out


def book_archive_table(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Fetched books as a Parquet table `venues.PMXT_LAYOUT` already reads.

    Writing a layout the importer supports, rather than inventing one, is what
    keeps the fetch step separable: the mirror on disk is the interface, and any
    later fetcher that produces the same columns needs no new ingest code.
    """
    import pyarrow as pa

    level = pa.struct([("price", pa.float64()), ("size", pa.float64())])
    schema = pa.schema(
        [
            pa.field("event_type", pa.string()),
            pa.field("timestamp", pa.int64()),
            pa.field("market", pa.string()),
            pa.field("asset_id", pa.string()),
            pa.field("bids", pa.list_(level)),
            pa.field("asks", pa.list_(level)),
            pa.field("price", pa.float64()),
            pa.field("size", pa.float64()),
            pa.field("side", pa.string()),
        ]
    )
    ordered = sorted(rows, key=lambda row: (int(row["timestamp"]), str(row["asset_id"])))
    return pa.table(
        {
            "event_type": pa.array([row["event_type"] for row in ordered], pa.string()),
            "timestamp": pa.array([int(row["timestamp"]) for row in ordered], pa.int64()),
            "market": pa.array([row["market"] for row in ordered], pa.string()),
            "asset_id": pa.array([row["asset_id"] for row in ordered], pa.string()),
            "bids": pa.array([row.get("bids") for row in ordered], pa.list_(level)),
            "asks": pa.array([row.get("asks") for row in ordered], pa.list_(level)),
            "price": pa.array([row.get("price") for row in ordered], pa.float64()),
            "size": pa.array([row.get("size") for row in ordered], pa.float64()),
            "side": pa.array([row.get("side") for row in ordered], pa.string()),
        },
        schema=schema,
    )


def collect_books(
    transport: Any,
    markets: Sequence[Mapping[str, Any]],
    *,
    now_ms: Optional[int] = None,
    base_url: str = CLOB_URL,
    on_error: str = "record",
) -> tuple[list[dict], list[dict]]:
    """One book row per (market, token). Returns (rows, failures).

    A token that fails is recorded and skipped rather than aborting the run: on a
    few hundred markets, one delisted token should not cost the whole mirror. The
    failures come back so the caller can decide, and the CLI prints them.
    """
    if on_error not in ("record", "raise"):
        raise ValueError("on_error must be 'record' or 'raise'")
    stamp = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows: list[dict] = []
    failures: list[dict] = []
    for market in markets:
        condition = str(
            market.get("condition_id") or market.get("conditionId") or market.get("id") or ""
        )
        try:
            tokens = token_ids(market)
        except FetchError as error:
            failures.append({"market": condition, "reason": str(error)})
            continue
        for token in tokens:
            try:
                book = fetch_book(transport, token, base_url=base_url)
            except FetchError as error:
                if on_error == "raise":
                    raise
                failures.append({"market": condition, "token": token, "reason": str(error)})
                continue
            rows.append(
                {
                    "event_type": "book",
                    "timestamp": int(book.get("timestamp") or stamp),
                    "market": condition,
                    "asset_id": token,
                    "bids": _level_rows(book.get("bids")),
                    "asks": _level_rows(book.get("asks")),
                    "price": None,
                    "size": None,
                    "side": None,
                }
            )
    return rows, failures


def write_mirror(rows: Sequence[Mapping[str, Any]], out_dir: Path) -> Path:
    """Write one Parquet file into the mirror, named by its own instant."""
    import pyarrow.parquet as pq

    if not rows:
        raise FetchError("nothing to write: no books were fetched")
    table = book_archive_table(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(min(int(row["timestamp"]) for row in rows))
    path = out_dir / f"books-{stamp}.parquet"
    pq.write_table(table, path)
    return path


# ---------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None, transport: Optional[Any] = None) -> int:
    parser = argparse.ArgumentParser(prog="fetch_polymarket.py", description=__doc__)
    # Shared transport flags live on a parent so they work *after* the
    # subcommand, which is where anyone will type them.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache", default="data/cache/polymarket-http",
                        help="response cache; re-runs are free. Empty string disables.")
    common.add_argument("--timeout", type=float, default=30.0)
    common.add_argument("--min-interval", type=float, default=0.2,
                        help="minimum seconds between successful requests")
    common.add_argument("--retry-delay", type=float, default=0.5,
                        help="first wait after a 429 or 5xx; grows by --backoff")
    common.add_argument("--backoff", type=float, default=1.5)
    sub = parser.add_subparsers(dest="command", required=True)

    markets = sub.add_parser("markets", parents=[common],
                             help="market definitions into a specs JSON")
    markets.add_argument("--out", required=True)
    markets.add_argument("--limit", type=int, default=200)
    markets.add_argument("--page-size", type=int, default=100)
    markets.add_argument("--max-pages", type=int, default=50)
    state = markets.add_mutually_exclusive_group()
    state.add_argument("--closed", dest="closed", action="store_true", default=None,
                       help="resolved markets only, which is what a study needs")
    state.add_argument("--open", dest="closed", action="store_false",
                       help="live markets only")

    books = sub.add_parser("books", parents=[common],
                           help="current books into a Parquet mirror")
    books.add_argument("--specs", required=True, help="specs JSON from the markets command")
    books.add_argument("--out", required=True, help="mirror directory")
    books.add_argument("--max-markets", type=int, default=25,
                       help="bound the crawl; one request per token")

    args = parser.parse_args(argv)
    if transport is None:
        transport = HttpTransport(
            cache_dir=Path(args.cache) if args.cache else None,
            timeout=args.timeout,
            min_interval=args.min_interval,
            retry_delay=args.retry_delay,
            backoff=args.backoff,
        )

    if args.command == "markets":
        records = fetch_markets(
            transport,
            closed=args.closed,
            limit=args.limit,
            page_size=args.page_size,
            max_pages=args.max_pages,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(records, indent=2), encoding="utf-8")
        resolved = sum(1 for record in records if record.get("closed"))
        print(
            json.dumps(
                {
                    "markets": len(records),
                    "closed": resolved,
                    "out": args.out,
                    "requests": getattr(transport, "requests_made", None),
                    "cache_hits": getattr(transport, "cache_hits", None),
                },
                indent=2,
            )
        )
        return 0 if records else 4

    if args.command == "books":
        records = json.loads(Path(args.specs).read_text(encoding="utf-8"))
        if not isinstance(records, list):
            parser.error(f"{args.specs} does not contain a list of markets")
        selected = records[: args.max_markets]
        rows, failures = collect_books(transport, selected)
        if not rows:
            print(json.dumps({"markets": len(selected), "failures": failures}, indent=2),
                  file=sys.stderr)
            return 4
        path = write_mirror(rows, Path(args.out))
        print(
            json.dumps(
                {
                    "markets": len(selected),
                    "book_rows": len(rows),
                    "failures": failures,
                    "wrote": str(path),
                    "requests": getattr(transport, "requests_made", None),
                    "next": [
                        f"python -m h5i_db.venues markets market.db {args.specs}",
                        f"python -m h5i_db.venues ingest market.db {args.specs} "
                        f"--root {args.out}",
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
