"""Offline tests for the Polymarket fetcher.

The transport is injected, so everything except the socket call is exercised
here: pagination and its stop conditions, retry and backoff, the response cache,
shape validation, and whether the Parquet mirror it writes is actually readable
by `h5i_db.venues.ingest_archive`. That last test is the one that matters, since
a fetcher whose output the importer cannot read is a fetcher that does not work.

Run with:  python -m pytest scripts/test_fetch_polymarket.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_polymarket as fp  # noqa: E402

YES = "111111111111111111"
NO = "222222222222222222"
CONDITION = "0xcondition"


class Recorded:
    """A transport that replays canned responses and records what was asked."""

    def __init__(self, responses):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get_json(self, url, params=None):
        self.calls.append((url, dict(params or {})))
        for matcher, payload in self.responses:
            if matcher(url, params or {}):
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"no canned response for {url} {params}")


def _market(index: int = 0, **overrides) -> dict:
    record = {
        "condition_id": f"{CONDITION}{index}",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": json.dumps([f"{YES}{index}", f"{NO}{index}"]),
        "outcomePrices": '["1", "0"]',
        "closed": True,
        "umaResolutionTime": "2026-03-10T00:00:00Z",
        "endDate": "2026-03-09T00:00:00Z",
    }
    record.update(overrides)
    return record


def _book(best_bid: float = 0.40, best_ask: float = 0.42) -> dict:
    return {
        "bids": [{"price": str(best_bid), "size": "120"}],
        "asks": [{"price": str(best_ask), "size": "90"}],
        "timestamp": 1_773_000_000_000,
    }


# ------------------------------------------------------------- pagination


def test_pagination_walks_offsets_and_stops_on_a_short_page():
    pages = {0: [_market(i) for i in range(3)], 3: [_market(3)]}
    transport = Recorded([
        (lambda url, params: url.endswith("/markets"),
         None),
    ])

    def get_json(url, params=None):
        transport.calls.append((url, dict(params or {})))
        return pages.get(params["offset"], [])

    transport.get_json = get_json  # type: ignore[method-assign]
    collected = list(fp.market_pages(transport, page_size=3, closed=True))
    assert [len(page) for page in collected] == [3, 1]
    # A short page ends the walk; offset 6 is never requested.
    assert [call[1]["offset"] for call in transport.calls] == [0, 3]
    assert transport.calls[0][1]["closed"] == "true"


def test_pagination_stops_on_an_empty_page_and_respects_max_pages():
    def empty(url, params=None):
        return []

    transport = Recorded([])
    transport.get_json = empty  # type: ignore[method-assign]
    assert list(fp.market_pages(transport, page_size=5)) == []

    def always_full(url, params=None):
        return [_market(i) for i in range(5)]

    transport.get_json = always_full  # type: ignore[method-assign]
    pages = list(fp.market_pages(transport, page_size=5, max_pages=2))
    assert len(pages) == 2, "max_pages must bound a crawl of someone else's API"

    with pytest.raises(ValueError):
        list(fp.market_pages(transport, page_size=0))


def test_fetch_markets_honours_the_limit_and_drops_untokened_markets():
    records = [_market(i) for i in range(4)] + [
        _market(9, clobTokenIds=None),
        _market(10, clobTokenIds=""),
    ]

    def get_json(url, params=None):
        return records if params["offset"] == 0 else []

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    kept = fp.fetch_markets(transport, page_size=10, limit=100)
    assert len(kept) == 4, "a market with no CLOB token is not ingestible"
    capped = fp.fetch_markets(transport, page_size=10, limit=2)
    assert len(capped) == 2


# ------------------------------------------------------- shape validation


def test_a_wrapped_list_is_unwrapped_and_anything_else_is_named():
    transport = Recorded([(lambda u, p: True, {"data": [_market(0)]})])
    assert len(list(fp.market_pages(transport, page_size=5))[0]) == 1

    transport = Recorded([(lambda u, p: True, {"unexpected": 1, "shape": 2})])
    with pytest.raises(fp.FetchError, match=r"keys \['shape', 'unexpected'\]"):
        list(fp.market_pages(transport, page_size=5))

    transport = Recorded([(lambda u, p: True, "a string")])
    with pytest.raises(fp.FetchError, match="expected a list, got str"):
        list(fp.market_pages(transport, page_size=5))


def test_a_missing_field_names_what_arrived_instead():
    transport = Recorded([(lambda u, p: True, [{"question": "no outcomes here"}])])
    with pytest.raises(fp.FetchError, match="missing \\['outcomes'\\]"):
        fp.fetch_markets(transport, page_size=5)

    transport = Recorded([(lambda u, p: True, {"bids": [], "no_asks": []})])
    with pytest.raises(fp.FetchError, match="missing \\['asks'\\]"):
        fp.fetch_book(transport, YES)

    transport = Recorded([(lambda u, p: True, [{"t": 1}])])
    with pytest.raises(fp.FetchError, match="missing \\['p'\\]"):
        fp.fetch_price_history(transport, YES)


def test_a_closed_market_book_404_explains_the_capability_limit():
    """The live endpoint serves books for open markets only."""

    def get_json(url, params=None):
        raise fp.FetchError(f"{url} returned HTTP 404")

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    with pytest.raises(fp.FetchError, match="live markets only"):
        fp.fetch_book(transport, YES)

    # Any other failure passes through unchanged rather than being relabelled.
    def other(url, params=None):
        raise fp.FetchError(f"{url} returned HTTP 500")

    transport.get_json = other  # type: ignore[method-assign]
    with pytest.raises(fp.FetchError, match="HTTP 500"):
        fp.fetch_book(transport, YES)


def test_token_ids_reads_both_spellings_and_refuses_neither():
    assert fp.token_ids({"clobTokenIds": [YES, NO]}) == [YES, NO]
    assert fp.token_ids({"clobTokenIds": json.dumps([YES, NO])}) == [YES, NO]
    with pytest.raises(fp.FetchError, match="not JSON"):
        fp.token_ids({"clobTokenIds": "not-json-at-all["})
    with pytest.raises(fp.FetchError, match="no usable tokens"):
        fp.token_ids({"condition_id": CONDITION})


# -------------------------------------------------------- retry and cache


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://x", code, "boom", {}, None)  # type: ignore[arg-type]


def test_retry_backs_off_on_a_rate_limit_then_succeeds(monkeypatch):
    slept: list[float] = []
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _http_error(429)

        class Response:
            def read(self):
                return json.dumps([_market(0)]).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response()

    monkeypatch.setattr(fp.urllib.request, "urlopen", fake_urlopen)
    transport = fp.HttpTransport(
        cache_dir=None, sleep=slept.append, min_interval=0.0, retry_delay=0.25
    )
    payload = transport.get_json(f"{fp.GAMMA_URL}/markets", {"limit": 1})
    assert isinstance(payload, list) and attempts["n"] == 3
    # Two failures, two waits, and the second is longer than the first. Backoff
    # is independent of pacing, so min_interval=0 must not flatten it.
    assert len(slept) == 2 and slept[1] > slept[0]
    assert slept[0] == pytest.approx(0.25)

    # And a transport cannot be built with a backoff that would hot-loop.
    with pytest.raises(ValueError, match="hot loop"):
        fp.HttpTransport(retry_delay=0.0)
    with pytest.raises(ValueError, match="hot loop"):
        fp.HttpTransport(backoff=0.5)


def test_a_permanent_status_is_not_retried_and_403_explains_itself(monkeypatch):
    attempts = {"n": 0}

    def fake_urlopen(request, timeout=None):
        attempts["n"] += 1
        raise _http_error(403)

    monkeypatch.setattr(fp.urllib.request, "urlopen", fake_urlopen)
    transport = fp.HttpTransport(cache_dir=None, sleep=lambda _: None, min_interval=0.0, retry_delay=0.01)
    with pytest.raises(fp.FetchError, match="the network, not the URL"):
        transport.get_json(f"{fp.GAMMA_URL}/markets")
    assert attempts["n"] == 1, "a 403 is not a transient failure"


def test_the_cache_makes_a_rerun_free(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1

        class Response:
            def read(self):
                return json.dumps({"bids": [], "asks": []}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response()

    monkeypatch.setattr(fp.urllib.request, "urlopen", fake_urlopen)
    transport = fp.HttpTransport(cache_dir=tmp_path, sleep=lambda _: None, min_interval=0.0, retry_delay=0.01)
    first = transport.get_json(f"{fp.CLOB_URL}/book", {"token_id": YES})
    second = transport.get_json(f"{fp.CLOB_URL}/book", {"token_id": YES})
    assert first == second
    assert calls["n"] == 1, "the second read must come from disk"
    assert transport.cache_hits == 1
    # A different token is a different key.
    transport.get_json(f"{fp.CLOB_URL}/book", {"token_id": NO})
    assert calls["n"] == 2


# ------------------------------------------------------------ collecting


def test_one_bad_token_does_not_lose_the_whole_mirror():
    def get_json(url, params=None):
        if params.get("token_id", "").startswith(NO):
            raise fp.FetchError("delisted")
        return _book()

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    rows, failures = collect = fp.collect_books(
        transport, [_market(0), _market(1)], now_ms=1_773_000_000_000
    )
    assert len(rows) == 2, "the YES side of both markets survived"
    assert len(failures) == 2 and all("delisted" in f["reason"] for f in failures)
    # And a caller who wants it strict can have that.
    with pytest.raises(fp.FetchError):
        fp.collect_books(transport, [_market(0)], on_error="raise")


def test_price_history_coerces_points_and_skips_junk():
    payload = {"history": [{"t": 1, "p": "0.5"}, "junk", {"t": 2, "p": 0.6}]}
    transport = Recorded([(lambda u, p: True, payload)])
    points = fp.fetch_price_history(transport, YES)
    assert points == [{"t": 1, "p": 0.5}, {"t": 2, "p": 0.6}]


# ------------------------------------------- the output the importer reads


def test_the_mirror_it_writes_is_ingestible_by_venues():
    """The test that decides whether this script works at all."""
    import h5i_db
    from h5i_db import venues

    def get_json(url, params=None):
        return _book(0.40, 0.42)

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    markets = [_market(0)]
    rows, failures = fp.collect_books(transport, markets, now_ms=1_773_000_000_000)
    assert not failures

    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp) / "mirror"
        written = fp.write_mirror(rows, mirror)
        assert written.exists()

        specs = venues.polymarket_markets_from_json(markets)
        db = h5i_db.Database(str(Path(tmp) / "market.db"), create=True)
        try:
            venues.write_markets(db, specs)
            report = venues.ingest_archive(
                db,
                files=venues.discover(mirror),
                markets=specs,
                layout=venues.PMXT_LAYOUT,
            )
            assert report.tables["book_deltas"].rows == 4  # 2 tokens x bid + ask
            book = db.sql(
                "SELECT outcome, side, price FROM book_deltas ORDER BY outcome, side"
            ).to_pandas()
            # Outcome attribution survived the round trip: the YES token's book
            # landed on outcome 0 with the prices the endpoint returned.
            assert list(book.outcome) == [0, 0, 1, 1]
            assert sorted(book[book.outcome == 0].price) == [0.40, 0.42]
            # And every event is one atomic snapshot of one outcome.
            grouped = db.sql(
                """
                SELECT count(*) AS bad FROM (
                    SELECT event_index FROM book_deltas GROUP BY event_index
                    HAVING count(DISTINCT outcome) > 1
                        OR sum(CASE WHEN is_last THEN 1 ELSE 0 END) <> 1)
                """
            ).to_pandas()
            assert int(grouped.bad.iloc[0]) == 0
        finally:
            db.close()


def test_writing_nothing_is_an_error_not_an_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(fp.FetchError, match="no books were fetched"):
            fp.write_mirror([], Path(tmp) / "mirror")


# -------------------------------------------------------------------- CLI


def test_cli_markets_writes_specs_and_reports_counts(capsys):
    def get_json(url, params=None):
        return [_market(i) for i in range(2)] if params["offset"] == 0 else []

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "specs.json"
        code = fp.main(["markets", "--out", str(out), "--page-size", "10"],
                       transport=transport)
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["markets"] == 2 and payload["closed"] == 2
        # The file it wrote is what the venues CLI consumes.
        from h5i_db import venues

        specs = venues.polymarket_markets_from_json(
            json.loads(out.read_text(encoding="utf-8"))
        )
        assert len(specs) == 2 and specs[0].is_resolved


def test_cli_books_writes_a_mirror_and_prints_the_next_commands(capsys):
    def get_json(url, params=None):
        return _book()

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory() as tmp:
        specs = Path(tmp) / "specs.json"
        specs.write_text(json.dumps([_market(0)]), encoding="utf-8")
        mirror = Path(tmp) / "mirror"
        code = fp.main(
            ["books", "--specs", str(specs), "--out", str(mirror)], transport=transport
        )
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["book_rows"] == 2 and payload["failures"] == []
        assert any("venues ingest" in line for line in payload["next"])
        assert list(mirror.glob("*.parquet"))


def test_cli_returns_a_distinct_code_when_nothing_came_back(capsys):
    def get_json(url, params=None):
        return []

    transport = Recorded([])
    transport.get_json = get_json  # type: ignore[method-assign]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "specs.json"
        code = fp.main(["markets", "--out", str(out)], transport=transport)
        assert code == 4, "an empty result is not a success"
