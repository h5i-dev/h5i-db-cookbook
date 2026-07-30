# %% [markdown]
# # From a vendor mirror to canonical tables
#
# Most prediction-market research starts with a directory of vendor Parquet
# already on disk: hourly full-feed archives, or per-day channel files. This
# recipe turns that directory into the tables a replay reads, and spends most of
# its length on the four things that decide whether the result is trustworthy:
# which outcome a row belongs to, what happens when you run the import twice,
# how much of the window you actually got, and what the importer does with a row
# it does not recognise.
#
# Nothing here fetches. Downloading belongs in a script where credentials and
# rate limits belong; `h5i_db.venues` is the part that must be reproducible.

# %% [markdown]
# ## Terms used here
#
# | term              | meaning |
# | ----------------- | --- |
# | vendor mirror     | a directory of vendor files already on disk, which is where most research starts |
# | canonical tables  | the normalized shape a replay reads, whatever the vendor's layout was |
# | market spec       | the definition of a market: its outcomes, tick size, and expiry |
# | layout            | which vendor columns mean what, declared as data rather than hard-coded |
# | content-addressed | keyed by the file's hash, so running the import twice replays rather than duplicates |
# | coverage          | how much of the intended window the ingest actually got |
# | quarantine        | what the importer does with a row it does not recognise, instead of dropping it |
#
# New to any of these? [GLOSSARY.md](../../GLOSSARY.md) defines them at more
# length, along with every other term the cookbook uses.

# %%
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import cookbook_utils as cu
import h5i_db
from h5i_db import venues

db = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp"), create=True)

# %% [markdown]
# ## A stand-in mirror
#
# `cu.write_polymarket_archive` de-normalises a synthetic panel into the shape
# the public hourly archives publish. It exists so this recipe runs offline; on
# your own machine, point the same code at your real mirror and delete this cell.
#
# One archive row is one event. A book state carries its levels nested; an
# incremental change or a print carries flat `price`/`size`/`side`.
#
# | column | type | meaning |
# |---|---|---|
# | `event_type` | `string` | `book`, `price_change`, `last_trade_price`, and others |
# | `timestamp` | `int64` | event time, **milliseconds** |
# | `market` | `string` | the condition id |
# | `asset_id` | `string` | the per-outcome token; this is the join key |
# | `bids` / `asks` | `list<struct<price, size>>` | levels of a book state |
# | `price` / `size` / `side` | `float64` / `float64` / `string` | one level, or one print |

# %%
panel = cu.make_prediction_markets(n_markets=40, steps=24, seed=11)
mirror = Path("data/cache/onramp-mirror")
files = cu.write_polymarket_archive(panel, mirror)
print(f"{len(files)} hourly files under {mirror}")
sample = pq.read_table(files[0])
print(f"{sample.num_rows:,} rows x {sample.num_columns} columns in {files[0].name}")
sample.to_pandas().head()

# %% [markdown]
# ## Step one: what the markets are
#
# A market spec is the identity of one tradeable event: the outcomes, the vendor
# token for each, when trading stops, and when the result became knowable. Get
# the outcome order wrong and every fill is attributed to the wrong side, so this
# is the step to be pedantic about.
#
# `polymarket_markets_from_json` reads the payloads a public market endpoint
# returns, including the awkward parts: the list fields arrive as JSON-encoded
# strings, and a resolution is expressed as settled prices plus a closed flag.

# %%
payloads = cu.polymarket_market_payloads(panel)
print(json.dumps(payloads[0], indent=2)[:420], "...")

specs = venues.polymarket_markets_from_json(payloads)
print(f"\n{len(specs)} markets parsed")
first = specs[0]
print(f"  {first.instrument_id}: outcomes {first.outcome_labels}")
print(f"  tokens              {first.tokens}")
print(f"  token -> outcome    {first.tokens[1]} is outcome {first.outcome_of_token(first.tokens[1])}")
print(f"  resolved            {first.is_resolved}, winner {first.winner_outcome}")

# %% [markdown]
# `outcome_labels` and `tokens` are positional: index `i` of each describes the
# same outcome. That is the entire contract, and the spec refuses every way of
# breaking it rather than resolving the ambiguity.

# %%
for description, build in (
    ("one outcome", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes",))),
    ("tokens and outcomes disagree", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes", "No"), tokens=("a",))),
    ("resolved with no resolution time", lambda: venues.MarketSpec(
        instrument_id="m", venue="v", outcome_labels=("Yes", "No"), winner_outcome=0)),
):
    try:
        build()
    except ValueError as error:
        print(f"{description:32} refused: {str(error).split(';')[0][:78]}")

# A token two markets both claim would make every row keyed by it ambiguous.
try:
    venues.token_index([
        venues.MarketSpec(instrument_id="a", venue="v",
                          outcome_labels=("Yes", "No"), tokens=("t1", "t2")),
        venues.MarketSpec(instrument_id="b", venue="v",
                          outcome_labels=("Yes", "No"), tokens=("t1", "t3")),
    ])
except ValueError as error:
    print(f"{'token claimed twice':32} refused: {error}")

# %% [markdown]
# ## Step two: write the definitions
#
# `instruments` gets one row per outcome; `resolutions` gets one row per resolved
# market, dated by the instant the result became **observable** rather than when
# the event happened. That column is what settlement is gated on later.

# %%
report = venues.write_markets(db, specs, note="market definitions")
print(report)
print(f"\ninstruments rows: {report.tables['instruments'].rows}")
print(f"resolutions rows: {report.tables['resolutions'].rows}")
db.sql("SELECT * FROM instruments ORDER BY instrument_id, outcome LIMIT 4").to_pandas()

# %% [markdown]
# ## Step three: ingest the archive
#
# Rows are filtered to the tokens of the markets you asked for, so a full-feed
# hour costs only the markets you care about. The window is a half-open
# `[start, end)` in epoch nanoseconds and bounds what is read.

# %%
expiry = int(db.sql("SELECT max(expiration_ns) AS e FROM instruments").to_pandas()["e"][0])
start = int(pd.Timestamp("2026-05-01T12:00:00Z").value)
ingest = venues.ingest_archive(
    db,
    files=venues.discover(mirror),
    markets=specs,
    layout=venues.PMXT_LAYOUT,
    window=(start, expiry + 1),
    note="mirror backfill",
)
print(ingest)
print("rows by table: ", {name: write.rows for name, write in ingest.tables.items()})
print(f"files read:     {len(ingest.sources)}")
print(f"coverage:      {ingest.coverage:.4f}")

# %% [markdown]
# The `book` events became grouped snapshots and the prints became trades. Note
# what the outcome column proves: each event describes one outcome of one
# instrument, which is the invariant the engine refuses to violate.

# %%
grouping = db.sql(
    """
    SELECT count(*) AS events,
           min(rows_per_event) AS min_rows, max(rows_per_event) AS max_rows,
           sum(CASE WHEN outcomes > 1 THEN 1 ELSE 0 END) AS events_mixing_outcomes,
           sum(CASE WHEN terminators <> 1 THEN 1 ELSE 0 END) AS badly_terminated
    FROM (
        SELECT event_index, count(*) AS rows_per_event,
               count(DISTINCT outcome) AS outcomes,
               sum(CASE WHEN is_last THEN 1 ELSE 0 END) AS terminators
        FROM book_deltas GROUP BY event_index
    )
    """
).to_pandas()
print(grouping.to_string(index=False))
db.sql(
    "SELECT * FROM book_deltas ORDER BY ts_init, event_index, is_last LIMIT 4"
).to_pandas()

# %% [markdown]
# ## Run it again
#
# Every commit is keyed by the hash of the rows it carries, so identical inputs
# produce identical keys and h5i-db recognises them. An interrupted backfill is
# safe to restart, and two sources serving the same hour converge on one commit
# instead of two.

# %%
before = db.sql("SELECT count(*) AS n FROM book_deltas").to_pandas()["n"][0]
versions_before = len(db.versions("book_deltas"))
again = venues.ingest_archive(
    db,
    files=venues.discover(mirror),
    markets=specs,
    layout=venues.PMXT_LAYOUT,
    window=(start, expiry + 1),
)
after = db.sql("SELECT count(*) AS n FROM book_deltas").to_pandas()["n"][0]
print(f"first  pass replayed: {ingest.replayed}")
print(f"second pass replayed: {again.replayed}")
print(f"rows    {before:,} -> {after:,}")
print(f"versions {versions_before} -> {len(db.versions('book_deltas'))}")
assert after == before

# %% [markdown]
# ## Coverage is a reported fact, not an assumption
#
# `requested_window` and `loaded_window` stay separate. Asking for a week and
# getting six hours is a normal thing to discover, and the point is to discover
# it from the result rather than from a strange backtest three steps later.
#
# `coverage` is deliberately `None` when no window was requested: a ratio against
# an unbounded request would mean nothing.

# %%
week = start + 7 * 24 * 3_600 * 1_000_000_000
probe = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_probe"), create=True)
short = venues.ingest_archive(
    probe, files=venues.discover(mirror), markets=specs,
    layout=venues.PMXT_LAYOUT, window=(start, week),
)
print(f"asked for {(week - start) / 3.6e12:.1f} hours")
print(f"loaded    {(short.loaded_window[1] - short.loaded_window[0]) / 3.6e12:.1f} hours")
print(f"coverage  {short.coverage:.4f}")

unbounded = venues.ingest_archive(
    probe, files=venues.discover(mirror), markets=specs, layout=venues.PMXT_LAYOUT
)
print(f"coverage with no window requested: {unbounded.coverage}")
probe.close()

# %% [markdown]
# ## What the importer will not guess
#
# Two failure modes that a permissive loader hides. A file missing the columns
# the layout needs is skipped with its name and the missing columns recorded. An
# event type present in the data but absent from the layout is counted, because
# silently dropping the update that mattered is how a book goes wrong quietly.

# %%
broken = Path("data/cache/onramp-broken")
broken.mkdir(parents=True, exist_ok=True)
import pyarrow as pa

pq.write_table(pa.table({"nonsense": pa.array([1, 2, 3])}), broken / "bad.parquet")
# A layout that does not know about prints leaves them unrecognised.
book_only = venues.ArchiveLayout(
    name="book-only",
    timestamp_unit="ms",
    instrument_column="market",
    snapshot_events=("book",),
)
strict = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_strict"), create=True)
picky = venues.ingest_archive(
    strict,
    files=[broken / "bad.parquet", *venues.discover(mirror)],
    markets=specs,
    layout=book_only,
)
for item in picky.skipped:
    print(json.dumps(item)[:150])
strict.close()

# %% [markdown]
# ## A vendor dialect is data
#
# `ArchiveLayout` carries the column names, the event vocabulary, the timestamp
# unit and the level shape. The two shipped layouts are literals of that type, so
# a third vendor is another literal rather than another module. Here is a
# house-format feed ingested with no new code.

# %%
print("PMXT_LAYOUT:")
for field in ("timestamp_column", "timestamp_unit", "token_column", "snapshot_events",
              "delta_events", "trade_events"):
    print(f"  {field:18} {getattr(venues.PMXT_LAYOUT, field)!r}")
print("\nTELONEX_LAYOUT differs only in these:")
for field in ("timestamp_unit", "event_type_column", "snapshot_events"):
    print(f"  {field:18} {getattr(venues.TELONEX_LAYOUT, field)!r}")

# %%
house = Path("data/cache/onramp-house")
house.mkdir(parents=True, exist_ok=True)
level = pa.struct([("px", pa.float64()), ("qty", pa.float64())])
token = specs[0].tokens[0]
pq.write_table(
    pa.table({
        "channel": pa.array(["depth"], pa.string()),
        "recv_ns": pa.array([start], pa.int64()),
        "token": pa.array([token], pa.string()),
        "buys": pa.array([[{"px": 0.31, "qty": 40.0}]], pa.list_(level)),
        "sells": pa.array([[{"px": 0.33, "qty": 35.0}]], pa.list_(level)),
    }),
    house / "day.parquet",
)
house_layout = venues.ArchiveLayout(
    name="house-feed",
    timestamp_column="recv_ns",
    timestamp_unit="ns",
    token_column="token",
    event_type_column="channel",
    snapshot_events=("depth",),
    levels=venues.LevelLayout(
        style="nested", bids_column="buys", asks_column="sells",
        price_field="px", size_field="qty",
    ),
)
other = h5i_db.Database(cu.fresh_db("05_vendor_data_onramp_house"), create=True)
house_report = venues.ingest_archive(
    other, files=[house / "day.parquet"], markets=specs, layout=house_layout
)
print(f"vendor: {house_report.vendor}, rows: {house_report.rows}")
print(other.sql("SELECT outcome, side, price, size FROM book_deltas").to_pandas().to_string(index=False))
other.close()

# %% [markdown]
# ## The same three steps from a shell
#
# Market definitions travel as a JSON file rather than as flags, because a market
# is a dozen fields and a flag list would be neither readable nor versionable.
# `--min-coverage` exits non-zero rather than letting a short load pass quietly,
# which is what makes this usable in a scheduled backfill.

# %%
spec_path = Path("data/cache/onramp-specs.json")
spec_path.write_text(json.dumps(payloads), encoding="utf-8")
print(f"""
python -m h5i_db.venues markets  market.db {spec_path}
python -m h5i_db.venues ingest   market.db {spec_path} --root {mirror} \\
    --start-ns {start} --end-ns {expiry + 1} --min-coverage 0.95
python -m h5i_db.venues inspect  market.db
""".strip())

# %%
from h5i_db.venues.__main__ import main as venues_cli

cli_db = cu.fresh_db("05_vendor_data_onramp_cli")
assert venues_cli(["markets", cli_db, str(spec_path)]) == 0
assert venues_cli(["ingest", cli_db, str(spec_path), "--root", str(mirror)]) == 0
assert venues_cli(["inspect", cli_db]) == 0
# The gate fires when the window is wider than the data.
code = venues_cli([
    "ingest", cli_db, str(spec_path), "--root", str(mirror),
    "--start-ns", str(start), "--end-ns", str(week), "--min-coverage", "0.95",
])
print(f"\nexit code when coverage falls short: {code}")

# %% [markdown]
# ## The tables are now replayable
#
# Which is the whole point. A snapshot pins what was ingested, and the run reads
# it through the same path as any other h5i-db data.

# %%
db.snapshot("mirror-v1", tables=["instruments", "book_deltas", "trades", "resolutions"],
            note="ingested from the vendor mirror")
from h5i_db import backtest

# One order, timed a microsecond after a quote instant so it transacts at the
# price it was decided from. Recipe 05/07 builds real strategies on this data.
quotes = db.sql(
    f"""
    SELECT instrument_id, ts_init,
           max(CASE WHEN side = 'sell' THEN price END) AS ask
    FROM h5i('book_deltas', 'mirror-v1')
    WHERE outcome = 0
    GROUP BY instrument_id, ts_init
    ORDER BY ts_init, instrument_id
    LIMIT 200
    """
).to_pandas()
decision = quotes.iloc[len(quotes) // 2]
backtest.create_signal_table(db, "signals")
db.append("signals", backtest.signal_table([{
    "ts": decision.ts_init.to_pydatetime() + pd.Timedelta(microseconds=1).to_pytimedelta(),
    "instrument_id": decision.instrument_id, "outcome": 0,
    "side": "buy", "quantity": 10.0, "tag": "onramp",
}]))

config = backtest.BacktestConfig(
    run_id="onramp-probe",
    data=backtest.DataConfig(signals="signals", snapshot="mirror-v1"),
    portfolio=backtest.PortfolioConfig(starting_cash=10_000.0),
    execution=backtest.ExecutionConfig(fee_kind="kalshi", fee_rate=0.07),
)
inspection = backtest.inspect(db, config)
print(f"replay fidelity: {inspection.fidelity}")
print(f"config accepted: {inspection.ok}")
for name, stats in sorted(inspection.tables.items()):
    print(f"  {name:12} {stats['row_count']:>7,} rows")

result = backtest.execute(db, config)
fill = result.fills.to_pandas()
print(f"\nfilled {fill.quantity.iloc[0]:.0f} at {fill.price.iloc[0]:.3f}, "
      f"decision ask was {decision.ask:.3f}")
assert fill.price.iloc[0] == decision.ask

# %% [markdown]
# ## Takeaways
#
# - Ingest is three steps: parse the market payloads, write `instruments` and
#   `resolutions`, then normalise the archive into `book_deltas` and `trades`.
#   Recipe 05/07 picks up from the snapshot this leaves behind.
# - Outcome identity is positional and every way of breaking it is refused.
#   A token two markets claim, a resolution with no observability instant, a
#   token list that does not match the outcome list: all errors, because the
#   silent versions of those mistakes are unrecoverable later.
# - Re-running an import is a replay. Commits are keyed by the hash of the
#   normalised rows, so a restarted backfill adds nothing and two sources for
#   one hour converge.
# - `coverage` compares the requested window with the loaded one and is `None`
#   when nothing was requested. Unknown event types and unusable files are
#   counted in `report.skipped`, never dropped quietly.
# - A vendor dialect is an `ArchiveLayout` literal, so a third vendor needs no
#   new code. This recipe ingested a house format to prove it.
# - h5i-db features doing the work: content-addressed idempotency keys turning
#   a re-import into a replay, a named snapshot pinning what was ingested, and
#   `versions()` showing that the second pass wrote nothing.

# %%
db.close()
