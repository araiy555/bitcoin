"""Binance's published daily archives, turned into a replayable recording.

Recording live was the bottleneck and it distorted the answer. An hour of
USUSDT took an hour to collect, and the two hours we managed were so unlike
each other that the same settings were struck 2,314 times in one and zero
times in the next. Choosing a parameter on one hour and testing it on one
other hour is not a test, it is a coin toss with extra steps.

Binance publishes the same data daily, for free, going back years:

    https://data.binance.vision/data/futures/um/daily/bookTicker/<SYM>/<SYM>-bookTicker-<YYYY-MM-DD>.zip
    https://data.binance.vision/data/futures/um/daily/aggTrades/<SYM>/<SYM>-aggTrades-<YYYY-MM-DD>.zip

so a month of quiet days and violent days can be replayed in minutes.

**What the archive is not.** It carries the best bid and ask each time either
changes, and every aggregated print, but not the full depth diff stream. The
rebuilt book is therefore one level deep. For a strategy that rests far from
the touch and is filled when price sweeps out to it, that is the part that
matters — the tape either reaches the price or it does not. For anything that
depends on queue position at the touch, it is not enough, and the missing
depth would flatter the result rather than hurt it.

**Column names over positions.** The archives sometimes carry a header row and
sometimes do not, and the column order has changed before. Reading by name
when a header exists, and only then falling back to the documented order, is
the difference between a wrong answer and a loud one.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta

from ..core.types import Instrument, Side
from ..feed.base import DepthSnapshot, TradeTick

BASE = "https://data.binance.vision/data/futures/um/daily"

BOOK_TICKER_COLUMNS = (
    "update_id",
    "best_bid_price",
    "best_bid_qty",
    "best_ask_price",
    "best_ask_qty",
    "transaction_time",
    "event_time",
)
AGG_TRADE_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)


def daily_url(datatype: str, symbol: str, day: date) -> str:
    sym = symbol.upper()
    return f"{BASE}/{datatype}/{sym}/{sym}-{datatype}-{day:%Y-%m-%d}.zip"


def days_between(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("the end date precedes the start date")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def read_zip_csv(blob: bytes, columns: tuple[str, ...]) -> Iterator[dict[str, str]]:
    """Rows as dicts, whether or not the archive carried a header.

    A file whose first row is data is read positionally against `columns`; a
    file with a header is read by name, so a reordered archive cannot quietly
    swap price for quantity.
    """
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError("the archive holds no csv")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            first = text.readline()
            if not first:
                return
            cells = next(csv.reader([first]))
            # The first column is an id in every archive we read, so a
            # non-numeric value there is a header. Looking for letters anywhere
            # in the row instead would call the first data row a header on any
            # file carrying a true/false flag, and silently drop it.
            has_header = not _is_number(cells[0] if cells else "")
            if has_header:
                header = [c.strip() for c in cells]
                yield from csv.DictReader(text, fieldnames=header)
            else:
                yield dict(zip(columns, cells, strict=False))
                for row in csv.reader(text):
                    if row:
                        yield dict(zip(columns, row, strict=False))


def _is_number(cell: str) -> bool:
    try:
        float(cell.strip())
    except ValueError:
        return False
    return True


def _pick(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    raise KeyError(f"none of {names} in {sorted(row)}")


def _ts_ns(raw: str) -> int:
    """Archive timestamps are milliseconds; some symbols carry microseconds."""
    value = int(float(raw))
    # 1e15 ms is the year 33658, so anything above it is a finer unit.
    return value * 1_000_000 if value < 1_000_000_000_000_000 else value * 1_000


def ts_ns_auto(raw: str) -> int:
    """A timestamp in whatever unit the venue chose, as nanoseconds.

    Bybit stamps seconds with a fractional part, Binance milliseconds, and
    some files microseconds. A wrong guess does not raise — it puts the day
    in 1970 or in the year 33658, and a merge of two venues then interleaves
    one venue's whole day before the other's first print, which reads as a
    spread of several percent that closes instantly.

    The magnitudes are four orders of magnitude apart, so the current era
    separates them unambiguously.
    """
    value = float(raw)
    if value < 1e11:  # seconds until the year 5138
        return int(value * 1e9)
    if value < 1e14:  # milliseconds
        return int(value * 1e6)
    if value < 1e17:  # microseconds
        return int(value * 1e3)
    return int(value)


@dataclass(frozen=True, slots=True)
class Stamped:
    ts_ns: int
    event: object


def book_events(rows: Iterator[dict[str, str]], instrument: Instrument) -> Iterator[Stamped]:
    """One top-of-book image per change.

    Emitted as a snapshot rather than a delta: a one-level book has no history
    to reconcile, and a delta stream implies update-id continuity the archive
    does not provide.
    """
    for row in rows:
        try:
            bid = instrument.to_ticks(_pick(row, "best_bid_price", "b"))
            ask = instrument.to_ticks(_pick(row, "best_ask_price", "a"))
            bid_qty = instrument.to_lots(_pick(row, "best_bid_qty", "B"))
            ask_qty = instrument.to_lots(_pick(row, "best_ask_qty", "A"))
            ts = _ts_ns(_pick(row, "transaction_time", "event_time", "T", "E"))
            update_id = int(float(_pick(row, "update_id", "u")))
        except (KeyError, ValueError, ArithmeticError):
            # Decimal raises InvalidOperation, not ValueError, on a malformed
            # price, so catching ValueError alone lets one bad row end the day.
            continue
        if bid <= 0 or ask <= 0 or ask <= bid:
            continue
        yield Stamped(
            ts,
            DepthSnapshot(
                bids=((bid, bid_qty),),
                asks=((ask, ask_qty),),
                last_update_id=update_id,
                ts_ns=ts,
            ),
        )


def trade_events(rows: Iterator[dict[str, str]], instrument: Instrument) -> Iterator[Stamped]:
    """Prints, with the aggressor read the way Binance means it.

    `is_buyer_maker` true means the resting order was the buy, so the side that
    crossed was the seller. Reading it the other way round flips the sign of
    every flow measurement downstream, silently.
    """
    for row in rows:
        try:
            price = instrument.to_ticks(_pick(row, "price", "p"))
            qty = instrument.to_lots(_pick(row, "quantity", "q"))
            ts = _ts_ns(_pick(row, "transact_time", "T"))
            maker = _pick(row, "is_buyer_maker", "m").strip().lower()
            trade_id = int(float(_pick(row, "agg_trade_id", "a")))
        except (KeyError, ValueError, ArithmeticError):
            # Decimal raises InvalidOperation, not ValueError, on a malformed
            # price, so catching ValueError alone lets one bad row end the day.
            continue
        if qty <= 0:
            continue
        buyer_was_maker = maker in ("true", "1", "t", "yes")
        yield Stamped(
            ts,
            TradeTick(
                price=price,
                qty=qty,
                aggressor=Side.SELL if buyer_was_maker else Side.BUY,
                trade_id=trade_id,
                ts_ns=ts,
            ),
        )


def book_from_tape(
    rows: Iterator[dict[str, str]],
    instrument: Instrument,
    trades=None,
) -> Iterator[Stamped]:
    """A touch inferred from the prints, for the days with no bookTicker.

    Binance publishes aggTrades daily but not the quote stream, and bookDepth
    is aggregated into percentage bands with no touch in it at all. The prints
    still carry the side that crossed: a trade with the buyer as maker took the
    bid, so that price *was* the bid, and one with the buyer as taker lifted the
    ask. Tracking the last of each reconstructs a two-sided touch.

    What this gives up, and in which direction:

      **It only moves when something trades.** A quote that widens or narrows
      without a print is invisible, so the reconstructed book is stale between
      trades and lags the real one during a sweep.

      **There is no size.** Both sides are emitted with a nominal one lot, so
      queue position at the touch is meaningless here and any strategy that
      depends on it cannot be judged from this stream.

    What survives is the question this was built for: whether the tape reached
    a price, and at what cost. That is carried entirely by the prints.

    `trades` names the reader for this venue's column layout — Binance's by
    default, Bybit's when that archive is being read. The reconstruction is
    identical once the aggressor is known; only the field that carries it
    differs, and the two venues spell it as opposites.
    """
    bid: int | None = None
    ask: int | None = None
    one = max(1, instrument.to_lots("1") or 1)
    for stamped in (trades or trade_events)(rows, instrument):
        tick = stamped.event
        if tick.aggressor is Side.SELL:
            bid = tick.price
        else:
            ask = tick.price
        if bid is not None and ask is not None and ask > bid:
            yield Stamped(
                stamped.ts_ns,
                DepthSnapshot(
                    bids=((bid, one),),
                    asks=((ask, one),),
                    last_update_id=tick.trade_id,
                    ts_ns=stamped.ts_ns,
                ),
            )
        yield stamped


def merge(*streams: Iterator[Stamped]) -> Iterator[Stamped]:
    """Interleave by timestamp.

    The book and the tape are separate files and both matter in order: a print
    applied before the quote it traded against measures the wrong thing.
    """
    import heapq

    yield from heapq.merge(*streams, key=lambda s: s.ts_ns)


async def fetch(url: str) -> bytes | None:
    """The archive for one day, or None when the venue has no such file.

    A missing day is ordinary — a symbol listed last month has no archive for
    last year, and the most recent day is not published until it ends. Treating
    that as an error would stop a month-long download on its first gap.
    """
    import aiohttp

    from ..net import make_session

    async with make_session() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            return await resp.read()
