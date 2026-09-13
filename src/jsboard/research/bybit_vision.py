"""Bybit's published daily trade archives.

Binance and Bybit both list a USDT perpetual on the same underlying. The two
contracts settle against near-identical indices, so their prices are tied
together by nothing more than participants moving between them — which is the
same shape as an ETF against its basket, minus the part that needs a licence.
Whether that tie is loose enough to pay for crossing twice is a measurement,
and it needs both venues' history rather than an hour of live recording.

Bybit publishes one gzipped CSV per symbol per day:

    https://public.bybit.com/trading/<SYM>/<SYM><YYYY-MM-DD>.csv.gz

Note the missing separator before the date — it is not the Binance layout
with a different host, and a URL built by analogy returns 404 on every day.

**`side` is the aggressor, not the maker.** Binance says `is_buyer_maker`,
where true means the *seller* crossed; Bybit says `side`, and Buy means the
*buyer* crossed. The two fields look interchangeable and mean opposite
things. Reading one as the other flips every flow measurement downstream
without raising, and on a cross-venue spread it would flip which venue looks
rich.
"""

from __future__ import annotations

import csv
import gzip
import io
from collections.abc import Iterator
from datetime import date

from ..core.types import Instrument, Side
from ..feed.base import TradeTick
from .vision import Stamped, _is_number, _pick, book_from_tape, ts_ns_auto

BASE = "https://public.bybit.com/trading"

TRADE_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "size",
    "price",
    "tickDirection",
    "trdMatchID",
    "grossValue",
    "homeNotional",
    "foreignNotional",
)


def daily_url(symbol: str, day: date) -> str:
    sym = symbol.upper()
    return f"{BASE}/{sym}/{sym}{day:%Y-%m-%d}.csv.gz"


def read_gzip_csv(blob: bytes, columns: tuple[str, ...] = TRADE_COLUMNS) -> Iterator[dict[str, str]]:
    """Rows as dicts, by header when the file carries one.

    Bybit has always shipped a header, but reading positionally when the
    first row is data costs one branch and saves a silent column shift if
    that ever changes.
    """
    text = io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(blob)), encoding="utf-8")
    first = text.readline()
    if not first:
        return
    cells = next(csv.reader([first]))
    if _is_number(cells[0] if cells else ""):
        yield dict(zip(columns, cells, strict=False))
        for row in csv.reader(text):
            if row:
                yield dict(zip(columns, row, strict=False))
    else:
        yield from csv.DictReader(text, fieldnames=[c.strip() for c in cells])


def trade_events(rows: Iterator[dict[str, str]], instrument: Instrument) -> Iterator[Stamped]:
    """Prints, with `side` read as the side that crossed."""
    for row in rows:
        try:
            price = instrument.to_ticks(_pick(row, "price", "p"))
            qty = instrument.to_lots(_pick(row, "size", "v", "q"))
            ts = ts_ns_auto(_pick(row, "timestamp", "T", "t"))
            side = _pick(row, "side", "S").strip().lower()
        except (KeyError, ValueError, ArithmeticError):
            # Decimal raises InvalidOperation, not ValueError, on a malformed
            # price, so catching ValueError alone lets one bad row end the day.
            continue
        if qty <= 0 or price <= 0:
            continue
        yield Stamped(
            ts,
            TradeTick(
                price=price,
                qty=qty,
                aggressor=Side.BUY if side.startswith("b") else Side.SELL,
                trade_id=0,
                ts_ns=ts,
            ),
        )


def book_events(rows: Iterator[dict[str, str]], instrument: Instrument) -> Iterator[Stamped]:
    """A touch inferred from the prints; Bybit publishes no daily quote file.

    Same construction as the Binance fallback, and the same limits: it only
    moves when something trades, and it carries no size. For a cross-venue
    spread that is survivable — the question is where each venue's price is,
    and a print is a price that actually happened on that venue — but it
    means the execution cost of size beyond the touch is not measurable here.
    """
    return book_from_tape(rows, instrument, trades=trade_events)


async def fetch(url: str) -> bytes | None:
    """The archive for one day, or None when the venue has no such file."""
    import aiohttp

    from ..net import make_session

    async with make_session() as session, session.get(
        url, timeout=aiohttp.ClientTimeout(total=300)
    ) as resp:
        if resp.status == 404:
            return None
        resp.raise_for_status()
        return await resp.read()
