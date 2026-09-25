"""Which Japanese books look like bitbank ADA did, today.

What separated the one book that paid from the ones that did not was not the
spread alone:

* GMO's leverage XRP had a 7bps spread and lost — takers pay nothing there,
  so a quote left stale by a move on Binance is taken for free.
* bitbank's XRP pays makers 2bps and still left nothing — the rebate had
  drawn enough makers to squeeze the spread to half a basis point.
* bitbank's ADA won: a few bps of spread, a maker rebate, and a 12bps taker
  fee that makes picking off a stale quote cost more than most moves.

So a book is a candidate when all three hold, and is ranked by what a maker
earns per side before being picked off: half the spread plus the rebate.
That is a screen, not a result — adverse selection only shows up in a
recorded replay, which is the second stage.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

MIN_TAKER_BPS = 3.0
"""Below this, stale quotes are picked off for free (GMO leverage)."""

MIN_SPREAD_BPS = 1.0
"""Below this, the rebate has already been competed away (bitbank XRP)."""

MIN_VOLUME_JPY = 50_000_000.0
"""Below this, too few fills to earn anything whatever the edge."""


@dataclass(slots=True)
class Book:
    venue: str
    symbol: str
    maker_bps: float
    """Positive is a fee; negative is a rebate paid to us."""
    taker_bps: float
    spreads_bps: list[float] = field(default_factory=list)
    volume_jpy: float = 0.0

    @property
    def spread_bps(self) -> float:
        return statistics.median(self.spreads_bps) if self.spreads_bps else float("nan")

    @property
    def rebate_bps(self) -> float:
        return -self.maker_bps

    @property
    def edge_bps(self) -> float:
        """Per side, before adverse selection: half the spread plus the rebate."""
        return self.spread_bps / 2.0 + self.rebate_bps

    def verdict(self) -> str:
        if not self.spreads_bps:
            return "板なし"
        if self.taker_bps < MIN_TAKER_BPS:
            return "狙われやすい"
        if self.spread_bps < MIN_SPREAD_BPS:
            return "業者が詰めている"
        if self.volume_jpy < MIN_VOLUME_JPY:
            return "取引が少ない"
        if self.edge_bps <= 0:
            return "取り分なし"
        return "候補"


def rate_bps(rate) -> float:
    """`"-0.0002"` → -2.0, without the float noise of a bare multiply."""
    return round(float(rate) * 1e4, 6)


def spread_bps(bid: float, ask: float) -> float | None:
    if bid <= 0 or ask <= bid:
        return None
    return (ask - bid) / ((ask + bid) / 2.0) * 1e4


def bitbank_books(pairs_payload: dict) -> dict[str, Book]:
    """`/v1/spot/pairs` → one book per enabled JPY pair, with its fees."""
    books = {}
    for info in (pairs_payload.get("data") or {}).get("pairs") or []:
        name = info.get("name", "")
        if not name.endswith("_jpy") or info.get("is_enabled") is False:
            continue
        maker = info.get("maker_fee_rate_quote", info.get("maker_fee_rate_base"))
        taker = info.get("taker_fee_rate_quote", info.get("taker_fee_rate_base"))
        if maker is None or taker is None:
            continue
        books[name] = Book("bitbank", name, rate_bps(maker), rate_bps(taker))
    return books


def gmo_books(symbols_payload: dict) -> dict[str, Book]:
    """`/public/v1/symbols` → one book per symbol, spot and leverage alike."""
    books = {}
    for rule in symbols_payload.get("data") or []:
        symbol = rule.get("symbol")
        if not symbol or rule.get("makerFee") is None or rule.get("takerFee") is None:
            continue
        books[symbol] = Book("gmo", symbol, rate_bps(rule["makerFee"]), rate_bps(rule["takerFee"]))
    return books


def add_bitbank_tickers(books: dict[str, Book], payload: dict) -> None:
    for row in payload.get("data") or []:
        book = books.get(row.get("pair"))
        if book is None:
            continue
        s = spread_bps(float(row.get("buy") or 0), float(row.get("sell") or 0))
        if s is not None:
            book.spreads_bps.append(s)
        book.volume_jpy = float(row.get("vol") or 0) * float(row.get("last") or 0)


def add_gmo_tickers(books: dict[str, Book], payload: dict) -> None:
    for row in payload.get("data") or []:
        book = books.get(row.get("symbol"))
        if book is None:
            continue
        s = spread_bps(float(row.get("bid") or 0), float(row.get("ask") or 0))
        if s is not None:
            book.spreads_bps.append(s)
        book.volume_jpy = float(row.get("volume") or 0) * float(row.get("last") or 0)


def ranked(books: list[Book]) -> list[Book]:
    """Candidates first by edge, then everything else by volume."""
    def key(book: Book) -> tuple[bool, float]:
        candidate = book.verdict() == "候補"
        return (not candidate, -(book.edge_bps if candidate else book.volume_jpy))

    return sorted(books, key=key)


def as_record(book: Book) -> dict:
    return {
        "venue": book.venue,
        "symbol": book.symbol,
        "spread_bps": round(book.spread_bps, 3),
        "rebate_bps": round(book.rebate_bps, 3),
        "taker_bps": round(book.taker_bps, 3),
        "edge_bps": round(book.edge_bps, 3),
        "volume_jpy": round(book.volume_jpy),
        "verdict": book.verdict(),
    }
