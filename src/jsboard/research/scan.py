"""Find the symbols where market making could clear its own costs.

The gate is arithmetic, not opinion. Quoting a bid and an offer earns at most
the spread once per round trip, and the venue charges the maker fee on *both*
legs. So the whole strategy is viable only where

    spread_bps  >  2 x maker_fee_bps

On BTCUSDT that comparison is 0.003 against 20 and the answer is no. This
module asks the same question of every symbol at once, using the two Binance
endpoints that return the entire market in a single request each.

Reading the output honestly matters more than running it. A wide spread is
not free money — it is usually the market pricing in the fact that nobody
wants to stand there. Three things have to hold together, and the scan
reports all three side by side:

  spread    must exceed twice the fee, or every round trip loses
  activity  a spread you never fill against pays nothing
  size      an edge in bps is only money when multiplied by real notional

A symbol that passes the first and fails the others is a trap, so `net_bps`
is deliberately not the only column.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp

from ..net import make_session

REST_BASE = "https://api.binance.com"
BOOK_TICKER = "/api/v3/ticker/bookTicker"
DAY_TICKER = "/api/v3/ticker/24hr"


@dataclass(slots=True)
class SymbolStats:
    """One symbol's top of book plus a day of activity."""

    symbol: str
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    quote_volume: float = 0.0
    trades: int = 0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def is_two_sided(self) -> bool:
        return self.bid > 0 and self.ask > self.bid

    @property
    def spread_bps(self) -> float:
        if not self.is_two_sided:
            return 0.0
        return (self.ask - self.bid) / self.mid * 10_000.0

    def net_bps(self, maker_bps: float) -> float:
        """Edge left after paying the maker fee on both legs."""
        return self.spread_bps - 2.0 * maker_bps

    def profit_per_round_trip(self, maker_bps: float, size_quote: float) -> float:
        """What one completed round trip pays, in quote currency."""
        return self.net_bps(maker_bps) / 10_000.0 * size_quote

    @property
    def top_of_book_quote(self) -> float:
        """Notional resting at the touch — how much size the level can absorb."""
        return min(self.bid_qty * self.bid, self.ask_qty * self.ask)

    def daily_ceiling(self, maker_bps: float, size_quote: float) -> float:
        """A generous upper bound on a day's take.

        Assumes we round-trip once per public trade, which no maker ever
        achieves — we are one participant among many in the same queue. Useful
        only for spotting symbols where even the fantasy number is negligible.
        """
        return self.profit_per_round_trip(maker_bps, size_quote) * self.trades / 2.0


@dataclass(slots=True)
class ScanFilters:
    quote_asset: str = "USDT"
    maker_bps: float = 10.0
    min_quote_volume: float = 1_000_000.0
    min_trades: int = 1_000
    size_quote: float = 1_000.0
    exclude_leveraged: bool = True
    """Drop UP/DOWN/BULL/BEAR tokens, which are not spot pairs in spirit."""


LEVERAGED_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")


def parse_book_tickers(rows: list[dict]) -> dict[str, SymbolStats]:
    out: dict[str, SymbolStats] = {}
    for row in rows:
        try:
            stats = SymbolStats(
                symbol=row["symbol"],
                bid=float(row["bidPrice"]),
                ask=float(row["askPrice"]),
                bid_qty=float(row["bidQty"]),
                ask_qty=float(row["askQty"]),
            )
        except (KeyError, ValueError):
            continue
        out[stats.symbol] = stats
    return out


def merge_day_tickers(book: dict[str, SymbolStats], rows: list[dict]) -> dict[str, SymbolStats]:
    for row in rows:
        stats = book.get(row.get("symbol", ""))
        if stats is None:
            continue
        try:
            stats.quote_volume = float(row["quoteVolume"])
            stats.trades = int(row["count"])
        except (KeyError, ValueError):
            continue
    return book


def apply_filters(book: dict[str, SymbolStats], f: ScanFilters) -> list[SymbolStats]:
    kept = []
    for stats in book.values():
        if not stats.symbol.endswith(f.quote_asset):
            continue
        if f.exclude_leveraged and stats.symbol.endswith(LEVERAGED_SUFFIXES):
            continue
        if not stats.is_two_sided:
            continue
        if stats.quote_volume < f.min_quote_volume:
            continue
        if stats.trades < f.min_trades:
            continue
        kept.append(stats)
    return kept


def rank(candidates: list[SymbolStats], maker_bps: float) -> list[SymbolStats]:
    return sorted(candidates, key=lambda s: s.net_bps(maker_bps), reverse=True)


async def fetch_market(session: aiohttp.ClientSession | None = None) -> dict[str, SymbolStats]:
    """Two requests, the whole spot market."""
    owns = session is None
    session = session or make_session()
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with session.get(REST_BASE + BOOK_TICKER, timeout=timeout) as resp:
            resp.raise_for_status()
            book_rows = await resp.json()
        async with session.get(REST_BASE + DAY_TICKER, timeout=timeout) as resp:
            resp.raise_for_status()
            day_rows = await resp.json()
    finally:
        if owns:
            await session.close()

    return merge_day_tickers(parse_book_tickers(book_rows), day_rows)


async def scan(filters: ScanFilters) -> tuple[list[SymbolStats], int]:
    """Returns (ranked survivors, how many symbols were considered)."""
    market = await fetch_market()
    considered = sum(
        1
        for s in market.values()
        if s.symbol.endswith(filters.quote_asset)
        and not (filters.exclude_leveraged and s.symbol.endswith(LEVERAGED_SUFFIXES))
    )
    survivors = apply_filters(market, filters)
    return rank(survivors, filters.maker_bps), considered


def summarise(results: list[SymbolStats], filters: ScanFilters) -> dict:
    """Headline numbers for the report."""
    viable = [s for s in results if s.net_bps(filters.maker_bps) > 0]
    return {
        "liquid": len(results),
        "viable": len(viable),
        "best_net_bps": viable[0].net_bps(filters.maker_bps) if viable else 0.0,
        "breakeven_spread_bps": 2.0 * filters.maker_bps,
        "median_spread_bps": _median([s.spread_bps for s in results]),
    }


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def run(filters: ScanFilters) -> tuple[list[SymbolStats], int]:
    return asyncio.run(scan(filters))
