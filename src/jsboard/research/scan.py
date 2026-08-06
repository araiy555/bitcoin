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
from dataclasses import dataclass, field

import aiohttp

from ..net import make_session

# Spot and USDⓈ-M futures answer the same two questions at different hosts
# and paths. Keeping them side by side rather than behind a flag deep in the
# code makes it obvious which market a result came from — and the fee that
# applies differs between them by roughly a factor of five, which is the
# whole reason for scanning the second one.
VENUES = {
    "spot": (
        "https://api.binance.com",
        "/api/v3/ticker/bookTicker",
        "/api/v3/ticker/24hr",
    ),
    "perp": (
        "https://fapi.binance.com",
        "/fapi/v1/ticker/bookTicker",
        "/fapi/v1/ticker/24hr",
    ),
}

REST_BASE = VENUES["spot"][0]
BOOK_TICKER = VENUES["spot"][1]
DAY_TICKER = VENUES["spot"][2]


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
    spread_samples: list[float] = field(default_factory=list)
    depth_samples: list[float] = field(default_factory=list)
    """Repeated observations. A single snapshot of a thin book is close to
    worthless — depth there swings by an order of magnitude between polls, so
    one reading can flip a symbol's verdict either way. When samples are
    present the medians are used instead."""

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def is_two_sided(self) -> bool:
        return self.bid > 0 and self.ask > self.bid

    @property
    def spread_bps(self) -> float:
        if self.spread_samples:
            return _median(self.spread_samples)
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
        if self.depth_samples:
            return _median(self.depth_samples)
        return min(self.bid_qty * self.bid, self.ask_qty * self.ask)

    @property
    def depth_swing(self) -> float:
        """How far the touch moved across samples, largest over smallest.

        1.0 means it never budged; 10 means a single snapshot could have been
        off by an order of magnitude, which is the normal state of affairs on
        a thin symbol.
        """
        usable = [d for d in self.depth_samples if d > 0]
        if len(usable) < 2:
            return 1.0
        return max(usable) / min(usable)

    @property
    def samples_taken(self) -> int:
        return len(self.depth_samples)

    def queue_ratio(self, size_quote: float) -> float:
        """How many times our own size is already queued ahead of us."""
        if size_quote <= 0:
            return float("inf")
        return self.top_of_book_quote / size_quote

    def capacity_verdict(
        self, size_quote: float, *, thin: float = 5.0, crowded: float = 50.0
    ) -> str:
        """Whether an order of this size can realistically work here.

        Clearing the fee is necessary and nowhere near sufficient. The two
        ways a surviving symbol still fails are opposites, and a symbol has
        to thread between them:

        too thin    our order is a large share of the level. We are not
                    joining a book, we *are* the book — the fill model stops
                    describing anything real, and there is no size to exit
                    into. The default draws the line at a fifth of the level.

        too deep    the queue ahead dwarfs us, so our turn rarely comes.
                    Queue position is most of a maker's edge and we would be
                    at the back of a very long line.

        Both thresholds are judgement, not law — a real desk would set them
        from its own fill data. They are arguments so they can be argued with.
        """
        ratio = self.queue_ratio(size_quote)
        if ratio < thin:
            return "板が薄い"
        if ratio > crowded:
            return "行列が長い"
        return "可"


@dataclass(slots=True)
class ScanFilters:
    quote_asset: str = "USDT"
    product: str = "spot"
    """Which market to look at. The maker fee differs between them, and a
    spot fee applied to a futures book answers a question nobody asked."""
    maker_bps: float = 10.0
    min_quote_volume: float = 1_000_000.0
    min_trades: int = 1_000
    size_quote: float = 1_000.0
    thin_ratio: float = 5.0
    """Below this many times our size at the touch, we would be the book."""
    crowded_ratio: float = 50.0
    """Above this, the queue ahead is long enough that we rarely reach it."""
    samples: int = 5
    """Polls of the touch. One is not enough to judge a thin book."""
    sample_interval: float = 2.0
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


async def _get_json(session, path, base=REST_BASE):
    timeout = aiohttp.ClientTimeout(total=30)
    async with session.get(base + path, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


def add_sample(book: dict[str, SymbolStats], rows: list[dict]) -> None:
    """Fold one more observation of the touch into the running samples."""
    for observed in parse_book_tickers(rows).values():
        stats = book.get(observed.symbol)
        if stats is None or not observed.is_two_sided:
            continue
        stats.spread_samples.append(observed.spread_bps)
        stats.depth_samples.append(observed.top_of_book_quote)


async def fetch_market(
    session: aiohttp.ClientSession | None = None,
    *,
    samples: int = 1,
    interval: float = 2.0,
    on_sample=None,
    product: str = "spot",
) -> dict[str, SymbolStats]:
    """The whole market for one product, optionally sampled several times.

    Volume and trade counts come from a single 24h call; only the touch is
    re-polled, since that is the part that moves between one look and the next.
    """
    if product not in VENUES:
        raise ValueError(f"product must be one of {tuple(VENUES)}")
    base, book_path, day_path = VENUES[product]

    owns = session is None
    session = session or make_session()
    try:
        book_rows = await _get_json(session, book_path, base)
        day_rows = await _get_json(session, day_path, base)
        book = merge_day_tickers(parse_book_tickers(book_rows), day_rows)

        add_sample(book, book_rows)
        if on_sample:
            on_sample(1, samples)

        for i in range(1, max(1, samples)):
            await asyncio.sleep(interval)
            add_sample(book, await _get_json(session, book_path, base))
            if on_sample:
                on_sample(i + 1, samples)
    finally:
        if owns:
            await session.close()

    return book


async def scan(filters: ScanFilters, *, on_sample=None) -> tuple[list[SymbolStats], int]:
    """Returns (ranked survivors, how many symbols were considered)."""
    market = await fetch_market(
        samples=filters.samples,
        interval=filters.sample_interval,
        on_sample=on_sample,
        product=filters.product,
    )
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
    def verdict(s: SymbolStats) -> str:
        return s.capacity_verdict(
            filters.size_quote, thin=filters.thin_ratio, crowded=filters.crowded_ratio
        )

    tradeable = [s for s in viable if verdict(s) == "可"]
    return {
        "liquid": len(results),
        "viable": len(viable),
        "tradeable": len(tradeable),
        # The placeable symbols are the answer, and ranking by edge buries
        # them: a wide spread nobody can reach outranks a narrow one that is
        # actually workable. Carry them out by name rather than as a count.
        "placeable": tradeable,
        "too_deep": sum(1 for s in viable if verdict(s) == "行列が長い"),
        "too_thin": sum(1 for s in viable if verdict(s) == "板が薄い"),
        "best_net_bps": viable[0].net_bps(filters.maker_bps) if viable else 0.0,
        "breakeven_spread_bps": 2.0 * filters.maker_bps,
        "median_spread_bps": _median([s.spread_bps for s in results]),
        "unstable": sum(1 for s in viable if s.depth_swing >= 3.0),
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


# --------------------------------------------------------------- persistence


@dataclass(slots=True)
class Persistence:
    """How reliably one symbol qualified, across repeated scans.

    A single scan answers "is this symbol viable right now", which turns out
    not to be the question. Two scans half an hour apart shared only half
    their surviving symbols, and of four that looked placeable in the first,
    one still did in the second — the touch on BONKUSDT fell to 0.15x and on
    MIRAUSDT to 0.09x, flipping both verdicts.

    Sampling the touch a few seconds apart does not catch this: within such a
    window these books barely move. The instability lives at the timescale you
    would actually run a maker over, so the only honest measurement is to
    repeat the whole scan over hours and count how often each symbol held up.
    """

    symbol: str
    rounds: int = 0
    """Rounds this symbol appeared in at all — it may drop out of the liquid
    set entirely between scans."""
    total_rounds: int = 0
    """Rounds the watch ran. Persistence is measured against this, not against
    `rounds`: a symbol that vanishes for half the watch has not been reliable,
    and scoring it only on the rounds it showed up for would say the opposite."""
    viable_rounds: int = 0
    tradeable_rounds: int = 0
    net_bps: list[float] = field(default_factory=list)
    depth: list[float] = field(default_factory=list)

    @property
    def persistence(self) -> float:
        """Fraction of the whole watch where the symbol was actually placeable."""
        denominator = self.total_rounds or self.rounds
        return self.tradeable_rounds / denominator if denominator else 0.0

    @property
    def presence(self) -> float:
        """Fraction of the watch where the symbol even made the liquid set."""
        return self.rounds / self.total_rounds if self.total_rounds else 0.0

    @property
    def median_net_bps(self) -> float:
        return _median(self.net_bps)

    @property
    def median_depth(self) -> float:
        return _median(self.depth)

    def breakeven_fee_bps(self, maker_bps: float) -> float:
        """The highest maker fee at which this symbol still clears its costs.

        Half the spread, because the fee is charged on both legs of a round
        trip. This is the number that actually decides things: running the
        scan at one fee answers only whether *that* fee works, while this
        says which fee tier would be needed — and a fee tier is negotiable
        in a way the spread is not.
        """
        return (self.median_net_bps + 2.0 * maker_bps) / 2.0

    @property
    def depth_swing(self) -> float:
        usable = [d for d in self.depth if d > 0]
        if len(usable) < 2:
            return 1.0
        return max(usable) / min(usable)


def fold_round(
    tally: dict[str, Persistence], results: list[SymbolStats], filters: ScanFilters
) -> None:
    """Add one completed scan to the running tally."""
    for stats in results:
        entry = tally.setdefault(stats.symbol, Persistence(symbol=stats.symbol))
        entry.rounds += 1
        net = stats.net_bps(filters.maker_bps)
        entry.net_bps.append(net)
        entry.depth.append(stats.top_of_book_quote)
        if net > 0:
            entry.viable_rounds += 1
            verdict = stats.capacity_verdict(
                filters.size_quote, thin=filters.thin_ratio, crowded=filters.crowded_ratio
            )
            if verdict == "可":
                entry.tradeable_rounds += 1


def rank_persistence(tally: dict[str, Persistence]) -> list[Persistence]:
    """Most reliably placeable first; ties broken by the edge on offer."""
    return sorted(
        tally.values(),
        key=lambda p: (p.persistence, p.median_net_bps),
        reverse=True,
    )


async def watch(
    filters: ScanFilters,
    *,
    rounds: int,
    every_seconds: float,
    on_round=None,
) -> dict[str, Persistence]:
    """Repeat the scan and record which symbols keep qualifying.

    Interrupting part-way is fine — whatever rounds completed are returned,
    which is usually what you want from a long watch.
    """
    tally: dict[str, Persistence] = {}
    for i in range(rounds):
        if i:
            await asyncio.sleep(every_seconds)
        results, considered = await scan(filters)
        fold_round(tally, results, filters)
        for entry in tally.values():
            entry.total_rounds = i + 1
        if on_round:
            on_round(i + 1, rounds, tally, considered)
    return tally
