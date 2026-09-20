"""Funding paid by one venue against funding paid by another.

Holding spot against a short perpetual on a single venue earned +5.21%/yr on
two years of BTCUSDT. That number is the whole market's view of the trade and
everyone collects the same one. The interesting question is whether venues
disagree: if one is paying the short leg more than another charges the long
leg, a position that is short where funding is high and long where it is low
collects the *difference*, and the difference is nobody's published rate.

Three things decide whether a gap that looks real is real.

**Settlement intervals differ.** Binance settles every eight hours,
Hyperliquid every hour. Comparing the raw numbers puts them on different
footings; annualising one with the other's frequency is wrong by a factor of
eight. Every rate here is carried with the interval it belongs to and
compared per-hour.

**A gap is only collectable while it is open.** Two venues that settle at
different times do not hand out their funding at the same instant, so a
position must be held across both settlements to collect both. The pairing
below matches settlements within a tolerance and ignores the rest rather than
interpolating a rate nobody ever paid.

**Both legs cost money to open.** The gap is per settlement; the entry is
paid once, on four legs across two venues. `CarryStudy` already turns that
into a payback distribution, so the output here feeds it rather than
reinventing it.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .carry import BPS, FundingPoint

HOURS_PER_YEAR = 365.0 * 24.0


@dataclass(frozen=True, slots=True)
class VenueFunding:
    """One venue's history, with the interval its rates are quoted over."""

    venue: str
    interval_hours: float
    points: list[FundingPoint]

    def hourly_bps(self, point: FundingPoint) -> float:
        """The rate put on a common footing.

        Without this a 0.01% hourly rate and a 0.01% eight-hourly rate read as
        equal, and the venue paying eight times as much looks the same as the
        one paying least.
        """
        return point.bps / self.interval_hours


@dataclass(frozen=True, slots=True)
class Spread:
    """One moment when two venues disagreed, priced per hour."""

    ts_ms: int
    short_venue: str
    long_venue: str
    hourly_bps: float
    """What the pair collects per hour of holding, before costs."""


def pair_spreads(
    venues: list[VenueFunding], *, tolerance_ms: int = 30 * 60_000
) -> list[Spread]:
    """The best available pair at each settlement of each venue.

    For every settlement on every venue, find the nearest settlement on each
    other venue within `tolerance_ms` and score the pair. The widest pair at
    that moment is kept. Settlements with no partner in range are dropped:
    a venue that has not settled yet has not paid anything, and inventing a
    rate for it would manufacture a gap that was never collectable.
    """
    if len(venues) < 2:
        return []

    indexed = [(v, sorted(v.points, key=lambda p: p.ts_ms)) for v in venues]
    out: list[Spread] = []
    seen: set[tuple[int, str, str]] = set()

    for venue, points in indexed:
        for point in points:
            rate = venue.hourly_bps(point)
            for other, other_points in indexed:
                if other.venue == venue.venue:
                    continue
                partner = _nearest(other_points, point.ts_ms, tolerance_ms)
                if partner is None:
                    continue
                other_rate = other.hourly_bps(partner)
                # Short where funding is higher, long where it is lower.
                if rate >= other_rate:
                    short, long_, gap = venue.venue, other.venue, rate - other_rate
                else:
                    short, long_, gap = other.venue, venue.venue, other_rate - rate
                key = (point.ts_ms // tolerance_ms, short, long_)
                if key in seen:
                    continue
                seen.add(key)
                out.append(Spread(point.ts_ms, short, long_, gap))
    out.sort(key=lambda s: s.ts_ms)
    return out


def _nearest(points: list[FundingPoint], ts_ms: int, tolerance_ms: int):
    """Closest settlement within the window, or None."""
    import bisect

    stamps = [p.ts_ms for p in points]
    i = bisect.bisect_left(stamps, ts_ms)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(points):
            gap = abs(points[j].ts_ms - ts_ms)
            if gap <= tolerance_ms and (best is None or gap < abs(best.ts_ms - ts_ms)):
                best = points[j]
    return best


@dataclass(frozen=True, slots=True)
class SpreadStudy:
    """What the cross-venue gap is worth, before and after the entry."""

    spreads: list[Spread]
    entry_cost_bps: float = 28.0

    @property
    def n(self) -> int:
        return len(self.spreads)

    @property
    def mean_hourly_bps(self) -> float:
        if not self.spreads:
            return 0.0
        return statistics.fmean(s.hourly_bps for s in self.spreads)

    @property
    def median_hourly_bps(self) -> float:
        if not self.spreads:
            return 0.0
        return statistics.median(s.hourly_bps for s in self.spreads)

    @property
    def annual_pct(self) -> float:
        """Simple, not compounded — the position is a fixed notional."""
        return self.mean_hourly_bps * HOURS_PER_YEAR / 100.0

    @property
    def payback_hours(self) -> float:
        """Hours of holding before the four entry legs are earned back."""
        rate = self.mean_hourly_bps
        return float("inf") if rate <= 0 else self.entry_cost_bps / rate

    def pair_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for s in self.spreads:
            key = (s.short_venue, s.long_venue)
            counts[key] = counts.get(key, 0) + 1
        return counts


def to_bps(rate: float) -> float:
    return rate * BPS


# --------------------------------------------------------------- fetching

# Settlement cadence, which is not the same everywhere and is not discoverable
# from the payload. Getting it wrong scales a venue's contribution by the
# ratio of the two intervals — eight times, between Binance and Hyperliquid.
INTERVAL_HOURS = {
    "binance": 8.0,
    "bybit": 8.0,
    "hyperliquid": 1.0,
}


async def fetch_binance(symbol: str, *, days: float, now_ms: int | None = None):
    from .carry import fetch_funding

    return VenueFunding(
        "binance", INTERVAL_HOURS["binance"], await fetch_funding(symbol, days=days, now_ms=now_ms)
    )


async def fetch_bybit(symbol: str, *, days: float, now_ms: int | None = None):
    """Bybit's public funding history. No credentials; the endpoint is open."""
    import time

    import aiohttp

    from ..net import make_session
    from .carry import MS_PER_DAY

    end = now_ms if now_ms is not None else int(time.time() * 1000)
    start = end - int(days * MS_PER_DAY)
    points: list[FundingPoint] = []
    async with make_session() as session:
        cursor = end
        while cursor > start:
            async with session.get(
                "https://api.bybit.com/v5/market/funding/history",
                params={
                    "category": "linear",
                    "symbol": symbol.upper(),
                    "endTime": cursor,
                    "limit": 200,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
            rows = ((body or {}).get("result") or {}).get("list") or []
            if not rows:
                break
            batch = [
                FundingPoint(int(r["fundingRateTimestamp"]), float(r["fundingRate"]))
                for r in rows
            ]
            points.extend(batch)
            oldest = min(p.ts_ms for p in batch)
            if oldest >= cursor or len(rows) < 200:
                break
            cursor = oldest - 1
    points = [p for p in points if start <= p.ts_ms <= end]
    points.sort(key=lambda p: p.ts_ms)
    return VenueFunding("bybit", INTERVAL_HOURS["bybit"], points)


async def fetch_hyperliquid(symbol: str, *, days: float, now_ms: int | None = None):
    """Hyperliquid settles hourly, and takes the bare coin rather than a pair."""
    import time

    import aiohttp

    from ..net import make_session
    from .carry import MS_PER_DAY

    end = now_ms if now_ms is not None else int(time.time() * 1000)
    start = end - int(days * MS_PER_DAY)
    coin = symbol.upper().removesuffix("USDT").removesuffix("USD")
    points: list[FundingPoint] = []
    async with make_session() as session:
        cursor = start
        while cursor < end:
            async with session.post(
                "https://api.hyperliquid.xyz/info",
                json={"type": "fundingHistory", "coin": coin, "startTime": cursor},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                rows = await resp.json()
            if not rows:
                break
            batch = [FundingPoint(int(r["time"]), float(r["fundingRate"])) for r in rows]
            points.extend(batch)
            newest = max(p.ts_ms for p in batch)
            if newest <= cursor:
                break
            cursor = newest + 1
    points = [p for p in points if start <= p.ts_ms <= end]
    points.sort(key=lambda p: p.ts_ms)
    return VenueFunding("hyperliquid", INTERVAL_HOURS["hyperliquid"], points)


FETCHERS = {
    "binance": fetch_binance,
    "bybit": fetch_bybit,
    "hyperliquid": fetch_hyperliquid,
}
