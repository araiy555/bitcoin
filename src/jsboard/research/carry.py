"""Whether holding spot long against a short perpetual pays for itself.

The basis replay closed the fast version of this trade: in the widest moment
of an hour the spot/perp gap was 3.43bps against 30.52bps of cost, so there is
nothing to capture by entering and exiting quickly. What remains is the slow
version — enter once, hold, and collect the funding the short perpetual is paid
every settlement — and it is a different question with a different failure
mode.

The expected value of that trade is not the interesting number. Funding is
positive most of the time and everyone knows it; the average is quoted in
every introduction to the trade. What decides it is the worst case:

  **How long does the entry cost take to earn back?** The four taker legs are
  paid up front. Until cumulative funding exceeds them the position is
  underwater no matter what the average says.

  **How deep and how long are the negative stretches?** Funding turns negative
  when the market is short-heavy, and then the position pays instead of being
  paid. A stretch long enough to undo months of accumulation is the whole risk.

So this module reports the distribution of payback times and the drawdown of
the cumulative funding curve, and treats the mean as context rather than as the
answer.

Two things it deliberately does not model. Liquidation risk on the perpetual
leg: a delta-neutral book still needs margin, and a violent move can force the
short closed at the worst moment. And exchange risk: the position is only as
good as the venue holding both legs. Neither is visible in a funding series,
and pretending otherwise would make the numbers here look safer than the trade.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime

NS_PER_S = 1_000_000_000
MS_PER_DAY = 86_400_000
BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class FundingPoint:
    """One settlement, as the exchange reported it."""

    ts_ms: int
    rate: float
    """Fraction of notional, signed. Positive means the short leg is paid."""

    @property
    def bps(self) -> float:
        return self.rate * BPS

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.ts_ms / 1000, tz=UTC)


@dataclass(frozen=True, slots=True)
class Payback:
    """How one entry fared."""

    started_ms: int
    days: float | None
    """None when the cost was never recovered before the data ran out."""


class CarryStudy:
    """Funding history turned into the numbers that decide the trade."""

    def __init__(
        self,
        points: list[FundingPoint],
        *,
        entry_cost_bps: float = 28.0,
        drawdown_window_days: int = 30,
    ) -> None:
        if entry_cost_bps < 0:
            raise ValueError("entry cost cannot be negative")
        if drawdown_window_days <= 0:
            raise ValueError("the drawdown window must be at least one day")
        self.points = sorted(points, key=lambda p: p.ts_ms)
        self.entry_cost_bps = entry_cost_bps
        self.drawdown_window_days = drawdown_window_days

    # ------------------------------------------------------------------ shape

    def __len__(self) -> int:
        return len(self.points)

    @property
    def span_days(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].ts_ms - self.points[0].ts_ms) / MS_PER_DAY

    @property
    def interval_hours(self) -> float:
        """Measured, not assumed.

        Most perpetuals settle every 8 hours, some every 4 or 1, and a symbol
        can change. Annualising with a hard-coded 3-per-day would silently
        misreport any of them by a factor of two or eight.
        """
        if len(self.points) < 2:
            return 0.0
        gaps = [
            (b.ts_ms - a.ts_ms) / 3_600_000
            for a, b in zip(self.points, self.points[1:], strict=False)
        ]
        return statistics.median(gaps)

    # ----------------------------------------------------------------- levels

    @property
    def mean_bps(self) -> float:
        return statistics.fmean(p.bps for p in self.points) if self.points else 0.0

    @property
    def median_bps(self) -> float:
        return statistics.median(p.bps for p in self.points) if self.points else 0.0

    @property
    def annual_pct(self) -> float:
        """Simple, not compounded: the position is a fixed notional, not a balance."""
        if not self.points or self.interval_hours <= 0:
            return 0.0
        per_year = 365.0 * 24.0 / self.interval_hours
        return self.mean_bps * per_year / 100.0

    @property
    def negative_share(self) -> float:
        if not self.points:
            return 0.0
        return sum(1 for p in self.points if p.rate < 0) / len(self.points)

    @property
    def worst_settlement_bps(self) -> float:
        return min((p.bps for p in self.points), default=0.0)

    # ------------------------------------------------------------- cumulative

    def cumulative_bps(self) -> list[float]:
        total = 0.0
        out = []
        for point in self.points:
            total += point.bps
            out.append(total)
        return out

    @property
    def max_drawdown_bps(self) -> float:
        """The deepest fall from a high in the cumulative funding curve.

        Reported positive. This is what a position opened at the worst moment
        gave back before recovering, on top of the entry cost it had already
        paid.
        """
        peak = 0.0
        worst = 0.0
        for value in self.cumulative_bps():
            peak = max(peak, value)
            worst = max(worst, peak - value)
        return worst

    @property
    def longest_underwater_days(self) -> float:
        """How long the curve stayed below a previous high."""
        peak = 0.0
        peak_ms = self.points[0].ts_ms if self.points else 0
        longest = 0.0
        for point, value in zip(self.points, self.cumulative_bps(), strict=False):
            # Measured to the recovery, not to the last point still down: the
            # stretch is not over until the old high is back, and stopping a
            # settlement early understates every episode by one interval.
            longest = max(longest, (point.ts_ms - peak_ms) / MS_PER_DAY)
            if value >= peak:
                peak, peak_ms = value, point.ts_ms
        return longest

    def worst_window_bps(self) -> float:
        """The worst run of `drawdown_window_days` consecutive settlements."""
        if self.interval_hours <= 0:
            return 0.0
        per_window = max(1, round(self.drawdown_window_days * 24 / self.interval_hours))
        if len(self.points) < per_window:
            return 0.0
        rates = [p.bps for p in self.points]
        running = sum(rates[:per_window])
        worst = running
        for i in range(per_window, len(rates)):
            running += rates[i] - rates[i - per_window]
            worst = min(worst, running)
        return worst

    # ---------------------------------------------------------------- payback

    def paybacks(self, *, max_hold_days: float = 180.0) -> list[Payback]:
        """For every possible entry, how long the cost took to earn back.

        Every settlement is a candidate entry rather than a sampled few: the
        question is what an unlucky entry looked like, and sampling is exactly
        what hides it.
        """
        rates = [p.bps for p in self.points]
        out: list[Payback] = []
        for i, start in enumerate(self.points):
            earned = 0.0
            recovered: float | None = None
            for j in range(i + 1, len(self.points)):
                elapsed = (self.points[j].ts_ms - start.ts_ms) / MS_PER_DAY
                if elapsed > max_hold_days:
                    break
                earned += rates[j]
                if earned >= self.entry_cost_bps:
                    recovered = elapsed
                    break
            out.append(Payback(start.ts_ms, recovered))
        return out

    def payback_summary(self, *, max_hold_days: float = 180.0) -> dict:
        """Median, tail and failure rate of the payback distribution.

        Entries too close to the end of the data cannot be judged — there was
        not enough history left for them to recover in — so they are excluded
        rather than counted as failures, which would make any series look worse
        the more recent it is.
        """
        results = self.paybacks(max_hold_days=max_hold_days)
        if not self.points:
            return {"judged": 0, "never": 0, "never_share": 0.0, "median_days": None, "p90_days": None}
        cutoff = self.points[-1].ts_ms - max_hold_days * MS_PER_DAY
        judged = [r for r in results if r.days is not None or r.started_ms <= cutoff]
        done = sorted(r.days for r in judged if r.days is not None)
        never = sum(1 for r in judged if r.days is None)
        return {
            "judged": len(judged),
            "never": never,
            "never_share": never / len(judged) if judged else 0.0,
            "median_days": statistics.median(done) if done else None,
            "p90_days": done[int(len(done) * 0.9)] if done else None,
        }


# ---------------------------------------------------------------------- timed


@dataclass(frozen=True, slots=True)
class TimedResult:
    """Holding the carry only while funding is rich enough to be worth it."""

    trips: int
    net_bps: float
    days_held: float
    span_days: float
    max_drawdown_bps: float
    equity_bps: list[float]

    @property
    def time_in_market(self) -> float:
        return self.days_held / self.span_days if self.span_days else 0.0

    @property
    def annual_pct_full(self) -> float:
        """Return on capital that sat ready the whole period, idle or not.

        This is what the strategy earns if it is the only use of the money.
        """
        if self.span_days <= 0:
            return 0.0
        return self.net_bps / self.span_days * 365.0 / 100.0

    @property
    def annual_pct_deployed(self) -> float:
        """Return per day actually in the position.

        Higher than the full-period figure whenever the rule sits out, and it
        says whether the position is good rather than whether the schedule is.
        """
        if self.days_held <= 0:
            return 0.0
        return self.net_bps / self.days_held * 365.0 / 100.0


def run_timed(
    points: list[FundingPoint],
    *,
    lookback: int,
    enter_bps: float,
    exit_bps: float,
    entry_cost_bps: float = 28.0,
) -> TimedResult:
    """Enter when trailing funding is rich, leave when it thins out.

    The signal is the mean of the `lookback` settlements **strictly before**
    the one being positioned for. Including the current settlement would let
    the rule see the payment it is deciding to collect, which turns any series
    into a winner and means nothing.

    The whole round-trip cost is charged at entry rather than split across the
    two ends. It is the same money either way, and charging it up front keeps
    a position that is closed by the end of the data from looking free.
    """
    if lookback < 1:
        raise ValueError("the lookback needs at least one settlement")
    if exit_bps > enter_bps:
        raise ValueError("the exit threshold cannot be above the entry threshold")
    if len(points) <= lookback:
        return TimedResult(0, 0.0, 0.0, 0.0, 0.0, [])

    rates = [p.bps for p in points]
    equity = 0.0
    curve: list[float] = []
    held = False
    trips = 0
    settlements_held = 0
    for i in range(lookback, len(points)):
        window = rates[i - lookback : i]
        signal = statistics.fmean(window)
        if not held and signal >= enter_bps:
            held = True
            trips += 1
            equity -= entry_cost_bps
        elif held and signal < exit_bps:
            held = False
        if held:
            equity += rates[i]
            settlements_held += 1
        curve.append(equity)

    peak = 0.0
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        worst = max(worst, peak - value)

    interval_days = (
        (points[-1].ts_ms - points[0].ts_ms) / MS_PER_DAY / (len(points) - 1)
        if len(points) > 1
        else 0.0
    )
    return TimedResult(
        trips=trips,
        net_bps=equity,
        days_held=settlements_held * interval_days,
        span_days=(points[-1].ts_ms - points[lookback].ts_ms) / MS_PER_DAY,
        max_drawdown_bps=worst,
        equity_bps=curve,
    )


# --------------------------------------------------------------------- source


def parse_funding(payload: list[dict]) -> list[FundingPoint]:
    """Binance's `/fapi/v1/fundingRate` rows, with the strings made numbers."""
    return [
        FundingPoint(ts_ms=int(row["fundingTime"]), rate=float(row["fundingRate"]))
        for row in payload
    ]


async def fetch_funding(symbol: str, *, days: float, now_ms: int | None = None) -> list[FundingPoint]:
    """Page back through the public funding history.

    No credentials: this endpoint is public, and the trade being studied does
    not exist yet, so there is nothing to authenticate as.
    """
    import time

    import aiohttp

    from ..net import make_session

    end = now_ms if now_ms is not None else int(time.time() * 1000)
    start = end - int(days * MS_PER_DAY)
    rows: list[dict] = []
    async with make_session() as session:
        cursor = start
        while cursor < end:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/fundingRate",
                params={"symbol": symbol.upper(), "startTime": cursor, "limit": 1000},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                page = await resp.json()
            if not page:
                break
            rows.extend(page)
            last = int(page[-1]["fundingTime"])
            if len(page) < 1000 or last <= cursor:
                break
            cursor = last + 1
    return parse_funding(rows)
