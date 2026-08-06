"""Minute bars with everything the archive can supply, and nothing it cannot.

The fixed-horizon study asked what the market offers at every timepoint and
found it offers less than the fee. The question now is narrower: are there
*conditions* under which it offers more. That needs features, and features
need a decision about which ones can be trusted.

What is here comes from spot and perp trades plus the open-interest metrics
file. What is deliberately absent is anything about the book — level
imbalance, cancel asymmetry, refill, microprice. The archive publishes book
depth as cumulative size at ±1…5% of mid once a minute, which is three orders
of magnitude too coarse to see a side of the touch vanish, and the
best-bid/ask feed stopped in March 2024. Those features have to come from a
live `capture` recording, so they get a slot rather than a guess.

Two things decide whether a feature here is honest.

**Nothing may see its own future.** Open interest is published every five
minutes, so the value attached to a minute is the last mark at or before it,
never the one after. A forward-filled series that fills from the right is the
most comfortable lookahead bug there is: everything still runs, and the
backtest quietly knows things.

**A z-score needs a trailing window, not the whole sample.** Scoring volume
against the full period's mean and standard deviation lets a quiet January
know about a violent March. Every z-score here is computed against the
preceding `window` minutes only.
"""

from __future__ import annotations

import bisect
import csv
import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .archive import SecondBar

MINUTE = 60


@dataclass(slots=True)
class MinuteBar:
    """One minute of both products, plus derived state."""

    minute: int  # unix seconds, floored to the minute

    spot_last: float = 0.0
    spot_buy: float = 0.0
    spot_sell: float = 0.0
    spot_trades: int = 0

    perp_last: float = 0.0
    perp_buy: float = 0.0
    perp_sell: float = 0.0
    perp_trades: int = 0

    perp_high: float = 0.0
    perp_low: float = 0.0

    open_interest: float = 0.0

    @property
    def perp_volume(self) -> float:
        return self.perp_buy + self.perp_sell

    @property
    def spot_volume(self) -> float:
        return self.spot_buy + self.spot_sell

    @property
    def perp_buy_ratio(self) -> float:
        """Share of perp volume that lifted the offer. 0.5 when balanced."""
        total = self.perp_volume
        return self.perp_buy / total if total > 0 else 0.5

    @property
    def spot_buy_ratio(self) -> float:
        total = self.spot_volume
        return self.spot_buy / total if total > 0 else 0.5

    @property
    def basis_bps(self) -> float:
        """Perp against spot. Positive means the perp trades rich."""
        if self.spot_last <= 0 or self.perp_last <= 0:
            return 0.0
        return (self.perp_last - self.spot_last) / self.spot_last * 10_000.0

    @property
    def has_both(self) -> bool:
        return self.spot_last > 0 and self.perp_last > 0


def to_minutes(spot: list[SecondBar], perp: list[SecondBar]) -> list[MinuteBar]:
    """Fold both tapes into aligned minute bars.

    Minutes where either product did not trade are dropped rather than
    filled: a basis computed against a stale spot print is not a basis, and
    every condition here compares the two products.
    """
    bars: dict[int, MinuteBar] = {}

    for sec in spot:
        m = sec.sec - sec.sec % MINUTE
        bar = bars.get(m) or bars.setdefault(m, MinuteBar(minute=m))
        bar.spot_last = sec.last
        bar.spot_buy += sec.buy_qty
        bar.spot_sell += sec.sell_qty
        bar.spot_trades += sec.trades

    for sec in perp:
        m = sec.sec - sec.sec % MINUTE
        bar = bars.get(m) or bars.setdefault(m, MinuteBar(minute=m))
        bar.perp_last = sec.last
        bar.perp_buy += sec.buy_qty
        bar.perp_sell += sec.sell_qty
        bar.perp_trades += sec.trades
        bar.perp_high = max(bar.perp_high, sec.high)
        bar.perp_low = sec.low if bar.perp_low == 0 else min(bar.perp_low, sec.low)

    return [bars[m] for m in sorted(bars) if bars[m].has_both]


def load_open_interest(path: Path) -> list[tuple[int, float]]:
    """(unix second, open interest) from a metrics archive, ascending.

    Published every five minutes. Returned as marks rather than resampled,
    so the caller decides how to attach them — and can only attach backwards.
    """
    out: list[tuple[int, float]] = []
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            for row in csv.reader(io.TextIOWrapper(raw, encoding="utf-8")):
                if len(row) < 3:
                    continue
                try:
                    stamp = _parse_time(row[0])
                    value = float(row[2])
                except (ValueError, IndexError):
                    continue  # header row
                out.append((stamp, value))
    out.sort()
    return out


def _parse_time(text: str) -> int:
    """`2026-08-01 00:05:00` to unix seconds, without a timezone library.

    The archive stamps these in UTC. Doing the arithmetic here keeps the
    parse total and obvious; `datetime.fromisoformat` would also work but
    would silently accept a local-time reading of the same string.
    """
    import datetime as _dt

    parsed = _dt.datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S")
    return int(parsed.replace(tzinfo=_dt.UTC).timestamp())


def attach_open_interest(bars: list[MinuteBar], marks: list[tuple[int, float]]) -> None:
    """Give each minute the last OI mark at or before it.

    At or *before*. Taking the nearest mark would let a minute see a value
    published up to five minutes later, which is the kind of lookahead that
    leaves every test passing and the backtest wrong.
    """
    if not marks:
        return
    stamps = [t for t, _ in marks]
    for bar in bars:
        i = bisect.bisect_right(stamps, bar.minute) - 1
        if i >= 0:
            bar.open_interest = marks[i][1]


# ------------------------------------------------------------------ features


@dataclass(frozen=True, slots=True)
class Features:
    """One row of the study. Every field is knowable at `minute`."""

    minute: int
    price: float  # perp last — what a trade would be priced against

    perp_ret_1m: float = 0.0
    perp_ret_5m: float = 0.0
    spot_ret_1m: float = 0.0
    spot_ret_5m: float = 0.0

    perp_buy_ratio: float = 0.5
    spot_buy_ratio: float = 0.5

    basis_bps: float = 0.0
    basis_z: float = 0.0
    futures_lead_bps: float = 0.0
    # Thresholds on the raw value have to be guessed, and a guessed absolute
    # is how a condition ends up never firing — spot and perp do not diverge
    # by whole basis points in a minute on BTC. The z-score says "unusual for
    # this market lately", which is what the condition actually means.
    futures_lead_z: float = 0.0

    volume_z: float = 0.0
    trade_count_z: float = 0.0
    volatility_z: float = 0.0

    oi_change_1m: float = 0.0
    oi_change_5m: float = 0.0
    oi_change_z: float = 0.0

    # Reserved for a live capture recording; the archive cannot supply these.
    book_imbalance_5: float = field(default=math.nan)
    ask_cancel_imbalance: float = field(default=math.nan)
    bid_cancel_imbalance: float = field(default=math.nan)
    microprice_deviation: float = field(default=math.nan)


def _z(values: list[float], i: int, window: int) -> float:
    """Trailing z-score of `values[i]` against the preceding `window` entries.

    Returns 0 rather than a huge number when the window has no spread — a
    constant stretch is not an extreme reading, and dividing by a near-zero
    standard deviation manufactures signals out of quiet periods.
    """
    lo = max(0, i - window)
    sample = values[lo:i]
    n = len(sample)
    if n < window // 2:
        return 0.0
    mean = sum(sample) / n
    var = sum((v - mean) ** 2 for v in sample) / n
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    return (values[i] - mean) / sd


def build(bars: list[MinuteBar], *, z_window: int = 1440) -> list[Features]:
    """Feature rows, one per minute, with a trailing window for z-scores.

    The default window is a day of minutes: long enough that "unusual volume"
    means unusual for this market rather than for the last hour, short enough
    to follow a regime instead of averaging over all of them.
    """
    if not bars:
        return []

    perp = [b.perp_last for b in bars]
    spot = [b.spot_last for b in bars]
    volume = [b.perp_volume for b in bars]
    counts = [float(b.perp_trades) for b in bars]
    basis = [b.basis_bps for b in bars]
    oi = [b.open_interest for b in bars]

    def ret(series: list[float], i: int, back: int) -> float:
        j = i - back
        if j < 0 or series[j] <= 0:
            return 0.0
        return (series[i] - series[j]) / series[j] * 10_000.0

    # Per-minute absolute move, as the raw material for the volatility score.
    moves = [0.0] + [abs(ret(perp, i, 1)) for i in range(1, len(bars))]
    leads = [ret(perp, i, 1) - ret(spot, i, 1) for i in range(len(bars))]
    oi_moves = [_pct_change(oi, i, 5) for i in range(len(bars))]

    out: list[Features] = []
    for i, bar in enumerate(bars):
        perp_1m = ret(perp, i, 1)
        spot_1m = ret(spot, i, 1)
        out.append(
            Features(
                minute=bar.minute,
                price=bar.perp_last,
                perp_ret_1m=perp_1m,
                perp_ret_5m=ret(perp, i, 5),
                spot_ret_1m=spot_1m,
                spot_ret_5m=ret(spot, i, 5),
                perp_buy_ratio=bar.perp_buy_ratio,
                spot_buy_ratio=bar.spot_buy_ratio,
                basis_bps=bar.basis_bps,
                basis_z=_z(basis, i, z_window),
                # Not a lead-lag estimate — just how much further the perp
                # moved this minute. A real lead needs sub-second alignment,
                # which is what a live capture is for.
                futures_lead_bps=perp_1m - spot_1m,
                futures_lead_z=_z(leads, i, z_window),
                volume_z=_z(volume, i, z_window),
                trade_count_z=_z(counts, i, z_window),
                volatility_z=_z(moves, i, z_window),
                oi_change_1m=_pct_change(oi, i, 1),
                oi_change_5m=_pct_change(oi, i, 5),
                oi_change_z=_z(oi_moves, i, z_window),
            )
        )
    return out


def _pct_change(series: list[float], i: int, back: int) -> float:
    j = i - back
    if j < 0 or series[j] <= 0:
        return 0.0
    return (series[i] - series[j]) / series[j] * 100.0
