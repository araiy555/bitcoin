"""Score a condition by net money, not by how often it points the right way.

Accuracy picked the wrong signal last time. A rule with a target and a stop
can be right less than half the time and still make money, or right most of
the time and lose it, so the thing being maximised has to be the thing that
matters:

    net = exit − entry − fees − slippage        (signed by direction)

Three traps decide whether a number here means anything.

**Overlapping trades are not independent observations.** Evaluating a
condition every minute and holding an hour produces sixty readings of one
hour. Three hundred such rows look like a sample and are closer to five.
Every entry here therefore blocks new entries until it closes.

**A handful of outliers can carry a whole result.** One violent hour inside
a fortnight can turn a losing rule profitable on the average, so the summary
carries the result with the best ten trades removed alongside the headline.

**A rule chosen on data it is then scored on is not tested.** This module
computes statistics; it does not choose. Selecting a condition on one period
and reporting it on another stays the caller's job, as it was in `predict`.

The acceptance thresholds encoded in `Verdict` are deliberately harder than
"profitable": at least a hundred non-overlapping trades, a profit factor of
1.3, a positive median, survival with the top ten removed, and survival at
1.5× the assumed cost.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass

from .archive import SecondBar
from .features import Features

Condition = Callable[[Features], bool]


@dataclass(frozen=True, slots=True)
class Trade:
    entry_minute: int
    exit_minute: int
    entry_price: float
    exit_price: float
    direction: int  # +1 long, -1 short
    cost_bps: float
    reason: str  # "target" | "stop" | "timeout"

    @property
    def gross_bps(self) -> float:
        return (self.exit_price - self.entry_price) / self.entry_price * 10_000.0 * self.direction

    @property
    def net_bps(self) -> float:
        return self.gross_bps - self.cost_bps


def _price_path(seconds: list[SecondBar], start: int, end: int) -> list[SecondBar]:
    """Second bars inside [start, end], for deciding which barrier came first.

    Minute bars cannot answer that question: a minute whose high cleared the
    target and whose low cleared the stop is ambiguous, and resolving it
    optimistically is the classic way to invent a profitable backtest.
    """
    stamps = [b.sec for b in seconds]
    lo = bisect.bisect_left(stamps, start)
    hi = bisect.bisect_right(stamps, end)
    return seconds[lo:hi]


def simulate(
    features: list[Features],
    seconds: list[SecondBar],
    condition: Condition,
    *,
    direction: int,
    horizon_min: int,
    cost_bps: float,
    target_bps: float | None = None,
    stop_bps: float | None = None,
) -> list[Trade]:
    """Every non-overlapping trade the condition would have taken.

    With a target and a stop, the exit is whichever the second-by-second path
    reaches first, and the horizon becomes a timeout. Without them the exit is
    the horizon, full stop.
    """
    if direction not in (1, -1):
        raise ValueError("direction must be +1 or -1")
    if horizon_min <= 0:
        raise ValueError("horizon must be positive")

    if not features or not seconds:
        return []

    # The last instant the data can speak about. An entry whose horizon runs
    # past it has no future to be scored against, and taking the final
    # available print instead would silently shorten the hold — turning a
    # 30-minute rule into a 2-minute one at the end of every file.
    last_second = seconds[-1].sec

    trades: list[Trade] = []
    blocked_until = -1

    for f in features:
        if f.minute < blocked_until or f.price <= 0:
            continue
        deadline = f.minute + horizon_min * 60
        if deadline > last_second:
            break  # every later entry runs past the data too
        if not condition(f):
            continue

        path = _price_path(seconds, f.minute + 1, deadline)
        exit_price, exit_at, reason = _resolve(
            f.price, direction, path, target_bps, stop_bps
        )
        if exit_price is None:
            continue

        trades.append(
            Trade(
                entry_minute=f.minute,
                exit_minute=exit_at,
                entry_price=f.price,
                exit_price=exit_price,
                direction=direction,
                cost_bps=cost_bps,
                reason=reason,
            )
        )
        blocked_until = exit_at

    return trades


def _resolve(
    entry: float,
    direction: int,
    path: list[SecondBar],
    target_bps: float | None,
    stop_bps: float | None,
) -> tuple[float | None, int, str]:
    """Walk the path second by second and take the first barrier touched.

    Within one second, a bar that touched both barriers is resolved as a
    stop. The true order is unknowable at this resolution, and assuming the
    favourable one is how a backtest flatters itself.
    """
    if not path:
        return None, 0, "timeout"
    if target_bps is None or stop_bps is None:
        # No barriers: the horizon is the only exit.
        return path[-1].last, path[-1].sec, "timeout"

    # Above and below entry, named by where they sit rather than by role,
    # because which one is the target flips with direction.
    above = entry * (1 + (target_bps if direction > 0 else stop_bps) / 10_000.0)
    below = entry * (1 - (stop_bps if direction > 0 else target_bps) / 10_000.0)

    for bar in path:
        hit_below = bar.low <= below
        hit_above = bar.high >= above
        # A second that touched both is resolved against us. The true order
        # is unknowable here, and taking the favourable one invents an edge.
        if direction > 0:
            if hit_below:
                return below, bar.sec, "stop"
            if hit_above:
                return above, bar.sec, "target"
        else:
            if hit_above:
                return above, bar.sec, "stop"
            if hit_below:
                return below, bar.sec, "target"
    return path[-1].last, path[-1].sec, "timeout"


# ------------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class Verdict:
    name: str
    trades: int
    mean_net_bps: float
    median_net_bps: float
    win_rate: float
    profit_factor: float
    max_drawdown_bps: float
    total_net_bps: float
    mean_without_top10_bps: float
    mean_at_15x_cost_bps: float
    months_positive: int
    months_total: int

    @property
    def passes(self) -> bool:
        """Every gate, not the headline alone.

        A rule that clears the average but fails with its ten best trades
        removed is a rule about ten trades.
        """
        return (
            self.trades >= 100
            and self.profit_factor >= 1.3
            and self.mean_net_bps >= 5.0
            and self.median_net_bps > 0
            and self.mean_without_top10_bps > 0
            and self.mean_at_15x_cost_bps > 0
            and self.months_total > 0
            and self.months_positive == self.months_total
        )

    def failures(self) -> list[str]:
        out = []
        if self.trades < 100:
            out.append(f"取引{self.trades}件<100")
        if self.profit_factor < 1.3:
            out.append(f"PF{self.profit_factor:.2f}<1.3")
        if self.mean_net_bps < 5.0:
            out.append(f"平均{self.mean_net_bps:+.1f}bps<5")
        if self.median_net_bps <= 0:
            out.append("中央値≤0")
        if self.mean_without_top10_bps <= 0:
            out.append("上位10件除くと赤")
        if self.mean_at_15x_cost_bps <= 0:
            out.append("手数料1.5倍で赤")
        if self.months_total and self.months_positive < self.months_total:
            out.append(f"黒字は{self.months_positive}/{self.months_total}期間")
        return out


def _month_of(minute: int) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(minute, _dt.UTC).strftime("%Y-%m")


def score(name: str, trades: list[Trade]) -> Verdict:
    if not trades:
        return Verdict(name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    nets = [t.net_bps for t in trades]
    wins = [n for n in nets if n > 0]
    losses = [-n for n in nets if n < 0]
    ordered = sorted(nets)

    # Equity is in bps of a constant stake, so drawdown is comparable across
    # price levels rather than being dominated by whatever BTC cost that week.
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for n in nets:
        equity += n
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)

    trimmed = sorted(nets)[: max(0, len(nets) - 10)]
    extra = [t.net_bps - t.cost_bps * 0.5 for t in trades]  # 1.5x the cost

    months: dict[str, float] = {}
    for t in trades:
        months[_month_of(t.entry_minute)] = months.get(_month_of(t.entry_minute), 0.0) + t.net_bps

    return Verdict(
        name=name,
        trades=len(trades),
        mean_net_bps=sum(nets) / len(nets),
        median_net_bps=ordered[len(ordered) // 2],
        win_rate=len(wins) / len(nets),
        profit_factor=(sum(wins) / sum(losses)) if losses else float("inf"),
        max_drawdown_bps=drawdown,
        total_net_bps=sum(nets),
        mean_without_top10_bps=(sum(trimmed) / len(trimmed)) if trimmed else 0.0,
        mean_at_15x_cost_bps=sum(extra) / len(extra),
        months_positive=sum(1 for v in months.values() if v > 0),
        months_total=len(months),
    )


# ---------------------------------------------------------------- conditions


def threshold(field_name: str, op: str, value: float) -> Condition:
    """A single comparison against one feature.

    NaN is treated as "not satisfied" rather than raising, so a condition
    naming a book feature simply never fires on archive-only data instead of
    crashing the run.
    """
    def check(f: Features) -> bool:
        got = getattr(f, field_name)
        if got != got:  # NaN
            return False
        return got > value if op == ">" else got < value

    return check


def combine(*conditions: Condition) -> Condition:
    return lambda f: all(c(f) for c in conditions)
