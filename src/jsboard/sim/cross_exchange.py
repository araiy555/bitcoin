"""Causal two-market mean-reversion with executable two-leg prices.

The signal is the rolling z-score of the Binance/Bybit log-mid spread.  The
trade, however, is never priced at mid: both entry legs and both exit legs
walk visible depth and pay taker fees.  This separates a statistical anomaly
from an opportunity that an account could actually execute.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field

from ..core.market import MarketView
from ..core.pretrade import (
    BookWalk,
    CostBreakdown,
    MarketDataPoint,
    RejectCode,
    assess_costs,
    assess_market_data,
    walk_book,
)
from ..core.types import Instrument, Side
from ..feed.base import FeedEvent, MarkPrice


@dataclass(frozen=True, slots=True)
class CrossArbConfig:
    size_base: float = 0.001
    lookback_s: float = 3600.0
    min_samples: int = 300
    entry_z: float = 2.5
    exit_z: float = 0.25
    max_hold_s: float = 1800.0
    min_expected_net_bps: float = 1.0
    max_age_ms: float = 250.0
    max_skew_ms: float = 250.0
    depth: int = 20
    hedge_latency_buffer_bps: float = 0.0
    fill_model_buffer_bps: float = 0.0
    safety_margin_bps: float = 0.0
    taker_bps: dict[str, float] = field(
        default_factory=lambda: {"binance": 4.0, "bybit": 5.5}
    )
    allowed_directions: tuple[tuple[str, str], ...] | None = None


@dataclass(slots=True)
class CrossArbPosition:
    long_source: str
    short_source: str
    qty_base: float
    opened_ns: int
    entry_z: float
    long_entry: float
    short_entry: float
    entry_fees: float
    expected_net_bps: float
    funding_quote: float = 0.0
    funding_settlements: int = 0


@dataclass(frozen=True, slots=True)
class CrossArbTrade:
    long_source: str
    short_source: str
    opened_ns: int
    closed_ns: int
    entry_z: float
    exit_z: float
    expected_net_bps: float
    gross_quote: float
    funding_quote: float
    fees_quote: float
    net_quote: float
    net_bps: float
    exit_reason: str

    @property
    def hold_s(self) -> float:
        return (self.closed_ns - self.opened_ns) / 1e9


@dataclass(slots=True)
class CrossArbStats:
    observations: int = 0
    fresh: int = 0
    warm: int = 0
    signals: int = 0
    rejected_depth: int = 0
    rejected_cost: int = 0
    rejected_quality: int = 0
    rejected_direction: int = 0
    entries: int = 0
    forced_exits: int = 0
    reject_codes: dict[str, int] = field(default_factory=dict)


class CrossExchangeArb:
    """One-position paper replay of any two comparable markets."""

    def __init__(self, instruments: dict[str, Instrument], config: CrossArbConfig) -> None:
        if len(instruments) != 2:
            raise ValueError("cross arb requires exactly two instruments")
        self.sources = tuple(instruments)
        if config.size_base <= 0 or config.lookback_s <= 0 or config.min_samples < 2:
            raise ValueError("size, lookback and min_samples must be positive")
        if config.entry_z <= config.exit_z or config.exit_z < 0:
            raise ValueError("entry_z must be greater than non-negative exit_z")
        if config.max_hold_s <= 0 or config.min_expected_net_bps < 0:
            raise ValueError("max_hold must be positive and expected edge non-negative")
        if config.max_age_ms < 0 or config.max_skew_ms < 0 or config.depth <= 0:
            raise ValueError("age/skew must be non-negative and depth positive")
        if any(
            value < 0
            for value in (
                config.hedge_latency_buffer_bps,
                config.fill_model_buffer_bps,
                config.safety_margin_bps,
            )
        ):
            raise ValueError("execution buffers must be non-negative")
        if any(config.taker_bps.get(source, -1) < 0 for source in self.sources):
            raise ValueError("both taker fees must be non-negative")
        if config.allowed_directions is not None:
            valid = set(self.sources)
            if any(
                long_source not in valid
                or short_source not in valid
                or long_source == short_source
                for long_source, short_source in config.allowed_directions
            ):
                raise ValueError("allowed directions must reference the two configured sources")

        self.instruments = instruments
        self.config = config
        self.now_ns = 0
        self.markets = {
            source: MarketView(instrument=instrument, depth=config.depth, clock=self._clock)
            for source, instrument in instruments.items()
        }
        self.history: deque[tuple[int, float]] = deque()
        self.position: CrossArbPosition | None = None
        self.marks: dict[str, MarkPrice] = {}
        self.funding_seen: set[tuple[str, int, int]] = set()
        self.trades: list[CrossArbTrade] = []
        self.stats = CrossArbStats()
        self.last_z = float("nan")

    def _clock(self) -> int:
        return self.now_ns

    def apply(self, source: str, event: FeedEvent, received_ns: int = 0) -> None:
        if source not in self.markets:
            return
        self.now_ns = max(self.now_ns, received_ns, getattr(event, "ts_ns", 0))
        if isinstance(event, MarkPrice):
            self._apply_funding(source, event)
            self.marks[source] = event
        self.markets[source].apply(event)

    def _apply_funding(self, source: str, current: MarkPrice) -> None:
        """Accrue the last observable predicted rate when its settlement passes."""
        position = self.position
        previous = self.marks.get(source)
        if position is None or previous is None or previous.next_funding_ns <= 0:
            return
        if not (position.opened_ns < previous.next_funding_ns <= current.ts_ns):
            return
        if source not in (position.long_source, position.short_source):
            return
        settlement = (source, previous.next_funding_ns, position.opened_ns)
        if settlement in self.funding_seen:
            return
        self.funding_seen.add(settlement)
        sign = 1 if source == position.long_source else -1
        mark_price = previous.mark * float(self.instruments[source].tick_size)
        position.funding_quote += (
            -sign * previous.funding_rate * mark_price * position.qty_base
        )
        position.funding_settlements += 1

    def _mid_price(self, source: str) -> float | None:
        mid = self.markets[source].mid
        if mid is None:
            return None
        return mid * float(self.instruments[source].tick_size)

    def _fresh(self) -> bool:
        points = tuple(
            MarketDataPoint(
                source=source,
                updated_ns=self.markets[source].last_update_ns,
                book_valid=(
                    self.markets[source].mid is not None
                    and self.markets[source].spread_ticks is not None
                    and self.markets[source].spread_ticks >= 0
                ),
                feed_state=self.markets[source].status,
                metadata_valid=bool(
                    self.instruments[source].base and self.instruments[source].quote
                ),
            )
            for source in self.sources
        )
        decision = assess_market_data(
            points,
            now_ns=self.now_ns,
            max_age_ms=self.config.max_age_ms,
            max_skew_ms=self.config.max_skew_ms,
        )
        if not decision.allowed:
            self.stats.rejected_quality += 1
            self._record_rejects(decision.reject_codes)
        return decision.allowed

    def _record_rejects(self, codes: tuple[RejectCode, ...]) -> None:
        for code in codes:
            self.stats.reject_codes[code.value] = self.stats.reject_codes.get(code.value, 0) + 1

    def _common_qty(self) -> float:
        quantities = [
            inst.qty_f(inst.to_lots(self.config.size_base)) for inst in self.instruments.values()
        ]
        return min(quantities)

    def _walk(self, source: str, sign: int, qty_base: float) -> BookWalk | None:
        result = walk_book(
            self.markets[source].snapshot(self.config.depth),
            self.instruments[source],
            Side.BUY if sign > 0 else Side.SELL,
            qty_base,
            max_levels=self.config.depth,
        )
        if not result.complete:
            return None
        return result

    def _human_price(self, source: str, result: BookWalk) -> float:
        return result.price(self.instruments[source])

    def _round_trip_costs(
        self, long_source: str, short_source: str, qty: float
    ) -> CostBreakdown | None:
        long_buy = self._walk(long_source, +1, qty)
        long_sell = self._walk(long_source, -1, qty)
        short_sell = self._walk(short_source, -1, qty)
        short_buy = self._walk(short_source, +1, qty)
        if None in (long_buy, long_sell, short_sell, short_buy):
            return None
        prices = {
            "long_buy": self._human_price(long_source, long_buy),
            "long_sell": self._human_price(long_source, long_sell),
            "short_sell": self._human_price(short_source, short_sell),
            "short_buy": self._human_price(short_source, short_buy),
        }
        reference = qty * (prices["long_buy"] + prices["short_sell"]) / 2.0
        if reference <= 0:
            return None
        crossing = qty * (
            prices["long_buy"] - prices["long_sell"]
            + prices["short_buy"] - prices["short_sell"]
        )
        entry_fees = qty * (
            prices["long_buy"] * self.config.taker_bps[long_source]
            + prices["short_sell"] * self.config.taker_bps[short_source]
        ) / 10_000.0
        exit_fees = qty * (
            prices["long_sell"] * self.config.taker_bps[long_source]
            + prices["short_buy"] * self.config.taker_bps[short_source]
        ) / 10_000.0
        scale = 10_000.0 / reference
        crossing_bps = crossing * scale
        return CostBreakdown(
            entry_fees_bps=entry_fees * scale,
            expected_exit_fees_bps=exit_fees * scale,
            entry_depth_slippage_bps=crossing_bps / 2.0,
            expected_exit_slippage_bps=crossing_bps / 2.0,
            funding_bps=self._expected_funding_cost_bps(long_source, short_source, qty, reference),
            hedge_latency_buffer_bps=self.config.hedge_latency_buffer_bps,
            fill_model_buffer_bps=self.config.fill_model_buffer_bps,
            safety_margin_bps=self.config.safety_margin_bps,
        )

    def _expected_funding_cost_bps(
        self,
        long_source: str,
        short_source: str,
        qty: float,
        reference: float,
    ) -> float:
        """Return signed funding cost visible at entry within the hold horizon.

        Positive is a cost and negative is a receipt.  Spot sources simply do
        not have a ``MarkPrice`` and therefore contribute zero.
        """
        if reference <= 0:
            return 0.0
        cost_quote = 0.0
        horizon_ns = self.now_ns + int(self.config.max_hold_s * 1e9)
        for source, sign in ((long_source, 1), (short_source, -1)):
            mark = self.marks.get(source)
            if mark is None or not (self.now_ns < mark.next_funding_ns <= horizon_ns):
                continue
            price = mark.mark * float(self.instruments[source].tick_size)
            cost_quote += sign * mark.funding_rate * price * qty
        return cost_quote / reference * 10_000.0

    def _open(self, z: float, mean: float, spread: float) -> None:
        self.stats.signals += 1
        source_a, source_b = self.sources
        long_source, short_source = (
            (source_b, source_a) if spread > mean else (source_a, source_b)
        )
        if (
            self.config.allowed_directions is not None
            and (long_source, short_source) not in self.config.allowed_directions
        ):
            self.stats.rejected_direction += 1
            self._record_rejects((RejectCode.BORROW_UNAVAILABLE,))
            return
        qty = self._common_qty()
        if qty <= 0:
            self.stats.rejected_depth += 1
            self._record_rejects((RejectCode.NO_DEPTH,))
            return
        costs = self._round_trip_costs(long_source, short_source, qty)
        if costs is None:
            self.stats.rejected_depth += 1
            self._record_rejects((RejectCode.NO_DEPTH,))
            return
        convergence_bps = abs(spread - mean) * 10_000.0
        decision = assess_costs(
            convergence_bps,
            costs,
            min_expected_net_bps=self.config.min_expected_net_bps,
        )
        if not decision.accepted:
            self.stats.rejected_cost += 1
            self._record_rejects(decision.reject_codes)
            return

        long_fill = self._walk(long_source, +1, qty)
        short_fill = self._walk(short_source, -1, qty)
        if long_fill is None or short_fill is None:
            self.stats.rejected_depth += 1
            self._record_rejects((RejectCode.NO_DEPTH,))
            return
        long_price = self._human_price(long_source, long_fill)
        short_price = self._human_price(short_source, short_fill)
        entry_fees = qty * (
            long_price * self.config.taker_bps[long_source]
            + short_price * self.config.taker_bps[short_source]
        ) / 10_000.0
        self.position = CrossArbPosition(
            long_source=long_source,
            short_source=short_source,
            qty_base=qty,
            opened_ns=self.now_ns,
            entry_z=z,
            long_entry=long_price,
            short_entry=short_price,
            entry_fees=entry_fees,
            expected_net_bps=decision.expected_net_bps,
        )
        self.stats.entries += 1

    def _close(self, z: float, reason: str) -> bool:
        position = self.position
        if position is None:
            return False
        long_exit = self._walk(position.long_source, -1, position.qty_base)
        short_exit = self._walk(position.short_source, +1, position.qty_base)
        if long_exit is None or short_exit is None:
            self.stats.rejected_depth += 1
            self._record_rejects((RejectCode.NO_DEPTH,))
            return False
        long_price = self._human_price(position.long_source, long_exit)
        short_price = self._human_price(position.short_source, short_exit)
        exit_fees = position.qty_base * (
            long_price * self.config.taker_bps[position.long_source]
            + short_price * self.config.taker_bps[position.short_source]
        ) / 10_000.0
        gross = position.qty_base * (
            long_price - position.long_entry + position.short_entry - short_price
        )
        fees = position.entry_fees + exit_fees
        reference = position.qty_base * (position.long_entry + position.short_entry) / 2.0
        net = gross + position.funding_quote - fees
        self.trades.append(
            CrossArbTrade(
                long_source=position.long_source,
                short_source=position.short_source,
                opened_ns=position.opened_ns,
                closed_ns=self.now_ns,
                entry_z=position.entry_z,
                exit_z=z,
                expected_net_bps=position.expected_net_bps,
                gross_quote=gross,
                funding_quote=position.funding_quote,
                fees_quote=fees,
                net_quote=net,
                net_bps=net / reference * 10_000.0 if reference > 0 else float("nan"),
                exit_reason=reason,
            )
        )
        if reason == "max hold":
            self.stats.forced_exits += 1
        self.position = None
        return True

    def evaluate(self) -> float | None:
        self.stats.observations += 1
        if not self._fresh():
            return None
        mids = {source: self._mid_price(source) for source in self.sources}
        if any(price is None or price <= 0 for price in mids.values()):
            return None
        self.stats.fresh += 1
        source_a, source_b = self.sources
        spread = math.log(mids[source_a]) - math.log(mids[source_b])
        cutoff = self.now_ns - int(self.config.lookback_s * 1e9)
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        warm = (
            len(self.history) >= self.config.min_samples
            and self.history
            and self.now_ns - self.history[0][0] >= int(self.config.lookback_s * 0.8e9)
        )
        z = None
        mean = float("nan")
        if warm:
            values = [value for _, value in self.history]
            mean = statistics.fmean(values)
            sigma = statistics.stdev(values)
            if sigma > 0:
                z = (spread - mean) / sigma
                self.last_z = z
                self.stats.warm += 1

        self.history.append((self.now_ns, spread))
        if z is None:
            return None
        if self.position is not None:
            held_s = (self.now_ns - self.position.opened_ns) / 1e9
            if abs(z) <= self.config.exit_z:
                self._close(z, "mean")
            elif held_s >= self.config.max_hold_s:
                self._close(z, "max hold")
        elif abs(z) >= self.config.entry_z:
            self._open(z, mean, spread)
        return z

    def finalize(self) -> bool:
        """Close any remaining paper position at the final observable books."""
        return self._close(self.last_z, "end of capture") if self.position else False

    def summary(self) -> dict[str, float]:
        net_values = [trade.net_quote for trade in self.trades]
        bps_values = [trade.net_bps for trade in self.trades if math.isfinite(trade.net_bps)]
        equity = peak = drawdown = 0.0
        for value in net_values:
            equity += value
            peak = max(peak, equity)
            drawdown = min(drawdown, equity - peak)
        wins = sum(value > 0 for value in net_values)
        return {
            "trades": float(len(self.trades)),
            "wins": float(wins),
            "win_rate": wins / len(net_values) if net_values else 0.0,
            "gross_quote": sum(trade.gross_quote for trade in self.trades),
            "funding_quote": sum(trade.funding_quote for trade in self.trades),
            "fees_quote": sum(trade.fees_quote for trade in self.trades),
            "net_quote": sum(net_values),
            "net_bps": sum(bps_values),
            "median_bps": statistics.median(bps_values) if bps_values else float("nan"),
            "max_drawdown_quote": drawdown,
        }
