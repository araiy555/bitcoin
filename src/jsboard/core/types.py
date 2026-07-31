"""Core domain types.

Prices and quantities are held as *integers* everywhere inside the engine:
prices in ticks, quantities in lots. Floating point never touches the book,
so price levels compare and hash exactly. `Instrument` is the only place
that converts to and from human units.
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal


class Side(enum.IntEnum):
    BUY = 1
    SELL = -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for buy, -1 for sell. Handy for inventory arithmetic."""
        return int(self.value)


class TimeInForce(enum.Enum):
    GTC = "GTC"
    """Rests on the book until filled or cancelled."""

    IOC = "IOC"
    """Fills what it can immediately, cancels the rest."""

    FOK = "FOK"
    """Fills completely and immediately, or not at all."""

    POST_ONLY = "POST_ONLY"
    """Rejected outright if it would take liquidity."""


class OrderStatus(enum.Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


MARKET = None
"""Sentinel price meaning "market order" — crosses whatever is there."""


@dataclass(frozen=True, slots=True)
class Instrument:
    """Contract spec plus the tick/lot conversions.

    tick_size and lot_size are Decimals so that the conversion to integer
    ticks is exact for the usual exchange values (0.01, 0.00001, ...).
    """

    symbol: str
    tick_size: Decimal
    lot_size: Decimal
    base: str = ""
    quote: str = ""

    def to_ticks(self, price: float | Decimal | str) -> int:
        """Human price -> integer ticks (round half up to nearest tick)."""
        return int((Decimal(str(price)) / self.tick_size).quantize(Decimal(1), rounding=ROUND_HALF_UP))

    def to_lots(self, qty: float | Decimal | str) -> int:
        """Human size -> integer lots (round *down*; never invent quantity)."""
        return int((Decimal(str(qty)) / self.lot_size).quantize(Decimal(1), rounding=ROUND_DOWN))

    def to_price(self, ticks: int) -> Decimal:
        return Decimal(ticks) * self.tick_size

    def to_qty(self, lots: int) -> Decimal:
        return Decimal(lots) * self.lot_size

    def price_f(self, ticks: int) -> float:
        return float(self.to_price(ticks))

    def qty_f(self, lots: int) -> float:
        return float(self.to_qty(lots))

    def notional(self, ticks: int, lots: int) -> float:
        return self.price_f(ticks) * self.qty_f(lots)


_order_ids = itertools.count(1)


def next_order_id() -> int:
    return next(_order_ids)


@dataclass(slots=True)
class Order:
    """A single resting or in-flight order.

    `price` is in ticks, or None for a market order. `qty` and `remaining`
    are in lots. `seq` is the engine-assigned arrival sequence and is what
    gives us time priority within a price level.
    """

    side: Side
    qty: int
    price: int | None = MARKET
    tif: TimeInForce = TimeInForce.GTC
    owner: str = "anon"
    order_id: int = field(default_factory=next_order_id)
    seq: int = 0
    ts_ns: int = 0
    remaining: int = -1
    status: OrderStatus = OrderStatus.NEW

    def __post_init__(self) -> None:
        if self.remaining < 0:
            self.remaining = self.qty
        if self.qty <= 0:
            raise ValueError(f"order qty must be positive, got {self.qty}")
        if self.price is MARKET and self.tif in (TimeInForce.GTC, TimeInForce.POST_ONLY):
            raise ValueError(f"market order cannot be {self.tif.name}; use IOC or FOK")

    @property
    def filled(self) -> int:
        return self.qty - self.remaining

    @property
    def is_market(self) -> bool:
        return self.price is MARKET

    @property
    def is_live(self) -> bool:
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED) and self.remaining > 0


@dataclass(frozen=True, slots=True)
class Fill:
    """One maker/taker match. Price is always the *maker's* price."""

    price: int
    qty: int
    maker_id: int
    taker_id: int
    maker_owner: str
    taker_owner: str
    aggressor: Side
    ts_ns: int
    seq: int = 0

    def signed_qty_for(self, owner: str) -> int:
        """Inventory delta this fill produces for `owner` (in lots)."""
        delta = 0
        if self.maker_owner == owner:
            # The maker takes the side opposite the aggressor.
            delta += self.aggressor.opposite.sign * self.qty
        if self.taker_owner == owner:
            delta += self.aggressor.sign * self.qty
        return delta


@dataclass(frozen=True, slots=True)
class Level:
    """One aggregated price level in an L2 view."""

    price: int
    qty: int
    orders: int = 0


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """Top-N L2 view. `bids` descend from best, `asks` ascend from best."""

    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    ts_ns: int = 0
    seq: int = 0

    @property
    def best_bid(self) -> int | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> int | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid(self) -> float | None:
        if not self.bids or not self.asks:
            return None
        return (self.bids[0].price + self.asks[0].price) / 2.0

    @property
    def spread(self) -> int | None:
        if not self.bids or not self.asks:
            return None
        return self.asks[0].price - self.bids[0].price

    @property
    def microprice(self) -> float | None:
        """Size-weighted mid — leans toward the side with *less* size.

        The standard formulation: a big bid means buyers are queued up and
        the next trade is more likely to lift the ask, so the fair value
        sits closer to the ask.
        """
        if not self.bids or not self.asks:
            return None
        bq, aq = self.bids[0].qty, self.asks[0].qty
        total = bq + aq
        if total == 0:
            return self.mid
        return (self.bids[0].price * aq + self.asks[0].price * bq) / total

    def imbalance(self, depth: int = 1) -> float:
        """Order-book imbalance in [-1, 1]; +1 = all bid, -1 = all ask."""
        bq = sum(lvl.qty for lvl in self.bids[:depth])
        aq = sum(lvl.qty for lvl in self.asks[:depth])
        total = bq + aq
        if total == 0:
            return 0.0
        return (bq - aq) / total
