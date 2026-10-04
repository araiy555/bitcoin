"""The dislocation cap means the same thing on every book."""

from decimal import Decimal

from jsboard.core.market import MarketView
from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot, FeedStatus
from jsboard.mm.inventory import FeeSchedule, Position
from jsboard.mm.risk import RiskAction, RiskManager


def decide(inst, bid, ask):
    now = 1_000_000_000
    view = MarketView(instrument=inst, clock=lambda: now)
    view.apply(FeedStatus("live"))
    view.apply(DepthSnapshot(tuple((bid - j, 10**6) for j in range(3)),
                             tuple((ask + j, 10**6) for j in range(3)), 1, now))
    return RiskManager().evaluate(view, Position(instrument=inst, fees=FeeSchedule(maker_bps=0.0)))


def test_a_coin_with_fine_ticks_is_quoted_at_a_normal_spread():
    oas = Instrument("oas_jpy", Decimal("0.0001"), Decimal("0.1"), "OAS", "JPY")
    # 4 yen, spread 0.07 yen = 700 ticks but 180bps: thin, not broken.
    assert decide(oas, 39_650, 40_350).action is not RiskAction.PULL


def test_a_broken_book_is_still_refused():
    oas = Instrument("oas_jpy", Decimal("0.0001"), Decimal("0.1"), "OAS", "JPY")
    d = decide(oas, 38_000, 42_000)  # 10% wide
    assert d.action is RiskAction.PULL and "dislocated" in d.reason
