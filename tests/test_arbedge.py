"""Trading bitbank against GMO only when their prices part by more than it costs."""

from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot
from jsboard.research.arbedge import NS, analyse

BB = Instrument("xrp_jpy", Decimal("0.001"), Decimal("0.0001"), "XRP", "JPY")
GM = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
T0 = 1_790_380_800_000_000_000


def book(bid, ask, t, qty=1_000_000_000):
    return DepthSnapshot(((bid, qty),), ((ask, qty),), 1, t)


def run(rows, **kw):
    kw.setdefault("thresholds", [10])
    return analyse(iter(rows), BB, GM, label="XRP", size_jpy=10_000, bb_fee_bps=12.0,
                   gm_fee_bps=0.0, gm_min=1.0, **kw)


def test_a_gap_wider_than_the_costs_opens_and_its_reversal_closes():
    rows = [
        ("bitbank", T0, book(99_900, 100_000, T0)),
        ("gmo", T0 + 1, book(100_300, 100_320, T0 + 1)),     # GMO 30bps dearer: buy bb, sell GMO
        ("gmo", T0 + 10 * NS, book(99_980, 100_000, T0 + 10 * NS)),
        ("bitbank", T0 + 11 * NS, book(100_100, 100_200, T0 + 11 * NS)),  # gap reversed
    ]
    r = run(rows)
    [leg] = [v for v in r.days.values() if v.n]
    assert leg.n == 1 and leg.wins == 1 and leg.pnl > 0 and leg.forced == 0
    # opened at +18bps (30 less 12 in fees), closed at -2bps (10 less 12)
    assert abs(leg.bps() - 16.0) < 0.2


def test_no_trade_when_the_venues_agree():
    rows = [("bitbank", T0, book(99_990, 100_010, T0)),
            ("gmo", T0 + 1, book(99_995, 100_005, T0 + 1))]
    r = run(rows)
    assert not any(v.n for v in r.days.values())
    assert max(r.best_gap_bps.values()) < 0


def test_a_gap_that_never_reverses_is_closed_when_time_runs_out():
    rows = [("bitbank", T0, book(99_900, 100_000, T0)),
            ("gmo", T0 + 1, book(100_300, 100_320, T0 + 1)),
            ("bitbank", T0 + 2 * 3600 * NS, book(99_900, 100_000, T0 + 2 * 3600 * NS))]
    r = run(rows, max_hold_s=3600)
    [leg] = [v for v in r.days.values() if v.n]
    assert leg.forced == 1 and leg.n == 1
