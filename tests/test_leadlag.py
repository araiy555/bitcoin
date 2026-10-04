"""Follow a jump in the lead market on GMO's real book."""

from decimal import Decimal

from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot
from jsboard.research.leadlag import NS, analyse

GMO = Instrument("XRP_JPY", Decimal("0.001"), Decimal("1"), "XRP", "JPY")
LEAD = Instrument("XRPUSDT", Decimal("0.0001"), Decimal("1"), "XRP", "USDT")
T0 = 1_790_380_800_000_000_000


def book(bid, ask, t):
    return DepthSnapshot(((bid, 1_000),), ((ask, 1_000),), 1, t)


def tape(gmo_follows_at=None, gmo_jumps_with_lead=False):
    rows = []
    for i in range(200):  # 100ms steps for 20s
        t = T0 + i * NS // 10
        jumped = i >= 100
        lead = (10_020, 10_022) if jumped else (10_000, 10_002)
        follow = (gmo_jumps_with_lead and jumped) or (
            gmo_follows_at is not None and i >= gmo_follows_at)
        gmo = (100_200, 100_220) if follow else (100_000, 100_020)
        # When both move at once GMO's update is seen first: nothing to follow.
        rows += [("gmo", t, book(*gmo, t)), ("lead", t, book(*lead, t))]
    return rows


def run(rows):
    return analyse(iter(rows), GMO, LEAD, label="xrp", thresholds=[10], holds=[5],
                   size_jpy=10_000)


def test_buying_before_gmo_follows_pays_the_move_less_the_spread():
    r = run(tape(gmo_follows_at=105))  # GMO catches up half a second later
    [cell] = r.cells.values()
    assert cell.n == 1 and cell.pnl > 0
    # bought at the old ask 100.02, sold at the new bid 100.20
    assert abs(cell.bps() - (100.20 - 100.02) / 100.02 * 1e4) < 0.01


def test_no_trade_when_gmo_moved_with_the_lead():
    r = run(tape(gmo_jumps_with_lead=True))
    assert not r.cells


def test_a_lead_that_gmo_never_follows_costs_the_spread():
    [cell] = run(tape()).cells.values()
    assert cell.pnl < 0 and cell.wins == 0
