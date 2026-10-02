"""Taking live and replayed fills apart the same way."""

from jsboard.research.fillcheck import NS, FillRow, MidLine, breakdown, from_trade, report, slices


def line(*points):
    m = MidLine()
    for t, mid in points:
        m.add(t * NS, mid)
    return m


def test_a_round_trip_splits_into_edge_and_carry():
    mids = line((0, 100.0), (30, 98.0), (100, 98.0))
    fills = [FillRow(0, 1, 99.9, 10), FillRow(50 * NS, -1, 98.1, 10)]  # bought, price fell, sold
    b = breakdown(fills, mids, 100 * NS)
    assert round(b.edge, 6) == round(0.1 * 10 + 0.1 * 10, 6)  # half spread both times
    assert round(b.move_60s, 6) == -20.0  # the buy's mid fell 2 yen within the minute
    cash = -99.9 * 10 + 98.1 * 10
    assert abs(b.pnl - (cash - b.fees)) < 1e-9
    assert abs(b.edge + b.carry - b.fees - b.pnl) < 1e-9


def test_slices_add_up_to_the_whole():
    mids = line((0, 100.0), (900, 101.0), (2000, 99.0))
    fills = [FillRow(10 * NS, 1, 99.9, 5), FillRow(2100 * NS, -1, 99.1, 2)]
    parts = slices(fills, mids, 0, 3600 * NS)
    assert len(parts) == 2
    assert abs(sum(p for _, p in parts) - breakdown(fills, mids, 3600 * NS).pnl) < 1e-9


def test_trades_parse_and_report_prints_both_sides():
    t = from_trade({"executed_at": 1000, "side": "sell", "price": "37.1", "amount": "5",
                    "maker_taker": "taker"})
    assert t.sign == -1 and not t.maker and t.fee > 0
    text = report([t], [], line((0, 37.0)), 0, 10 * NS)
    assert "本番\t検証" in text and "30分ごとの損益" in text
