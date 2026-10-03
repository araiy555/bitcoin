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


def test_a_print_inside_a_wide_spread_pays_the_maker():
    from decimal import Decimal

    from jsboard.core.types import Instrument, Side
    from jsboard.feed.base import DepthSnapshot, TradeTick
    from jsboard.research.printedge import collect, report

    inst = Instrument("ada_jpy", Decimal("0.001"), Decimal("0.0001"), "ADA", "JPY")
    lead = Instrument("ADAUSDT", Decimal("0.0001"), Decimal("1"), "ADA", "USDT")
    rows = []
    for i in range(120):
        rows.append(("bitbank", i * NS, DepthSnapshot(((36_990, 10**7),), ((37_010, 10**7),), i)))
        if i % 10 == 0:  # sellers hit the bid; the mid never moves
            rows.append(("bitbank", i * NS + 1, TradeTick(36_990, 10**6, Side.SELL, i)))
    prints, mids, spreads = collect(rows, inst, lead)
    assert len(prints) == 12 and all(p.sign == 1 for p in prints)
    text = report(prints, mids, spreads, 10_000)
    row = next(line for line in text.splitlines() if line.startswith("すべての約定"))
    edge, *after = (float(x) for x in row.split("\t")[3:])
    assert edge > 2.5 and all(a > 4 for a in after)  # half spread 2.7bps + 2bps rebate


def test_holding_measures_how_much_and_how_long():
    from jsboard.research.fillcheck import holding

    fills = [FillRow(10 * NS, 1, 40.0, 100), FillRow(70 * NS, -1, 40.0, 100)]
    avg, biggest, share, longest = holding(fills, 0, 100 * NS)
    assert biggest == 100 and abs(avg - 60.0) < 1e-9
    assert abs(share - 0.6) < 1e-9 and abs(longest - 1.0) < 1e-9


def test_the_listing_interleaves_both_runs_with_their_inventory():
    from jsboard.research.fillcheck import listing

    mids = line((0, 40.0), (200, 39.0))
    live = [FillRow(10 * NS, 1, 40.0, 100), FillRow(100 * NS, -1, 39.5, 100)]
    sim = [FillRow(20 * NS, 1, 40.0, 50)]
    text = listing(live, sim, mids, 0, 150 * NS, 200 * NS)
    rows = text.splitlines()[2:]
    assert [r.split("\t")[1] for r in rows] == ["本番", "検証", "本番"]
    assert rows[-1].endswith("+0") and rows[1].endswith("+50")


def test_a_single_run_splits_into_quick_and_held_parts():
    from jsboard.research.fillcheck import single

    mids = line((0, 40.0), (100, 39.0), (300, 39.0))
    text = single([FillRow(10 * NS, 1, 40.0, 100)], mids, 0, 300 * NS)
    assert "この日の値動き(始め→終わり)\t-2.50%" in text
    assert "終わりの在庫(枚)\t+100" in text and "損益(円)\t-99" in text and "60秒より長く持った分(円)\t-100" in text
