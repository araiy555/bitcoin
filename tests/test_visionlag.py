import io
import zipfile

from jsboard.research.visionlag import (
    Scan,
    Signal,
    book_tops,
    evaluate,
    lead_prints,
    report,
    signals,
    tape_tops,
)


def _zip(tmp_path, name, text):
    path = tmp_path / f"{name}.zip"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{name}.csv", text)
    path.write_bytes(buf.getvalue())
    return str(path)


def test_reads_with_and_without_header(tmp_path):
    with_header = _zip(tmp_path, "a", (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
        "1,100.0,1,1,1,1000,true\n2,101.0,1,2,2,1100,false\n"))
    bare = _zip(tmp_path, "b", "1,100.0,1,1,1,1000,true\n2,101.0,1,2,2,1100,false\n")
    assert list(lead_prints(with_header)) == [(1000, 100.0), (1100, 101.0)]
    assert list(lead_prints(bare)) == [(1000, 100.0), (1100, 101.0)]
    assert list(tape_tops(bare))[0][1:4:2] == (100.0, 101.0)


def test_book_tops_in_header_order(tmp_path):
    path = _zip(tmp_path, "c", (
        "update_id,best_bid_price,best_bid_qty,best_ask_price,best_ask_qty,transaction_time,event_time\n"
        "1,10.0,5,10.1,6,1000,1001\n2,10.2,5,10.1,6,1100,1101\n"))
    assert list(book_tops(path)) == [(1000, 10.0, 5.0, 10.1, 6.0)]  # crossed row dropped


def test_signal_needs_the_move_within_the_window():
    prints = [(0, 100.0), (50, 100.0), (100, 100.06), (150, 100.0), (1300, 100.0), (1400, 99.9)]
    sigs = signals(prints, [5], window_ms=100, cooldown_ms=1000)
    assert [(s.ms, s.sign) for s in sigs] == [(100, 1), (1400, -1)]


def test_follower_that_lags_pays_and_one_that_led_shows_before():
    sig = [Signal(1000, 1, 5, 6.0)]
    # Follower jumps 20 bps 300ms after the lead; spread 1 bp.
    tops = [(0, 99.995, 100, 100.005, 100), (1300, 100.195, 100, 100.205, 100), (20000, 100.195, 100, 100.205, 100)]
    scan = Scan()
    evaluate(iter(tops), sig, scan, follower="X", day="d", latency_ms=50, window_ms=100,
             fee_bps=5, size_usd=1000)
    fast = scan.cells[("X", "d", 5, 100)]
    slow = scan.cells[("X", "d", 5, 1000)]
    assert fast.n == 1 and fast.net_bps < 0  # nothing moved yet: pays spread and fees
    assert slow.same == 1 and slow.net_bps > 5  # caught the move after costs
    assert slow.depth_ok == 1 and slow.before_bps == 0
    scan.sources[("X", "d")] = "book"
    every, best = report(scan, ["X"], [5], ["d"], min_signals=1)
    assert len(every) == 7 and best[0].split("\t")[2] == "0.5秒"


def test_structure_lists_the_mid_path_by_horizon():
    from jsboard.research.visionlag import structure

    sig = [Signal(1000, 1, 5, 6.0)]
    tops = [(0, 99.995, 100, 100.005, 100), (1300, 100.195, 100, 100.205, 100),
            (20000, 100.195, 100, 100.205, 100)]
    scan = Scan()
    evaluate(iter(tops), sig, scan, follower="X", day="d")
    scan.sources[("X", "d")] = "book"
    lines = structure(scan, ["X"], 5, ["d"])
    cols = lines[1].split("\t")
    assert cols[0] == "X" and cols[3] == "+0.0" and cols[-2] == "+20.0"
