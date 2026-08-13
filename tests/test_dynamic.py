"""Dynamic screen: adverse selection over 100ms against the spread.

The sign convention is the whole measurement. Reading the aggressor backwards
turns a market that runs a maker over into one that pays it, so the direction
cases matter more than the arithmetic.
"""

import math

import pytest

from jsboard.feed.multi import Print, Quote, _parse
from jsboard.research.dynamic import SymbolProbe, rank, tally

MS = 1_000_000


def probe(**kw):
    return SymbolProbe("XUSDT", **kw)


def quote(p, bid, ask, ms):
    p.on_book(bid, ask, ms * MS)


class TestSign:
    """An aggressive buy means the maker sold, and vice versa."""

    def test_a_taker_buy_that_keeps_running_hurts_the_maker(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)
        p.on_trade(aggressor_sign=+1, ts_ns=0)  # we sold
        quote(p, 100.9, 101.1, 200)  # price ran up 100 bps
        assert p.markout_bps == pytest.approx(-100.0, rel=1e-3)
        assert p.ratio > 0

    def test_a_taker_sell_that_keeps_falling_hurts_the_maker(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)
        p.on_trade(aggressor_sign=-1, ts_ns=0)  # we bought
        quote(p, 98.9, 99.1, 200)
        assert p.markout_bps == pytest.approx(-100.0, rel=1e-3)

    def test_a_reverting_market_pays_the_maker(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)
        p.on_trade(aggressor_sign=+1, ts_ns=0)  # we sold
        quote(p, 98.9, 99.1, 200)  # price fell back
        assert p.markout_bps == pytest.approx(+100.0, rel=1e-3)
        assert p.ratio < 0


class TestRatio:
    def test_the_two_symbols_this_project_measured(self):
        # USUSDT: 12.48 bps of adverse selection against a 9.53 bps spread.
        p = probe()
        p.spread_samples.append(9.53)
        p.markout.windows[0].weight = 1.0
        p.markout.windows[0].weighted_vs_mid = -12.48
        assert p.ratio == pytest.approx(1.31, abs=0.01)

    def test_a_gentle_market_scores_well_below_one(self):
        p = probe()
        p.spread_samples.append(7.37)
        p.markout.windows[0].weight = 1.0
        p.markout.windows[0].weighted_vs_mid = -2.26
        assert p.ratio == pytest.approx(0.31, abs=0.01)

    def test_no_spread_yet_is_not_a_ratio(self):
        assert math.isnan(probe().ratio)

    def test_the_spread_is_a_median_not_a_last_look(self):
        p = probe()
        for bid, ask in ((99.9, 100.1), (99.99, 100.01), (99.95, 100.05)):
            quote(p, bid, ask, 0)
        assert p.spread_bps == pytest.approx(10.0, rel=1e-3)


class TestVerdict:
    def loaded(self, spread, markout, n):
        p = probe()
        p.spread_samples.append(spread)
        w = p.markout.windows[0]
        w.weight, w.weighted_vs_mid, w.n = float(n), markout * n, n
        return p

    def test_a_market_faster_than_its_spread_is_impossible(self):
        assert self.loaded(9.53, -12.48, 5000).verdict() == "不可"

    def test_a_ratio_under_one_but_over_a_half_is_not_enough(self):
        # Nothing left for fees, inventory, or the fill model's optimism.
        assert self.loaded(10.0, -7.0, 5000).verdict() == "見込み薄"

    def test_a_quiet_market_is_worth_recording(self):
        assert self.loaded(10.0, -2.0, 5000).verdict() == "研究候補"

    def test_too_few_prints_decides_nothing(self):
        assert self.loaded(10.0, -2.0, 10).verdict() == "サンプル不足"

    def test_a_market_that_pays_the_maker_is_a_candidate(self):
        assert self.loaded(10.0, +1.0, 5000).verdict() == "研究候補"


class TestRankAndTally:
    def loaded(self, symbol, spread, markout, n):
        p = SymbolProbe(symbol)
        p.spread_samples.append(spread)
        w = p.markout.windows[0]
        w.weight, w.weighted_vs_mid, w.n = float(n), markout * n, n
        return p

    def market(self):
        return [
            self.loaded("BADUSDT", 9.53, -12.48, 5000),
            self.loaded("GOODUSDT", 10.0, -1.0, 5000),
            self.loaded("MEHUSDT", 10.0, -7.0, 5000),
            self.loaded("THINUSDT", 10.0, -0.1, 5),
        ]

    def test_best_ratio_first_and_unmeasurable_last(self):
        got = rank(self.market())
        assert [p.symbol for p in got][:3] == ["GOODUSDT", "MEHUSDT", "BADUSDT"]
        assert got[-1].symbol == "THINUSDT"

    def test_the_tally_separates_impossible_from_unmeasured(self):
        counts = tally(self.market())
        assert counts == {"研究候補": 1, "見込み薄": 1, "不可": 1, "サンプル不足": 1}


class TestParsing:
    def test_a_book_ticker_becomes_a_quote(self):
        got = _parse({"data": {"e": "bookTicker", "s": "AUSDT", "b": "1.5", "a": "1.6", "T": 5}})
        assert got == Quote("AUSDT", 1.5, 1.6, 5 * MS)

    def test_maker_buyer_means_the_aggressor_sold(self):
        got = _parse({"data": {"e": "aggTrade", "s": "AUSDT", "m": True, "T": 7}})
        assert got == Print("AUSDT", -1, 7 * MS)

    def test_taker_buyer_means_the_aggressor_bought(self):
        got = _parse({"data": {"e": "aggTrade", "s": "AUSDT", "m": False, "T": 7}})
        assert got.aggressor_sign == +1

    def test_an_unwrapped_payload_parses_too(self):
        assert _parse({"e": "aggTrade", "s": "AUSDT", "m": False, "T": 1}) is not None

    def test_anything_else_is_ignored(self):
        assert _parse({"result": None, "id": 1}) is None
        assert _parse({"data": {"e": "markPrice"}}) is None

    def test_a_malformed_quote_is_dropped_rather_than_guessed(self):
        assert _parse({"data": {"e": "bookTicker", "s": "A", "b": "x", "a": "1"}}) is None


class TestClock:
    def test_a_late_stamp_does_not_rewind_the_settlement_clock(self):
        p = probe()
        quote(p, 99.9, 100.1, 1000)
        p.on_trade(+1, 1000 * MS)
        quote(p, 99.9, 100.1, 500)  # arrives out of order
        assert p.last_ns == 1000 * MS


class TestSelectionBias:
    """Which prints count changed the answer by a factor of five."""

    def loaded(self, spread=10.0):
        p = probe()
        p.spread_samples.append(spread)
        return p

    def test_a_sweep_outweighs_a_shower_of_dust(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)
        # 100 harmless one-lot prints, then one sweep of 1,000 that runs.
        for i in range(100):
            p.on_trade(+1, i * MS, qty=1.0)
        quote(p, 99.9, 100.1, 150)  # nothing moved for those
        p.on_trade(+1, 150 * MS, qty=1000.0)
        quote(p, 100.9, 101.1, 400)  # the sweep ran the price up
        # Size weighting lets the sweep dominate, as it should.
        assert p.markout_bps < -50

    def test_only_prints_that_clear_the_touch_count_as_sweeps(self):
        p = probe()
        p.on_book(99.9, 100.1, 0, bid_qty=500, ask_qty=500)
        p.on_trade(+1, 0, qty=100.0)  # too small to reach a queued maker
        assert p.sweeps == 0
        p.on_trade(+1, MS, qty=600.0)  # clears the offer
        assert p.sweeps == 1

    def test_the_side_that_matters_is_the_one_being_hit(self):
        p = probe()
        p.on_book(99.9, 100.1, 0, bid_qty=10, ask_qty=1000)
        p.on_trade(-1, 0, qty=50.0)  # sells into a thin bid: clears it
        assert p.sweeps == 1
        p.on_trade(+1, MS, qty=50.0)  # buys into a deep offer: does not
        assert p.sweeps == 1

    def test_an_unknown_touch_size_is_not_treated_as_a_sweep(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)  # no quantities carried
        p.on_trade(+1, 0, qty=1e9)
        assert p.sweeps == 0

    def test_the_verdict_uses_the_sweeps_once_there_are_enough(self):
        p = self.loaded()
        for tracker, mo, n in (
            (p.markout, -1.0, 5000),  # optimistic: ratio 0.10
            (p.sweep_markout, -9.0, 500),  # realistic: ratio 0.90
        ):
            w = tracker.windows[0]
            w.weight, w.weighted_vs_mid, w.n = float(n), mo * n, n
        assert p.ratio == pytest.approx(0.10)
        assert p.sweep_ratio == pytest.approx(0.90)
        assert p.decisive_ratio() == pytest.approx(0.90)
        assert p.verdict() == "見込み薄"

    def test_too_few_sweeps_falls_back_to_every_print(self):
        p = self.loaded()
        w = p.markout.windows[0]
        w.weight, w.weighted_vs_mid, w.n = 5000.0, -1.0 * 5000, 5000
        sw = p.sweep_markout.windows[0]
        sw.weight, sw.weighted_vs_mid, sw.n = 3.0, -9.0 * 3, 3
        assert p.decisive_ratio() == pytest.approx(0.10)
        assert p.verdict() == "研究候補"

    def test_a_zero_size_print_is_ignored(self):
        p = probe()
        quote(p, 99.9, 100.1, 0)
        p.on_trade(+1, 0, qty=0.0)
        assert p.trades == 0
