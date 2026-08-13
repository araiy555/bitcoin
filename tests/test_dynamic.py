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
