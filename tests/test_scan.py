"""Symbol scan: the spread-versus-fee gate, and the filters guarding it."""

import pytest

from jsboard.research.scan import (
    ScanFilters,
    SymbolStats,
    apply_filters,
    merge_day_tickers,
    parse_book_tickers,
    rank,
    summarise,
)


def book_row(symbol, bid, ask, bid_qty=10.0, ask_qty=10.0):
    return {
        "symbol": symbol,
        "bidPrice": str(bid),
        "askPrice": str(ask),
        "bidQty": str(bid_qty),
        "askQty": str(ask_qty),
    }


def day_row(symbol, volume, trades):
    return {"symbol": symbol, "quoteVolume": str(volume), "count": trades}


def stats(symbol="XUSDT", bid=100.0, ask=101.0, volume=5_000_000.0, trades=50_000, **kw):
    s = SymbolStats(symbol=symbol, bid=bid, ask=ask, bid_qty=kw.pop("bid_qty", 10.0),
                    ask_qty=kw.pop("ask_qty", 10.0))
    s.quote_volume = volume
    s.trades = trades
    return s


class TestSpreadArithmetic:
    def test_spread_is_measured_against_the_mid(self):
        s = stats(bid=99.0, ask=101.0)  # 2 wide on a 100 mid

        assert s.spread_bps == pytest.approx(200.0)

    def test_a_one_tick_spread_on_a_large_price_is_tiny(self):
        # The BTCUSDT case: $0.01 on $64,000.
        s = stats(bid=64_000.00, ask=64_000.01)

        assert s.spread_bps == pytest.approx(0.0016, abs=1e-4)

    def test_fee_is_charged_on_both_legs(self):
        s = stats(bid=99.0, ask=101.0)  # 200bps of spread

        assert s.net_bps(maker_bps=10.0) == pytest.approx(180.0)
        assert s.net_bps(maker_bps=100.0) == pytest.approx(0.0)
        assert s.net_bps(maker_bps=150.0) == pytest.approx(-100.0)

    def test_btcusdt_loses_money_at_a_retail_fee(self):
        s = stats(bid=64_000.00, ask=64_000.02)

        assert s.net_bps(maker_bps=10.0) < -19.0

    def test_profit_per_round_trip_scales_with_size(self):
        s = stats(bid=99.0, ask=101.0)

        assert s.profit_per_round_trip(10.0, 1_000.0) == pytest.approx(18.0)
        assert s.profit_per_round_trip(10.0, 10_000.0) == pytest.approx(180.0)

    def test_one_sided_book_has_no_spread(self):
        assert stats(bid=0.0, ask=101.0).spread_bps == 0.0
        assert not stats(bid=0.0, ask=101.0).is_two_sided

    def test_crossed_book_is_rejected(self):
        assert not stats(bid=102.0, ask=101.0).is_two_sided

    def test_top_of_book_is_the_thinner_side(self):
        s = stats(bid=100.0, ask=101.0, bid_qty=1.0, ask_qty=50.0)

        # 1 * 100 vs 50 * 101 -> the bid side is the binding constraint.
        assert s.top_of_book_quote == pytest.approx(100.0)


class TestParsing:
    def test_book_tickers_become_stats(self):
        book = parse_book_tickers([book_row("AAAUSDT", 1.0, 1.01), book_row("BBBUSDT", 2.0, 2.02)])

        assert set(book) == {"AAAUSDT", "BBBUSDT"}
        assert book["AAAUSDT"].ask == 1.01

    def test_malformed_rows_are_skipped(self):
        book = parse_book_tickers(
            [book_row("GOODUSDT", 1.0, 1.01), {"symbol": "BADUSDT"}, {"nonsense": 1}]
        )

        assert set(book) == {"GOODUSDT"}

    def test_day_tickers_merge_by_symbol(self):
        book = parse_book_tickers([book_row("AAAUSDT", 1.0, 1.01)])
        merge_day_tickers(book, [day_row("AAAUSDT", 5_000_000, 42_000)])

        assert book["AAAUSDT"].quote_volume == 5_000_000
        assert book["AAAUSDT"].trades == 42_000

    def test_unknown_symbols_in_day_data_are_ignored(self):
        book = parse_book_tickers([book_row("AAAUSDT", 1.0, 1.01)])
        merge_day_tickers(book, [day_row("ZZZUSDT", 1, 1)])

        assert book["AAAUSDT"].quote_volume == 0.0


class TestFilters:
    def _market(self):
        return {
            s.symbol: s
            for s in [
                stats("LIQUIDUSDT", 100.0, 101.0),
                stats("THINUSDT", 100.0, 110.0, volume=100.0, trades=5),
                stats("QUIETUSDT", 100.0, 110.0, volume=9_000_000.0, trades=10),
                stats("BTCBUSD", 100.0, 101.0),
                stats("ETHUPUSDT", 100.0, 120.0),
                stats("BROKENUSDT", 0.0, 101.0),
            ]
        }

    def test_quote_asset_is_enforced(self):
        kept = apply_filters(self._market(), ScanFilters())

        assert all(s.symbol.endswith("USDT") for s in kept)
        assert "BTCBUSD" not in [s.symbol for s in kept]

    def test_illiquid_symbols_are_dropped(self):
        kept = [s.symbol for s in apply_filters(self._market(), ScanFilters())]

        assert "THINUSDT" not in kept

    def test_quiet_symbols_are_dropped_even_when_liquid(self):
        """A wide spread nobody trades against pays nothing."""
        kept = [s.symbol for s in apply_filters(self._market(), ScanFilters())]

        assert "QUIETUSDT" not in kept

    def test_leveraged_tokens_are_excluded_by_default(self):
        kept = [s.symbol for s in apply_filters(self._market(), ScanFilters())]

        assert "ETHUPUSDT" not in kept

    def test_leveraged_tokens_can_be_kept(self):
        f = ScanFilters(exclude_leveraged=False)
        kept = [s.symbol for s in apply_filters(self._market(), f)]

        assert "ETHUPUSDT" in kept

    def test_one_sided_books_are_dropped(self):
        kept = [s.symbol for s in apply_filters(self._market(), ScanFilters())]

        assert "BROKENUSDT" not in kept


class TestRankingAndSummary:
    def test_ranking_is_by_net_edge(self):
        market = [
            stats("NARROWUSDT", 100.0, 100.1),
            stats("WIDEUSDT", 100.0, 105.0),
            stats("MIDUSDT", 100.0, 101.0),
        ]

        ordered = [s.symbol for s in rank(market, maker_bps=10.0)]

        assert ordered == ["WIDEUSDT", "MIDUSDT", "NARROWUSDT"]

    def test_summary_counts_only_symbols_that_clear_the_fee(self):
        market = rank(
            [stats("WIDEUSDT", 100.0, 105.0), stats("NARROWUSDT", 100.0, 100.1)],
            maker_bps=10.0,
        )

        s = summarise(market, ScanFilters(maker_bps=10.0))

        assert s["liquid"] == 2
        assert s["viable"] == 1
        assert s["breakeven_spread_bps"] == 20.0

    def test_nothing_is_viable_at_a_high_fee(self):
        market = rank([stats("WIDEUSDT", 100.0, 105.0)], maker_bps=500.0)

        assert summarise(market, ScanFilters(maker_bps=500.0))["viable"] == 0

    def test_median_spread_is_reported(self):
        market = [
            stats("AUSDT", 100.0, 100.1),  # 10bps
            stats("BUSDT", 100.0, 100.2),  # 20bps
            stats("CUSDT", 100.0, 100.3),  # 30bps
        ]

        s = summarise(market, ScanFilters())

        assert s["median_spread_bps"] == pytest.approx(19.98, abs=0.05)

    def test_summary_survives_an_empty_result(self):
        s = summarise([], ScanFilters())

        assert s["viable"] == 0
        assert s["median_spread_bps"] == 0.0


class TestCapacity:
    """Clearing the fee is necessary and nowhere near sufficient."""

    def test_a_long_queue_is_flagged(self):
        # PEPEUSDT shape: $128k resting ahead of a $1,000 order.
        s = stats(bid=100.0, ask=100.4, bid_qty=1_280.0, ask_qty=1_280.0)

        assert s.queue_ratio(1_000.0) == pytest.approx(128.0)
        assert s.capacity_verdict(1_000.0) == "行列が長い"

    def test_a_book_thinner_than_our_order_is_flagged(self):
        # AEVOUSDT shape: $117 at the touch, nowhere to put $1,000.
        s = stats(bid=100.0, ask=100.4, bid_qty=1.17, ask_qty=1.17)

        assert s.queue_ratio(1_000.0) < 1.0
        assert s.capacity_verdict(1_000.0) == "板が薄い"

    def test_a_workable_depth_passes(self):
        s = stats(bid=100.0, ask=100.4, bid_qty=50.0, ask_qty=50.0)

        assert s.capacity_verdict(1_000.0) == "可"

    def test_shrinking_the_order_escapes_the_thin_book(self):
        s = stats(bid=100.0, ask=100.4, bid_qty=1.17, ask_qty=1.17)

        assert s.capacity_verdict(1_000.0) == "板が薄い"
        assert s.capacity_verdict(20.0, thin=0.1) == "可"

    def test_summary_separates_the_two_failure_modes(self):
        market = rank(
            [
                stats("DEEPUSDT", 100.0, 100.4, bid_qty=1_280.0, ask_qty=1_280.0),
                stats("THINUSDT", 100.0, 100.4, bid_qty=1.17, ask_qty=1.17),
                stats("OKUSDT", 100.0, 100.4, bid_qty=50.0, ask_qty=50.0),
            ],
            maker_bps=10.0,
        )

        s = summarise(market, ScanFilters(maker_bps=10.0, size_quote=1_000.0))

        assert s["viable"] == 3
        assert s["tradeable"] == 1
        assert s["too_deep"] == 1
        assert s["too_thin"] == 1

    def test_symbols_below_the_fee_are_not_counted_as_tradeable(self):
        market = rank([stats("NARROWUSDT", 100.0, 100.01, bid_qty=50.0, ask_qty=50.0)], 10.0)

        s = summarise(market, ScanFilters(maker_bps=10.0))

        assert s["viable"] == 0
        assert s["tradeable"] == 0


class TestSampling:
    """One snapshot of a thin book proves nothing; medians are the point."""

    def _sampled(self, spreads, depths):
        s = stats(bid=100.0, ask=101.0)
        s.spread_samples = list(spreads)
        s.depth_samples = list(depths)
        return s

    def test_median_spread_is_used_when_sampled(self):
        s = self._sampled([10.0, 20.0, 30.0, 40.0, 100.0], [1_000.0] * 5)

        assert s.spread_bps == pytest.approx(30.0)

    def test_median_depth_is_used_when_sampled(self):
        # The exact swing observed live: SHIBUSDT 20,723 -> 4,841.
        s = self._sampled([20.0] * 3, [20_723.0, 4_841.0, 9_000.0])

        assert s.top_of_book_quote == pytest.approx(9_000.0)

    def test_a_single_outlier_no_longer_decides_the_verdict(self):
        """One unlucky poll used to flip a symbol between 可 and 板が薄い."""
        s = self._sampled([20.0] * 5, [8_000.0, 9_000.0, 200.0, 8_500.0, 9_500.0])

        assert s.capacity_verdict(1_000.0) == "可"

    def test_unsampled_stats_fall_back_to_the_live_touch(self):
        s = stats(bid=100.0, ask=101.0, bid_qty=10.0, ask_qty=10.0)

        assert s.spread_bps == pytest.approx(99.5, abs=0.5)
        assert s.top_of_book_quote == pytest.approx(1_000.0)

    def test_swing_reports_how_far_the_touch_moved(self):
        s = self._sampled([20.0] * 3, [100.0, 500.0, 1_000.0])

        assert s.depth_swing == pytest.approx(10.0)

    def test_a_steady_book_reports_no_swing(self):
        s = self._sampled([20.0] * 3, [1_000.0] * 3)

        assert s.depth_swing == pytest.approx(1.0)

    def test_swing_needs_two_samples_to_mean_anything(self):
        assert self._sampled([20.0], [1_000.0]).depth_swing == 1.0
        assert stats().depth_swing == 1.0

    def test_zero_depth_samples_do_not_blow_up_the_swing(self):
        s = self._sampled([20.0] * 3, [0.0, 500.0, 1_000.0])

        assert s.depth_swing == pytest.approx(2.0)

    def test_add_sample_accumulates_across_polls(self):
        from jsboard.research.scan import add_sample

        book = parse_book_tickers([book_row("AAAUSDT", 100.0, 101.0, 10.0, 10.0)])
        add_sample(book, [book_row("AAAUSDT", 100.0, 101.0, 10.0, 10.0)])
        add_sample(book, [book_row("AAAUSDT", 100.0, 101.0, 50.0, 50.0)])

        assert book["AAAUSDT"].samples_taken == 2
        assert book["AAAUSDT"].depth_swing == pytest.approx(5.0)

    def test_one_sided_observations_are_not_sampled(self):
        from jsboard.research.scan import add_sample

        book = parse_book_tickers([book_row("AAAUSDT", 100.0, 101.0)])
        add_sample(book, [book_row("AAAUSDT", 0.0, 101.0)])

        assert book["AAAUSDT"].samples_taken == 0

    def test_summary_counts_unstable_symbols(self):
        steady = self._sampled([200.0] * 3, [10_000.0] * 3)
        steady.symbol = "STEADYUSDT"
        jumpy = self._sampled([200.0] * 3, [1_000.0, 9_000.0, 5_000.0])
        jumpy.symbol = "JUMPYUSDT"

        s = summarise(rank([steady, jumpy], 10.0), ScanFilters(maker_bps=10.0))

        assert s["unstable"] == 1


class TestPersistence:
    """One scan answers "right now", which turned out not to be the question."""

    def _round(self, symbol, spread_bps, depth_quote):
        s = stats(symbol, 100.0, 100.0 * (1 + spread_bps / 10_000))
        s.bid_qty = s.ask_qty = depth_quote / 100.0
        return s

    def test_a_symbol_that_always_qualifies_scores_full_persistence(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        for i in range(5):
            fold_round(tally, [self._round("STEADYUSDT", 40.0, 20_000.0)], f)
            for e in tally.values():
                e.total_rounds = i + 1

        assert tally["STEADYUSDT"].persistence == 1.0
        assert tally["STEADYUSDT"].tradeable_rounds == 5

    def test_a_symbol_that_comes_and_goes_scores_partially(self):
        """The live case: BONKUSDT's touch fell to 0.15x and the verdict flipped."""
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        for i, depth in enumerate((20_000.0, 20_000.0, 500.0, 500.0)):
            fold_round(tally, [self._round("BONKUSDT", 36.0, depth)], f)
            for e in tally.values():
                e.total_rounds = i + 1

        entry = tally["BONKUSDT"]
        assert entry.rounds == 4
        assert entry.viable_rounds == 4  # the spread held up
        assert entry.tradeable_rounds == 2  # the depth did not
        assert entry.persistence == 0.5

    def test_clearing_the_fee_is_counted_separately_from_being_placeable(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        # Wide enough to clear the fee, always far too thin to use.
        for i in range(3):
            fold_round(tally, [self._round("THINUSDT", 40.0, 200.0)], f)
            for e in tally.values():
                e.total_rounds = i + 1

        assert tally["THINUSDT"].viable_rounds == 3
        assert tally["THINUSDT"].tradeable_rounds == 0
        assert tally["THINUSDT"].persistence == 0.0

    def test_symbols_absent_from_a_round_do_not_gain_credit(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        fold_round(tally, [self._round("AUSDT", 40.0, 20_000.0)], f)
        fold_round(tally, [self._round("BUSDT", 40.0, 20_000.0)], f)

        assert tally["AUSDT"].rounds == 1
        assert tally["BUSDT"].rounds == 1

    def test_ranking_puts_the_reliable_symbol_first(self):
        from jsboard.research.scan import Persistence, fold_round, rank_persistence

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        for _ in range(4):
            fold_round(tally, [self._round("STEADYUSDT", 30.0, 20_000.0)], f)
        for depth in (20_000.0, 200.0, 200.0, 200.0):
            # A bigger edge, but almost never usable.
            fold_round(tally, [self._round("FLAKYUSDT", 90.0, depth)], f)

        ordered = [p.symbol for p in rank_persistence(tally)]

        assert ordered[0] == "STEADYUSDT"

    def test_swing_spans_the_whole_watch(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        for depth in (15_793.0, 2_354.0):  # the observed BONKUSDT collapse
            fold_round(tally, [self._round("BONKUSDT", 36.0, depth)], f)

        assert tally["BONKUSDT"].depth_swing == pytest.approx(6.71, abs=0.05)

    async def test_watch_runs_the_requested_rounds(self, monkeypatch):
        import jsboard.research.scan as scan_mod

        calls = []

        async def fake_scan(filters, on_sample=None):
            calls.append(1)
            return [self._round("AUSDT", 40.0, 20_000.0)], 1

        monkeypatch.setattr(scan_mod, "scan", fake_scan)

        tally = await scan_mod.watch(
            ScanFilters(maker_bps=10.0), rounds=3, every_seconds=0.0
        )

        assert len(calls) == 3
        assert tally["AUSDT"].rounds == 3

    def test_a_symbol_that_vanishes_is_not_credited_for_the_rounds_it_missed(self):
        """The live failure: half the surviving list turned over in 30 minutes."""
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=10.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        rounds = [
            [self._round("STEADYUSDT", 30.0, 20_000.0)],
            [self._round("STEADYUSDT", 30.0, 20_000.0), self._round("LATEUSDT", 30.0, 20_000.0)],
            [self._round("STEADYUSDT", 30.0, 20_000.0), self._round("LATEUSDT", 30.0, 20_000.0)],
            [self._round("STEADYUSDT", 30.0, 20_000.0)],
        ]
        for i, results in enumerate(rounds):
            fold_round(tally, results, f)
            for e in tally.values():
                e.total_rounds = i + 1

        # LATEUSDT qualified whenever it appeared, but only appeared half the time.
        assert tally["LATEUSDT"].tradeable_rounds == 2
        assert tally["LATEUSDT"].presence == 0.5
        assert tally["LATEUSDT"].persistence == 0.5
        assert tally["STEADYUSDT"].persistence == 1.0

    async def test_watch_scores_against_the_whole_run(self, monkeypatch):
        import jsboard.research.scan as scan_mod

        rounds = iter([
            [self._round("AUSDT", 40.0, 20_000.0), self._round("BUSDT", 40.0, 20_000.0)],
            [self._round("AUSDT", 40.0, 20_000.0)],
            [self._round("AUSDT", 40.0, 20_000.0)],
        ])

        async def fake_scan(filters, on_sample=None):
            return next(rounds), 1

        monkeypatch.setattr(scan_mod, "scan", fake_scan)
        tally = await scan_mod.watch(ScanFilters(maker_bps=10.0), rounds=3, every_seconds=0.0)

        assert tally["AUSDT"].persistence == 1.0
        assert tally["BUSDT"].persistence == pytest.approx(1 / 3)

    def test_breakeven_fee_is_half_the_spread(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=0.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        # ASTERUSDT's observed shape: 16.65bps of spread at a 0bps fee.
        for i in range(3):
            fold_round(tally, [self._round("ASTERUSDT", 16.65, 20_000.0)], f)
            for e in tally.values():
                e.total_rounds = i + 1

        assert tally["ASTERUSDT"].breakeven_fee_bps(0.0) == pytest.approx(8.32, abs=0.02)

    def test_breakeven_is_independent_of_the_fee_it_was_scanned_at(self):
        """The same symbol must report the same ceiling from either run."""
        from jsboard.research.scan import Persistence, fold_round

        results = {}
        for scanned_at in (0.0, 5.0, 10.0):
            f = ScanFilters(maker_bps=scanned_at, size_quote=1_000.0)
            tally: dict[str, Persistence] = {}
            fold_round(tally, [self._round("XUSDT", 16.65, 20_000.0)], f)
            tally["XUSDT"].total_rounds = 1
            results[scanned_at] = tally["XUSDT"].breakeven_fee_bps(scanned_at)

        assert results[0.0] == pytest.approx(results[5.0], abs=0.01)
        assert results[0.0] == pytest.approx(results[10.0], abs=0.01)

    def test_a_one_tick_major_has_essentially_no_ceiling(self):
        from jsboard.research.scan import Persistence, fold_round

        f = ScanFilters(maker_bps=0.0, size_quote=1_000.0)
        tally: dict[str, Persistence] = {}
        fold_round(tally, [self._round("BTCUSDT", 0.003, 125_000.0)], f)
        tally["BTCUSDT"].total_rounds = 1

        assert tally["BTCUSDT"].breakeven_fee_bps(0.0) < 0.01
