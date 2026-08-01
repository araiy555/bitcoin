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
