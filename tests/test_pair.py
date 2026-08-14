"""Cross-market fair value and the pre-trade executable-edge gate."""

import json
from decimal import Decimal

import pytest

from jsboard.core.market import MARKET_OWNER, MarketView
from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, FeedStatus, TradeTick
from jsboard.feed.replay import _encode
from jsboard.mm.quoter import Quote, QuoteSet
from jsboard.sim.hedge import HedgeConfig, Hedger
from jsboard.sim.pair import CrossMarketFairValue, PairQuoteGate

MAKER = Instrument("PAIR", Decimal("1"), Decimal("1"), "P", "U")
HEDGE = Instrument("PAIR", Decimal("0.5"), Decimal("1"), "P", "U")


def market(instrument, bids, asks, ts_ns=1_000_000_000):
    view = MarketView(instrument=instrument)
    view.book.replace_l2(list(bids), list(asks), owner=MARKET_OWNER)
    view.last_update_ns = ts_ns
    return view


def gate(*, bids=((202, 100),), asks=((204, 100),), now=1_000_000_000, **kw):
    maker_view = market(MAKER, [(99, 100)], [(101, 100)])
    hedge_view = market(HEDGE, bids, asks)
    hedger = Hedger(HEDGE, hedge_view, HedgeConfig(max_levels=20))
    return PairQuoteGate(
        MAKER,
        HEDGE,
        maker_view,
        hedge_view,
        hedger,
        clock=lambda: now,
        **kw,
    )


def test_cross_market_fair_value_converts_tick_units():
    hedge_view = market(HEDGE, [(199, 10)], [(201, 10)])
    estimator = CrossMarketFairValue(MAKER, HEDGE, hedge_view)
    # Hedge microprice is 200 ticks * 0.5 = 100 quote units = 100 maker ticks.
    assert estimator.estimate(market(MAKER, [(99, 1)], [(101, 1)])) == pytest.approx(100)


def test_maker_buy_is_compared_with_executable_hedge_bid():
    g = gate(maker_bps=1, taker_bps=4)
    edge = g.edge(Quote(Side.BUY, 99, 10))
    assert edge.hedge_price == pytest.approx(101)
    assert edge.gross_bps == pytest.approx((101 - 99) / 99 * 10_000)
    assert edge.net_bps < edge.gross_bps
    assert edge.executable


def test_maker_sell_is_compared_with_executable_hedge_ask():
    g = gate(bids=((196, 100),), asks=((198, 100),), taker_bps=0)
    edge = g.edge(Quote(Side.SELL, 101, 10))
    assert edge.hedge_price == pytest.approx(99)
    assert edge.gross_bps == pytest.approx((101 - 99) / 101 * 10_000)


def test_book_walking_is_included_before_the_quote_passes():
    g = gate(bids=((202, 5), (198, 5)), asks=((204, 100),), taker_bps=0)
    edge = g.edge(Quote(Side.BUY, 99, 10))
    assert edge.hedge_price == pytest.approx(100)
    assert edge.net_bps == pytest.approx((100 - 99) / 99 * 10_000)


def test_partial_hedge_is_rejected_instead_of_leaving_directional_risk():
    g = gate(bids=((202, 5),), asks=((204, 100),))
    edge = g.edge(Quote(Side.BUY, 99, 10))
    assert not edge.executable
    assert edge.reason == "insufficient hedge depth"


def test_stale_hedge_book_rejects_the_quote():
    g = gate(now=2_000_000_000, max_hedge_age_ms=250)
    edge = g.edge(Quote(Side.BUY, 99, 10))
    assert not edge.executable
    assert edge.reason == "hedge book stale"


def test_filter_keeps_only_the_side_above_the_net_threshold():
    # Buy at 99 and sell hedge at 101 is profitable; maker sell at 101 and
    # hedge buy at 102 is not.
    g = gate(min_net_bps=100, taker_bps=0)
    quotes = QuoteSet(
        bids=(Quote(Side.BUY, 99, 10),),
        asks=(Quote(Side.SELL, 101, 10),),
    )
    kept = g.filter(quotes)
    assert len(kept.bids) == 1
    assert kept.asks == ()
    assert g.stats.quotes_tested == 2
    assert g.stats.quotes_passed == 1
    assert g.stats.pass_share == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_pair_command_prices_then_fills_and_hedges(tmp_path, capsys):
    from jsboard.cli import build_parser

    path = tmp_path / "pair.jsonl"
    base_ns = 1_800_000_000_000_000_000
    events = [
        ("spot", base_ns, FeedStatus("live", "ok", base_ns)),
        ("spot", base_ns + 1_000_000, DepthSnapshot(
            ((99, 100), (98, 100)), ((101, 100), (102, 100)), 1, base_ns + 1_000_000
        )),
        ("perp", base_ns + 2_000_000, FeedStatus("live", "ok", base_ns + 2_000_000)),
        ("perp", base_ns + 3_000_000, DepthSnapshot(
            ((202, 1_000), (200, 1_000)), ((204, 1_000), (206, 1_000)),
            1, base_ns + 3_000_000
        )),
        # The sell print consumes the public bid queue and then our paper bid.
        ("spot", base_ns + 20_000_000, TradeTick(
            99, 300, Side.SELL, 1, base_ns + 20_000_000
        )),
        # The maker market then moves rich to the same hedge book.  The
        # strategy can sell the maker leg and buy the hedge, closing both.
        ("spot", base_ns + 30_000_000, DepthSnapshot(
            ((102, 100), (101, 100)), ((104, 100), (105, 100)),
            2, base_ns + 30_000_000
        )),
        ("spot", base_ns + 50_000_000, TradeTick(
            103, 300, Side.BUY, 2, base_ns + 50_000_000
        )),
    ]
    rows = []
    for src, received_ns, event in events:
        row = _encode(event)
        row.update({"src": src, "rx_ns": received_ns})
        rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    path.with_suffix(".jsonl.meta.json").write_text(json.dumps({
        "sources": {
            "spot": {
                "symbol": "PAIR", "tick_size": "1", "lot_size": "1",
                "base": "P", "quote": "U",
            },
            "perp": {
                "symbol": "PAIR", "tick_size": "0.5", "lot_size": "1",
                "base": "P", "quote": "U",
            },
        }
    }))

    args = build_parser().parse_args([
        "pair", str(path), "--maker-fees", "0", "--taker-bps", "0",
        "--pair-edge-bps", "0", "--requote-ms", "1", "--latency-ms", "0",
        "--hedge-latency-ms", "0", "--plain",
    ])
    assert await args.func(args) == 0
    output = capsys.readouterr().out
    assert "相対価値MM" in output
    assert "全執行コスト後でプラス" in output
