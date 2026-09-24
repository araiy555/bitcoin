"""Hyperliquid's book, as the venue we would quote on.

The prices have no fixed tick on the venue, which is the part most likely to
be wrong silently: a tick too coarse rounds real prices together, one too fine
lets the quoter "improve" by a step the venue would reject.
"""

import json
from decimal import Decimal

import pytest

from jsboard.core.types import Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.feed.hyperliquid import HyperliquidFeed, coin_for, instrument_for, tick_for


class TestTick:
    def test_five_significant_figures_below_one(self):
        assert tick_for(0.81234, sz_decimals=0) == Decimal("0.00001")

    def test_five_significant_figures_above_one(self):
        assert tick_for(12.345, sz_decimals=1) == Decimal("0.001")

    def test_decimal_cap_binds_for_small_prices_with_fine_sizes(self):
        # 6 - szDecimals = 1 decimal allowed, tighter than 5 sig figs would give.
        assert tick_for(0.5, sz_decimals=5) == Decimal("0.1")

    def test_whole_numbers_are_always_allowed(self):
        assert tick_for(97_123.0, sz_decimals=5) == Decimal("1")

    def test_symbol_maps_to_coin(self):
        assert coin_for("ADAUSDT") == "ADA"
        assert coin_for("kPEPEUSDT") == "KPEPE"


def feed():
    return HyperliquidFeed(instrument_for("ADA", sz_decimals=0, price=0.8))


class TestMessages:
    def test_a_book_message_is_a_full_snapshot(self):
        f = feed()
        [event] = f.parse_message(
            {
                "channel": "l2Book",
                "data": {
                    "coin": "ADA",
                    "time": 1_757_000_000_123,
                    "levels": [
                        [{"px": "0.80010", "sz": "1500", "n": 3}, {"px": "0.80000", "sz": "900", "n": 1}],
                        [{"px": "0.80030", "sz": "700", "n": 2}],
                    ],
                },
            }
        )
        assert isinstance(event, DepthSnapshot)
        assert event.bids == ((80010, 1500), (80000, 900))
        assert event.asks == ((80030, 700),)
        assert event.ts_ns == 1_757_000_000_123 * 1_000_000

    def test_trades_carry_the_aggressor(self):
        events = feed().parse_message(
            {
                "channel": "trades",
                "data": [
                    {"coin": "ADA", "side": "B", "px": "0.8003", "sz": "100", "time": 1, "tid": 7},
                    {"coin": "ADA", "side": "A", "px": "0.8001", "sz": "50", "time": 2, "tid": 8},
                ],
            }
        )
        assert [e.aggressor for e in events] == [Side.BUY, Side.SELL]
        assert all(isinstance(e, TradeTick) for e in events)
        assert events[0].price == 80030 and events[0].qty == 100

    def test_other_coins_and_channels_are_ignored(self):
        f = feed()
        assert f.parse_message({"channel": "l2Book", "data": {"coin": "BTC", "levels": [[], []]}}) == []
        assert f.parse_message({"channel": "subscriptionResponse", "data": {}}) == []

    def test_dust_below_one_lot_is_not_a_level(self):
        [event] = feed().parse_message(
            {"channel": "l2Book", "data": {"coin": "ADA", "time": 1, "levels": [[{"px": "0.8", "sz": "0.4", "n": 1}], []]}}
        )
        assert event.bids == ()


@pytest.mark.asyncio
async def test_sweep_quotes_hyperliquid_while_watching_binance(tmp_path, capsys):
    """The two venues have different ticks; the lead must be rescaled, not reread."""
    from jsboard.cli import build_parser
    from jsboard.feed.replay import _encode

    NS = 1_000_000_000
    base = 1_757_000_000 * NS
    hl_spec = {"symbol": "ADA", "tick_size": "0.00001", "lot_size": "1",
               "base": "ADA", "quote": "USDC", "market": "perp"}
    bn_spec = {"symbol": "ADAUSDT", "tick_size": "0.0001", "lot_size": "1",
               "base": "ADA", "quote": "USDT", "market": "perp"}
    rows = [{"k": "status", "state": "live", "detail": "", "ts_ns": base, "src": "hyperliquid"}]
    for i in range(200):
        ts = base + i * 250_000_000
        hl = _encode(DepthSnapshot(
            bids=tuple((80000 - 3 * j, 5000) for j in range(5)),
            asks=tuple((80006 + 3 * j, 5000) for j in range(5)),
            last_update_id=i, ts_ns=ts))
        hl.update(src="hyperliquid", rx_ns=ts)
        bn = _encode(DepthSnapshot(
            bids=tuple((8000 - j, 90000) for j in range(5)),
            asks=tuple((8001 + j, 90000) for j in range(5)),
            last_update_id=i, ts_ns=ts))
        bn.update(src="binance", rx_ns=ts)
        rows += [hl, bn]
    path = tmp_path / "dex.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"sources": {"hyperliquid": hl_spec, "binance": bn_spec}})
    )
    args = build_parser().parse_args([
        "sweep", str(path), "--source", "hyperliquid", "--lead-source", "binance",
        "--axis=lead_threshold_bps=0,2", "--plain",
    ])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "tick=0.00001" in out
    # Binance mid 0.80005 against Hyperliquid mid 0.80003: a 0.25bps standing gap,
    # learned as basis, so a 2bps gate never fires.
    assert "2 通り" in out
