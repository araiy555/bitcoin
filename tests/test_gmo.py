"""GMO Coin's book, as the venue we would quote on, led by Binance.

The lead is priced in dollars and our book in yen. The gate has to see a
0.1% move on Binance as 10bps, not as 10bps divided by the exchange rate.
"""

import json
from decimal import Decimal

import pytest

from jsboard.core.types import Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.feed.gmo import GmoFeed, instrument_from_rule, ts_ns

RULE = {"symbol": "XRP_JPY", "minOrderSize": "10", "sizeStep": "1", "tickSize": "0.001"}


def feed():
    return GmoFeed(instrument_from_rule(RULE))


class TestSpec:
    def test_symbols_row_becomes_an_instrument(self):
        inst = instrument_from_rule(RULE)
        assert inst.tick_size == Decimal("0.001")
        assert inst.lot_size == Decimal("1")
        assert (inst.base, inst.quote) == ("XRP", "JPY")

    def test_a_whole_number_step_prints_as_one(self):
        inst = instrument_from_rule({**RULE, "sizeStep": "10"})
        assert str(inst.lot_size) == "10"

    def test_timestamps_are_utc(self):
        assert ts_ns("2026-09-24T00:00:00.500Z") == 1_790_208_000_500_000_000


class TestMessages:
    def test_a_book_message_is_a_full_snapshot(self):
        [event] = feed().parse_message(
            {
                "channel": "orderbooks",
                "symbol": "XRP_JPY",
                "timestamp": "2026-09-24T00:00:00.000Z",
                "bids": [{"price": "350.100", "size": "1200"}, {"price": "350.050", "size": "300"}],
                "asks": [{"price": "350.350", "size": "800"}],
            }
        )
        assert isinstance(event, DepthSnapshot)
        assert event.bids == ((350100, 1200), (350050, 300))
        assert event.asks == ((350350, 800),)

    def test_a_taker_trade_carries_its_side(self):
        [event] = feed().parse_message(
            {
                "channel": "trades",
                "symbol": "XRP_JPY",
                "side": "SELL",
                "price": "350.100",
                "size": "50",
                "timestamp": "2026-09-24T00:00:01.000Z",
            }
        )
        assert isinstance(event, TradeTick)
        assert event.aggressor is Side.SELL and event.qty == 50

    def test_another_symbol_is_ignored(self):
        assert feed().parse_message({"channel": "orderbooks", "symbol": "XRP", "bids": [], "asks": []}) == []


class TestLeadInAnotherCurrency:
    def test_a_dollar_move_reads_at_full_size_in_yen(self):
        from jsboard.core.market import MarketView
        from jsboard.mm.toxicity import ToxicityConfig, ToxicityGate

        inst = instrument_from_rule(RULE)
        market = MarketView(inst)
        market.apply(
            DepthSnapshot(bids=((350000, 1000),), asks=((350010, 1000),), last_update_id=1, ts_ns=1)
        )
        now = [1_000_000_000]
        market.clock = lambda: now[0]

        class Lead:
            price = 350005 / 150.0  # the same coin in dollars, in our tick units

            def estimate(self, _market):
                return self.price

        lead = Lead()
        gate = ToxicityGate(ToxicityConfig(lead_threshold_bps=5.0), lead=lead)
        gate.evaluate(market)  # learns the exchange rate as basis
        now[0] += 250_000_000
        lead.price *= 1.001  # Binance up 10bps
        assert gate.lead_bps(market) == pytest.approx(10.0, abs=0.1)


@pytest.mark.asyncio
async def test_sweep_quotes_gmo_while_watching_binance(tmp_path, capsys):
    from jsboard.cli import build_parser
    from jsboard.feed.replay import _encode

    NS = 1_000_000_000
    base = 1_790_208_000 * NS
    gmo_spec = {"symbol": "XRP_JPY", "tick_size": "0.001", "lot_size": "1",
                "base": "XRP", "quote": "JPY", "market": "leverage"}
    bn_spec = {"symbol": "XRPUSDT", "tick_size": "0.0001", "lot_size": "0.1",
               "base": "XRP", "quote": "USDT", "market": "perp"}
    rows = [{"k": "status", "state": "live", "detail": "", "ts_ns": base, "src": "gmo"}]
    for i in range(200):
        ts = base + i * 250_000_000
        g = _encode(DepthSnapshot(
            bids=tuple((350000 - 10 * j, 5000) for j in range(5)),
            asks=tuple((350250 + 10 * j, 5000) for j in range(5)),
            last_update_id=i, ts_ns=ts))
        g.update(src="gmo", rx_ns=ts)
        b = _encode(DepthSnapshot(
            bids=tuple((23340 - j, 90000) for j in range(5)),
            asks=tuple((23341 + j, 90000) for j in range(5)),
            last_update_id=i, ts_ns=ts))
        b.update(src="binance", rx_ns=ts)
        rows += [g, b]
    path = tmp_path / "gmo.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"sources": {"gmo": gmo_spec, "binance": bn_spec}})
    )
    args = build_parser().parse_args([
        "sweep", str(path), "--source", "gmo", "--lead-source", "binance",
        "--axis=lead_threshold_bps=0,5", "--plain",
    ])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "XRP_JPY" in out and "2 通り" in out
    # The day's drift is kept out of one column, so skill can be read apart from luck.
    assert "10秒内計" in out
