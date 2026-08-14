"""Executable, causal cross-exchange spread replay."""

import json
from decimal import Decimal

import pytest

from jsboard.cli import build_parser
from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot, MarkPrice
from jsboard.feed.replay import _encode
from jsboard.sim.cross_exchange import CrossArbConfig, CrossExchangeArb

INST = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.1"), "BTC", "USDT")
BASE_NS = 1_800_000_000_000_000_000


def book(mid: float, ts_ns: int, qty: float = 100.0) -> DepthSnapshot:
    bid = INST.to_ticks(mid - 0.1)
    ask = INST.to_ticks(mid + 0.1)
    lots = INST.to_lots(qty)
    return DepthSnapshot(((bid, lots),), ((ask, lots),), ts_ns, ts_ns)


def update(engine: CrossExchangeArb, seconds: int, binance_mid: float, bybit_mid: float = 100.0):
    now = BASE_NS + int(seconds * 1e9)
    engine.apply("binance", book(binance_mid, now), now)
    engine.apply("bybit", book(bybit_mid, now), now)
    return engine.evaluate()


def engine(*, fees: float = 0.0, min_expected: float = 0.0) -> CrossExchangeArb:
    return CrossExchangeArb(
        {"binance": INST, "bybit": INST},
        CrossArbConfig(
            size_base=1.0,
            lookback_s=10.0,
            min_samples=4,
            entry_z=2.0,
            exit_z=1.0,
            max_hold_s=100.0,
            min_expected_net_bps=min_expected,
            max_age_ms=100.0,
            depth=5,
            taker_bps={"binance": fees, "bybit": fees},
        ),
    )


def warm(engine: CrossExchangeArb):
    for seconds, price in ((0, 100.0), (2, 100.1), (4, 99.9), (6, 100.1), (8, 99.9)):
        update(engine, seconds, price)


def basis_engine(*, allow_spot_short: bool = False) -> CrossExchangeArb:
    return CrossExchangeArb(
        {"perp": INST, "spot": INST},
        CrossArbConfig(
            size_base=1.0,
            lookback_s=10.0,
            min_samples=4,
            entry_z=2.0,
            exit_z=1.0,
            max_hold_s=100.0,
            min_expected_net_bps=0.0,
            max_age_ms=100.0,
            depth=5,
            taker_bps={"spot": 0.0, "perp": 0.0},
            allowed_directions=None if allow_spot_short else (("spot", "perp"),),
        ),
    )


def update_basis(engine: CrossExchangeArb, seconds: int, perp_mid: float, spot_mid: float = 100.0):
    now = BASE_NS + int(seconds * 1e9)
    engine.apply("perp", book(perp_mid, now), now)
    engine.apply("spot", book(spot_mid, now), now)
    return engine.evaluate()


def warm_basis(engine: CrossExchangeArb):
    for seconds, price in ((0, 100.0), (2, 100.1), (4, 99.9), (6, 100.1), (8, 99.9)):
        update_basis(engine, seconds, price)


def test_cross_exchange_enters_only_after_causal_window_and_realises_four_legs():
    strategy = engine()
    warm(strategy)
    # The first complete window is evaluated only from the four prior points.
    assert strategy.stats.warm == 1
    assert strategy.position is None

    z = update(strategy, 10, 110.0)
    assert z is not None and z > 2
    assert strategy.position is not None
    assert strategy.position.long_source == "bybit"
    assert strategy.position.short_source == "binance"

    update(strategy, 12, 100.0)
    assert strategy.position is None
    assert len(strategy.trades) == 1
    trade = strategy.trades[0]
    assert trade.gross_quote > 0
    assert trade.fees_quote == 0
    assert trade.net_quote == trade.gross_quote
    assert trade.exit_reason == "mean"


def test_cross_exchange_rejects_z_signal_when_round_trip_cost_exceeds_convergence():
    strategy = engine(fees=500.0, min_expected=1.0)
    warm(strategy)
    update(strategy, 10, 110.0)
    assert strategy.position is None
    assert strategy.stats.signals == 1
    assert strategy.stats.rejected_cost == 1


def test_cross_exchange_rejects_stale_other_venue():
    strategy = engine()
    now = BASE_NS
    strategy.apply("binance", book(100.0, now), now)
    strategy.apply("bybit", book(100.0, now), now)
    strategy.now_ns += 1_000_000_000
    assert strategy.evaluate() is None
    assert strategy.stats.fresh == 0


def test_cross_exchange_rejects_receive_time_skew_with_machine_readable_reason():
    strategy = engine()
    now = BASE_NS
    strategy.apply("binance", book(100.0, now), now)
    strategy.apply("bybit", book(100.0, now + 500_000_000), now + 500_000_000)
    strategy.config = CrossArbConfig(
        size_base=1.0, lookback_s=10.0, min_samples=4, entry_z=2.0, exit_z=1.0,
        max_hold_s=100.0, min_expected_net_bps=0.0, max_age_ms=1_000.0,
        max_skew_ms=100.0, depth=5, taker_bps={"binance": 0.0, "bybit": 0.0},
    )
    assert strategy.evaluate() is None
    assert strategy.stats.reject_codes["SKEW"] == 1


def test_execution_buffers_are_part_of_expected_net():
    strategy = CrossExchangeArb(
        {"binance": INST, "bybit": INST},
        CrossArbConfig(
            size_base=1.0, lookback_s=10.0, min_samples=4, entry_z=2.0, exit_z=1.0,
            max_hold_s=100.0, min_expected_net_bps=1_000.0, max_age_ms=100.0,
            depth=5, taker_bps={"binance": 0.0, "bybit": 0.0},
            hedge_latency_buffer_bps=10.0, fill_model_buffer_bps=10.0,
            safety_margin_bps=10.0,
        ),
    )
    warm(strategy)
    update(strategy, 10, 110.0)
    assert strategy.position is None
    assert strategy.stats.rejected_cost == 1
    assert strategy.stats.reject_codes["NET_NEGATIVE"] == 1


def test_cross_exchange_accounts_for_funding_settlement_on_both_perps():
    strategy = engine()
    warm(strategy)
    update(strategy, 10, 110.0)
    assert strategy.position is not None
    settle_ns = BASE_NS + int(11 * 1e9)
    for source, rate in (("binance", 0.001), ("bybit", 0.0002)):
        strategy.apply(
            source,
            MarkPrice(1000, 1000, rate, settle_ns, BASE_NS + int(10.5 * 1e9)),
            BASE_NS + int(10.5 * 1e9),
        )
        strategy.apply(
            source,
            MarkPrice(1000, 1000, rate, settle_ns + int(8 * 3600 * 1e9), settle_ns),
            settle_ns,
        )
    # Long Bybit pays 0.02 quote; short Binance receives 0.10 quote.
    assert strategy.position.funding_quote == pytest.approx(0.08)
    update(strategy, 12, 100.0)
    assert strategy.trades[0].funding_quote == pytest.approx(0.08)


def test_basis_enters_cash_and_carry_but_rejects_unborrowed_spot_short():
    strategy = basis_engine()
    warm_basis(strategy)
    update_basis(strategy, 10, 110.0)
    assert strategy.position is not None
    assert strategy.position.long_source == "spot"
    assert strategy.position.short_source == "perp"

    reverse = basis_engine()
    warm_basis(reverse)
    update_basis(reverse, 10, 90.0)
    assert reverse.position is None
    assert reverse.stats.rejected_direction == 1
    assert reverse.stats.reject_codes["BORROW_UNAVAILABLE"] == 1


def test_basis_expected_funding_receipt_improves_entry_net():
    strategy = basis_engine()
    warm_basis(strategy)
    now = BASE_NS + int(9 * 1e9)
    strategy.apply(
        "perp",
        MarkPrice(1000, 1000, 0.001, BASE_NS + int(11 * 1e9), now),
        now,
    )
    update_basis(strategy, 10, 110.0)
    assert strategy.position is not None
    # A positive funding rate is received by the short perpetual leg.
    assert strategy.position.expected_net_bps > 900.0


@pytest.mark.asyncio
async def test_xarb_cli_replays_tagged_capture(tmp_path, capsys):
    path = tmp_path / "xarb.jsonl"
    rows = []
    points = ((0, 100.0), (2, 100.1), (4, 99.9), (6, 100.1), (8, 99.9), (10, 110.0), (12, 100.0))
    for seconds, binance_mid in points:
        now = BASE_NS + int(seconds * 1e9)
        for source, mid in (("binance", binance_mid), ("bybit", 100.0)):
            row = _encode(book(mid, now))
            row.update({"src": source, "rx_ns": now})
            rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    spec = {
        "symbol": "BTCUSDT",
        "tick_size": "0.1",
        "lot_size": "0.1",
        "base": "BTC",
        "quote": "USDT",
        "market": "perp",
    }
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps({"sources": {"binance": spec, "bybit": spec}})
    )
    args = build_parser().parse_args(
        [
            "xarb",
            str(path),
            "--size-base",
            "1",
            "--lookback-minutes",
            str(10 / 60),
            "--min-samples",
            "4",
            "--entry-z",
            "2",
            "--exit-z",
            "1",
            "--binance-taker-bps",
            "0",
            "--bybit-taker-bps",
            "0",
            "--min-expected-net-bps",
            "0",
            "--sample-ms",
            "0",
            "--plain",
        ]
    )
    assert await args.func(args) == 0
    output = capsys.readouterr().out
    assert "取引所間Zスコア" in output
    assert "完了取引 : 1" in output
    assert "往復全コスト" in output


@pytest.mark.asyncio
async def test_basis_cli_replays_spot_perp_capture(tmp_path, capsys):
    path = tmp_path / "basis.jsonl"
    rows = []
    points = (
        (0, 100.0),
        (2, 100.1),
        (4, 99.9),
        (6, 100.1),
        (8, 99.9),
        (10, 110.0),
        (12, 100.0),
    )
    for seconds, perp_mid in points:
        now = BASE_NS + int(seconds * 1e9)
        for source, mid in (("perp", perp_mid), ("spot", 100.0)):
            row = _encode(book(mid, now))
            row.update({"src": source, "rx_ns": now})
            rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    common = {
        "symbol": "BTCUSDT",
        "tick_size": "0.1",
        "lot_size": "0.1",
        "base": "BTC",
        "quote": "USDT",
    }
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "sources": {
                    "spot": {**common, "market": "spot"},
                    "perp": {**common, "market": "perp"},
                }
            }
        )
    )
    args = build_parser().parse_args(
        [
            "basis",
            str(path),
            "--size-base",
            "1",
            "--lookback-minutes",
            str(10 / 60),
            "--min-samples",
            "4",
            "--entry-z",
            "2",
            "--exit-z",
            "1",
            "--spot-taker-bps",
            "0",
            "--perp-taker-bps",
            "0",
            "--min-expected-net-bps",
            "0",
            "--sample-ms",
            "0",
            "--plain",
        ]
    )
    assert await args.func(args) == 0
    output = capsys.readouterr().out
    assert "現物–先物ベーシス" in output
    assert "完了取引 : 1" in output
    assert "Funding" in output
