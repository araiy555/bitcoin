"""Automatic Binance spot/perpetual route selection."""

import json
from decimal import Decimal

import pytest

from jsboard.cli import build_parser
from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, MarkPrice
from jsboard.feed.replay import _encode
from jsboard.sim.dealer import BinanceDealer, DealerRoute

SPOT = Instrument("PAIR", Decimal("1"), Decimal("1"), "P", "U")
PERP = Instrument("PAIR", Decimal("1"), Decimal("1"), "P", "U")
BASE_NS = 1_800_000_000_000_000_000


def dealer(**kwargs):
    return BinanceDealer(
        instruments={"spot": SPOT, "perp": PERP},
        maker_bps={"spot": 0.0, "perp": 0.0},
        taker_bps={"spot": 0.0, "perp": 0.0},
        size_base=10.0,
        min_net_bps=0.0,
        max_age_ms=250.0,
        **kwargs,
    )


def seed_books(engine):
    engine.apply(
        "spot",
        DepthSnapshot(((99, 100),), ((101, 100),), 1, BASE_NS),
        BASE_NS,
    )
    engine.apply(
        "perp",
        DepthSnapshot(((103, 100),), ((105, 100),), 1, BASE_NS + 1_000_000),
        BASE_NS + 1_000_000,
    )


def test_dealer_compares_all_routes_and_selects_best_after_funding():
    engine = dealer(funding_horizon_h=8.0)
    seed_books(engine)
    engine.apply(
        "perp",
        MarkPrice(104, 104, 0.001, BASE_NS + 3_600_000_000_000, BASE_NS + 2_000_000),
        BASE_NS + 2_000_000,
    )

    opportunities = engine.evaluate()
    assert len(opportunities) == 4
    by_route = {row.route: row for row in opportunities}
    spot_buy = by_route[DealerRoute("spot", "perp", Side.BUY)]
    spot_sell = by_route[DealerRoute("spot", "perp", Side.SELL)]
    assert spot_buy.gross_bps == pytest.approx((103 - 99) / 99 * 10_000)
    assert spot_buy.funding_bps > 0  # short perp receives positive funding
    assert spot_sell.funding_bps < 0  # long perp pays it
    assert spot_buy.candidate
    assert engine.stats[spot_buy.route].selected == 1


def test_dealer_rejects_stale_maker_and_insufficient_hedge_depth():
    stale = dealer()
    seed_books(stale)
    stale.now_ns = BASE_NS + 1_000_000_000
    opportunity = stale.opportunity(DealerRoute("spot", "perp", Side.BUY))
    assert not opportunity.executable
    assert opportunity.reason == "maker book stale"

    thin = dealer()
    thin.apply(
        "spot",
        DepthSnapshot(((99, 100),), ((101, 100),), 1, BASE_NS),
        BASE_NS,
    )
    thin.apply(
        "perp",
        DepthSnapshot(((103, 5),), ((105, 5),), 1, BASE_NS + 1_000_000),
        BASE_NS + 1_000_000,
    )
    opportunity = thin.opportunity(DealerRoute("spot", "perp", Side.BUY))
    assert not opportunity.executable
    assert opportunity.reason == "insufficient hedge depth"


@pytest.mark.asyncio
async def test_dealer_cli_routes_a_capture_without_claiming_realised_profit(tmp_path, capsys):
    path = tmp_path / "dealer.jsonl"
    events = [
        (
            "spot",
            BASE_NS,
            DepthSnapshot(((99, 100),), ((101, 100),), 1, BASE_NS),
        ),
        (
            "perp",
            BASE_NS + 1_000_000,
            DepthSnapshot(((103, 100),), ((105, 100),), 1, BASE_NS + 1_000_000),
        ),
        (
            "perp",
            BASE_NS + 2_000_000,
            MarkPrice(
                104,
                104,
                0.001,
                BASE_NS + 3_600_000_000_000,
                BASE_NS + 2_000_000,
            ),
        ),
    ]
    rows = []
    for source, received_ns, event in events:
        row = _encode(event)
        row.update({"src": source, "rx_ns": received_ns})
        rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    path.with_suffix(".jsonl.meta.json").write_text(
        json.dumps(
            {
                "sources": {
                    "spot": {
                        "symbol": "PAIR",
                        "tick_size": "1",
                        "lot_size": "1",
                        "base": "P",
                        "quote": "U",
                    },
                    "perp": {
                        "symbol": "PAIR",
                        "tick_size": "1",
                        "lot_size": "1",
                        "base": "P",
                        "quote": "U",
                    },
                }
            }
        )
    )

    args = build_parser().parse_args(
        [
            "dealer",
            str(path),
            "--size-base",
            "10",
            "--spot-maker-bps",
            "0",
            "--spot-taker-bps",
            "0",
            "--perp-maker-bps",
            "0",
            "--perp-taker-bps",
            "0",
            "--dealer-edge-bps",
            "0",
            "--sample-ms",
            "0",
            "--plain",
        ]
    )
    assert await args.func(args) == 0
    output = capsys.readouterr().out
    assert "spot買い → perp売り" in output
    assert "提示候補" in output
    assert "これは利益ではありません" in output

