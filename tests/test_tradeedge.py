"""Every bitbank pair from its published executions."""

import pytest

from jsboard.research.tradeedge import PairEdge, day_edge, parse


def rows(spec):
    return [{"executed_at": t, "side": side, "price": str(p), "amount": "10"} for t, side, p in spec]


def test_sellers_hitting_a_steady_bid_pay_the_maker():
    # Sellers hit 99.9 and buyers lift 100.1 all day; the price never moves.
    day = parse(rows([(i * 4_000, "sell" if i % 2 else "buy", 99.9 if i % 2 else 100.1)
                      for i in range(2_000)]))
    edge = PairEdge("x_jpy", rebate_bps=2.0, days=[day_edge(day, [])])
    assert edge.bps(60) > 2.0 and edge.good_days == 1 and edge.passed


def test_a_seller_ahead_of_a_fall_costs_the_maker():
    spec = []
    price = 100.0
    for i in range(2_000):
        spec.append((i * 5_000, "sell", price))
        price -= 0.01  # every seller is followed by a lower price
    edge = PairEdge("x_jpy", rebate_bps=2.0, days=[day_edge(parse(rows(spec)), [])])
    assert edge.bps(60) < 0 and not edge.passed
    assert "不合格" in edge.row()


@pytest.mark.asyncio
async def test_the_command_ranks_every_pair(monkeypatch, capsys):
    import jsboard.cli as cli

    tape = rows([(1_790_380_800_000 + i * 5_000, "sell" if i % 2 else "buy",
                  99.9 if i % 2 else 100.1) for i in range(2_000)])

    async def get_json(session, url):
        if url == cli.BITBANK_PAIRS_URL:
            return {"data": {"pairs": [
                {"name": n, "maker_fee_rate_quote": "-0.0002", "taker_fee_rate_quote": "0.0012"}
                for n in ("ada_jpy", "xrp_jpy")]}}
        return {"data": {"transactions": tape}}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(cli, "_get_json", get_json)
    monkeypatch.setattr(cli, "make_session", Session)
    args = cli.build_parser().parse_args(["tradeedge", "--days", "2", "--pause", "0"])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "合格: ada_jpy, xrp_jpy" in out and "60秒がプラスの日" in out
