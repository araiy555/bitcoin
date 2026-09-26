"""Paper trading on a live feed: the daily ledger, and the loop end to end."""

from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, Feed, FeedStatus, TradeTick
from jsboard.mm.papertrade import DailyTally, slack_text


def test_the_day_closes_when_the_date_changes():
    tally = DailyTally("2026-09-26")
    assert tally.roll("2026-09-26", {"total": 50.0, "fills": 10, "fees": -4.0}) is None
    report = tally.roll("2026-09-27", {"total": 120.0, "fills": 30, "fees": -9.0, "position": 5})
    assert report == {"day": "2026-09-26", "pnl": 120.0, "fills": 30, "fees": -9.0, "position": 5}
    # The next day is measured from where this one ended.
    nxt = tally.roll("2026-09-28", {"total": 100.0, "fills": 45, "fees": -12.0, "position": 0})
    assert nxt["pnl"] == -20.0 and nxt["fills"] == 15


def test_a_rebate_is_called_a_rebate():
    text = slack_text("bitbank ada_jpy", "JPY", "ADA",
                      {"day": "d", "pnl": 812.0, "fills": 900, "fees": -40.0, "position": 12.5},
                      {"total": 812.0, "fills": 900.0})
    assert "リベート 40" in text and "+812 JPY" in text


INST = Instrument("ada_jpy", Decimal("0.001"), Decimal("0.0001"), "ADA", "JPY")
LEAD = Instrument("ADAUSDT", Decimal("0.0001"), Decimal("1"), "ADA", "USDT")


class Book(Feed):
    async def stream(self):
        yield FeedStatus("live", "fake")
        for i in range(300):
            mid = 37_200 + (i // 40) % 3
            yield DepthSnapshot(tuple((mid - 5 - j, 9_000_000) for j in range(5)),
                                tuple((mid + 5 + j, 9_000_000) for j in range(5)), i)
            yield TradeTick(mid + 12 if i % 2 else mid - 12, 5_000_000,
                            Side.BUY if i % 2 else Side.SELL, i)


class Lead(Feed):
    async def stream(self):
        yield FeedStatus("live", "fake")
        yield DepthSnapshot(((2480, 90_000),), ((2481, 90_000),), 1)


@pytest.mark.asyncio
async def test_the_loop_runs_the_maker_and_reports_when_the_feed_ends(monkeypatch, capsys):
    import jsboard.cli as cli

    async def live_book(target):
        return INST, Book(INST)

    async def futures(symbol):
        assert symbol == "ADAUSDT"
        return LEAD

    posted = []

    async def post(text):
        posted.append(text)
        return True

    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "fetch_futures_instrument", futures)
    monkeypatch.setattr(cli, "_binance_lead", lambda inst, ms: Lead(inst))
    monkeypatch.setattr(cli, "_post_slack", post)
    args = cli.build_parser().parse_args(["paper", "--log-every", "0", "--slack"])
    assert await args.func(args) == 1
    out = capsys.readouterr().out
    assert "注文は一切出しません" in out
    assert "約定" in out
    assert posted and "止まりました" in posted[-1]
