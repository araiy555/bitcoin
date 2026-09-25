"""The daily screen must reproduce what the replays already found.

bitbank ADA won, GMO's leverage XRP lost to free pick-offs, and bitbank XRP
had its spread competed away. A screen that ranked those three any other
way would be screening for something else.
"""

from jsboard.research.jpscan import (
    add_bitbank_tickers,
    add_gmo_tickers,
    bitbank_books,
    gmo_books,
    ranked,
)

BB_PAIRS = {"data": {"pairs": [
    {"name": "ada_jpy", "is_enabled": True, "maker_fee_rate_quote": "-0.0002",
     "taker_fee_rate_quote": "0.0012"},
    {"name": "xrp_jpy", "is_enabled": True, "maker_fee_rate_quote": "-0.0002",
     "taker_fee_rate_quote": "0.0012"},
    {"name": "btc_usdt", "is_enabled": True, "maker_fee_rate_quote": "0",
     "taker_fee_rate_quote": "0.001"},
]}}
BB_TICKERS = {"data": [
    {"pair": "ada_jpy", "buy": "37.000", "sell": "37.018", "last": "37.01", "vol": "4500000"},
    {"pair": "xrp_jpy", "buy": "241.500", "sell": "241.511", "last": "241.5", "vol": "9000000"},
]}
GMO_SYMBOLS = {"data": [
    {"symbol": "XRP_JPY", "makerFee": "0", "takerFee": "0"},
    {"symbol": "SOL_JPY", "makerFee": "0", "takerFee": "0.0003"},
    {"symbol": "ADA", "makerFee": "-0.0003", "takerFee": "0.0009"},
]}
GMO_TICKERS = {"data": [
    {"symbol": "XRP_JPY", "bid": "241.40", "ask": "241.57", "last": "241.5", "volume": "20000000"},
    {"symbol": "SOL_JPY", "bid": "30000", "ask": "30030", "last": "30010", "volume": "20000"},
    {"symbol": "ADA", "bid": "36.95", "ask": "37.01", "last": "37.0", "volume": "100000"},
]}


def books():
    bb, gmo = bitbank_books(BB_PAIRS), gmo_books(GMO_SYMBOLS)
    add_bitbank_tickers(bb, BB_TICKERS)
    add_gmo_tickers(gmo, GMO_TICKERS)
    return {f"{b.venue} {b.symbol}": b for b in [*bb.values(), *gmo.values()]}


def test_only_jpy_pairs_are_screened():
    assert "btc_usdt" not in bitbank_books(BB_PAIRS)


def test_fees_become_bps_with_rebates_positive():
    ada = books()["bitbank ada_jpy"]
    assert ada.rebate_bps == 2.0
    assert ada.taker_bps == 12.0


def test_the_book_that_won_is_a_candidate():
    ada = books()["bitbank ada_jpy"]
    assert ada.verdict() == "候補"
    assert ada.edge_bps > 2.0  # half of ~4.9bps plus the 2bps rebate


def test_free_taking_is_flagged_whatever_the_spread():
    assert books()["gmo XRP_JPY"].verdict() == "狙われやすい"


def test_a_competed_away_spread_is_flagged_despite_the_rebate():
    assert books()["bitbank xrp_jpy"].verdict() == "業者が詰めている"


def test_a_thin_book_is_flagged():
    assert books()["gmo ADA"].verdict() == "取引が少ない"


def test_candidates_rank_first():
    order = [f"{b.venue} {b.symbol}" for b in ranked(list(books().values()))]
    assert order[0] == "bitbank ada_jpy"


def test_the_slack_message_leads_with_candidates_and_says_why_others_fell_out():
    from jsboard.research.jpscan import slack_text

    text = slack_text(ranked(list(books().values())), "2026-09-25 09:00")
    lines = text.splitlines()
    assert "候補 1 件" in lines[1]
    assert "ada_jpy" in lines[2]
    assert "狙われやすい 2" in text and "業者が詰めている 1" in text


def test_a_day_without_candidates_says_so():
    from jsboard.research.jpscan import slack_text

    gmo = gmo_books(GMO_SYMBOLS)
    add_gmo_tickers(gmo, GMO_TICKERS)
    assert "条件を満たす銘柄がありません" in slack_text(list(gmo.values()), "x")


def test_slack_without_a_webhook_refuses(monkeypatch, capsys):
    import asyncio

    import jsboard.cli as cli

    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)

    class Resp:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        async def json(self):
            return {"data": {"pairs": []}} if "pairs" in self.url else {"data": []}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def get(self, url, timeout=None):
            r = Resp()
            r.url = url
            return r

    monkeypatch.setattr(cli, "make_session", lambda **kw: Session())
    args = cli.build_parser().parse_args(["jpscan", "--samples", "1", "--slack"])
    assert asyncio.run(args.func(args)) == 1
    assert "SLACK_WEBHOOK_URL" in capsys.readouterr().out


def test_a_thin_toll_on_takers_is_still_too_thin():
    # GMO's other leverage books charge takers 3bps: the XRP_JPY mechanism
    # that lost, with a small fee on it.
    assert books()["gmo SOL_JPY"].verdict() == "狙われやすい"


def test_a_wide_spread_on_an_empty_book_does_not_outrank_a_busy_one():
    from jsboard.research.jpscan import Book

    busy = Book("bitbank", "busy", -2.0, 12.0, [5.0], volume_jpy=3e8)
    empty = Book("gmo", "empty", -3.0, 9.0, [30.0], volume_jpy=0.6e8)
    assert busy.edge_bps < empty.edge_bps
    assert ranked([empty, busy])[0] is busy
