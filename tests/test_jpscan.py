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
    {"symbol": "ADA", "makerFee": "-0.0003", "takerFee": "0.0009"},
]}
GMO_TICKERS = {"data": [
    {"symbol": "XRP_JPY", "bid": "241.40", "ask": "241.57", "last": "241.5", "volume": "20000000"},
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
