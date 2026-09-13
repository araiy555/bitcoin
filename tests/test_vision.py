"""Reading Binance's published daily archives.

The failures worth guarding are silent ones. A swapped column turns price into
quantity without raising; a misread `is_buyer_maker` flips the sign of every
flow measurement downstream; a timestamp read in the wrong unit puts a day of
data in 1970 and the merge with the tape becomes meaningless.
"""

import io
import zipfile
from datetime import date
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.research.vision import (
    AGG_TRADE_COLUMNS,
    BOOK_TICKER_COLUMNS,
    Stamped,
    book_events,
    daily_url,
    days_between,
    merge,
    read_zip_csv,
    trade_events,
)

INST = Instrument("USUSDT", Decimal("0.000001"), Decimal("1"), "US", "USDT")
MS = 1_789_000_000_000


def archive(lines, name="x.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")
    return buf.getvalue()


def book_rows(blob):
    return list(read_zip_csv(blob, BOOK_TICKER_COLUMNS))


class TestUrls:
    def test_the_path_matches_the_published_layout(self):
        url = daily_url("bookTicker", "ususdt", date(2026, 9, 1))
        assert url.endswith(
            "/futures/um/daily/bookTicker/USUSDT/USUSDT-bookTicker-2026-09-01.zip"
        )

    def test_a_range_covers_both_ends(self):
        days = days_between(date(2026, 9, 1), date(2026, 9, 3))
        assert [d.day for d in days] == [1, 2, 3]

    def test_a_backwards_range_is_refused(self):
        with pytest.raises(ValueError):
            days_between(date(2026, 9, 3), date(2026, 9, 1))


class TestHeaders:
    """The archives carry a header on some symbols and not on others."""

    def test_a_file_without_a_header_is_read_positionally(self):
        blob = archive([f"1,0.0298,100,0.0299,200,{MS},{MS}"])
        assert book_rows(blob)[0]["best_bid_price"] == "0.0298"

    def test_a_true_false_flag_does_not_look_like_a_header(self):
        """Every aggTrades row ends in true or false; calling that a header
        drops the first print of every file."""
        blob = archive([f"5,0.0298,10,1,2,{MS},true", f"6,0.0299,10,3,4,{MS},false"])
        assert len(list(read_zip_csv(blob, AGG_TRADE_COLUMNS))) == 2

    def test_a_file_with_a_header_is_read_by_name(self):
        blob = archive(
            [
                "update_id,best_ask_price,best_ask_qty,best_bid_price,best_bid_qty,transaction_time,event_time",
                f"1,0.0299,200,0.0298,100,{MS},{MS}",
            ]
        )
        # Columns are deliberately out of the documented order: reading
        # positionally here would report the ask as the bid.
        assert book_rows(blob)[0]["best_bid_price"] == "0.0298"

    def test_the_first_data_row_is_not_eaten(self):
        blob = archive([f"{i},0.0298,100,0.0299,200,{MS},{MS}" for i in range(3)])
        assert len(book_rows(blob)) == 3

    def test_an_archive_without_a_csv_says_so(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "nothing here")
        with pytest.raises(ValueError):
            list(read_zip_csv(buf.getvalue(), BOOK_TICKER_COLUMNS))


class TestBook:
    def test_a_row_becomes_a_one_level_snapshot(self):
        blob = archive([f"7,0.0298,100,0.0299,200,{MS},{MS}"])
        events = list(book_events(book_rows(blob), INST))
        assert len(events) == 1
        snap = events[0].event
        assert isinstance(snap, DepthSnapshot)
        assert snap.bids == ((INST.to_ticks("0.0298"), 100),)
        assert snap.asks == ((INST.to_ticks("0.0299"), 200),)
        assert snap.last_update_id == 7

    def test_milliseconds_become_nanoseconds(self):
        blob = archive([f"1,0.0298,100,0.0299,200,{MS},{MS}"])
        assert list(book_events(book_rows(blob), INST))[0].ts_ns == MS * 1_000_000

    def test_a_crossed_or_empty_book_is_dropped(self):
        blob = archive(
            [
                f"1,0.0299,100,0.0298,200,{MS},{MS}",  # bid above ask
                f"2,0,100,0.0299,200,{MS},{MS}",  # no bid
            ]
        )
        assert list(book_events(book_rows(blob), INST)) == []

    def test_an_unparsable_row_is_skipped_rather_than_fatal(self):
        blob = archive(
            [
                "1,not-a-price,100,0.0299,200,x,y",
                f"2,0.0298,100,0.0299,200,{MS},{MS}",
            ]
        )
        assert len(list(book_events(book_rows(blob), INST))) == 1


class TestTape:
    def rows(self, blob):
        return list(read_zip_csv(blob, AGG_TRADE_COLUMNS))

    def test_buyer_as_maker_means_the_seller_crossed(self):
        """Reading this backwards flips every flow measurement, silently."""
        blob = archive([f"5,0.0298,10,1,2,{MS},true"])
        tick = list(trade_events(self.rows(blob), INST))[0].event
        assert isinstance(tick, TradeTick)
        assert tick.aggressor is Side.SELL

    def test_buyer_as_taker_is_a_buy(self):
        blob = archive([f"5,0.0298,10,1,2,{MS},false"])
        assert list(trade_events(self.rows(blob), INST))[0].event.aggressor is Side.BUY

    def test_a_numeric_flag_is_understood_too(self):
        blob = archive([f"5,0.0298,10,1,2,{MS},1"])
        assert list(trade_events(self.rows(blob), INST))[0].event.aggressor is Side.SELL

    def test_price_and_size_survive_the_conversion(self):
        blob = archive([f"5,0.0298,10,1,2,{MS},false"])
        tick = list(trade_events(self.rows(blob), INST))[0].event
        assert tick.price == INST.to_ticks("0.0298")
        assert tick.qty == 10

    def test_a_zero_size_print_is_dropped(self):
        blob = archive([f"5,0.0298,0,1,2,{MS},false"])
        assert list(trade_events(self.rows(blob), INST)) == []


class TestMerge:
    def test_the_two_files_interleave_by_time(self):
        book = iter([Stamped(10, "b1"), Stamped(30, "b2")])
        tape = iter([Stamped(20, "t1"), Stamped(40, "t2")])
        assert [s.event for s in merge(book, tape)] == ["b1", "t1", "b2", "t2"]

    def test_a_print_never_precedes_the_quote_it_traded_against(self):
        book = iter([Stamped(10, "quote")])
        tape = iter([Stamped(10, "print")])
        order = [s.event for s in merge(book, tape)]
        assert order.index("quote") < order.index("print")


class TestEndToEnd:
    def test_a_written_file_replays(self, tmp_path):
        """The writer and the reader have disagreed before, over envelope keys."""
        import json

        from jsboard.feed.replay import RX_KEY, SOURCE_KEY, ReplayFeed, _encode

        blob = archive([f"{i},0.0298,100,0.0299,200,{MS + i},{MS + i}" for i in range(3)])
        path = tmp_path / "hist.jsonl"
        with path.open("w") as fh:
            for stamped in book_events(book_rows(blob), INST):
                row = _encode(stamped.event)
                row[SOURCE_KEY] = "perp"
                row[RX_KEY] = stamped.ts_ns
                fh.write(json.dumps(row) + "\n")

        import asyncio

        async def drain():
            return [e async for e in ReplayFeed(INST, path, speed=0, source="perp").stream()]

        events = [e for e in asyncio.run(drain()) if isinstance(e, DepthSnapshot)]
        assert len(events) == 3
        assert events[0].bids[0][0] == INST.to_ticks("0.0298")


class TestBookFromTape:
    """The days with no published quote stream still have every print."""

    def rows(self, lines):
        return list(read_zip_csv(archive(lines), AGG_TRADE_COLUMNS))

    def stream(self, lines):
        from jsboard.research.vision import book_from_tape

        return list(book_from_tape(iter(self.rows(lines)), INST))

    def test_a_sell_aggressor_sets_the_bid_and_a_buy_sets_the_ask(self):
        events = self.stream(
            [
                f"1,0.0298,10,1,1,{MS},true",  # seller crossed: that was the bid
                f"2,0.0299,10,2,2,{MS + 1},false",  # buyer crossed: that was the ask
            ]
        )
        snaps = [s.event for s in events if isinstance(s.event, DepthSnapshot)]
        assert snaps[-1].bids[0][0] == INST.to_ticks("0.0298")
        assert snaps[-1].asks[0][0] == INST.to_ticks("0.0299")

    def test_no_book_before_both_sides_have_printed(self):
        events = self.stream([f"1,0.0298,10,1,1,{MS},true"])
        assert not any(isinstance(s.event, DepthSnapshot) for s in events)

    def test_the_prints_themselves_are_still_emitted(self):
        events = self.stream([f"1,0.0298,10,1,1,{MS},true"])
        assert [type(s.event) for s in events] == [TradeTick]

    def test_a_crossed_reconstruction_is_suppressed(self):
        """Prints arrive out of order relative to the quote; a bid above the
        ask is not a book anyone can quote against."""
        events = self.stream(
            [
                f"1,0.0299,10,1,1,{MS},true",  # bid at 0.0299
                f"2,0.0298,10,2,2,{MS + 1},false",  # ask below it
            ]
        )
        assert not any(isinstance(s.event, DepthSnapshot) for s in events)
