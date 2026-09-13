"""Bybit's daily archives, and the two-venue recording built from them.

The failures worth guarding here are the silent ones, and two of them are
specific to putting two exchanges in one file:

  * `side` means the aggressor on Bybit and the maker on Binance. The fields
    look interchangeable and mean opposites, so reading one as the other
    flips which venue looks rich — a sign error in the answer, not a crash.
  * Bybit stamps seconds, Binance milliseconds. A wrong unit puts one venue's
    whole day before the other's first print, which reads as a spread of
    several percent that closes instantly.
"""

import asyncio
import gzip
import io
import json
import zipfile
from datetime import date
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.research import bybit_vision
from jsboard.research.vision import ts_ns_auto

INST = Instrument("BTCUSDT", Decimal("0.1"), Decimal("0.001"), "BTC", "USDT")

BYBIT_HEADER = (
    "timestamp,symbol,side,size,price,tickDirection,trdMatchID,"
    "grossValue,homeNotional,foreignNotional"
)
SECONDS = 1_710_720_000.123
MS = 1_710_720_000_123


def bybit_archive(lines, header=BYBIT_HEADER):
    body = "\n".join(([header] if header else []) + lines) + "\n"
    return gzip.compress(body.encode())


def binance_archive(lines, name="x.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")
    return buf.getvalue()


def bybit_rows(blob):
    return list(bybit_vision.read_gzip_csv(blob))


class TestUrl:
    def test_the_date_is_glued_to_the_symbol(self):
        """Not the Binance layout on another host; a dash here 404s every day."""
        url = bybit_vision.daily_url("btcusdt", date(2024, 3, 18))
        assert url.endswith("/trading/BTCUSDT/BTCUSDT2024-03-18.csv.gz")


class TestReading:
    def test_a_header_is_read_by_name(self):
        rows = bybit_rows(
            bybit_archive([f"{SECONDS},BTCUSDT,Buy,0.5,67000.5,ZeroPlusTick,abc,1,1,1"])
        )
        assert rows[0]["price"] == "67000.5"
        assert rows[0]["side"] == "Buy"

    def test_a_headerless_file_is_read_positionally(self):
        rows = bybit_rows(
            bybit_archive(
                [f"{SECONDS},BTCUSDT,Sell,0.5,67000.5,MinusTick,abc,1,1,1"], header=""
            )
        )
        assert rows[0]["side"] == "Sell"

    def test_a_reordered_header_does_not_swap_price_for_size(self):
        header = BYBIT_HEADER.replace("side,size,price", "side,price,size")
        rows = bybit_rows(
            bybit_archive([f"{SECONDS},BTCUSDT,Buy,67000.5,0.5,Z,abc,1,1,1"], header)
        )
        assert rows[0]["price"] == "67000.5"
        assert rows[0]["size"] == "0.5"


class TestAggressor:
    """Binance says `is_buyer_maker`; Bybit says `side`. They are opposites."""

    def _one(self, side):
        blob = bybit_archive([f"{SECONDS},BTCUSDT,{side},0.5,67000.0,Z,abc,1,1,1"])
        return next(iter(bybit_vision.trade_events(bybit_vision.read_gzip_csv(blob), INST)))

    def test_buy_means_the_buyer_crossed(self):
        assert self._one("Buy").event.aggressor is Side.BUY

    def test_sell_means_the_seller_crossed(self):
        assert self._one("Sell").event.aggressor is Side.SELL

    def test_the_reconstructed_touch_puts_the_bid_below_the_ask(self):
        """A seller crossing took the bid; a buyer crossing lifted the ask."""
        blob = bybit_archive(
            [
                f"{SECONDS},BTCUSDT,Sell,0.5,67000.0,Z,a,1,1,1",
                f"{SECONDS + 1},BTCUSDT,Buy,0.5,67001.0,Z,b,1,1,1",
            ]
        )
        books = [
            s.event
            for s in bybit_vision.book_events(bybit_vision.read_gzip_csv(blob), INST)
            if isinstance(s.event, DepthSnapshot)
        ]
        assert books, "no touch reconstructed"
        bid = books[-1].bids[0][0]
        ask = books[-1].asks[0][0]
        assert bid < ask
        assert INST.to_ticks("67000.0") == bid


class TestTimestamps:
    def test_seconds_milliseconds_and_micros_land_on_the_same_instant(self):
        ns = int(SECONDS * 1e9)
        assert ts_ns_auto(str(SECONDS)) == pytest.approx(ns, abs=1_000)
        assert ts_ns_auto(str(MS)) == pytest.approx(ns, abs=1_000_000)
        assert ts_ns_auto(str(MS * 1_000)) == pytest.approx(ns, abs=1_000_000)

    def test_a_bybit_print_is_not_read_as_1970(self):
        stamped = next(
            iter(
                bybit_vision.trade_events(
                    bybit_vision.read_gzip_csv(
                        bybit_archive([f"{SECONDS},BTCUSDT,Buy,0.5,67000.0,Z,a,1,1,1"])
                    ),
                    INST,
                )
            )
        )
        assert stamped.ts_ns > 1_700_000_000_000_000_000


class TestBadRows:
    def test_a_malformed_price_does_not_end_the_day(self):
        blob = bybit_archive(
            [
                f"{SECONDS},BTCUSDT,Buy,0.5,not-a-price,Z,a,1,1,1",
                f"{SECONDS + 1},BTCUSDT,Buy,0.5,67000.0,Z,b,1,1,1",
            ]
        )
        events = list(bybit_vision.trade_events(bybit_vision.read_gzip_csv(blob), INST))
        assert len(events) == 1

    def test_a_zero_size_print_is_dropped(self):
        blob = bybit_archive([f"{SECONDS},BTCUSDT,Buy,0,67000.0,Z,a,1,1,1"])
        assert list(bybit_vision.trade_events(bybit_vision.read_gzip_csv(blob), INST)) == []


class TestTheCommand:
    """`vision` died twice on lines no unit test executed. This runs the body."""

    def _serve(self, monkeypatch, *, bybit_present=True, binance_book=True):
        from jsboard.research import vision

        book = binance_archive([f"1,66999.0,1.0,67001.0,2.0,{MS},{MS}"])
        tape = binance_archive(
            [
                f"1,67000.0,0.01,1,1,{MS},true",
                f"2,67000.5,0.01,2,2,{MS + 5},false",
            ]
        )
        by = bybit_archive(
            [
                f"{SECONDS},BTCUSDT,Sell,0.01,67010.0,Z,a,1,1,1",
                f"{SECONDS + 0.002},BTCUSDT,Buy,0.01,67012.0,Z,b,1,1,1",
            ]
        )

        async def binance_fetch(url):
            if "bookTicker" in url:
                return book if binance_book else None
            return tape

        async def bybit_fetch(url):
            return by if bybit_present else None

        monkeypatch.setattr(vision, "fetch", binance_fetch)
        monkeypatch.setattr(bybit_vision, "fetch", bybit_fetch)

    def _run(self, tmp_path, monkeypatch, **kw):
        from jsboard.cli import build_parser, cmd_xvision

        self._serve(monkeypatch, **kw)
        out = tmp_path / "x.jsonl"
        args = build_parser().parse_args(
            [
                "xvision",
                "--symbol", "BTCUSDT",
                "--start", "2024-03-18",
                "--out", str(out),
                "--tick-size", "0.1",
                "--lot-size", "0.001",
            ]
        )
        assert asyncio.run(cmd_xvision(args)) == 0
        return out

    def _lines(self, out):
        return [json.loads(line) for line in out.read_text().splitlines()]

    def test_both_venues_land_in_one_file(self, tmp_path, monkeypatch):
        rows = self._lines(self._run(tmp_path, monkeypatch))
        sources = {r["src"] for r in rows}
        assert sources == {"binance", "bybit"}

    def test_the_meta_carries_both_specs(self, tmp_path, monkeypatch):
        self._run(tmp_path, monkeypatch)
        meta = json.loads((tmp_path / "x.jsonl.meta.json").read_text())
        assert set(meta["sources"]) == {"binance", "bybit"}
        assert meta["sources"]["bybit"]["venue"] == "bybit"

    def test_each_venue_announces_itself_before_its_own_first_event(
        self, tmp_path, monkeypatch
    ):
        """xarb holds everything until both sides report live."""
        rows = self._lines(self._run(tmp_path, monkeypatch))
        seen: set[str] = set()
        for row in rows:
            source = row["src"]
            if source not in seen:
                assert row.get("state") == "live", f"{source} traded before saying live"
                seen.add(source)
        assert seen == {"binance", "bybit"}

    def test_the_file_is_ordered_in_time_across_venues(self, tmp_path, monkeypatch):
        stamps = [r["rx_ns"] for r in self._lines(self._run(tmp_path, monkeypatch))]
        assert stamps == sorted(stamps)

    def test_the_two_venues_land_in_the_same_era(self, tmp_path, monkeypatch):
        """Seconds read as milliseconds would separate them by 50 years."""
        rows = self._lines(self._run(tmp_path, monkeypatch))
        per_venue = {}
        for row in rows:
            per_venue.setdefault(row["src"], []).append(row["rx_ns"])
        spread_s = abs(
            min(per_venue["binance"]) - min(per_venue["bybit"])
        ) / 1e9
        assert spread_s < 60.0

    def test_a_day_missing_on_one_venue_is_refused_rather_than_half_measured(
        self, tmp_path, monkeypatch
    ):
        from jsboard.cli import ConfigError, build_parser, cmd_xvision

        self._serve(monkeypatch, bybit_present=False)
        args = build_parser().parse_args(
            [
                "xvision",
                "--symbol", "BTCUSDT",
                "--start", "2024-03-18",
                "--out", str(tmp_path / "x.jsonl"),
                "--tick-size", "0.1",
                "--lot-size", "0.001",
            ]
        )
        with pytest.raises(ConfigError):
            asyncio.run(cmd_xvision(args))

    def test_it_still_works_when_binance_has_no_quote_file(self, tmp_path, monkeypatch):
        rows = self._lines(self._run(tmp_path, monkeypatch, binance_book=False))
        assert {r["src"] for r in rows} == {"binance", "bybit"}

    def test_what_it_writes_can_be_replayed_as_two_sources(self, tmp_path, monkeypatch):
        from jsboard.feed.replay import iter_tagged

        out = self._run(tmp_path, monkeypatch)
        tagged = list(iter_tagged(out))

        assert {source for source, _ in tagged} == {"binance", "bybit"}
        assert any(isinstance(e, TradeTick) for _, e in tagged)
        assert any(isinstance(e, DepthSnapshot) for _, e in tagged)


class TestCompressed:
    """The default output is gzipped; a day of two venues does not fit twice."""

    def test_a_gzipped_recording_is_readable_by_the_cross_venue_reader(
        self, tmp_path, monkeypatch
    ):
        from jsboard.cli import build_parser, cmd_xvision
        from jsboard.feed.replay import iter_tagged, iter_tagged_timed

        TestTheCommand()._serve(monkeypatch)
        out = tmp_path / "x.jsonl.gz"
        args = build_parser().parse_args(
            [
                "xvision",
                "--symbol", "BTCUSDT",
                "--start", "2024-03-18",
                "--out", str(out),
                "--tick-size", "0.1",
                "--lot-size", "0.001",
            ]
        )
        assert asyncio.run(cmd_xvision(args)) == 0

        assert {s for s, _ in iter_tagged(out)} == {"binance", "bybit"}
        assert {s for s, _, _ in iter_tagged_timed(out)} == {"binance", "bybit"}

    def test_the_meta_sits_where_xarb_looks_for_it(self, tmp_path, monkeypatch):
        from jsboard.cli import build_parser, cmd_xvision

        TestTheCommand()._serve(monkeypatch)
        out = tmp_path / "x.jsonl.gz"
        args = build_parser().parse_args(
            [
                "xvision",
                "--symbol", "BTCUSDT",
                "--start", "2024-03-18",
                "--out", str(out),
                "--tick-size", "0.1",
                "--lot-size", "0.001",
            ]
        )
        asyncio.run(cmd_xvision(args))

        # xarb builds this path itself; a mismatch means a download that
        # finished successfully and then cannot be replayed.
        assert out.with_suffix(out.suffix + ".meta.json").exists()
