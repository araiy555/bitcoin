"""Mark-out split by the signal showing at the fill.

The aggregate mark-out cannot distinguish "every fill is adverse" from "a few
fills are catastrophic and the rest are fine". Those call for opposite
decisions — close the strategy, or quote only in the good band — so the split
is the measurement that matters, and these tests guard the two ways it can
lie: mixing the two sides together, and only ever seeing the band the gate
already allows.
"""

import asyncio
import json

import pytest

from jsboard.research.edge import DEFAULT_EDGES, EdgeStudy, bucket_of

NS = 1_000_000_000


class TestBuckets:
    def test_a_score_lands_in_the_band_that_contains_it(self):
        assert bucket_of(-0.7, DEFAULT_EDGES) == 0
        assert bucket_of(-0.1, DEFAULT_EDGES) == 2
        assert bucket_of(+0.1, DEFAULT_EDGES) == 3
        assert bucket_of(+0.7, DEFAULT_EDGES) == 5

    def test_a_score_past_the_end_is_kept_not_dropped(self):
        """The extremes are the rows most likely to carry the answer."""
        assert bucket_of(-9.0, DEFAULT_EDGES) == 0
        assert bucket_of(+9.0, DEFAULT_EDGES) == len(DEFAULT_EDGES) - 2

    def test_a_missing_score_is_refused(self):
        assert bucket_of(float("nan"), DEFAULT_EDGES) == -1


class TestStudy:
    def _study(self):
        return EdgeStudy(edges=(-1.0, 0.0, 1.0), horizons_s=(1.0,))

    def test_a_buy_followed_by_a_rise_scores_positive(self):
        s = self._study()
        s.on_fill(0, score=0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)

        row = next(r for r in s.rows() if r["side"] == "buy")
        assert row["mo_1s"] == pytest.approx(100.0, rel=1e-3)

    def test_a_sell_followed_by_a_rise_scores_negative(self):
        s = self._study()
        s.on_fill(0, score=0.5, sign=-1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)

        row = next(r for r in s.rows() if r["side"] == "sell")
        assert row["mo_1s"] == pytest.approx(-100.0, rel=1e-3)

    def test_the_two_sides_are_not_averaged_together(self):
        """A directional signal helps one side and hurts the other.

        Averaging them cancels exactly the asymmetry the study exists to find,
        and reports a flat zero for a signal that works.
        """
        s = self._study()
        s.on_fill(0, score=0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.on_fill(0, score=0.5, sign=-1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)

        by_side = {r["side"]: r["mo_1s"] for r in s.rows()}
        assert by_side["buy"] > 0 > by_side["sell"]

    def test_bands_are_reported_separately(self):
        s = self._study()
        s.on_fill(0, score=-0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.on_fill(0, score=+0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)

        bands = [(r["low"], r["high"]) for r in s.rows()]
        assert (-1.0, 0.0) in bands and (0.0, 1.0) in bands

    def test_a_band_with_no_fills_is_absent_rather_than_zero(self):
        """An empty band is no evidence, not evidence of neutrality."""
        s = self._study()
        s.on_fill(0, score=0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)

        assert [(r["low"], r["high"]) for r in s.rows()] == [(0.0, 1.0)]

    def test_a_fill_too_recent_to_have_matured_is_not_counted(self):
        s = self._study()
        s.on_fill(0, score=0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(NS // 2, 101.0)

        assert s.rows() == [] or all(r["n"] == 0 for r in s.rows())

    def test_size_weights_the_average(self):
        s = self._study()
        s.on_fill(0, score=0.5, sign=+1, price_ticks=100.0, weight=99.0, mid_ticks=100.0)
        s.poll(2 * NS, 101.0)
        s.on_fill(2 * NS, score=0.5, sign=+1, price_ticks=100.0, weight=1.0, mid_ticks=100.0)
        s.poll(4 * NS, 100.0)

        row = next(r for r in s.rows() if r["side"] == "buy")
        # 99 lots at +100bps and 1 lot at -100bps: the big one dominates.
        assert row["mo_1s"] > 90.0


class TestTheCommand:
    """Two earlier commands shipped broken on lines no unit test executed."""

    def _recording(self, tmp_path):
        """A book that walks with prints on the way, tagged like xcapture."""
        path = tmp_path / "r.jsonl"
        rows = []
        price = 670_000
        ts = 1_700_000_000_000_000_000
        rows.append(
            {"k": "status", "state": "live", "detail": "t", "ts_ns": ts,
             "src": "binance", "rx_ns": ts}
        )
        for i in range(600):
            ts += 10_000_000
            price += 10 if (i // 5) % 2 == 0 else -10
            rows.append(
                {"k": "snapshot",
                 "bids": [[price - j, 1000] for j in range(1, 4)],
                 "asks": [[price + j, 1000] for j in range(1, 4)],
                 "last_update_id": i + 1, "ts_ns": ts,
                 "src": "binance", "rx_ns": ts}
            )
            ts += 1_000_000
            rows.append(
                {"k": "trade", "price": price + (1 if i % 2 else -1), "qty": 500,
                 "aggressor": 1 if i % 2 else -1, "trade_id": i,
                 "ts_ns": ts, "src": "binance", "rx_ns": ts}
            )
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (tmp_path / "r.jsonl.meta.json").write_text(
            json.dumps({"sources": {"binance": {
                "symbol": "BTCUSDT", "tick_size": "0.1", "lot_size": "0.001",
                "base": "BTC", "quote": "USDT"}}})
        )
        return path

    def _args(self, path, *extra):
        from jsboard.cli import build_parser

        return build_parser().parse_args(
            ["edge", str(path), "--symbol", "BTCUSDT", "--source", "binance",
             "--tick-size", "0.1", "--lot-size", "0.001",
             "--min-edge-bps", "0", "--min-half-spread", "1",
             "--min-book-levels", "2", "--max-position", "0.02",
             "--requote-ms", "1", "--latency-ms", "0.5", *extra]
        )

    def test_it_runs_and_reports_bands(self, tmp_path, capsys):
        from jsboard.cli import cmd_edge

        path = self._recording(tmp_path)
        assert asyncio.run(cmd_edge(self._args(path, "--plain"))) == 0

        out = capsys.readouterr().out
        assert "スコア帯" in out

    def test_the_gate_is_forced_off_so_every_band_can_be_seen(self, tmp_path, capsys):
        """With the gate on, only the allowed band ever fills."""
        from jsboard.cli import cmd_edge

        path = self._recording(tmp_path)
        # A threshold this tight would normally silence one side almost always.
        asyncio.run(cmd_edge(self._args(path, "--plain", "--toxicity-threshold", "0.01")))

        assert "毒性ゲートは無効" in capsys.readouterr().out

    def test_the_caller_s_namespace_is_not_mutated(self, tmp_path):
        """Forcing the gate off must not leak into the caller's settings."""
        from jsboard.cli import cmd_edge

        path = self._recording(tmp_path)
        args = self._args(path, "--plain", "--toxicity-threshold", "0.4")
        asyncio.run(cmd_edge(args))

        assert args.toxicity_threshold == 0.4
