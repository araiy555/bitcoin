"""Testing settings across many published days without filling the disk.

Two mistakes this command exists to prevent were both made for real in this
repo, so the tests are about the discipline rather than the arithmetic:

  * a setting measured on one stretch of market and reported as a result
  * a setting chosen and judged on the same days

The third thing it has to get right is mundane and just as fatal: a day of
BTC quotes is bigger than the free space on the machine, so if the working
file is not gone before the next day is fetched, the run dies halfway with
nothing to show.
"""

import asyncio
import io
import zipfile

import pytest

from jsboard.cli import (
    ConfigError,
    _walk_aggregate,
    _walk_key,
    build_parser,
    cmd_walk,
)

MS = 1_710_000_000_000


def archive(lines, name="x.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, "\n".join(lines) + "\n")
    return buf.getvalue()


def a_day(base_ms, *, moves=400):
    """A book that walks, with a print on every step, so quotes get filled."""
    book, tape = [], []
    price = 29_800
    for i in range(moves):
        price += 10 if (i // 7) % 2 == 0 else -10
        t = base_ms + i * 100
        book.append(f"{i + 1},0.0{price},100,0.0{price + 2},200,{t},{t}")
        tape.append(f"{i + 1},0.0{price},5,{i},{i},{t},{'true' if i % 2 else 'false'}")
    return archive(book), archive(tape)


@pytest.fixture
def served(monkeypatch):
    """Serve a distinct synthetic day per date, and count what was asked for."""
    from jsboard.research import vision

    asked: list[str] = []
    days: dict[str, tuple[bytes, bytes]] = {}

    async def fake_fetch(url):
        asked.append(url)
        stamp = url.rsplit("-", 1)[-1].removesuffix(".zip")
        if stamp not in days:
            offset = int(stamp.replace("-", "")[-2:])
            days[stamp] = a_day(MS + offset * 86_400_000)
        book, tape = days[stamp]
        return book if "bookTicker" in url else tape

    monkeypatch.setattr(vision, "fetch", fake_fetch)
    return asked


def walk_args(tmp_path, *extra):
    return build_parser().parse_args(
        [
            "walk",
            "--symbol", "USUSDT",
            "--tick-size", "0.000001",
            "--lot-size", "1",
            "--work", str(tmp_path / "day.jsonl.gz"),
            "--min-book-levels", "1",
            *extra,
        ]
    )


class TestItRuns:
    def test_a_range_of_days_is_measured(self, tmp_path, served, capsys):
        args = walk_args(tmp_path, "--start", "2024-03-01", "--end", "2024-03-03")

        assert asyncio.run(cmd_walk(args)) == 0

        out = capsys.readouterr().out
        assert "2024-03-01" in out and "2024-03-03" in out

    def test_the_working_file_is_gone_when_it_finishes(self, tmp_path, served):
        args = walk_args(tmp_path, "--start", "2024-03-01", "--end", "2024-03-03")
        asyncio.run(cmd_walk(args))

        assert not (tmp_path / "day.jsonl.gz").exists()
        assert not (tmp_path / "day.jsonl.gz.meta.json").exists()

    def test_keep_leaves_the_last_day_behind(self, tmp_path, served):
        args = walk_args(tmp_path, "--start", "2024-03-01", "--end", "2024-03-02", "--keep")
        asyncio.run(cmd_walk(args))

        assert (tmp_path / "day.jsonl.gz").exists()

    def test_an_interrupted_run_still_clears_the_disk(self, tmp_path, served, monkeypatch):
        """The delete is in `finally` precisely so a crash is not also a full disk."""
        import jsboard.cli as cli

        async def boom(*a, **k):
            raise RuntimeError("replay exploded")

        monkeypatch.setattr(cli, "_run_combos", boom)
        args = walk_args(tmp_path, "--start", "2024-03-01", "--end", "2024-03-02")

        with pytest.raises(RuntimeError):
            asyncio.run(cmd_walk(args))
        assert not (tmp_path / "day.jsonl.gz").exists()

    def test_one_setting_across_days_is_not_multiplied_into_a_distance_sweep(
        self, tmp_path, served
    ):
        """`sweep` invents an axis when given none; measuring days must not."""
        from jsboard.cli import _has_axes

        assert not _has_axes(walk_args(tmp_path, "--start", "2024-03-01"))
        assert _has_axes(walk_args(tmp_path, "--start", "2024-03-01", "--latencies", "1,5"))

    def test_each_day_is_fetched_once_however_many_settings_run(self, tmp_path, served):
        args = walk_args(
            tmp_path, "--start", "2024-03-01", "--end", "2024-03-02", "--latencies", "1,5,20"
        )
        asyncio.run(cmd_walk(args))

        books = [u for u in served if "bookTicker" in u]
        assert len(books) == 2, "the day was re-downloaded per setting"


class TestHoldOut:
    """The +13.21 was chosen on one hour and read off that same hour."""

    def test_the_split_must_actually_split(self, tmp_path, served):
        args = walk_args(
            tmp_path,
            "--start", "2024-03-01",
            "--end", "2024-03-03",
            "--decide-until", "2024-03-09",
        )
        with pytest.raises(ConfigError, match="2つに割"):
            asyncio.run(cmd_walk(args))

    def test_a_malformed_cutoff_is_refused(self, tmp_path, served):
        args = walk_args(tmp_path, "--start", "2024-03-01", "--decide-until", "3月9日")
        with pytest.raises(ConfigError, match="YYYY-MM-DD"):
            asyncio.run(cmd_walk(args))

    def test_both_blocks_are_reported(self, tmp_path, served, capsys):
        args = walk_args(
            tmp_path,
            "--start", "2024-03-01",
            "--end", "2024-03-04",
            "--decide-until", "2024-03-02",
            "--latencies", "1,20",
        )
        assert asyncio.run(cmd_walk(args)) == 0

        out = capsys.readouterr().out
        assert "決定用" in out
        assert "検証用" in out
        assert "順位はつけない" in out

    def test_without_a_split_it_says_the_table_cannot_choose(
        self, tmp_path, served, capsys
    ):
        args = walk_args(
            tmp_path, "--start", "2024-03-01", "--end", "2024-03-02", "--latencies", "1,20"
        )
        asyncio.run(cmd_walk(args))

        assert "選んだ日で成績を測る" in capsys.readouterr().out


class TestAggregate:
    def _rows(self, *values):
        return [
            {"fills": 100, "pre_fee_bps": v, "gap_share": 10.0, "max_maker_bps": v / 2}
            for v in values
        ]

    def test_a_winning_total_built_from_one_day_is_visible_as_such(self):
        """+30 then -10 four times totals positive and loses four days in five."""
        agg = _walk_aggregate(self._rows(30.0, -10.0, -10.0, -10.0, -10.0))

        assert agg["pre_fee_weighted"] == pytest.approx(-2.0)
        assert agg["negative_days"] == 4
        assert agg["pre_fee_worst"] == pytest.approx(-10.0)

    def test_days_that_never_traded_do_not_count_as_flat_days(self):
        """A day with no fills is a day with no evidence, not a day of zero."""
        rows = self._rows(2.0, 2.0)
        rows.append({"fills": 0, "pre_fee_bps": float("nan"), "gap_share": float("nan"),
                     "max_maker_bps": float("nan")})

        agg = _walk_aggregate(rows)

        assert agg["days"] == 3
        assert agg["traded_days"] == 2
        assert agg["pre_fee_mean"] == pytest.approx(2.0)

    def test_the_weighted_mean_follows_where_the_volume_was(self):
        rows = [
            {"fills": 1, "pre_fee_bps": 100.0, "gap_share": 0.0, "max_maker_bps": 50.0},
            {"fills": 99, "pre_fee_bps": 0.0, "gap_share": 0.0, "max_maker_bps": 0.0},
        ]
        agg = _walk_aggregate(rows)

        assert agg["pre_fee_weighted"] == pytest.approx(1.0)
        assert agg["pre_fee_mean"] == pytest.approx(50.0)

    def test_settings_are_keyed_independently_of_their_order(self):
        assert _walk_key({"a": 1, "b": 2}) == _walk_key({"b": 2, "a": 1})
