"""Parsing sweep axes.

A mistyped axis is the dangerous case: it would run the same configuration
several times and print a table that looks like a comparison.
"""

import argparse

import pytest

from jsboard.cli import ConfigError, _cast_like, _grid, _label, _sweep_axes


def args(**kw):
    base = {
        "distances": "",
        "sizes": "",
        "axis": None,
        "gamma": 0.6,
        "levels": 3,
        "size": "0.01",
        "max_distance": None,
        "max_position": "0.10",
    }
    base.update(kw)
    return argparse.Namespace(**base)


class TestGrid:
    def test_none_means_the_uncapped_setting(self):
        assert _grid("0,2,none", int) == [0, 2, None]

    def test_blanks_and_spacing_are_forgiven(self):
        assert _grid(" 1 , , 2 ", int) == [1, 2]

    def test_off_is_a_synonym_for_none(self):
        assert _grid("off", int) == [None]


class TestCastLike:
    def test_a_float_setting_stays_float(self):
        assert _cast_like(0.6)("3") == 3.0

    def test_an_int_setting_stays_int(self):
        assert _cast_like(3)("5") == 5

    def test_a_size_stays_a_string_so_decimal_keeps_it_exact(self):
        assert _cast_like("0.01")("0.001") == "0.001"

    def test_a_none_default_narrows_to_the_tightest_type(self):
        guess = _cast_like(None)
        assert guess("4") == 4
        assert guess("4.5") == 4.5
        assert guess("wide") == "wide"


class TestAxes:
    def test_the_shorthands_map_onto_real_settings(self):
        axes = _sweep_axes(args(distances="0,2", sizes="1,2"))
        assert axes == {"max_distance": [0, 2], "size": ["1", "2"]}

    def test_axis_reaches_anything_else(self):
        axes = _sweep_axes(args(axis=["gamma=0.6,4"]))
        assert axes == {"gamma": [0.6, 4.0]}

    def test_dashes_are_accepted_the_way_the_flag_is_written(self):
        assert _sweep_axes(args(axis=["max-position=1,2"])) == {"max_position": ["1", "2"]}

    def test_a_name_that_is_not_a_setting_is_refused(self):
        with pytest.raises(ConfigError, match="gama"):
            _sweep_axes(args(axis=["gama=1"]))

    def test_a_missing_value_list_is_refused(self):
        with pytest.raises(ConfigError, match="name=v1"):
            _sweep_axes(args(axis=["gamma"]))

    def test_an_empty_value_list_is_refused(self):
        with pytest.raises(ConfigError, match="name=v1"):
            _sweep_axes(args(axis=["gamma="]))

    def test_axes_combine_rather_than_replace(self):
        axes = _sweep_axes(args(distances="0", axis=["gamma=1", "levels=1,3"]))
        assert list(axes) == ["max_distance", "gamma", "levels"]

    def test_nothing_specified_is_no_axes(self):
        assert _sweep_axes(args()) == {}


class TestLabel:
    def test_zero_distance_reads_as_the_touch(self):
        assert _label("max_distance", 0) == "touch"

    def test_uncapped_distance_reads_as_none(self):
        assert _label("max_distance", None) == "none"

    def test_other_settings_print_plainly(self):
        assert _label("gamma", 0.6) == "0.6"
        assert _label("levels", 3) == "3"


class TestANameThatIsNotASetting:
    """The flag and the field it feeds are not always spelled the same.

    `--min-half-spread` lands on the namespace as `min_half_spread`, but the
    quoter field it sets is `min_half_spread_ticks`. Reading the field name
    out of the source and handing it to --axis is the natural mistake, and it
    is one edit from correct — so the error should say which edit.
    """

    def _axes_for(self, spec):
        from jsboard.cli import ConfigError, _sweep_axes, build_parser

        args = build_parser().parse_args(["sweep", "x.jsonl", "--axis", spec])
        try:
            _sweep_axes(args)
        except ConfigError as exc:
            return str(exc)
        return ""

    def test_a_near_miss_names_the_real_setting(self):
        assert "min_half_spread" in self._axes_for("min_half_spread_ticks=1,2")

    def test_a_typo_is_corrected(self):
        assert "gamma" in self._axes_for("gama=1,2")

    def test_something_unrecognisable_lists_what_is_available(self):
        message = self._axes_for("zzzzzzzz=1,2")
        assert "指定できるもの" in message
        assert "gamma" in message

    def test_plumbing_is_not_offered_as_a_setting(self):
        """`--axis plain=...` would run the same configuration twice."""
        message = self._axes_for("zzzzzzzz=1,2")
        for plumbing in ("func", "plain", "path", "axis"):
            assert f" {plumbing}," not in message
