from jsboard.cli import (
    _check_sizes,
    _prepare_sweep_defaults,
    _sweep_axes,
    build_instrument,
    build_parser,
)


def test_latency_sweep_uses_only_requested_axes() -> None:
    args = build_parser().parse_args(
        [
            "sweep",
            "us.jsonl",
            "--requotes",
            "100,50,20,10",
            "--latencies",
            "20,10,5,2",
        ]
    )

    assert _sweep_axes(args) == {
        "requote_ms": [100.0, 50.0, 20.0, 10.0],
        "latency_ms": [20.0, 10.0, 5.0, 2.0],
    }


def test_sweep_without_axes_keeps_historical_distance_default() -> None:
    args = build_parser().parse_args(["sweep", "us.jsonl"])

    assert _sweep_axes(args) == {
        "max_distance": [0, 1, 2, 4, None],
    }


def test_latency_sweep_can_explicitly_include_distance() -> None:
    args = build_parser().parse_args(
        [
            "sweep",
            "us.jsonl",
            "--distances",
            "0,1",
            "--requotes",
            "100,10",
            "--latencies",
            "5,1",
        ]
    )

    assert _sweep_axes(args) == {
        "max_distance": [0, 1],
        "requote_ms": [100.0, 10.0],
        "latency_ms": [5.0, 1.0],
    }


def test_sweep_adapts_btc_defaults_to_whole_token_lot() -> None:
    args = build_parser().parse_args(
        [
            "sweep",
            "us.jsonl",
            "--requotes",
            "100,10",
            "--latencies",
            "20,2",
        ]
    )
    instrument = build_instrument("USUSDT", "0.000001", "1")
    axes = _sweep_axes(args)

    changed = _prepare_sweep_defaults(instrument, args, axes)

    assert changed == ["--size 100", "--max-position 1000"]
    assert args.size == "100"
    assert args.max_position == "1000"
    _check_sizes(instrument, args)


def test_sweep_keeps_valid_btc_defaults() -> None:
    args = build_parser().parse_args(["sweep", "btc.jsonl"])
    instrument = build_instrument("BTCUSDT", "0.01", "0.00001")
    axes = _sweep_axes(args)

    assert _prepare_sweep_defaults(instrument, args, axes) == []
    assert args.size == "0.01"
    assert args.max_position == "0.10"


def test_explicit_size_axis_is_not_silently_rewritten() -> None:
    args = build_parser().parse_args(
        ["sweep", "us.jsonl", "--sizes", "0.01,100"]
    )
    instrument = build_instrument("USUSDT", "0.000001", "1")
    axes = _sweep_axes(args)

    # The swept sizes survive untouched, so the 0.01 row is still refused on
    # its own merits. Only the position cap moves, and it has to: left at the
    # 0.10 default the 100 row could not open a single lot either, which would
    # throw away the half of the sweep the user actually asked about.
    assert _prepare_sweep_defaults(instrument, args, axes) == ["--max-position 1000"]
    assert axes["size"] == ["0.01", "100"]


def test_toxicity_threshold_sweep_is_an_explicit_axis() -> None:
    args = build_parser().parse_args(
        [
            "sweep",
            "us.jsonl",
            "--requotes",
            "100",
            "--latencies",
            "2",
            "--toxicity-thresholds",
            "0,0.2,0.4,0.6,0.8",
        ]
    )

    assert _sweep_axes(args) == {
        "requote_ms": [100.0],
        "latency_ms": [2.0],
        "toxicity_threshold": [0.0, 0.2, 0.4, 0.6, 0.8],
    }

def test_a_plain_replay_adapts_to_a_whole_token_lot() -> None:
    """The two refusals a user hits before seeing a single number.

    `replay us.jsonl` on USUSDT used to fail on --size, then fail again on
    --max-position, each time asking for a value only the lot size explains.
    Nobody chose 0.01; it is what the parser fills in when nothing is said.
    """
    from jsboard.cli import _adapt_generic_defaults

    args = build_parser().parse_args(["replay", "us.jsonl"])
    instrument = build_instrument("USUSDT", "0.000001", "1")

    assert _adapt_generic_defaults(instrument, args) == [
        "--size 100",
        "--max-position 1000",
    ]
    _check_sizes(instrument, args)


def test_an_instrument_the_defaults_already_fit_is_left_alone() -> None:
    args = build_parser().parse_args(["replay", "btc.jsonl"])
    instrument = build_instrument("BTCUSDT", "0.01", "0.00001")

    from jsboard.cli import _adapt_generic_defaults

    assert _adapt_generic_defaults(instrument, args) == []
    assert args.size == "0.01"


def test_a_size_the_user_typed_is_still_refused() -> None:
    """Adapting a default is help; adapting an instruction is a silent swap."""
    import pytest

    from jsboard.cli import ConfigError, _adapt_generic_defaults

    args = build_parser().parse_args(["replay", "us.jsonl", "--size", "0.5"])
    instrument = build_instrument("USUSDT", "0.000001", "1")

    assert _adapt_generic_defaults(instrument, args) == []
    with pytest.raises(ConfigError, match="最小単位"):
        _check_sizes(instrument, args)
