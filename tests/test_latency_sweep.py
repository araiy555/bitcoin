from jsboard.cli import _sweep_axes, build_parser


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
