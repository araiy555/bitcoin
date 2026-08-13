"""Reading the instrument spec back off a recording.

`record` and `capture` write two different meta shapes. Picking the wrong one
does not fail loudly — it rescales every price in the file by the ratio of two
tick sizes — so the selection is worth pinning down.
"""

import argparse
import json

import pytest

from jsboard.cli import ConfigError, _instrument_for_recording

FLAT = {
    "symbol": "BTCUSDT",
    "tick_size": "0.01",
    "lot_size": "0.00001",
    "base": "BTC",
    "quote": "USDT",
}
NESTED = {
    "sources": {
        "spot": {**FLAT, "symbol": "WIFUSDT", "tick_size": "0.0001", "lot_size": "1"},
        "perp": {**FLAT, "symbol": "WIFUSDT", "tick_size": "0.0001", "lot_size": "0.1"},
    }
}


def recording(tmp_path, meta=None, name="cap.jsonl"):
    path = tmp_path / name
    path.write_text("")
    if meta is not None:
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta))
    return path


def args(**kw):
    base = {"symbol": "BTCUSDT", "tick_size": None, "lot_size": None, "source": None}
    base.update(kw)
    return argparse.Namespace(**base)


def test_a_record_style_meta_is_read_directly(tmp_path):
    inst = _instrument_for_recording(recording(tmp_path, FLAT), args())
    assert inst.symbol == "BTCUSDT"
    assert str(inst.tick_size) == "0.01"


def test_a_capture_meta_selects_the_named_source(tmp_path):
    inst = _instrument_for_recording(recording(tmp_path, NESTED), args(source="perp"))
    assert inst.symbol == "WIFUSDT"
    assert str(inst.lot_size) == "0.1"


def test_the_other_source_has_its_own_lot_size(tmp_path):
    inst = _instrument_for_recording(recording(tmp_path, NESTED), args(source="spot"))
    assert str(inst.lot_size) == "1"


def test_two_sources_and_no_choice_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="--source"):
        _instrument_for_recording(recording(tmp_path, NESTED), args())


def test_a_single_source_needs_no_choice(tmp_path):
    only = {"sources": {"perp": NESTED["sources"]["perp"]}}
    inst = _instrument_for_recording(recording(tmp_path, only), args())
    assert inst.symbol == "WIFUSDT"


def test_an_unknown_source_names_the_ones_that_exist(tmp_path):
    with pytest.raises(ConfigError, match="perp"):
        _instrument_for_recording(recording(tmp_path, NESTED), args(source="futures"))


def test_without_meta_the_cli_overrides_are_used(tmp_path):
    inst = _instrument_for_recording(
        recording(tmp_path, None), args(symbol="ETHUSDT", tick_size="0.05")
    )
    assert inst.symbol == "ETHUSDT"
    assert str(inst.tick_size) == "0.05"
