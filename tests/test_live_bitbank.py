"""bitbank's private API client and the order/cancel latency probe."""

import hashlib
import hmac
import json
from decimal import Decimal

import pytest

from jsboard.live.bitbank import (
    BitbankError,
    BitbankPrivate,
    ProbeResult,
    far_bid,
    load_keys,
    probe,
    sign,
)


def test_signature_is_hmac_sha256_of_nonce_and_message():
    expected = hmac.new(b"s3cret", b"1700/v1/user/assets", hashlib.sha256).hexdigest()
    assert sign("s3cret", "1700/v1/user/assets") == expected


def test_keys_come_from_the_environment_or_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("BITBANK_API_KEY", raising=False)
    monkeypatch.delenv("BITBANK_API_SECRET", raising=False)
    env = tmp_path / "jsboard.env"
    env.write_text("SLACK_WEBHOOK_URL=x\nBITBANK_API_KEY=k1\nBITBANK_API_SECRET='s1'\n")
    assert load_keys(str(env)) == ("k1", "s1")
    assert load_keys(None) is None
    monkeypatch.setenv("BITBANK_API_KEY", "k2")
    monkeypatch.setenv("BITBANK_API_SECRET", "s2")
    assert load_keys(str(env)) == ("k2", "s2")


def test_a_probe_price_sits_well_under_the_bid_on_the_tick_grid():
    assert far_bid(Decimal("37.215"), Decimal("0.001")) == Decimal("33.493")


class Recorder(BitbankPrivate):
    def __init__(self, replies):
        super().__init__("key", "secret")
        self.replies = list(replies)
        self.sent = []

    async def _send(self, method, url, headers, body):
        self.sent.append((method, url, headers, body))
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_calls_are_signed_over_the_path_or_the_body():
    api = Recorder([
        {"success": 1, "data": {"assets": [{"asset": "jpy", "free_amount": "1000"}]}},
        {"success": 1, "data": {"order_id": 42}},
    ])
    assert (await api.assets())["jpy"] == Decimal("1000")
    assert await api.order("ada_jpy", "buy", Decimal("33.493"), Decimal("1")) == 42

    method, url, headers, body = api.sent[0]
    assert (method, url, body) == ("GET", "https://api.bitbank.cc/v1/user/assets", None)
    assert headers["ACCESS-SIGNATURE"] == sign("secret", headers["ACCESS-NONCE"] + "/v1/user/assets")

    method, url, headers, body = api.sent[1]
    assert json.loads(body) == {"pair": "ada_jpy", "amount": "1", "price": "33.493",
                                "side": "buy", "type": "limit", "post_only": True}
    assert headers["ACCESS-SIGNATURE"] == sign("secret", headers["ACCESS-NONCE"] + body)
    assert int(api.sent[1][2]["ACCESS-NONCE"]) > int(api.sent[0][2]["ACCESS-NONCE"])


@pytest.mark.asyncio
async def test_an_error_reply_raises_with_its_code():
    api = Recorder([{"success": 0, "data": {"code": 20001}}])
    with pytest.raises(BitbankError, match="20001"):
        await api.assets()


@pytest.mark.asyncio
async def test_the_probe_times_each_round_and_always_cancels():
    class Fake:
        def __init__(self):
            self.open = set()
            self.fail_next_cancel = True

        async def order(self, pair, side, price, amount):
            self.open.add(len(self.open) + 1)
            return max(self.open)

        async def cancel(self, pair, order_id):
            if self.fail_next_cancel:
                self.fail_next_cancel = False
                raise BitbankError(50009, "/v1/user/spot/cancel_order")
            self.open.discard(order_id)

    api = Fake()
    with pytest.raises(BitbankError):
        await probe(api, "ada_jpy", Decimal("30"), Decimal("1"), rounds=3, pause_s=0)
    assert api.open == set()  # the failed cancel was retried, nothing left resting

    result = await probe(api, "ada_jpy", Decimal("30"), Decimal("1"), rounds=3, pause_s=0)
    assert len(result.order_ms) == len(result.cancel_ms) == 3
    assert api.open == set()


def test_the_verdict_follows_the_replay_thresholds():
    assert "ほぼ落ちない" in ProbeResult([80.0], [120.0, 150.0, 900.0]).verdict()
    assert "半分" in ProbeResult([80.0], [600.0]).verdict()
    assert "見送り" in ProbeResult([80.0], [1500.0]).verdict()


@pytest.mark.asyncio
async def test_without_yes_nothing_is_ordered(monkeypatch, capsys, tmp_path):
    import jsboard.cli as cli

    env = tmp_path / "jsboard.env"
    env.write_text("BITBANK_API_KEY=k\nBITBANK_API_SECRET=s\n")
    monkeypatch.delenv("BITBANK_API_KEY", raising=False)
    monkeypatch.delenv("BITBANK_API_SECRET", raising=False)

    async def get_json(session, url):
        if url.endswith("/ticker"):
            return {"data": {"buy": "37.215"}}
        return {"data": {"pairs": [{"name": "ada_jpy", "price_digits": 3, "amount_digits": 4,
                                    "base_asset": "ada", "quote_asset": "jpy",
                                    "unit_amount": "1"}]}}

    sent = []

    async def send(self, method, url, headers, body):
        sent.append(url)
        return {"success": 1, "data": {"assets": [{"asset": "jpy", "free_amount": "500"}]}}

    monkeypatch.setattr(cli, "_get_json", get_json)
    monkeypatch.setattr(BitbankPrivate, "_send", send)
    args = cli.build_parser().parse_args(["bbprobe", "--env-file", str(env)])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "33.493" in out and "まだ注文は出していません" in out
    assert sent == ["https://api.bitbank.cc/v1/user/assets"]
