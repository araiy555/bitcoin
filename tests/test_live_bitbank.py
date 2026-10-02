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


class Venue:
    """Accepts an order a few polls after the POST returns, as bitbank did."""

    def __init__(self, accept_after=2, cancel_error=None):
        self.accept_after = accept_after
        self.cancel_error = cancel_error
        self.polls = {}
        self.open = set()
        self.next_id = 100

    async def order(self, pair, side, price, amount):
        self.next_id += 1
        self.polls[self.next_id] = 0
        return self.next_id

    def _accepted(self, order_id):
        self.polls[order_id] += 1
        if self.polls[order_id] >= self.accept_after:
            self.open.add(order_id)
            return True
        return False

    async def status(self, pair, order_id):
        if order_id not in self.open and not self._accepted(order_id):
            raise BitbankError(50009, "/v1/user/spot/order")
        return "UNFILLED"

    async def cancel(self, pair, order_id):
        if self.cancel_error:
            raise BitbankError(self.cancel_error, "/v1/user/spot/cancel_order")
        if order_id not in self.open and not self._accepted(order_id):
            raise BitbankError(50009, "/v1/user/spot/cancel_order")
        self.open.discard(order_id)

    async def active_orders(self, pair):
        return sorted(self.open) + [7]  # 7: someone else's order, never touched


@pytest.mark.asyncio
async def test_the_probe_waits_for_the_book_then_cancels_every_order():
    venue = Venue(accept_after=3)
    result = await probe(venue, "ada_jpy", Decimal("30"), Decimal("1"), rounds=3, pause_s=0)
    assert len(result.order_ms) == len(result.live_ms) == len(result.cancel_ms) == 3
    assert venue.open == set()


@pytest.mark.asyncio
async def test_a_cancel_sent_before_the_order_lands_is_retried():
    from jsboard.live.bitbank import cancel_until_gone

    venue = Venue(accept_after=4)
    order_id = await venue.order("ada_jpy", "buy", Decimal("30"), Decimal("1"))
    await cancel_until_gone(venue, "ada_jpy", order_id)
    assert venue.open == set() and venue.polls[order_id] == 4


@pytest.mark.asyncio
async def test_orders_left_by_an_error_are_swept_at_the_end():
    venue = Venue(accept_after=1, cancel_error=50010)
    with pytest.raises(RuntimeError, match="取り消せなかった注文"):
        await probe(venue, "ada_jpy", Decimal("30"), Decimal("1"), rounds=2, pause_s=0)
    venue.cancel_error = None
    # The sweep ran while cancels were failing; a later run's sweep clears it.
    await probe(venue, "ada_jpy", Decimal("30"), Decimal("1"), rounds=1, pause_s=0)
    assert 101 in venue.open  # placed by the first run, not this one: left alone


def test_the_verdict_follows_the_replay_thresholds():
    assert "ほぼ落ちない" in ProbeResult([80.0], [90.0], [120.0, 150.0, 900.0]).verdict()
    assert "半分" in ProbeResult([80.0], [90.0], [600.0]).verdict()
    assert "見送り" in ProbeResult([80.0], [90.0], [1500.0]).verdict()


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
        if "active_orders" in url:
            return {"success": 1, "data": {"orders": []}}
        return {"success": 1, "data": {"assets": [{"asset": "jpy", "free_amount": "500"}]}}

    monkeypatch.setattr(cli, "_get_json", get_json)
    monkeypatch.setattr(BitbankPrivate, "_send", send)
    args = cli.build_parser().parse_args(["bbprobe", "--env-file", str(env)])
    assert await args.func(args) == 0
    out = capsys.readouterr().out
    assert "33.493" in out and "まだ注文は出していません" in out
    assert sent == ["https://api.bitbank.cc/v1/user/assets",
                    "https://api.bitbank.cc/v1/user/spot/active_orders?pair=ada_jpy"]


@pytest.mark.asyncio
async def test_calls_reach_the_venue_one_at_a_time_in_nonce_order():
    import asyncio

    arrived = []

    class Wire(BitbankPrivate):
        def __init__(self):
            super().__init__("key", "secret")

        async def _send(self, method, url, headers, body):
            nonce = int(headers["ACCESS-NONCE"])
            # A later call that overtook an earlier one would arrive first.
            await asyncio.sleep(0.02 if len(arrived) % 2 == 0 else 0)
            arrived.append(nonce)
            return {"success": 1, "data": {"assets": [], "trades": [], "orders": []}}

    api = Wire()
    await asyncio.gather(api.assets(), api.trade_history("ada_jpy", 0),
                         api.active_orders("ada_jpy"), api.assets())
    assert arrived == sorted(arrived) and len(set(arrived)) == 4


@pytest.mark.asyncio
async def test_a_post_only_order_the_venue_refused_is_not_treated_as_resting():
    from jsboard.live.bitbank import OrderRefused

    api = Recorder([{"success": 1, "data": {"order_id": 77, "status": "REJECTED"}}])
    with pytest.raises(OrderRefused):
        await api.order("ada_jpy", "buy", Decimal("37"), Decimal("1"))
