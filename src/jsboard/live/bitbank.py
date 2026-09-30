"""bitbank's private REST API, and a probe of how fast it answers.

The replay says the quoting edge barely moves while a cancel takes up to
300ms and halves near one second, so the venue's real round trip decides
whether the paper numbers carry over. The probe measures it with orders
that cannot fill: post-only buys well below the best bid, cancelled at once.

Keys come from the environment (BITBANK_API_KEY / BITBANK_API_SECRET) or an
env file readable only by root; they are never printed or logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from pathlib import Path

API_URL = "https://api.bitbank.cc"
PUBLIC_URL = "https://public.bitbank.cc"
ERRORS_URL = "https://github.com/bitbankinc/bitbank-api-docs/blob/master/errors.md"
KEY_VARS = ("BITBANK_API_KEY", "BITBANK_API_SECRET")


class BitbankError(RuntimeError):
    def __init__(self, code: int | None, path: str) -> None:
        self.code = code
        super().__init__(
            f"bitbank がエラーを返しました（コード {code}、{path}）。意味: {ERRORS_URL}"
        )


def sign(secret: str, message: str) -> str:
    """HMAC-SHA256 of nonce+path (GET) or nonce+body (POST), as bitbank signs."""
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def load_keys(env_file: str | None = None) -> tuple[str, str] | None:
    """The key pair from the environment, else from KEY=VALUE lines in a file."""
    values = {k: os.environ.get(k, "") for k in KEY_VARS}
    if env_file and not all(values.values()):
        for line in Path(env_file).read_text().splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name in KEY_VARS and not values[name]:
                values[name] = value.strip().strip('"').strip("'")
    key, secret = (values[k] for k in KEY_VARS)
    return (key, secret) if key and secret else None


@dataclass
class BitbankPrivate:
    key: str = field(repr=False)
    secret: str = field(repr=False)
    session: object = field(default=None, repr=False)
    _nonce: int = 0

    def _next_nonce(self) -> str:
        # Must rise on every call, even two in the same millisecond.
        self._nonce = max(self._nonce + 1, int(time.time() * 1000))
        return str(self._nonce)

    def _headers(self, message_tail: str) -> dict:
        nonce = self._next_nonce()
        return {
            "ACCESS-KEY": self.key,
            "ACCESS-NONCE": nonce,
            "ACCESS-SIGNATURE": sign(self.secret, nonce + message_tail),
            "Content-Type": "application/json",
        }

    async def _send(self, method: str, url: str, headers: dict, body: str | None) -> dict:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=15)
        async with self.session.request(
            method, url, headers=headers, data=body, timeout=timeout
        ) as resp:
            return await resp.json(content_type=None)

    async def _call(self, method: str, path: str, payload: dict | None = None) -> dict:
        if method == "GET":
            reply = await self._send("GET", API_URL + path, self._headers(path), None)
        else:
            body = json.dumps(payload or {}, separators=(",", ":"))
            reply = await self._send("POST", API_URL + path, self._headers(body), body)
        if reply.get("success") != 1:
            raise BitbankError((reply.get("data") or {}).get("code"), path)
        return reply["data"]

    async def assets(self) -> dict[str, Decimal]:
        data = await self._call("GET", "/v1/user/assets")
        return {a["asset"]: Decimal(str(a.get("free_amount", "0"))) for a in data["assets"]}

    async def order(self, pair: str, side: str, price: Decimal, amount: Decimal) -> int:
        """A post-only limit order: rejected rather than filled if it would take."""
        data = await self._call("POST", "/v1/user/spot/order", {
            "pair": pair, "amount": f"{amount:f}", "price": f"{price:f}",
            "side": side, "type": "limit", "post_only": True,
        })
        return int(data["order_id"])

    async def cancel(self, pair: str, order_id: int) -> None:
        await self._call(
            "POST", "/v1/user/spot/cancel_order", {"pair": pair, "order_id": order_id}
        )


@dataclass(frozen=True)
class ProbeResult:
    order_ms: list[float]
    cancel_ms: list[float]

    @staticmethod
    def _describe(values: list[float]) -> str:
        if not values:
            return "—"
        return f"中央値 {statistics.median(values):.0f}ms  最大 {max(values):.0f}ms"

    def verdict(self) -> str:
        if not self.cancel_ms:
            return "取消しを測れませんでした。"
        typical = statistics.median(self.cancel_ms)
        if typical <= 300:
            return "300ms 以内: 録画の検証では成績はほぼ落ちない範囲です（約5%減）。"
        if typical <= 1000:
            return "300ms〜1秒: 取り分が落ち始める範囲です。1秒で約半分でした。"
        return "1秒超: 引っ込めるのが間に合わず、狙われやすくなります。本番は見送りを検討。"

    def text(self) -> str:
        return (
            f"  注文 {self._describe(self.order_ms)}\n"
            f"  取消 {self._describe(self.cancel_ms)}\n"
            f"  → {self.verdict()}"
        )


def far_bid(best_bid: Decimal, tick: Decimal, below: Decimal = Decimal("0.1")) -> Decimal:
    """A price `below` under the best bid, on the tick grid: rests, never fills."""
    return (best_bid * (1 - below) / tick).to_integral_value(ROUND_DOWN) * tick


async def probe(
    api: BitbankPrivate,
    pair: str,
    price: Decimal,
    amount: Decimal,
    rounds: int,
    pause_s: float = 1.0,
) -> ProbeResult:
    """Place and cancel `rounds` unfillable orders, timing each call."""
    order_ms: list[float] = []
    cancel_ms: list[float] = []
    for i in range(rounds):
        order_id = None
        try:
            t0 = time.perf_counter()
            order_id = await api.order(pair, "buy", price, amount)
            t1 = time.perf_counter()
            await api.cancel(pair, order_id)
            t2 = time.perf_counter()
            order_id = None
            order_ms.append((t1 - t0) * 1000)
            cancel_ms.append((t2 - t1) * 1000)
        finally:
            if order_id is not None:
                # A failed cancel would leave a real order resting: try again.
                await api.cancel(pair, order_id)
        if i + 1 < rounds:
            await asyncio.sleep(pause_s)
    return ProbeResult(order_ms, cancel_ms)
