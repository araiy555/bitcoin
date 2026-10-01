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
from urllib.parse import urlencode

API_URL = "https://api.bitbank.cc"
PUBLIC_URL = "https://public.bitbank.cc"
ERRORS_URL = "https://github.com/bitbankinc/bitbank-api-docs/blob/master/errors.md"
KEY_VARS = ("BITBANK_API_KEY", "BITBANK_API_SECRET")
NOT_FOUND = 50009
"""What bitbank answers for an order it has not finished accepting: an order
cancelled the instant its POST returns can still be on its way to the book."""
DONE = ("CANCELED_UNFILLED", "CANCELED_PARTIALLY_FILLED", "FULLY_FILLED")


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
        try:
            text = Path(env_file).read_text()
        except OSError:
            text = ""
        for line in text.splitlines():
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
    _lock: asyncio.Lock | None = field(default=None, repr=False)

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

    async def _call(
        self, method: str, path: str, payload: dict | None = None, params: dict | None = None
    ) -> dict:
        # One call at a time. bitbank refuses a nonce lower than the last one
        # it saw (20001), and two calls in flight together can arrive in
        # either order: the live run's order sender and fill reader did.
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if method == "GET":
                path = path + ("?" + urlencode(params) if params else "")
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

    async def cancel_many(self, pair: str, order_ids: list[int]) -> None:
        """Up to 30 cancels in one request, which counts once against the limit."""
        await self._call(
            "POST", "/v1/user/spot/cancel_orders", {"pair": pair, "order_ids": order_ids}
        )

    async def status(self, pair: str, order_id: int) -> str:
        data = await self._call(
            "GET", "/v1/user/spot/order", params={"pair": pair, "order_id": order_id}
        )
        return str(data["status"])

    async def onhand(self) -> dict[str, Decimal]:
        """Total held per asset, including what resting orders have locked."""
        data = await self._call("GET", "/v1/user/assets")
        return {a["asset"]: Decimal(str(a.get("onhand_amount", "0"))) for a in data["assets"]}

    async def trade_history(self, pair: str, since_ms: int) -> list[dict]:
        """Our executions on `pair` since `since_ms`, oldest first."""
        data = await self._call(
            "GET", "/v1/user/spot/trade_history",
            params={"pair": pair, "since": since_ms, "order": "asc", "count": 1000},
        )
        return list(data.get("trades", []))

    async def active_orders(self, pair: str) -> list[int]:
        data = await self._call("GET", "/v1/user/spot/active_orders", params={"pair": pair})
        return [int(o["order_id"]) for o in data.get("orders", [])]


async def wait_live(api, pair: str, order_id: int, timeout_s: float = 5.0) -> bool:
    """Poll until the order rests on the book (or is already done)."""
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            state = await api.status(pair, order_id)
        except BitbankError as exc:
            if exc.code != NOT_FOUND:
                raise
        else:
            if state in ("UNFILLED", "PARTIALLY_FILLED", *DONE):
                return True
        await asyncio.sleep(0.05)
    return False


async def cancel_until_gone(api, pair: str, order_id: int, timeout_s: float = 5.0) -> None:
    """Cancel, retrying while bitbank still says it has no such order."""
    deadline = time.perf_counter() + timeout_s
    delay = 0.05
    while True:
        try:
            await api.cancel(pair, order_id)
            return
        except BitbankError as exc:
            if exc.code != NOT_FOUND:
                try:
                    if await api.status(pair, order_id) in DONE:
                        return
                except BitbankError:
                    pass
                raise
            if time.perf_counter() > deadline:
                raise
        await asyncio.sleep(delay)
        delay = min(delay * 2, 0.5)


@dataclass(frozen=True)
class ProbeResult:
    order_ms: list[float]
    live_ms: list[float]
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
            f"  注文の返事 {self._describe(self.order_ms)}\n"
            f"  板に載るまで {self._describe(self.live_ms)}\n"
            f"  取消し     {self._describe(self.cancel_ms)}\n"
            f"  → {self.verdict()}"
        )


def far_bid(best_bid: Decimal, tick: Decimal, below: Decimal = Decimal("0.1")) -> Decimal:
    """A price `below` under the best bid, on the tick grid: rests, never fills."""
    return (best_bid * (1 - below) / tick).to_integral_value(ROUND_DOWN) * tick


async def probe(
    api,
    pair: str,
    price: Decimal,
    amount: Decimal,
    rounds: int,
    pause_s: float = 1.0,
) -> ProbeResult:
    """Place and cancel `rounds` unfillable orders, timing each step.

    Every order placed is tracked, and a last sweep of the open orders
    cancels any of them still resting, whatever went wrong on the way.
    """
    order_ms: list[float] = []
    live_ms: list[float] = []
    cancel_ms: list[float] = []
    placed: list[int] = []
    try:
        for i in range(rounds):
            t0 = time.perf_counter()
            order_id = await api.order(pair, "buy", price, amount)
            placed.append(order_id)
            t1 = time.perf_counter()
            if await wait_live(api, pair, order_id):
                live_ms.append((time.perf_counter() - t0) * 1000)
            t2 = time.perf_counter()
            await cancel_until_gone(api, pair, order_id)
            t3 = time.perf_counter()
            order_ms.append((t1 - t0) * 1000)
            cancel_ms.append((t3 - t2) * 1000)
            if i + 1 < rounds:
                await asyncio.sleep(pause_s)
    finally:
        stuck = []
        for order_id in sorted(set(placed) & set(await api.active_orders(pair))):
            try:
                await cancel_until_gone(api, pair, order_id)
            except BitbankError:
                stuck.append(order_id)
        if stuck:
            raise RuntimeError(
                f"取り消せなかった注文があります: {stuck}。bitbank の画面で取り消してください。"
            )
    return ProbeResult(order_ms, live_ms, cancel_ms)
