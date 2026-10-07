"""GMO Coin's private REST API, for the leverage (取引所レバレッジ) book.

Only what the follow test needs: a market order that opens a position, a
market order that closes exactly that position, the fills of an order, and
the open positions. Keys come from the environment (GMO_API_KEY /
GMO_API_SECRET) or an env file readable only by root; they are never
printed or logged.

Signing, as GMO documents it: HMAC-SHA256 over timestamp + method + path +
body, with the path taken without its query string.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

API_URL = "https://api.coin.z.com/private"
KEY_VARS = ("GMO_API_KEY", "GMO_API_SECRET")


class GmoError(RuntimeError):
    def __init__(self, codes: list[str], path: str, text: str = "") -> None:
        self.codes = codes
        super().__init__(f"GMO がエラーを返しました（{', '.join(codes) or '不明'}、{path}）{text}")


def sign(secret: str, text: str) -> str:
    return hmac.new(secret.encode(), text.encode(), hashlib.sha256).hexdigest()


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


@dataclass(frozen=True)
class Fill:
    price: float
    size: float
    position_id: int | None
    fee: float
    loss_gain: float
    timestamp: str


@dataclass
class GmoPrivate:
    key: str = field(repr=False)
    secret: str = field(repr=False)
    session: object = field(default=None, repr=False)
    _last_ts: int = 0
    _lock: asyncio.Lock | None = field(default=None, repr=False)

    def _headers(self, method: str, path: str, body: str) -> dict:
        # Rising even within one millisecond, so two signatures never share one.
        self._last_ts = max(self._last_ts + 1, int(time.time() * 1000))
        ts = str(self._last_ts)
        return {
            "API-KEY": self.key,
            "API-TIMESTAMP": ts,
            "API-SIGN": sign(self.secret, ts + method + path + body),
            "Content-Type": "application/json",
        }

    async def _send(self, method: str, url: str, headers: dict, body: str | None) -> dict:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=10)
        async with self.session.request(method, url, headers=headers, data=body,
                                        timeout=timeout) as resp:
            return await resp.json(content_type=None)

    async def _call(self, method: str, path: str, payload: dict | None = None,
                    params: dict | None = None):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            body = json.dumps(payload, separators=(",", ":")) if payload is not None else ""
            url = API_URL + path + ("?" + urlencode(params) if params else "")
            reply = await self._send(method, url, self._headers(method, path, body), body or None)
        if not isinstance(reply, dict) or reply.get("status") != 0:
            msgs = (reply or {}).get("messages") or [] if isinstance(reply, dict) else []
            codes = [str(m.get("message_code", "")) for m in msgs if isinstance(m, dict)]
            text = " ".join(str(m.get("message_string", "")) for m in msgs if isinstance(m, dict))
            raise GmoError(codes, path, text)
        return reply.get("data")

    async def market_open(self, symbol: str, side: str, size: str) -> str:
        """A market order that opens a position; its order id."""
        data = await self._call("POST", "/v1/order", {
            "symbol": symbol, "side": side, "executionType": "MARKET", "size": size})
        return str(data)

    async def market_close(self, symbol: str, side: str, positions: list[tuple[int, str]]) -> str:
        """A market order that closes exactly these positions (id, size)."""
        data = await self._call("POST", "/v1/closeOrder", {
            "symbol": symbol, "side": side, "executionType": "MARKET",
            "settlePosition": [{"positionId": pid, "size": size} for pid, size in positions]})
        return str(data)

    async def fills(self, order_id: str) -> list[Fill]:
        data = await self._call("GET", "/v1/executions", params={"orderId": order_id}) or {}
        out = []
        for e in data.get("list") or []:
            out.append(Fill(
                price=float(e["price"]), size=float(e["size"]),
                position_id=int(e["positionId"]) if e.get("positionId") is not None else None,
                fee=float(e.get("fee") or 0), loss_gain=float(e.get("lossGain") or 0),
                timestamp=str(e.get("timestamp", "")),
            ))
        return out

    async def open_positions(self, symbol: str) -> list[dict]:
        data = await self._call("GET", "/v1/openPositions", params={"symbol": symbol}) or {}
        return list(data.get("list") or [])
