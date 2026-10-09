"""Follow Bybit and Binance on GMO's leverage book with real orders, and
record every step so the live run can be held against the replay.

The rule is the one the replay tested on unseen data: when a lead perp's
mid has moved at least `threshold_bps` more than GMO's over the last
second, and the lead itself moved that far that way, buy (or sell) on GMO
at market, and close exactly that position at market `hold_s` later. One
position at a time. The run stops for good once realised losses reach
`max_loss_jpy`.

This is a measurement first. Each trade records the signal, every
timestamp from sending the order to seeing the fill, the prices GMO
actually gave, and next to them what the replay would have assumed at the
same moments (the book walked `latency_ms` after the signal, and `hold_s`
after that), so the gap between the two is visible trade by trade.

Without live keys the same code runs against `ShadowApi`, which fills at
the live book and sends nothing.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass, field

from .gmo import Fill, GmoError

NS = 1_000_000_000


class Detector:
    """The replay's signal, one lead at a time; a lead re-arms once its gap
    has closed below the threshold, so one move is one signal."""

    def __init__(self, threshold_bps: float = 8.0, window_ms: float = 1000.0) -> None:
        self.threshold = threshold_bps
        self.window = int(window_ms * 1e6)
        self.hist: dict[str, deque] = {}
        self.armed: dict[str, bool] = {}

    def reset(self, src: str) -> None:
        """Forget a source's history, as after a reconnect: a gap across a
        disconnection is not a move."""
        self.hist.pop(src, None)

    def _ago(self, src: str, now: int) -> float | None:
        h = self.hist.get(src)
        if not h:
            return None
        while len(h) >= 2 and h[1][0] <= now - self.window:
            h.popleft()
        return h[0][1] if h[0][0] <= now - self.window else None

    def update(self, src: str, rx: int, mid: float | None) -> tuple[str, int, float] | None:
        """(lead, +1 buy / -1 sell, gap bps) when `src` just gave a signal."""
        if mid is None:
            return None
        h = self.hist.setdefault(src, deque())
        if not h or h[-1][1] != mid:
            h.append((rx, mid))
        if src == "gmo":
            self._ago("gmo", rx)
            return None
        gmo = self.hist.get("gmo")
        lead_then, gmo_then = self._ago(src, rx), self._ago("gmo", rx)
        if not gmo or not lead_then or not gmo_then:
            return None
        lead_move = math.log(mid / lead_then) * 1e4
        gap = lead_move - math.log(gmo[-1][1] / gmo_then) * 1e4
        t = self.threshold
        if abs(gap) < t or abs(lead_move) < t or lead_move * gap <= 0:
            self.armed[src] = True
            return None
        if not self.armed.get(src, True):
            return None
        self.armed[src] = False
        return src, (1 if gap > 0 else -1), gap


class ShadowApi:
    """Sends nothing. Each order is filled at the live book as it stands
    `latency_s` after it was sent, as a real order would only land then:
    filling at the instant of the signal flattered the shadow by about
    1.3bps a trade, the move GMO makes in the first 0.2 s."""

    def __init__(self, walk, latency_s: float = 0.2, sleep=asyncio.sleep) -> None:
        self.walk = walk  # (side +1/-1, size) -> price or None
        self.latency_s = latency_s
        self.sleep = sleep
        self._fills: dict[str, list[Fill]] = {}
        self._seq = 0

    async def market_open(self, symbol: str, side: str, size: str) -> str:
        return await self._fill(1 if side == "BUY" else -1, float(size), opening=True)

    async def market_close(self, symbol: str, side: str, positions) -> str:
        return await self._fill(1 if side == "BUY" else -1, sum(float(s) for _, s in positions),
                                opening=False)

    async def _fill(self, side: int, size: float, opening: bool) -> str:
        if self.latency_s:
            await self.sleep(self.latency_s)
        self._seq += 1
        oid = f"shadow-{self._seq}"
        price = self.walk(side, size)
        if price is None:
            raise GmoError(["SHADOW-NO-BOOK"], "/shadow")
        self._fills[oid] = [Fill(price, size, self._seq if opening else None, 0.0, 0.0, "")]
        return oid

    async def fills(self, order_id: str) -> list[Fill]:
        return self._fills.get(order_id, [])

    async def open_positions(self, symbol: str) -> list[dict]:
        return []


@dataclass
class Tally:
    n: int = 0
    wins: int = 0
    pnl: float = 0.0
    notional: float = 0.0
    expected_pnl: float = 0.0
    slip_bps: float = 0.0
    send_to_ack_ms: float = 0.0
    errors: int = 0
    signals: int = 0
    skipped: int = 0
    """Signals that came while a trade was already open, so were not taken."""

    def text(self) -> str:
        head = f"合図 {self.signals}回（取引中で見送り {self.skipped}回）  "
        if not self.n:
            return head + f"取引 0回（エラー {self.errors}回）"
        bps = self.pnl / self.notional * 1e4
        exp = self.expected_pnl / self.notional * 1e4
        return head + (f"取引 {self.n}回  勝率 {self.wins / self.n:.0%}  損益 {self.pnl:+,.1f}円（{bps:+.1f}bps）"
                f"  検証の計算なら {exp:+.1f}bps  入りの値段のずれ 平均 {self.slip_bps / self.n:+.1f}bps"
                f"  注文→受付 平均 {self.send_to_ack_ms / self.n:.0f}ms  エラー {self.errors}回")


@dataclass
class Follower:
    api: object
    symbol: str
    size: str
    walk: object
    """(side +1/-1, size) -> the price the live book gives now, or None."""
    touch: object
    """() -> (best bid, best ask) now, or (None, None)."""
    hold_s: float = 10.0
    latency_ms: float = 200.0
    max_loss_jpy: float = 500.0
    max_trades: int = 1000
    log: object = None
    notify: object = None
    clock: object = time.time_ns
    sleep: object = asyncio.sleep
    busy: bool = False
    stopped: str | None = None
    open_positions: list = field(default_factory=list)
    open_back: str = "SELL"
    """The side that closes what is open."""
    tally: Tally = field(default_factory=Tally)
    _consecutive_errors: int = 0

    def ready(self) -> bool:
        return not self.busy and self.stopped is None

    def saw_signal(self, lead: str, side: int, gap: float, signal_ns: int, extra=None) -> bool:
        """Count a signal; record it if a trade in hand means it is passed
        over. True when it can be taken now."""
        if self.stopped is not None:
            return False
        self.tally.signals += 1
        if self.busy:
            self.tally.skipped += 1
            self._record({"skipped": True, "signal_ns": signal_ns, "lead": lead,
                          "side": "BUY" if side > 0 else "SELL", "gap_bps": round(gap, 2),
                          **(extra or {})})
            return False
        return True

    async def _say(self, text: str) -> None:
        if self.notify:
            await self.notify(text)

    def _record(self, rec: dict) -> None:
        if self.log:
            self.log(rec)

    async def stop(self, reason: str) -> None:
        if self.stopped is None:
            self.stopped = reason
            await self._say(f":octagonal_sign: GMO 後追いを止めました: {reason}\n  {self.tally.text()}")

    async def _wait_fills(self, order_id: str, size: float, timeout_s: float = 5.0) -> list[Fill]:
        deadline = self.clock() + int(timeout_s * NS)
        while True:
            fills = await self.api.fills(order_id)
            if sum(f.size for f in fills) >= size - 1e-9:
                return fills
            if self.clock() >= deadline:
                return fills
            await self.sleep(0.2)  # five reads a second, inside GMO's private API limit

    async def _expected(self, side: int, signal_ns: int, size: float) -> dict:
        """What the replay assumes: the book `latency_ms` after the signal,
        and again `hold_s` after that."""
        await self.sleep(max(0.0, (signal_ns + self.latency_ms * 1e6 - self.clock()) / NS))
        entry = self.walk(side, size)
        await self.sleep(self.hold_s)
        exit_ = self.walk(-side, size)
        return {"expected_entry": entry, "expected_exit": exit_}

    async def trade(self, lead: str, side: int, gap: float, signal_ns: int,
                    extra: dict | None = None) -> dict | None:
        if not self.ready():
            return None
        self.busy = True
        try:
            return await self._trade(lead, side, gap, signal_ns, extra or {})
        finally:
            self.busy = False

    async def _trade(self, lead: str, side: int, gap: float, signal_ns: int,
                     extra: dict) -> dict | None:
        size = float(self.size)
        word, back = ("BUY", "SELL") if side > 0 else ("SELL", "BUY")
        self.open_back = back
        bid, ask = self.touch()
        rec: dict = {"signal_ns": signal_ns, "lead": lead, "gap_bps": round(gap, 2), "side": word,
                     "size": self.size, "bid_at_signal": bid, "ask_at_signal": ask, **extra}
        expected = asyncio.ensure_future(self._expected(side, signal_ns, size))
        t_send = self.clock()
        try:
            oid = await self.api.market_open(self.symbol, word, self.size)
        except Exception as exc:  # noqa: BLE001 - every refusal is recorded and counted
            expected.cancel()
            return await self._failed(rec, "入りの注文", exc)
        t_ack = self.clock()
        fills = await self._wait_fills(oid, size)
        t_filled = self.clock()
        got = sum(f.size for f in fills)
        rec.update(order_id=oid, send_ns=t_send, ack_ns=t_ack, filled_seen_ns=t_filled,
                   send_to_ack_ms=(t_ack - t_send) / 1e6, ack_to_fill_seen_ms=(t_filled - t_ack) / 1e6,
                   signal_to_send_ms=(t_send - signal_ns) / 1e6)
        if got < size - 1e-9:
            # Part or none of it shows as filled: a position may exist that
            # this process does not know the size of. Stop and say so.
            expected.cancel()
            self.open_positions = [(f.position_id, f"{f.size:g}") for f in fills if f.position_id]
            rec["error"] = f"入りの約定が {got:g}/{size:g} しか確認できない"
            self._record(rec)
            await self.stop(f"{rec['error']}。GMO の画面でポジションを確認してください")
            return rec
        entry = sum(f.price * f.size for f in fills) / got
        self.open_positions = [(f.position_id, f"{f.size:g}") for f in fills if f.position_id]
        rec.update(entry_price=entry, entry_fee=sum(f.fee for f in fills),
                   entry_fill_time=fills[-1].timestamp, positions=self.open_positions)

        await self.sleep(max(0.0, (t_ack + self.hold_s * NS - self.clock()) / NS))
        exit_fills, rec2 = await self.close(back)
        rec.update(rec2)
        if not exit_fills:
            expected.cancel()
            self._record(rec)
            await self.stop("決済できませんでした。GMO の画面でポジションを決済してください")
            return rec
        out = sum(f.size for f in exit_fills)
        exit_ = sum(f.price * f.size for f in exit_fills) / out
        fees = rec["entry_fee"] + sum(f.fee for f in exit_fills)
        pnl = side * (exit_ - entry) * size - fees
        exp = await expected
        rec.update(exit_price=exit_, fees=fees, pnl_jpy=pnl, pnl_bps=pnl / (entry * size) * 1e4,
                   loss_gain=sum(f.loss_gain for f in exit_fills), **exp)
        if exp["expected_entry"] and exp["expected_exit"]:
            e_in, e_out = exp["expected_entry"], exp["expected_exit"]
            rec["expected_pnl_bps"] = side * (e_out - e_in) / e_in * 1e4
            rec["entry_slip_bps"] = side * (entry - e_in) / e_in * 1e4
            rec["exit_slip_bps"] = side * (e_out - exit_) / e_out * 1e4
        self._record(rec)
        t = self.tally
        t.n += 1
        t.wins += pnl > 0
        t.pnl += pnl
        t.notional += entry * size
        t.expected_pnl += rec.get("expected_pnl_bps", 0.0) / 1e4 * entry * size
        t.slip_bps += rec.get("entry_slip_bps", 0.0)
        t.send_to_ack_ms += rec["send_to_ack_ms"]
        self._consecutive_errors = 0
        if t.pnl <= -self.max_loss_jpy:
            await self.stop(f"損失が上限 {self.max_loss_jpy:,.0f}円 に達しました")
        elif t.n >= self.max_trades:
            await self.stop(f"取引回数が上限 {self.max_trades}回 に達しました")
        return rec

    async def close(self, back: str, tries: int = 3) -> tuple[list[Fill], dict]:
        """Close the open positions at market; (fills, what happened)."""
        rec: dict = {}
        if not self.open_positions:
            return [], {"exit_error": "決済するポジションがない"}
        for attempt in range(tries):
            t_send = self.clock()
            try:
                oid = await self.api.market_close(self.symbol, back, self.open_positions)
            except Exception as exc:  # noqa: BLE001
                rec["exit_error"] = f"{type(exc).__name__}: {exc}"
                await self.sleep(0.5 * (attempt + 1))
                continue
            t_ack = self.clock()
            size = sum(float(s) for _, s in self.open_positions)
            fills = await self._wait_fills(oid, size)
            rec.update(exit_order_id=oid, exit_send_ns=t_send, exit_ack_ns=t_ack,
                       exit_send_to_ack_ms=(t_ack - t_send) / 1e6)
            if sum(f.size for f in fills) >= size - 1e-9:
                self.open_positions = []
                rec.pop("exit_error", None)
                return fills, rec
            rec["exit_error"] = "決済の約定が確認できない"
            await self.sleep(0.5 * (attempt + 1))
        return [], rec

    async def _failed(self, rec: dict, what: str, exc: Exception) -> dict:
        rec["error"] = f"{what}: {type(exc).__name__}: {exc}"
        self._record(rec)
        self.tally.errors += 1
        self._consecutive_errors += 1
        codes = getattr(exc, "codes", []) or []
        # Not enough margin, or the key is refused: retrying will not help.
        # Anything else stops after three refusals in a row.
        fatal = any(c in ("ERR-201", "ERR-5010", "ERR-5011", "ERR-5012") for c in codes)
        if fatal or self._consecutive_errors >= 3:
            await self.stop(rec["error"])
        return rec


def summary(records: list[dict]) -> list[str]:
    """The numbers the live run is judged by, from its log."""
    import statistics

    done = [r for r in records if "pnl_jpy" in r]
    skipped = [r for r in records if r.get("skipped")]
    failed = [r for r in records if r.get("error") and "pnl_jpy" not in r]
    tried = len(done) + len(failed)

    def mean(key: str, rows=done) -> str:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return f"{statistics.fmean(vals):+.2f}" if vals else "-"

    def ms(key: str) -> str:
        vals = [r[key] for r in done if isinstance(r.get(key), (int, float))]
        return f"{statistics.median(vals):.0f}" if vals else "-"

    yen = sum(r["pnl_jpy"] for r in done)
    wins = sum(r["pnl_jpy"] > 0 for r in done)
    lines = [
        f"合図 {len(done) + len(failed) + len(skipped)}回（取引 {len(done)}・失敗 {len(failed)}・取引中で見送り {len(skipped)}）",
        f"約定率 {len(done) / tried:.0%}" if tried else "約定率 -",
        f"勝率 {wins / len(done):.0%}" if done else "勝率 -",
        f"実質の損益 {yen:+,.1f}円  1回あたり {mean('pnl_bps')}bps",
        f"検証の計算なら 1回あたり {mean('expected_pnl_bps')}bps",
        f"予想した値動き（フェア価格）平均 {mean('predicted_bps')}bps  その時の往復コスト 平均 {mean('cost_bps')}bps",
        f"入りの値段のずれ 平均 {mean('entry_slip_bps')}bps  出の値段のずれ 平均 {mean('exit_slip_bps')}bps（プラスは不利）",
        f"合図→注文 {ms('signal_to_send_ms')}ms  注文→受付 {ms('send_to_ack_ms')}ms  受付→約定確認 {ms('ack_to_fill_seen_ms')}ms（中央値）",
    ]
    lines += skew(done)
    return lines


def skew(done: list[dict]) -> list[str]:
    """Whether the result rests on a few hours or a few trades: by hour of
    the day (Japan time), and without the best and the worst trades."""
    import statistics
    from datetime import UTC, datetime, timedelta

    if not done:
        return []
    jst = timedelta(hours=9)
    by_hour: dict = {}
    for r in done:
        hour = (datetime.fromtimestamp(int(r["signal_ns"]) / 1e9, UTC) + jst).hour
        by_hour.setdefault(hour, []).append(r)
    out = ["時間帯ごと（日本時間）:"]
    plus_hours = 0
    for h, rows in sorted(by_hour.items()):
        mean = statistics.fmean(x["pnl_bps"] for x in rows)
        plus_hours += mean > 0
        wins = sum(x["pnl_jpy"] > 0 for x in rows)
        out.append(f"  {h:02d}時  {len(rows)}回  勝率 {wins / len(rows):.0%}  "
                   f"{sum(x['pnl_jpy'] for x in rows):+,.1f}円  1回あたり {mean:+.2f}bps")
    out.append(f"プラスだった時間帯 {plus_hours}/{len(by_hour)}")
    bps = sorted(r["pnl_bps"] for r in done)
    # Fixed before any result was seen: the best and the worst 5%, at least one.
    k = max(1, len(bps) // 20)
    if len(bps) > 2 * k:
        out.append(f"一番良い {k}回を除くと 1回あたり {statistics.fmean(bps[:-k]):+.2f}bps、"
                   f"一番悪い {k}回を除くと {statistics.fmean(bps[k:]):+.2f}bps、"
                   f"両方除くと {statistics.fmean(bps[k:-k]):+.2f}bps")
    return out
