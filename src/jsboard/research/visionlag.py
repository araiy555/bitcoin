"""Which coins follow BTC late enough to trade, from Binance's free archives.

The lead's prints (aggTrades) give the signal: its price moved at least a
threshold within `window_ms`. Each follower is then checked against that
list of signals, using its top of book (bookTicker: best bid and ask with
their sizes, millisecond stamps):

- before: how far the follower had already moved in the signal's
  direction over the same window (if it moves with the lead, there is no
  lag to take);
- entry `latency_ms` after the signal at the follower's best ask (a buy)
  or best bid (a sell), and whether the size at that price covered the
  order;
- exit `h` later at the other side of the book, taker fee on both legs.

Days with no bookTicker published fall back to a touch read off the
follower's prints (the last price sold into is the bid, the last bought
the ask). That touch is stale between prints and has no size, so those
rows are marked and should not be trusted for the after-cost number.

Files are read straight out of the zip, one row at a time, and deleted
after use: a day of BTC is too big for a small machine's memory and disk
to hold whole.
"""

from __future__ import annotations

import heapq
import io
import math
import os
import shutil
import zipfile
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

BASE = "https://data.binance.vision/data/futures/um/daily"

HORIZONS_MS = (100, 200, 500, 1000, 2000, 5000, 10000)

FOLLOWERS = (
    "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "BNBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "SUIUSDT", "TRXUSDT", "NEARUSDT",
    "APTUSDT", "ARBUSDT", "OPUSDT", "FILUSDT", "ATOMUSDT", "UNIUSDT",
)

BOOK_COLUMNS = ("update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
                "best_ask_qty", "transaction_time", "event_time")
TRADE_COLUMNS = ("agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id",
                 "transact_time", "is_buyer_maker")


def url(datatype: str, symbol: str, day: str) -> str:
    return f"{BASE}/{datatype}/{symbol}/{symbol}-{datatype}-{day}.zip"


def _ms(raw: str) -> int:
    value = int(raw)
    return value // 1000 if value > 100_000_000_000_000 else value  # microseconds


def read_rows(path: str, columns: tuple[str, ...], wanted: tuple[str, ...]) -> Iterator[list[str]]:
    """The `wanted` fields of every row, read from the zip one line at a time.

    A header row is used when the file has one (its first cell is not a
    number), so a reordered archive cannot swap price and size."""
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError(f"{path} holds no csv")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            first = text.readline()
            if not first:
                return
            cells = first.strip().split(",")
            try:
                float(cells[0])
                header = None
            except ValueError:
                header = [c.strip() for c in cells]
            names_ = header or list(columns)
            idx = [names_.index(w) for w in wanted]
            if header is None:
                yield [cells[i] for i in idx]
            for line in text:
                c = line.split(",")
                if len(c) > max(idx):
                    yield [c[i] for i in idx]


def lead_prints(path: str) -> Iterator[tuple[int, float]]:
    """(ms, price) for every aggregated print."""
    for price, ts in read_rows(path, TRADE_COLUMNS, ("price", "transact_time")):
        try:
            yield _ms(ts), float(price)
        except ValueError:
            continue


def book_tops(path: str) -> Iterator[tuple[int, float, float, float, float]]:
    """(ms, bid, bid size, ask, ask size) from a bookTicker archive."""
    wanted = ("transaction_time", "best_bid_price", "best_bid_qty", "best_ask_price", "best_ask_qty")
    for ts, b, bq, a, aq in read_rows(path, BOOK_COLUMNS, wanted):
        try:
            bid, ask = float(b), float(a)
            if 0 < bid < ask:
                yield _ms(ts), bid, float(bq), ask, float(aq)
        except ValueError:
            continue


def tape_tops(path: str) -> Iterator[tuple[int, float, float, float, float]]:
    """A touch read off the prints, for days without bookTicker; size unknown (nan)."""
    bid = ask = None
    nan = float("nan")
    for price, ts, maker in read_rows(path, TRADE_COLUMNS, ("price", "transact_time", "is_buyer_maker")):
        try:
            p = float(price)
        except ValueError:
            continue
        if maker.strip().lower() in ("true", "1"):
            bid = p  # the buyer rested, a seller hit the bid
        else:
            ask = p
        if bid is not None and ask is not None and ask > bid:
            yield _ms(ts), bid, nan, ask, nan


@dataclass(frozen=True, slots=True)
class Signal:
    ms: int
    sign: int
    threshold: float
    move_bps: float


def signals(prints, thresholds, *, window_ms: int, cooldown_ms: int) -> list[Signal]:
    """Each time the lead's price moved at least `threshold` bps within the
    last `window_ms`, at most once per `cooldown_ms` for each threshold."""
    hist: deque = deque()
    next_ok = {t: -1 for t in thresholds}
    out: list[Signal] = []
    last = None
    for ms, price in prints:
        if price == last:
            continue
        last = price
        hist.append((ms, price))
        while len(hist) >= 2 and hist[1][0] <= ms - window_ms:
            hist.popleft()
        if hist[0][0] > ms - window_ms:
            continue
        move = math.log(price / hist[0][1]) * 1e4
        for t in thresholds:
            if abs(move) >= t and ms >= next_ok[t]:
                next_ok[t] = ms + cooldown_ms
                out.append(Signal(ms, 1 if move > 0 else -1, t, move))
    return out


@dataclass
class Cell:
    n: int = 0
    same: int = 0
    opposite: int = 0
    mid_bps: float = 0.0
    net_bps: float = 0.0
    wins: int = 0
    before_bps: float = 0.0
    depth_ok: int = 0
    depth_known: int = 0
    spread_bps: float = 0.0

    def add(self, other: Cell) -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass
class Scan:
    """(follower, day, threshold, horizon) -> Cell, and each day's data source."""
    cells: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)


def evaluate(tops, sigs: list[Signal], scan: Scan, *, follower: str, day: str,
             horizons_ms=HORIZONS_MS, latency_ms: int = 50, window_ms: int = 100,
             fee_bps: float = 5.0, size_usd: float = 1000.0) -> None:
    """Score every signal on the follower's book; see the module docstring."""
    tasks: list = []  # (due ms, seq, kind, payload)
    seq = 0
    i = 0
    bid = ask = bq = aq = None
    hist: deque = deque()  # (ms, mid)

    def mid_ago(now: int):
        while len(hist) >= 2 and hist[1][0] <= now - window_ms:
            hist.popleft()
        return hist[0][1] if hist and hist[0][0] <= now - window_ms else None

    def cell(t: float, h: int) -> Cell:
        key = (follower, day, t, h)
        c = scan.cells.get(key)
        if c is None:
            c = scan.cells[key] = Cell()
        return c

    def run(now: int) -> None:
        """Everything due before `now`, against the book as it stood."""
        nonlocal i, seq
        while True:
            next_sig = sigs[i].ms if i < len(sigs) else None
            next_task = tasks[0][0] if tasks else None
            due = min(x for x in (next_sig, next_task, now) if x is not None)
            if due >= now:
                return
            if next_sig is not None and next_sig == due:
                s = sigs[i]
                i += 1
                if bid is None:
                    continue
                mid = (bid + ask) / 2
                before = mid_ago(s.ms)
                pre = s.sign * math.log(mid / before) * 1e4 if before else 0.0
                seq += 1
                heapq.heappush(tasks, (s.ms + latency_ms, seq, "in", (s, mid, pre)))
                continue
            _, _, kind, payload = heapq.heappop(tasks)
            if kind == "in":
                s, mid0, pre = payload
                price, size = (ask, aq) if s.sign > 0 else (bid, bq)
                spread = (ask - bid) / ((ask + bid) / 2) * 1e4
                for h in horizons_ms:
                    seq += 1
                    heapq.heappush(tasks, (due + h, seq, "out", (s, h, mid0, pre, price, size, spread)))
            else:
                s, h, mid0, pre, entry, size, spread = payload
                exit_ = bid if s.sign > 0 else ask
                c = cell(s.threshold, h)
                c.n += 1
                move = s.sign * math.log((bid + ask) / 2 / mid0) * 1e4
                c.same += move > 1e-9
                c.opposite += move < -1e-9
                c.mid_bps += move
                net = s.sign * (exit_ - entry) / entry * 1e4 - 2 * fee_bps
                c.net_bps += net
                c.wins += net > 0
                c.before_bps += pre
                c.spread_bps += spread
                if size == size:  # not nan
                    c.depth_known += 1
                    c.depth_ok += size * entry >= size_usd

    for ms, b, bsz, a, asz in tops:
        run(ms)
        bid, bq, ask, aq = b, bsz, a, asz
        m = (b + a) / 2
        if not hist or hist[-1][1] != m:
            hist.append((ms, m))
            # Trim as it grows, not only when a signal asks: between rare
            # signals a whole day of mids would otherwise pile up in memory.
            while len(hist) >= 2 and hist[1][0] <= ms - window_ms:
                hist.popleft()
    # Trades still open when the file ends are dropped, not closed at a stale price.


def total(scan: Scan, follower: str, threshold: float, h: int, days) -> tuple[Cell, int]:
    """The cell over all days, and how many days were positive after costs."""
    out, plus = Cell(), 0
    for day in days:
        c = scan.cells.get((follower, day, threshold, h))
        if c is None or not c.n:
            continue
        out.add(c)
        plus += c.net_bps > 0
    return out, plus


HEADER = "\t".join([
    "追随する銘柄", "しきいbps", "時間", "合図", "直前に動いてた bps", "同じ向き",
    "中値 bps", "手数料込み bps", "勝率", "板足りた", "平均スプレッド bps", "プラスの日", "データ",
])

NOTE = (
    "合図 = 先行銘柄の約定値段が「窓」の間に しきいbps 以上動いた瞬間。\n"
    "直前に動いてた = 追随側が同じ窓の間に、もう同じ向きに動いていた分（大きいと遅れがない）。\n"
    "中値 bps = 合図からその時間後の追随側の中値の動き（合図の向きを正）。手数料・スプレッドなし。\n"
    "手数料込み = 遅れ時間後に最良気配で成行で入り、その時間後に反対の最良気配で出た損益（片道手数料×2込み）。\n"
    "板足りた = 入る瞬間の最良気配の量が注文額以上あった割合。データ「約定推定」の日は板の量がなく値段も粗い。"
)


def fmt(follower: str, threshold: float, h: int, c: Cell, plus: int, days: int, source: str) -> str:
    def pct(x: int, n: int) -> str:
        return f"{x / n:.0%}" if n else "-"

    n = c.n
    return "\t".join([
        follower, f"{threshold:g}", f"{h / 1000:g}秒", f"{n:,}",
        f"{c.before_bps / n:+.1f}", pct(c.same, n), f"{c.mid_bps / n:+.2f}",
        f"{c.net_bps / n:+.2f}", pct(c.wins, n), pct(c.depth_ok, c.depth_known),
        f"{c.spread_bps / n:.1f}", f"{plus}/{days}", source,
    ])


def report(scan: Scan, followers, thresholds, days, *, min_signals: int = 30, top: int = 20):
    """(every row, the best `top` rows by the after-cost result)."""
    rows = []
    for f in followers:
        srcs = {scan.sources.get((f, d)) for d in days} - {None}
        if not srcs:
            continue
        source = "板" if srcs == {"book"} else ("約定推定" if srcs == {"tape"} else "混在")
        used = [d for d in days if (f, d) in scan.sources]
        for t in thresholds:
            for h in HORIZONS_MS:
                c, plus = total(scan, f, t, h, used)
                if c.n:
                    rows.append((c.net_bps / c.n, c.n, fmt(f, t, h, c, plus, len(used), source)))
    every = [r[2] for r in rows]
    best = [r[2] for r in sorted((r for r in rows if r[1] >= min_signals), key=lambda r: -r[0])[:top]]
    return every, best


def structure(scan: Scan, followers, threshold: float, days) -> list[str]:
    """For one threshold: how far each follower's mid had gone, in the
    signal's direction, by each horizon after the lead moved (no costs)."""
    head = "\t".join(["銘柄", "合図", "合図の時点で既に"] + [f"{h / 1000:g}秒後" for h in HORIZONS_MS]
                     + ["1秒後に同じ向き"])
    out = [head]
    for f in followers:
        used = [d for d in days if (f, d) in scan.sources]
        cells = [total(scan, f, threshold, h, used)[0] for h in HORIZONS_MS]
        if not cells[0].n:
            continue
        one = cells[HORIZONS_MS.index(1000)]
        out.append("\t".join(
            [f, f"{cells[0].n:,}", f"{cells[0].before_bps / cells[0].n:+.1f}"]
            + [f"{c.mid_bps / c.n:+.1f}" if c.n else "-" for c in cells]
            + [f"{one.same / one.n:.0%}" if one.n else "-"]))
    return out


async def download(link: str, path: str, *, keep_free: int = 500 * 2**20) -> int | None:
    """Save `link` to `path`; None if the archive does not exist.

    Refuses, before writing anything, a file that would leave less than
    `keep_free` bytes on the disk."""
    import aiohttp

    from ..net import make_session

    async with make_session() as session:
        timeout = aiohttp.ClientTimeout(total=3600, sock_read=120)
        async with session.get(link, timeout=timeout) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            free = shutil.disk_usage(os.path.dirname(path) or ".").free
            size = resp.content_length or 0
            if size and free - size < keep_free:
                raise OSError(f"空き {free / 2**30:.1f}GB に {size / 2**30:.1f}GB は入りません: {link}")
            done = 0
            with open(path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
            return done
