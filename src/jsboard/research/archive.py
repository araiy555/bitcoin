"""Daily archive files from Binance, downloaded once and kept.

The live recorder only accumulates forward. The archive goes back to 2017 on
spot and 2019 on the perp, and runs to yesterday, so anything that can be
answered from the tape can be answered today instead of in a fortnight.

What the archive does *not* contain shaped the plan more than what it does:
`bookDepth` is cumulative size at ±1…5% of mid, once a minute, and `bookTicker`
stopped being published in March 2024. So order-book features — level
imbalance, cancel/add asymmetry, microprice — cannot come from here at all.
Trades can, in full, to the millisecond. Liquidations are published for
neither product.

Files are cached on disk because they are immutable: a day that has closed
never changes, so re-downloading it is pure waste.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import aiohttp

from ..net import make_session

BASE = "https://data.binance.vision/data"
DEFAULT_CACHE = Path("data/archive")

PRODUCTS = {
    "spot": "spot",
    "perp": "futures/um",
}


def archive_url(product: str, dataset: str, symbol: str, day: date) -> str:
    root = PRODUCTS[product]
    name = f"{symbol.upper()}-{dataset}-{day:%Y-%m-%d}.zip"
    return f"{BASE}/{root}/daily/{dataset}/{symbol.upper()}/{name}"


def days_ending(last: date, count: int) -> list[date]:
    """`count` consecutive days ending at `last`, oldest first."""
    return [last - timedelta(days=i) for i in range(count - 1, -1, -1)]


async def fetch_day(
    product: str,
    dataset: str,
    symbol: str,
    day: date,
    *,
    cache: Path = DEFAULT_CACHE,
    session: aiohttp.ClientSession | None = None,
) -> Path | None:
    """Local path to one day's archive, downloading it if absent.

    Returns None when the venue has no file for that day — a symbol that had
    not listed yet, or a gap — which is a fact about the data, not an error to
    raise through the caller.
    """
    target = cache / product / dataset / symbol.upper() / f"{day:%Y-%m-%d}.zip"
    if target.exists() and target.stat().st_size > 0:
        return target

    owned = session is None
    session = session or make_session()
    try:
        async with session.get(
            archive_url(product, dataset, symbol, day),
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status == 404:
                return None
            resp.raise_for_status()
            blob = await resp.read()
    finally:
        if owned:
            await session.close()

    target.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted download does not
    # leave a truncated file that the cache check would then trust.
    tmp = target.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(target)
    return target


# ------------------------------------------------------------------ parsing


@dataclass(slots=True)
class SecondBar:
    """One second of tape. `last` is the final trade price in that second."""

    sec: int
    last: float
    high: float
    low: float
    buy_qty: float
    sell_qty: float
    trades: int

    @property
    def signed_qty(self) -> float:
        return self.buy_qty - self.sell_qty


def _to_seconds(stamp: int) -> int:
    """Archive timestamps are milliseconds in older files and microseconds in
    newer ones. Deciding by magnitude beats hard-coding a cutover date that
    differs between datasets."""
    if stamp > 1_000_000_000_000_000:  # microseconds
        return stamp // 1_000_000
    if stamp > 1_000_000_000_000:  # milliseconds
        return stamp // 1_000
    return stamp


def load_seconds(path: Path) -> list[SecondBar]:
    """Collapse one day of aggregated trades into per-second bars.

    Columns are id, price, qty, first, last, time, isBuyerMaker. Older files
    have no header row and newer ones do, so the first row is tested rather
    than assumed. `isBuyerMaker` true means the *seller* crossed the spread.
    """
    bars: dict[int, SecondBar] = {}
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    price = float(row[1])
                    qty = float(row[2])
                    sec = _to_seconds(int(row[5]))
                except ValueError:
                    continue  # the header row, if this file has one
                buyer_is_maker = row[6].strip().lower() in ("true", "1")

                bar = bars.get(sec)
                if bar is None:
                    bars[sec] = SecondBar(sec, price, price, price, 0.0, 0.0, 0)
                    bar = bars[sec]
                bar.last = price
                bar.high = max(bar.high, price)
                bar.low = min(bar.low, price)
                bar.trades += 1
                if buyer_is_maker:
                    bar.sell_qty += qty
                else:
                    bar.buy_qty += qty
    return [bars[k] for k in sorted(bars)]
