"""Sanity checks on a two-venue recording before trusting a replay of it.

    python scripts/leadcheck.py /tmp/rec-XRP_JPY [--source gmo] [--lead binance]

Answers the questions a losing sweep raises: are the trade sides read the
right way round, how often does the book update, how wide is the spread
really, how far did the price travel, and does the lead venue actually move
first.
"""

import argparse
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

from jsboard.core.market import MarketView
from jsboard.core.types import Instrument
from jsboard.feed.base import DepthSnapshot, TradeTick
from jsboard.feed.replay import iter_tagged


def quantile(xs: list[float], p: float) -> float:
    return sorted(xs)[int(len(xs) * p)] if xs else float("nan")


def one_second_return(series: dict[int, float], t: int) -> float | None:
    a, b = series.get(t), series.get(t + 1)
    return math.log(b / a) * 1e4 if a and b else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--source", default="gmo")
    ap.add_argument("--lead", default="binance")
    opts = ap.parse_args()
    folder = Path(opts.folder)

    spec = json.loads((folder / "meta.json").read_text())["sources"]
    views = {
        name: MarketView(
            Instrument(
                s["symbol"], Decimal(s["tick_size"]), Decimal(s["lot_size"]), s["base"], s["quote"]
            )
        )
        for name, s in spec.items()
    }

    side_ok = side_bad = 0
    gaps: list[float] = []
    spreads: list[float] = []
    last_book = None
    first = last = None
    hi, lo = 0.0, math.inf
    grid: dict[str, dict[int, float]] = {name: {} for name in views}

    for src, event in iter_tagged(folder):
        view = views.get(src)
        if view is None:
            continue
        if src == opts.source and isinstance(event, TradeTick) and view.book.best_bid() is not None:
            buy = int(event.aggressor) > 0
            mid = view.mid
            if (buy and event.price >= mid) or (not buy and event.price <= mid):
                side_ok += 1
            else:
                side_bad += 1
        view.apply(event)
        ts = getattr(event, "ts_ns", 0)
        price = view.mid_price
        if price is None or not ts:
            continue
        grid[src][ts // 1_000_000_000] = float(price)
        if src == opts.source and isinstance(event, DepthSnapshot):
            bid, ask = view.book.best_bid(), view.book.best_ask()
            spreads.append((ask - bid) / ((ask + bid) / 2) * 1e4)
            if last_book is not None:
                gaps.append((ts - last_book) / 1e9)
            last_book = ts
            first = first or float(price)
            last = float(price)
            hi, lo = max(hi, float(price)), min(lo, float(price))

    total = max(1, side_ok + side_bad)
    print(f"約定の向き: 整合 {side_ok:,} / 逆 {side_bad:,}  ({side_ok / total:.0%} 整合)")
    print(f"板の更新間隔: 中央 {quantile(gaps, .5):.2f}秒  90%点 {quantile(gaps, .9):.2f}秒")
    print(
        f"スプレッド: 中央 {quantile(spreads, .5):.2f}bps  "
        f"(10%点 {quantile(spreads, .1):.2f} / 90%点 {quantile(spreads, .9):.2f})"
    )
    if first and last:
        print(
            f"値動き: 開始 {first:.3f} → 終了 {last:.3f}  ({(last / first - 1) * 1e4:+.0f}bps)  "
            f"高値 {hi:.3f} 安値 {lo:.3f}  (幅 {(hi / lo - 1) * 1e4:.0f}bps)"
        )

    ours, lead = grid[opts.source], grid[opts.lead]
    for lag in (0, 1, 2, 3, 5):
        xs, ys = [], []
        for t in sorted(lead):
            x, y = one_second_return(lead, t), one_second_return(ours, t + lag)
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) > 50:
            corr = statistics.correlation(xs, ys)
            print(
                f"先行市場の1秒変化 → 自分の市場の{lag}秒後の1秒変化  "
                f"相関 {corr:+.2f}  (n={len(xs):,})"
            )


if __name__ == "__main__":
    main()
