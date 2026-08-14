"""Medium-horizon, market-neutral research on Binance USD-M futures.

This module deliberately does not place orders.  It downloads immutable kline
archives plus realised funding history, then tests one pre-declared idea:
cross-sectional mean reversion after removing the BTC market move.  Keeping
the downloader and the accounting here (rather than in the CLI) makes the
assumptions independently testable.
"""

from __future__ import annotations

import asyncio
import bisect
import csv
import io
import json
import math
import statistics
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import aiohttp

from ..net import make_session

ARCHIVE_BASE = "https://data.binance.vision/data/futures/um"
FUTURES_BASE = "https://fapi.binance.com"
DEFAULT_ROOT = Path("data/statarb")
STABLE_BASES = {
    "USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "EUR", "TRY",
}


@dataclass(frozen=True, slots=True)
class ArchivePart:
    scope: str  # monthly or daily
    stamp: str


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    lookback_h: int
    hold_h: int
    strategy: str = "symmetric"
    entry_z: float = 0.0

    @property
    def label(self) -> str:
        base = f"lookback={self.lookback_h}h hold={self.hold_h}h"
        if self.strategy == "loser_btc":
            return f"{base} entry≤-{self.entry_z:g}σ"
        return base


@dataclass(slots=True)
class TradeReturn:
    hour: int
    price_bps: float
    long_price_bps: float
    short_price_bps: float
    funding_bps: float
    cost_bps: float
    net_bps: float
    gross_scale: float
    long_symbols: tuple[str, ...]
    short_symbols: tuple[str, ...]
    symbol_bps: dict[str, float]


@dataclass(slots=True)
class Performance:
    config: StrategyConfig
    trades: int
    total_bps: float
    annual_bps: float
    avg_bps: float
    win_rate: float
    max_drawdown_bps: float
    price_bps: float
    long_price_bps: float
    short_price_bps: float
    funding_bps: float
    cost_bps: float
    final_equity: float
    bankrupt: bool
    largest_win_bps: float
    largest_loss_bps: float
    top10_profit_share: float
    symbol_bps: dict[str, float]
    monthly_bps: dict[str, float]
    returns: list[TradeReturn]


@dataclass(slots=True)
class WalkForwardFold:
    train_end_hour: int
    test_end_hour: int
    selected: StrategyConfig | None
    train_bps: float
    test_bps: float
    test_trades: int


def parse_hours(value: str) -> int:
    """Parse CLI durations such as 6h, 24h and 3d into whole hours."""
    text = value.strip().lower()
    try:
        if text.endswith("h"):
            hours = int(text[:-1])
        elif text.endswith("d"):
            hours = int(text[:-1]) * 24
        else:
            hours = int(text)
    except ValueError as exc:
        raise ValueError(f"時間は 6h / 24h / 3d の形式で指定してください: {value}") from exc
    if hours <= 0:
        raise ValueError("時間は1時間以上で指定してください")
    return hours


def archive_parts(start: date, end: date) -> list[ArchivePart]:
    """Use one monthly file for complete months and daily files at the edges."""
    if end < start:
        return []
    parts: list[ArchivePart] = []
    cursor = start
    while cursor <= end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        # A complete archive month may include days before `start`; the loader
        # trims them.  Downloading it once is much cheaper than 20 daily ZIPs.
        if month_end <= end:
            parts.append(ArchivePart("monthly", cursor.strftime("%Y-%m")))
            cursor = next_month
        else:
            parts.append(ArchivePart("daily", cursor.isoformat()))
            cursor += timedelta(days=1)
    return parts


def kline_url(symbol: str, interval: str, part: ArchivePart) -> str:
    name = f"{symbol.upper()}-{interval}-{part.stamp}.zip"
    return (
        f"{ARCHIVE_BASE}/{part.scope}/klines/{symbol.upper()}/{interval}/{name}"
    )


def _part_path(root: Path, symbol: str, interval: str, part: ArchivePart) -> Path:
    return root / "klines" / symbol.upper() / interval / part.scope / f"{part.stamp}.zip"


async def select_universe(
    top: int,
    *,
    session: aiohttp.ClientSession,
) -> list[dict]:
    """Current liquid USDT perpetuals, saved to the manifest for auditability."""
    async with session.get(f"{FUTURES_BASE}/fapi/v1/exchangeInfo") as info_resp:
        info_resp.raise_for_status()
        info = await info_resp.json()
    async with session.get(f"{FUTURES_BASE}/fapi/v1/ticker/24hr") as tick_resp:
        tick_resp.raise_for_status()
        tickers = await tick_resp.json()

    eligible: dict[str, dict] = {}
    for item in info.get("symbols", []):
        base = item.get("baseAsset", "")
        if (
            item.get("status") == "TRADING"
            and item.get("contractType") == "PERPETUAL"
            and item.get("quoteAsset") == "USDT"
            and base not in STABLE_BASES
        ):
            eligible[item["symbol"]] = {
                "symbol": item["symbol"],
                "base": base,
                "onboard_date": item.get("onboardDate"),
            }

    ranked = []
    for item in tickers:
        symbol = item.get("symbol")
        if symbol not in eligible:
            continue
        try:
            volume = float(item.get("quoteVolume", 0.0))
        except (TypeError, ValueError):
            continue
        row = dict(eligible[symbol])
        row["quote_volume_24h"] = volume
        ranked.append(row)
    ranked.sort(key=lambda row: row["quote_volume_24h"], reverse=True)
    chosen = ranked[:top]

    # BTC is the market factor.  It must be present even if an unusually tiny
    # --top value was requested.
    if "BTCUSDT" in eligible and all(row["symbol"] != "BTCUSDT" for row in chosen):
        btc = dict(eligible["BTCUSDT"])
        btc["quote_volume_24h"] = next(
            (float(x.get("quoteVolume", 0.0)) for x in tickers if x.get("symbol") == "BTCUSDT"),
            0.0,
        )
        if len(chosen) >= top:
            chosen[-1] = btc
        else:
            chosen.append(btc)
    return chosen


async def _fetch_archive(
    symbol: str,
    interval: str,
    part: ArchivePart,
    *,
    root: Path,
    session: aiohttp.ClientSession,
) -> bool:
    target = _part_path(root, symbol, interval, part)
    if target.exists() and target.stat().st_size > 0:
        return True
    async with session.get(
        kline_url(symbol, interval, part), timeout=aiohttp.ClientTimeout(total=300)
    ) as resp:
        if resp.status == 404:
            return False
        resp.raise_for_status()
        blob = await resp.read()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(target)
    return True


async def _fetch_funding(
    symbol: str,
    start: date,
    end: date,
    *,
    root: Path,
    session: aiohttp.ClientSession,
) -> int:
    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    end_ms = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
        * 1000
    ) - 1
    rows: list[dict] = []
    cursor = start_ms
    while cursor <= end_ms:
        async with session.get(
            f"{FUTURES_BASE}/fapi/v1/fundingRate",
            params={
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            page = await resp.json()
        if not page:
            break
        rows.extend(page)
        nxt = int(page[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(page) < 1000:
            break

    target = root / "funding" / f"{symbol}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    tmp.replace(target)
    return len(rows)


async def download_dataset(
    *,
    days: int,
    top: int,
    end: date,
    root: Path = DEFAULT_ROOT,
    interval: str = "5m",
    symbols: Iterable[str] | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Download a reproducible research snapshot and return its manifest."""
    if days <= 0 or top <= 0:
        raise ValueError("--days と --top は1以上で指定してください")
    start = end - timedelta(days=days - 1)
    parts = archive_parts(start, end)

    async with make_session() as session:
        if symbols:
            universe = [
                {"symbol": s.strip().upper(), "base": "", "quote_volume_24h": None}
                for s in symbols
                if s.strip()
            ]
            if all(row["symbol"] != "BTCUSDT" for row in universe):
                universe.append({"symbol": "BTCUSDT", "base": "BTC", "quote_volume_24h": None})
        else:
            universe = await select_universe(top, session=session)

        semaphore = asyncio.Semaphore(6)

        async def one(row: dict) -> dict:
            symbol = row["symbol"]
            async with semaphore:
                found = 0
                for part in parts:
                    found += int(
                        await _fetch_archive(
                            symbol, interval, part, root=root, session=session
                        )
                    )
                funding = await _fetch_funding(
                    symbol, start, end, root=root, session=session
                )
            return {"symbol": symbol, "archives": found, "funding": funding}

        tasks = [asyncio.create_task(one(row)) for row in universe]
        results = []
        for done, task in enumerate(asyncio.as_completed(tasks), 1):
            result = await task
            results.append(result)
            if progress:
                progress(result["symbol"], done, len(tasks))

    manifest = {
        "version": 1,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "selection": "explicit" if symbols else "current_24h_quote_volume",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": days,
        "interval": interval,
        "top": top,
        "universe": universe,
        "files": sorted(results, key=lambda row: row["symbol"]),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _timestamp_ms(value: str) -> int:
    stamp = int(value)
    if stamp > 1_000_000_000_000_000:  # microseconds
        return stamp // 1000
    return stamp


def load_hourly_market(
    paths: Iterable[Path], start: date, end: date
) -> tuple[dict[int, float], dict[int, float]]:
    """Hourly signal closes and the next bar open used for causal execution.

    A signal for hour H uses the final kline close inside H.  Its executable
    price is the open of the first archive kline in H+1, stored under key H.
    Thus the backtest never observes a close and fills at that same close.
    """
    start_h = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp() // 3600)
    end_h = int(
        datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
        // 3600
    )
    latest: dict[int, tuple[int, float]] = {}
    next_open: dict[int, tuple[int, float]] = {}
    for path in sorted(paths):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                with zf.open(name) as raw:
                    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
                    for row in reader:
                        if len(row) < 5:
                            continue
                        try:
                            stamp = _timestamp_ms(row[0])
                            open_price = float(row[1])
                            close = float(row[4])
                        except ValueError:
                            continue
                        hour = stamp // 3_600_000
                        if close <= 0 or open_price <= 0:
                            continue
                        if start_h <= hour < end_h:
                            previous = latest.get(hour)
                            if previous is None or stamp > previous[0]:
                                latest[hour] = (stamp, close)
                        # The first kline of hour H is the first executable
                        # price after the signal made at the end of H-1.
                        signal_hour = hour - 1
                        if start_h <= signal_hour < end_h:
                            previous_open = next_open.get(signal_hour)
                            if previous_open is None or stamp < previous_open[0]:
                                next_open[signal_hour] = (stamp, open_price)
    return (
        {hour: value[1] for hour, value in latest.items()},
        {hour: value[1] for hour, value in next_open.items()},
    )


def load_hourly_closes(paths: Iterable[Path], start: date, end: date) -> dict[int, float]:
    """Compatibility helper used by earlier research and tests."""
    closes, _ = load_hourly_market(paths, start, end)
    return closes


def load_funding(path: Path) -> tuple[list[int], list[float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    pairs = sorted(
        (int(row["fundingTime"]) // 3_600_000, float(row["fundingRate"]))
        for row in rows
    )
    hours = [hour for hour, _ in pairs]
    prefix = [0.0]
    for _, rate in pairs:
        prefix.append(prefix[-1] + rate)
    return hours, prefix


def _funding_between(series: tuple[list[int], list[float]], start_h: int, end_h: int) -> float:
    hours, prefix = series
    left = bisect.bisect_right(hours, start_h)
    right = bisect.bisect_right(hours, end_h)
    return prefix[right] - prefix[left]


def load_dataset(
    root: Path = DEFAULT_ROOT,
) -> tuple[
    dict,
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[str, tuple[list[int], list[float]]],
]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} がありません。先に `jsboard statarb download` を実行してください"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = date.fromisoformat(manifest["start"])
    end = date.fromisoformat(manifest["end"])
    interval = manifest["interval"]
    prices: dict[str, dict[int, float]] = {}
    execution: dict[str, dict[int, float]] = {}
    funding: dict[str, tuple[list[int], list[float]]] = {}
    for row in manifest["universe"]:
        symbol = row["symbol"]
        paths = (root / "klines" / symbol / interval).rglob("*.zip")
        closes, next_opens = load_hourly_market(paths, start, end)
        if closes:
            prices[symbol] = closes
        if next_opens:
            execution[symbol] = next_opens
        funding_path = root / "funding" / f"{symbol}.json"
        if funding_path.exists():
            funding[symbol] = load_funding(funding_path)
    return manifest, prices, execution, funding


def _beta(prices: dict[int, float], btc: dict[int, float], hour: int, window: int) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for h in range(hour - window + 1, hour + 1):
        if h - 1 not in prices or h not in prices or h - 1 not in btc or h not in btc:
            continue
        xs.append(math.log(btc[h] / btc[h - 1]))
        ys.append(math.log(prices[h] / prices[h - 1]))
    if len(xs) < max(24, window // 3):
        return 1.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    variance = sum((x - x_mean) ** 2 for x in xs)
    if variance <= 1e-18:
        return 1.0
    value = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / variance
    return min(3.0, max(0.0, value))


def _portfolio_annual_vol(
    prices: dict[str, dict[int, float]],
    longs: tuple[str, ...],
    shorts: tuple[str, ...],
    long_weight: float,
    short_weight: float,
    hour: int,
    window: int,
) -> float:
    """Trailing beta-hedged hourly volatility, annualised without look-ahead."""
    returns: list[float] = []
    for h in range(hour - window + 1, hour + 1):
        if any(h - 1 not in prices[s] or h not in prices[s] for s in (*longs, *shorts)):
            continue
        long_return = statistics.fmean(
            math.log(prices[s][h] / prices[s][h - 1]) for s in longs
        )
        short_return = statistics.fmean(
            math.log(prices[s][h] / prices[s][h - 1]) for s in shorts
        )
        returns.append(long_weight * long_return - short_weight * short_return)
    if len(returns) < max(24, window // 3):
        return 0.0
    return statistics.pstdev(returns) * math.sqrt(24 * 365)


def simulate_config(
    prices: dict[str, dict[int, float]],
    funding: dict[str, tuple[list[int], list[float]]],
    config: StrategyConfig,
    *,
    execution_prices: dict[str, dict[int, float]] | None = None,
    fee_bps: float,
    slippage_bps: float,
    include_funding: bool,
    target_vol_pct: float = 0.0,
    beta_window_h: int = 168,
    side_fraction: float = 0.20,
) -> list[TradeReturn]:
    """Non-overlapping beta-neutral residual-reversion portfolios.

    ``symmetric`` buys the bottom residual quintile and shorts the top one.
    ``loser_btc`` buys only cross-sectionally extreme losers and shorts BTC in
    the amount required to remove their estimated market beta.  The latter is
    a direct test of the long-side-only effect found by the first experiment.
    """
    btc = prices.get("BTCUSDT")
    if not btc:
        raise ValueError("BTCUSDTがありません。市場要因を除去できません")
    execution = execution_prices or prices
    btc_execution = execution.get("BTCUSDT")
    if not btc_execution:
        raise ValueError("BTCUSDTの次足約定価格がありません")
    symbols = sorted(symbol for symbol in prices if symbol != "BTCUSDT")
    if len(symbols) < 4:
        raise ValueError("BTC以外に最低4銘柄必要です")
    # Newer listings must not truncate the entire study.  They simply join the
    # ranked universe once their own lookback and forward prices exist.
    first = min(btc)
    last = max(btc)
    start = first + max(beta_window_h, config.lookback_h)
    full_gross_cost = 2.0 * (fee_bps + slippage_bps)
    trades: list[TradeReturn] = []

    for hour in range(start, last - config.hold_h + 1, config.hold_h):
        btc_start = btc.get(hour - config.lookback_h)
        btc_now = btc.get(hour)
        if (
            btc_start is None
            or btc_now is None
            or hour not in btc_execution
            or hour + config.hold_h not in btc_execution
        ):
            continue
        btc_move = math.log(btc_now / btc_start)
        ranked: list[tuple[float, str, float]] = []
        for symbol in symbols:
            series = prices[symbol]
            p0 = series.get(hour - config.lookback_h)
            p1 = series.get(hour)
            executable = execution.get(symbol, {})
            if (
                p0 is None
                or p1 is None
                or hour not in executable
                or hour + config.hold_h not in executable
            ):
                continue
            beta = _beta(series, btc, hour, beta_window_h)
            residual = math.log(p1 / p0) - beta * btc_move
            ranked.append((residual, symbol, beta))
        if len(ranked) < 4:
            continue
        ranked.sort()
        if config.strategy == "loser_btc":
            residuals = [residual for residual, _, _ in ranked]
            dispersion = statistics.pstdev(residuals)
            if dispersion <= 1e-12:
                continue
            centre = statistics.fmean(residuals)
            cutoff = centre - config.entry_z * dispersion
            selected = [row for row in ranked if row[0] <= cutoff]
            if not selected:
                continue
            longs = tuple(symbol for _, symbol, _ in selected)
            shorts = ("BTCUSDT",)
            long_beta = statistics.fmean(beta for _, _, beta in selected)
            # Long gross + short gross = 1, while
            # long_weight*long_beta == short_weight*BTC_beta(1).
            long_weight = 1.0 / (1.0 + long_beta)
            short_weight = long_beta / (1.0 + long_beta)
        elif config.strategy == "symmetric":
            count = max(1, int(len(ranked) * side_fraction))
            selected = ranked[:count]
            short_selected = ranked[-count:]
            longs = tuple(symbol for _, symbol, _ in selected)
            shorts = tuple(symbol for _, symbol, _ in short_selected)
            long_beta = statistics.fmean(beta for _, _, beta in selected)
            short_beta = statistics.fmean(beta for _, _, beta in short_selected)
            beta_sum = long_beta + short_beta
            if beta_sum <= 1e-12:
                long_weight = short_weight = 0.5
            else:
                # Total gross stays at 1x while BTC beta is neutralised:
                # long_weight*long_beta == short_weight*short_beta.
                long_weight = short_beta / beta_sum
                short_weight = long_beta / beta_sum
        else:
            raise ValueError(f"unknown statarb strategy: {config.strategy}")
        annual_vol = _portfolio_annual_vol(
            prices,
            longs,
            shorts,
            long_weight,
            short_weight,
            hour,
            beta_window_h,
        )
        gross_scale = 1.0
        if target_vol_pct > 0 and annual_vol > 0:
            gross_scale = min(1.0, target_vol_pct / 100.0 / annual_vol)
        long_return = statistics.fmean(
            execution[s][hour + config.hold_h] / execution[s][hour] - 1.0 for s in longs
        )
        short_return = statistics.fmean(
            execution[s][hour + config.hold_h] / execution[s][hour] - 1.0 for s in shorts
        )
        long_price_bps = gross_scale * long_weight * long_return * 10_000.0
        short_price_bps = -gross_scale * short_weight * short_return * 10_000.0
        price_bps = long_price_bps + short_price_bps

        funding_bps = 0.0
        funding_start = hour + 1
        funding_end = hour + config.hold_h + 1
        if include_funding:
            long_funding = statistics.fmean(
                _funding_between(funding.get(s, ([], [0.0])), funding_start, funding_end)
                for s in longs
            )
            short_funding = statistics.fmean(
                _funding_between(funding.get(s, ([], [0.0])), funding_start, funding_end)
                for s in shorts
            )
            # Longs pay positive funding; shorts receive it.
            funding_bps = gross_scale * (
                -long_weight * long_funding + short_weight * short_funding
            ) * 10_000.0
        cost = full_gross_cost * gross_scale
        symbol_bps: dict[str, float] = {}
        for symbol in longs:
            weight = gross_scale * long_weight / len(longs)
            symbol_funding = (
                _funding_between(funding.get(symbol, ([], [0.0])), funding_start, funding_end)
                if include_funding
                else 0.0
            )
            symbol_bps[symbol] = (
                weight
                * (execution[symbol][hour + config.hold_h] / execution[symbol][hour] - 1.0)
                * 10_000.0
                - weight * symbol_funding * 10_000.0
                - full_gross_cost * weight
            )
        for symbol in shorts:
            weight = gross_scale * short_weight / len(shorts)
            symbol_funding = (
                _funding_between(funding.get(symbol, ([], [0.0])), funding_start, funding_end)
                if include_funding
                else 0.0
            )
            symbol_bps[symbol] = symbol_bps.get(symbol, 0.0) + (
                -weight
                * (execution[symbol][hour + config.hold_h] / execution[symbol][hour] - 1.0)
                * 10_000.0
                + weight * symbol_funding * 10_000.0
                - full_gross_cost * weight
            )
        trades.append(
            TradeReturn(
                hour=hour,
                price_bps=price_bps,
                long_price_bps=long_price_bps,
                short_price_bps=short_price_bps,
                funding_bps=funding_bps,
                cost_bps=cost,
                net_bps=price_bps + funding_bps - cost,
                gross_scale=gross_scale,
                long_symbols=longs,
                short_symbols=shorts,
                symbol_bps=symbol_bps,
            )
        )
    return trades


def summarise_performance(
    config: StrategyConfig,
    returns: list[TradeReturn],
    *,
    observed_days: float,
) -> Performance:
    equity = peak = 1.0
    drawdown = 0.0
    bankrupt = False
    processed: list[TradeReturn] = []
    price_bps = long_price_bps = short_price_bps = funding_bps = cost_bps = 0.0
    symbol_bps: dict[str, float] = {}
    positive_contributions: list[float] = []
    monthly: dict[str, float] = {}
    for row in returns:
        before = equity
        contribution = before * row.net_bps
        price_bps += before * row.price_bps
        long_price_bps += before * row.long_price_bps
        short_price_bps += before * row.short_price_bps
        funding_bps += before * row.funding_bps
        cost_bps += before * row.cost_bps
        for symbol, value in row.symbol_bps.items():
            symbol_bps[symbol] = symbol_bps.get(symbol, 0.0) + before * value
        equity = before * max(0.0, 1.0 + row.net_bps / 10_000.0)
        peak = max(peak, equity)
        drawdown = max(drawdown, (peak - equity) / peak if peak > 0 else 1.0)
        month = datetime.fromtimestamp(row.hour * 3600, tz=UTC).strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0) + contribution
        if contribution > 0:
            positive_contributions.append(contribution)
        processed.append(row)
        if equity <= 0:
            bankrupt = True
            break
    n = len(processed)
    total = (equity - 1.0) * 10_000.0
    if equity <= 0:
        annual = -10_000.0
    else:
        try:
            annual = (equity ** (365.0 / observed_days) - 1.0) * 10_000.0
        except OverflowError:
            annual = math.inf
    positive_total = sum(positive_contributions)
    top10 = sum(sorted(positive_contributions, reverse=True)[:10])
    return Performance(
        config=config,
        trades=n,
        total_bps=total,
        annual_bps=annual,
        avg_bps=statistics.fmean(row.net_bps for row in processed) if n else 0.0,
        win_rate=sum(row.net_bps > 0 for row in processed) / n if n else 0.0,
        max_drawdown_bps=drawdown * 10_000.0,
        price_bps=price_bps,
        long_price_bps=long_price_bps,
        short_price_bps=short_price_bps,
        funding_bps=funding_bps,
        cost_bps=cost_bps,
        final_equity=equity,
        bankrupt=bankrupt,
        largest_win_bps=max((row.net_bps for row in processed), default=0.0),
        largest_loss_bps=min((row.net_bps for row in processed), default=0.0),
        top10_profit_share=top10 / positive_total if positive_total > 0 else 0.0,
        symbol_bps=symbol_bps,
        monthly_bps=monthly,
        returns=processed,
    )


def run_backtests(
    prices: dict[str, dict[int, float]],
    funding: dict[str, tuple[list[int], list[float]]],
    configs: Iterable[StrategyConfig],
    *,
    execution_prices: dict[str, dict[int, float]] | None = None,
    fee_bps: float,
    slippage_bps: float,
    include_funding: bool,
    target_vol_pct: float = 0.0,
) -> list[Performance]:
    if not prices:
        return []
    first = min(min(series) for series in prices.values())
    last = max(max(series) for series in prices.values())
    days = max(1.0, (last - first) / 24.0)
    results = []
    for config in configs:
        returns = simulate_config(
            prices,
            funding,
            config,
            execution_prices=execution_prices,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            include_funding=include_funding,
            target_vol_pct=target_vol_pct,
        )
        results.append(summarise_performance(config, returns, observed_days=days))
    return results


def walk_forward(
    results: list[Performance],
    folds: int = 3,
    *,
    min_train_bps: float = 0.0,
    min_train_trades: int = 30,
) -> tuple[list[WalkForwardFold], Performance]:
    """Expanding-window selection with cash as a real alternative.

    A parameter set is never forced into the next period merely because it is
    the least bad.  If every sufficiently sampled candidate earned no more
    than ``min_train_bps`` on data available at the time, that fold holds cash
    and contributes exactly zero to the out-of-sample result.
    """
    all_returns = [row for result in results for row in result.returns]
    if not all_returns:
        raise ValueError("walk-forwardに使える取引がありません")
    first = min(row.hour for row in all_returns)
    last = max(row.hour for row in all_returns) + 1
    warmup_end = first + int((last - first) * 0.40)
    test_span = last - warmup_end
    selected_returns: list[TradeReturn] = []
    fold_rows: list[WalkForwardFold] = []
    walk_bankrupt = False

    for index in range(folds):
        test_start = warmup_end + test_span * index // folds
        test_end = warmup_end + test_span * (index + 1) // folds
        if walk_bankrupt:
            fold_rows.append(
                WalkForwardFold(test_start, test_end, None, 0.0, 0.0, 0)
            )
            continue
        scored = []
        for result in results:
            train = [row for row in result.returns if row.hour < test_start]
            if len(train) >= min_train_trades:
                train_days = max(1.0, (test_start - first) / 24.0)
                train_perf = summarise_performance(
                    result.config, train, observed_days=train_days
                )
                scored.append((train_perf.total_bps, result, train))
        if not scored:
            fold_rows.append(
                WalkForwardFold(
                    train_end_hour=test_start,
                    test_end_hour=test_end,
                    selected=None,
                    train_bps=0.0,
                    test_bps=0.0,
                    test_trades=0,
                )
            )
            continue
        best_train_bps, chosen, train = max(scored, key=lambda item: item[0])
        if best_train_bps <= min_train_bps:
            fold_rows.append(
                WalkForwardFold(
                    train_end_hour=test_start,
                    test_end_hour=test_end,
                    selected=None,
                    train_bps=best_train_bps,
                    test_bps=0.0,
                    test_trades=0,
                )
            )
            continue
        test = [row for row in chosen.returns if test_start <= row.hour < test_end]
        test_perf = summarise_performance(
            chosen.config,
            test,
            observed_days=max(1.0, (test_end - test_start) / 24.0),
        )
        selected_returns.extend(test_perf.returns)
        walk_bankrupt = test_perf.bankrupt
        fold_rows.append(
            WalkForwardFold(
                train_end_hour=test_start,
                test_end_hour=test_end,
                selected=chosen.config,
                train_bps=best_train_bps,
                test_bps=test_perf.total_bps,
                test_trades=test_perf.trades,
            )
        )
    observed_days = max(1.0, (last - warmup_end) / 24.0)
    summary = summarise_performance(
        StrategyConfig(0, 0, strategy="walk_forward"),
        selected_returns,
        observed_days=observed_days,
    )
    return fold_rows, summary


def performance_dict(performance: Performance) -> dict:
    """Stable JSON-friendly representation for future report exporters."""
    value = asdict(performance)
    value["config"] = asdict(performance.config)
    value.pop("returns", None)
    return value
