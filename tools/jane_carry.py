#!/usr/bin/env python3
"""Funding carry over years: the one edge that is larger than the fee.

Every microstructure family tested so far died on the same wall. The spread on
BTCUSDT perp averages 0.0316 bps against a 4 bps round-trip maker fee - the fee
is 127x the edge - and the spread clears that floor for about a second a day.
No amount of speed, signal or queue modelling closes a 127x gap.

Funding is the opposite shape. It is paid every 8 hours, it is measured in bps
rather than hundredths of one, and it does not depend on being fast. A
delta-neutral position - long spot, short perp, same size - has no price
exposure and collects funding whenever the rate is positive.

What that costs is the two round trips: taker in and taker out on both legs.
So the question this tool answers is not "is funding positive" (it usually is)
but "does it stay positive long enough, often enough, to repay the fees" - and
what the worst holding window in the sample looked like.

Deliberately NOT modelled, and each one can move the answer:
  * basis P&L at entry and exit; entering at a rich basis adds to this, and
    unwinding at a wide one takes from it,
  * margin on the short perp leg and the liquidation risk that comes with it,
  * that funding is charged on the perp notional, which drifts with the mark.

Research only. No authentication, no order entry, no profit claims.
"""
from __future__ import annotations

import argparse
import bisect
import json
import ssl
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise SystemExit(
        f"jane_carry needs Python 3.11+ (pyproject requires-python), got "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        f"Run it with the project venv, e.g. .venv/bin/python3 tools/jane_carry.py ..."
    )

try:
    from tools.jane_history import SSL_CONTEXT, tls_hint
    from tools.jane_lab import DEFAULT_TAKER_BPS
except ImportError:  # pragma: no cover - exercised only outside the repo root
    from jane_history import SSL_CONTEXT, tls_hint  # type: ignore[no-redef]
    from jane_lab import DEFAULT_TAKER_BPS  # type: ignore[no-redef]

VERSION = "2026-08-17-carry-v1"
FUTURES_BASE = "https://fapi.binance.com"
DEFAULT_CACHE = Path("data/archive/funding")
DAY_MS = 86_400_000
UTC = timezone.utc


def to_ms(day: date) -> int:
    return int(datetime.combine(day, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)


def fetch_funding(symbol: str, start: date, end: date, *, cache: Path,
                  timeout: float, refresh: bool) -> list[dict[str, Any]]:
    """Every funding print in the range, paginated and cached on disk.

    The endpoint is public and returns at most 1000 rows, oldest first, so the
    cursor walks forward from the last print seen.
    """
    target = cache / f"{symbol.upper()}-fundingRate-{start:%Y%m%d}-{end:%Y%m%d}.json"
    if target.exists() and target.stat().st_size > 0 and not refresh:
        return json.loads(target.read_text(encoding="utf-8"))

    start_ms, end_ms = to_ms(start), to_ms(end + timedelta(days=1)) - 1
    rows: list[dict[str, Any]] = []
    cursor = start_ms
    while cursor <= end_ms:
        q = urllib.parse.urlencode({"symbol": symbol.upper(), "startTime": cursor,
                                    "endTime": end_ms, "limit": 1000})
        url = f"{FUTURES_BASE}/fapi/v1/fundingRate?{q}"
        try:
            with urllib.request.urlopen(url, timeout=timeout, context=SSL_CONTEXT) as resp:
                page = json.loads(resp.read())
        except OSError as exc:
            hint = tls_hint(exc)
            if hint:
                raise SystemExit(f"[jane-carry] {hint}") from exc
            raise
        if not page:
            break
        rows.extend(page)
        nxt = int(page[-1]["fundingTime"]) + 1
        if nxt <= cursor:
            break
        cursor = nxt
        print(f"[jane-carry] {len(rows):,}件取得 "
              f"({datetime.fromtimestamp(cursor / 1000, UTC):%Y-%m-%d})")
        if len(page) < 1000:
            break

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    tmp.replace(target)
    return rows


def to_series(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    """(times_ms, rates_bps), sorted and de-duplicated by funding time."""
    seen: dict[int, float] = {}
    for row in rows:
        seen[int(row["fundingTime"])] = float(row["fundingRate"]) * 10_000.0
    times = sorted(seen)
    return times, [seen[t] for t in times]


def prefix_sums(rates: list[float]) -> list[float]:
    out = [0.0]
    for r in rates:
        out.append(out[-1] + r)
    return out


def window_stats(times: list[int], rates: list[float], *, hold_days: float,
                 round_trip_fee_bps: float) -> dict[str, Any]:
    """Net bps of a delta-neutral hold started at each funding print.

    Windows are cut on the clock rather than on a count of prints, so a change
    in funding interval - 8h is not guaranteed forever, and is not universal
    across symbols - does not silently change the holding period.
    """
    pre = prefix_sums(rates)
    span = int(hold_days * DAY_MS)
    nets: list[float] = []
    starts: list[int] = []
    for i, t in enumerate(times):
        end = t + span
        if times[-1] < end:
            break  # the window would run past the data
        j = bisect.bisect_right(times, end)
        # Entering at t_i means not having held through the print at t_i, so
        # the window starts at the next one and runs through the print landing
        # exactly at the exit.
        nets.append(pre[j] - pre[i + 1] - round_trip_fee_bps)
        starts.append(t)
    if not nets:
        return {"hold_days": hold_days, "windows": 0}

    # Overlapping windows share almost all of their prints, so a rate computed
    # over them counts one regime hundreds of times. Walking forward a whole
    # holding period at a time gives the independent observations, which for a
    # long hold over a few years is a single-digit number - and that is the
    # honest sample size for any claim about how often this works.
    indep: list[float] = []
    cursor_t = starts[0]
    k = 0
    while k < len(starts):
        if starts[k] < cursor_t:
            k += 1
            continue
        indep.append(nets[k])
        cursor_t = starts[k] + span
        k += 1

    worst_i = min(range(len(nets)), key=lambda k: nets[k])
    best_i = max(range(len(nets)), key=lambda k: nets[k])
    return {
        "hold_days": hold_days,
        "windows": len(nets),
        "independent_windows": len(indep),
        "independent_positive_rate": sum(1 for v in indep if v > 0) / len(indep),
        "independent_mean_net_bps": statistics.fmean(indep),
        "independent_median_net_bps": statistics.median(indep),
        "round_trip_fee_bps": round_trip_fee_bps,
        "mean_net_bps": statistics.fmean(nets),
        "median_net_bps": statistics.median(nets),
        "positive_rate": sum(1 for v in nets if v > 0) / len(nets),
        "worst_net_bps": nets[worst_i],
        "worst_start": datetime.fromtimestamp(starts[worst_i] / 1000, UTC).date().isoformat(),
        "best_net_bps": nets[best_i],
        "best_start": datetime.fromtimestamp(starts[best_i] / 1000, UTC).date().isoformat(),
        # Annualising a holding-period return is only meaningful if the window
        # can be repeated back to back, which assumes it stays available.
        "annualised_pct_if_repeated": statistics.fmean(nets) / 100.0 * (365.0 / hold_days),
    }


def yearly(times: list[int], rates: list[float]) -> list[dict[str, Any]]:
    buckets: dict[int, list[float]] = {}
    for t, r in zip(times, rates):
        buckets.setdefault(datetime.fromtimestamp(t / 1000, UTC).year, []).append(r)
    out = []
    for year in sorted(buckets):
        v = buckets[year]
        out.append({
            "year": year,
            "prints": len(v),
            "mean_bps_per_print": statistics.fmean(v),
            "positive_rate": sum(1 for x in v if x > 0) / len(v),
            "sum_bps": sum(v),
            "min_bps": min(v),
            "max_bps": max(v),
        })
    return out


def analyze(rows: list[dict[str, Any]], *, hold_days: tuple[float, ...],
            spot_taker_bps: float, perp_taker_bps: float,
            recent_days: float | None = None) -> dict[str, Any]:
    times, rates = to_series(rows)
    if len(times) < 2:
        return {"version": VERSION, "available": False, "reason": "funding prints not found"}
    # Two legs in, two legs out.
    round_trip = 2.0 * (spot_taker_bps + perp_taker_bps)
    gaps = [(b - a) / 3_600_000 for a, b in zip(times, times[1:])]
    interval_h = statistics.median(gaps)
    per_year = 365.0 * 24.0 / interval_h
    mean_bps = statistics.fmean(rates)
    return {
        "version": VERSION,
        "available": True,
        "prints": len(times),
        "first": datetime.fromtimestamp(times[0] / 1000, UTC).date().isoformat(),
        "last": datetime.fromtimestamp(times[-1] / 1000, UTC).date().isoformat(),
        "median_interval_hours": interval_h,
        "spot_taker_bps": spot_taker_bps,
        "perp_taker_bps": perp_taker_bps,
        "round_trip_fee_bps": round_trip,
        "mean_bps_per_print": mean_bps,
        "median_bps_per_print": statistics.median(rates),
        "positive_rate": sum(1 for r in rates if r > 0) / len(rates),
        "min_bps": min(rates),
        "max_bps": max(rates),
        # Gross of fees, and only if the rate holds - it does not.
        "gross_annualised_pct": mean_bps / 100.0 * per_year,
        "prints_to_repay_fees": round_trip / mean_bps if mean_bps > 0 else None,
        "days_to_repay_fees": (round_trip / mean_bps * interval_h / 24.0)
                              if mean_bps > 0 else None,
        "by_year": yearly(times, rates),
        # The full-sample average is carried by 2021. What the strategy is
        # worth now needs the tail of the data on its own.
        "recent": recent_slice(rows, times, hold_days=hold_days,
                               spot_taker_bps=spot_taker_bps,
                               perp_taker_bps=perp_taker_bps,
                               recent_days=recent_days),
        "windows": [window_stats(times, rates, hold_days=d, round_trip_fee_bps=round_trip)
                    for d in hold_days],
        "notes": [
            "Delta-neutral means long spot and short perp in equal size: no price exposure, funding collected while the rate is positive.",
            "Fees are two taker legs in and two out at the repo's standard-tier assumptions. A maker entry would lower this and is the obvious first optimisation.",
            "Basis P&L at entry and exit is not modelled, nor is margin or liquidation risk on the short perp leg.",
            "positive_rate over windows is the number that matters: a high average with a low positive rate means the mean is carried by a few regimes.",
            "Funding is not an arbitrage. It is payment for taking the other side of crowded leverage, and it turns negative exactly when that crowd flips.",
        ],
    }


def recent_slice(rows: list[dict[str, Any]], times: list[int], *,
                 hold_days: tuple[float, ...], spot_taker_bps: float,
                 perp_taker_bps: float, recent_days: float | None) -> dict[str, Any] | None:
    if not recent_days or not times:
        return None
    cut = times[-1] - int(recent_days * DAY_MS)
    kept = [r for r in rows if int(r["fundingTime"]) >= cut]
    if len(kept) < 2:
        return None
    sub = analyze(kept, hold_days=hold_days, spot_taker_bps=spot_taker_bps,
                  perp_taker_bps=perp_taker_bps, recent_days=None)
    sub["recent_days"] = recent_days
    return sub


def print_summary(rep: dict[str, Any]) -> None:
    print(f"[jane-carry] version {VERSION}")
    print("\n============ ファンディング・キャリー検証 ============")
    if not rep.get("available"):
        print("  利用不可:", rep.get("reason"))
        print("=" * 56)
        return
    print(f"期間: {rep['first']} 〜 {rep['last']}  ({rep['prints']:,}回, "
          f"{rep['median_interval_hours']:.0f}時間ごと)")
    print(f"1回あたり: 平均={rep['mean_bps_per_print']:+.4f}bps "
          f"中央={rep['median_bps_per_print']:+.4f}bps "
          f"最小={rep['min_bps']:+.3f} 最大={rep['max_bps']:+.3f}")
    print(f"プラスだった割合: {rep['positive_rate']*100:.2f}%")
    print(f"手数料控除前の年率: {rep['gross_annualised_pct']:+.2f}%")
    print(f"往復手数料: {rep['round_trip_fee_bps']:.1f}bps "
          f"(spot {rep['spot_taker_bps']:g} + perp {rep['perp_taker_bps']:g}, 各2回)")
    if rep["days_to_repay_fees"]:
        print(f"手数料回収に必要な保有: {rep['days_to_repay_fees']:.1f}日 "
              f"({rep['prints_to_repay_fees']:.1f}回分)")

    print("\n年別:")
    print(f"  {'年':6s} {'回数':>6s} {'平均bps':>9s} {'プラス率':>8s} {'年間合計bps':>11s} "
          f"{'最小':>8s} {'最大':>8s}")
    for y in rep["by_year"]:
        print(f"  {y['year']:<6d} {y['prints']:6d} {y['mean_bps_per_print']:+9.4f} "
              f"{y['positive_rate']*100:7.2f}% {y['sum_bps']:+11.1f} "
              f"{y['min_bps']:+8.3f} {y['max_bps']:+8.3f}")

    print("\n保有期間別（手数料控除後、開始時点を全通り試行）:")
    print(f"  {'保有':>6s} {'試行数':>7s} {'net平均':>9s} {'net中央':>9s} {'黒字率':>8s} "
          f"{'最悪':>9s} {'最悪の開始日':>12s} {'年率換算':>9s}")
    for w in rep["windows"]:
        if not w["windows"]:
            print(f"  {w['hold_days']:5g}日 {0:7d}  （データ期間が足りません）")
            continue
        print(
            f"  {w['hold_days']:5g}日 {w['windows']:7d} {w['mean_net_bps']:+9.2f} "
            f"{w['median_net_bps']:+9.2f} {w['positive_rate']*100:7.2f}% "
            f"{w['worst_net_bps']:+9.2f} {w['worst_start']:>12s} "
            f"{w['annualised_pct_if_repeated']:+8.2f}%"
        )

    print("\n重複を除いた独立窓のみ（これが本当の標本数）:")
    print(f"  {'保有':>6s} {'独立数':>7s} {'net平均':>9s} {'net中央':>9s} {'黒字率':>8s}")
    for w in rep["windows"]:
        if not w["windows"]:
            continue
        print(
            f"  {w['hold_days']:5g}日 {w['independent_windows']:7d} "
            f"{w['independent_mean_net_bps']:+9.2f} "
            f"{w['independent_median_net_bps']:+9.2f} "
            f"{w['independent_positive_rate']*100:7.2f}%"
        )

    rec = rep.get("recent")
    if rec:
        print(f"\n直近{rec['recent_days']:g}日のみ（{rec['first']} 〜 {rec['last']}）:")
        print(f"  1回あたり平均={rec['mean_bps_per_print']:+.4f}bps "
              f"プラス率={rec['positive_rate']*100:.2f}% "
              f"手数料控除前年率={rec['gross_annualised_pct']:+.2f}%")
        for w in rec["windows"]:
            if not w["windows"]:
                continue
            print(f"  {w['hold_days']:5g}日: net中央={w['median_net_bps']:+8.2f}bps "
                  f"黒字率={w['positive_rate']*100:6.2f}% "
                  f"(独立{w['independent_windows']}窓で{w['independent_positive_rate']*100:.0f}%)")
    print("\n※ ファンディングは裁定ではなく、偏ったレバレッジの反対側を引き受ける対価です。")
    print("※ ベーシス損益・証拠金・清算リスクは未モデル化。")
    print("=" * 56)


def selftest() -> None:
    t0 = to_ms(date(2024, 1, 1))
    step = 8 * 3_600_000
    # 90 days of a flat +1 bps funding print every 8 hours.
    rows = [{"fundingTime": t0 + i * step, "fundingRate": "0.0001"} for i in range(270)]
    times, rates = to_series(rows)
    assert len(times) == 270 and abs(rates[0] - 1.0) < 1e-12

    # Duplicates collapse rather than double-counting a print.
    dup = rows + [{"fundingTime": t0, "fundingRate": "0.0001"}]
    assert len(to_series(dup)[0]) == 270

    rep = analyze(rows, hold_days=(30.0,), spot_taker_bps=10.0, perp_taker_bps=4.0)
    assert rep["available"] and abs(rep["median_interval_hours"] - 8.0) < 1e-9
    assert abs(rep["round_trip_fee_bps"] - 28.0) < 1e-12
    # 1 bps three times a day repays 28 bps of fees in 28/3 days.
    assert abs(rep["days_to_repay_fees"] - 28.0 / 3.0) < 1e-9, rep["days_to_repay_fees"]
    # 270 prints spans 89.7 days, so 30-day holds starting at day 0 and day 30
    # both fit; one starting at day 60 would run past the last print.
    w30 = rep["windows"][0]
    assert w30["independent_windows"] == 2, w30["independent_windows"]
    assert w30["independent_positive_rate"] == 1.0
    assert abs(w30["independent_mean_net_bps"] - 62.0) < 1e-9

    # The recent slice must see only the tail, and must not recurse.
    r = analyze(rows, hold_days=(30.0,), spot_taker_bps=10.0, perp_taker_bps=4.0,
                recent_days=30.0)
    assert r["recent"] is not None and r["recent"]["recent"] is None
    assert r["recent"]["prints"] < r["prints"], (r["recent"]["prints"], r["prints"])
    assert r["recent"]["recent_days"] == 30.0

    w = rep["windows"][0]
    # 30 days x 3 prints x 1 bps = 90 bps gross, minus 28 bps of fees.
    assert abs(w["mean_net_bps"] - 62.0) < 1e-9, w
    assert w["positive_rate"] == 1.0
    assert w["windows"] == 270 - 90, w["windows"]

    # A regime that pays nothing must show up as a loss of exactly the fees.
    flat = [{"fundingTime": t0 + i * step, "fundingRate": "0"} for i in range(270)]
    wf = analyze(flat, hold_days=(30.0,), spot_taker_bps=10.0,
                 perp_taker_bps=4.0)["windows"][0]
    assert abs(wf["mean_net_bps"] + 28.0) < 1e-9 and wf["positive_rate"] == 0.0
    print("[jane-carry] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Years-scale delta-neutral funding carry from Binance funding history")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--hold-days", type=float, nargs="+", default=[7, 30, 90, 180])
    p.add_argument("--spot-taker-bps", type=float, default=DEFAULT_TAKER_BPS["binance_spot"])
    p.add_argument("--perp-taker-bps", type=float, default=DEFAULT_TAKER_BPS["binance_perp"])
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--refresh", action="store_true", help="ignore the cached download")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--recent-days", type=float, default=365.0,
                   help="直近この日数だけの再計算も出す。0で無効")
    p.add_argument("--report", default="jane-carry-report.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    print(f"[jane-carry] version {VERSION}")
    print(f"[jane-carry] {args.symbol} {start} 〜 {end}")
    rows = fetch_funding(args.symbol, start, end, cache=Path(args.cache),
                         timeout=args.timeout, refresh=args.refresh)
    print(f"[jane-carry] ファンディング {len(rows):,} 件")
    rep = analyze(rows, hold_days=tuple(sorted(set(args.hold_days))),
                  spot_taker_bps=args.spot_taker_bps, perp_taker_bps=args.perp_taker_bps,
                  recent_days=args.recent_days or None)
    rep["config"] = vars(args)
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
