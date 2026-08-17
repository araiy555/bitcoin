#!/usr/bin/env python3
"""Years-scale spread history from the Binance daily archive.

One hour of live capture cannot say whether a quiet market was the reason
nothing cleared the fee bar. The archive can: it goes back to 2019 on the perp
and is free. What it cannot do is answer the *same* question, because the
datasets differ from a live L2 recording:

* ``bookTicker`` - best bid/ask on every change, to the millisecond. Available
  from listing until Binance stopped publishing it in March 2024. Enough for
  spread and top-of-book relative pricing; not enough for depth or queue.
* ``aggTrades`` - the full trade tape to the millisecond, listing to yesterday.
* ``bookDepth`` - cumulative size at +-1..5% of mid, once a minute. Too coarse
  for microstructure.

So the years-scale question this tool answers is the structural one: how wide
is the top of book relative to the fee a round trip pays, across every regime
in the sample including crashes. If the spread never clears twice the maker
fee even in the worst turmoil of five years, no amount of signal work on this
symbol can pay for itself.

Research only. No authentication, no order entry, no profit claims.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import ssl
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise SystemExit(
        f"jane_history needs Python 3.11+ (pyproject requires-python), got "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        f"Run it with the project venv, e.g. .venv/bin/python3 tools/jane_history.py ..."
    )

try:
    from tools.jane_lab import DEFAULT_MAKER_BPS
    from tools.jane_screen import SPREAD_BINS, SpreadAcc
except ImportError:  # pragma: no cover - exercised only outside the repo root
    from jane_lab import DEFAULT_MAKER_BPS  # type: ignore[no-redef]
    from jane_screen import SPREAD_BINS, SpreadAcc  # type: ignore[no-redef]

VERSION = "2026-08-17-history-v1"
BASE = "https://data.binance.vision/data"
PRODUCTS = {"spot": "spot", "perp": "futures/um"}
DEFAULT_CACHE = Path("data/archive")
# Same fee assumption the rest of the pipeline uses, keyed by product.
MAKER_BPS = {"spot": DEFAULT_MAKER_BPS["binance_spot"],
             "perp": DEFAULT_MAKER_BPS["binance_perp"]}


def _ssl_context() -> ssl.SSLContext:
    """Verify against the bundled roots, exactly as jsboard.net does.

    A Homebrew Python on macOS points OpenSSL at a CA store that is usually
    empty, so every chain fails verification while curl succeeds. certifi ships
    Mozilla's roots with the package and behaves the same on every machine.
    Verification is never disabled: a certificate error means the far end could
    not be identified, and silencing the check does not fix that.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


SSL_CONTEXT = _ssl_context()


def tls_hint(exc: BaseException) -> str | None:
    text = f"{type(exc).__name__}: {exc}"
    if "CERTIFICATE_VERIFY_FAILED" not in text and "SSLCertVerification" not in text:
        return None
    return (
        "TLS証明書の検証に失敗しました。Pythonが空のCAストアを見ています。\n"
        "  .venv/bin/pip install certifi   で解決します"
        "（このリポジトリの依存に含まれているので、通常は -e インストールで入ります）。"
    )


def archive_url(product: str, dataset: str, symbol: str, day: date) -> str:
    """Identical to jsboard.research.archive.archive_url; selftest asserts it."""
    name = f"{symbol.upper()}-{dataset}-{day:%Y-%m-%d}.zip"
    return f"{BASE}/{PRODUCTS[product]}/daily/{dataset}/{symbol.upper()}/{name}"


def cache_path(product: str, dataset: str, symbol: str, day: date, *,
               cache: Path) -> Path:
    """Same layout as the existing archive cache, so downloads are shared."""
    return cache / product / dataset / symbol.upper() / f"{day:%Y-%m-%d}.zip"


def sample_days(start: date, end: date, every_n: int) -> list[date]:
    if end < start:
        raise ValueError("end is before start")
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    out, d = [], start
    while d <= end:
        out.append(d)
        d += timedelta(days=every_n)
    return out


def head_ok(url: str, *, timeout: float) -> tuple[int, str | None]:
    """(status, error). Status 0 means the request never reached the server."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
            return int(resp.status), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except OSError as exc:
        return 0, tls_hint(exc) or f"{type(exc).__name__}: {exc}"


def fetch_day(product: str, dataset: str, symbol: str, day: date, *,
              cache: Path, timeout: float) -> Path | None:
    """Cached download. Returns None for a day the venue never published."""
    target = cache_path(product, dataset, symbol, day, cache=cache)
    if target.exists() and target.stat().st_size > 0:
        return target
    url = archive_url(product, dataset, symbol, day)
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=SSL_CONTEXT) as resp:
            blob = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except OSError as exc:
        hint = tls_hint(exc)
        if hint:
            raise SystemExit(f"[jane-history] {hint}") from exc
        raise
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and rename, so an interrupted download cannot
    # leave a truncated file that the cache check would then trust.
    tmp = target.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(target)
    return target


def _bid_ask_columns(first_row: list[str]) -> tuple[int, int] | None:
    """Locate bid/ask columns, tolerating the header that newer files carry.

    Returns None when the row *is* the header, so the caller skips it.
    Positional defaults match both products: update_id, bid, bid_qty, ask,
    ask_qty, [transaction_time, event_time].
    """
    lowered = [c.strip().lower() for c in first_row]
    if any(c.replace("_", "").startswith("bestbid") for c in lowered):
        bid = next(i for i, c in enumerate(lowered)
                   if c.replace("_", "").startswith("bestbidprice"))
        ask = next(i for i, c in enumerate(lowered)
                   if c.replace("_", "").startswith("bestaskprice"))
        return bid, ask
    return None


def spread_day(path: Path, *, row_stride: int, maker_fee_bps: float) -> dict[str, Any]:
    """Sampled top-of-book spread statistics for one archived day."""
    acc = SpreadAcc()
    bid_i, ask_i = 1, 3
    skipped = 0
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"))
            for i, row in enumerate(reader):
                if i == 0:
                    found = _bid_ask_columns(row)
                    if found is not None:
                        bid_i, ask_i = found
                        continue
                if i % row_stride:
                    continue
                if len(row) <= max(bid_i, ask_i):
                    skipped += 1
                    continue
                try:
                    bid = float(row[bid_i])
                    ask = float(row[ask_i])
                except ValueError:
                    skipped += 1
                    continue
                if not (bid > 0 and ask > bid):
                    skipped += 1
                    continue
                mid = (bid + ask) * 0.5
                acc.add((ask - bid) / mid * 10_000.0,
                        fee_floor_bps=2.0 * maker_fee_bps)
    return {
        "sampled": acc.n,
        "skipped": skipped,
        "mean_spread_bps": acc.total / acc.n if acc.n else None,
        "min_spread_bps": acc.min_v if acc.n else None,
        "max_spread_bps": acc.max_v if acc.n else None,
        "samples_over_fee_floor": acc.over_fee,
        "rate_over_fee_floor": acc.over_fee / acc.n if acc.n else None,
        "bins": {"edges_bps": list(SPREAD_BINS), "counts": list(acc.bins)},
    }


def probe(product: str, symbol: str, days: list[date], *, timeout: float) -> dict[str, Any]:
    datasets = ("bookTicker", "aggTrades", "bookDepth")
    rows = []
    errors: list[str] = []
    for d in days:
        row: dict[str, Any] = {"day": d.isoformat()}
        for ds in datasets:
            status, err = head_ok(archive_url(product, ds, symbol, d), timeout=timeout)
            row[ds] = status
            if err and err not in errors:
                errors.append(err)
        rows.append(row)
    return {"product": product, "symbol": symbol, "datasets": list(datasets),
            "rows": rows, "errors": errors}


def run(product: str, symbol: str, days: list[date], *, cache: Path, row_stride: int,
        timeout: float, maker_fee_bps: float) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    total = SpreadAcc()
    for d in days:
        path = fetch_day(product, "bookTicker", symbol, d, cache=cache, timeout=timeout)
        if path is None:
            missing.append(d.isoformat())
            print(f"[jane-history] {d}: bookTicker なし (404)")
            continue
        stats = spread_day(path, row_stride=row_stride, maker_fee_bps=maker_fee_bps)
        stats["day"] = d.isoformat()
        out.append(stats)
        if stats["sampled"]:
            total.n += stats["sampled"]
            total.total += stats["mean_spread_bps"] * stats["sampled"]
            total.min_v = min(total.min_v, stats["min_spread_bps"])
            total.max_v = max(total.max_v, stats["max_spread_bps"])
            total.over_fee += stats["samples_over_fee_floor"]
            for i, c in enumerate(stats["bins"]["counts"]):
                total.bins[i] += c
        print(
            f"[jane-history] {d}: n={stats['sampled']:,} "
            f"平均={stats['mean_spread_bps']:.4f}bps "
            f"最大={stats['max_spread_bps']:.4f}bps "
            f"手数料超過={stats['rate_over_fee_floor']*100:.4f}%"
        )
    return {
        "version": VERSION,
        "product": product,
        "symbol": symbol,
        "maker_fee_bps": maker_fee_bps,
        "round_trip_fee_floor_bps": 2.0 * maker_fee_bps,
        "row_stride": row_stride,
        "days_requested": len(days),
        "days_loaded": len(out),
        "days_missing": missing,
        "aggregate": {
            "sampled": total.n,
            "mean_spread_bps": total.total / total.n if total.n else None,
            "min_spread_bps": total.min_v if total.n else None,
            "max_spread_bps": total.max_v if total.n else None,
            "samples_over_fee_floor": total.over_fee,
            "rate_over_fee_floor": total.over_fee / total.n if total.n else None,
            "bins": {"edges_bps": list(SPREAD_BINS), "counts": list(total.bins)},
        },
        "days": out,
        "notes": [
            "bookTicker is best bid/ask only: this measures the spread, not depth, imbalance or queue position.",
            "Binance stopped publishing bookTicker in March 2024, so later days return 404. That is a fact about the archive, not a failure.",
            "Rows are sampled with row_stride; the spread distribution is unaffected in shape but exact counts are 1/stride of the true ones.",
            "A spread below twice the maker fee means no strategy that earns the spread on this symbol can pay for itself, whatever the signal.",
        ],
    }


def print_summary(rep: dict[str, Any]) -> None:
    print(f"\n========== 数年分スプレッド履歴 ({rep['symbol']} {rep['product']}) ==========")
    a = rep["aggregate"]
    print(f"取得日数: {rep['days_loaded']} / {rep['days_requested']}"
          f"（欠測 {len(rep['days_missing'])}日）")
    if not a["sampled"]:
        print("  有効サンプルなし")
        print("=" * 60)
        return
    print(f"サンプル: {a['sampled']:,}行（{rep['row_stride']}行に1行）")
    print(f"往復Maker手数料の下限: {rep['round_trip_fee_floor_bps']:.1f} bps")
    print(f"スプレッド 平均={a['mean_spread_bps']:.4f} 最小={a['min_spread_bps']:.4f} "
          f"最大={a['max_spread_bps']:.4f} bps")
    print(f"手数料下限を超えたサンプル: {a['samples_over_fee_floor']:,} "
          f"({a['rate_over_fee_floor']*100:.4f}%)")
    edges = a["bins"]["edges_bps"]
    labels = [f"<{e:g}" for e in edges] + [f">={edges[-1]:g}"]
    print("スプレッド分布:")
    for lab, c in zip(labels, a["bins"]["counts"]):
        share = c / a["sampled"] * 100 if a["sampled"] else 0.0
        print(f"  {lab:>7s} bps: {c:12,} ({share:6.3f}%)")
    worst = max(rep["days"], key=lambda d: d["max_spread_bps"] or -1, default=None)
    if worst:
        print(f"最もスプレッドが開いた日: {worst['day']} "
              f"(最大 {worst['max_spread_bps']:.4f} bps, "
              f"超過率 {worst['rate_over_fee_floor']*100:.4f}%)")
    print("=" * 60)


def selftest() -> None:
    assert archive_url("perp", "bookTicker", "btcusdt", date(2023, 6, 15)) == (
        "https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/"
        "BTCUSDT-bookTicker-2023-06-15.zip")
    try:  # stay in lockstep with the module the rest of the repo downloads with
        from jsboard.research import archive as ja
        assert archive_url("spot", "aggTrades", "BTCUSDT", date(2024, 1, 2)) == \
            ja.archive_url("spot", "aggTrades", "BTCUSDT", date(2024, 1, 2))
    except ImportError:
        pass

    # The context must carry roots; an empty store is what broke this on macOS.
    assert SSL_CONTEXT.verify_mode == ssl.CERT_REQUIRED
    assert SSL_CONTEXT.cert_store_stats()["x509_ca"] > 0, "CA store is empty"
    try:  # same roots the rest of the repo uses
        from jsboard.net import ssl_context as jsb_ctx
        assert jsb_ctx().cert_store_stats()["x509_ca"] == \
            SSL_CONTEXT.cert_store_stats()["x509_ca"]
    except ImportError:
        pass
    err = ssl.SSLCertVerificationError("certificate verify failed")
    assert tls_hint(err) is not None
    assert tls_hint(OSError("connection refused")) is None

    assert sample_days(date(2024, 1, 1), date(2024, 1, 10), 4) == [
        date(2024, 1, 1), date(2024, 1, 5), date(2024, 1, 9)]

    header = ["update_id", "best_bid_price", "best_bid_qty", "best_ask_price",
              "best_ask_qty", "transaction_time", "event_time"]
    assert _bid_ask_columns(header) == (1, 3)
    assert _bid_ask_columns(["1", "100.0", "1", "101.0", "1", "0", "0"]) is None

    import tempfile
    rows = [header]
    # 4 sampled rows at stride 1: spreads of 2, 2, 20 and 200 bps on ~100k.
    for bid, ask in ((100_000.0, 100_020.0), (100_000.0, 100_020.0),
                     (100_000.0, 100_200.0), (100_000.0, 102_000.0)):
        rows.append(["1", f"{bid}", "1", f"{ask}", "1", "0", "0"])
    rows.append(["1", "0", "1", "0", "1", "0", "0"])  # unusable, must be skipped
    with tempfile.TemporaryDirectory() as td:
        zp = Path(td) / "d.zip"
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("d.csv", buf.getvalue())
        st = spread_day(zp, row_stride=1, maker_fee_bps=2.0)
    assert st["sampled"] == 4 and st["skipped"] == 1, st
    assert abs(st["min_spread_bps"] - 2.0) < 0.01, st
    assert abs(st["max_spread_bps"] - 198.02) < 0.05, st
    # fee floor is 4 bps: only the 20 and 200 bps rows clear it.
    assert st["samples_over_fee_floor"] == 2, st
    print("[jane-history] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Years-scale spread-vs-fee history from the Binance daily archive")
    p.add_argument("--product", default="perp", choices=sorted(PRODUCTS))
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2024-02-28")
    p.add_argument("--every-n-days", type=int, default=30)
    p.add_argument("--row-stride", type=int, default=100,
                   help="parse every Nth row; the spread distribution keeps its shape")
    p.add_argument("--cache", default=str(DEFAULT_CACHE))
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--report", default="jane-history-report.json")
    p.add_argument("--probe", action="store_true",
                   help="only check which datasets exist for the sampled days")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    days = sample_days(start, end, args.every_n_days)
    print(f"[jane-history] version {VERSION}")
    print(f"[jane-history] {args.symbol} {args.product}: {len(days)}日を"
          f"{args.start}〜{args.end}から抽出")

    if args.probe:
        rep = probe(args.product, args.symbol, days, timeout=args.timeout)
        print(f"\n{'day':12s} " + " ".join(f"{d:>10s}" for d in rep["datasets"]))
        for row in rep["rows"]:
            print(f"{row['day']:12s} " +
                  " ".join(f"{row[d]:>10d}" for d in rep["datasets"]))
        print("\n200=あり 404=なし 0=サーバに到達できず（下のエラーを参照）")
        for err in rep["errors"]:
            print(f"\n[jane-history] {err}")
        Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\nreport: {args.report}")
        return 1 if rep["errors"] else 0

    rep = run(args.product, args.symbol, days, cache=Path(args.cache),
              row_stride=args.row_stride, timeout=args.timeout,
              maker_fee_bps=MAKER_BPS[args.product])
    rep["config"] = vars(args)
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
