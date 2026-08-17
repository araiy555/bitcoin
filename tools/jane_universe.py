#!/usr/bin/env python3
"""Find the markets where the market-making equation actually closes.

BTCUSDT failed one test and one test only:

    spread > 2 x maker fee

at 0.0316 bps against 4.0 bps, a 127x shortfall. That is a fact about BTC, not
about market making. The maker fee is the same 2 bps on every perp Binance
lists, while the spread is set by how contested the symbol is - and across the
listed universe that varies by orders of magnitude. So the same equation is
worth asking of every symbol, not just the most efficient one in the world.

One call to the all-symbol bookTicker endpoint prices the whole universe at
once, so a poll every few seconds for an hour is enough to rank several hundred
markets by how often their top of book clears the fee floor. No download, no
authentication.

A wide spread is not profit. It is compensation for the risk of quoting into a
market that moves before you can cancel, and the symbols with the widest
spreads are the ones where that is worst. This tool finds candidates for the
adverse-selection work that the repo's mm/toxicity machinery already exists to
do; it does not find winners.

Research only. No authentication, no order entry, no profit claims.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise SystemExit(
        f"jane_universe needs Python 3.11+ (pyproject requires-python), got "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        f"Run it with the project venv, e.g. .venv/bin/python3 tools/jane_universe.py ..."
    )

try:
    from tools.jane_history import SSL_CONTEXT, tls_hint
    from tools.jane_screen import SPREAD_BINS, SpreadAcc
except ImportError:  # pragma: no cover - exercised only outside the repo root
    from jane_history import SSL_CONTEXT, tls_hint  # type: ignore[no-redef]
    from jane_screen import SPREAD_BINS, SpreadAcc  # type: ignore[no-redef]

VERSION = "2026-08-17-universe-v1"
FUTURES_BASE = "https://fapi.binance.com"
SPOT_BASE = "https://api.binance.com"
# Standard-tier perp maker fee, the same on every symbol Binance lists.
DEFAULT_MAKER_BPS = 2.0


@dataclass(slots=True)
class SymbolAcc:
    spread: SpreadAcc = field(default_factory=SpreadAcc)
    mid_sum: float = 0.0
    bid_notional_sum: float = 0.0
    ask_notional_sum: float = 0.0


def get_json(url: str, *, timeout: float) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=SSL_CONTEXT) as resp:
            return json.loads(resp.read())
    except OSError as exc:
        hint = tls_hint(exc)
        if hint:
            raise SystemExit(f"[jane-universe] {hint}") from exc
        raise


def fetch_book_tickers(base: str, *, timeout: float) -> list[dict[str, Any]]:
    """Best bid/ask for every symbol on the venue, in one request."""
    rows = get_json(f"{base}/fapi/v1/ticker/bookTicker" if "fapi" in base
                    else f"{base}/api/v3/ticker/bookTicker", timeout=timeout)
    return rows if isinstance(rows, list) else [rows]


def fetch_quote_volume(base: str, *, timeout: float) -> dict[str, float]:
    """24h quote volume per symbol, so a wide spread on a dead market is visible."""
    rows = get_json(f"{base}/fapi/v1/ticker/24hr" if "fapi" in base
                    else f"{base}/api/v3/ticker/24hr", timeout=timeout)
    out: dict[str, float] = {}
    for r in rows if isinstance(rows, list) else [rows]:
        try:
            out[r["symbol"]] = float(r.get("quoteVolume") or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def observe(acc: dict[str, SymbolAcc], rows: list[dict[str, Any]], *,
            maker_fee_bps: float, quote_suffix: str | None) -> int:
    """Fold one snapshot into the accumulators. Returns rows actually used."""
    used = 0
    floor = 2.0 * maker_fee_bps
    for r in rows:
        sym = r.get("symbol")
        if not sym or (quote_suffix and not sym.endswith(quote_suffix)):
            continue
        try:
            bid = float(r["bidPrice"])
            ask = float(r["askPrice"])
            bid_q = float(r.get("bidQty") or 0.0)
            ask_q = float(r.get("askQty") or 0.0)
        except (KeyError, TypeError, ValueError):
            continue
        if not (bid > 0 and ask > bid):
            continue
        mid = (bid + ask) * 0.5
        a = acc.setdefault(sym, SymbolAcc())
        a.spread.add((ask - bid) / mid * 10_000.0, fee_floor_bps=floor)
        a.mid_sum += mid
        a.bid_notional_sum += bid * bid_q
        a.ask_notional_sum += ask * ask_q
        used += 1
    return used


def summarize(acc: dict[str, SymbolAcc], volumes: dict[str, float], *,
              maker_fee_bps: float, min_samples: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    floor = 2.0 * maker_fee_bps
    for sym, a in acc.items():
        s = a.spread
        if s.n < min_samples:
            continue
        mean = s.total / s.n
        out.append({
            "symbol": sym,
            "samples": s.n,
            "mean_spread_bps": mean,
            "min_spread_bps": s.min_v,
            "max_spread_bps": s.max_v,
            "fee_floor_bps": floor,
            "rate_over_fee_floor": s.over_fee / s.n,
            # What one perfect round trip would keep, before adverse selection.
            "edge_over_floor_bps": mean - floor,
            "mean_mid": a.mid_sum / s.n,
            "mean_top_notional_usd": (a.bid_notional_sum + a.ask_notional_sum) / (2 * s.n),
            "quote_volume_24h": volumes.get(sym),
            "bins": {"edges_bps": list(SPREAD_BINS), "counts": list(s.bins)},
        })
    out.sort(key=lambda r: (r["rate_over_fee_floor"], r["edge_over_floor_bps"]),
             reverse=True)
    return out


def run(base: str, *, duration_s: float, interval_s: float, timeout: float,
        maker_fee_bps: float, quote_suffix: str | None,
        min_samples: int) -> dict[str, Any]:
    print(f"[jane-universe] 24h出来高を取得...")
    volumes = fetch_quote_volume(base, timeout=timeout)
    print(f"[jane-universe] {len(volumes):,} 銘柄")

    acc: dict[str, SymbolAcc] = {}
    polls = 0
    started = time.monotonic()
    deadline = started + duration_s
    while True:
        t0 = time.monotonic()
        used = observe(acc, fetch_book_tickers(base, timeout=timeout),
                       maker_fee_bps=maker_fee_bps, quote_suffix=quote_suffix)
        polls += 1
        if polls == 1 or polls % 12 == 0:
            print(f"[jane-universe] {polls}回目 {used:,}銘柄 "
                  f"経過{time.monotonic() - started:.0f}s / {duration_s:.0f}s")
        if time.monotonic() >= deadline:
            break
        time.sleep(max(0.0, interval_s - (time.monotonic() - t0)))

    return {
        "version": VERSION,
        "venue": base,
        "polls": polls,
        "duration_s": time.monotonic() - started,
        "interval_s": interval_s,
        "maker_fee_bps": maker_fee_bps,
        "fee_floor_bps": 2.0 * maker_fee_bps,
        "symbols": summarize(acc, volumes, maker_fee_bps=maker_fee_bps,
                             min_samples=min_samples),
        "notes": [
            "The test is spread > 2 x maker fee: a round trip that rests on both sides earns the spread once and pays the maker fee twice.",
            "Snapshots are polled, so a spread that widens and closes between two polls is missed. This undercounts brief blowouts and is the conservative direction.",
            "A wide spread is compensation for adverse selection, not free money. These are candidates for the toxicity and markout work, not results.",
            "mean_top_notional_usd caps the size a quote can work at, and quote_volume_24h caps the size the strategy can ever reach. A wide spread on a dead market is not a business.",
            "Fees are the standard-tier maker assumption, identical across symbols; the spread is what varies.",
        ],
    }


def print_summary(rep: dict[str, Any], *, top_n: int, min_volume: float) -> None:
    print(f"\n[jane-universe] version {VERSION}")
    print("\n======== マーケットメイクが成立する市場の探索 ========")
    print(f"venue: {rep['venue']}  ポーリング: {rep['polls']}回 / "
          f"{rep['duration_s']:.0f}秒")
    print(f"判定式: スプレッド > 2 × Maker手数料 = {rep['fee_floor_bps']:.1f} bps")

    rows = [r for r in rep["symbols"]
            if (r["quote_volume_24h"] or 0.0) >= min_volume]
    live = [r for r in rows if r["rate_over_fee_floor"] > 0.5]
    print(f"\n24h出来高 {min_volume:,.0f} 以上の銘柄: {len(rows)}")
    print(f"うち半分以上の時間で手数料下限を超えていた銘柄: {len(live)}")

    print(f"\n  {'symbol':16s} {'平均spread':>10s} {'超過率':>8s} {'手数料超過分':>11s} "
          f"{'板厚(USD)':>11s} {'24h出来高':>14s}")
    for r in rows[:top_n]:
        print(
            f"  {r['symbol']:16s} {r['mean_spread_bps']:10.3f} "
            f"{r['rate_over_fee_floor']*100:7.2f}% {r['edge_over_floor_bps']:+11.3f} "
            f"{r['mean_top_notional_usd']:11,.0f} {r['quote_volume_24h'] or 0:14,.0f}"
        )
    btc = next((r for r in rep["symbols"] if r["symbol"] == "BTCUSDT"), None)
    if btc:
        print(f"\n  参考 BTCUSDT: 平均={btc['mean_spread_bps']:.4f}bps "
              f"超過率={btc['rate_over_fee_floor']*100:.4f}% "
              f"(手数料÷スプレッド={rep['fee_floor_bps'] / btc['mean_spread_bps']:.0f}倍)")
    print("\n※ 広いスプレッドは逆選択の対価です。候補であって、勝てる市場ではありません。")
    print("※ 板厚と出来高が小さい銘柄は、式が成立しても規模が出せません。")
    print("=" * 56)


def selftest() -> None:
    acc: dict[str, SymbolAcc] = {}
    # 2 bps maker -> 4 bps floor. WIDE clears it, TIGHT does not.
    snap = [
        {"symbol": "WIDEUSDT", "bidPrice": "100.0", "askPrice": "100.2",
         "bidQty": "10", "askQty": "10"},
        {"symbol": "TIGHTUSDT", "bidPrice": "100.0", "askPrice": "100.001",
         "bidQty": "10", "askQty": "10"},
        {"symbol": "CROSSEDUSDT", "bidPrice": "100.0", "askPrice": "99.0",
         "bidQty": "1", "askQty": "1"},
        {"symbol": "OTHERBUSD", "bidPrice": "100.0", "askPrice": "101.0",
         "bidQty": "1", "askQty": "1"},
    ]
    used = observe(acc, snap, maker_fee_bps=2.0, quote_suffix="USDT")
    # The crossed book is refused and the non-USDT symbol is filtered out.
    assert used == 2 and set(acc) == {"WIDEUSDT", "TIGHTUSDT"}, acc.keys()
    observe(acc, snap, maker_fee_bps=2.0, quote_suffix="USDT")

    rows = summarize(acc, {"WIDEUSDT": 5e6, "TIGHTUSDT": 9e9},
                     maker_fee_bps=2.0, min_samples=2)
    assert [r["symbol"] for r in rows] == ["WIDEUSDT", "TIGHTUSDT"], rows
    wide, tight = rows
    assert wide["rate_over_fee_floor"] == 1.0 and tight["rate_over_fee_floor"] == 0.0
    assert abs(wide["mean_spread_bps"] - 19.98) < 0.05, wide
    assert wide["edge_over_floor_bps"] > 0 and tight["edge_over_floor_bps"] < 0
    # Top-of-book notional is the average of the two sides, ~100 * 10.
    assert abs(wide["mean_top_notional_usd"] - 1001.0) < 2.0, wide
    # min_samples keeps a symbol seen once out of the ranking entirely.
    assert summarize(acc, {}, maker_fee_bps=2.0, min_samples=3) == []
    print("[jane-universe] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Rank every listed symbol by spread against the maker fee floor")
    p.add_argument("--venue", default="perp", choices=("perp", "spot"))
    p.add_argument("--duration", type=float, default=600.0, help="秒")
    p.add_argument("--interval", type=float, default=5.0, help="秒")
    p.add_argument("--maker-fee-bps", type=float, default=DEFAULT_MAKER_BPS)
    p.add_argument("--quote", default="USDT", help="この文字列で終わる銘柄のみ")
    p.add_argument("--min-samples", type=int, default=10)
    p.add_argument("--min-volume", type=float, default=10_000_000.0,
                   help="24h建値出来高の下限（表示のみ）")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--report", default="jane-universe-report.json")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    base = FUTURES_BASE if args.venue == "perp" else SPOT_BASE
    print(f"[jane-universe] version {VERSION}")
    print(f"[jane-universe] {args.venue} を {args.duration:.0f}秒 "
          f"({args.interval:.0f}秒間隔) 観測します")
    rep = run(base, duration_s=args.duration, interval_s=args.interval,
              timeout=args.timeout, maker_fee_bps=args.maker_fee_bps,
              quote_suffix=args.quote or None, min_samples=args.min_samples)
    rep["config"] = vars(args)
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep, top_n=args.top, min_volume=args.min_volume)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
