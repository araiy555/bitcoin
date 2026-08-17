#!/usr/bin/env python3
"""Step 1: re-price every hedge venue from raw depth at the maker-fill time.

The maker->taker candidates surfaced so far all hedge into ``binance_spot``,
whose ~10 bps taker assumption dominates the economics. Swapping the hedge to a
perp venue is *not* a matter of substituting 4 bps for 10 bps in the existing
number: binance_perp and okx_perp trade at their own prices, so the gross has
to be rebuilt from their own books at the same fill timestamp.

For every de-duplicated maker fill this tool walks the real ladder of *each*
candidate hedge venue and reports

    gross(venue, latency) - maker fee - venue taker fee = net

with the executed price taken from depth, so queue-independent slippage is
measured rather than assumed. The break-even hedge taker fee it emits per venue
is the direct input to step 2 (which fee tier, if any, makes this work).

Research only. No authentication, no order entry, no profit claims.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    # Importing jane_lab under the macOS system python (3.9) fails with an
    # opaque "dataclass() got an unexpected keyword argument 'slots'". Say what
    # is actually wrong and which interpreter to use instead.
    raise SystemExit(
        f"jane_hedge needs Python 3.11+ (pyproject requires-python), got "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        f"Run it with the project venv, e.g. .venv/bin/python3 tools/jane_hedge.py ..."
    )

try:  # package import, and the direct-script path via tools/tools.py
    from tools import jane_lab as base
except ImportError:  # pragma: no cover - exercised only outside the repo root
    import jane_lab as base  # type: ignore[no-redef]

VERSION = "2026-08-16-hedge-v1"
SOURCES = base.SOURCES
MARKET_GROUP = base.MARKET_GROUP
DEFAULT_TAKER_BPS = base.DEFAULT_TAKER_BPS
DEFAULT_MAKER_BPS = base.DEFAULT_MAKER_BPS

Level = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Execution:
    qty_base: float
    quote: float
    avg_price: float
    levels_used: int
    complete: bool


def walk_sell_base(bids: tuple[Level, ...], qty_target: float) -> Execution:
    """Sell ``qty_target`` base into the bid ladder, best price first."""
    remaining = qty_target
    sold = proceeds = 0.0
    used = 0
    for price, available in bids:
        if remaining <= 1e-15:
            break
        take = min(remaining, available)
        sold += take
        proceeds += take * price
        remaining -= take
        used += 1
    complete = remaining <= max(1e-15, qty_target * 1e-9)
    return Execution(sold, proceeds, proceeds / sold if sold > 0 else 0.0, used, complete)


def walk_buy_base(asks: tuple[Level, ...], qty_target: float) -> Execution:
    """Buy ``qty_target`` base from the ask ladder, best price first."""
    remaining = qty_target
    bought = spent = 0.0
    used = 0
    for price, available in asks:
        if remaining <= 1e-15:
            break
        take = min(remaining, available)
        bought += take
        spent += take * price
        remaining -= take
        used += 1
    complete = remaining <= max(1e-15, qty_target * 1e-9)
    return Execution(bought, spent, spent / bought if bought > 0 else 0.0, used, complete)


@dataclass(slots=True)
class MakerEvent:
    ts_ns: int
    fill_ts_ns: int
    maker_source: str
    maker_side: str
    maker_price: float
    qty_base: float
    notional: float
    quote_gross_bps: float
    recorded_hedge_source: str
    cluster_samples: int = 1


def load_maker_events(report_path: Path) -> list[MakerEvent]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("mt", {}).get("top_fill_proxy_passed", []) or []
    out: list[MakerEvent] = []
    for row in rows:
        fill = row.get("fill_proxy") or {}
        fill_ts = fill.get("fill_ts_ns")
        if not fill_ts:
            continue
        out.append(MakerEvent(
            ts_ns=int(row["ts_ns"]),
            fill_ts_ns=int(fill_ts),
            maker_source=row["maker_source"],
            maker_side=row["maker_side"],
            maker_price=float(row["maker_price"]),
            qty_base=float(row["qty_base"]),
            notional=float(row["notional"]),
            quote_gross_bps=float(row["gross_bps_at_quote"]),
            recorded_hedge_source=row["hedge_source"],
        ))
    return out


def dedup_events(events: list[MakerEvent], *, cluster_ms: float) -> list[MakerEvent]:
    """One dislocation spans several samples of the upstream top-N heap."""
    groups: dict[tuple[Any, ...], list[MakerEvent]] = {}
    for e in events:
        groups.setdefault((e.maker_source, e.maker_side, e.notional), []).append(e)
    out: list[MakerEvent] = []
    for items in groups.values():
        items.sort(key=lambda e: e.ts_ns)
        cluster: list[MakerEvent] = []
        for e in items:
            if cluster and (e.ts_ns - cluster[-1].ts_ns) / 1e6 > cluster_ms:
                out.append(_pick(cluster))
                cluster = []
            cluster.append(e)
        if cluster:
            out.append(_pick(cluster))
    out.sort(key=lambda e: e.ts_ns)
    return out


def _pick(cluster: list[MakerEvent]) -> MakerEvent:
    best = max(cluster, key=lambda e: e.quote_gross_bps)
    best.cluster_samples = len(cluster)
    return best


def collect_ladders(capture: Path, wanted: dict[str, list[int]], *,
                    max_late_ms: float) -> dict[tuple[str, int], dict[str, Any] | None]:
    """Stream the capture once and keep only the ladders the events need.

    For each (source, target_ts) this keeps the first book at or after the
    target, which is what a hedge order arriving at ``target_ts`` could actually
    have traded against. Books are written in receive order, so a single
    forward pass with one cursor per source is enough.
    """
    cursor = {s: 0 for s in wanted}
    found: dict[tuple[str, int], dict[str, Any] | None] = {}
    with capture.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") != "book":
                continue
            src = row.get("source")
            targets = wanted.get(src)
            if not targets:
                continue
            ts = int(row.get("receive_ts_ns") or 0)
            if ts <= 0:
                continue
            i = cursor[src]
            while i < len(targets) and ts >= targets[i]:
                late_ms = (ts - targets[i]) / 1e6
                found[(src, targets[i])] = None if late_ms > max_late_ms else {
                    "ts_ns": ts,
                    "bids": tuple((float(p), float(q)) for p, q in (row.get("bids") or ())),
                    "asks": tuple((float(p), float(q)) for p, q in (row.get("asks") or ())),
                    "late_ms": late_ms,
                }
                i += 1
            cursor[src] = i
    for src, targets in wanted.items():
        for t in targets[cursor[src]:]:
            found[(src, t)] = None  # capture ended before the hedge could land
    return found


def evaluate_hedge(event: MakerEvent, ladder: dict[str, Any] | None, hedge_source: str,
                   *, latency_ms: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "hedge_source": hedge_source,
        "hedge_latency_ms": latency_ms,
        "same_market_group": MARKET_GROUP[hedge_source] == MARKET_GROUP[event.maker_source],
        "self_hedge": hedge_source == event.maker_source,
    }
    if ladder is None:
        row["usable"] = False
        row["reason"] = "no hedge book within the freshness limit"
        return row
    levels = ladder["bids"] if event.maker_side == "buy" else ladder["asks"]
    if not levels:
        row["usable"] = False
        row["reason"] = "empty ladder"
        return row

    if event.maker_side == "buy":
        ex = walk_sell_base(levels, event.qty_base)
        top_px = levels[0][0]
        gross = (ex.avg_price / event.maker_price - 1.0) * 10_000.0
        top_gross = (top_px / event.maker_price - 1.0) * 10_000.0
    else:
        ex = walk_buy_base(levels, event.qty_base)
        top_px = levels[0][0]
        gross = (event.maker_price / ex.avg_price - 1.0) * 10_000.0
        top_gross = (event.maker_price / top_px - 1.0) * 10_000.0

    if not ex.complete:
        row["usable"] = False
        row["reason"] = "ladder too thin for the hedge size"
        row["fillable_qty_base"] = ex.qty_base
        return row

    maker_fee = DEFAULT_MAKER_BPS[event.maker_source]
    taker_fee = DEFAULT_TAKER_BPS[hedge_source]
    row.update({
        "usable": True,
        "hedge_book_late_ms": ladder["late_ms"],
        "hedge_avg_price": ex.avg_price,
        "levels_used": ex.levels_used,
        "top_of_book_gross_bps": top_gross,
        "gross_bps": gross,
        "slippage_bps": top_gross - gross,
        "maker_fee_bps": maker_fee,
        "hedge_taker_fee_bps": taker_fee,
        "net_bps": gross - maker_fee - taker_fee,
        # Step 2 input: the hedge taker fee at which this event breaks even.
        "break_even_hedge_taker_fee_bps": gross - maker_fee,
    })
    return row


def analyze(capture: Path, report_path: Path, *, latencies_ms: tuple[float, ...],
            max_late_ms: float, cluster_ms: float, hedge_sources: tuple[str, ...],
            top_n: int) -> dict[str, Any]:
    raw = load_maker_events(report_path)
    events = dedup_events(raw, cluster_ms=cluster_ms)
    if not events:
        return {"version": VERSION, "available": False,
                "reason": "no filled maker candidates in the xarb report"}

    wanted: dict[str, set[int]] = {s: set() for s in hedge_sources}
    for e in events:
        for lat in latencies_ms:
            t = e.fill_ts_ns + int(lat * 1e6)
            for s in hedge_sources:
                wanted[s].add(t)
    ordered = {s: sorted(v) for s, v in wanted.items()}
    ladders = collect_ladders(capture, ordered, max_late_ms=max_late_ms)

    detail: list[dict[str, Any]] = []
    for e in events:
        hedges: list[dict[str, Any]] = []
        for lat in latencies_ms:
            t = e.fill_ts_ns + int(lat * 1e6)
            for s in hedge_sources:
                hedges.append(evaluate_hedge(e, ladders.get((s, t)), s, latency_ms=lat))
        usable = [h for h in hedges if h["usable"]]
        detail.append({
            "ts_ns": e.ts_ns,
            "maker_source": e.maker_source,
            "maker_side": e.maker_side,
            "maker_price": e.maker_price,
            "notional": e.notional,
            "qty_base": e.qty_base,
            "quote_gross_bps": e.quote_gross_bps,
            "recorded_hedge_source": e.recorded_hedge_source,
            "maker_fill_delay_ms": (e.fill_ts_ns - e.ts_ns) / 1e6,
            "cluster_samples": e.cluster_samples,
            "best_hedge": max(usable, key=lambda h: h["net_bps"]) if usable else None,
            "hedges": hedges,
        })

    return {
        "version": VERSION,
        "available": True,
        "capture": str(capture),
        "xarb_report": str(report_path),
        "distinct_events": len(events),
        "raw_candidate_rows": len(raw),
        "cluster_ms": cluster_ms,
        "latencies_ms": list(latencies_ms),
        "max_hedge_book_late_ms": max_late_ms,
        "venue_ranking": rank_venues(detail, latencies_ms=latencies_ms,
                                     hedge_sources=hedge_sources),
        "events": detail[:top_n],
        "notes": [
            "Gross is rebuilt per venue from that venue's own ladder at the fill time; it is never derived by swapping a fee into another venue's gross.",
            "The executed hedge price is volume-weighted over real depth, so slippage from size is measured. Queue position on the maker leg is deliberately NOT modelled here (step 3); modelling it can only make results worse.",
            "Fee tables are the repo's standard-tier research assumptions, not the operator's tier. break_even_hedge_taker_fee_bps is the number to carry into the fee-tier step.",
            "Perp-vs-spot hedges leave basis exposure that a same-group hedge does not; same_market_group flags this per row.",
            "Source events are the extreme tail of one capture. A positive net here is a candidate to test further, not an expectation and not a fill rate.",
        ],
    }


def rank_venues(detail: list[dict[str, Any]], *, latencies_ms: tuple[float, ...],
                hedge_sources: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in hedge_sources:
        for lat in latencies_ms:
            nets, breaks, slips = [], [], []
            unusable = 0
            for ev in detail:
                for h in ev["hedges"]:
                    if h["hedge_source"] != s or h["hedge_latency_ms"] != lat:
                        continue
                    if not h["usable"]:
                        unusable += 1
                        continue
                    nets.append(h["net_bps"])
                    breaks.append(h["break_even_hedge_taker_fee_bps"])
                    slips.append(h["slippage_bps"])
            if not nets:
                out.append({"hedge_source": s, "hedge_latency_ms": lat, "n": 0,
                            "unusable": unusable})
                continue
            out.append({
                "hedge_source": s,
                "hedge_latency_ms": lat,
                "n": len(nets),
                "unusable": unusable,
                "net_positive": sum(1 for v in nets if v > 0),
                "median_net_bps": statistics.median(nets),
                "mean_net_bps": statistics.fmean(nets),
                "best_net_bps": max(nets),
                "median_break_even_taker_fee_bps": statistics.median(breaks),
                "best_break_even_taker_fee_bps": max(breaks),
                "median_slippage_bps": statistics.median(slips),
                "assumed_taker_fee_bps": DEFAULT_TAKER_BPS[s],
            })
    out.sort(key=lambda r: (r.get("median_net_bps") is not None,
                            r.get("median_net_bps", float("-inf"))), reverse=True)
    return out


def print_summary(rep: dict[str, Any]) -> None:
    print(f"[jane-hedge] version {VERSION}")
    print("\n============ ①ヘッジ先の実価格比較 ============")
    if not rep.get("available"):
        print("  利用不可:", rep.get("reason"))
        print("===============================================")
        return
    print(f"capture: {rep['capture']}")
    print(f"独立イベント: {rep['distinct_events']}件（元候補{rep['raw_candidate_rows']}行, "
          f"{rep['cluster_ms']:g}msで集約）")
    print(f"ヘッジ遅延: {', '.join(f'{v:g}ms' for v in rep['latencies_ms'])}")

    print("\nヘッジ先ランキング（ネット中央値, 板を実際に歩いた価格）:")
    print(f"  {'venue':14s} {'lat':>6s} {'n':>4s} {'net中央':>9s} {'net最良':>9s} "
          f"{'黒字':>5s} {'滑り中央':>9s} {'損益分岐Taker':>13s}")
    for r in rep["venue_ranking"]:
        if not r["n"]:
            print(f"  {r['hedge_source']:14s} {r['hedge_latency_ms']:5g}ms "
                  f"{0:4d}  （成立なし: 板不足/鮮度切れ {r['unusable']}件）")
            continue
        print(
            f"  {r['hedge_source']:14s} {r['hedge_latency_ms']:5g}ms {r['n']:4d} "
            f"{r['median_net_bps']:+9.3f} {r['best_net_bps']:+9.3f} "
            f"{r['net_positive']:4d}件 {r['median_slippage_bps']:9.3f} "
            f"{r['median_break_even_taker_fee_bps']:+13.3f}"
        )

    print("\nイベント別の最良ヘッジ:")
    for ev in rep["events"][:10]:
        best = ev["best_hedge"]
        if best is None:
            print(f"  {ev['maker_source']} {ev['maker_side']} ${ev['notional']:g}: "
                  f"成立するヘッジなし")
            continue
        flag = "" if best["same_market_group"] else " [basis]"
        print(
            f"  {ev['maker_source']} {ev['maker_side']} ${ev['notional']:g} "
            f"(quote={ev['quote_gross_bps']:+.3f}bps) -> {best['hedge_source']}"
            f"@{best['hedge_latency_ms']:g}ms{flag}: "
            f"gross={best['gross_bps']:+.3f} 滑り={best['slippage_bps']:.3f} "
            f"-maker{best['maker_fee_bps']:g}-taker{best['hedge_taker_fee_bps']:g} "
            f"=> net={best['net_bps']:+.3f}bps"
        )
    print("\n※ 損益分岐Taker = そのヘッジ先のTaker手数料が何bpsならトントンか（②の入力）。")
    print("※ キュー位置は未モデル化（③）。現実化すると結果は改善ではなく悪化します。")
    print("===============================================")


def selftest() -> None:
    bids = ((100.0, 1.0), (99.0, 5.0))
    ex = walk_sell_base(bids, 2.0)
    assert ex.complete and abs(ex.avg_price - 99.5) < 1e-12 and ex.levels_used == 2
    assert not walk_sell_base(bids, 10.0).complete
    asks = ((101.0, 1.0), (102.0, 5.0))
    ex = walk_buy_base(asks, 2.0)
    assert ex.complete and abs(ex.avg_price - 101.5) < 1e-12

    ev = MakerEvent(ts_ns=0, fill_ts_ns=1_000_000_000, maker_source="bybit_perp",
                    maker_side="buy", maker_price=100.0, qty_base=2.0, notional=200.0,
                    quote_gross_bps=6.0, recorded_hedge_source="binance_spot")
    # Thin top level forces the second level: slippage must be positive and the
    # walked gross must be worse than the top-of-book gross.
    lad = {"ts_ns": 1_000_000_000, "late_ms": 0.0, "bids": bids, "asks": asks}
    r = evaluate_hedge(ev, lad, "binance_perp", latency_ms=0.0)
    assert r["usable"] and r["slippage_bps"] > 0
    assert r["gross_bps"] < r["top_of_book_gross_bps"]
    # net must subtract both legs' fees, and break-even is net + the taker fee.
    assert abs(r["net_bps"] - (r["gross_bps"] - DEFAULT_MAKER_BPS["bybit_perp"]
                               - DEFAULT_TAKER_BPS["binance_perp"])) < 1e-12
    assert abs(r["break_even_hedge_taker_fee_bps"]
               - (r["net_bps"] + DEFAULT_TAKER_BPS["binance_perp"])) < 1e-12
    # Same gross on a cheaper venue must rank better purely through the fee.
    cheap = evaluate_hedge(ev, lad, "binance_perp", latency_ms=0.0)["net_bps"]
    dear = evaluate_hedge(ev, lad, "binance_spot", latency_ms=0.0)["net_bps"]
    assert cheap > dear
    assert not evaluate_hedge(ev, None, "okx_perp", latency_ms=0.0)["usable"]
    thin = {"ts_ns": 0, "late_ms": 0.0, "bids": ((100.0, 0.1),), "asks": asks}
    assert not evaluate_hedge(ev, thin, "okx_perp", latency_ms=0.0)["usable"]

    dup = [MakerEvent(ts_ns=int(ms * 1e6), fill_ts_ns=int(ms * 1e6) + 1, maker_source="bybit_perp",
                      maker_side="buy", maker_price=100.0, qty_base=1.0, notional=100.0,
                      quote_gross_bps=g, recorded_hedge_source="binance_spot")
           for ms, g in ((0, 6.0), (100, 6.5), (200, 6.2), (5_000, 3.0))]
    got = dedup_events(dup, cluster_ms=1000.0)
    assert len(got) == 2 and got[0].cluster_samples == 3 and got[0].quote_gross_bps == 6.5
    print("[jane-hedge] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare every hedge venue at real depth for maker->taker candidates")
    p.add_argument("capture", nargs="?", default="xarb-lab.jsonl")
    p.add_argument("--xarb-report", default="xarb-lab-report.json")
    p.add_argument("--report", default="jane-hedge-report.json")
    p.add_argument("--hedge-latency-ms", type=float, nargs="+", default=[0, 5, 20, 50])
    p.add_argument("--max-hedge-book-late-ms", type=float, default=250.0)
    p.add_argument("--cluster-ms", type=float, default=1000.0)
    p.add_argument("--hedge-source", nargs="+", default=list(SOURCES),
                   choices=list(SOURCES))
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    rep = analyze(
        Path(args.capture), Path(args.xarb_report),
        latencies_ms=tuple(sorted(set(args.hedge_latency_ms))),
        max_late_ms=args.max_hedge_book_late_ms,
        cluster_ms=args.cluster_ms,
        hedge_sources=tuple(args.hedge_source),
        top_n=args.top,
    )
    rep["config"] = vars(args)
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
