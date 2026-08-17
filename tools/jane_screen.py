#!/usr/bin/env python3
"""Step 1b: screen maker->taker pairs on basis-adjusted residual, not raw gross.

The upstream candidate generator ranks by raw quote-time gross. Raw gross
carries the persistent perp-vs-spot basis, so the top-N is structurally filled
with cross-group rows whose "edge" is a carry position, and genuine same-group
dislocations never reach the list. Step 1 confirmed this: the surviving
candidate's +6.761 bps against binance_spot was only +1.556 bps against
binance_perp, the ~5.2 bps difference being the perp premium at that instant.

This tool re-screens the whole capture with two changes:

* rank on residual = raw gross - EWMA basis between the two venues, so only
  deviation from the recent normal relationship counts as edge,
* report the whole sampled population - positive rate and distinct episodes -
  instead of the extreme tail, so a family can be judged rather than a moment.

Fills are deliberately not modelled: this measures whether the spread is ever
there to capture at all. If nothing clears fees here, the queue model (step 3)
cannot rescue it, because a realistic queue only makes results worse.

Research only. No authentication, no order entry, no profit claims.
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 11):
    raise SystemExit(
        f"jane_screen needs Python 3.11+ (pyproject requires-python), got "
        f"{sys.version.split()[0]} at {sys.executable}.\n"
        f"Run it with the project venv, e.g. .venv/bin/python3 tools/jane_screen.py ..."
    )

try:  # package import, and the direct-script path via tools/tools.py
    from tools import jane_lab as base
    from tools.jane_hedge import walk_buy_base, walk_sell_base
except ImportError:  # pragma: no cover - exercised only outside the repo root
    import jane_lab as base  # type: ignore[no-redef]
    from jane_hedge import walk_buy_base, walk_sell_base  # type: ignore[no-redef]

VERSION = "2026-08-17-screen-v1"
SOURCES = base.SOURCES
MARKET_GROUP = base.MARKET_GROUP
DEFAULT_TAKER_BPS = base.DEFAULT_TAKER_BPS
DEFAULT_MAKER_BPS = base.DEFAULT_MAKER_BPS


@dataclass(slots=True)
class Acc:
    """Population stats for one (maker, hedge, side, notional) family."""
    n: int = 0
    sum_adj: float = 0.0
    max_adj: float = -math.inf
    pos_adj: int = 0
    pos_raw: int = 0
    sum_raw: float = 0.0
    max_raw: float = -math.inf
    episodes: int = 0
    last_pos_ts: int | None = None
    top: list[tuple[float, int, dict[str, Any]]] = field(default_factory=list)

    def add(self, ts: int, *, net_adj: float, net_raw: float, row: dict[str, Any],
            episode_gap_ms: float, keep_top: int) -> None:
        self.n += 1
        self.sum_adj += net_adj
        self.sum_raw += net_raw
        self.max_adj = max(self.max_adj, net_adj)
        self.max_raw = max(self.max_raw, net_raw)
        if net_raw > 0:
            self.pos_raw += 1
        if net_adj > 0:
            self.pos_adj += 1
            if self.last_pos_ts is None or (ts - self.last_pos_ts) / 1e6 > episode_gap_ms:
                self.episodes += 1
            self.last_pos_ts = ts
            # Only the kept tail needs a materialised record.
            heapq.heappush(self.top, (net_adj, ts, {"ts_ns": ts} | row))
            if len(self.top) > keep_top:
                heapq.heappop(self.top)


def ewma_alpha(dt_ms: float, halflife_ms: float) -> float:
    if halflife_ms <= 0:
        return 1.0
    return 1.0 - math.exp(-math.log(2.0) * max(0.0, dt_ms) / halflife_ms)


def residual_bps(raw_gross_bps: float, basis_bps: float, side: str) -> float:
    """Strip the persistent venue-vs-venue basis out of a raw gross.

    ``basis_bps`` is how rich the hedge venue is versus the maker venue. A buy
    at the maker venue hedged by a sell at the hedge venue earns that basis
    without any dislocation, so it is subtracted; the sell side earns the
    negative of it, so it is added back.
    """
    return raw_gross_bps - basis_bps if side == "buy" else raw_gross_bps + basis_bps


def evaluate_sample(maker_book: dict[str, Any], hedge_book: dict[str, Any], *,
                    side: str, notional: float, basis_bps: float,
                    maker_fee_bps: float, taker_fee_bps: float) -> dict[str, Any] | None:
    """Quote-time economics of resting on one venue and hedging on the other."""
    if side == "buy":
        levels = maker_book["bids"]
        hedge_levels = hedge_book["bids"]
    else:
        levels = maker_book["asks"]
        hedge_levels = hedge_book["asks"]
    if not levels or not hedge_levels:
        return None
    maker_px = levels[0][0]
    if maker_px <= 0:
        return None
    qty = notional / maker_px

    if side == "buy":
        ex = walk_sell_base(hedge_levels, qty)
        if not ex.complete:
            return None
        raw = (ex.avg_price / maker_px - 1.0) * 10_000.0
    else:
        ex = walk_buy_base(hedge_levels, qty)
        if not ex.complete:
            return None
        raw = (maker_px / ex.avg_price - 1.0) * 10_000.0

    adj = residual_bps(raw, basis_bps, side)
    fees = maker_fee_bps + taker_fee_bps
    return {
        "maker_price": maker_px,
        "hedge_avg_price": ex.avg_price,
        "levels_used": ex.levels_used,
        "raw_gross_bps": raw,
        "basis_bps": basis_bps,
        "residual_bps": adj,
        "fees_bps": fees,
        "net_raw_bps": raw - fees,
        "net_residual_bps": adj - fees,
    }


def screen(capture: Path, *, notionals: tuple[float, ...], sample_ms: float,
           max_age_ms: float, halflife_ms: float, warmup_ms: float,
           episode_gap_ms: float, sources: tuple[str, ...], keep_top: int,
           same_group_only: bool) -> dict[str, Any]:
    pairs = [(m, h) for m in sources for h in sources
             if m != h and (not same_group_only or MARKET_GROUP[m] == MARKET_GROUP[h])]
    latest: dict[str, dict[str, Any]] = {}
    basis: dict[tuple[str, str], float] = {}
    basis_ts: dict[tuple[str, str], int] = {}
    acc: dict[tuple[str, str, str, float], Acc] = {}
    first_ts = 0
    samples = 0
    step = int(sample_ms * 1e6)
    next_sample: int | None = None

    def do_sample(ts: int) -> None:
        nonlocal samples
        samples += 1
        fresh = {s: b for s, b in latest.items()
                 if (ts - b["ts_ns"]) / 1e6 <= max_age_ms}
        for m, h in pairs:
            bm, bh = fresh.get(m), fresh.get(h)
            if bm is None or bh is None:
                continue
            key = (m, h)
            b_now = (bh["mid"] / bm["mid"] - 1.0) * 10_000.0
            prev = basis.get(key)
            if prev is None:
                basis[key] = b_now
                basis_ts[key] = ts
                continue
            a = ewma_alpha((ts - basis_ts[key]) / 1e6, halflife_ms)
            basis[key] = prev + a * (b_now - prev)
            basis_ts[key] = ts
            if (ts - first_ts) / 1e6 < warmup_ms:
                continue
            for side in ("buy", "sell"):
                for notional in notionals:
                    row = evaluate_sample(
                        bm, bh, side=side, notional=notional, basis_bps=basis[key],
                        maker_fee_bps=DEFAULT_MAKER_BPS[m],
                        taker_fee_bps=DEFAULT_TAKER_BPS[h],
                    )
                    if row is None:
                        continue
                    acc.setdefault((m, h, side, notional), Acc()).add(
                        ts, net_adj=row["net_residual_bps"], net_raw=row["net_raw_bps"],
                        row=row,
                        episode_gap_ms=episode_gap_ms, keep_top=keep_top,
                    )

    with capture.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") != "book":
                continue
            src = row.get("source")
            if src not in sources:
                continue
            ts = int(row.get("receive_ts_ns") or 0)
            if ts <= 0:
                continue
            if next_sample is None:
                first_ts = ts
                next_sample = ts + step
            while next_sample <= ts:
                do_sample(next_sample)
                next_sample += step
            bids = tuple((float(p), float(q)) for p, q in (row.get("bids") or ()))
            asks = tuple((float(p), float(q)) for p, q in (row.get("asks") or ()))
            if not bids or not asks:
                continue
            latest[src] = {"ts_ns": ts, "bids": bids, "asks": asks,
                           "mid": (bids[0][0] + asks[0][0]) * 0.5}

    return {
        "version": VERSION,
        "capture": str(capture),
        "samples": samples,
        "families": summarize(acc),
        "notes": [
            "Residual = raw gross - EWMA basis between the two venues; only deviation from the recent normal relationship is counted as edge.",
            "net_residual subtracts the maker fee of the resting venue and the taker fee of the hedge venue at the repo's standard-tier assumptions.",
            "The hedge price is walked over real depth, so size-driven slippage is included. Maker fills are NOT modelled: every sample assumes the resting order would have been filled at the touch, which is optimistic.",
            "positive_episodes merges consecutive positive samples so one dislocation counts once.",
            "raw_* columns are the unadjusted view the old ranking used, kept only to show how much of it was basis.",
        ],
    }


def summarize(acc: dict[tuple[str, str, str, float], Acc]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (m, h, side, notional), a in acc.items():
        if not a.n:
            continue
        out.append({
            "maker_source": m,
            "hedge_source": h,
            "side": side,
            "notional": notional,
            "same_market_group": MARKET_GROUP[m] == MARKET_GROUP[h],
            "samples": a.n,
            "fees_bps": DEFAULT_MAKER_BPS[m] + DEFAULT_TAKER_BPS[h],
            "net_residual_mean_bps": a.sum_adj / a.n,
            "net_residual_best_bps": a.max_adj,
            "net_residual_positive": a.pos_adj,
            "net_residual_positive_rate": a.pos_adj / a.n,
            "positive_episodes": a.episodes,
            "net_raw_mean_bps": a.sum_raw / a.n,
            "net_raw_best_bps": a.max_raw,
            "net_raw_positive": a.pos_raw,
            "top": [d for _, _, d in sorted(a.top, reverse=True)],
        })
    out.sort(key=lambda r: (r["positive_episodes"], r["net_residual_best_bps"]), reverse=True)
    return out


def print_summary(rep: dict[str, Any], *, top_n: int) -> None:
    print(f"[jane-screen] version {VERSION}")
    print("\n======== ①b ベーシス調整後の候補スクリーニング ========")
    print(f"capture: {rep['capture']}")
    print(f"サンプル数: {rep['samples']:,}")
    fams = rep["families"]
    if not fams:
        print("  成立したファミリーなし（板不足/鮮度切れ）")
        print("=======================================================")
        return

    live = [f for f in fams if f["positive_episodes"] > 0]
    print(f"\n手数料を超えた（残差ベース）ファミリー: {len(live)} / {len(fams)}")
    print(f"  {'maker':13s}->{'hedge':13s} {'side':4s} {'$':>7s} {'黒字回数':>7s} "
          f"{'ep':>4s} {'黒字率':>7s} {'net最良':>8s} {'net平均':>8s} {'手数料':>6s} {'旧黒字':>7s}")
    for f in fams[:top_n]:
        flag = "" if f["same_market_group"] else " [basis]"
        print(
            f"  {f['maker_source']:13s}->{f['hedge_source']:13s} {f['side']:4s} "
            f"{f['notional']:7g} {f['net_residual_positive']:7d} "
            f"{f['positive_episodes']:4d} {f['net_residual_positive_rate']*100:6.3f}% "
            f"{f['net_residual_best_bps']:+8.3f} {f['net_residual_mean_bps']:+8.3f} "
            f"{f['fees_bps']:6.1f} {f['net_raw_positive']:7d}{flag}"
        )
    print("\n※ net = 残差 − Maker手数料 − Taker手数料。旧黒字 = ベーシス調整前の黒字サンプル数。")
    print("※ 約定は未モデル化。指値が必ず約定した前提なので、この数字は上振れ側です。")
    print("=======================================================")


def selftest() -> None:
    assert abs(residual_bps(6.7, 5.2, "buy") - 1.5) < 1e-12
    assert abs(residual_bps(-4.0, 5.2, "sell") - 1.2) < 1e-12
    assert ewma_alpha(0.0, 1000.0) == 0.0
    assert abs(ewma_alpha(1000.0, 1000.0) - 0.5) < 1e-12
    assert ewma_alpha(100.0, 0.0) == 1.0

    def book(mid: float, qty: float = 10.0) -> dict[str, Any]:
        return {"ts_ns": 0, "bids": ((mid - 1.0, qty),), "asks": ((mid + 1.0, qty),),
                "mid": mid}

    # A pure, persistent 5 bps basis is not edge: once the EWMA has learnt it,
    # the residual must collapse even though the raw gross stays large.
    maker, hedge = book(100_000.0), book(100_050.0)
    b = (hedge["mid"] / maker["mid"] - 1.0) * 10_000.0
    row = evaluate_sample(maker, hedge, side="buy", notional=1000.0, basis_bps=b,
                          maker_fee_bps=2.0, taker_fee_bps=4.0)
    assert row is not None
    assert row["raw_gross_bps"] > 4.0
    assert abs(row["residual_bps"]) < 0.3, row
    assert row["net_residual_bps"] < row["net_raw_bps"]

    # A real dislocation on top of the same basis must survive the adjustment.
    row2 = evaluate_sample(maker, book(100_100.0), side="buy", notional=1000.0,
                           basis_bps=b, maker_fee_bps=2.0, taker_fee_bps=4.0)
    assert row2 is not None and row2["residual_bps"] > 4.0

    # Thin hedge depth is refused rather than silently filled at the touch.
    assert evaluate_sample(maker, book(100_050.0, qty=1e-9), side="buy",
                           notional=1000.0, basis_bps=b, maker_fee_bps=2.0,
                           taker_fee_bps=4.0) is None

    a = Acc()
    det = {"x": 1}
    for ms in (0, 100, 200, 5_000):
        a.add(int(ms * 1e6), net_adj=1.0, net_raw=1.0, row=det,
              episode_gap_ms=1000.0, keep_top=5)
    assert a.n == 4 and a.pos_adj == 4 and a.episodes == 2, a
    a2 = Acc()
    a2.add(0, net_adj=-1.0, net_raw=1.0, row=det, episode_gap_ms=1000.0, keep_top=5)
    assert a2.pos_adj == 0 and a2.pos_raw == 1 and a2.episodes == 0
    print("[jane-screen] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Screen maker->taker families on basis-adjusted residual")
    p.add_argument("capture", nargs="?", default="xarb-lab.jsonl")
    p.add_argument("--report", default="jane-screen-report.json")
    p.add_argument("--notional", type=float, nargs="+", default=[1000.0])
    p.add_argument("--sample-ms", type=float, default=100.0)
    p.add_argument("--max-age-ms", type=float, default=250.0)
    p.add_argument("--basis-halflife-ms", type=float, default=30_000.0)
    p.add_argument("--warmup-ms", type=float, default=60_000.0)
    p.add_argument("--episode-gap-ms", type=float, default=1000.0)
    p.add_argument("--source", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--same-group-only", action="store_true",
                   help="restrict to perp->perp and spot->spot, dropping basis pairs entirely")
    p.add_argument("--keep-top", type=int, default=20)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    rep = screen(
        Path(args.capture),
        notionals=tuple(args.notional), sample_ms=args.sample_ms,
        max_age_ms=args.max_age_ms, halflife_ms=args.basis_halflife_ms,
        warmup_ms=args.warmup_ms, episode_gap_ms=args.episode_gap_ms,
        sources=tuple(args.source), keep_top=args.keep_top,
        same_group_only=args.same_group_only,
    )
    rep["config"] = vars(args)
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep, top_n=args.top)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
