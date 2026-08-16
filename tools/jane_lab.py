#!/usr/bin/env python3
"""Integrated BTC relative-value / prediction research over xarb-lab captures.

This is deliberately an *offline* screening tool.  It reuses one capture and
checks multiple edge families together instead of asking the operator to run a
new experiment for each idea:

* cross-venue lead/lag momentum,
* top-of-book imbalance prediction,
* fair-value dislocation / mean reversion within spot and perp groups,
* execution-latency stress,
* fee-tier stress,
* maker/taker candidates re-priced at the observed maker-fill time.

No authentication. No order entry. No claims of guaranteed profit.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

VERSION = "2026-08-16-fusion-v2"
SOURCES = (
    "binance_spot", "binance_perp",
    "bybit_spot", "bybit_perp",
    "okx_spot", "okx_perp",
)
SPOTS = ("binance_spot", "bybit_spot", "okx_spot")
PERPS = ("binance_perp", "bybit_perp", "okx_perp")
MARKET_GROUP = {s: "spot" for s in SPOTS} | {s: "perp" for s in PERPS}

# Same research defaults already used by xarb_scan. They are assumptions, not
# a claim about the user's exact account tier.
DEFAULT_TAKER_BPS = {
    "binance_spot": 10.0,
    "binance_perp": 4.0,
    "bybit_spot": 10.0,
    "bybit_perp": 5.5,
    "okx_spot": 10.0,
    "okx_perp": 5.0,
}

# Standard (non-VIP) maker assumptions for the resting leg of a maker->taker
# hedge. Same status as the taker table: research defaults, not the user's tier.
DEFAULT_MAKER_BPS = {
    "binance_spot": 10.0,
    "binance_perp": 2.0,
    "bybit_spot": 10.0,
    "bybit_perp": 2.0,
    "okx_spot": 8.0,
    "okx_perp": 2.0,
}


@dataclass(slots=True)
class BookSeries:
    t: array = field(default_factory=lambda: array("q"))
    bid: array = field(default_factory=lambda: array("d"))
    bid_q: array = field(default_factory=lambda: array("d"))
    ask: array = field(default_factory=lambda: array("d"))
    ask_q: array = field(default_factory=lambda: array("d"))

    def append(self, ts: int, bids: list | tuple, asks: list | tuple) -> None:
        if not bids or not asks:
            return
        bp, bq = float(bids[0][0]), float(bids[0][1])
        ap, aq = float(asks[0][0]), float(asks[0][1])
        if not (bp > 0 and ap > bp and bq > 0 and aq > 0):
            return
        self.t.append(ts); self.bid.append(bp); self.bid_q.append(bq)
        self.ask.append(ap); self.ask_q.append(aq)

    def __len__(self) -> int:
        return len(self.t)

    def mid(self, i: int) -> float:
        return (self.bid[i] + self.ask[i]) * 0.5

    def imbalance(self, i: int) -> float:
        d = self.bid_q[i] + self.ask_q[i]
        return 0.0 if d <= 0 else (self.bid_q[i] - self.ask_q[i]) / d

    def latest_index(self, ts: int, max_age_ms: float | None = None) -> int | None:
        i = bisect.bisect_right(self.t, ts) - 1
        if i < 0:
            return None
        if max_age_ms is not None and ts - self.t[i] > int(max_age_ms * 1e6):
            return None
        return i

    def first_index(self, ts: int, max_late_ms: float | None = None) -> int | None:
        i = bisect.bisect_left(self.t, ts)
        if i >= len(self.t):
            return None
        if max_late_ms is not None and self.t[i] - ts > int(max_late_ms * 1e6):
            return None
        return i


@dataclass(slots=True)
class Dist:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    pos: int = 0
    min_v: float = math.inf
    max_v: float = -math.inf

    def add(self, x: float) -> None:
        self.n += 1; self.s += x; self.ss += x * x
        self.pos += int(x > 0)
        self.min_v = min(self.min_v, x); self.max_v = max(self.max_v, x)

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    @property
    def sd(self) -> float:
        if self.n < 2:
            return float("nan")
        v = max(0.0, (self.ss - self.s * self.s / self.n) / (self.n - 1))
        return math.sqrt(v)

    @property
    def tstat(self) -> float:
        if self.n < 2 or not math.isfinite(self.sd) or self.sd <= 0:
            return 0.0
        return self.mean / (self.sd / math.sqrt(self.n))


@dataclass(slots=True)
class SplitStats:
    train: Dist = field(default_factory=Dist)
    valid: Dist = field(default_factory=Dist)

    def add(self, x: float, is_valid: bool) -> None:
        (self.valid if is_valid else self.train).add(x)


@dataclass(frozen=True, slots=True)
class SignalKey:
    family: str
    leader: str
    follower: str
    feature_window_ms: int
    threshold: float
    horizon_ms: int
    latency_ms: int


def load_capture(path: Path) -> tuple[dict[str, BookSeries], int, int]:
    books = {s: BookSeries() for s in SOURCES}
    start = 2**63 - 1
    end = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") != "book":
                continue
            src = row.get("source")
            if src not in books:
                continue
            ts = int(row.get("receive_ts_ns") or 0)
            if ts <= 0:
                continue
            books[src].append(ts, row.get("bids") or (), row.get("asks") or ())
            start = min(start, ts); end = max(end, ts)
    if end <= 0 or start >= end:
        raise RuntimeError("capture contains no usable book events")
    missing = [s for s, bs in books.items() if not bs]
    if missing:
        raise RuntimeError("missing book series: " + ", ".join(missing))
    return books, start, end


def executable_roundtrip(
    bs: BookSeries,
    signal_ts: int,
    *,
    direction: int,
    horizon_ms: int,
    latency_ms: int,
    notional: float,
    max_book_age_ms: float,
    max_exit_late_ms: float,
) -> float | None:
    """Return gross bps for a taker entry then taker exit on one venue.

    Uses displayed top level only and refuses the sample if that level cannot
    carry the requested notional. This is intentionally conservative and does
    not invent hidden depth.
    """
    entry_ts = signal_ts + int(latency_ms * 1e6)
    ei = bs.first_index(entry_ts, max_late_ms=max_exit_late_ms)
    if ei is None or bs.t[ei] - entry_ts > int(max_book_age_ms * 1e6):
        return None
    exit_target = entry_ts + int(horizon_ms * 1e6)
    xi = bs.first_index(exit_target, max_late_ms=max_exit_late_ms)
    if xi is None:
        return None

    if direction > 0:
        entry_px, entry_q = bs.ask[ei], bs.ask_q[ei]
        qty = notional / entry_px
        if entry_q + 1e-15 < qty or bs.bid_q[xi] + 1e-15 < qty:
            return None
        return (bs.bid[xi] / entry_px - 1.0) * 10_000.0
    entry_px, entry_q = bs.bid[ei], bs.bid_q[ei]
    qty = notional / entry_px
    if entry_q + 1e-15 < qty or bs.ask_q[xi] + 1e-15 < qty:
        return None
    return (entry_px / bs.ask[xi] - 1.0) * 10_000.0


def add_thresholded(
    out: dict[SignalKey, SplitStats],
    *,
    family: str,
    leader: str,
    follower: str,
    feature_window_ms: int,
    feature_strength: float,
    thresholds: tuple[float, ...],
    horizon_ms: int,
    latency_ms: int,
    gross_bps: float,
    is_valid: bool,
) -> None:
    for th in thresholds:
        if feature_strength + 1e-12 < th:
            break
        key = SignalKey(family, leader, follower, feature_window_ms, th, horizon_ms, latency_ms)
        out.setdefault(key, SplitStats()).add(gross_bps, is_valid)


def scan_lead_lag(
    books: dict[str, BookSeries],
    *,
    split_ts: int,
    lookbacks_ms: tuple[int, ...],
    thresholds_bps: tuple[float, ...],
    horizons_ms: tuple[int, ...],
    latencies_ms: tuple[int, ...],
    notional: float,
    max_book_age_ms: float,
    max_exit_late_ms: float,
) -> dict[SignalKey, SplitStats]:
    out: dict[SignalKey, SplitStats] = {}
    min_th = min(thresholds_bps)
    for leader in SOURCES:
        ls = books[leader]
        for lb in lookbacks_ms:
            lb_ns = int(lb * 1e6)
            next_signal = 0
            for i in range(len(ls)):
                t = ls.t[i]
                if t < next_signal:
                    continue
                j = bisect.bisect_right(ls.t, t - lb_ns) - 1
                if j < 0 or j >= i:
                    continue
                old = ls.mid(j); now = ls.mid(i)
                move = (now / old - 1.0) * 10_000.0
                strength = abs(move)
                if strength < min_th:
                    continue
                direction = 1 if move > 0 else -1
                next_signal = t + lb_ns
                valid = t >= split_ts
                for follower in SOURCES:
                    if follower == leader:
                        continue
                    fs = books[follower]
                    for lat in latencies_ms:
                        for h in horizons_ms:
                            gross = executable_roundtrip(
                                fs, t, direction=direction, horizon_ms=h, latency_ms=lat,
                                notional=notional, max_book_age_ms=max_book_age_ms,
                                max_exit_late_ms=max_exit_late_ms,
                            )
                            if gross is None:
                                continue
                            add_thresholded(
                                out, family="lead_lag", leader=leader, follower=follower,
                                feature_window_ms=lb, feature_strength=strength,
                                thresholds=thresholds_bps, horizon_ms=h, latency_ms=lat,
                                gross_bps=gross, is_valid=valid,
                            )
    return out


def scan_imbalance(
    books: dict[str, BookSeries],
    *,
    split_ts: int,
    thresholds: tuple[float, ...],
    horizons_ms: tuple[int, ...],
    latencies_ms: tuple[int, ...],
    cooldown_ms: int,
    notional: float,
    max_book_age_ms: float,
    max_exit_late_ms: float,
) -> dict[SignalKey, SplitStats]:
    out: dict[SignalKey, SplitStats] = {}
    min_th = min(thresholds)
    for leader in SOURCES:
        ls = books[leader]
        next_signal = 0
        for i in range(len(ls)):
            t = ls.t[i]
            if t < next_signal:
                continue
            imb = ls.imbalance(i)
            strength = abs(imb)
            if strength < min_th:
                continue
            direction = 1 if imb > 0 else -1
            next_signal = t + int(cooldown_ms * 1e6)
            valid = t >= split_ts
            for follower in SOURCES:
                fs = books[follower]
                for lat in latencies_ms:
                    for h in horizons_ms:
                        gross = executable_roundtrip(
                            fs, t, direction=direction, horizon_ms=h, latency_ms=lat,
                            notional=notional, max_book_age_ms=max_book_age_ms,
                            max_exit_late_ms=max_exit_late_ms,
                        )
                        if gross is None:
                            continue
                        add_thresholded(
                            out, family="imbalance", leader=leader, follower=follower,
                            feature_window_ms=cooldown_ms, feature_strength=strength,
                            thresholds=thresholds, horizon_ms=h, latency_ms=lat,
                            gross_bps=gross, is_valid=valid,
                        )
    return out


def scan_fair_value(
    books: dict[str, BookSeries],
    *,
    start_ts: int,
    end_ts: int,
    split_ts: int,
    sample_ms: int,
    thresholds_bps: tuple[float, ...],
    horizons_ms: tuple[int, ...],
    latencies_ms: tuple[int, ...],
    notional: float,
    max_book_age_ms: float,
    max_exit_late_ms: float,
) -> dict[SignalKey, SplitStats]:
    out: dict[SignalKey, SplitStats] = {}
    min_th = min(thresholds_bps)
    step = int(sample_ms * 1e6)
    t = start_ts
    groups = {"spot": SPOTS, "perp": PERPS}
    while t <= end_ts:
        for gname, members in groups.items():
            mids: list[tuple[str, float]] = []
            for s in members:
                bs = books[s]
                i = bs.latest_index(t, max_age_ms=max_book_age_ms)
                if i is not None:
                    mids.append((s, bs.mid(i)))
            if len(mids) < 3:
                continue
            fair = statistics.median(m for _, m in mids)
            for follower, mid in mids:
                dev = (mid / fair - 1.0) * 10_000.0
                strength = abs(dev)
                if strength < min_th:
                    continue
                direction = 1 if dev < 0 else -1
                valid = t >= split_ts
                for lat in latencies_ms:
                    for h in horizons_ms:
                        gross = executable_roundtrip(
                            books[follower], t, direction=direction, horizon_ms=h,
                            latency_ms=lat, notional=notional,
                            max_book_age_ms=max_book_age_ms,
                            max_exit_late_ms=max_exit_late_ms,
                        )
                        if gross is None:
                            continue
                        add_thresholded(
                            out, family=f"fair_{gname}", leader=gname, follower=follower,
                            feature_window_ms=sample_ms, feature_strength=strength,
                            thresholds=thresholds_bps, horizon_ms=h, latency_ms=lat,
                            gross_bps=gross, is_valid=valid,
                        )
        t += step
    return out


def row_from_stats(key: SignalKey, st: SplitStats, *, fee_multiplier: float,
                   buffer_bps: float) -> dict[str, Any]:
    taker_each = DEFAULT_TAKER_BPS[key.follower] * fee_multiplier
    round_fee = 2.0 * taker_each
    train_net = st.train.mean - round_fee - buffer_bps if st.train.n else float("nan")
    valid_net = st.valid.mean - round_fee - buffer_bps if st.valid.n else float("nan")
    return {
        "family": key.family,
        "leader": key.leader,
        "follower": key.follower,
        "feature_window_ms": key.feature_window_ms,
        "threshold": key.threshold,
        "horizon_ms": key.horizon_ms,
        "latency_ms": key.latency_ms,
        "train_n": st.train.n,
        "valid_n": st.valid.n,
        "train_gross_mean_bps": st.train.mean,
        "valid_gross_mean_bps": st.valid.mean,
        "train_gross_positive_rate": st.train.pos / st.train.n if st.train.n else None,
        "valid_gross_positive_rate": st.valid.pos / st.valid.n if st.valid.n else None,
        "train_tstat": st.train.tstat,
        "valid_tstat": st.valid.tstat,
        "round_trip_fee_bps": round_fee,
        "buffer_bps": buffer_bps,
        "train_net_mean_bps": train_net,
        "valid_net_mean_bps": valid_net,
        "break_even_taker_each_bps_valid": (st.valid.mean - buffer_bps) / 2.0 if st.valid.n else None,
    }


def rank_stats(all_stats: Iterable[dict[SignalKey, SplitStats]], *,
               fee_multipliers: tuple[float, ...], buffer_bps: float,
               min_train: int, min_valid: int, top_n: int) -> dict[str, Any]:
    merged: dict[SignalKey, SplitStats] = {}
    for d in all_stats:
        merged.update(d)
    scenarios: dict[str, list[dict[str, Any]]] = {}
    for mult in fee_multipliers:
        rows = [row_from_stats(k, s, fee_multiplier=mult, buffer_bps=buffer_bps)
                for k, s in merged.items() if s.train.n >= min_train and s.valid.n >= min_valid]
        rows.sort(key=lambda r: (r["valid_net_mean_bps"], r["valid_tstat"], r["valid_n"]), reverse=True)
        scenarios[f"fee_x{mult:g}"] = rows[:top_n]
    robust = [row_from_stats(k, s, fee_multiplier=1.0, buffer_bps=buffer_bps)
              for k, s in merged.items() if s.train.n >= min_train and s.valid.n >= min_valid]
    robust = [r for r in robust if r["train_net_mean_bps"] > 0 and r["valid_net_mean_bps"] > 0]
    robust.sort(key=lambda r: (r["valid_net_mean_bps"], r["valid_tstat"]), reverse=True)
    return {"tested_configs": len(merged), "scenarios": scenarios,
            "robust_default_fee": robust[:top_n]}


def dedup_maker_rows(rows: list[dict[str, Any]], *, cluster_ms: float) -> list[dict[str, Any]]:
    """Collapse candidates that describe the same quote into one event.

    The upstream xarb report keeps the top-N samples by quote-time gross, so a
    single dislocation that survives several 100 ms samples appears many times.
    Counting those as separate opportunities inflates every downstream count.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["maker_source"], r["hedge_source"], r["maker_side"], r["notional"])
        groups.setdefault(key, []).append(r)
    events: list[dict[str, Any]] = []
    for key, items in groups.items():
        items.sort(key=lambda r: r["ts_ns"])
        cluster: list[dict[str, Any]] = []
        for r in items:
            if cluster and (r["ts_ns"] - cluster[-1]["ts_ns"]) / 1e6 > cluster_ms:
                events.append(_collapse_cluster(cluster))
                cluster = []
            cluster.append(r)
        if cluster:
            events.append(_collapse_cluster(cluster))
    events.sort(key=lambda r: r["net_bps_default_fees"], reverse=True)
    return events


def _collapse_cluster(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(cluster, key=lambda r: r["net_bps_default_fees"])
    out = dict(best)
    out["cluster_samples"] = len(cluster)
    out["cluster_span_ms"] = (cluster[-1]["ts_ns"] - cluster[0]["ts_ns"]) / 1e6
    return out


def summarize_maker_events(events: list[dict[str, Any]], *,
                           fee_multipliers: tuple[float, ...],
                           maker_fee_grid: tuple[float, ...]) -> dict[str, Any]:
    """Count how many *distinct* events survive each fee assumption."""
    grid: list[dict[str, Any]] = []
    for fm in fee_multipliers:
        for maker_fee in maker_fee_grid:
            nets = []
            for e in events:
                taker_fee = DEFAULT_TAKER_BPS[e["hedge_source"]] * fm
                nets.append(e["fill_time_gross_bps"] - taker_fee - maker_fee)
            grid.append({
                "fee_multiplier": fm,
                "maker_fee_bps": maker_fee,
                "net_positive_events": sum(1 for v in nets if v > 0),
                "best_net_bps": max(nets) if nets else None,
                "mean_net_bps": statistics.fmean(nets) if nets else None,
            })
    return {
        "distinct_events": len(events),
        "net_positive_events_default_fees": sum(1 for e in events if e["net_bps_default_fees"] > 0),
        "best_net_bps_default_fees": max((e["net_bps_default_fees"] for e in events), default=None),
        "best_gross_bps_at_fill": max((e["fill_time_gross_bps"] for e in events), default=None),
        "fee_grid": grid,
    }


def reprice_maker_candidates(report_path: Path | None, books: dict[str, BookSeries], *,
                             fee_multipliers: tuple[float, ...],
                             maker_fee_grid: tuple[float, ...],
                             max_book_age_ms: float,
                             cluster_ms: float = 1000.0) -> dict[str, Any]:
    if report_path is None or not report_path.exists():
        return {"available": False, "reason": "xarb report not supplied"}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("mt", {}).get("top_fill_proxy_passed", []) or []
    out: list[dict[str, Any]] = []
    for row in rows:
        fill = row.get("fill_proxy") or {}
        fill_ts = fill.get("fill_ts_ns")
        if not fill_ts:
            continue
        hedge_source = row["hedge_source"]
        bs = books[hedge_source]
        hi = bs.first_index(int(fill_ts), max_late_ms=max_book_age_ms)
        if hi is None:
            continue
        qty = float(row["qty_base"])
        side = row["maker_side"]
        maker_px = float(row["maker_price"])
        if side == "buy":
            if bs.bid_q[hi] + 1e-15 < qty:
                continue
            hedge_px = bs.bid[hi]
            gross = (hedge_px / maker_px - 1.0) * 10_000.0
        else:
            if bs.ask_q[hi] + 1e-15 < qty:
                continue
            hedge_px = bs.ask[hi]
            gross = (maker_px / hedge_px - 1.0) * 10_000.0
        scen = []
        for fm in fee_multipliers:
            taker_fee = DEFAULT_TAKER_BPS[hedge_source] * fm
            for maker_fee in maker_fee_grid:
                scen.append({
                    "fee_multiplier": fm,
                    "maker_fee_bps": maker_fee,
                    "net_bps_before_extra_buffers": gross - taker_fee - maker_fee,
                })
        maker_source = row["maker_source"]
        default_taker = DEFAULT_TAKER_BPS[hedge_source]
        default_maker = DEFAULT_MAKER_BPS[maker_source]
        out.append({
            "maker_source": maker_source,
            "hedge_source": hedge_source,
            "maker_side": side,
            "notional": row["notional"],
            "ts_ns": int(row["ts_ns"]),
            "quote_time_gross_bps": row["gross_bps_at_quote"],
            "fill_time_gross_bps": gross,
            # The headline number. Gross is not a P&L: the hedge leg always pays
            # taker fee and the resting leg pays the maker fee of its venue.
            "assumed_maker_fee_bps": default_maker,
            "assumed_hedge_taker_fee_bps": default_taker,
            "net_bps_default_fees": gross - default_taker - default_maker,
            # Maker fee (bps) at which this event would exactly break even.
            "break_even_maker_fee_bps": gross - default_taker,
            "maker_fill_delay_ms": (int(fill_ts) - int(row["ts_ns"])) / 1e6,
            "hedge_book_delay_ms": (int(bs.t[hi]) - int(fill_ts)) / 1e6,
            "scenarios": scen,
        })
    events = dedup_maker_rows(out, cluster_ms=cluster_ms)
    summary = summarize_maker_events(
        events, fee_multipliers=fee_multipliers, maker_fee_grid=maker_fee_grid,
    )
    return {
        "available": True,
        "checked": len(out),
        "cluster_ms": cluster_ms,
        "summary": summary,
        "top": events[:50],
        "selection_note": (
            "Source rows are the top-N samples by quote-time gross from the xarb "
            "report, i.e. the extreme tail of the capture. They are an upper "
            "bound on the opportunity, not a rate or an expectation."
        ),
    }


def print_summary(rep: dict[str, Any]) -> None:
    print(f"[jane-lab] version {VERSION}")
    print("\n================ 統合エッジ検証 ================")
    print(f"capture: {rep['capture']}")
    print("books : " + ", ".join(f"{s}={rep['book_counts'][s]:,}" for s in SOURCES))
    robust = rep["prediction"]["robust_default_fee"]
    print(f"\n現行想定手数料+bufferで学習/検証の両方が黒字: {len(robust)}件")
    for r in robust[:10]:
        print(
            f"{r['family']:11s} {r['leader']:14s}->{r['follower']:14s} "
            f"thr={r['threshold']:g} h={r['horizon_ms']}ms lat={r['latency_ms']}ms "
            f"train={r['train_net_mean_bps']:+.3f}bps(n={r['train_n']}) "
            f"valid={r['valid_net_mean_bps']:+.3f}bps(n={r['valid_n']},t={r['valid_tstat']:+.2f})"
        )
    if not robust:
        print("  なし")

    print("\n手数料を下げた場合の上位候補:")
    for label, rows in rep["prediction"]["scenarios"].items():
        if not rows:
            continue
        r = rows[0]
        print(
            f"  {label:9s}: {r['family']} {r['leader']}->{r['follower']} "
            f"valid={r['valid_net_mean_bps']:+.3f}bps n={r['valid_n']} "
            f"片道損益分岐手数料={r['break_even_taker_each_bps_valid']:+.3f}bps"
        )

    mk = rep["maker_repriced"]
    print("\nMaker→Taker（Maker約定時点の実ヘッジ価格で再計算）:")
    if not mk.get("available"):
        print("  利用不可:", mk.get("reason"))
    elif not mk.get("top"):
        print("  約定後ヘッジまで成立した候補なし")
    else:
        s = mk["summary"]
        print(
            f"  重複除去後の独立イベント: {s['distinct_events']}件"
            f"（元サンプル{mk['checked']}件, {mk['cluster_ms']:g}msで集約）"
        )
        print(
            f"  想定手数料でネット黒字: {s['net_positive_events_default_fees']}件 / "
            f"最良ネット={s['best_net_bps_default_fees']:+.3f}bps "
            f"(最良グロス={s['best_gross_bps_at_fill']:+.3f}bps)"
        )
        for r in mk["top"][:5]:
            print(
                f"  {r['maker_source']}->{r['hedge_source']} ${r['notional']:g} "
                f"gross={r['fill_time_gross_bps']:+.3f}bps "
                f"-maker{r['assumed_maker_fee_bps']:g}-taker{r['assumed_hedge_taker_fee_bps']:g} "
                f"=> net={r['net_bps_default_fees']:+.3f}bps "
                f"(損益分岐Maker手数料={r['break_even_maker_fee_bps']:+.3f}bps, "
                f"fill={r['maker_fill_delay_ms']:.1f}ms, x{r['cluster_samples']})"
            )
        print("  ※ 上記は捕捉期間中の最良サンプル（上振れ側の上限）であり、期待値ではありません。")
    print("=================================================")


def selftest() -> None:
    bs = BookSeries()
    bs.append(1_000_000_000, [[100, 10]], [[101, 10]])
    bs.append(1_100_000_000, [[102, 10]], [[103, 10]])
    g = executable_roundtrip(bs, 1_000_000_000, direction=1, horizon_ms=100,
                             latency_ms=0, notional=100, max_book_age_ms=10,
                             max_exit_late_ms=10)
    assert g is not None and g > 0
    d = Dist(); d.add(1); d.add(-1); assert d.n == 2 and abs(d.mean) < 1e-12

    def mk_row(ts_ms: float, gross: float) -> dict[str, Any]:
        taker = DEFAULT_TAKER_BPS["binance_spot"]
        maker = DEFAULT_MAKER_BPS["bybit_perp"]
        return {
            "maker_source": "bybit_perp", "hedge_source": "binance_spot",
            "maker_side": "buy", "notional": 100.0, "ts_ns": int(ts_ms * 1e6),
            "fill_time_gross_bps": gross,
            "net_bps_default_fees": gross - taker - maker,
        }

    # Three samples of one dislocation collapse to a single event; a later one
    # stays separate.
    ev = dedup_maker_rows(
        [mk_row(0, 6.0), mk_row(100, 6.5), mk_row(200, 6.2), mk_row(5_000, 3.0)],
        cluster_ms=1000.0,
    )
    assert len(ev) == 2, ev
    assert ev[0]["cluster_samples"] == 3 and abs(ev[0]["fill_time_gross_bps"] - 6.5) < 1e-12
    # 6.5 gross - 10 taker - 2 maker is a loss: gross alone is never the P&L.
    assert ev[0]["net_bps_default_fees"] < 0
    summ = summarize_maker_events(ev, fee_multipliers=(1.0, 0.0), maker_fee_grid=(2.0, 0.0))
    assert summ["distinct_events"] == 2 and summ["net_positive_events_default_fees"] == 0
    zero_fee = [g for g in summ["fee_grid"] if g["fee_multiplier"] == 0.0 and g["maker_fee_bps"] == 0.0][0]
    assert zero_fee["net_positive_events"] == 2
    print("[jane-lab] selftest PASS")


def main() -> int:
    p = argparse.ArgumentParser(description="Integrated Jane-Street-style screening over xarb-lab raw data")
    p.add_argument("capture", nargs="?", default="xarb-lab.jsonl")
    p.add_argument("--xarb-report", default="xarb-lab-report.json")
    p.add_argument("--report", default="jane-lab-report.json")
    p.add_argument("--notional", type=float, default=1000.0)
    p.add_argument("--lookback-ms", type=int, nargs="+", default=[50, 100, 250, 500])
    p.add_argument("--lead-threshold-bps", type=float, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--imbalance-threshold", type=float, nargs="+", default=[0.30, 0.50, 0.70])
    p.add_argument("--fair-threshold-bps", type=float, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--horizon-ms", type=int, nargs="+", default=[50, 100, 250, 500, 1000, 2000, 5000])
    p.add_argument("--latency-ms", type=int, nargs="+", default=[0, 25, 50, 100])
    p.add_argument("--fee-multiplier", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    p.add_argument("--maker-fee-bps", type=float, nargs="+", default=[2, 1, 0, -0.5, -1, -2])
    p.add_argument("--buffer-bps", type=float, default=1.0)
    p.add_argument("--max-book-age-ms", type=float, default=250.0)
    p.add_argument("--max-exit-late-ms", type=float, default=100.0)
    p.add_argument("--fair-sample-ms", type=int, default=100)
    p.add_argument("--imbalance-cooldown-ms", type=int, default=250)
    p.add_argument("--min-train", type=int, default=30)
    p.add_argument("--min-valid", type=int, default=20)
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    selftest()
    if args.selftest:
        return 0

    capture = Path(args.capture)
    books, start, end = load_capture(capture)
    split = start + int((end - start) * 0.70)
    print(f"[jane-lab] version {VERSION}")
    print("[jane-lab] loaded " + ", ".join(f"{s}={len(books[s]):,}" for s in SOURCES))
    print("[jane-lab] 1/4 取引所間の先行→追随を総当たり...")
    lead = scan_lead_lag(
        books, split_ts=split,
        lookbacks_ms=tuple(sorted(set(args.lookback_ms))),
        thresholds_bps=tuple(sorted(set(args.lead_threshold_bps))),
        horizons_ms=tuple(sorted(set(args.horizon_ms))),
        latencies_ms=tuple(sorted(set(args.latency_ms))),
        notional=args.notional, max_book_age_ms=args.max_book_age_ms,
        max_exit_late_ms=args.max_exit_late_ms,
    )
    print("[jane-lab] 2/4 板の偏りを総当たり...")
    imb = scan_imbalance(
        books, split_ts=split,
        thresholds=tuple(sorted(set(args.imbalance_threshold))),
        horizons_ms=tuple(sorted(set(args.horizon_ms))),
        latencies_ms=tuple(sorted(set(args.latency_ms))),
        cooldown_ms=args.imbalance_cooldown_ms, notional=args.notional,
        max_book_age_ms=args.max_book_age_ms, max_exit_late_ms=args.max_exit_late_ms,
    )
    print("[jane-lab] 3/4 合成適正価格からの乖離を総当たり...")
    fair = scan_fair_value(
        books, start_ts=start, end_ts=end, split_ts=split,
        sample_ms=args.fair_sample_ms,
        thresholds_bps=tuple(sorted(set(args.fair_threshold_bps))),
        horizons_ms=tuple(sorted(set(args.horizon_ms))),
        latencies_ms=tuple(sorted(set(args.latency_ms))),
        notional=args.notional, max_book_age_ms=args.max_book_age_ms,
        max_exit_late_ms=args.max_exit_late_ms,
    )
    print("[jane-lab] 4/4 手数料階級・Maker約定後ヘッジを再評価...")
    pred = rank_stats(
        [lead, imb, fair], fee_multipliers=tuple(args.fee_multiplier),
        buffer_bps=args.buffer_bps, min_train=args.min_train,
        min_valid=args.min_valid, top_n=args.top,
    )
    maker = reprice_maker_candidates(
        Path(args.xarb_report) if args.xarb_report else None, books,
        fee_multipliers=tuple(args.fee_multiplier), maker_fee_grid=tuple(args.maker_fee_bps),
        max_book_age_ms=args.max_book_age_ms,
    )
    rep = {
        "version": VERSION,
        "capture": str(capture),
        "book_counts": {s: len(books[s]) for s in SOURCES},
        "train_fraction": 0.70,
        "config": vars(args),
        "notes": [
            "Lead/lag and imbalance use receive timestamps from the same local capture clock.",
            "Prediction trades are taker-in/taker-out and require displayed top-level quantity to cover the requested notional.",
            "Fee multipliers are hypothetical sensitivity scenarios applied to the existing default research fee assumptions, not claims about a specific VIP tier.",
            "Funding is not present in xarb-lab v2 raw captures; for sub-5-second horizons it is negligible unless a funding timestamp is crossed. Longer-hold basis strategies need a funding-aware capture before promotion.",
            "Maker->taker rows report net after the resting leg's maker fee and the hedge leg's taker fee; the gross figure alone is not a P&L.",
            "Maker->taker source rows are the extreme tail selected by the xarb report, and are de-duplicated here because one dislocation spans several samples. Treat them as an upper bound, never as a fill rate or an expectation.",
            "A positive result is only a research candidate; require out-of-sample persistence, more days, and deployment-latency validation before live trading.",
        ],
        "prediction": pred,
        "maker_repriced": maker,
    }
    Path(args.report).write_text(json.dumps(rep, indent=2, allow_nan=False), encoding="utf-8")
    print_summary(rep)
    print(f"\nreport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
