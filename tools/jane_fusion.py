#!/usr/bin/env python3
"""True multivariate fusion screening over an existing xarb-lab capture.

Unlike jane_lab.py v1, this combines all six markets in one model instead of
ranking lead/lag, imbalance, and fair-value signals separately. It trains on
60%, tunes model/entry threshold on 10%, and leaves the final 30% untouched.
Research only: no authentication and no order entry.
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
from typing import Any

from tools import jane_lab as base

VERSION = "2026-08-16-fusion-v2"
SOURCES = base.SOURCES
SPOTS = base.SPOTS
PERPS = base.PERPS
DEFAULT_TAKER_BPS = base.DEFAULT_TAKER_BPS


@dataclass(slots=True)
class MarketSeries:
    t: array = field(default_factory=lambda: array("q"))
    bid: array = field(default_factory=lambda: array("d"))
    bid_q: array = field(default_factory=lambda: array("d"))
    ask: array = field(default_factory=lambda: array("d"))
    ask_q: array = field(default_factory=lambda: array("d"))
    imb1: array = field(default_factory=lambda: array("d"))
    imb5: array = field(default_factory=lambda: array("d"))

    def append(self, ts: int, bids, asks) -> None:
        if not bids or not asks:
            return
        try:
            bp, bq = float(bids[0][0]), float(bids[0][1])
            ap, aq = float(asks[0][0]), float(asks[0][1])
        except (TypeError, ValueError, IndexError):
            return
        if not (bp > 0 and ap > bp and bq > 0 and aq > 0):
            return
        b5 = a5 = 0.0
        for r in list(bids)[:5]:
            try:
                q = float(r[1])
                if q > 0:
                    b5 += q
            except (TypeError, ValueError, IndexError):
                pass
        for r in list(asks)[:5]:
            try:
                q = float(r[1])
                if q > 0:
                    a5 += q
            except (TypeError, ValueError, IndexError):
                pass
        d1, d5 = bq + aq, b5 + a5
        self.t.append(ts); self.bid.append(bp); self.bid_q.append(bq)
        self.ask.append(ap); self.ask_q.append(aq)
        self.imb1.append((bq - aq) / d1 if d1 else 0.0)
        self.imb5.append((b5 - a5) / d5 if d5 else 0.0)

    def __len__(self) -> int:
        return len(self.t)

    def mid(self, i: int) -> float:
        return 0.5 * (self.bid[i] + self.ask[i])

    def latest_index(self, ts: int, max_age_ms: float | None = None) -> int | None:
        i = bisect.bisect_right(self.t, ts) - 1
        if i < 0:
            return None
        if max_age_ms is not None and ts - self.t[i] > int(max_age_ms * 1e6):
            return None
        return i


@dataclass(slots=True)
class Dataset:
    times: array = field(default_factory=lambda: array("q"))
    x: array = field(default_factory=lambda: array("d"))
    p: int = 0
    feature_names: tuple[str, ...] = ()

    def append(self, ts: int, row: list[float]) -> None:
        if self.p == 0:
            self.p = len(row)
        if len(row) != self.p:
            raise ValueError("inconsistent feature width")
        self.times.append(ts)
        self.x.extend(row)

    def __len__(self) -> int:
        return len(self.times)


@dataclass(slots=True)
class Dist:
    n: int = 0
    s: float = 0.0
    ss: float = 0.0
    pos: int = 0

    def add(self, x: float) -> None:
        self.n += 1; self.s += x; self.ss += x * x; self.pos += int(x > 0)

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    @property
    def sd(self) -> float:
        if self.n < 2:
            return float("nan")
        return math.sqrt(max(0.0, (self.ss - self.s * self.s / self.n) / (self.n - 1)))

    @property
    def tstat(self) -> float:
        if self.n < 2 or not math.isfinite(self.sd) or self.sd <= 0:
            return 0.0
        return self.mean / (self.sd / math.sqrt(self.n))


def load_capture(path: Path) -> tuple[dict[str, MarketSeries], int, int]:
    markets = {s: MarketSeries() for s in SOURCES}
    start, end = 2**63 - 1, 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("kind") != "book" or row.get("source") not in markets:
                continue
            ts = int(row.get("receive_ts_ns") or 0)
            if ts <= 0:
                continue
            src = row["source"]
            n0 = len(markets[src])
            markets[src].append(ts, row.get("bids") or (), row.get("asks") or ())
            if len(markets[src]) > n0:
                start = min(start, ts); end = max(end, ts)
    missing = [s for s, ms in markets.items() if not ms]
    if missing:
        raise RuntimeError("missing book series: " + ", ".join(missing))
    return markets, start, end


def build_dataset(markets: dict[str, MarketSeries], *, start_ts: int, end_ts: int,
                  sample_ms: int, lookbacks_ms: tuple[int, ...],
                  max_book_age_ms: float) -> Dataset:
    names: list[str] = []
    for s in SOURCES:
        names += [f"{s}:ret_{lb}ms_bps" for lb in lookbacks_ms]
        names += [f"{s}:spread_bps", f"{s}:imb1", f"{s}:imb5"]
    names += [f"{s}:fair_dev_bps" for s in SOURCES]
    names += [f"{v}:perp_minus_spot_bps" for v in ("binance", "bybit", "okx")]
    ds = Dataset(feature_names=tuple(names))
    step = int(sample_ms * 1e6)
    t = start_ts
    while t <= end_ts:
        idx: dict[str, int] = {}
        for s in SOURCES:
            i = markets[s].latest_index(t, max_age_ms=max_book_age_ms)
            if i is None:
                break
            idx[s] = i
        if len(idx) != len(SOURCES):
            t += step; continue
        mids = {s: markets[s].mid(idx[s]) for s in SOURCES}
        row: list[float] = []
        ok = True
        for s in SOURCES:
            ms, i, mid = markets[s], idx[s], mids[s]
            for lb in lookbacks_ms:
                j = ms.latest_index(t - int(lb * 1e6), max_age_ms=max_book_age_ms)
                if j is None:
                    ok = False; break
                row.append((mid / ms.mid(j) - 1.0) * 10_000.0)
            if not ok:
                break
            row += [(ms.ask[i] / ms.bid[i] - 1.0) * 10_000.0, ms.imb1[i], ms.imb5[i]]
        if not ok:
            t += step; continue
        spot_fair = statistics.median(mids[s] for s in SPOTS)
        perp_fair = statistics.median(mids[s] for s in PERPS)
        for s in SOURCES:
            fair = spot_fair if s in SPOTS else perp_fair
            row.append((mids[s] / fair - 1.0) * 10_000.0)
        for v in ("binance", "bybit", "okx"):
            row.append((mids[f"{v}_perp"] / mids[f"{v}_spot"] - 1.0) * 10_000.0)
        ds.append(t, row)
        t += step
    return ds


def future_mid_return(ms: MarketSeries, signal_ts: int, horizon_ms: int) -> float | None:
    i0 = ms.latest_index(signal_ts)
    i1 = ms.latest_index(signal_ts + int(horizon_ms * 1e6))
    if i0 is None or i1 is None:
        return None
    return (ms.mid(i1) / ms.mid(i0) - 1.0) * 10_000.0


def executable_roundtrip(ms: MarketSeries, signal_ts: int, *, direction: int,
                         horizon_ms: int, latency_ms: int, notional: float,
                         max_book_age_ms: float) -> float | None:
    entry_ts = signal_ts + int(latency_ms * 1e6)
    exit_ts = entry_ts + int(horizon_ms * 1e6)
    ei = ms.latest_index(entry_ts, max_age_ms=max_book_age_ms)
    xi = ms.latest_index(exit_ts, max_age_ms=max_book_age_ms)
    if ei is None or xi is None:
        return None
    if direction > 0:
        px = ms.ask[ei]; qty = notional / px
        if ms.ask_q[ei] + 1e-15 < qty or ms.bid_q[xi] + 1e-15 < qty:
            return None
        return (ms.bid[xi] / px - 1.0) * 10_000.0
    px = ms.bid[ei]; qty = notional / px
    if ms.bid_q[ei] + 1e-15 < qty or ms.ask_q[xi] + 1e-15 < qty:
        return None
    return (px / ms.ask[xi] - 1.0) * 10_000.0


def split_bounds(start: int, end: int) -> tuple[int, int]:
    return start + int((end - start) * 0.60), start + int((end - start) * 0.70)


def scaler(ds: Dataset, train_end: int) -> tuple[list[float], list[float], list[int]]:
    p = ds.p; sums = [0.0] * p; sums2 = [0.0] * p; idxs: list[int] = []
    for i, ts in enumerate(ds.times):
        if ts >= train_end:
            break
        idxs.append(i); off = i * p
        for j in range(p):
            v = ds.x[off + j]; sums[j] += v; sums2[j] += v * v
    n = len(idxs)
    if n < 100:
        raise RuntimeError("not enough training rows")
    means = [v / n for v in sums]
    stds = []
    for j in range(p):
        sd = math.sqrt(max(0.0, sums2[j] / n - means[j] * means[j]))
        stds.append(sd if sd > 1e-9 else 1.0)
    return means, stds, idxs


def build_xtx(ds: Dataset, idxs: list[int], means: list[float], stds: list[float]) -> list[list[float]]:
    p, q = ds.p, ds.p + 1
    a = [[0.0] * q for _ in range(q)]
    for i in idxs:
        off = i * p; z = [1.0] + [(ds.x[off+j] - means[j]) / stds[j] for j in range(p)]
        for r in range(q):
            for c in range(r, q):
                a[r][c] += z[r] * z[c]
    for r in range(q):
        for c in range(r):
            a[r][c] = a[c][r]
    return a


def build_xty(ds: Dataset, idxs: list[int], means: list[float], stds: list[float],
               ms: MarketSeries, horizon_ms: int) -> list[float]:
    p = ds.p; b = [0.0] * (p + 1)
    for i in idxs:
        y = future_mid_return(ms, int(ds.times[i]), horizon_ms)
        if y is None:
            continue
        b[0] += y; off = i * p
        for j in range(p):
            b[j+1] += ((ds.x[off+j] - means[j]) / stds[j]) * y
    return b


def solve_ridge(xtx: list[list[float]], xty: list[float], alpha: float) -> list[float]:
    n = len(xty); a = [r[:] for r in xtx]; b = xty[:]
    for i in range(1, n):
        a[i][i] += alpha
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise RuntimeError("singular normal equation")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]; b[col], b[pivot] = b[pivot], b[col]
        inv = 1.0 / a[col][col]
        for j in range(col, n): a[col][j] *= inv
        b[col] *= inv
        for r in range(n):
            if r == col: continue
            f = a[r][col]
            if abs(f) < 1e-18: continue
            for j in range(col, n): a[r][j] -= f * a[col][j]
            b[r] -= f * b[col]
    return b


def predict(ds: Dataset, i: int, coef: list[float], means: list[float], stds: list[float]) -> float:
    off = i * ds.p; y = coef[0]
    for j in range(ds.p):
        y += coef[j+1] * ((ds.x[off+j] - means[j]) / stds[j])
    return y


def choose_alpha(ds: Dataset, ms: MarketSeries, horizon_ms: int,
                 coefs: dict[float, list[float]], means: list[float], stds: list[float],
                 train_end: int, tune_end: int) -> tuple[float, dict[str, float]]:
    winner = None; best_mse = math.inf; meta: dict[str, float] = {}
    for alpha, coef in coefs.items():
        n = 0; se = sy = sp = syy = spp = syp = 0.0
        for i, ts in enumerate(ds.times):
            if ts < train_end: continue
            if ts >= tune_end: break
            y = future_mid_return(ms, int(ts), horizon_ms)
            if y is None: continue
            pr = predict(ds, i, coef, means, stds); e = y - pr
            n += 1; se += e*e; sy += y; sp += pr; syy += y*y; spp += pr*pr; syp += y*pr
        if not n: continue
        mse = se / n
        den = math.sqrt(max(0.0, (syy-sy*sy/n)*(spp-sp*sp/n)))
        corr = (syp-sy*sp/n)/den if den else 0.0
        if mse < best_mse:
            winner, best_mse, meta = alpha, mse, {"tune_mse": mse, "tune_corr": corr, "tune_n": n}
    if winner is None: raise RuntimeError("no alpha evaluated")
    return winner, meta


def evaluate_model(ds: Dataset, ms: MarketSeries, coef: list[float], means: list[float], stds: list[float],
                   *, train_end: int, tune_end: int, end_ts: int, horizon_ms: int,
                   latencies_ms: tuple[int, ...], thresholds: tuple[float, ...],
                   notional: float, max_book_age_ms: float) -> tuple[dict[tuple[int,float],Dist], dict[tuple[int,float],Dist]]:
    tune = {(lat, th): Dist() for lat in latencies_ms for th in thresholds}
    valid = {(lat, th): Dist() for lat in latencies_ms for th in thresholds}
    for i, ts0 in enumerate(ds.times):
        ts = int(ts0)
        if ts < train_end: continue
        if ts > end_ts: break
        pr = predict(ds, i, coef, means, stds); strength = abs(pr)
        eligible = [th for th in thresholds if th <= strength + 1e-12]
        if not eligible: continue
        direction = 1 if pr > 0 else -1
        bucket = tune if ts < tune_end else valid
        for lat in latencies_ms:
            gross = executable_roundtrip(ms, ts, direction=direction, horizon_ms=horizon_ms,
                                         latency_ms=lat, notional=notional,
                                         max_book_age_ms=max_book_age_ms)
            if gross is None: continue
            for th in eligible:
                bucket[(lat, th)].add(gross)
    return tune, valid


def scan(ds: Dataset, markets: dict[str, MarketSeries], *, start_ts: int, end_ts: int,
         horizons: tuple[int,...], latencies: tuple[int,...], alphas: tuple[float,...],
         thresholds: tuple[float,...], fee_mults: tuple[float,...], buffer_bps: float,
         notional: float, max_book_age_ms: float, min_tune: int, min_valid: int) -> dict[str,Any]:
    train_end, tune_end = split_bounds(start_ts, end_ts)
    means, stds, idxs = scaler(ds, train_end)
    print(f"[jane-fusion] samples={len(ds):,} features={ds.p} train_rows={len(idxs):,}")
    print("[jane-fusion] building shared normal matrix...")
    xtx = build_xtx(ds, idxs, means, stds)
    rows: list[dict[str,Any]] = []; model_cache: dict[tuple[str,int],tuple[list[float],float]] = {}
    total = len(SOURCES)*len(horizons); k = 0
    for target in SOURCES:
        for h in horizons:
            k += 1; ms = markets[target]; xty = build_xty(ds, idxs, means, stds, ms, h)
            coefs = {a: solve_ridge(xtx, xty, a) for a in alphas}
            alpha, mm = choose_alpha(ds, ms, h, coefs, means, stds, train_end, tune_end)
            coef = coefs[alpha]; model_cache[(target,h)] = (coef, alpha)
            print(f"[jane-fusion] model {k:02d}/{total} {target} h={h}ms alpha={alpha:g} corr={mm['tune_corr']:+.3f}")
            tune, valid = evaluate_model(ds, ms, coef, means, stds, train_end=train_end,
                                         tune_end=tune_end, end_ts=end_ts, horizon_ms=h,
                                         latencies_ms=latencies, thresholds=thresholds,
                                         notional=notional, max_book_age_ms=max_book_age_ms)
            for lat in latencies:
                for fm in fee_mults:
                    cost = 2*DEFAULT_TAKER_BPS[target]*fm + buffer_bps
                    elig = [(th, tune[(lat,th)]) for th in thresholds if tune[(lat,th)].n >= min_tune]
                    if not elig: continue
                    th, td = max(elig, key=lambda z: (z[1].mean-cost, z[1].tstat, z[1].n))
                    vd = valid[(lat,th)]
                    if vd.n < min_valid: continue
                    rows.append({"target":target,"horizon_ms":h,"latency_ms":lat,"ridge_alpha":alpha,
                                 "model_tune_corr":mm["tune_corr"],"entry_threshold_bps":th,
                                 "fee_multiplier":fm,"round_trip_taker_fee_bps":2*DEFAULT_TAKER_BPS[target]*fm,
                                 "buffer_bps":buffer_bps,"tune_n":td.n,"tune_gross_mean_bps":td.mean,
                                 "tune_net_mean_bps":td.mean-cost,"tune_positive_rate":td.pos/td.n,
                                 "valid_n":vd.n,"valid_gross_mean_bps":vd.mean,"valid_net_mean_bps":vd.mean-cost,
                                 "valid_positive_rate":vd.pos/vd.n,"valid_tstat_gross":vd.tstat,
                                 "break_even_taker_each_bps_valid":(vd.mean-buffer_bps)/2})
    by_fee: dict[str,list[dict[str,Any]]] = {}
    for fm in fee_mults:
        rr = [r for r in rows if abs(r["fee_multiplier"]-fm)<1e-12]
        rr.sort(key=lambda r:(r["valid_net_mean_bps"],r["valid_tstat_gross"],r["valid_n"]), reverse=True)
        by_fee[f"fee_x{fm:g}"] = rr[:30]
    robust = [r for r in rows if abs(r["fee_multiplier"]-1)<1e-12 and r["tune_net_mean_bps"]>0 and r["valid_net_mean_bps"]>0]
    robust.sort(key=lambda r:(r["valid_net_mean_bps"],r["valid_tstat_gross"]), reverse=True)
    zero = [r for r in rows if abs(r["fee_multiplier"])<1e-12 and r["tune_net_mean_bps"]>0 and r["valid_net_mean_bps"]>0]
    zero.sort(key=lambda r:(r["valid_net_mean_bps"],r["valid_tstat_gross"]), reverse=True)
    ref = (robust or zero or sorted(rows,key=lambda r:r["valid_net_mean_bps"],reverse=True))[:1]
    importance: list[dict[str,Any]] = []
    if ref:
        r = ref[0]; coef,_ = model_cache[(r["target"],r["horizon_ms"])]
        pairs = sorted(((ds.feature_names[j],coef[j+1]) for j in range(ds.p)), key=lambda z:abs(z[1]), reverse=True)
        importance = [{"feature":n,"standardized_coef":w} for n,w in pairs[:20]]
    return {"split":{"train":0.60,"tune":0.10,"validation":0.30},"samples":len(ds),"features":ds.p,
            "feature_names":list(ds.feature_names),"tested_rows":len(rows),"robust_current_fee":robust[:30],
            "robust_even_zero_fee":zero[:30],"by_fee":by_fee,"reference_feature_importance":importance}


def print_report(rep: dict[str,Any]) -> None:
    f = rep["fusion"]
    print("\n================ 真の統合エッジ検証 ================")
    print(f"version : {rep['version']}"); print(f"capture : {rep['capture']}")
    print(f"samples : {f['samples']:,} / features={f['features']}")
    print("split   : 学習60% / 調整10% / 最終検証30%")
    print(f"\n現行想定手数料+bufferで調整/最終検証の両方が黒字: {len(f['robust_current_fee'])}件")
    if not f["robust_current_fee"]: print("  なし")
    for r in f["robust_current_fee"][:10]:
        print(f"  {r['target']:14s} h={r['horizon_ms']:4d}ms lat={r['latency_ms']:3d}ms valid={r['valid_net_mean_bps']:+.3f}bps n={r['valid_n']} th={r['entry_threshold_bps']:.2f}")
    print(f"\n手数料ゼロ（bufferは残す）でも調整/最終検証の両方が黒字: {len(f['robust_even_zero_fee'])}件")
    if not f["robust_even_zero_fee"]: print("  なし")
    for r in f["robust_even_zero_fee"][:10]:
        print(f"  {r['target']:14s} h={r['horizon_ms']:4d}ms lat={r['latency_ms']:3d}ms valid={r['valid_net_mean_bps']:+.3f}bps n={r['valid_n']} gross={r['valid_gross_mean_bps']:+.3f} hit={100*r['valid_positive_rate']:.1f}%")
    print("\n手数料別の最良候補:")
    for label, rr in f["by_fee"].items():
        if rr:
            r=rr[0]; print(f"  {label:9s}: {r['target']:14s} h={r['horizon_ms']:4d}ms lat={r['latency_ms']:3d}ms valid={r['valid_net_mean_bps']:+.3f}bps n={r['valid_n']} 損益分岐片道={r['break_even_taker_each_bps_valid']:+.3f}bps")
    if f["reference_feature_importance"]:
        print("\n最良モデルが強く使った特徴:")
        for r in f["reference_feature_importance"][:10]: print(f"  {r['feature']:38s} {r['standardized_coef']:+.4f}")
    mk=rep.get("maker_repriced",{}); print("\nMaker→Taker再評価:")
    if mk.get("available") and mk.get("top"):
        for r in mk["top"][:5]: print(f"  {r['maker_source']}->{r['hedge_source']} ${r['notional']:g} 約定後gross={r['fill_time_gross_bps']:+.3f}bps maker約定={r['maker_fill_delay_ms']:.1f}ms")
    else: print("  利用可能な候補なし")
    print("=====================================================")


def selftest() -> None:
    ms=MarketSeries(); ms.append(1_000_000_000,[[100,10],[99,8]],[[101,5],[102,7]]); ms.append(1_100_000_000,[[102,10],[101,8]],[[103,5],[104,7]])
    assert len(ms)==2 and ms.imb1[0]>0 and ms.imb5[0]>0
    g=executable_roundtrip(ms,1_000_000_000,direction=1,horizon_ms=100,latency_ms=0,notional=100,max_book_age_ms=1000)
    assert g is not None and g>0
    c=solve_ridge([[2.0,0.0],[0.0,2.0]],[4.0,6.0],0.0); assert abs(c[0]-2)<1e-9 and abs(c[1]-3)<1e-9
    print("[jane-fusion] selftest PASS")


def main() -> int:
    p=argparse.ArgumentParser(description="True multivariate six-market fusion screening")
    p.add_argument("capture",nargs="?",default="xarb-lab.jsonl"); p.add_argument("--xarb-report",default="xarb-lab-report.json"); p.add_argument("--report",default="jane-fusion-report.json")
    p.add_argument("--notional",type=float,default=1000.0); p.add_argument("--sample-ms",type=int,default=100); p.add_argument("--lookback-ms",type=int,nargs="+",default=[50,100,250,500])
    p.add_argument("--horizon-ms",type=int,nargs="+",default=[250,500,1000,2000,5000]); p.add_argument("--latency-ms",type=int,nargs="+",default=[0,25,50,100])
    p.add_argument("--ridge-alpha",type=float,nargs="+",default=[0.1,1,10,100]); p.add_argument("--entry-threshold-bps",type=float,nargs="+",default=[0.25,0.5,1,2,3,5])
    p.add_argument("--fee-multiplier",type=float,nargs="+",default=[1,0.75,0.5,0.25,0]); p.add_argument("--buffer-bps",type=float,default=1.0)
    p.add_argument("--max-book-age-ms",type=float,default=250.0); p.add_argument("--min-tune",type=int,default=20); p.add_argument("--min-valid",type=int,default=30); p.add_argument("--selftest",action="store_true")
    a=p.parse_args(); selftest()
    if a.selftest: return 0
    markets,start,end=load_capture(Path(a.capture)); print(f"[jane-fusion] version {VERSION}"); print("[jane-fusion] loaded "+", ".join(f"{s}={len(markets[s]):,}" for s in SOURCES))
    print("[jane-fusion] building synchronized multi-market features...")
    ds=build_dataset(markets,start_ts=start,end_ts=end,sample_ms=a.sample_ms,lookbacks_ms=tuple(sorted(set(a.lookback_ms))),max_book_age_ms=a.max_book_age_ms)
    fusion=scan(ds,markets,start_ts=start,end_ts=end,horizons=tuple(sorted(set(a.horizon_ms))),latencies=tuple(sorted(set(a.latency_ms))),alphas=tuple(sorted(set(a.ridge_alpha))),thresholds=tuple(sorted(set(a.entry_threshold_bps))),fee_mults=tuple(a.fee_multiplier),buffer_bps=a.buffer_bps,notional=a.notional,max_book_age_ms=a.max_book_age_ms,min_tune=a.min_tune,min_valid=a.min_valid)
    maker=base.reprice_maker_candidates(Path(a.xarb_report) if a.xarb_report else None,{s:base.BookSeries(t=markets[s].t,bid=markets[s].bid,bid_q=markets[s].bid_q,ask=markets[s].ask,ask_q=markets[s].ask_q) for s in SOURCES},fee_multipliers=tuple(a.fee_multiplier),maker_fee_grid=(2,1,0,-0.5,-1,-2),max_book_age_ms=a.max_book_age_ms)
    rep={"version":VERSION,"capture":a.capture,"config":vars(a),"notes":["All six markets are combined in one multivariate model; validation is not used for model or threshold selection.","Features: multi-venue returns, spread, top-level and five-level imbalance, fair-value deviations, spot-perp basis.","Scoring uses executable bid/ask round trips, displayed top-level capacity, latency, fees and buffer.","Spot short signals need pre-positioned BTC/borrow.","Funding is absent from xarb-lab v2 raw data; tests are <=5s.","Maker/Taker reuses conservative queue/tape candidates and reprices hedge at maker fill."],"book_counts":{s:len(markets[s]) for s in SOURCES},"fusion":fusion,"maker_repriced":maker}
    Path(a.report).write_text(json.dumps(rep,indent=2,allow_nan=False),encoding="utf-8"); print_report(rep); print(f"\nreport: {a.report}"); return 0

if __name__=="__main__": raise SystemExit(main())
