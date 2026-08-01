#!/usr/bin/env python3
"""Inspect a market-data file and report what it can and cannot drive.

Point it at one day of your generated data:

    python tools/inspect_data.py ~/environment/opt/gen/2026-03-09.csv

It prints the schema it found and then answers the only question that
matters for a market-making backtest: is there enough here to fill a
resting order honestly?

Three things are needed, and they are not equally easy to come by:

  price + size + timestamp   replay the tape at all
  aggressor side             know whether a print would have hit our bid
                             or our offer — without it a resting order
                             cannot be filled, only guessed at
  book depth                 seed queue position. Depth is what tells us
                             how much size was already standing at our
                             price when we arrived, and queue position is
                             most of a maker's edge

Trades alone reconstruct a tape, not a book. No amount of post-processing
recovers depth that was never recorded.

Runs on the standard library. pandas/pyarrow are used only if a Parquet
file is passed and they happen to be installed.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Column-name heuristics. Japanese venues and vendors are all over the place,
# so match generously and let the human confirm from the sample rows.
PATTERNS: dict[str, tuple[str, ...]] = {
    "timestamp": ("ts", "time", "timestamp", "datetime", "date", "時刻", "日時", "時間"),
    "price": ("price", "px", "last", "trade_price", "価格", "約定値", "約定価格", "値段"),
    "size": ("size", "qty", "quantity", "volume", "vol", "amount", "数量", "出来高", "約定数量"),
    "side": ("side", "aggressor", "direction", "taker", "buy_sell", "bs", "売買", "売買区分"),
    "bid_price": ("bid", "bid_price", "bidpx", "best_bid", "買気配", "買気配値", "bid1"),
    "ask_price": ("ask", "ask_price", "askpx", "best_ask", "offer", "売気配", "売気配値", "ask1"),
    "bid_size": ("bid_size", "bidqty", "bid_volume", "bidsz", "買気配数量", "bid_qty"),
    "ask_size": ("ask_size", "askqty", "ask_volume", "asksz", "売気配数量", "ask_qty"),
    "ohlc": ("open", "high", "low", "close", "始値", "高値", "安値", "終値"),
}

ENCODINGS = ("utf-8-sig", "utf-8", "cp932", "euc-jp")


@dataclass
class Findings:
    path: Path
    size_bytes: int
    fmt: str = "unknown"
    encoding: str = ""
    delimiter: str = ""
    columns: list[str] = field(default_factory=list)
    rows: int = 0
    sample: list[list[str]] = field(default_factory=list)
    matched: dict[str, list[str]] = field(default_factory=dict)
    note: str = ""


# ------------------------------------------------------------------ opening


def sniff_format(path: Path) -> str:
    with path.open("rb") as fh:
        head = fh.read(8)
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"PAR1":
        return "parquet"
    if head[:1] in (b"{", b"["):
        return "jsonl"
    return "text"


def open_text(path: Path, gzipped: bool) -> tuple[io.TextIOBase, str]:
    """Open with whichever encoding actually decodes. cp932 is common for JPX."""
    opener = gzip.open if gzipped else open
    last: Exception | None = None
    for enc in ENCODINGS:
        try:
            fh = opener(path, "rt", encoding=enc, newline="")
            fh.read(64_000)
            fh.seek(0)
            return fh, enc
        except (UnicodeDecodeError, LookupError) as exc:
            last = exc
            continue
    raise RuntimeError(f"could not decode {path} with any of {ENCODINGS}: {last}")


# ------------------------------------------------------------------ reading


def read_delimited(path: Path, gzipped: bool) -> Findings:
    out = Findings(path=path, size_bytes=path.stat().st_size)
    out.fmt = "CSV/TSV" + (" (gzip)" if gzipped else "")

    fh, enc = open_text(path, gzipped)
    out.encoding = enc
    with fh:
        head = fh.read(64_000)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(head, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        out.delimiter = {"\t": "TAB"}.get(delimiter, delimiter)

        reader = csv.reader(fh, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            out.note = "file is empty"
            return out

        # A header of pure numbers means there is no header row.
        if all(_looks_numeric(c) for c in header if c.strip()):
            out.columns = [f"col{i}" for i in range(len(header))]
            out.sample.append(header)
            out.note = "no header row detected; columns are positional"
        else:
            out.columns = [c.strip() for c in header]

        for row in reader:
            if len(out.sample) < 5:
                out.sample.append(row)
            out.rows += 1
        out.rows += len(out.sample) if out.note else 0
    return out


def read_jsonl(path: Path, gzipped: bool) -> Findings:
    out = Findings(path=path, size_bytes=path.stat().st_size)
    out.fmt = "JSONL" + (" (gzip)" if gzipped else "")

    fh, enc = open_text(path, gzipped)
    out.encoding = enc
    keys: list[str] = []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.rows += 1
            if len(out.sample) < 5:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    for k in obj:
                        if k not in keys:
                            keys.append(k)
                    out.sample.append([_short(obj.get(k)) for k in keys])
    out.columns = keys
    return out


def read_parquet(path: Path) -> Findings:
    out = Findings(path=path, size_bytes=path.stat().st_size, fmt="Parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        out.note = "install pyarrow to inspect Parquet: pip install pyarrow"
        return out

    pf = pq.ParquetFile(path)
    out.columns = list(pf.schema_arrow.names)
    out.rows = pf.metadata.num_rows
    batch = next(pf.iter_batches(batch_size=5), None)
    if batch is not None:
        cols = batch.to_pydict()
        for i in range(min(5, batch.num_rows)):
            out.sample.append([_short(cols[c][i]) for c in out.columns])
    return out


def _looks_numeric(value: str) -> bool:
    try:
        float(value.replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


def _short(value, width: int = 26) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= width else text[: width - 1] + "…"


# ----------------------------------------------------------------- matching


def match_columns(columns: list[str]) -> dict[str, list[str]]:
    """Map each requirement to the columns that plausibly satisfy it."""
    found: dict[str, list[str]] = {}
    lowered = [(c, c.strip().lower()) for c in columns]
    for role, needles in PATTERNS.items():
        hits = []
        for original, low in lowered:
            if any(n in low for n in needles):
                hits.append(original)
        if hits:
            found[role] = hits

    # Substring matching over-fires in predictable ways: "bid_size" contains
    # "bid", "ask_px" contains "px". Quote columns are the more specific
    # reading, so let them win and strip the looser roles.
    def drop(role: str, taken: set[str]) -> None:
        if role in found:
            kept = [c for c in found[role] if c not in taken]
            if kept:
                found[role] = kept
            else:
                del found[role]

    for side in ("bid", "ask"):
        drop(f"{side}_price", set(found.get(f"{side}_size", [])))

    quote_cols = set()
    for role in ("bid_price", "ask_price", "bid_size", "ask_size"):
        quote_cols |= set(found.get(role, []))
    drop("price", quote_cols)
    drop("size", quote_cols)

    return found


# ------------------------------------------------------------------ report


def report(f: Findings) -> int:
    mb = f.size_bytes / 1024 / 1024
    print()
    print(f"file      : {f.path.name}  ({mb:,.1f} MB)")
    line = f"format    : {f.fmt}"
    if f.encoding:
        line += f", encoding={f.encoding}"
    if f.delimiter:
        line += f", delimiter={f.delimiter}"
    print(line)
    print(f"rows      : {f.rows:,}")
    if f.note:
        print(f"note      : {f.note}")

    if not f.columns:
        print("\nno columns could be read.")
        return 2

    print(f"\ncolumns ({len(f.columns)})")
    for i, name in enumerate(f.columns):
        values = [row[i] for row in f.sample if i < len(row)]
        print(f"  {name:<24} {' | '.join(_short(v, 18) for v in values[:3])}")

    m = f.matched
    has_tape = all(k in m for k in ("timestamp", "price", "size"))
    has_side = "side" in m
    has_bbo = "bid_price" in m and "ask_price" in m
    has_depth = has_bbo and "bid_size" in m and "ask_size" in m
    looks_like_bars = "ohlc" in m

    print("\njsboard に必要なもの")
    _row("約定の時刻・価格・数量", has_tape, m, ("timestamp", "price", "size"))
    _row("約定の攻め手（売り崩し / 買い上がり）", has_side, m, ("side",))
    _row("気配値（最良買い / 最良売り）", has_bbo, m, ("bid_price", "ask_price"))
    _row("気配数量（キュー位置の初期値）", has_depth, m, ("bid_size", "ask_size"))

    print("\n判定")
    code = 0
    if looks_like_bars and not has_tape:
        print("  ✗ バー（OHLC）に集約されています。")
        print("    個々の約定が失われているため、ペーパー約定モデルは動きません。")
        print("    MM の検証をするなら、生成側で約定を残す必要があります。")
        code = 1
    elif has_depth:
        print("  ✓ マーケットメイクのバックテストに必要なものが揃っています。")
        print("    キュー位置モデルをそのまま使えます。ReplayFeed のアダプタを書けば繋がります。")
    elif has_bbo and has_side:
        print("  △ 約定と気配値はあるが、気配数量がありません。")
        print("    約定判定はできますが、キュー位置は推定になります。")
        print("    「自分の前に何枚並んでいるか」が不明なので、約定率は楽観側に偏ります。")
        code = 1
    elif has_tape and has_side:
        print("  △ 約定列のみ（気配なし）。")
        print("    テイカー戦略の検証はできますが、マーケットメイクには板が要ります。")
        print("    板は約定データから復元できません。生成側で別途取る必要があります。")
        code = 1
    elif has_tape:
        print("  ✗ 約定はありますが、攻め手が判別できません。")
        print("    自分の指値が約定したかを判定できないため、このままでは MM を回せません。")
        print("    価格が最良買い / 最良売りのどちらで起きたかが分かる列を探してください。")
        code = 1
    else:
        print("  ? 想定した列名に当てはまりませんでした。")
        print("    上の columns を見せてもらえれば、こちらで対応付けます。")
        code = 2

    print()
    return code


def _row(label: str, ok: bool, matched: dict[str, list[str]], roles: tuple[str, ...]) -> None:
    mark = "✓" if ok else "✗"
    names = []
    for role in roles:
        names.extend(matched.get(role, []))
    detail = ", ".join(names) if names else "見つからず"
    print(f"  {mark} {label:<38} {detail}")


def inspect(path: Path) -> int:
    if not path.exists():
        print(f"not found: {path}", file=sys.stderr)
        return 2

    fmt = sniff_format(path)
    if fmt == "parquet":
        findings = read_parquet(path)
    elif fmt == "gzip":
        # Peek past the gzip wrapper to see what is actually inside.
        with gzip.open(path, "rb") as fh:
            inner = fh.read(1)
        findings = read_jsonl(path, True) if inner in (b"{", b"[") else read_delimited(path, True)
    elif fmt == "jsonl":
        findings = read_jsonl(path, False)
    else:
        findings = read_delimited(path, False)

    findings.matched = match_columns(findings.columns)
    return report(findings)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print(__doc__)
        print("usage: python tools/inspect_data.py <file> [file ...]", file=sys.stderr)
        return 2
    worst = 0
    for raw in args:
        worst = max(worst, inspect(Path(raw).expanduser()))
    return worst


if __name__ == "__main__":
    sys.exit(main())
