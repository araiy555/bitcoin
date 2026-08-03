"""What history is downloadable, rather than what we remember being there.

The live recorder only accumulates from now on. Binance also publishes daily
archives, and what is in them decides whether analysis can start today or has
to wait for a recording to fill up.

Two questions decide the plan, and neither should be taken on faith:

  how far back and how recent   a dataset that stops in 2024 cannot be used
                                to study today's market
  what a row contains           the archive ships headerless CSV, so the
                                columns get printed rather than assumed

The bucket lists at most 1000 keys per request, and each day contributes both
a .zip and a .zip.CHECKSUM — so an unpaginated listing silently stops at 500
days and reports a "newest" file from years ago. This walks every page.

    python tools/probe_archive.py              # inventory for BTCUSDT
    python tools/probe_archive.py ETHUSDT
"""

from __future__ import annotations

import asyncio
import io
import sys
import zipfile
from xml.etree import ElementTree

import aiohttp

from jsboard.net import make_session

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DOWNLOAD = "https://data.binance.vision"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

ROOTS = {
    "spot": "data/spot/daily",
    "perp": "data/futures/um/daily",
}

# Small enough to fetch just to read the header. aggTrades and bookTicker run
# to hundreds of megabytes a day, and their shape can wait until they are
# actually being ingested.
SAMPLE = ("bookDepth", "metrics")


async def page(session: aiohttp.ClientSession, prefix: str, marker: str | None):
    params = {"delimiter": "/", "prefix": prefix}
    if marker:
        params["marker"] = marker
    async with session.get(
        BUCKET, params=params, timeout=aiohttp.ClientTimeout(total=30)
    ) as resp:
        resp.raise_for_status()
        return ElementTree.fromstring(await resp.text())


async def list_dirs(session: aiohttp.ClientSession, prefix: str) -> list[str]:
    root = await page(session, prefix, None)
    return sorted(
        {
            node.text.rstrip("/").rsplit("/", 1)[-1]
            for node in root.iter(f"{NS}Prefix")
            if node.text and node.text != prefix
        }
    )


async def list_files(session: aiohttp.ClientSession, prefix: str) -> list[tuple[str, int]]:
    """Every .zip under `prefix`, following pagination to the end."""
    out: list[tuple[str, int]] = []
    marker: str | None = None
    while True:
        root = await page(session, prefix, marker)
        keys = list(root.iter(f"{NS}Contents"))
        if not keys:
            break
        for node in keys:
            key = node.findtext(f"{NS}Key") or ""
            if key.endswith(".zip"):
                out.append((key, int(node.findtext(f"{NS}Size") or 0)))
        truncated = (root.findtext(f"{NS}IsTruncated") or "false") == "true"
        if not truncated:
            break
        marker = root.findtext(f"{NS}NextMarker") or (
            keys[-1].findtext(f"{NS}Key") or ""
        )
        if not marker:
            break
    return sorted(out)


def day_of(key: str) -> str:
    return key.rsplit("-", 3)[-3:] and "-".join(key.rsplit(".zip", 1)[0].rsplit("-", 3)[-3:])


async def main() -> None:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT").upper()
    session = make_session()
    samples: list[str] = []
    try:
        print(f"{symbol} in the Binance daily archive (paginated).\n")
        for product, root in ROOTS.items():
            datasets = await list_dirs(session, f"{root}/")
            print(f"=== {product}   {root}/")
            print(f"    published: {', '.join(datasets)}")
            if "liquidationSnapshot" not in datasets:
                print("    NOTE: no liquidationSnapshot here")

            for dataset in datasets:
                try:
                    files = await list_files(session, f"{root}/{dataset}/{symbol}/")
                except Exception:  # noqa: BLE001 - a missing symbol is an answer
                    continue
                if not files:
                    continue
                total_gb = sum(size for _, size in files) / 1e9
                recent = files[-1][1] / 1e6
                print(
                    f"      {dataset:<20} {len(files):>5,} days  "
                    f"{day_of(files[0][0])} → {day_of(files[-1][0])}  "
                    f"{recent:>7,.1f} MB/day   {total_gb:>6,.1f} GB total"
                )
                if dataset in SAMPLE:
                    samples.append(files[-1][0])
            print()

        for key in samples:
            print(f"=== columns: {key.rsplit('/', 1)[-1]}")
            try:
                async with session.get(
                    f"{DOWNLOAD}/{key}", timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    resp.raise_for_status()
                    blob = await resp.read()
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    name = zf.namelist()[0]
                    with zf.open(name) as fh:
                        head = fh.read(500).decode("utf-8", "replace").splitlines()
                for line in head[:3]:
                    print(f"      {line[:140]}")
            except Exception as exc:  # noqa: BLE001
                print(f"      FAILED: {type(exc).__name__}: {exc}")
            print()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
