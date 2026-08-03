"""What history is downloadable, rather than what we remember being there.

The live recorder accumulates from now onwards, which means waiting days
before there is anything to analyse. Binance also publishes daily archives,
and if BTCUSDT is in there the analysis can start today against years of
data instead.

Two things matter and neither should be taken on faith:

  which datasets exist   trades are certainly there; whether book depth,
                         liquidations and open interest are is the question,
                         and liquidations in particular are the one thing the
                         live feed cannot get at all
  what a file contains   the archive ships headerless CSV in places, so the
                         columns get printed rather than assumed

Listing comes from the bucket's XML index; one small file is then fetched and
its first rows shown, so the answer is the data itself.

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

# Where each product's daily files live.
ROOTS = {
    "spot": "data/spot/daily",
    "perp (USDⓈ-M)": "data/futures/um/daily",
}


async def listing(session: aiohttp.ClientSession, prefix: str) -> tuple[list[str], list[str]]:
    """Sub-directories and file keys directly under `prefix`."""
    async with session.get(
        BUCKET,
        params={"delimiter": "/", "prefix": prefix},
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        resp.raise_for_status()
        body = await resp.text()

    root = ElementTree.fromstring(body)
    dirs = [
        node.text.rstrip("/").rsplit("/", 1)[-1]
        for node in root.iter(f"{NS}Prefix")
        if node.text and node.text != prefix
    ]
    files = [
        node.text
        for node in root.iter(f"{NS}Key")
        if node.text and node.text.endswith(".zip")
    ]
    return sorted(set(dirs)), files


async def inventory(session: aiohttp.ClientSession, symbol: str) -> list[tuple[str, str, str]]:
    """(product, dataset, newest file) for every dataset carrying `symbol`."""
    found = []
    for product, root in ROOTS.items():
        datasets, _ = await listing(session, f"{root}/")
        print(f"\n=== {product}   {root}/")
        if not datasets:
            print("    (nothing listed)")
            continue
        print(f"    datasets: {', '.join(datasets)}")

        for dataset in datasets:
            try:
                _, files = await listing(session, f"{root}/{dataset}/{symbol}/")
            except Exception:  # noqa: BLE001 - a missing symbol is an answer
                continue
            if not files:
                continue
            newest = sorted(files)[-1]
            oldest = sorted(files)[0]
            print(
                f"      {dataset:<22} {len(files):>5,} days   "
                f"{oldest.rsplit('-', 3)[-3:][0]}… → {newest.rsplit('/', 1)[-1]}"
            )
            found.append((product, dataset, newest))
    return found


async def peek(session: aiohttp.ClientSession, key: str) -> None:
    """Download one archive and show what its rows actually look like."""
    url = f"{DOWNLOAD}/{key}"
    print(f"\n=== sample: {key.rsplit('/', 1)[-1]}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            resp.raise_for_status()
            blob = await resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"    FAILED: {type(exc).__name__}: {exc}")
        return

    print(f"    {len(blob) / 1e6:,.1f} MB compressed")
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = fh.read(600).decode("utf-8", "replace").splitlines()
        size = zf.getinfo(name).file_size
    print(f"    {name}  →  {size / 1e6:,.1f} MB uncompressed")
    for line in head[:4]:
        print(f"      {line[:150]}")


async def main() -> None:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT").upper()
    session = make_session()
    try:
        print(f"Looking for {symbol} in the Binance daily archive.")
        found = await inventory(session, symbol)
        if not found:
            print("\nNothing found — the archive layout may have moved.")
            return

        # Trades are the dataset every plan depends on, so sample that one.
        pick = next(
            (k for p, d, k in found if "perp" in p and d == "aggTrades"),
            found[0][2],
        )
        await peek(session, pick)

        print(
            "\nWhat to look for: liquidationSnapshot and bookDepth under perp.\n"
            "Liquidations are the one thing the live feed cannot get here, and\n"
            "book depth decides whether order-book features can be built from\n"
            "history or only from a live recording."
        )
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
