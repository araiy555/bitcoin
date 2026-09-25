"""Separate recordings that were accidentally sent to the same folder.

    python scripts/pickrun.py /tmp/rec-XRP-spot [--rotate-minutes 5]

Two captures of one symbol started together land their parts side by side,
and a replay of the folder would feed every event twice. Each capture numbers
its parts 1, 2, 3 ... at a fixed rotation interval, so a part belongs to the
run whose first part, projected forward by that interval, lands nearest its
own start time. The longest run stays; the others move to <folder>-other-N.
"""

import argparse
import re
import shutil
from pathlib import Path

PART = re.compile(r"(\d{16,20})-(\d{5})\.jsonl(?:\.gz)?$")


def runs_of(folder: Path, rotate_ns: int) -> list[list[Path]]:
    parts = []
    for path in folder.rglob("*.jsonl*"):
        match = PART.search(path.name)
        if match and path.is_file():
            parts.append((int(match[1]), int(match[2]), path))
    starts = sorted(start for start, seq, _ in parts if seq == 1)
    runs: dict[int, list[Path]] = {start: [] for start in starts}
    for start, seq, path in parts:
        origin = min(starts, key=lambda s: abs(start - (s + (seq - 1) * rotate_ns)))
        runs[origin].append(path)
    return sorted(runs.values(), key=len, reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--rotate-minutes", type=float, default=5.0)
    opts = ap.parse_args()
    folder = Path(opts.folder)
    runs = runs_of(folder, int(opts.rotate_minutes * 60 * 1e9))
    if not runs:
        print("分割ファイルがありません")
        return
    print(f"録画の本数: {len(runs)}  (各 {', '.join(str(len(r)) for r in runs)} 個)")
    for n, run in enumerate(runs[1:], start=1):
        dest = folder.parent / f"{folder.name}-other-{n}"
        for path in run:
            target = dest / path.relative_to(folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), target)
        print(f"  {len(run)} 個を {dest} へ移動")
    print(f"残した録画: {len(runs[0])} 個")


if __name__ == "__main__":
    main()
