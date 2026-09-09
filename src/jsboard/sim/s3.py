"""Ship recordings to S3 while the capture is still running.

An hour of one symbol is ~84 MB of JSONL, so a day of several symbols is
tens of gigabytes: too much to leave on a laptop, and the interesting
question needs days rather than one hour. The screen and the replay
disagreed about USUSDT by a factor of ten and neither side could be settled,
because one looked at an hour and the other at five minutes. Continuous
recording is what closes that.

Three things this deliberately does not do:

  **Upload per event.** A print-by-print PUT would cost more in requests than
  in storage and would stall the receive loop on every message. Lines
  accumulate in a local part file, which is closed on a size or time
  boundary, gzipped, and sent as one object.

  **Upload on the event loop.** boto3 is synchronous and a 20 MB part takes
  seconds. Rotation hands the finished part to a worker thread, so recording
  never pauses for the network; `close` is what waits.

  **Touch credentials.** The client comes from boto3's own chain —
  environment, shared config, or an instance role. There is no key parameter
  here and there must never be one: a recording that carries a credential is
  a credential in a bucket.

Keys are laid out for later querying rather than for browsing:

    <prefix>/symbol=BTCUSDT/date=2026-09-09/hour=07/<start_ns>.jsonl.gz

Hive-style partitions mean Athena or a Glue crawler can read a day, an hour
or one symbol without listing everything, and the start timestamp in the
name keeps parts sortable inside a partition.
"""

from __future__ import annotations

import gzip
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

NS_PER_S = 1_000_000_000
DEFAULT_ROTATE_S = 900.0
DEFAULT_ROTATE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class S3Target:
    bucket: str
    prefix: str = "raw"

    def key_for(self, symbol: str, start_ns: int, seq: int = 0) -> str:
        """Where one part lands.

        The sequence number is not decoration. A size-triggered rotation can
        close two parts inside the same clock tick, and without it the second
        object overwrites the first — data loss that leaves no trace, since
        both uploads report success. Fixed widths keep parts sortable by name
        within a partition.
        """
        when = datetime.fromtimestamp(start_ns / NS_PER_S, tz=UTC)
        return (
            f"{self.prefix.strip('/')}/symbol={symbol.upper()}"
            f"/date={when:%Y-%m-%d}/hour={when:%H}/{start_ns}-{seq:05d}.jsonl.gz"
        )


def default_client():
    """boto3's own credential chain, and nothing else.

    Imported here rather than at module scope so the rest of the project
    keeps working without boto3 installed — S3 is opt-in.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "S3 への保存には boto3 が必要です: pip install -e '.[s3]'"
        ) from exc
    return boto3.client("s3")


@dataclass(slots=True)
class RotatingJsonlSink:
    """A file-like sink that rotates, compresses and uploads as it goes.

    Stands in for the plain file handle the capture writes to, so the
    recorder needs no knowledge of any of this.
    """

    path: Path
    """Template for the part names; the parts sit beside it."""

    target: S3Target
    symbol: str
    client: object = field(default=None)
    rotate_seconds: float = DEFAULT_ROTATE_S
    rotate_bytes: int = DEFAULT_ROTATE_BYTES
    keep_local: bool = False
    clock: object = time.time_ns

    _fh: object = field(default=None, init=False)
    _part: Path | None = field(default=None, init=False)
    _started_ns: int = field(default=0, init=False)
    _bytes: int = field(default=0, init=False)
    _pool: ThreadPoolExecutor = field(default=None, init=False)
    _pending: list[Future] = field(default_factory=list, init=False)
    _closing: bool = field(default=False, init=False)
    _seq: int = field(default=0, init=False)
    uploaded: list[str] = field(default_factory=list, init=False)
    failed: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = default_client()
        # Two workers: one uploading while the next part is already closing,
        # without letting a slow network build an unbounded backlog.
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3")
        self._open_part()

    # ------------------------------------------------------------- file-like

    def write(self, text: str) -> int:
        # Rotate before writing, not after: a part should cover the interval
        # it is named for. Checking afterwards puts the line that crossed the
        # boundary in the part that had already ended.
        if self._should_rotate():
            self.rotate()
        n = self._fh.write(text)
        self._bytes += len(text.encode("utf-8"))
        return n

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.rotate()
        self._pool.shutdown(wait=True)
        self._pending.clear()

    def __enter__(self) -> RotatingJsonlSink:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -------------------------------------------------------------- rotation

    def _should_rotate(self) -> bool:
        if self.rotate_bytes and self._bytes >= self.rotate_bytes:
            return True
        elapsed = (self.clock() - self._started_ns) / NS_PER_S
        return bool(self.rotate_seconds) and elapsed >= self.rotate_seconds

    def _open_part(self) -> None:
        self._started_ns = self.clock()
        self._bytes = 0
        self._seq += 1
        stem, suffix = self.path.stem, self.path.suffix or ".jsonl"
        self._part = self.path.parent / f"{stem}-{self._started_ns}-{self._seq:05d}{suffix}"
        self._part.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._part.open("w", encoding="utf-8")

    def rotate(self) -> None:
        """Close the current part and hand it to the uploader."""
        if self._fh is None:
            return
        self._fh.close()
        self._fh = None
        part, started = self._part, self._started_ns
        self._part = None
        if part is not None and part.exists() and part.stat().st_size > 0:
            key = self.target.key_for(self.symbol, started, self._seq)
            self._pending.append(self._pool.submit(self._upload, part, key))
        elif part is not None and part.exists():
            part.unlink()  # nothing was written to it
        if not self._closing:
            self._open_part()

    def _upload(self, part: Path, key: str) -> None:
        archive = part.with_suffix(part.suffix + ".gz")
        try:
            with part.open("rb") as raw, gzip.open(archive, "wb") as out:
                shutil.copyfileobj(raw, out)
            self.client.upload_file(str(archive), self.target.bucket, key)
            self.uploaded.append(key)
        except Exception as exc:  # noqa: BLE001 - a failed part must not stop recording
            self.failed.append(f"{key}: {exc}")
            # Keep the local copy: it is the only remaining evidence.
            return
        finally:
            archive.unlink(missing_ok=True)
        if not self.keep_local:
            part.unlink(missing_ok=True)

    # ---------------------------------------------------------------- status

    def summary(self) -> dict:
        return {
            "bucket": self.target.bucket,
            "uploaded": list(self.uploaded),
            "failed": list(self.failed),
        }
