"""Reproducible data acquisition.

Two mechanisms, both standard-library HTTP only:

* :func:`download_file` — whole-file download with atomic write, size check
  and SHA-256 checksum (used for small metadata files).
* :class:`HTTPRangeFile` — a read-only, seekable file object backed by HTTP
  ``Range`` requests with a block cache. Passing it to ``h5py.File`` lets us
  read selected traces from a remote HDF5 file without downloading the whole
  file (the PNW ComCat waveform file is ~67 GB).

Destination paths are always supplied by the caller (resolved through
:mod:`seismic_ml.paths`); nothing here knows about machine-specific locations.
"""

from __future__ import annotations

import hashlib
import io
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_USER_AGENT = "seismic-ml-research/0.1 (+https://github.com/niyiyu/PNW-ML dataset user)"


@dataclass(frozen=True, slots=True)
class RemoteFileInfo:
    """Server-side facts about a remote file (from a HEAD request)."""

    url: str
    size: int
    etag: str | None
    last_modified: str | None
    accept_ranges: bool


@dataclass(frozen=True, slots=True)
class DownloadRecord:
    """Provenance of one downloaded file. ``path`` is relative to the dataset raw dir."""

    path: str
    url: str
    size: int
    sha256: str
    etag: str | None
    last_modified: str | None
    acquired_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _request(url: str, method: str = "GET", headers: dict[str, str] | None = None,
             timeout: float = 60.0):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": _USER_AGENT,
                                                              **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310 (https URLs from config)


def head(url: str, timeout: float = 60.0) -> RemoteFileInfo:
    """Return size, ETag and range support of a remote file."""
    with _request(url, method="HEAD", timeout=timeout) as resp:
        headers = resp.headers
        return RemoteFileInfo(
            url=url,
            size=int(headers["Content-Length"]),
            etag=(headers.get("ETag") or "").strip('"') or None,
            last_modified=headers.get("Last-Modified"),
            accept_ranges=headers.get("Accept-Ranges", "").lower() == "bytes",
        )


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def download_file(
    url: str,
    dest: str | Path,
    *,
    relative_to: str | Path,
    overwrite: bool = False,
    timeout: float = 60.0,
    chunk_size: int = 1 << 20,
) -> DownloadRecord:
    """Download ``url`` to ``dest`` atomically and return its provenance record.

    An existing file with the server's size is kept (idempotent re-runs) unless
    ``overwrite``. The file is written to ``<dest>.part`` and renamed only after
    the byte count matches ``Content-Length``.

    Args:
        relative_to: Directory the recorded ``path`` is expressed relative to
            (keeps provenance files free of machine-specific absolute paths).
    """
    dest = Path(dest)
    info = head(url, timeout=timeout)
    if not (dest.exists() and dest.stat().st_size == info.size and not overwrite):
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        written = 0
        with _request(url, timeout=timeout) as resp, part.open("wb") as fh:
            for chunk in iter(lambda: resp.read(chunk_size), b""):
                fh.write(chunk)
                written += len(chunk)
        if written != info.size:
            part.unlink(missing_ok=True)
            raise OSError(f"Incomplete download of {url}: {written} of {info.size} bytes.")
        os.replace(part, dest)
    return DownloadRecord(
        path=dest.resolve().relative_to(Path(relative_to).resolve()).as_posix(),
        url=url,
        size=dest.stat().st_size,
        sha256=sha256_file(dest),
        etag=info.etag,
        last_modified=info.last_modified,
        acquired_utc=utc_now_iso(),
    )


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable view of a remote file via HTTP ``Range`` requests.

    Reads are served from fixed-size, aligned blocks kept in an LRU cache, so
    the many small reads HDF5 makes for its internal metadata cost few requests.

    Args:
        url: Remote file URL; the server must support byte ranges.
        block_size: Bytes fetched per request.
        max_cached_blocks: LRU cache capacity (memory ≈ block_size × this).
        retries: Attempts per block before giving up (exponential back-off).
    """

    def __init__(
        self,
        url: str,
        *,
        block_size: int = 1 << 20,
        max_cached_blocks: int = 64,
        timeout: float = 60.0,
        retries: int = 4,
    ) -> None:
        super().__init__()
        info = head(url, timeout=timeout)
        if not info.accept_ranges:
            raise OSError(f"Server does not advertise byte-range support for {url}.")
        self.url = url
        self.info = info
        self.size = info.size
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.timeout = timeout
        self.retries = retries
        self._pos = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self.requests = 0
        self.bytes_fetched = 0

    # --- io.RawIOBase interface -------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if new < 0:
            raise ValueError("negative seek position")
        self._pos = new
        return self._pos

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        wanted = min(len(view), max(0, self.size - self._pos))
        done = 0
        while done < wanted:
            block_index, offset = divmod(self._pos + done, self.block_size)
            block = self._block(block_index)
            n = min(len(block) - offset, wanted - done)
            view[done:done + n] = block[offset:offset + n]
            done += n
        self._pos += done
        return done

    # --- internals ----------------------------------------------------------------

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        start = index * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = self._fetch(start, end)
        self._cache[index] = data
        if len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return data

    def _fetch(self, start: int, end: int) -> bytes:
        data = fetch_range(self.url, start, end - start + 1, timeout=self.timeout,
                           retries=self.retries)
        self.requests += 1
        self.bytes_fetched += len(data)
        return data


def fetch_range(url: str, start: int, length: int, *, timeout: float = 60.0,
                retries: int = 4) -> bytes:
    """Fetch ``length`` bytes starting at ``start`` with retries and back-off.

    Raises:
        OSError: If the server does not answer with HTTP 206 and exactly
            ``length`` bytes after ``retries`` attempts.
    """
    end = start + length - 1
    for attempt in range(retries):
        try:
            with _request(url, headers={"Range": f"bytes={start}-{end}"}, timeout=timeout) as resp:
                if resp.status != 206:
                    raise OSError(f"Expected HTTP 206 for a range request, got {resp.status}.")
                data = resp.read()
            if len(data) != length:
                raise OSError(f"Short range read: {len(data)} of {length} bytes.")
            return data
        except (OSError, urllib.error.URLError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")
