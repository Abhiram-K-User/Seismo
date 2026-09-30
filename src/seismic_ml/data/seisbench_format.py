"""Reading (and subsetting) waveform files in the SeisBench HDF5 layout.

Layout (as used by the PNW dataset)::

    /data_format/component_order   e.g. b"ENZ"
    /data/bucket<k>                float array [n_traces, n_channels, n_samples]

A metadata row points to its waveform with ``trace_name``, e.g.
``"bucket4$17,:3,:15001"`` = row 17 of ``/data/bucket4``, first 3 channels,
first 15001 samples.

Local *subset* files written by :func:`extract_subset` keep exactly this
layout (same dataset names and shapes, so the original ``trace_name`` values
stay valid). Only the selected rows are stored: datasets are chunked one trace
per chunk and unwritten chunks occupy no disk space; they read back as NaN.
The list of stored traces is kept in ``/subset/trace_names``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from seismic_ml.data.acquisition import HTTPRangeFile, fetch_range

_TRACE_NAME = re.compile(r"^(?P<bucket>[^$]+)\$(?P<row>\d+),:(?P<channels>\d+),:(?P<samples>\d+)$")


@dataclass(frozen=True, slots=True)
class TraceLocation:
    """Where one trace lives inside a SeisBench HDF5 file."""

    bucket: str
    row: int
    n_channels: int
    n_samples: int


def parse_trace_name(trace_name: str) -> TraceLocation:
    """Parse ``"bucket4$17,:3,:15001"`` into a :class:`TraceLocation`.

    Raises:
        ValueError: If the name does not follow the SeisBench bucket convention.
    """
    match = _TRACE_NAME.match(trace_name)
    if match is None:
        raise ValueError(f"Unrecognised SeisBench trace_name: {trace_name!r}")
    return TraceLocation(
        bucket=match["bucket"],
        row=int(match["row"]),
        n_channels=int(match["channels"]),
        n_samples=int(match["samples"]),
    )


def read_component_order(h5: h5py.File) -> tuple[str, ...]:
    """Channel order of the waveform arrays, e.g. ``("E", "N", "Z")``."""
    raw = h5["data_format"]["component_order"][()]
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    return tuple(text)


class SeisBenchWaveformStore:
    """Read traces by ``trace_name`` from a local SeisBench-layout HDF5 file.

    Works for full files and for subset files from :func:`extract_subset`.
    Use as a context manager or call :meth:`close`.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        self.component_order = read_component_order(self._h5)
        if "subset" in self._h5:
            names = self._h5["subset"]["trace_names"].asstr()[()]
            self._available: set[str] | None = set(names)
        else:
            self._available = None  # full file: everything is available

    @property
    def is_subset(self) -> bool:
        return self._available is not None

    def __contains__(self, trace_name: str) -> bool:
        if self._available is not None:
            return trace_name in self._available
        loc = parse_trace_name(trace_name)
        return loc.bucket in self._h5["data"] and loc.row < self._h5["data"][loc.bucket].shape[0]

    def available_trace_names(self) -> set[str] | None:
        """Stored traces for a subset file; ``None`` for a full file."""
        return None if self._available is None else set(self._available)

    def get(self, trace_name: str) -> np.ndarray:
        """Return the ``[channels, samples]`` array of one trace.

        Raises:
            KeyError: If the trace is not stored in this file.
        """
        if trace_name not in self:
            raise KeyError(f"{trace_name!r} is not stored in {self.path.name}")
        loc = parse_trace_name(trace_name)
        return self._h5["data"][loc.bucket][loc.row, : loc.n_channels, : loc.n_samples]

    def close(self) -> None:
        self._h5.close()

    def __enter__(self) -> SeisBenchWaveformStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class _RemoteBucket:
    offset: int
    shape: tuple[int, ...]
    dtype: np.dtype


def _remote_layout(url: str) -> tuple[dict[str, _RemoteBucket], tuple[str, ...], HTTPRangeFile]:
    """Read bucket byte offsets, shapes and dtypes from a remote HDF5 header."""
    remote = HTTPRangeFile(url, block_size=1 << 18)
    with h5py.File(remote, "r") as h5:
        order = read_component_order(h5)
        buckets = {}
        for name, dset in h5["data"].items():
            if dset.chunks is not None or dset.compression is not None:
                raise ValueError(f"/data/{name} is chunked/compressed; byte-offset reads unsupported.")
            buckets[name] = _RemoteBucket(dset.id.get_offset(), dset.shape, dset.dtype)
    return buckets, order, remote


def _row_written(dset: h5py.Dataset, row: int) -> bool:
    """True if the one-trace chunk holding ``row`` has been written (allocated)."""
    return dset.id.get_chunk_info_by_coord((row,) + (0,) * (dset.ndim - 1)).byte_offset is not None


def extract_subset(
    url: str,
    trace_names: Sequence[str],
    dest: str | Path,
    *,
    workers: int = 6,
    timeout: float = 120.0,
    retries: int = 5,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Copy selected traces of a remote SeisBench HDF5 file into a local subset file.

    Trace arrays are copied byte-for-byte (same dtype, no processing). Work
    happens in ``<dest>.part``, which is renamed to ``dest`` only when every
    trace is present. The extraction is resumable: re-running with the same
    arguments skips traces already written to the ``.part`` file. Fetches that
    fail after ``retries`` attempts are retried once more at the end.

    Args:
        url: Remote HDF5 URL (server must support byte ranges; datasets must be
            contiguous and uncompressed, as in the PNW files).
        trace_names: SeisBench ``trace_name`` values to copy (duplicates ignored).
        dest: Local destination path.
        workers: Parallel HTTP connections.
        progress: Optional callback ``(done, total)``.

    Returns:
        Summary with the number of traces, bytes fetched and remote ETag.

    Raises:
        OSError: If some traces still cannot be fetched (the ``.part`` file is
            kept so a later run can resume).
    """
    names = sorted(set(trace_names), key=lambda n: (parse_trace_name(n).bucket,
                                                      parse_trace_name(n).row))
    buckets, order, remote = _remote_layout(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    jobs = []
    for name in names:
        loc = parse_trace_name(name)
        if loc.bucket not in buckets:
            raise KeyError(f"{name}: bucket not present in remote file")
        b = buckets[loc.bucket]
        if loc.row >= b.shape[0]:
            raise IndexError(f"{name}: row {loc.row} outside bucket of {b.shape[0]} rows")
        row_bytes = int(np.prod(b.shape[1:])) * b.dtype.itemsize
        jobs.append((name, loc, b.offset + loc.row * row_bytes, row_bytes))

    with h5py.File(part, "r+" if part.exists() else "w") as out:
        if "data_format" not in out:
            out.create_group("data_format").create_dataset(
                "component_order", data="".join(order).encode()
            )
        dsets = {}
        for bucket in sorted({loc.bucket for _, loc, _, _ in jobs}):
            b = buckets[bucket]
            if f"data/{bucket}" in out:
                dsets[bucket] = out[f"data/{bucket}"]
                continue
            fill = np.nan if np.issubdtype(b.dtype, np.floating) else 0
            dsets[bucket] = out.create_dataset(
                f"data/{bucket}", shape=b.shape, dtype=b.dtype,
                chunks=(1, *b.shape[1:]), fillvalue=fill,
            )

        pending = [job for job in jobs if not _row_written(dsets[job[1].bucket], job[1].row)]
        done = len(jobs) - len(pending)
        fetched = 0

        def run(batch: list, n_workers: int) -> list:
            nonlocal done, fetched
            failed = []
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for job in batch:
                    _, _, start, length = job
                    future = pool.submit(fetch_range, url, start, length,
                                         timeout=timeout, retries=retries)
                    futures[future] = job
                for future in as_completed(futures):
                    job = futures[future]
                    _, loc, _, length = job
                    try:
                        data = future.result()
                    except OSError:
                        failed.append(job)
                        continue
                    b = buckets[loc.bucket]
                    dsets[loc.bucket][loc.row] = np.frombuffer(data, dtype=b.dtype).reshape(b.shape[1:])
                    done += 1
                    fetched += length
                    if progress is not None:
                        progress(done, len(jobs))
            return failed

        failed = run(pending, workers)
        if failed:
            failed = run(failed, 1)
        if failed:
            out.flush()
            raise OSError(f"{len(failed)} traces could not be fetched; re-run to resume "
                          f"(first: {failed[0][0]}).")

        if "subset" in out:
            del out["subset"]
        subset = out.create_group("subset")
        subset.create_dataset("trace_names", data=np.array(names, dtype=h5py.string_dtype()))
        subset.attrs["source_url"] = url
        subset.attrs["source_etag"] = remote.info.etag or ""
        subset.attrs["source_last_modified"] = remote.info.last_modified or ""

    part.replace(dest)
    return {
        "n_traces": len(names),
        "bytes_fetched_this_run": fetched,
        "source_url": url,
        "source_etag": remote.info.etag,
        "source_last_modified": remote.info.last_modified,
        "source_size": remote.size,
        "component_order": "".join(order),
    }


def trace_names_for_events(metadata: Any, event_ids: Iterable[str]) -> list[str]:
    """All ``trace_name`` values belonging to ``event_ids`` (event-complete selection)."""
    wanted = set(event_ids)
    return metadata.loc[metadata["event_id"].isin(wanted), "trace_name"].tolist()
