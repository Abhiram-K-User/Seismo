"""Downloads, HTTP byte-range reading and remote subset extraction (offline, local server)."""

from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
import pytest

from seismic_ml.data.acquisition import HTTPRangeFile, download_file, fetch_range, head, sha256_file
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore, extract_subset


def test_head_reports_size_etag_and_ranges(range_server: str, toy_raw_dir: Path) -> None:
    info = head(range_server + "comcat_metadata.csv")
    assert info.size == (toy_raw_dir / "comcat_metadata.csv").stat().st_size
    assert info.etag == "toy-etag" and info.accept_ranges


def test_download_file_is_atomic_and_checksummed(range_server: str, toy_raw_dir: Path,
                                                 tmp_path: Path) -> None:
    dest = tmp_path / "dl" / "comcat_metadata.csv"
    record = download_file(range_server + "comcat_metadata.csv", dest, relative_to=tmp_path)
    assert dest.read_bytes() == (toy_raw_dir / "comcat_metadata.csv").read_bytes()
    assert record.path == "dl/comcat_metadata.csv"            # relative, portable
    assert record.sha256 == sha256_file(dest)
    assert not dest.with_name(dest.name + ".part").exists()
    again = download_file(range_server + "comcat_metadata.csv", dest, relative_to=tmp_path)
    assert again.sha256 == record.sha256                       # idempotent re-run


def test_fetch_range_returns_exact_bytes(range_server: str, toy_raw_dir: Path) -> None:
    data = (toy_raw_dir / "comcat_waveforms.hdf5").read_bytes()
    assert fetch_range(range_server + "comcat_waveforms.hdf5", 10, 50) == data[10:60]


def test_http_range_file_behaves_like_a_file(range_server: str, toy_raw_dir: Path) -> None:
    local = (toy_raw_dir / "comcat_waveforms.hdf5").read_bytes()
    remote = HTTPRangeFile(range_server + "comcat_waveforms.hdf5", block_size=64)
    assert remote.seekable() and remote.readable()
    remote.seek(100)
    assert remote.read(300) == local[100:400]                  # spans several blocks
    remote.seek(-10, io.SEEK_END)
    assert remote.read() == local[-10:]
    assert remote.read(5) == b""                               # at EOF
    with pytest.raises(ValueError):
        remote.seek(-1)


def test_h5py_reads_remote_file(range_server: str, toy_raw_dir: Path) -> None:
    with h5py.File(HTTPRangeFile(range_server + "comcat_waveforms.hdf5"), "r") as remote, \
            h5py.File(toy_raw_dir / "comcat_waveforms.hdf5", "r") as local:
        np.testing.assert_array_equal(remote["data/bucket1"][3], local["data/bucket1"][3])


def test_extract_subset_copies_selected_traces_byte_exact(range_server: str, toy_raw_dir: Path,
                                                          tmp_path: Path) -> None:
    names = ["bucket1$1,:3,:200", "bucket1$4,:3,:200"]
    dest = tmp_path / "subset.hdf5"
    summary = extract_subset(range_server + "comcat_waveforms.hdf5", names, dest, workers=2)
    assert summary["n_traces"] == 2 and summary["component_order"] == "ENZ"
    with SeisBenchWaveformStore(dest) as sub, \
            SeisBenchWaveformStore(toy_raw_dir / "comcat_waveforms.hdf5") as full:
        assert sub.is_subset and sub.available_trace_names() == set(names)
        assert sub.component_order == ("E", "N", "Z")
        for name in names:
            np.testing.assert_array_equal(sub.get(name), full.get(name))
        assert "bucket1$0,:3,:200" not in sub
        with pytest.raises(KeyError):
            sub.get("bucket1$0,:3,:200")
        assert np.isnan(sub._h5["data/bucket1"][0]).all()      # unselected rows are not stored


def test_extract_subset_resumes_from_part_file(range_server: str, tmp_path: Path) -> None:
    url = range_server + "comcat_waveforms.hdf5"
    dest = tmp_path / "subset.hdf5"
    names = ["bucket1$0,:3,:200", "bucket1$2,:3,:200", "bucket1$5,:3,:200"]
    extract_subset(url, names[:1], dest, workers=1)
    dest.rename(dest.with_name(dest.name + ".part"))            # pretend it was interrupted
    summary = extract_subset(url, names, dest, workers=1)
    row_bytes = 3 * 200 * 8
    assert summary["bytes_fetched_this_run"] == 2 * row_bytes   # only the missing traces
    with SeisBenchWaveformStore(dest) as sub:
        assert sub.available_trace_names() == set(names)


def test_extract_subset_rejects_unknown_rows(range_server: str, tmp_path: Path) -> None:
    with pytest.raises(IndexError):
        extract_subset(range_server + "comcat_waveforms.hdf5", ["bucket1$999,:3,:200"],
                       tmp_path / "x.hdf5")
    with pytest.raises(KeyError):
        extract_subset(range_server + "comcat_waveforms.hdf5", ["bucket9$0,:3,:200"],
                       tmp_path / "y.hdf5")


def test_extract_subset_recovers_from_corrupted_part_file(range_server: str, tmp_path: Path) -> None:
    url = range_server + "comcat_waveforms.hdf5"
    dest = tmp_path / "corrupt_subset.hdf5"
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(b"corrupted header data not hdf5")
    names = ["bucket1$0,:3,:200", "bucket1$2,:3,:200"]
    summary = extract_subset(url, names, dest, workers=1)
    assert summary["n_traces"] == 2
    with SeisBenchWaveformStore(dest) as sub:
        assert sub.available_trace_names() == set(names)

