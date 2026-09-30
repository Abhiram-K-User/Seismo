"""Tiny synthetic dataset in the PNW / SeisBench layout — FOR UNIT TESTS ONLY.

The arrays are deterministic toy signals that only exercise code paths
(parsing, grouping, schema conversion, subsetting). They are not seismic data
and must never be used for, or reported as, results.

Layout written under ``<data_root>/raw/pnw/``::

    comcat_metadata.csv, comcat_waveforms.hdf5
    exotic_metadata.csv, exotic_waveforms.hdf5
    PROVENANCE.json
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

T = 200            # default samples per trace
RATE = 100.0       # Hz
P_SAMPLE, S_SAMPLE = 60, 90

# (event, pnsn label, comcat source_type, [(net, sta, loc, band, missing)], lat, lon)
COMCAT_EVENTS = [
    ("uw1", "eq", "earthquake", [("UW", "AAA", "--", "HH", 0), ("UW", "BBB", "00", "BH", 0),
                                 ("CC", "CCC", "--", "EH", 2)], 47.0, -122.0),
    ("uw2", "px", "explosion", [("UW", "AAA", "--", "HH", 0)], 47.5, -122.5),
    ("uw3", "eq", "explosion", [("UW", "BBB", "00", "BH", 0), ("UW", "DDD", "--", "EH", 2)],
     46.0, -121.0),                                   # label conflict ComCat vs PNSN
    ("uw4", "lf", "earthquake", [("UW", "AAA", "--", "HH", 0), ("CC", "CCC", "--", "EH", 2)],
     46.2, -121.8),
]
EXOTIC_EVENTS = [
    ("pnsn1", "surface event", [("CC", "CCC", "--", "EH", 2)]),
    ("pnsn2", "sonic boom", [("UW", "AAA", "--", "HH", 0), ("UW", "BBB", "00", "BH", 0)]),
]
STATION_COORDS = {"AAA": (47.1, -122.1, 100.0), "BBB": (46.9, -121.9, 200.0),
                  "CCC": (46.5, -121.5, 1500.0), "DDD": (45.9, -120.9, 300.0)}


def toy_trace(index: int, missing: int, n_samples: int = T) -> np.ndarray:
    """Deterministic [3, n_samples] toy array (ENZ); ``missing=2`` zero-fills E and N."""
    t = np.arange(n_samples) / RATE
    data = np.stack([np.sin(2 * np.pi * (1 + index + k) * t) * (k + 1) for k in range(3)])
    if missing == 2:
        data[:2] = 0.0
    return data.astype(np.float64)


def _row(event, net, sta, loc, band, missing, bucket_row, start, extra, n_samples=T):
    lat, lon, elev = STATION_COORDS[sta]
    snr = "nan|nan|5.0" if missing == 2 else "4.0|3.0|5.0"
    return {"event_id": event, "station_network_code": net, "station_channel_code": band,
            "station_code": sta, "station_location_code": loc, "station_latitude_deg": lat,
            "station_longitude_deg": lon, "station_elevation_m": elev,
            "trace_name": f"bucket1${bucket_row},:3,:{n_samples}", "trace_sampling_rate_hz": 100,
            "trace_start_time": start, "trace_snr_db": snr, **extra}


def _write_h5(path: Path, arrays: list[np.ndarray]) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_group("data_format").create_dataset("component_order", data=b"ENZ")
        h5.create_dataset("data/bucket1", data=np.stack(arrays))


def build(data_root: Path, *, n_samples: int = T, p_sample: int = P_SAMPLE,
          s_sample: int = S_SAMPLE) -> Path:
    """Write the toy dataset under ``data_root`` and return the dataset raw dir.

    ``n_samples`` / ``p_sample`` / ``s_sample`` allow longer traces for
    end-to-end preprocessing tests (defaults keep the original tiny traces).
    """
    raw = data_root / "raw" / "pnw"
    raw.mkdir(parents=True, exist_ok=True)

    rows, arrays = [], []
    for event, pnsn, src, stations, lat, lon in COMCAT_EVENTS:
        for net, sta, loc, band, missing in stations:
            arrays.append(toy_trace(len(arrays), missing, n_samples))
            rows.append(_row(event, net, sta, loc, band, missing, len(arrays) - 1,
                             "2020-01-01T00:00:00.000000Z", n_samples=n_samples, extra={
                                 "source_origin_time": "2020-01-01T00:00:00.500000Z",
                                 "source_latitude_deg": lat, "source_longitude_deg": lon,
                                 "source_type": src, "source_depth_km": 5.0,
                                 "preferred_source_magnitude": 1.5,
                                 "trace_P_arrival_sample": p_sample,
                                 "trace_S_arrival_sample": s_sample,
                                 "source_type_pnsn_label": pnsn,
                                 "trace_missing_channel": missing, "trace_has_offset": 0}))
    pd.DataFrame(rows).to_csv(raw / "comcat_metadata.csv", index=False)
    _write_h5(raw / "comcat_waveforms.hdf5", arrays)

    rows, arrays = [], []
    for event, src, stations in EXOTIC_EVENTS:
        for net, sta, loc, band, missing in stations:
            arrays.append(toy_trace(len(arrays), missing, n_samples))
            rows.append(_row(event, net, sta, loc, band, missing, len(arrays) - 1,
                             "2021-06-01T12:00:00.000000Z", n_samples=n_samples, extra={
                                 "source_type": src, "trace_P_arrival_sample": float(p_sample + 10),
                                 "trace_S_arrival_sample": np.nan}))
    pd.DataFrame(rows).to_csv(raw / "exotic_metadata.csv", index=False)
    _write_h5(raw / "exotic_waveforms.hdf5", arrays)

    files = {}
    for name in ("comcat_metadata.csv", "exotic_metadata.csv"):
        files[name] = {"path": name, "url": f"https://example.invalid/{name}",
                       "size": (raw / name).stat().st_size, "sha256": "0" * 64, "etag": None,
                       "last_modified": None, "acquired_utc": "2026-01-01T00:00:00+00:00"}
    (raw / "PROVENANCE.json").write_text(json.dumps({"files": files, "subsets": {}}),
                                         encoding="utf-8")
    return raw
