"""Inspection measurements and manifest generation (toy fixture; measures only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismic_ml.data import pnw
from seismic_ml.data.inspection import (
    label_agreement,
    label_counts,
    metadata_quality,
    missing_components,
    possible_duplicate_events,
    station_count_table,
    waveform_metrics,
    waveform_quality_table,
)
from seismic_ml.data.manifest import build_manifest, collection_summary, write_json
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore


@pytest.fixture
def comcat(toy_raw_dir: Path) -> pd.DataFrame:
    return pnw.load_metadata(toy_raw_dir / "comcat_metadata.csv")


def test_station_count_table(comcat: pd.DataFrame) -> None:
    overall = station_count_table(comcat)
    assert overall.loc["all"].to_dict() == {"1": 1, "2": 2, "3": 1, "4+": 0}
    by_label = station_count_table(comcat, "source_type_pnsn_label")
    assert by_label.loc["eq"].to_dict() == {"1": 0, "2": 1, "3": 1, "4+": 0}
    assert by_label.loc["px", "1"] == 1


def test_label_counts_and_agreement(comcat: pd.DataFrame) -> None:
    counts = label_counts(comcat, "source_type_pnsn_label")
    assert counts.loc["eq"].to_dict() == {"events": 2, "traces": 5}
    agreement = label_agreement(comcat, "source_type", "source_type_pnsn_label")
    assert agreement.loc["explosion", "eq"] == 1                  # uw3 conflict is surfaced


def test_missing_components_from_snr(comcat: pd.DataFrame) -> None:
    missing = missing_components(comcat, ("E", "N", "Z"))
    assert set(missing) == {"", "EN"}
    assert (missing == "EN").sum() == (comcat["trace_missing_channel"] == 2).sum()


def test_metadata_quality(comcat: pd.DataFrame) -> None:
    report = metadata_quality(comcat, origin_offset_s=-0.5)
    assert report["rows"] == 8 and report["duplicate_trace_name"] == 0
    assert report["duplicate_event_station"] == 0
    assert report["sampling_rates_hz"] == [100.0] and report["samples_per_trace"] == [200]
    assert report["picks"]["s_before_p"] == 0
    assert report["trace_missing_channel"] == {"0": 5, "2": 3}
    assert report["start_minus_origin_s"]["n_deviating_gt_0p01s"] == 0


def test_possible_duplicate_events(comcat: pd.DataFrame) -> None:
    # all toy events share one origin time; only uw events close in space pair up
    dups = possible_duplicate_events(comcat, max_dt_s=1.0, max_dist_deg=1.0)
    assert len(dups) >= 1 and (dups["dt_s"] <= 1.0).all()


def test_waveform_metrics_detects_issues() -> None:
    data = np.zeros((3, 50))
    data[1] = 7.0                                   # flat, non-zero
    data[2] = np.sin(np.linspace(0, 6, 50))
    data[2, 10:15] = 1.0                            # 5 samples at the peak (clip-like)
    data[2, 30:40] = 0.0                            # zero run of 10
    m = waveform_metrics(data, ("E", "N", "Z"))
    assert m["E"]["all_zero"] and m["E"]["longest_zero_run"] == 50
    assert m["N"]["flat_nonzero"] and not m["N"]["all_zero"]
    assert m["Z"]["n_at_peak"] >= 5 and m["Z"]["longest_zero_run"] >= 10
    assert m["Z"]["n_nan"] == 0 and m["shape"] == (3, 50)


def test_waveform_quality_table(comcat: pd.DataFrame, toy_raw_dir: Path) -> None:
    with SeisBenchWaveformStore(toy_raw_dir / "comcat_waveforms.hdf5") as store:
        table = waveform_quality_table(comcat, store)
    assert len(table) == 8 * 3
    zero = table.groupby("trace_name")["all_zero"].sum()
    flags = comcat.set_index("trace_name")["trace_missing_channel"]
    assert (zero.to_numpy() == flags.loc[zero.index].to_numpy()).all()   # flag agrees with the stored zeros


def test_manifest_is_measured_portable_and_json(comcat: pd.DataFrame, toy_raw_dir: Path,
                                                tmp_path: Path, data_config: dict) -> None:
    summary = collection_summary(comcat, label_columns=["source_type", "source_type_pnsn_label"],
                                 primary_label_column="source_type_pnsn_label",
                                 component_order=("E", "N", "Z"))
    assert summary["num_events"] == 4 and summary["num_traces"] == 8
    assert summary["num_stations"] == 4 and summary["station_ids"][0] == "CC.CCC"
    assert summary["stations_per_event"]["max"] == 3
    provenance = json.loads((toy_raw_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    manifest = build_manifest(dataset_config=data_config["dataset"],
                              collections={"comcat": summary}, provenance=provenance,
                              raw_dir_relative="raw/pnw", notes=["toy"],
                              generated_utc="2026-01-01T00:00:00+00:00")
    assert manifest["dataset_name"] == "pnw" and manifest["num_events"] == 4
    assert manifest["sampling_rates"] == [100.0] and manifest["channels"] == ["ENZ"]
    assert all(f["path"].startswith("raw/pnw/") for f in manifest["files"])
    out = tmp_path / "processed" / "dataset_manifest.json"
    write_json(manifest, out)
    assert json.loads(out.read_text(encoding="utf-8"))["num_traces"] == 8
