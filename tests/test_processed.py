"""End-to-end: toy raw data -> preprocessing -> processed store -> events -> EventBatch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

import pnw_fixture
from seismic_ml.data import pnw
from seismic_ml.data.loader import collate_events
from seismic_ml.data.processed import ProcessedEventDataset, write_processed_collection
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore
from seismic_ml.data.splits import load_split_table

# Window sized for the 45 s toy traces (P at 15 s); the real config uses 60 s windows.
TOY_PREPROCESSING = {
    "detrend": "linear", "taper_s": 2.0,
    "bandpass": {"freqmin": 1.0, "freqmax": 16.0, "corners": 4, "zerophase": True},
    "window": {"pre_p_s": 5.0, "post_p_s": 15.0, "edge_margin_s": 2.0, "resample_margin_s": 1.0},
    "target_sample_rate": 50, "normalization": {"method": "std", "scope": "station"},
    "p_wave_alignment": True, "quality_checks": {"require_p_pick": True, "clip_min_samples": 10},
}


@pytest.fixture
def long_toy_root(tmp_path: Path) -> Path:
    root = tmp_path / "data_root"
    raw = pnw_fixture.build(root, n_samples=4500, p_sample=1500, s_sample=1800)
    meta = pd.read_csv(raw / "comcat_metadata.csv")
    meta.loc[meta.index[1], ["trace_P_arrival_sample", "trace_S_arrival_sample"]] = [4200, 4300]
    #                                                            ^ P too late -> window_out_of_range
    meta.to_csv(raw / "comcat_metadata.csv", index=False)
    ex = pd.read_csv(raw / "exotic_metadata.csv")
    ex.loc[ex.index[0], "trace_P_arrival_sample"] = np.nan        # pnsn1 has no P -> dropped
    ex.to_csv(raw / "exotic_metadata.csv", index=False)
    return root


def test_write_processed_collection(long_toy_root: Path, tmp_path: Path) -> None:
    raw = long_toy_root / "raw" / "pnw"
    meta = pnw.load_metadata(raw / "comcat_metadata.csv")
    with SeisBenchWaveformStore(raw / "comcat_waveforms.hdf5") as store:
        counts = write_processed_collection(tmp_path / "out", meta, store,
                                            label_column="source_type_pnsn_label",
                                            config=TOY_PREPROCESSING)
    assert counts["traces_input"] == 8 and counts["traces_kept"] == 7
    assert counts["traces_dropped"] == {"window_out_of_range": 1}
    assert counts["samples_per_trace"] == 1000 and counts["events_kept"] == 4
    traces = pd.read_csv(tmp_path / "out" / "traces.csv")
    with h5py.File(tmp_path / "out" / "waveforms.hdf5", "r") as h5:
        w, m = h5["waveforms"][()], h5["channel_mask"][()]
        assert h5.attrs["sampling_rate"] == 50.0 and h5.attrs["channels"] == "ENZ"
    assert w.shape == (7, 3, 1000) and w.dtype == np.float32 and not np.isnan(w).any()
    assert (w[~m] == 0).all()                                      # missing components stay zero
    assert np.allclose([w[i][m[i]].std() for i in range(7)], 1.0, atol=1e-4)
    assert (traces["p_index"] == 250).all()                        # P exactly at pre_p_s * rate
    assert set(traces["live_components"]) == {"ENZ", "Z"}
    dropped = pd.read_csv(tmp_path / "out" / "dropped_traces.csv")
    assert dropped["reason"].tolist() == ["window_out_of_range"]


def _toy_config_dir(repo_root: Path, tmp_path: Path) -> Path:
    cfg_dir = tmp_path / "configs"
    shutil.copytree(repo_root / "configs", cfg_dir)
    data = yaml.safe_load((cfg_dir / "data.yaml").read_text(encoding="utf-8"))
    data["preprocessing"] = TOY_PREPROCESSING
    data["split"].update({"train_end": "2021-01-01", "val_end": "2022-01-01"})
    (cfg_dir / "data.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return cfg_dir


def test_prepare_data_script_end_to_end(repo_root: Path, long_toy_root: Path, tmp_path: Path) -> None:
    cfg = _toy_config_dir(repo_root, tmp_path) / "config.yaml"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(repo_root / "src"),
                                                                    os.environ.get("PYTHONPATH")])))
    result = subprocess.run([sys.executable, "scripts/prepare_data.py", "--config", str(cfg),
                             "--data-root", str(long_toy_root)], cwd=repo_root, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    processed = long_toy_root / "processed"
    split = load_split_table(processed / "splits" / "s1_temporal" / "events.csv")
    assert split.set_index("event_id").loc[["uw1", "uw2", "uw3", "uw4", "pnsn1", "pnsn2"],
                                           "split"].tolist() == ["train", "train", "train",
                                                                 "ood", "ood", "ood"]
    info = json.loads((processed / "pnw" / "comcat" / "preprocessing.json").read_text(encoding="utf-8"))
    assert info["inputs"]["waveforms"] == "raw/pnw/comcat_waveforms.hdf5"   # relative, portable
    assert info["counts"]["traces_kept"] == 7
    assert long_toy_root.parent.name not in json.dumps(info)
    ex_info = json.loads((processed / "pnw" / "exotic" / "preprocessing.json").read_text(encoding="utf-8"))
    assert ex_info["counts"]["traces_dropped"] == {"no_p_pick": 1}

    train = ProcessedEventDataset(processed / "pnw" / "comcat", split, ["train"])
    assert train.event_ids == ["uw1", "uw2", "uw3"]
    events = [train[i] for i in range(len(train))]
    assert [e.label for e in events] == ["earthquake", "anthropogenic", "earthquake"]
    assert events[2].metadata["label_raw"] == "eq" and events[2].metadata["split"] == "train"
    batch = collate_events(events, ["earthquake", "anthropogenic"])
    assert tuple(batch.waveforms.shape) == (3, 2, 3, 1000)       # N = largest event (2)
    assert batch.station_mask.sum(dim=1).tolist() == [2, 1, 2]   # uw1 lost its late-P trace
    assert batch.channel_mask[0].tolist()[0] == [False, False, True]   # CC.CCC vertical-only
    train.close()

    ood = ProcessedEventDataset(processed / "pnw" / "exotic", split, ["ood"])
    assert ood.event_ids == ["pnsn2"] and ood[0].label is None      # pnsn1 had no usable trace
    assert collate_events([ood[0]], ["earthquake", "anthropogenic"],
                          allow_unlabeled=True).labels.tolist() == [-1]
    ood.close()
