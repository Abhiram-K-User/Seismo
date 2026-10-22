"""Configuration files load; values are either measured facts or explicitly undecided."""

from __future__ import annotations

from pathlib import Path

import pytest

from seismic_ml.config import load_config, load_yaml


@pytest.mark.parametrize("name", ["config.yaml", "data.yaml", "model.yaml", "training.yaml"])
def test_every_config_file_parses(repo_root: Path, name: str) -> None:
    assert isinstance(load_yaml(repo_root / "configs" / name), dict)


def test_includes_are_resolved(config: dict) -> None:
    assert "includes" not in config
    for section in ("data", "model", "training"):
        assert isinstance(config[section], dict) and config[section]


def test_seed(config: dict) -> None:
    assert config["seed"] == 42
    assert isinstance(config["deterministic"], bool)


def test_storage_paths_are_relative_and_overridable(config: dict) -> None:
    paths = config["paths"]
    assert set(paths) == {"data_root", "output_root"}
    for value in paths.values():
        assert not Path(value).is_absolute()


def test_selected_dataset_is_documented(data_config: dict) -> None:
    dataset = data_config["dataset"]
    assert dataset["name"] == "pnw" and dataset["format"] == "seisbench"
    assert dataset["license"] == "CC BY 4.0" and dataset["raw_subdir"] == "pnw"
    assert not Path(dataset["raw_subdir"]).is_absolute()
    for spec in dataset["collections"].values():
        for key in ("metadata_file", "waveform_file", "subset_file"):
            assert not Path(spec[key]).is_absolute()
        assert spec["label_column"]


def test_class_definition_is_consistent(config: dict) -> None:
    labels = config["data"]["labels"]
    assert labels["num_classes"] == 2 == len(labels["class_names"])
    assert labels["class_names"] == ["earthquake", "anthropogenic"]
    assert set(labels["class_map"].values()) <= set(labels["class_names"])
    assert config["model"]["num_classes"] == labels["num_classes"]


def test_ood_and_training_labels_are_disjoint(data_config: dict) -> None:
    labels = data_config["labels"]
    trained = set(labels["class_map"])
    for collection, ood in labels["ood_labels"].items():
        assert not trained & set(ood), collection
    assert not trained & set(labels["excluded_labels"]["comcat"])


def test_measured_waveform_facts(data_config: dict) -> None:
    waveform = data_config["waveform"]
    assert waveform["sample_rate"] == 100
    assert waveform["num_channels"] == 3 == len(waveform["channel_order"])
    assert waveform["channel_order"] == ["E", "N", "Z"]


def test_phase2_preprocessing_parameters_are_consistent(data_config: dict) -> None:
    pre = data_config["preprocessing"]
    win, band = pre["window"], pre["bandpass"]
    assert data_config["waveform"]["window_length_s"] == win["pre_p_s"] + win["post_p_s"]
    assert 0 < band["freqmin"] < band["freqmax"] < pre["target_sample_rate"] / 2
    assert band["freqmax"] < 20.0          # BH channels carry no content above ~20 Hz (§11H)
    assert pre["target_sample_rate"] <= data_config["waveform"]["sample_rate"]
    assert pre["normalization"] == {"method": "std", "scope": "station"}
    assert pre["quality_checks"]["require_p_pick"] is True
    assert data_config["stations"]["num_stations"] is None      # dynamic padding, no truncation


def test_phase2_split_parameters(data_config: dict) -> None:
    split = data_config["split"]
    assert split["strategy"] == "temporal" and split["group_by"] == "event"
    assert split["train_end"] < split["val_end"]
    assert set(split["exclude_label_conflicts_from"]) == {"val", "test"}


def test_later_phase_settings_are_still_open(config: dict) -> None:
    assert config["model"]["encoder"]["embedding_dim"] == 128
    assert config["training"]["optimizer"]["name"] == "adamw"
    assert config["data"]["spectrogram"]["n_fft"] is None
    assert config["training"]["loss"]["evidential"]["variant"] is None


def test_future_components_are_disabled(config: dict) -> None:
    model = config["model"]
    for component in ("graph", "gat", "stgnn"):
        assert model[component]["enabled"] is False
    assert config["data"]["spectrogram"]["enabled"] is False


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)


def test_load_yaml_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_yaml(path) == {}


def test_load_config_requires_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 1\npaths: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(path)


def test_load_config_rejects_duplicate_section(tmp_path: Path) -> None:
    (tmp_path / "data.yaml").write_text("a: 1\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text("data: {}\nincludes: {data: data.yaml}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="both inline and via includes"):
        load_config(path)
