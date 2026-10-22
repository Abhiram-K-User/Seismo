"""Schemas validate their inputs; dataset and loader read events (toy fixture)."""

from __future__ import annotations

import dataclasses
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from seismic_ml.data import (
    EventBatch,
    SeismicEvent,
    SeismicEventDataset,
    Station,
    StationObservation,
    WaveformSample,
    collate_events,
    create_dataloader,
    load_events,
    waveform_path,
)
from seismic_ml.placeholders import NotYetImplementedError

# --- Station -----------------------------------------------------------------


def test_station_fields(station: Station) -> None:
    assert station.station_id == "XX.STA1"
    assert station.elevation_m is None


def test_station_is_immutable(station: Station) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        station.latitude = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"station_id": "", "latitude": 0.0, "longitude": 0.0},
        {"station_id": "A", "latitude": 91.0, "longitude": 0.0},
        {"station_id": "A", "latitude": 0.0, "longitude": 181.0},
        {"station_id": "A", "latitude": float("nan"), "longitude": 0.0},
    ],
)
def test_station_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Station(**kwargs)


# --- WaveformSample ----------------------------------------------------------


def test_waveform_properties(waveform: WaveformSample) -> None:
    assert waveform.channels == ("Z", "N", "E")
    assert waveform.num_channels == 3
    assert waveform.num_samples == 200
    assert waveform.duration_s == pytest.approx(2.0)


def test_waveform_start_time_optional() -> None:
    sample = WaveformSample(data=np.zeros((1, 10)), sampling_rate=10.0, channels=("Z",))
    assert sample.start_time is None


@pytest.mark.parametrize(
    ("data", "rate", "channels", "error"),
    [
        (np.zeros(10), 10.0, ("Z",), ValueError),               # not 2-D
        (np.zeros((2, 10)), 10.0, ("Z", "N", "E"), ValueError),  # channel count mismatch
        (np.zeros((2, 10)), 10.0, ("Z", "Z"), ValueError),       # duplicate channels
        (np.zeros((3, 10)), 0.0, ("Z", "N", "E"), ValueError),   # non-positive rate
        (np.zeros((3, 10)), float("inf"), ("Z", "N", "E"), ValueError),
        ([[0.0] * 10] * 3, 10.0, ("Z", "N", "E"), TypeError),   # not an ndarray
        (np.array([["a"] * 10] * 3), 10.0, ("Z", "N", "E"), TypeError),  # non-numeric
    ],
)
def test_waveform_rejects_invalid_values(data, rate, channels, error) -> None:
    with pytest.raises(error):
        WaveformSample(data=data, sampling_rate=rate, channels=channels)


# --- StationObservation ------------------------------------------------------


def test_observation_station_id(station: Station, waveform: WaveformSample) -> None:
    assert StationObservation(station=station, waveform=waveform).station_id == "XX.STA1"


def test_observation_rejects_s_before_p(station: Station, waveform: WaveformSample) -> None:
    p_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        StationObservation(
            station=station,
            waveform=waveform,
            p_arrival_time=p_time,
            s_arrival_time=p_time - timedelta(seconds=1),
        )


# --- SeismicEvent ------------------------------------------------------------


def test_event_properties(event: SeismicEvent) -> None:
    assert event.num_stations == 2
    assert event.station_ids == ("XX.STA1", "XX.STA2")
    assert all(isinstance(s, Station) for s in event.stations)
    assert event.label == "earthquake"
    assert event.magnitude is None and event.metadata == {}


def test_event_label_may_be_unknown(event: SeismicEvent) -> None:
    unlabeled = SeismicEvent(event_id="evt-x", observations=event.observations)
    assert unlabeled.label is None


def test_event_rejects_duplicate_stations(station: Station, waveform: WaveformSample) -> None:
    obs = StationObservation(station=station, waveform=waveform)
    with pytest.raises(ValueError, match="duplicate"):
        SeismicEvent(event_id="evt", observations=(obs, obs))


def test_event_rejects_empty_id(event: SeismicEvent) -> None:
    with pytest.raises(ValueError):
        SeismicEvent(event_id="", observations=event.observations)


# --- EventBatch --------------------------------------------------------------


def _batch(**overrides) -> EventBatch:
    kwargs = {
        "waveforms": torch.zeros(2, 4, 3, 50),
        "station_mask": torch.ones(2, 4, dtype=torch.bool),
        "event_ids": ("a", "b"),
        "coordinates": torch.zeros(2, 4, 2),
        "labels": torch.zeros(2, dtype=torch.long),
    }
    kwargs.update(overrides)
    return EventBatch(**kwargs)


def test_event_batch_shapes() -> None:
    batch = _batch()
    assert (batch.batch_size, batch.num_stations, batch.num_channels, batch.num_samples) == (
        2,
        4,
        3,
        50,
    )


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"waveforms": torch.zeros(2, 4, 50)}, ValueError),
        ({"station_mask": torch.ones(2, 3, dtype=torch.bool)}, ValueError),
        ({"station_mask": torch.ones(2, 4)}, TypeError),
        ({"event_ids": ("a",)}, ValueError),
        ({"coordinates": torch.zeros(2, 4, 3)}, ValueError),
        ({"labels": torch.zeros(3, dtype=torch.long)}, ValueError),
        ({"station_ids": (("s1", "s2"), ("s1", "s2"))}, ValueError),
    ],
)
def test_event_batch_rejects_inconsistent_shapes(overrides: dict, error: type) -> None:
    with pytest.raises(error):
        _batch(**overrides)


# --- Dataset and loader (Phase 1, toy PNW-layout fixture) ----------------------------


def test_waveform_path_prefers_full_file_then_subset(toy_raw_dir: Path, data_config: dict) -> None:
    assert waveform_path(toy_raw_dir, data_config, "comcat").name == "comcat_waveforms.hdf5"
    (toy_raw_dir / "comcat_waveforms.hdf5").rename(toy_raw_dir / "elsewhere.hdf5")
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        waveform_path(toy_raw_dir, data_config, "comcat")


def test_unknown_collection_is_rejected(toy_raw_dir: Path, data_config: dict) -> None:
    with pytest.raises(ValueError, match="Unknown collection"):
        load_events(toy_raw_dir, data_config, collection="volcanoes")


def test_load_events(toy_raw_dir: Path, data_config: dict) -> None:
    events = load_events(toy_raw_dir, data_config, collection="comcat")
    assert [e.event_id for e in events] == ["uw1", "uw2", "uw3", "uw4"]
    assert [e.label for e in events] == ["eq", "px", "eq", "lf"]      # PNSN analyst label
    assert len(load_events(toy_raw_dir, data_config, collection="exotic", limit=1)) == 1


def test_dataset_from_config(toy_raw_dir: Path, data_config: dict) -> None:
    ds = SeismicEventDataset.from_config(toy_raw_dir, data_config, collection="comcat",
                                         event_ids=["uw3", "uw1"], split="train")
    assert len(ds) == 2 and ds.split == "train"
    first = ds[0]
    assert isinstance(first, SeismicEvent) and first.event_id == "uw3"
    assert first.metadata["source_type"] == "explosion"               # label conflict kept visible
    clone = pickle.loads(pickle.dumps(ds))                            # DataLoader workers
    assert clone._store is None and clone[1].event_id == "uw1"
    ds.close()
    clone.close()


def test_dataset_rejects_unavailable_events(toy_raw_dir: Path, data_config: dict) -> None:
    with pytest.raises(KeyError):
        SeismicEventDataset.from_config(toy_raw_dir, data_config, event_ids=["uw1", "nope"])


# --- Batching (Phase 2) -------------------------------------------------------------------


def _toy_event(event_id: str, n_stations: int, label: str | None, *, t: int = 40,
               vertical_only: set[int] = frozenset()) -> SeismicEvent:
    obs = []
    for j in range(n_stations):
        data = np.full((3, t), float(j + 1), dtype=np.float32)
        mask = (False, False, True) if j in vertical_only else (True, True, True)
        data[~np.array(mask)] = 0.0
        obs.append(StationObservation(
            station=Station(f"XX.S{j}", latitude=10.0 + j, longitude=20.0 - j),
            waveform=WaveformSample(data=data, sampling_rate=50.0, channels=("E", "N", "Z")),
            metadata={"channel_mask": mask}))
    return SeismicEvent(event_id=event_id, observations=tuple(obs), label=label)


CLASSES = ["earthquake", "anthropogenic"]


def test_collate_events_pads_and_masks() -> None:
    events = [_toy_event("a", 2, "earthquake", vertical_only={1}),
              _toy_event("b", 3, "anthropogenic")]
    batch = collate_events(events, CLASSES)
    assert tuple(batch.waveforms.shape) == (2, 3, 3, 40) and batch.waveforms.dtype == torch.float32
    assert batch.station_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.labels.tolist() == [0, 1]
    assert batch.channel_mask[0, 1].tolist() == [False, False, True]    # vertical-only station
    assert not batch.channel_mask[0, 2].any()                           # padded slot
    assert batch.station_ids[0] == ("XX.S0", "XX.S1", "")
    assert batch.coordinates[1, 2].tolist() == [12.0, 18.0]
    assert float(batch.waveforms[0, 2].abs().sum()) == 0.0              # padding is zeros
    assert collate_events(events, CLASSES, max_stations=5).num_stations == 5


def test_collate_events_mask_falls_back_to_nonzero_components() -> None:
    event = _toy_event("a", 1, "earthquake")
    object.__setattr__(event.observations[0], "metadata", {})
    event.observations[0].waveform.data[0] = 0.0
    assert collate_events([event], CLASSES).channel_mask[0, 0].tolist() == [False, True, True]


def test_collate_events_labels_are_never_silently_mapped() -> None:
    with pytest.raises(ValueError, match="not in"):
        collate_events([_toy_event("a", 1, "eq")], CLASSES)             # raw label, not a class
    with pytest.raises(ValueError, match="no label"):
        collate_events([_toy_event("a", 1, None)], CLASSES)
    ood = collate_events([_toy_event("a", 1, None)], CLASSES, allow_unlabeled=True)
    assert ood.labels.tolist() == [-1]


def test_collate_events_rejects_inconsistent_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        collate_events([], CLASSES)
    with pytest.raises(ValueError, match="max_stations"):
        collate_events([_toy_event("a", 3, "earthquake")], CLASSES, max_stations=2)
    with pytest.raises(ValueError, match="preprocess first"):
        collate_events([_toy_event("a", 1, "earthquake", t=40),
                        _toy_event("b", 1, "earthquake", t=41)], CLASSES)


def test_event_batch_validates_channel_mask() -> None:
    ok = _batch(channel_mask=torch.ones(2, 4, 3, dtype=torch.bool))
    assert ok.channel_mask.shape == (2, 4, 3)
    with pytest.raises(TypeError):
        _batch(channel_mask=torch.ones(2, 4, 3))
    with pytest.raises(ValueError):
        _batch(channel_mask=torch.ones(2, 4, 2, dtype=torch.bool))
    padded = torch.ones(2, 4, dtype=torch.bool)
    padded[0, 3] = False
    with pytest.raises(ValueError, match="padded"):
        _batch(station_mask=padded, channel_mask=torch.ones(2, 4, 3, dtype=torch.bool))


# --- Placeholders (later phases) --------------------------------------------------------


def test_create_dataloader_batches_events(toy_raw_dir: Path, data_config: dict) -> None:
    ds = SeismicEventDataset.from_config(toy_raw_dir, data_config)
    loader = create_dataloader(ds, batch_size=2, shuffle=False, class_names=["eq", "px", "lf"], seed=42)
    batches = list(loader)
    assert len(batches) == (len(ds) + 1) // 2
    batch = batches[0]
    assert batch.waveforms.shape[0] == 2
    assert batch.station_mask.shape[0] == 2
    ds.close()
