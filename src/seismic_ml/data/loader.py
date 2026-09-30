"""Data loading: locate dataset files and turn them into :class:`SeismicEvent` objects.

All locations derive from a dataset raw directory that the caller resolves
through :mod:`seismic_ml.paths` (``--data-root`` / ``$SEISMIC_DATA_ROOT`` /
config); file names come from ``configs/data.yaml``. Nothing is hard-coded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from seismic_ml.data import pnw
from seismic_ml.data.schemas import EventBatch, SeismicEvent
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore
from seismic_ml.placeholders import NotYetImplementedError

SUPPORTED_FORMATS: tuple[str, ...] = ("seisbench",)


def collection_config(data_config: Mapping[str, Any], collection: str) -> Mapping[str, Any]:
    """The ``data.dataset.collections.<collection>`` block.

    Raises:
        ValueError: If the dataset format is unsupported or the collection unknown.
    """
    dataset = data_config["dataset"]
    if dataset.get("format") not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported dataset format {dataset.get('format')!r}; "
                         f"supported: {SUPPORTED_FORMATS}")
    collections = dataset.get("collections") or {}
    if collection not in collections:
        raise ValueError(f"Unknown collection {collection!r}; configured: {sorted(collections)}")
    return collections[collection]


def waveform_path(raw_dir: str | Path, data_config: Mapping[str, Any], collection: str) -> Path:
    """Local waveform file for ``collection``: the full file if present, else the subset.

    Raises:
        FileNotFoundError: If neither exists (with instructions to acquire data).
    """
    spec = collection_config(data_config, collection)
    raw_dir = Path(raw_dir)
    for candidate in (spec.get("waveform_file"), spec.get("subset_file")):
        if candidate and (raw_dir / candidate).exists():
            return raw_dir / candidate
    raise FileNotFoundError(
        f"No waveform file for collection {collection!r} under {raw_dir}. Run "
        "`python scripts/download_data.py` (optionally with --data-root) first."
    )


def load_collection_metadata(raw_dir: str | Path, data_config: Mapping[str, Any],
                             collection: str) -> pd.DataFrame:
    """Read the metadata table of one collection."""
    spec = collection_config(data_config, collection)
    return pnw.load_metadata(Path(raw_dir) / spec["metadata_file"])


def load_events(
    raw_dir: str | Path,
    data_config: Mapping[str, Any],
    *,
    collection: str = "comcat",
    event_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[SeismicEvent]:
    """Load events with their waveforms into memory.

    Args:
        raw_dir: Dataset raw directory (``ProjectPaths.dataset_raw_dir(...)``).
        data_config: The ``data`` configuration section.
        collection: Configured collection name (``"comcat"`` or ``"exotic"``).
        event_ids: Events to load; default = every event whose traces are all
            available locally (the whole subset, or the whole dataset).
        limit: Load at most this many events (in order).

    Raises:
        KeyError: If a requested event is not completely available locally.
    """
    spec = collection_config(data_config, collection)
    metadata = load_collection_metadata(raw_dir, data_config, collection)
    with SeisBenchWaveformStore(waveform_path(raw_dir, data_config, collection)) as store:
        events = pnw.iter_events(metadata, store, label_column=spec["label_column"],
                                 event_ids=event_ids)
        return list(islice(events, limit))


def collate_events(
    events: Sequence[SeismicEvent],
    class_names: Sequence[str],
    max_stations: int | None = None,
    *,
    allow_unlabeled: bool = False,
) -> EventBatch:
    """Pad a list of preprocessed events into an :class:`EventBatch`.

    Stations keep the order of ``event.observations``. Padded station slots
    are all-zero with ``station_mask`` False. ``channel_mask`` comes from each
    observation's ``metadata["channel_mask"]`` (written by preprocessing) or,
    if absent, from which components contain non-zero samples.

    Args:
        events: Events whose waveforms share channel names, rate and length.
        class_names: Ordered class names; ``class_names.index(label)`` is the
            class index. Labels not listed are rejected, never silently mapped.
        max_stations: Padding size ``N`` (default: the largest event). Events
            with more stations raise — truncation would be a silent choice.
        allow_unlabeled: Accept ``label=None`` (OOD events); encoded as ``-1``.

    Returns:
        ``EventBatch`` (float32 waveforms ``[B, N, C, T]``, masks, labels, coordinates).

    Raises:
        ValueError: Empty input, inconsistent waveforms, unknown/missing labels,
            or an event larger than ``max_stations``.
    """
    if not events:
        raise ValueError("cannot collate an empty list of events")
    first = events[0].observations[0].waveform
    channels, rate, length = first.channels, first.sampling_rate, first.num_samples
    for event in events:
        if not event.observations:
            raise ValueError(f"event {event.event_id} has no observations")
        for obs in event.observations:
            wf = obs.waveform
            if (wf.channels, wf.sampling_rate, wf.num_samples) != (channels, rate, length):
                raise ValueError(f"{event.event_id}/{obs.station_id}: waveform "
                                 f"{wf.channels}@{wf.sampling_rate}Hz x{wf.num_samples} differs "
                                 f"from {channels}@{rate}Hz x{length}; preprocess first")
    n_max = max(e.num_stations for e in events)
    n_pad = n_max if max_stations is None else max_stations
    if n_max > n_pad:
        raise ValueError(f"an event has {n_max} stations > max_stations={n_pad}")

    labels = []
    for event in events:
        if event.label is None:
            if not allow_unlabeled:
                raise ValueError(f"event {event.event_id} has no label (set allow_unlabeled=True)")
            labels.append(-1)
        elif event.label in class_names:
            labels.append(list(class_names).index(event.label))
        else:
            raise ValueError(f"event {event.event_id}: label {event.label!r} not in {list(class_names)}")

    b, c = len(events), len(channels)
    waveforms = np.zeros((b, n_pad, c, length), dtype=np.float32)
    station_mask = np.zeros((b, n_pad), dtype=bool)
    channel_mask = np.zeros((b, n_pad, c), dtype=bool)
    coordinates = np.zeros((b, n_pad, 2), dtype=np.float32)
    station_ids = [["" for _ in range(n_pad)] for _ in range(b)]
    for i, event in enumerate(events):
        for j, obs in enumerate(event.observations):
            waveforms[i, j] = obs.waveform.data
            station_mask[i, j] = True
            mask = obs.metadata.get("channel_mask")
            channel_mask[i, j] = mask if mask is not None else np.any(obs.waveform.data != 0, axis=-1)
            coordinates[i, j] = (obs.station.latitude, obs.station.longitude)
            station_ids[i][j] = obs.station_id
    return EventBatch(
        waveforms=torch.from_numpy(waveforms),
        station_mask=torch.from_numpy(station_mask),
        event_ids=tuple(e.event_id for e in events),
        coordinates=torch.from_numpy(coordinates),
        labels=torch.tensor(labels, dtype=torch.long),
        station_ids=tuple(tuple(row) for row in station_ids),
        channel_mask=torch.from_numpy(channel_mask),
    )


def create_dataloader(
    dataset: Dataset[SeismicEvent],
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """Build a seeded DataLoader using :func:`collate_events`.

    TODO(Phase 3): implement together with the training pipeline.
    """
    raise NotYetImplementedError("create_dataloader", phase=3)
