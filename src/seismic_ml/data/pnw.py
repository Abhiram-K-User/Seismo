"""PNW dataset (Ni et al., 2023): metadata, event selection, schema conversion.

Mapping from the SeisBench metadata to the project schemas (Phase 1)::

    SeismicEvent            <- all rows sharing ``event_id``
      label                 <- configured label column (default: PNSN analyst label)
      origin/location/mag   <- source_* columns (ComCat only; absent for exotic events)
      observations          <- one StationObservation per row
        Station             <- station_id = "<station_network_code>.<station_code>"
                               lat/lon/elevation from the SAME row (coordinates of a
                               site can change over time, so no static station table)
        WaveformSample      <- HDF5 array at ``trace_name`` ([3, T], order from
                               /data_format/component_order = "ENZ"), 100 Hz,
                               start_time = trace_start_time
        P/S arrival times   <- trace_start_time + trace_{P,S}_arrival_sample / rate
        metadata            <- location/channel codes, SNR, missing-channel and
                               offset flags, pick quality columns

Nothing here filters, merges or relabels data: conversion is lossless apart
from dtype-preserving array reads.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seismic_ml.data.schemas import SeismicEvent, Station, StationObservation, WaveformSample
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore

#: String-typed metadata columns (codes such as location "00" must not become ints).
STRING_COLUMNS: tuple[str, ...] = (
    "event_id", "source_type", "source_type_pnsn_label", "station_network_code",
    "station_code", "station_location_code", "station_channel_code", "trace_name",
    "trace_start_time", "source_origin_time",
)

#: Per-trace columns copied into ``StationObservation.metadata`` when present.
OBSERVATION_METADATA_COLUMNS: tuple[str, ...] = (
    "trace_name", "station_location_code", "station_channel_code", "trace_snr_db",
    "trace_missing_channel", "trace_has_offset", "trace_P_polarity", "trace_P_onset",
    "trace_S_onset", "trace_P_arrival_uncertainty_s", "trace_S_arrival_uncertainty_s",
    "trace_P_arrival_sample", "trace_S_arrival_sample",
)

#: Event-level columns copied into ``SeismicEvent.metadata`` when present.
EVENT_METADATA_COLUMNS: tuple[str, ...] = (
    "source_type", "source_type_pnsn_label", "preferred_source_magnitude_type",
    "preferred_source_magnitude_uncertainty", "source_depth_uncertainty_km",
    "source_horizontal_uncertainty_km",
)


def load_metadata(path: str | Path) -> pd.DataFrame:
    """Read a PNW metadata CSV with code columns kept as strings."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "string" for c in STRING_COLUMNS if c in header}
    return pd.read_csv(path, dtype=dtypes, low_memory=False, keep_default_na=True,
                       na_values=[""])


def station_id(network: str, station: str) -> str:
    """Graph-node identity of a recording site: ``"NET.STA"``."""
    return f"{network}.{station}"


def parse_utc(text: str) -> datetime:
    """Parse ISO timestamps such as ``2002-10-03T01:55:59.530000Z`` (timezone-aware UTC)."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def parse_snr_db(text: Any) -> tuple[float, float, float] | None:
    """Parse ``"6.1|3.0|11.7"`` (per component, same order as the waveform) or NaN."""
    if text is None or pd.isna(text):
        return None
    return tuple(float(v) for v in str(text).split("|"))  # type: ignore[return-value]


def event_table(metadata: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """One row per event: label and number of traces (stations).

    Raises:
        ValueError: If an event carries more than one label in ``label_column``.
    """
    grouped = metadata.groupby("event_id", sort=True)
    table = grouped.agg(label=(label_column, "first"), n_labels=(label_column, "nunique"),
                        n_traces=("trace_name", "size"))
    inconsistent = table.index[table["n_labels"] > 1]
    if len(inconsistent):
        raise ValueError(f"{len(inconsistent)} events have several {label_column!r} values, "
                         f"e.g. {list(inconsistent[:3])}")
    return table.drop(columns="n_labels")


def select_events(
    metadata: pd.DataFrame,
    *,
    label_column: str,
    events_per_label: Mapping[str, int],
    seed: int,
) -> list[str]:
    """Seeded, event-level sample: ``n`` events per label (``-1`` = all).

    Each label gets its own random stream derived from ``(seed, label)``, so
    changing the count of one label does not change the events chosen for
    another. Sampling is uniform over the (sorted) event IDs of that label.

    Raises:
        ValueError: If a requested label does not occur in the data.
    """
    table = event_table(metadata, label_column)
    selected: list[str] = []
    for label in sorted(events_per_label):
        ids = np.array(sorted(table.index[table["label"] == label]))
        if len(ids) == 0:
            raise ValueError(f"label {label!r} not found in column {label_column!r}")
        n = events_per_label[label]
        if n < 0 or n >= len(ids):
            selected.extend(ids.tolist())
            continue
        rng = np.random.default_rng([seed, zlib.crc32(label.encode())])
        selected.extend(rng.choice(ids, size=n, replace=False).tolist())
    return sorted(selected)


def select_events_balanced_temporal(
    metadata: pd.DataFrame,
    *,
    label_column: str,
    class_map: Mapping[str, str],
    train_end: str = "2019-01-01",
    val_end: str = "2021-01-01",
    seed: int = 42,
) -> list[str]:
    """Select 100% of anthropogenic blast events, and match natural earthquakes within each temporal split.

    Guarantees that train (< 2019), val (2019-2020), and test (2021+) splits are individually 50/50 balanced!
    """
    agg = {
        "label": (label_column, "first"),
        "origin_time": ("source_origin_time" if "source_origin_time" in metadata else "trace_start_time", "first"),
        "trace_start": ("trace_start_time", "min"),
    }
    events = metadata.groupby("event_id", sort=True).agg(**agg).reset_index()
    ref_time = pd.to_datetime(events["origin_time"], utc=True, format="ISO8601").fillna(
        pd.to_datetime(events["trace_start"], utc=True, format="ISO8601")
    )
    events["ref_time"] = ref_time
    events["class"] = events["label"].map(dict(class_map))

    t_tr = pd.Timestamp(train_end, tz="UTC")
    t_v = pd.Timestamp(val_end, tz="UTC")

    events["split_group"] = np.where(
        events["ref_time"] < t_tr, "train",
        np.where(events["ref_time"] < t_v, "val", "test")
    )

    rng = np.random.default_rng(seed)
    chosen_ids: list[str] = []

    # 1. Take all non-earthquake events (anthropogenic blasts + OOD categories like lf, su, sn)
    non_eq = events[events["class"] != "earthquake"]
    chosen_ids.extend(non_eq["event_id"].tolist())

    # 2. For each split (train, val, test), match earthquakes to the exact blast count in that split
    for s_name in ("train", "val", "test"):
        blasts_in_split = events[(events["split_group"] == s_name) & (events["class"] == "anthropogenic")]
        eqs_in_split = events[(events["split_group"] == s_name) & (events["class"] == "earthquake")]
        n_needed = len(blasts_in_split)
        eq_pool = np.array(sorted(eqs_in_split["event_id"].unique()))

        if n_needed >= len(eq_pool) or n_needed <= 0:
            chosen_ids.extend(eq_pool.tolist())
        else:
            sampled = rng.choice(eq_pool, size=n_needed, replace=False)
            chosen_ids.extend(sampled.tolist())

    return sorted(set(chosen_ids))


def _optional_float(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def row_to_station(row: Mapping[str, Any]) -> Station:
    """Build the :class:`Station` of one metadata row."""
    return Station(
        station_id=station_id(row["station_network_code"], row["station_code"]),
        latitude=float(row["station_latitude_deg"]),
        longitude=float(row["station_longitude_deg"]),
        elevation_m=_optional_float(row.get("station_elevation_m")),
        network=str(row["station_network_code"]),
    )


def row_to_observation(
    row: Mapping[str, Any],
    data: np.ndarray,
    component_order: Sequence[str],
) -> StationObservation:
    """Build a :class:`StationObservation` from a metadata row and its array."""
    rate = float(row["trace_sampling_rate_hz"])
    start = parse_utc(row["trace_start_time"])

    def arrival(column: str) -> datetime | None:
        sample = row.get(column)
        return None if sample is None or pd.isna(sample) else start + timedelta(seconds=float(sample) / rate)

    extra = {c: (None if pd.isna(row[c]) else row[c]) for c in OBSERVATION_METADATA_COLUMNS if c in row}
    if "trace_snr_db" in extra:
        extra["trace_snr_db"] = parse_snr_db(extra["trace_snr_db"])
    return StationObservation(
        station=row_to_station(row),
        waveform=WaveformSample(data=data, sampling_rate=rate, channels=tuple(component_order),
                                start_time=start),
        p_arrival_time=arrival("trace_P_arrival_sample"),
        s_arrival_time=arrival("trace_S_arrival_sample"),
        metadata=extra,
    )


def rows_to_event(
    rows: pd.DataFrame,
    store: SeisBenchWaveformStore,
    *,
    label_column: str,
) -> SeismicEvent:
    """Build one :class:`SeismicEvent` from all metadata rows of that event.

    Observations are ordered by station ID for determinism.

    Raises:
        ValueError: If the rows belong to more than one event.
        KeyError: If a trace of the event is not in ``store`` (incomplete event).
    """
    event_ids = rows["event_id"].unique()
    if len(event_ids) != 1:
        raise ValueError(f"rows belong to {len(event_ids)} events, expected 1")
    rows = rows.assign(_sid=rows["station_network_code"] + "." + rows["station_code"])
    rows = rows.sort_values("_sid")
    first = rows.iloc[0]
    observations = tuple(
        row_to_observation(row, store.get(row["trace_name"]), store.component_order)
        for row in (r.to_dict() for _, r in rows.iterrows())
    )

    def col(name: str) -> Any:
        return None if name not in rows or pd.isna(first[name]) else first[name]

    origin = col("source_origin_time")
    magnitude = col("preferred_source_magnitude")
    return SeismicEvent(
        event_id=str(event_ids[0]),
        observations=observations,
        label=None if pd.isna(first[label_column]) else str(first[label_column]),
        origin_time=parse_utc(origin) if origin is not None else None,
        latitude=_optional_float(col("source_latitude_deg")),
        longitude=_optional_float(col("source_longitude_deg")),
        depth_km=_optional_float(col("source_depth_km")),
        magnitude=_optional_float(magnitude),
        metadata={c: col(c) for c in EVENT_METADATA_COLUMNS if c in rows},
    )


def complete_event_ids(metadata: pd.DataFrame, store: SeisBenchWaveformStore) -> list[str]:
    """Events whose traces are ALL stored in ``store`` (sorted)."""
    available = store.available_trace_names()
    if available is None:
        return sorted(metadata["event_id"].unique())
    present = metadata["trace_name"].isin(available)
    complete = present.groupby(metadata["event_id"]).all()
    return sorted(complete.index[complete])


def iter_events(
    metadata: pd.DataFrame,
    store: SeisBenchWaveformStore,
    *,
    label_column: str,
    event_ids: Sequence[str] | None = None,
) -> Iterator[SeismicEvent]:
    """Yield events (in ``event_ids`` order; default: all complete events)."""
    ids = complete_event_ids(metadata, store) if event_ids is None else list(event_ids)
    groups = metadata[metadata["event_id"].isin(set(ids))].groupby("event_id")
    for event_id in ids:
        yield rows_to_event(groups.get_group(event_id), store, label_column=label_column)
