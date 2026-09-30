"""Typed data structures for stations, waveforms and seismic events.

These schemas are dataset-agnostic on purpose: the dataset has not been
selected yet (Phase 1). They describe the *concepts* the rest of the code
base relies on, not a file format. Dataset-specific fields go into the
``metadata`` mappings until a decision is recorded in PROJECT_CONTEXT.md.

Hierarchy::

    SeismicEvent                         one labelled event (one ML sample)
      └── observations: StationObservation, ...
            ├── station:  Station        station id + location (one graph node)
            └── waveform: WaveformSample [C, T] array, e.g. Z/N/E components

``EventBatch`` is the padded, model-facing tensor form of several events.
Its shape convention is provisional (Decision 003 in PROJECT_CONTEXT.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch

#: Default component order of waveform arrays (rows of ``WaveformSample.data``).
DEFAULT_CHANNELS: tuple[str, ...] = ("Z", "N", "E")


def _check_latitude(value: float, name: str = "latitude") -> None:
    if not -90.0 <= value <= 90.0:  # also rejects NaN
        raise ValueError(f"{name} must be in [-90, 90] degrees, got {value}.")


def _check_longitude(value: float, name: str = "longitude") -> None:
    if not -180.0 <= value <= 180.0:  # also rejects NaN
        raise ValueError(f"{name} must be in [-180, 180] degrees, got {value}.")


@dataclass(frozen=True, slots=True)
class Station:
    """A seismic recording station — one node of the station graph.

    Attributes:
        station_id: Unique identifier. The FDSN ``"NET.STA"`` form is a natural
            choice but the final convention depends on the dataset.
        latitude: Degrees north, in [-90, 90].
        longitude: Degrees east, in [-180, 180] (convert 0–360 inputs first).
        elevation_m: Elevation above sea level in metres, if known.
        network: Network code, if known.
    """

    station_id: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    network: str | None = None

    def __post_init__(self) -> None:
        if not self.station_id:
            raise ValueError("station_id must be a non-empty string.")
        _check_latitude(self.latitude)
        _check_longitude(self.longitude)


@dataclass(slots=True)
class WaveformSample:
    """A multi-component waveform segment recorded at one station.

    Attributes:
        data: Numeric array of shape ``[channels, time]``. Raw counts or
            physical units — the unit is dataset-dependent (Phase 1).
        sampling_rate: Samples per second (Hz), finite and > 0.
        channels: Component name of each row of ``data``, e.g. ``("Z", "N", "E")``.
        start_time: Time of the first sample (timezone-aware UTC recommended),
            if known.
    """

    data: np.ndarray
    sampling_rate: float
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    start_time: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"data must be a numpy.ndarray, got {type(self.data).__name__}.")
        if not np.issubdtype(self.data.dtype, np.number):
            raise TypeError(f"data must be numeric, got dtype {self.data.dtype}.")
        if self.data.ndim != 2:
            raise ValueError(f"data must have shape [channels, time], got {self.data.shape}.")
        self.channels = tuple(self.channels)
        if len(self.channels) != self.data.shape[0]:
            raise ValueError(
                f"{len(self.channels)} channel names given for {self.data.shape[0]} data rows."
            )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"channel names must be unique, got {self.channels}.")
        if not (math.isfinite(self.sampling_rate) and self.sampling_rate > 0):
            raise ValueError(f"sampling_rate must be finite and > 0, got {self.sampling_rate}.")

    @property
    def num_channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration_s(self) -> float:
        """Duration in seconds (``num_samples / sampling_rate``)."""
        return self.num_samples / self.sampling_rate


@dataclass(slots=True)
class StationObservation:
    """The recording of one event at one station.

    Attributes:
        station: The recording station.
        waveform: The station's multi-component waveform for this event.
        p_arrival_time: P-phase arrival time, if picked/available.
        s_arrival_time: S-phase arrival time, if picked/available.
        metadata: Dataset-specific extras (e.g. SNR, pick uncertainty, quality flags).
    """

    station: Station
    waveform: WaveformSample
    p_arrival_time: datetime | None = None
    s_arrival_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            self.p_arrival_time is not None
            and self.s_arrival_time is not None
            and self.s_arrival_time < self.p_arrival_time
        ):
            raise ValueError("s_arrival_time must not precede p_arrival_time.")

    @property
    def station_id(self) -> str:
        return self.station.station_id


@dataclass(slots=True)
class SeismicEvent:
    """One seismic event observed by one or more stations (one ML sample).

    Attributes:
        event_id: Unique event identifier (e.g. catalog ID).
        observations: Station recordings of this event; station IDs must be unique.
        label: Raw class label as a string (e.g. ``"earthquake"``), or ``None``
            if unlabelled/unknown. Mapping to class indices is configured in
            ``configs/data.yaml`` (``labels.class_names``); never infer labels.
        origin_time, latitude, longitude, depth_km, magnitude: Catalog origin
            information, if available.
        metadata: Dataset-specific extras (e.g. catalog event type, author).

    Warning:
        Origin information is for splitting, stratification and analysis. It
        must not be used as a model input without a recorded decision: it can
        act as a shortcut/confounder (e.g. blasts occur at known quarry sites,
        during working hours, at shallow depth and small magnitude).
        See PROJECT_CONTEXT.md §12.
    """

    event_id: str
    observations: tuple[StationObservation, ...]
    label: str | None = None
    origin_time: datetime | None = None
    latitude: float | None = None
    longitude: float | None = None
    depth_km: float | None = None
    magnitude: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be a non-empty string.")
        self.observations = tuple(self.observations)
        ids = [obs.station_id for obs in self.observations]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate station IDs in event {self.event_id!r}: {ids}.")
        if self.latitude is not None:
            _check_latitude(self.latitude)
        if self.longitude is not None:
            _check_longitude(self.longitude)

    @property
    def num_stations(self) -> int:
        return len(self.observations)

    @property
    def station_ids(self) -> tuple[str, ...]:
        return tuple(obs.station_id for obs in self.observations)

    @property
    def stations(self) -> tuple[Station, ...]:
        return tuple(obs.station for obs in self.observations)


@dataclass(slots=True)
class EventBatch:
    """Padded, model-facing batch of events (provisional convention).

    Shapes (B = batch, N = max stations per event, C = channels, T = samples):

    ============== ====== ================ =========================================
    field          dtype  shape            meaning
    ============== ====== ================ =========================================
    waveforms      float  ``[B, N, C, T]`` padded station waveforms
    station_mask   bool   ``[B, N]``       True where a real station is present
    event_ids      —      length B         event identifiers
    coordinates    float  ``[B, N, 2]``    (latitude, longitude) in degrees, optional
    labels         long   ``[B]``          class indices, optional
    station_ids    —      B × N            station IDs per slot ("" = padding), optional
    channel_mask   bool   ``[B, N, C]``    True where a component holds real data, optional
    ============== ====== ================ =========================================

    Padded slots (mask False) carry no information and must be ignored by
    models. ``channel_mask`` marks components that were missing (zero-filled)
    in the raw data — they are zeros, not signal (Decision 022). Shapes follow
    Decision 003 (provisional).
    """

    waveforms: torch.Tensor
    station_mask: torch.Tensor
    event_ids: tuple[str, ...]
    coordinates: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    station_ids: tuple[tuple[str, ...], ...] | None = None
    channel_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.waveforms.ndim != 4:
            raise ValueError(f"waveforms must be [B, N, C, T], got {tuple(self.waveforms.shape)}.")
        batch, stations = self.waveforms.shape[:2]
        if self.station_mask.dtype != torch.bool:
            raise TypeError(f"station_mask must be bool, got {self.station_mask.dtype}.")
        if tuple(self.station_mask.shape) != (batch, stations):
            raise ValueError(
                f"station_mask must be [{batch}, {stations}], got {tuple(self.station_mask.shape)}."
            )
        self.event_ids = tuple(self.event_ids)
        if len(self.event_ids) != batch:
            raise ValueError(f"expected {batch} event_ids, got {len(self.event_ids)}.")
        if self.coordinates is not None and tuple(self.coordinates.shape) != (batch, stations, 2):
            raise ValueError(
                f"coordinates must be [{batch}, {stations}, 2], got {tuple(self.coordinates.shape)}."
            )
        if self.labels is not None and tuple(self.labels.shape) != (batch,):
            raise ValueError(f"labels must be [{batch}], got {tuple(self.labels.shape)}.")
        if self.station_ids is not None:
            self.station_ids = tuple(tuple(row) for row in self.station_ids)
            if len(self.station_ids) != batch or any(len(r) != stations for r in self.station_ids):
                raise ValueError(f"station_ids must be {batch} rows of {stations} IDs.")
        if self.channel_mask is not None:
            expected = tuple(self.waveforms.shape[:3])
            if self.channel_mask.dtype != torch.bool:
                raise TypeError(f"channel_mask must be bool, got {self.channel_mask.dtype}.")
            if tuple(self.channel_mask.shape) != expected:
                raise ValueError(f"channel_mask must be {list(expected)}, "
                                 f"got {tuple(self.channel_mask.shape)}.")
            if bool((self.channel_mask & ~self.station_mask[..., None]).any()):
                raise ValueError("channel_mask marks components of padded stations as present.")

    @property
    def batch_size(self) -> int:
        return self.waveforms.shape[0]

    @property
    def num_stations(self) -> int:
        return self.waveforms.shape[1]

    @property
    def num_channels(self) -> int:
        return self.waveforms.shape[2]

    @property
    def num_samples(self) -> int:
        return self.waveforms.shape[3]
