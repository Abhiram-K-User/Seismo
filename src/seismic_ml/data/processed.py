"""Preprocessed (model-ready) waveform store: writer and reader (Phase 2).

Layout of one processed collection directory (e.g. ``<processed>/pnw/comcat/``)::

    waveforms.hdf5       /waveforms     float32 [n_traces, C, T]  P-aligned, filtered, normalised
                         /channel_mask  bool    [n_traces, C]     True = real data (not zero-filled)
                         attrs: sampling_rate, pre_p_s, post_p_s, channels
    traces.csv           one row per kept trace (row index into the arrays + metadata)
    dropped_traces.csv   one row per trace that could not be processed, with the reason
    preprocessing.json   exact configuration, inputs (relative paths + SHA-256), counts

A trace is dropped only for hard reasons (no P pick, not enough data around P,
non-finite samples, no live component). Soft issues are kept as flags. The
split assignment is *not* stored here — it lives in the split table and is
joined at read time, so re-splitting never requires re-processing.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from seismic_ml.data import pnw
from seismic_ml.data.schemas import SeismicEvent, Station, StationObservation, WaveformSample
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore
from seismic_ml.preprocessing.waveform import WindowOutOfRangeError, preprocess_observation

WAVEFORMS_FILE = "waveforms.hdf5"
TRACES_FILE = "traces.csv"
DROPPED_FILE = "dropped_traces.csv"
INFO_FILE = "preprocessing.json"


def _drop_reason(exc: Exception) -> str:
    if isinstance(exc, WindowOutOfRangeError):
        return "window_out_of_range"
    text = str(exc)
    if "no P pick" in text:
        return "no_p_pick"
    if "non_finite_samples" in text:
        return "non_finite_samples"
    if "no_live_components" in text:
        return "no_live_components"
    raise exc


def write_processed_collection(
    dest_dir: str | Path,
    metadata: pd.DataFrame,
    store: SeisBenchWaveformStore,
    *,
    label_column: str,
    config: Mapping[str, Any],
    event_ids: Sequence[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Preprocess every (locally complete) event of a collection into ``dest_dir``.

    Args:
        metadata: Collection metadata (``pnw.load_metadata``).
        store: Raw waveform store (full file or exploration subset).
        label_column: Raw label column (kept as ``label_raw`` in traces.csv).
        config: ``data.preprocessing`` section.
        event_ids: Events to process (default: all complete events in ``store``).

    Returns:
        Counts of processed/dropped traces and events.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ids = pnw.complete_event_ids(metadata, store) if event_ids is None else list(event_ids)
    groups = metadata[metadata["event_id"].isin(set(ids))].groupby("event_id")
    win = config["window"]
    rate = float(config["target_sample_rate"])
    n_samples = int(round((win["pre_p_s"] + win["post_p_s"]) * rate))
    channels = tuple(store.component_order)

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    tmp = dest / (WAVEFORMS_FILE + ".part")
    with h5py.File(tmp, "w") as h5:
        wave = h5.create_dataset("waveforms", shape=(0, len(channels), n_samples),
                                 maxshape=(None, len(channels), n_samples), dtype="float32",
                                 chunks=(1, len(channels), n_samples))
        cmask = h5.create_dataset("channel_mask", shape=(0, len(channels)),
                                  maxshape=(None, len(channels)), dtype=bool)
        for done, event_id in enumerate(ids, start=1):
            event = pnw.rows_to_event(groups.get_group(event_id), store, label_column=label_column)
            processed = []
            for obs in event.observations:
                try:
                    processed.append((obs, *preprocess_observation(obs, config)))
                except (ValueError, WindowOutOfRangeError) as exc:
                    dropped.append({"event_id": event.event_id, "trace_name": obs.metadata["trace_name"],
                                    "station_id": obs.station_id, "reason": _drop_reason(exc)})
            if processed:
                first_p = min(o.p_arrival_time for o, _, _ in processed)
            for raw, out, info in processed:
                row = wave.shape[0]
                wave.resize(row + 1, axis=0)
                cmask.resize(row + 1, axis=0)
                wave[row] = out.waveform.data
                cmask[row] = info["channel_mask"]
                s_index = ((raw.s_arrival_time - out.waveform.start_time).total_seconds() * rate
                           if raw.s_arrival_time is not None else np.nan)
                kept.append({
                    "row": row, "event_id": event.event_id, "trace_name": raw.metadata["trace_name"],
                    "station_id": raw.station_id, "network": raw.station.network,
                    "latitude": raw.station.latitude, "longitude": raw.station.longitude,
                    "elevation_m": raw.station.elevation_m,
                    "channel_code": raw.metadata.get("station_channel_code"),
                    "location_code": raw.metadata.get("station_location_code"),
                    "label_raw": event.label,
                    "p_arrival_time": raw.p_arrival_time.isoformat(),
                    "s_arrival_time": raw.s_arrival_time.isoformat() if raw.s_arrival_time else None,
                    "window_start_time": out.waveform.start_time.isoformat(),
                    "p_index": win["pre_p_s"] * rate,
                    "s_index": s_index,
                    "p_time_offset_s": (raw.p_arrival_time - first_p).total_seconds(),
                    "live_components": "".join(c for c, m in zip(channels, info["channel_mask"]) if m),
                    "norm_scale": "|".join(f"{v:.6g}" for v in info["norm_scale"]),
                    "clipped_components": "".join(info["clipped_components"]),
                    "flat_components": "".join(info["flat_components"]),
                    "has_offset": raw.metadata.get("trace_has_offset"),
                })
            if progress is not None:
                progress(done, len(ids))
        h5.attrs["sampling_rate"] = rate
        h5.attrs["pre_p_s"] = float(win["pre_p_s"])
        h5.attrs["post_p_s"] = float(win["post_p_s"])
        h5.attrs["channels"] = "".join(channels)
    tmp.replace(dest / WAVEFORMS_FILE)

    traces = pd.DataFrame(kept)
    drops = pd.DataFrame(dropped, columns=["event_id", "trace_name", "station_id", "reason"])
    traces.to_csv(dest / TRACES_FILE, index=False)
    drops.to_csv(dest / DROPPED_FILE, index=False)
    events_kept = traces["event_id"].nunique() if len(traces) else 0
    return {
        "events_input": len(ids), "events_kept": int(events_kept),
        "events_dropped": int(len(ids) - events_kept),
        "traces_input": int(len(traces) + len(drops)), "traces_kept": int(len(traces)),
        "traces_dropped": {r: int(n) for r, n in drops["reason"].value_counts().items()},
        "samples_per_trace": n_samples, "sampling_rate": rate, "channels": "".join(channels),
    }


def write_info(dest_dir: str | Path, info: Mapping[str, Any]) -> None:
    """Write ``preprocessing.json`` (configuration, inputs, counts)."""
    (Path(dest_dir) / INFO_FILE).write_text(json.dumps(info, indent=2, default=str) + "\n",
                                            encoding="utf-8")


class ProcessedEventDataset(Dataset[SeismicEvent]):
    """Events from a processed collection, restricted to the chosen splits.

    ``SeismicEvent.label`` is the supervised class name (``earthquake`` /
    ``anthropogenic``) for train/val/test events and ``None`` for OOD events;
    the raw label, split, role and ``site_seen_in_train`` are in
    ``SeismicEvent.metadata``. Waveforms are the processed windows (start =
    P - pre_p_s). The HDF5 file is opened lazily per process.

    Args:
        processed_dir: Directory written by :func:`write_processed_collection`.
        split_table: Output of :func:`seismic_ml.data.splits.temporal_split`.
        splits: Split values to include (e.g. ``("train",)`` or ``("ood",)``).
    """

    def __init__(self, processed_dir: str | Path, split_table: pd.DataFrame,
                 splits: Sequence[str]) -> None:
        self.processed_dir = Path(processed_dir)
        traces = pd.read_csv(self.processed_dir / TRACES_FILE,
                             dtype={"event_id": str, "location_code": str, "network": str})
        chosen = split_table[split_table["split"].isin(list(splits))].set_index("event_id")
        traces = traces[traces["event_id"].isin(chosen.index)]
        self.event_ids: list[str] = sorted(traces["event_id"].unique())
        self._traces = traces.reset_index(drop=True)
        self._rows = self._traces.groupby("event_id").indices
        self._events = chosen
        self._h5: h5py.File | None = None
        with h5py.File(self.processed_dir / WAVEFORMS_FILE, "r") as h5:
            self.sampling_rate = float(h5.attrs["sampling_rate"])
            self.pre_p_s = float(h5.attrs["pre_p_s"])
            self.channels = tuple(str(h5.attrs["channels"]))

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.processed_dir / WAVEFORMS_FILE, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, index: int) -> SeismicEvent:
        event_id = self.event_ids[index]
        rows = self._traces.iloc[self._rows[event_id]].sort_values("station_id")
        info = self._events.loc[event_id]
        observations = []
        for r in rows.itertuples():
            data = self.h5["waveforms"][int(r.row)]
            mask = tuple(bool(v) for v in self.h5["channel_mask"][int(r.row)])
            start = pnw.parse_utc(r.window_start_time)
            observations.append(StationObservation(
                station=Station(r.station_id, float(r.latitude), float(r.longitude),
                                None if pd.isna(r.elevation_m) else float(r.elevation_m),
                                r.network),
                waveform=WaveformSample(data=data, sampling_rate=self.sampling_rate,
                                        channels=self.channels, start_time=start),
                p_arrival_time=pnw.parse_utc(r.p_arrival_time),
                s_arrival_time=None if pd.isna(r.s_arrival_time) else pnw.parse_utc(r.s_arrival_time),
                metadata={"channel_mask": mask, "trace_name": r.trace_name,
                          "channel_code": r.channel_code, "p_time_offset_s": r.p_time_offset_s,
                          "norm_scale": r.norm_scale, "clipped_components": r.clipped_components,
                          "has_offset": r.has_offset},
            ))
        klass = info["class"] if info["split"] in ("train", "val", "test") else None
        return SeismicEvent(
            event_id=event_id, observations=tuple(observations),
            label=None if pd.isna(klass) else str(klass),
            metadata={"label_raw": info["label_raw"], "split": info["split"], "role": info["role"],
                      "collection": info["collection"],
                      "site_seen_in_train": info["site_seen_in_train"]},
        )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
