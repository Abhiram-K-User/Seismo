"""PyTorch dataset of seismic events (one :class:`SeismicEvent` per index)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from seismic_ml.data import pnw
from seismic_ml.data.loader import collection_config, load_collection_metadata, waveform_path
from seismic_ml.data.schemas import SeismicEvent
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore


class SeismicEventDataset(Dataset[SeismicEvent]):
    """Map-style dataset yielding one complete :class:`SeismicEvent` per index.

    The waveform file is opened lazily in each process (``h5py`` handles
    cannot be pickled into DataLoader worker processes). Events are returned
    unprocessed; preprocessing and batching are Phase 2.

    Args:
        metadata: Metadata table of one collection (see ``pnw.load_metadata``).
        waveform_file: Local SeisBench-layout HDF5 file (full file or subset).
        label_column: Metadata column used as ``SeismicEvent.label``.
        event_ids: Events to expose, in this order. Default: every event whose
            traces are all present in ``waveform_file``. A split (Phase 2) is
            simply a list of event IDs — never a list of traces (§12).
        split: Optional name for bookkeeping (e.g. ``"train"``).

    Raises:
        KeyError: If an explicitly requested event is not completely available.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        waveform_file: str | Path,
        *,
        label_column: str,
        event_ids: Sequence[str] | None = None,
        split: str | None = None,
    ) -> None:
        self.waveform_file = Path(waveform_file)
        self.label_column = label_column
        self.split = split
        self._store: SeisBenchWaveformStore | None = None
        available = pnw.complete_event_ids(metadata, self.store)
        if event_ids is None:
            self.event_ids = available
        else:
            missing = sorted(set(event_ids) - set(available))
            if missing:
                raise KeyError(f"{len(missing)} requested events are not completely available "
                               f"locally, e.g. {missing[:3]}")
            self.event_ids = list(event_ids)
        self._metadata = metadata[metadata["event_id"].isin(set(self.event_ids))].reset_index(drop=True)
        self._rows = self._metadata.groupby("event_id").indices   # event_id -> positional rows

    @classmethod
    def from_config(
        cls,
        raw_dir: str | Path,
        data_config: Mapping[str, Any],
        *,
        collection: str = "comcat",
        event_ids: Sequence[str] | None = None,
        split: str | None = None,
    ) -> SeismicEventDataset:
        """Build from the configured collection files under ``raw_dir``."""
        spec = collection_config(data_config, collection)
        return cls(
            load_collection_metadata(raw_dir, data_config, collection),
            waveform_path(raw_dir, data_config, collection),
            label_column=spec["label_column"],
            event_ids=event_ids,
            split=split,
        )

    @property
    def store(self) -> SeisBenchWaveformStore:
        if self._store is None:
            self._store = SeisBenchWaveformStore(self.waveform_file)
        return self._store

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, index: int) -> SeismicEvent:
        rows = self._metadata.iloc[self._rows[self.event_ids[index]]]
        return pnw.rows_to_event(rows, self.store, label_column=self.label_column)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_store"] = None  # reopened lazily in the worker process
        return state

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
