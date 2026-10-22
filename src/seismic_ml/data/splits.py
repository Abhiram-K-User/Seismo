"""Leakage-aware event splits (Phase 2; design in PROJECT_CONTEXT.md §11F).

The unit is always the **event**: a split is a table of event IDs, never a
list of traces. Every event of every collection receives exactly one row with
its role and — if it is not used for supervised training/evaluation — the
reason. Nothing is dropped silently.

``split`` values:

- ``train`` / ``val`` / ``test`` — supervised events (class = earthquake |
  anthropogenic), assigned by origin time (S1, temporal).
- ``ood`` — OOD candidates (never used for training, tuning or thresholds).
- ``excluded`` — with ``reason``: excluded label, boundary buffer, label
  conflict in val/test, duplicate straddling a boundary.

Built from metadata only; waveform availability is handled separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seismic_ml.data.inspection import possible_duplicate_events

SPLIT_VALUES: tuple[str, ...] = ("train", "val", "test", "ood", "excluded")

#: ComCat ``source_type`` -> supervised class, used only to detect label conflicts.
COMCAT_SOURCE_TYPE_CLASS: dict[str, str] = {"earthquake": "earthquake",
                                             "explosion": "anthropogenic"}


def event_table(metadata: pd.DataFrame, *, collection: str, label_column: str) -> pd.DataFrame:
    """One row per event with the fields needed for splitting.

    Columns: ``event_id, collection, label_raw, source_type, origin_time,
    latitude, longitude, n_stations`` (origin fields are NaN/NaT for exotic
    events, which have none; ``reference_time`` falls back to the earliest
    trace start so every event has a date).
    """
    agg: dict[str, Any] = {"label_raw": (label_column, "first"), "n_stations": ("trace_name", "size"),
                           "first_trace_start": ("trace_start_time", "min")}
    for column, name in [("source_type", "source_type"), ("source_origin_time", "origin_time"),
                         ("source_latitude_deg", "latitude"), ("source_longitude_deg", "longitude")]:
        if column in metadata:
            agg[name] = (column, "first")
    events = metadata.groupby("event_id", sort=True).agg(**agg).reset_index()
    events.insert(1, "collection", collection)
    for name in ("source_type", "origin_time", "latitude", "longitude"):
        if name not in events:
            events[name] = np.nan
    origin = pd.to_datetime(events["origin_time"], utc=True, format="ISO8601")
    first = pd.to_datetime(events["first_trace_start"], utc=True, format="ISO8601")
    events["reference_time"] = origin.fillna(first)
    events["year"] = events["reference_time"].dt.year
    events["event_id"] = events["event_id"].astype(str)
    return events.drop(columns="first_trace_start")


def assign_roles(
    events: pd.DataFrame,
    *,
    class_map: Mapping[str, str],
    ood_labels: Sequence[str],
    excluded_labels: Sequence[str],
) -> pd.DataFrame:
    """Add ``role`` (supervised | ood | excluded) and ``class`` columns.

    Raises:
        ValueError: If a raw label has no explicit role (never silently mapped).
    """
    out = events.copy()
    labels = out["label_raw"].astype(str)
    out["class"] = labels.map(dict(class_map))
    role = pd.Series(np.select([labels.isin(list(class_map)), labels.isin(list(ood_labels)),
                                labels.isin(list(excluded_labels))],
                               ["supervised", "ood", "excluded"], default="UNMAPPED"),
                     index=out.index)
    unmapped = sorted(labels[role == "UNMAPPED"].unique())
    if unmapped:
        raise ValueError(f"labels without a configured role: {unmapped}")
    out["role"] = role
    return out


def label_conflicts(events: pd.DataFrame) -> pd.Series:
    """Boolean Series: ComCat ``source_type`` class disagrees with the PNSN-derived class."""
    comcat_class = events["source_type"].map(COMCAT_SOURCE_TYPE_CLASS)
    return (events["role"] == "supervised") & comcat_class.notna() & (comcat_class != events["class"])


def site_cells(events: pd.DataFrame, cell_deg: float) -> pd.Series:
    """Spatial source-site cell id (``"<i>_<j>"`` on a ``cell_deg`` grid; NaN without location)."""
    lat_i = np.round(events["latitude"] / cell_deg)
    lon_i = np.round(events["longitude"] / cell_deg)
    cells = lat_i.astype("Int64").astype(str) + "_" + lon_i.astype("Int64").astype(str)
    return cells.where(events["latitude"].notna() & events["longitude"].notna())


def temporal_split(
    metadata_by_collection: Mapping[str, pd.DataFrame],
    *,
    label_columns: Mapping[str, str],
    class_map: Mapping[str, str],
    ood_labels: Mapping[str, Sequence[str]],
    excluded_labels: Mapping[str, Sequence[str]],
    train_end: str,
    val_end: str,
    buffer_days: float,
    site_cell_deg: float,
    exclude_label_conflicts_from: Sequence[str] = ("val", "test"),
    duplicate_max_dt_s: float = 2.0,
    duplicate_max_dist_deg: float = 0.05,
) -> pd.DataFrame:
    """Build the S1 temporal split for every event of every collection.

    Supervised events: ``train`` if origin < ``train_end``, ``val`` if
    < ``val_end``, else ``test`` (UTC). Events within ``buffer_days`` *before*
    a boundary are excluded (sequences straddling a boundary). Label-conflict
    events are excluded from the splits in ``exclude_label_conflicts_from``.
    Candidate duplicate pairs in different splits are both excluded.
    ``site_seen_in_train`` marks val/test/ood events whose source-site cell
    contains a training event of the same class (for seen vs new site reporting).

    Returns:
        One row per event: identifiers, times, labels, ``role``, ``class``,
        ``split``, ``reason``, ``site_cell``, ``site_seen_in_train``,
        ``label_conflict``.
    """
    frames = []
    for collection, metadata in metadata_by_collection.items():
        events = event_table(metadata, collection=collection,
                             label_column=label_columns[collection])
        frames.append(assign_roles(events, class_map=class_map if collection == "comcat" else {},
                                   ood_labels=ood_labels.get(collection, ()),
                                   excluded_labels=excluded_labels.get(collection, ())))
    table = pd.concat(frames, ignore_index=True)

    t_train, t_val = pd.Timestamp(train_end, tz="UTC"), pd.Timestamp(val_end, tz="UTC")
    if not t_train < t_val:
        raise ValueError("train_end must be before val_end")
    buffer = pd.Timedelta(days=int(buffer_days))
    t = table["reference_time"]
    supervised = table["role"] == "supervised"

    split = pd.Series(np.where(t < t_train, "train", np.where(t < t_val, "val", "test")),
                      index=table.index)
    reason = pd.Series("", index=table.index)
    split[table["role"] == "ood"] = "ood"
    split[table["role"] == "excluded"] = "excluded"
    reason[table["role"] == "excluded"] = "excluded_label"

    in_buffer = supervised & (((t >= t_train - buffer) & (t < t_train))
                              | ((t >= t_val - buffer) & (t < t_val)))
    split[in_buffer], reason[in_buffer] = "excluded", "boundary_buffer"

    table["label_conflict"] = label_conflicts(table)
    conflict = table["label_conflict"] & split.isin(list(exclude_label_conflicts_from))
    split[conflict], reason[conflict] = "excluded", "label_conflict"

    comcat = metadata_by_collection.get("comcat")
    if comcat is not None and "source_origin_time" in comcat:
        pairs = possible_duplicate_events(comcat, max_dt_s=duplicate_max_dt_s,
                                          max_dist_deg=duplicate_max_dist_deg)
        where = dict(zip(table["event_id"], split))
        for a, b in zip(pairs["event_a"], pairs["event_b"]):
            if where.get(a) != where.get(b) and {where.get(a), where.get(b)} <= {"train", "val", "test"}:
                both = table["event_id"].isin([a, b])
                split[both], reason[both] = "excluded", "duplicate_straddles_split"

    table["split"], table["reason"] = split, reason
    table["site_cell"] = site_cells(table, site_cell_deg)
    train_sites = set(zip(table.loc[table["split"] == "train", "site_cell"],
                          table.loc[table["split"] == "train", "class"]))
    seen = [(cell, cls) in train_sites for cell, cls in zip(table["site_cell"], table["class"])]
    table["site_seen_in_train"] = pd.Series(seen, index=table.index).where(
        table["split"].isin(["val", "test"]) & table["site_cell"].notna())
    columns = ["event_id", "collection", "label_raw", "source_type", "class", "role", "split",
               "reason", "reference_time", "year", "origin_time", "latitude", "longitude",
               "n_stations", "site_cell", "site_seen_in_train", "label_conflict"]
    return table[columns].sort_values(["collection", "event_id"]).reset_index(drop=True)


def load_split_table(path: str | Path) -> pd.DataFrame:
    """Read a split table written by ``scripts/prepare_data.py`` with explicit dtypes."""
    table = pd.read_csv(path, dtype={"event_id": str, "collection": str, "label_raw": str,
                                     "source_type": str, "class": str, "role": str, "split": str,
                                     "reason": str, "site_cell": str, "site_seen_in_train": str},
                        keep_default_na=False, na_values=[""])
    table["site_seen_in_train"] = table["site_seen_in_train"].map({"True": True, "False": False})
    table["label_conflict"] = table["label_conflict"].astype(bool)
    table["reason"] = table["reason"].fillna("")
    for column in ("reference_time", "origin_time"):
        table[column] = pd.to_datetime(table[column], utc=True, format="ISO8601")
    return table


def split_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Counts per split/class/collection and exclusion reasons (JSON-friendly)."""
    sup = table[table["split"].isin(["train", "val", "test"])]
    counts = pd.crosstab(sup["split"], sup["class"])
    return {
        "events_total": int(len(table)),
        "supervised": {s: {c: int(counts.loc[s, c]) for c in counts.columns}
                       for s in ["train", "val", "test"] if s in counts.index},
        "anthropogenic_share": {s: round(float(counts.loc[s, "anthropogenic"] / counts.loc[s].sum()), 4)
                                for s in ["train", "val", "test"] if s in counts.index
                                and "anthropogenic" in counts.columns},
        "ood": {f"{c}/{lbl}": int(n) for (c, lbl), n in
                table[table["split"] == "ood"].groupby(["collection", "label_raw"]).size().items()},
        "excluded": {r: int(n) for r, n in
                     table.loc[table["split"] == "excluded", "reason"].value_counts().items()},
        "test_sites_seen_in_train": {
            str(cls): {"seen": int((g["site_seen_in_train"] == True).sum()),  # noqa: E712
                       "new": int((g["site_seen_in_train"] == False).sum())}  # noqa: E712
            for cls, g in table[table["split"] == "test"].groupby("class")},
    }
