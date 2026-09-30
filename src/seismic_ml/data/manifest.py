"""Machine-readable dataset manifest (``<processed>/dataset_manifest.json``).

Every number in the manifest is computed from the files on disk; nothing is
typed in by hand. Paths are stored relative to the data root so the manifest
is identical on every machine (laptop, Colab + Drive, cluster).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from seismic_ml.data.inspection import label_counts, missing_components, station_count_table

MANIFEST_VERSION = 1


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and containers to JSON types."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if np.isnan(value) else float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if value is pd.NA:
        return None
    return value


def collection_summary(
    metadata: pd.DataFrame,
    *,
    label_columns: Sequence[str],
    primary_label_column: str,
    component_order: Sequence[str],
) -> dict[str, Any]:
    """Measured summary of one metadata collection (e.g. ComCat or exotic events)."""
    sid = metadata["station_network_code"] + "." + metadata["station_code"]
    per_event = metadata.groupby("event_id").size()
    samples = metadata["trace_name"].astype(str).str.rsplit(":", n=1).str[-1].astype(int)
    summary: dict[str, Any] = {
        "num_events": int(metadata["event_id"].nunique()),
        "num_traces": int(len(metadata)),
        "num_stations": int(sid.nunique()),
        "station_ids": sorted(sid.dropna().unique().tolist()),
        "num_networks": int(metadata["station_network_code"].nunique()),
        "networks": sorted(metadata["station_network_code"].dropna().unique().tolist()),
        "component_order": "".join(component_order),
        "channel_band_codes": metadata["station_channel_code"].value_counts().to_dict(),
        "sampling_rates_hz": sorted(float(v) for v in metadata["trace_sampling_rate_hz"].unique()),
        "samples_per_trace": sorted(int(v) for v in samples.unique()),
        "missing_component_patterns": missing_components(metadata, component_order)
        .replace("", "none").value_counts().to_dict(),
        "stations_per_event": {
            "mean": float(per_event.mean()), "median": float(per_event.median()),
            "max": int(per_event.max()),
            "bins": station_count_table(metadata).iloc[0].to_dict(),
            "bins_by_label": station_count_table(metadata, primary_label_column).to_dict("index"),
        },
        "labels": {col: label_counts(metadata, col).to_dict("index") for col in label_columns
                   if col in metadata},
        "trace_start_time_range": [str(metadata["trace_start_time"].min()),
                                   str(metadata["trace_start_time"].max())],
        "station_extent_deg": {
            "lat": [float(metadata["station_latitude_deg"].min()),
                    float(metadata["station_latitude_deg"].max())],
            "lon": [float(metadata["station_longitude_deg"].min()),
                    float(metadata["station_longitude_deg"].max())],
        },
    }
    if "source_latitude_deg" in metadata:
        summary["event_extent_deg"] = {
            "lat": [float(metadata["source_latitude_deg"].min()),
                    float(metadata["source_latitude_deg"].max())],
            "lon": [float(metadata["source_longitude_deg"].min()),
                    float(metadata["source_longitude_deg"].max())],
        }
        summary["origin_time_range"] = [str(metadata["source_origin_time"].min()),
                                        str(metadata["source_origin_time"].max())]
        mags = metadata.groupby("event_id")["preferred_source_magnitude"].first()
        summary["magnitude_range"] = [float(mags.min()), float(mags.max())]
    return summary


def build_manifest(
    *,
    dataset_config: Mapping[str, Any],
    collections: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    raw_dir_relative: str,
    notes: Sequence[str] = (),
    generated_utc: str,
) -> dict[str, Any]:
    """Assemble the manifest.

    Args:
        dataset_config: ``data.dataset`` section of the configuration.
        collections: ``{name: collection_summary(...)}``.
        provenance: Content of ``<raw>/<dataset>/PROVENANCE.json``.
        raw_dir_relative: Dataset raw directory relative to the data root.
        notes: Human-readable caveats (observed facts only).
    """
    files = [{"path": f"{raw_dir_relative}/{rec['path']}", "size": rec["size"],
              "sha256": rec["sha256"], "source_url": rec["url"], "source_etag": rec["etag"],
              "source_last_modified": rec["last_modified"]}
             for rec in provenance.get("files", {}).values()]
    subsets = {name: {"path": f"{raw_dir_relative}/{info['file']}",
                      "selection_path": f"{raw_dir_relative}/{info['selection_file']}",
                      "size": info["size"], "sha256": info["sha256"],
                      "num_events": info["n_events"], "num_traces": info["n_traces"],
                      "selection": info["selection"],
                      "source_url": info["extraction"].get("source_url"),
                      "source_etag": info["extraction"].get("source_etag")}
               for name, info in provenance.get("subsets", {}).items()}
    stations = {s for c in collections.values() for s in c["station_ids"]}
    return _jsonable({
        "manifest_version": MANIFEST_VERSION,
        "generated_utc": generated_utc,
        "dataset_name": dataset_config["name"],
        "full_name": dataset_config.get("full_name"),
        "version": dataset_config.get("version"),
        "license": dataset_config.get("license"),
        "citation": dataset_config.get("citation"),
        "data_source": dataset_config.get("source"),
        "format": dataset_config.get("format"),
        "num_events": sum(c["num_events"] for c in collections.values()),
        "num_traces": sum(c["num_traces"] for c in collections.values()),
        "num_stations": len(stations),
        "sampling_rates": sorted({r for c in collections.values() for r in c["sampling_rates_hz"]}),
        "channels": sorted({c["component_order"] for c in collections.values()}),
        "collections": dict(collections),
        "files": files,
        "waveform_subsets": subsets,
        "paths_relative_to": "data_root",
        "notes": list(notes),
    })


def write_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as indented UTF-8 JSON (parent directories created)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
