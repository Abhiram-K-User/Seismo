"""Dataset inspection: catalogue statistics and data-quality measurements.

Phase 1 tooling for *understanding* the data. Every function here measures;
none of them filters, repairs or transforms the data (that is Phase 2).
Shared by ``scripts/inspect_data.py``, the manifest and the exploration notebook.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from seismic_ml.data.seisbench_format import SeisBenchWaveformStore, parse_trace_name

#: Bins used throughout the project for "stations per event".
STATION_COUNT_BINS: tuple[str, ...] = ("1", "2", "3", "4+")


def _bin_station_counts(counts: pd.Series) -> pd.Series:
    return pd.cut(counts, bins=[0, 1, 2, 3, np.inf], labels=list(STATION_COUNT_BINS))


def stations_per_event(metadata: pd.DataFrame) -> pd.Series:
    """Number of traces (= distinct stations, verified unique) per event."""
    return metadata.groupby("event_id").size().rename("n_stations")


def station_count_table(metadata: pd.DataFrame, label_column: str | None = None) -> pd.DataFrame:
    """Events per stations-per-event bin (1, 2, 3, 4+), optionally split by label.

    Returns:
        Rows = labels (or ``"all"``), columns = bins, values = event counts.
    """
    counts = stations_per_event(metadata)
    bins = _bin_station_counts(counts)
    if label_column is None:
        table = bins.value_counts().reindex(list(STATION_COUNT_BINS), fill_value=0).to_frame("all").T
    else:
        labels = metadata.groupby("event_id")[label_column].first()
        table = pd.crosstab(labels, bins).reindex(columns=list(STATION_COUNT_BINS), fill_value=0)
    table.columns = [str(c) for c in table.columns]
    return table


def label_counts(metadata: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """Events and traces per label value (missing labels counted as ``"<NA>"``)."""
    labels = metadata[label_column].astype("string").fillna("<NA>")
    traces = labels.value_counts().rename("traces")
    events = labels.groupby(metadata["event_id"]).first().value_counts().rename("events")
    return pd.concat([events, traces], axis=1).fillna(0).astype(int).sort_values("events",
                                                                               ascending=False)


def label_agreement(metadata: pd.DataFrame, column_a: str, column_b: str) -> pd.DataFrame:
    """Event-level cross-tabulation of two label columns."""
    events = metadata.groupby("event_id")[[column_a, column_b]].first()
    return pd.crosstab(events[column_a], events[column_b])


def missing_components(metadata: pd.DataFrame, component_order: Sequence[str]) -> pd.Series:
    """Per-trace string of components whose SNR is NaN (e.g. ``"EN"``; ``""`` = none).

    The dataset authors fill absent components with zeros and their SNR with
    NaN, so NaN SNR identifies the missing component(s).
    """
    def which(text: Any) -> str:
        if text is None or pd.isna(text):
            return "".join(component_order)
        values = str(text).split("|")
        return "".join(c for c, v in zip(component_order, values) if v.strip().lower() == "nan")

    return metadata["trace_snr_db"].map(which).rename("missing_components")


def metadata_quality(metadata: pd.DataFrame, *, origin_offset_s: float | None = None) -> dict[str, Any]:
    """Counts of potential metadata problems (nothing is removed).

    Args:
        origin_offset_s: Expected ``trace_start_time - source_origin_time`` in
            seconds (``-50`` for PNW ComCat); ``None`` skips the check.
    """
    report: dict[str, Any] = {
        "rows": int(len(metadata)),
        "missing_values": {c: int(n) for c, n in metadata.isna().sum().items() if n},
        "duplicate_trace_name": int(metadata["trace_name"].duplicated().sum()),
        "duplicate_event_station": int(metadata.duplicated(
            ["event_id", "station_network_code", "station_code"]).sum()),
        "sampling_rates_hz": sorted(float(v) for v in metadata["trace_sampling_rate_hz"].unique()),
        "samples_per_trace": sorted(int(v) for v in metadata["trace_name"].astype(str)
                                    .str.rsplit(":", n=1).str[-1].astype(int).unique()),
    }
    p, s = metadata["trace_P_arrival_sample"], metadata["trace_S_arrival_sample"]
    report["picks"] = {
        "p_missing": int(p.isna().sum()), "s_missing": int(s.isna().sum()),
        "s_before_p": int((s < p).sum()), "s_equals_p": int((s == p).sum()),
    }
    sites = metadata.assign(sid=metadata["station_network_code"] + "." + metadata["station_code"])
    n_coords = sites.groupby("sid")[["station_latitude_deg", "station_longitude_deg"]].nunique()
    report["sites_with_multiple_coordinates"] = int((n_coords.max(axis=1) > 1).sum())
    for flag in ("trace_missing_channel", "trace_has_offset"):
        if flag in metadata:
            report[flag] = {str(k): int(v) for k, v in metadata[flag].value_counts().items()}
    snr = metadata["trace_snr_db"].astype("string").str.split("|", expand=True).apply(
        pd.to_numeric, errors="coerce")
    report["snr_db"] = {
        "all_components_nan": int(snr.isna().all(axis=1).sum()),
        "max_component_below_0db": int((snr.max(axis=1) < 0).sum()),
    }
    if origin_offset_s is not None and "source_origin_time" in metadata:
        start = pd.to_datetime(metadata["trace_start_time"], utc=True, format="ISO8601")
        origin = pd.to_datetime(metadata["source_origin_time"], utc=True, format="ISO8601")
        offset = (start - origin).dt.total_seconds()
        report["start_minus_origin_s"] = {
            "expected": origin_offset_s,
            "n_deviating_gt_0p01s": int(((offset - origin_offset_s).abs() > 0.01).sum()),
            "min": float(offset.min()), "max": float(offset.max()),
        }
    return report


def possible_duplicate_events(metadata: pd.DataFrame, *, max_dt_s: float = 2.0,
                              max_dist_deg: float = 0.05) -> pd.DataFrame:
    """Pairs of *different* event IDs with origins close in time and space.

    A cheap screen (consecutive events in time order), not a definitive
    de-duplication; results need manual review.
    """
    events = (metadata.groupby("event_id")[["source_origin_time", "source_latitude_deg",
                                            "source_longitude_deg"]].first().dropna())
    events["t"] = pd.to_datetime(events["source_origin_time"], utc=True, format="ISO8601")
    events = events.sort_values("t")
    dt = events["t"].diff().dt.total_seconds()
    dlat = events["source_latitude_deg"].diff().abs()
    dlon = events["source_longitude_deg"].diff().abs()
    close = (dt <= max_dt_s) & (dlat <= max_dist_deg) & (dlon <= max_dist_deg)
    idx = np.flatnonzero(close.to_numpy())
    ids = events.index.to_numpy()
    return pd.DataFrame({"event_a": ids[idx - 1], "event_b": ids[idx], "dt_s": dt.iloc[idx].to_numpy()})


def waveform_metrics(data: np.ndarray, component_order: Sequence[str]) -> dict[str, Any]:
    """Per-trace waveform measurements (no modification of ``data``).

    Returns, per component: whether it is all zeros, flat (constant, non-zero),
    NaN count, peak |amplitude|, standard deviation, number of samples at the
    peak magnitude (many samples at exactly the peak suggest clipping), and the
    longest run of exact zeros.
    """
    out: dict[str, Any] = {"shape": tuple(int(v) for v in data.shape), "dtype": str(data.dtype)}
    for i, comp in enumerate(component_order):
        x = np.asarray(data[i])
        finite = x[np.isfinite(x)]
        peak = float(np.abs(finite).max()) if finite.size else float("nan")
        zero = x == 0
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, zero.astype(np.int8), 0])))[::2]
        out[comp] = {
            "all_zero": bool(zero.all()),
            "flat_nonzero": bool(finite.size and not zero.all() and np.ptp(finite) == 0),
            "n_nan": int(np.isnan(x).sum()),
            "peak_abs": peak,
            "std": float(finite.std()) if finite.size else float("nan"),
            "n_at_peak": int((np.abs(finite) >= peak).sum()) if peak > 0 else 0,
            "longest_zero_run": int(runs.max()) if runs.size else 0,
        }
    return out


def waveform_quality_table(
    metadata: pd.DataFrame,
    store: SeisBenchWaveformStore,
    trace_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """One row per (trace, component) with :func:`waveform_metrics` values.

    Only traces present in ``store`` are measured.
    """
    names = list(trace_names) if trace_names is not None else [
        t for t in metadata["trace_name"] if t in store]
    records = []
    for name in names:
        metrics = waveform_metrics(store.get(name), store.component_order)
        for comp in store.component_order:
            records.append({"trace_name": name, "component": comp, **metrics[comp]})
    return pd.DataFrame.from_records(records)
