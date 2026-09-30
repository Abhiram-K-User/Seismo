"""Inspect the acquired dataset: write the dataset manifest and a data-quality report.

Outputs (under the resolved data root; nothing is written into the raw data):
    <data_root>/processed/dataset_manifest.json          measured dataset description
    <data_root>/interim/<dataset>_quality_report.json    metadata + waveform issue counts
    <data_root>/interim/<dataset>_<collection>_waveform_metrics.csv   per-trace/component values

Usage:
    python scripts/inspect_data.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from seismic_ml.config import load_config
from seismic_ml.data.acquisition import utc_now_iso
from seismic_ml.data.inspection import (
    label_agreement,
    metadata_quality,
    missing_components,
    possible_duplicate_events,
    waveform_quality_table,
)
from seismic_ml.data.loader import collection_config, load_collection_metadata, waveform_path
from seismic_ml.data.manifest import build_manifest, collection_summary, write_json
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore
from seismic_ml.paths import project_root_from_config, relative_to_root, resolve_paths
from seismic_ml.utils.logging import configure_logging

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

#: Expected ``trace_start_time - source_origin_time`` for ComCat traces (paper, §Data prep.).
COMCAT_START_OFFSET_S = -50.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--data-root", default=None,
                        help="Data root (overrides $SEISMIC_DATA_ROOT and the config).")
    return parser.parse_args(argv)


def waveform_summary(table: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, Any]:
    """Aggregate per-component waveform metrics into issue counts."""
    per_trace = table.groupby("trace_name").agg(
        zero_components=("all_zero", "sum"), any_nan=("n_nan", "max"),
        flat=("flat_nonzero", "max"), max_n_at_peak=("n_at_peak", "max"),
        max_zero_run=("longest_zero_run", "max"))
    per_trace = per_trace.join(metadata.set_index("trace_name")[
        [c for c in ("trace_missing_channel", "station_channel_code") if c in metadata]])
    out: dict[str, Any] = {
        "traces_measured": int(len(per_trace)),
        "traces_with_nan": int((per_trace["any_nan"] > 0).sum()),
        "traces_with_flat_nonzero_component": int(per_trace["flat"].sum()),
        "zero_components_per_trace": per_trace["zero_components"].value_counts().sort_index().to_dict(),
        "traces_with_>=10_samples_at_peak (clipping indicator)": int((per_trace["max_n_at_peak"] >= 10).sum()),
        "traces_with_zero_run_>=100_samples (on a non-missing component)": int(
            (table[~table["all_zero"]].groupby("trace_name")["longest_zero_run"].max() >= 100).sum()),
    }
    if "trace_missing_channel" in per_trace:
        agree = per_trace["zero_components"] == per_trace["trace_missing_channel"]
        out["missing_channel_flag_matches_zero_components"] = {
            "agree": int(agree.sum()), "disagree": int((~agree).sum())}
    peaks = table[~table["all_zero"]].merge(metadata[["trace_name", "station_channel_code"]],
                                            on="trace_name")
    out["peak_abs_counts_by_band"] = {
        band: {q: float(v) for q, v in grp["peak_abs"].quantile([0.05, 0.5, 0.95]).items()}
        for band, grp in peaks.groupby("station_channel_code")}
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    config = load_config(args.config)
    paths = resolve_paths(config, project_root=project_root_from_config(args.config),
                          data_root=args.data_root)
    data_cfg = config["data"]
    dataset = data_cfg["dataset"]
    raw_dir = paths.dataset_raw_dir(data_cfg)
    provenance = json.loads((raw_dir / "PROVENANCE.json").read_text(encoding="utf-8"))

    paths.interim.mkdir(parents=True, exist_ok=True)
    summaries, quality = {}, {}
    notes: list[str] = []
    for collection in dataset["collections"]:
        spec = collection_config(data_cfg, collection)
        meta = load_collection_metadata(raw_dir, data_cfg, collection)
        label_cols = [c for c in ("source_type", "source_type_pnsn_label") if c in meta]
        with SeisBenchWaveformStore(waveform_path(raw_dir, data_cfg, collection)) as store:
            order = store.component_order
            logger.info("%s: measuring waveforms in %s", collection, store.path.name)
            table = waveform_quality_table(meta, store)
        summaries[collection] = collection_summary(meta, label_columns=label_cols,
                                                   primary_label_column=spec["label_column"],
                                                   component_order=order)
        q: dict[str, Any] = {"metadata": metadata_quality(
            meta, origin_offset_s=COMCAT_START_OFFSET_S if "source_origin_time" in meta else None)}
        q["missing_components_by_band"] = pd.crosstab(
            meta["station_channel_code"], missing_components(meta, order).replace("", "none")
        ).to_dict("index")
        if len(label_cols) == 2:
            agreement = label_agreement(meta, *label_cols)
            q["label_agreement_events"] = agreement.to_dict("index")
        if "source_origin_time" in meta:
            dups = possible_duplicate_events(meta)
            q["possible_duplicate_event_pairs"] = {"count": int(len(dups)),
                                                   "examples": dups.head(10).to_dict("records")}
        q["waveforms"] = waveform_summary(table, meta)
        quality[collection] = q
        table.to_csv(paths.interim / f"{dataset['name']}_{collection}_waveform_metrics.csv",
                     index=False)

    cc = summaries.get("comcat", {})
    if cc:
        counts = ", ".join(f"{label}={info['events']}"
                           for label, info in cc["labels"]["source_type_pnsn_label"].items())
        notes.append("ComCat `source_type` merges PNSN labels; `source_type_pnsn_label` is the "
                     f"analyst label (events: {counts}).")
    notes += [
        "Waveforms are stored in counts (no instrument-response removal; per the dataset paper), "
        "dtype float64, component order ENZ, 100 Hz; the paper states missing components are "
        "zero-filled (checked on the subsets: see the quality report).",
        "Documented selection: ComCat traces exist only where both P and S were picked. Observed: "
        "explosion events have fewer stations per event than earthquakes (see stations_per_event).",
        "Exotic events have no origin time, location or magnitude and are mostly single-station.",
        "Waveform-level statistics describe the Phase 1 exploration subsets only (see waveform_subsets).",
    ]
    manifest = build_manifest(dataset_config=dataset, collections=summaries, provenance=provenance,
                              raw_dir_relative=relative_to_root(raw_dir, paths.data_root),
                              notes=notes, generated_utc=utc_now_iso())
    write_json(manifest, paths.processed / "dataset_manifest.json")
    write_json({"generated_utc": manifest["generated_utc"], "dataset": dataset["name"],
                "collections": quality},
               paths.interim / f"{dataset['name']}_quality_report.json")
    logger.info("Wrote %s and quality report (%d events, %d traces, %d stations)",
                relative_to_root(paths.processed / "dataset_manifest.json", paths.data_root),
                manifest["num_events"], manifest["num_traces"], manifest["num_stations"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
