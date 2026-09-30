"""Prepare the dataset: build the S1 split and preprocess the locally available waveforms.

Inputs come from <data_root>/raw/<dataset>/ (see scripts/download_data.py);
the data root is resolved as --data-root > $SEISMIC_DATA_ROOT > configs/config.yaml.
Outputs (relative to the data root; raw data are never modified):

    processed/splits/s1_temporal/events.csv      one row per event: split, class, reason, ...
    processed/splits/s1_temporal/summary.json    counts per split / class / exclusion reason
    processed/<dataset>/<collection>/            waveforms.hdf5, traces.csv, dropped_traces.csv,
                                                 preprocessing.json

The split uses the full metadata. Preprocessing covers every event whose
traces are all available locally (the exploration subsets, or the full files
if they have been placed in the raw directory).

Usage:
    python scripts/prepare_data.py [--data-root PATH] [--splits-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from seismic_ml.config import load_config
from seismic_ml.data.acquisition import sha256_file, utc_now_iso
from seismic_ml.data.loader import collection_config, load_collection_metadata, waveform_path
from seismic_ml.data.processed import write_info, write_processed_collection
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore
from seismic_ml.data.splits import split_summary, temporal_split
from seismic_ml.paths import project_root_from_config, relative_to_root, resolve_paths
from seismic_ml.utils.logging import configure_logging
from seismic_ml.utils.reproducibility import get_environment_info

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--data-root", default=None,
                        help="Data root (overrides $SEISMIC_DATA_ROOT and the config).")
    parser.add_argument("--collections", nargs="+", default=None,
                        help="Collections to preprocess (default: all configured).")
    parser.add_argument("--splits-only", action="store_true",
                        help="Only build the split table (no waveform preprocessing).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    config = load_config(args.config)
    paths = resolve_paths(config, project_root=project_root_from_config(args.config),
                          data_root=args.data_root)
    data_cfg = config["data"]
    dataset = data_cfg["dataset"]
    raw_dir = paths.dataset_raw_dir(data_cfg)
    collections = list(dataset["collections"])
    metadata = {c: load_collection_metadata(raw_dir, data_cfg, c) for c in collections}

    # --- S1 temporal split (metadata only) --------------------------------------------------
    split_cfg, labels = data_cfg["split"], data_cfg["labels"]
    if split_cfg["strategy"] != "temporal":
        raise ValueError(f"unsupported split strategy {split_cfg['strategy']!r}")
    table = temporal_split(
        metadata,
        label_columns={c: collection_config(data_cfg, c)["label_column"] for c in collections},
        class_map=labels["class_map"], ood_labels=labels["ood_labels"],
        excluded_labels=labels["excluded_labels"],
        train_end=split_cfg["train_end"], val_end=split_cfg["val_end"],
        buffer_days=split_cfg["buffer_days"], site_cell_deg=split_cfg["site_cell_deg"],
        exclude_label_conflicts_from=split_cfg["exclude_label_conflicts_from"],
        duplicate_max_dt_s=split_cfg["duplicate_max_dt_s"],
        duplicate_max_dist_deg=split_cfg["duplicate_max_dist_deg"],
    )
    split_dir = paths.processed / "splits" / "s1_temporal"
    split_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(split_dir / "events.csv", index=False)
    summary = {"strategy": "S1 temporal", "parameters": split_cfg, "generated_utc": utc_now_iso(),
               "inputs": {c: {"metadata": relative_to_root(raw_dir / collection_config(data_cfg, c)
                                                            ["metadata_file"], paths.data_root),
                              "sha256": sha256_file(raw_dir / collection_config(data_cfg, c)
                                                    ["metadata_file"])} for c in collections},
               **split_summary(table)}
    (split_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n",
                                            encoding="utf-8")
    logger.info("S1 split: %s", json.dumps(summary["supervised"]))
    logger.info("S1 excluded: %s | ood: %s", summary["excluded"], summary["ood"])
    if args.splits_only:
        return 0

    # --- waveform preprocessing (locally available events) ------------------------------------
    for collection in args.collections or collections:
        spec = collection_config(data_cfg, collection)
        source = waveform_path(raw_dir, data_cfg, collection)
        dest = paths.processed / dataset["name"] / collection
        logger.info("%s: preprocessing events available in %s", collection,
                    relative_to_root(source, paths.data_root))
        with SeisBenchWaveformStore(source) as store:
            step = [0]

            def progress(done: int, total: int, _c: str = collection) -> None:
                if done == total or done >= step[0]:
                    logger.info("%s: %d/%d events", _c, done, total)
                    step[0] = done + max(1, total // 10)

            counts = write_processed_collection(dest, metadata[collection], store,
                                                label_column=spec["label_column"],
                                                config=data_cfg["preprocessing"], progress=progress)
        write_info(dest, {
            "generated_utc": utc_now_iso(), "collection": collection,
            "preprocessing": data_cfg["preprocessing"],
            "inputs": {"waveforms": relative_to_root(source, paths.data_root),
                       "waveforms_sha256": sha256_file(source),
                       "metadata": relative_to_root(raw_dir / spec["metadata_file"], paths.data_root)},
            "split_table": relative_to_root(split_dir / "events.csv", paths.data_root),
            "counts": counts, "environment": get_environment_info(),
        })
        logger.info("%s: %s", collection, json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
