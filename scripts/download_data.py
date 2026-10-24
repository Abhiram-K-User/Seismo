"""Acquire the configured dataset: metadata files + event-complete waveform subsets.

Where data goes is resolved at run time (never hard-coded):
    --data-root  >  $SEISMIC_DATA_ROOT  >  paths.data_root in configs/config.yaml
Files are written under <data_root>/raw/<dataset.raw_subdir>/ and a
PROVENANCE.json (relative paths only) records URLs, ETags and SHA-256 sums.

Usage:
    python scripts/download_data.py                        # local ./data
    python scripts/download_data.py --data-root /content/drive/MyDrive/seismic-data
    python scripts/download_data.py --metadata-only

Re-running is safe: finished files are kept, interrupted subsets resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from seismic_ml.config import load_config
from seismic_ml.data.acquisition import download_file, sha256_file, utc_now_iso
from seismic_ml.data.pnw import load_metadata, select_events
from seismic_ml.data.seisbench_format import SeisBenchWaveformStore, extract_subset
from seismic_ml.paths import project_root_from_config, resolve_paths
from seismic_ml.utils.logging import configure_logging

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--data-root", default=None,
                        help="Data root (overrides $SEISMIC_DATA_ROOT and the config).")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Download metadata files only (no waveform subsets).")
    parser.add_argument("--full", action="store_true",
                        help="Download the complete remote HDF5 waveform files instead of the 2 GB exploration subset.")
    parser.add_argument("--scale", type=int, default=None,
                        help="Scale factor for events per label (e.g. 5 for 5x larger dataset).")
    parser.add_argument("--all-events", action="store_true",
                        help="Extract all available events across all labels via parallel range requests.")
    parser.add_argument("--collections", nargs="+", default=None,
                        help="Subset collections to extract (default: all configured).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel HTTP connections (default: from config).")
    return parser.parse_args(argv)


def _load_provenance(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}, "subsets": {}}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    config = load_config(args.config)
    paths = resolve_paths(config, project_root=project_root_from_config(args.config),
                          data_root=args.data_root)
    data_cfg = config["data"]
    dataset = data_cfg["dataset"]
    base_url = dataset["remote_base_url"]
    collections_cfg = dataset["collections"]
    raw_dir = paths.dataset_raw_dir(data_cfg)
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset %r -> raw dir %s", dataset["name"], raw_dir)

    provenance_path = raw_dir / "PROVENANCE.json"
    provenance = _load_provenance(provenance_path)
    provenance.update({"dataset": dataset["name"], "source": dataset["source"],
                       "license": dataset["license"], "citation": dataset["citation"]})

    for key, spec in collections_cfg.items():
        filename = spec["metadata_file"]
        record = download_file(base_url + filename, raw_dir / filename, relative_to=raw_dir)
        provenance["files"][record.path] = record.to_dict()
        logger.info("metadata %-7s %s (%d bytes, sha256 %s...)", key, record.path, record.size,
                    record.sha256[:12])
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    if args.metadata_only:
        return 0

    if args.full:
        collections = args.collections or list(collections_cfg.keys())
        for collection in collections:
            spec = collections_cfg[collection]
            waveform_filename = spec.get("waveform_file")
            if not waveform_filename:
                continue
            dest = raw_dir / waveform_filename
            logger.info("Downloading FULL %s waveforms (%s) to %s ...", collection, waveform_filename, dest)
            record = download_file(base_url + waveform_filename, dest, relative_to=raw_dir)
            provenance["files"][record.path] = record.to_dict()
            logger.info("FULL %s waveforms complete (%d bytes, sha256 %s...)", collection, record.size, record.sha256[:12])
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        return 0

    subset_cfg = data_cfg["dataset"]["exploration_subset"]
    workers = args.workers or subset_cfg.get("workers", 4)
    collections = args.collections or [c for c in ("comcat", "exotic") if c in subset_cfg]
    for collection in collections:
        spec = subset_cfg[collection]
        metadata = load_metadata(raw_dir / collections_cfg[collection]["metadata_file"])
        events_per_label = dict(spec["events_per_label"])
        if args.all_events:
            events_per_label = {k: -1 for k in events_per_label}
        elif args.scale and args.scale > 1:
            events_per_label = {
                k: (v * args.scale if v > 0 else v) for k, v in events_per_label.items()
            }

        event_ids = select_events(metadata, label_column=spec["label_column"],
                                  events_per_label=events_per_label,
                                  seed=subset_cfg["seed"])
        rows = metadata[metadata["event_id"].isin(set(event_ids))]
        subset_dir = raw_dir / "subsets"
        subset_dir.mkdir(exist_ok=True)
        selection_path = subset_dir / f"{collection}_exploration_selection.csv"
        rows[["event_id", spec["label_column"], "trace_name"]].to_csv(selection_path, index=False)
        dest = raw_dir / collections_cfg[collection]["subset_file"]
        logger.info("%s: %d events, %d traces selected (seed %d)", collection, len(event_ids),
                    len(rows), subset_cfg["seed"])

        wanted = set(rows["trace_name"])
        if dest.exists():
            with SeisBenchWaveformStore(dest) as store:
                have = store.available_trace_names()
            if have == wanted:
                logger.info("%s: subset already complete, skipping download", collection)
                summary = provenance["subsets"].get(collection, {}).get("extraction", {})
            else:
                raise RuntimeError(f"{dest} exists but holds a different trace selection; "
                                   "move it away to re-extract.")
        else:
            step = max(1, len(wanted) // 20)

            def progress(done: int, total: int, _name: str = collection) -> None:
                if done % step == 0 or done == total:
                    logger.info("%s: %d/%d traces (%.0f%%)", _name, done, total, 100 * done / total)

            summary = extract_subset(base_url + collections_cfg[collection]["waveform_file"],
                                     sorted(wanted), dest, workers=workers, progress=progress)

        provenance["subsets"][collection] = {
            "file": dest.relative_to(raw_dir).as_posix(),
            "selection_file": selection_path.relative_to(raw_dir).as_posix(),
            "size": dest.stat().st_size,
            "sha256": sha256_file(dest),
            "n_events": len(event_ids),
            "n_traces": len(wanted),
            "selection": {"seed": subset_cfg["seed"], "label_column": spec["label_column"],
                          "events_per_label": spec["events_per_label"],
                          "method": "uniform random events per label, all traces per event"},
            "extraction": summary,
            "completed_utc": utc_now_iso(),
        }
        provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        logger.info("%s: done -> %s", collection, provenance["subsets"][collection]["file"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
