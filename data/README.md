# data/

Default location of the project's data root. **Nothing in this directory is
committed** except this README and `.gitkeep` placeholders (see `.gitignore`).

Dataset (Phase 1): **Curated Pacific Northwest AI-ready Seismic Dataset (PNW)**,
Ni et al. (2023), *Seismica* 2(1), doi:10.26443/seismica.v2i1.368 — CC BY 4.0.
See `PROJECT_CONTEXT.md` → *DATASET SELECTED*.

## Where the data lives — configurable, never hard-coded

The data root is resolved at run time (first match wins):

1. command-line flag `--data-root PATH` (all data scripts accept it),
2. environment variable `SEISMIC_DATA_ROOT`,
3. `paths.data_root` in `configs/config.yaml` (default `data`, i.e. this directory).

Relative paths are resolved against the project root, not the working
directory. Outputs follow the same rule with `--output-root` /
`SEISMIC_OUTPUT_ROOT` / `paths.output_root`. Implementation: `src/seismic_ml/paths.py`.

### Local machine

```bash
python scripts/download_data.py            # -> ./data/raw/pnw/
python scripts/inspect_data.py             # -> ./data/processed/dataset_manifest.json
python scripts/prepare_data.py             # -> ./data/processed/splits/ + ./data/processed/pnw/
```

or keep the data elsewhere (e.g. an external disk):

```bash
python scripts/download_data.py --data-root /path/to/seismic-data
```

### Google Colab with Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
import os
os.environ["SEISMIC_DATA_ROOT"] = "/content/drive/MyDrive/seismic-data"
!python scripts/download_data.py
```

The notebooks read the same variable (see their setup cell).

## Layout under the data root

| Directory | Content | Rules |
|---|---|---|
| `raw/pnw/` | Files exactly as obtained + `PROVENANCE.json` + `subsets/` | Immutable. Never edit in place. |
| `interim/` | Inspection outputs (quality report, per-trace waveform metrics) | Reproducible by `scripts/inspect_data.py`. |
| `processed/` | `dataset_manifest.json`; `splits/s1_temporal/` (split table + summary); `pnw/<collection>/` (preprocessed windows) | Reproducible by `scripts/inspect_data.py` / `scripts/prepare_data.py`. |

`raw/pnw/` after `scripts/download_data.py`:

| File | What | Size |
|---|---|---|
| `comcat_metadata.csv` | full ComCat metadata (178,939 traces, 46,054 events) | 50.0 MB |
| `exotic_metadata.csv` | full exotic-event metadata (9,267 traces, 5,657 events) | 1.5 MB |
| `subsets/comcat_exploration_subset.hdf5` | event-complete waveform subset (1,221 events, 4,019 traces) | ~1.45 GB |
| `subsets/exotic_exploration_subset.hdf5` | event-complete waveform subset (732 events, 1,133 traces) | ~0.49 GB |
| `subsets/*_exploration_selection.csv` | the selected `event_id` / label / `trace_name` rows | small |
| `PROVENANCE.json` | URLs, server ETags, SHA-256, selection parameters (relative paths only) | small |

The full waveform files (`comcat_waveforms.hdf5`, 67.3 GB; `exotic_waveforms.hdf5`,
4.1 GB) are **not** downloaded. Subsets are extracted from them with HTTP byte-range
requests and keep the original SeisBench layout, so every `trace_name` in the
metadata stays valid. If you place a full waveform file in `raw/pnw/`, the loader
uses it automatically instead of the subset.

## Processed data (Phase 2)

| Path (under the data root) | Content |
|---|---|
| `processed/splits/s1_temporal/events.csv` | one row per event (all collections): `split` (train/val/test/ood/excluded), `class`, `reason`, `site_seen_in_train`, … |
| `processed/splits/s1_temporal/summary.json` | counts per split/class, exclusions, OOD sets, parameters, input checksums |
| `processed/pnw/<collection>/waveforms.hdf5` | `/waveforms` float32 `[n, 3, 3000]` (E, N, Z; 50 Hz; P − 10 s … P + 50 s), `/channel_mask` bool `[n, 3]` |
| `processed/pnw/<collection>/traces.csv` | one row per kept trace (row index, event, station, coordinates, picks, flags) |
| `processed/pnw/<collection>/dropped_traces.csv` | traces that could not be processed, with the reason |
| `processed/pnw/<collection>/preprocessing.json` | exact configuration, input paths (relative) + SHA-256, counts, versions |

Read with `seismic_ml.data.processed.ProcessedEventDataset` (split table via
`seismic_ml.data.splits.load_split_table`) and batch with
`seismic_ml.data.loader.collate_events`.

## Policy

- Never commit data. Never modify files under `raw/`.
- Record provenance for every file (done automatically in `PROVENANCE.json`).
- Split assignments (Phase 2) are saved as event-ID lists — never trace lists — and
  must follow `PROJECT_CONTEXT.md` §12 and *PROPOSED DATA SPLIT*.
- Do not place synthetic or fabricated data here (toy data lives only in `tests/`).
- Never write machine-specific absolute paths into manifests or configs; store paths
  relative to the data root.
