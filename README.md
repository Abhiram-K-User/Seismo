# Uncertainty-Aware Spatio-Temporal GNN for Seismic Event Classification

> **Status: Phase 3 (CNN baseline) implemented.**
> The dataset is selected and characterised, the leakage-aware train/val/test
> split is built, and Baseline 1 (1D Dilated Residual CNN / TCN with station pooling and Softmax)
> is fully implemented with PyTorch DataLoader, Trainer, and classification metrics.

## Motivation

Seismic event classifiers can perform well on the event types they were
trained on and still be **confidently wrong** on unfamiliar events: a source
type never seen in training, a new region, or a degraded station network.
For monitoring, an honest "uncertain" is more useful than a confident error.

## Problem

1. **Classification:** distinguish natural tectonic earthquakes from
   anthropogenic seismic events (e.g. quarry blasts) using multi-station,
   three-component waveforms.
2. **Uncertainty:** produce uncertainty estimates that are high for
   unfamiliar / out-of-distribution (OOD) events.

The research question is whether combining CNN feature extraction, a graph of
recording stations, spatio-temporal graph neural networks and Evidential Deep
Learning (EDL) improves both — tested against progressively stronger
baselines and ablations. These are hypotheses, not established results.

## Dataset

**Curated Pacific Northwest AI-ready Seismic Dataset (PNW)** — Ni et al. (2023),
*Seismica* 2(1), doi:10.26443/seismica.v2i1.368, CC BY 4.0
([github.com/niyiyu/PNW-ML](https://github.com/niyiyu/PNW-ML)).

- ComCat collection: 46,054 events / 178,939 three-component traces from 639
  stations (2002–2022); analyst labels include earthquakes (39,366 events) and
  probable/confirmed explosions and quarry blasts (6,396 events).
- Exotic collection: 5,657 events (surface events, sonic booms, thunder, one
  plane crash) — candidate out-of-distribution sources from the same network.
- Important caveats (station counts per event, zero-filled components,
  confounders, window alignment): `PROJECT_CONTEXT.md` → *DATA EXPLORATION FINDINGS*.

## Planned architecture

```text
Multi-station 3-channel waveforms
  → signal preprocessing
  → CNN / time-frequency encoder          (one embedding per station)
  → dynamic seismic graph                 (nodes = stations)
  → graph attention network
  → spatio-temporal fusion
  → evidential classifier → Dirichlet distribution
  → class prediction + uncertainty
```

## Current implementation status

| Component | Status |
|---|---|
| Repository structure, configs, docs | ✅ |
| Data schemas (`Station`, `WaveformSample`, `StationObservation`, `SeismicEvent`, `EventBatch`) | ✅ |
| Config loading, configurable data/output locations, logging, seeding | ✅ |
| Dataset acquisition (`scripts/download_data.py`) + provenance | ✅ Phase 1 |
| Event loading into schemas, `SeismicEventDataset` | ✅ Phase 1 |
| Dataset manifest + data-quality report (`scripts/inspect_data.py`) | ✅ Phase 1 |
| Exploration notebook (`notebooks/01_data_exploration.ipynb`) | ✅ Phase 1 |
| Preprocessing (1–16 Hz, P-aligned 60 s @ 50 Hz, station normalisation), channel masks | ✅ Phase 2 |
| S1 temporal split (all 51,711 events), processed store, `collate_events` → `EventBatch` | ✅ Phase 2 |
| Preprocessing justification notebook (`notebooks/02_waveform_analysis.ipynb`) | ✅ Phase 2 |
| DataLoader (`create_dataloader`), training pipeline, Trainer | ✅ Phase 3 |
| CNN encoder (`CNNEncoder`), Baseline 1 (`CNNBaseline`) | ✅ Phase 3 |
| Training & evaluation entry points (`scripts/train.py`, `scripts/evaluate.py`) | ✅ Phase 3 |
| Classification metrics (`accuracy`, `macro_f1`, `confusion_matrix`) | ✅ Phase 3 |
| GAT, STGNN, EDL head, graph construction | ❌ interfaces only |
| OOD metrics, evidential loss | ❌ interfaces only |

Unimplemented interfaces raise `NotYetImplementedError`, which names the
phase in which they are planned.

## Repository structure

```text
configs/            YAML configuration (config.yaml entry + data/model/training)
data/               default data root: raw/ interim/ processed/ (never committed)
notebooks/          exploration notebooks
scripts/            download_data.py, inspect_data.py (Phase 1); prepare_data.py (Phase 2);
                    train.py, evaluate.py (Phase 3); run_notebooks.py
src/seismic_ml/     package: paths, data, preprocessing, features, graph, models,
                    uncertainty, training, evaluation, utils
tests/              pytest suite (toy fixtures only; no real data needed)
experiments/        experiment records (none yet)
outputs/            default output root (empty)
PROJECT_CONTEXT.md  full project context, rules, decisions and state
```

## Roadmap

| Phase | Content |
|---|---|
| 0 | Project initialization ✅ |
| 1 | Dataset selection + data exploration ✅ |
| 2 | Waveform preprocessing ✅ |
| 3 | CNN baseline ✅ |
| 4 | Time-frequency experiments |
| 5 | Spatial graph construction |
| 6 | GAT |
| 7 | Spatio-temporal GNN |
| 8 | Evidential Deep Learning |
| 9 | OOD + uncertainty evaluation |
| 10 | Ablation + robustness experiments |
| 11 | Final experiments + documentation |

## Setup

Requires Python ≥ 3.11. Install PyTorch for your platform/CUDA version first
(see pytorch.org) if it is not already available.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev,notebooks]"
python -m pytest
```

## Getting the data

Data locations are never hard-coded. The data root comes from `--data-root`,
then `$SEISMIC_DATA_ROOT`, then `paths.data_root` in `configs/config.yaml`
(default `./data`). Raw data stays outside version control.

```bash
python scripts/download_data.py      # metadata (~52 MB) + waveform exploration subsets (~1.9 GB)
python scripts/inspect_data.py       # writes <data_root>/processed/dataset_manifest.json
python scripts/prepare_data.py       # S1 split + preprocessed windows under <data_root>/processed/
```

Only the Phase 1 exploration subsets are downloaded by default; the pipeline
runs unchanged on the full waveform files once they are placed in
`<data_root>/raw/pnw/`.

Google Colab with Drive: mount Drive, then
`os.environ["SEISMIC_DATA_ROOT"] = "/content/drive/MyDrive/seismic-data"` (or pass
`--data-root`). Details: [`data/README.md`](data/README.md).

## Contributing

Read [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) first. It defines the
development phases, data-leakage rules, evaluation plan, decision log and
current project state, and it applies to human and AI contributors alike.

## License

Code: not chosen yet. Data: PNW dataset under CC BY 4.0 — cite Ni et al. (2023).
