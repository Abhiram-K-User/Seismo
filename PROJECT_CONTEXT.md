# PROJECT_CONTEXT.md

> **This file is the persistent source of truth for the project.**
> It is written so that a developer or AI coding agent (Claude Code, Claude,
> Google Antigravity, or others) can continue the work months later without
> access to the original conversation. Read it completely before changing anything.
>
> **Quick orientation for agents**
> 1. Current phase and what exists → [§22 CURRENT PROJECT STATE](#22-current-project-state)
> 2. Rules you must follow → [§19 INSTRUCTIONS FOR FUTURE AI AGENTS](#19-instructions-for-future-ai-agents)
> 3. Decisions already made → [§21 DECISION LOG](#21-decision-log)
> 4. Per-phase procedure → [§15 Phase Completion Rules](#15-phase-completion-rules)

_Last updated: 2026-09-19, end of Phase 2 (waveform preprocessing)._

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Core Research Question](#2-core-research-question)
3. [Problem Definition](#3-problem-definition)
4. [Input Representation](#4-input-representation)
5. [Model Architecture](#5-model-architecture)
6. [Important Scientific Constraints](#6-important-scientific-constraints)
7. [Baselines](#7-baselines)
8. [Ablation Study Plan](#8-ablation-study-plan)
9. [Evaluation Plan](#9-evaluation-plan)
10. [Robustness Experiments](#10-robustness-experiments)
11. [Data Policy](#11-data-policy)
    - 11A. [DATASET SELECTED](#11a-dataset-selected)
    - 11B. [DATASET COMPARISON](#11b-dataset-comparison)
    - 11C. [DATA EXPLORATION FINDINGS](#11c-data-exploration-findings)
    - 11D. [DATA QUALITY ISSUES](#11d-data-quality-issues)
    - 11E. [OOD STRATEGY](#11e-ood-strategy)
    - 11F. [PROPOSED DATA SPLIT](#11f-proposed-data-split)
    - 11G. [PHASE 1 DECISIONS](#11g-phase-1-decisions)
    - 11H. [PREPROCESSING PIPELINE (PHASE 2)](#11h-preprocessing-pipeline-phase-2)
    - 11I. [S1 SPLIT AND PROCESSED DATA (PHASE 2)](#11i-s1-split-and-processed-data-phase-2)
    - 11J. [PHASE 2 DECISIONS](#11j-phase-2-decisions)
12. [Data Leakage Prevention](#12-data-leakage-prevention)
13. [Reproducibility](#13-reproducibility)
14. [Development Phases](#14-development-phases)
15. [Phase Completion Rules](#15-phase-completion-rules)
16. [Repository Map and Code Conventions](#16-repository-map-and-code-conventions)
17. [Open Questions and Known Risks](#17-open-questions-and-known-risks)
18. [RESEARCH HYPOTHESES](#18-research-hypotheses)
19. [INSTRUCTIONS FOR FUTURE AI AGENTS](#19-instructions-for-future-ai-agents)
20. [References (Starting Points)](#20-references-starting-points)
21. [DECISION LOG](#21-decision-log)
22. [CURRENT PROJECT STATE](#22-current-project-state)

---

## 1. Project Identity

```text
Project Name:
Uncertainty-Aware Spatio-Temporal Graph Neural Network
for Seismic Event Classification

Project Type:
Research-oriented ML project

Primary Task:
Natural earthquake vs anthropogenic seismic event classification

Secondary Task:
Reliable uncertainty estimation for unfamiliar/OOD events
```

| Item | Value |
|---|---|
| Repository directory | `seismic-event-classification/` |
| Python package (import name) | `seismic_ml` (in `src/seismic_ml/`) |
| Distribution name | `seismic-ml` (see `pyproject.toml`) |
| Python | ≥ 3.11 |
| Licence | Not chosen yet |

### Motivation

Traditional seismic event classifiers — hand-crafted discriminants as well as
deep networks with a softmax output — can perform well on the classes they
were trained on while being **overconfident on unfamiliar events** (an event
type absent from training, a new region, a degraded station network). In a
monitoring context a confident wrong answer is worse than an honest
"I don't know".

The project investigates whether combining

```text
CNN feature extraction
+
spatial graph modeling
+
temporal/spatial GNN reasoning
+
Evidential Deep Learning
```

can improve **both** classification and uncertainty estimation compared with
simpler baselines. The contribution of every component must be demonstrated
experimentally (§7, §8); none is assumed to help.

---

## 2. Core Research Question

> **Can a spatially aware spatio-temporal neural architecture combined with
> evidential uncertainty estimation accurately distinguish natural and
> anthropogenic seismic events while producing appropriately high uncertainty
> for unfamiliar seismic events?**

Secondary questions:

1. Does graph-based spatial information improve classification?
2. Does waveform similarity provide useful graph relationships?
3. Does dynamic graph construction outperform a static geographic graph?
4. Does GAT improve over non-graph baselines?
5. Does EDL improve uncertainty calibration?
6. Can EDL help identify OOD seismic events?
7. How does missing station data affect prediction and uncertainty?

These map to the hypotheses in §18.

---

## 3. Problem Definition

Conceptual classes:

```text
Known classes:
    earthquake
    anthropogenic event

Potential OOD classes:
    geothermal/fracture events
    volcanic events
    other unfamiliar seismic sources
```

**Important:** do NOT assume any of the potential OOD classes will actually be
present. Which unfamiliar event types exist depends on the dataset (Phase 1).
Dataset-dependent decisions must be recorded in §11 and the Decision Log.

**Phase 1 update (observed in the selected PNW dataset, §11E):** geothermal /
induced / fracture events are **not** labelled in this dataset. Available
unfamiliar sources are low-frequency events (`lf`, volcanic/glacial), surface
events (mass movements), sonic booms, thunder and one plane crash.

Working formulation (subject to revision after dataset selection):

- One ML sample = one **seismic event** recorded by a variable-size set of
  stations. Each station provides a location and a 3-component waveform.
- Output 1: a class prediction over the known classes (currently K = 2).
- Output 2: an uncertainty value that should be **high** for unfamiliar /
  OOD events and for poorly constrained inputs (e.g. few stations, noisy data).
- Class index mapping is configured in `configs/data.yaml`
  (`labels.class_names`; list order = class index). "Anthropogenic" may be a
  single class (e.g. quarry blasts) or an umbrella of several explosion types —
  to be decided from the dataset's actual labels.

---

## 4. Input Representation

```text
Station
    ├── location
    │     ├── latitude
    │     └── longitude
    │
    └── waveform
          ├── Z
          ├── N
          └── E
```

Expected conceptual tensor:

```text
[batch, stations, channels, time]   =   [B, N, C, T]
```

⚠️ **All dimensions are subject to change after dataset selection.** N (max
stations per event), T (window length × sampling rate) and even C (some
datasets label horizontals `1`/`2` instead of `N`/`E`, or have missing
components) are dataset-dependent. Events with fewer than N stations are
padded, and a boolean `station_mask [B, N]` marks real stations
(implemented as `seismic_ml.data.schemas.EventBatch`; provisional, Decision 003).

**Phase 1 measurements (PNW):** C = 3 stored as `E, N, Z`; 100 Hz everywhere;
stored T = 15,001 (ComCat) or 18,001 (exotic) samples; stations per event:
median 3, max 41 (ComCat); ~28 % of ComCat traces are vertical-only
(horizontals zero-filled). See §11C.

**Phase 2 model input (implemented, §11H):** `EventBatch.waveforms` float32
`[B, N, 3, 3000]` — components `E, N, Z`; 50 Hz; 60 s window per station from
P − 10 s to P + 50 s (P at sample 500); band-pass 1–16 Hz; one std-scale per
station over its live components. `station_mask [B, N]` marks real stations
(N = largest event in the batch) and `channel_mask [B, N, 3]` marks components
that hold real data (missing components are exact zeros).

Preprocessing **may** eventually include (no method is committed yet — choose
in Phase 2 once sampling characteristics are known):

- resampling
- detrending
- filtering (band-pass)
- normalization
- windowing
- signal quality checks
- optional P-wave alignment
- spectrogram / STFT generation (Phase 4)

Scientific notes to consider when choosing:

- **Normalization scope matters.** Relative amplitudes (e.g. P/S amplitude
  ratios, vertical vs horizontal energy, amplitude decay across stations) are
  classic physical discriminants between explosions and earthquakes.
  Per-channel normalization destroys inter-component ratios; per-station
  normalization destroys inter-station amplitude information.
- Zero-phase filtering can smear energy before the P onset.
- Any statistic fitted on data (e.g. global scaling) must use the training
  split only (§12).

---

## 5. Model Architecture

Overall intended pipeline:

```text
Multi-Station 3-Channel Waveforms
                │
                ▼
        Signal Preprocessing
                │
                ▼
     CNN / Time-Frequency Encoder
                │
                ▼
       Station-Level Embeddings
                │
                ▼
       Dynamic Seismic Graph
                │
                ▼
       Graph Attention Network
                │
                ▼
      Spatio-Temporal Fusion
                │
                ▼
       Evidential Classifier
                │
                ▼
         Dirichlet Distribution
                │
        ┌───────┴────────┐
        ▼                ▼
 Classification      Uncertainty
```

### Stage A — Feature Extraction

```text
3-channel waveform
        ↓
CNN encoder
        ↓
station embedding
```

- One **shared** encoder is applied to every station (same weights), so
  station embeddings are comparable graph node features.
- The CNN should learn temporal / time-frequency characteristics relevant to
  seismic discrimination (e.g. spectral content, P/S character, coda).
- Input: raw waveforms `[B, N, C, T]` (Phase 3) or spectrograms
  `[B, N, C, F, T']` (Phase 4). Output: `[B, N, D]`, or `[B, N, T', D]` when a
  temporal axis is kept for Stage D.
- Code: `features/cnn_encoder.py::CNNEncoder` (interface only).
  `models/cnn.py::CNNBaseline` = Baseline 1 (encoder + station pooling + softmax head).

### Stage B — Graph Construction

```text
one node = one seismic station
```

Potential edge information:

```text
geographical distance
waveform similarity
learned relationship
```

Conceptual edge formulation — **a starting point, not a final mathematical commitment**:

```text
A_ij = α * geographic_similarity(i,j)
     + β * waveform_similarity(i,j)
```

- Geographic term: e.g. k-NN, radius, or Gaussian kernel of great-circle distance.
- Waveform term: e.g. lag-tolerant normalized cross-correlation or coherence,
  computed **within one event only** (never across events — see §12).
- Learned term: data-dependent relation from station embeddings (dynamic graph).
- Code: `graph/construction.py`, `graph/similarity.py`, `graph/dynamic_graph.py` (interfaces only).

### Stage C — GAT

- The GAT should learn which stations are relevant to one another.
- Attention weights must be **inspectable** (the interface returns them on
  request) for interpretability. Attention is not automatically a faithful
  explanation — analyse it critically.
- Candidate variants: GAT, GATv2; with or without edge weights as priors.
- Code: `models/gat.py::GAT` (interface only).

### Stage D — Spatio-Temporal Modeling

The final architecture should capture:

```text
Temporal information
+
Spatial station relationships
```

The exact STGNN architecture is to be **determined experimentally** in
Phase 7, not hard-coded now. Candidates: temporal model per station followed
by spatial attention; interleaved temporal/spatial blocks (STGCN-style);
spatial attention per time step over a time-varying graph.
Code: `models/stgnn.py::STGNN` (interface only).

### Stage E — Evidential Deep Learning

The eventual classifier produces non-negative evidence:

```text
e_k >= 0
```

Then:

```text
α_k = e_k + 1
```

with:

```text
S = Σ α_k
```

and belief:

```text
b_k = e_k / S
```

For K classes, the conceptual uncertainty measure:

```text
u = K / S
```

Also: expected class probability `p_k = α_k / S`, and the identity
`Σ_k b_k + u = 1` (useful as a unit-test invariant in Phase 8).

> ⚠️ **EDL does NOT guarantee perfect OOD detection.** Its effectiveness must
> be experimentally evaluated (H4, H5). The literature contains documented
> pitfalls of EDL-style uncertainty (see §20); these must be reviewed before
> Phase 8.

Code: `models/edl.py::EvidentialClassifier`, `uncertainty/dirichlet.py`
(container `DirichletOutput` + function interfaces), `training/losses.py`
(`evidential_loss` interface). **No EDL computation or loss is implemented.**

### Composition (suggestion, not a decision)

All baselines can be expressed as
`encoder → [graph module] → station pooling → head (softmax | evidential)`.
A future phase may introduce a generic composed model; record it in the
Decision Log if adopted.

---

## 6. Important Scientific Constraints

1. **No unsupported claims.** Do not describe the model as "state-of-the-art"
   (or "better", "robust", "reliable") unless supported by actual literature
   comparisons and experiments in this repository.
2. **Accuracy is insufficient.** Evaluation must eventually include
   classification, calibration, uncertainty and OOD detection (§9).
3. **EDL is not magic OOD detection.** Whether uncertainty behaves
   appropriately must be tested experimentally.
4. **Avoid unnecessary complexity.** Every architectural component needs an
   experimental justification (baseline comparison or ablation). A component
   that does not help is removed or reported as not helping.
5. **No fabricated data or results.** Synthetic signals are allowed **only**
   inside unit tests to exercise code paths; they must never be reported,
   plotted or described as experimental results.
6. **Negative results are results.** Record them (experiments/ and Decision Log).
7. **Report variability.** Final numbers come from multiple random seeds
   (mean ± spread), on a leakage-safe split (§12).

---

## 7. Baselines

Progressively stronger baselines (planned experiments, **not current implementations**):

| # | Model | Needs phase(s) | Isolates the effect of |
|---|---|---|---|
| Baseline 1 | Simple CNN + Softmax | 3 | reference point |
| Baseline 2 | CNN + EDL | 3, 8 | EDL without graph |
| Baseline 3 | CNN + static spatial GAT + Softmax | 5, 6 | spatial graph + attention |
| Baseline 4 | CNN + static spatial GAT + EDL | 5, 6, 8 | EDL on graph model |
| Baseline 5 | CNN + dynamic graph + GAT + Softmax | 7 | dynamic graph |
| Final | CNN + dynamic STGNN + EDL | 7, 8 | full system |

Uncertainty reference points for softmax models (Phase 9): maximum softmax
probability and predictive entropy. Optional additional references (decide
after the Phase 9 literature review): temperature scaling for calibration,
MC dropout or deep ensembles for uncertainty.

---

## 8. Ablation Study Plan

Compare the contribution of each component (remove or replace it, keep
everything else fixed, multiple seeds):

| Component | Ablation |
|---|---|
| CNN | replace learned encoder by simpler features (e.g. classic discriminants) |
| Spectrogram representation | raw waveform vs time-frequency input |
| Spatial graph | graph model vs station pooling without edges |
| Waveform similarity | geographic-only edges vs geographic + similarity |
| GAT | attention vs uniform / fixed-weight aggregation |
| Dynamic graph | dynamic vs static graph |
| Temporal modeling | STGNN vs time-pooled embeddings |
| EDL | evidential head vs softmax head |

Purpose: determine which components actually improve classification,
calibration and OOD behaviour — not to justify a fixed design.

---

## 9. Evaluation Plan

Planned metrics (exact set may be modified after reviewing relevant literature):

**Classification**

```text
Accuracy
Precision
Recall
F1
Confusion Matrix
```

Report per-class and macro-averaged values (plus balanced accuracy) because
the class distribution is unknown and may be imbalanced.

**Calibration**

```text
Expected Calibration Error
Brier Score
Reliability Diagram
```

Report the ECE binning scheme with every value.

**OOD**

```text
AUROC
AUPRC
OOD detection rate
Uncertainty distributions
Risk-coverage analysis
```

Conventions (code: `evaluation/ood.py`, `uncertainty/metrics.py`):
higher score = more OOD-like; OOD = positive class; thresholds are never tuned
on the OOD test set.

Statistical practice: multiple seeds; report mean and spread; consider
bootstrap confidence intervals for small test sets.

---

## 10. Robustness Experiments

- **Noise:** test performance and uncertainty under increasing noise levels
  (noise model to be defined — e.g. real noise windows if available).
- **Missing stations:** remove one or more stations and observe
  classification performance and uncertainty (expected: uncertainty rises as
  information is removed; to be tested — H6). The `station_mask` exists for this.
- **Geographic generalization:** where data permits, train on one region and
  evaluate on another.
- **Unseen event types:** train on known categories and evaluate on an event
  type excluded from training (only if the dataset contains such events).

---

## 11. Data Policy

> **Dataset selected in Phase 1: PNW (Decision 008).** Do not fabricate dataset
> assumptions: every value below was measured from the files or is quoted from
> the dataset documentation (marked *documented*).

| Field | Value |
|---|---|
| Dataset name | Curated Pacific Northwest AI-ready Seismic Dataset ("PNW") — collections *ComCat* and *Exotic* |
| Source | Ni et al. (2023), *Seismica* 2(1), doi:10.26443/seismica.v2i1.368; files: github.com/niyiyu/PNW-ML (served from the UW-ESS server) |
| License | CC BY 4.0 (paper; SeisBench dataset record). Repository code: MIT |
| Version / access date / checksum | No data release tag; snapshot pinned by server ETag + SHA-256 in `<raw>/pnw/PROVENANCE.json`; accessed 2026-09-18 (§11A) |
| Station count | 639 (ComCat), 146 (Exotic, 143 shared), 642 combined; 9 FDSN networks |
| Event count | 46,054 ComCat events (178,939 traces); 5,657 exotic events (9,267 traces) |
| Sampling frequency | 100 Hz for every trace (curators resampled) |
| Channels | 3 components stored as E, N, Z; instrument band codes EH / BH / HH; missing components zero-filled |
| Geographic coverage | Pacific Northwest (WA, OR; offshore and ID margins): events 41.5–50.0 °N, 127.0–116.0 °W; stations 41.0–50.7 °N, 127.4–114.4 °W |
| Event labels | ComCat `source_type` (earthquake / explosion) **and** PNSN analyst label `source_type_pnsn_label` (eq, px, lf, ex, su, qb, sn, uk); exotic `source_type` (surface event, sonic boom, thunder, plane crash) |
| Class distribution | Proposed classes (Decision 010): earthquake 39,366 events vs anthropogenic 6,396 events (6.2 : 1) |
| Train/validation/test strategy | Proposed in §11F (Decision 012); implemented in Phase 2 |
| OOD strategy | §11E (Decision 011) |

**Storage and paths (Decision 009).** Dataset locations are *never* hard-coded.
The data root is resolved as `--data-root` flag → `$SEISMIC_DATA_ROOT` →
`paths.data_root` in `configs/config.yaml` (default `data`, relative to the
project root); raw files live in `<data_root>/raw/pnw/`. This works unchanged
on a local disk, on Google Colab with Google Drive mounted, or on any other
mounted storage. Raw data are immutable and never committed (`.gitignore`);
only `data/processed/dataset_manifest.json` (a description with relative paths
and checksums, no data) may be versioned. See `data/README.md`.

---

## 11A. DATASET SELECTED

```text
Selected dataset:  Curated Pacific Northwest AI-ready Seismic Dataset (PNW)
                   - ComCat collection -> primary supervised data (earthquake vs anthropogenic)
                   - Exotic collection (+ non-tectonic labels hidden inside ComCat) -> OOD evaluation
Source:            Ni, Hutko, Skene, Denolle, Malone, Bodin, Hartog & Wright (2023),
                   Seismica 2(1), doi:10.26443/seismica.v2i1.368;
                   https://github.com/niyiyu/PNW-ML  (data: https://dasway.ess.washington.edu/shared/niyiyu/PNW-ML/)
License:           CC BY 4.0
Version:           server snapshot, accessed 2026-09-18 (no release tag covers the data files)
  comcat_metadata.csv    50,025,733 B  ETag f6cbb870a72fa23d34e2ae34fde4df50-3  Last-Modified 2025-11-21  SHA-256 8cc7fba99351b833...
  exotic_metadata.csv     1,476,207 B  ETag 2014a67e2999c3023b511a0e3f9bde71    Last-Modified 2025-06-03  SHA-256 2ace01d8759e2aff...
  comcat_waveforms.hdf5  67,343,580,072 B ETag 7738fe98168abe7d14dd9795dcda1f47-123 (2023-12-17)  NOT downloaded in full
  exotic_waveforms.hdf5   4,111,581,160 B ETag b6fd5d8cd6fac45c2472f6b73e2b30fa-8   (2023-12-17)  NOT downloaded in full
  (full SHA-256 values: <raw>/pnw/PROVENANCE.json and data/processed/dataset_manifest.json)
Data format:       SeisBench layout — CSV metadata (one row = one station recording of one event)
                   + HDF5 /data/bucket<k> float64 [n_traces, 3, T], /data_format/component_order = "ENZ"
Subset acquired:   event-complete exploration subsets read from the remote HDF5 files by
                   HTTP byte ranges (Decision 013), seed 42, uniform per label:
                   ComCat 1,221 events / 4,019 traces (~1.45 GB); Exotic 732 events / 1,133 traces (~0.49 GB)
Event count:       46,054 ComCat + 5,657 exotic
Station count:     639 ComCat, 146 exotic (143 shared), 642 combined
Channel structure: 3 rows E, N, Z; band codes (ComCat traces) EH 55.0 %, BH 30.2 %, HH 14.8 %
Class structure:   see §11C.1 and Decision 010
```

**Event → stations → waveforms: exact fields** (implemented in
`src/seismic_ml/data/pnw.py`, tested in `tests/test_pnw.py`)

| Concept (schema) | Metadata field(s) | Notes |
|---|---|---|
| `SeismicEvent.event_id` | `event_id` | `uw…` = ComCat, `pnsn…` = exotic; all rows with one `event_id` form one event |
| `SeismicEvent.label` | `source_type_pnsn_label` (ComCat), `source_type` (exotic) | configured per collection (`data.yaml`) |
| origin / location / depth / magnitude | `source_origin_time`, `source_latitude_deg`, `source_longitude_deg`, `source_depth_km`, `preferred_source_magnitude` | ComCat only; **never model inputs** (§12.5) |
| `Station.station_id` (graph node) | `station_network_code` + `.` + `station_code` | unique per event (observed) |
| `Station` coordinates | `station_latitude_deg`, `station_longitude_deg`, `station_elevation_m` of the same row | per trace, not per site |
| `WaveformSample.data` | HDF5 array at `trace_name` (`bucket$row,:3,:T`) | `[3, T]`, rows `E, N, Z` (`/data_format/component_order`) |
| `WaveformSample.start_time`, rate | `trace_start_time`, `trace_sampling_rate_hz` | UTC; 100 Hz |
| P / S arrival times | `trace_P_arrival_sample`, `trace_S_arrival_sample` (÷ rate + start) | exotic: often no S, sometimes no P |
| observation metadata | `station_location_code`, `station_channel_code`, `trace_snr_db`, `trace_missing_channel`, `trace_has_offset`, pick polarity/onset/uncertainty | kept verbatim |

**Why it fits the research question**

| Requirement | Status | Evidence |
|---|---|---|
| Natural earthquakes | ✅ | 39,366 events labelled `eq` by PNSN analysts (observed) |
| Anthropogenic events | ✅ | 6,396 events: `px` probable explosion 6,362, `ex` explosion 30, `qb` quarry blast 4 (observed); `px` = analyst judgement "strong P + near known quarries" (documented) |
| Multiple stations per event | ⚠️ partial | `event_id` groups stations; median 3 stations (eq) vs 2 (anthropogenic); ≥ 4 stations: 49.8 % of eq, 21.9 % of anthropogenic events (observed) |
| Three components | ⚠️ partial | 70.1 % of ComCat traces have all three; 27.9 % vertical-only; 2.0 % lack the vertical or one horizontal (observed) |
| Station geographic information | ✅ | latitude/longitude/elevation on every row (observed) |
| Sufficient metadata | ✅ ComCat / ⚠️ exotic | origin, location, depth, magnitude, picks, SNR, quality flags for ComCat; exotic events have no origin/location/magnitude (documented + observed) |
| Unseen / OOD event categories | ✅ with caveats | low-frequency (`lf`), surface events, sonic booms, thunder from the **same network** (observed); few are multi-station (§11E) |
| Licence / access | ✅ | CC BY 4.0; public HTTPS with byte-range support (observed) |
| Student workstation | ✅ | metadata 51.5 MB; exploration subsets 1.94 GB; the 67 GB ComCat waveform file is only needed from Phase 2/3 and can be streamed by byte ranges or placed on external storage |

**Important limitations** (details §11C–§11D): few stations for many events
(especially explosions); inherited selection rule (ComCat traces exist only
where P **and** S were picked — documented); ~28 % vertical-only traces;
amplitudes in counts (no response removal — documented); strong label proxies
(time of day, depth, repeat sites); labelling-practice drift after ~2011;
exotic events mostly single-station and pick-aligned; no induced/geothermal
events.

---

## 11B. DATASET COMPARISON

Candidates investigated in Phase 1 (2026-09-18). *Observed* = measured or
listed by us; *documented* = stated by the dataset authors; *derived* =
computed from documented totals; **Unknown** = not determinable without data
we did not obtain.

| Dataset | Earthquakes | Anthropogenic | Multi-station | 3C | Station coords | OOD potential | Size | License | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **PNW (selected)** | 39,366 events (observed) | 6,396 events px/ex/qb (observed) | yes, `event_id`; median 3 / 2 stations (observed) | yes; 28 % vertical-only (observed) | yes, per trace | lf, su, sn in ComCat; exotic surface events, sonic booms, thunder (observed) | metadata 51.5 MB; waveforms 67.3 + 4.1 GB | CC BY 4.0 | counts; P+S-picked stations only; peer-reviewed data report |
| CEED (AI4EPS/CEED, HuggingFace) | ~99.4–99.5 % of 2.44 M traces (documented) | `quarry_blast` 0.2 % (NC) / 0.5 % (SC) of traces (documented) | yes, event-based; 2.8 (NC) / 4.3 (SC) traces per event (derived) | 76 % ENZ (documented) | yes | `long_period` (documented); geothermal/volcanic regions labelled earthquake | 190 GiB (documented) | MIT | response removed (µm/s); needs P+S+first motion; **release mid-rebuild at access: `main` held 29 of 42 NCEDC years and no SCEDC files; no version tags (observed)** |
| QuakeFlow SC / NC (AI4EPS, earlier releases) | Unknown by type | Unknown — `events.csv` has no event-type column (observed) | event-based HDF5 (documented) | Unknown | Unknown | Unknown | 454.6 GB / 163.5 GB; per-year HDF5 8–33 GB (observed) | MIT | no practical partial download of labelled blasts |
| STEAD (Mousavi et al., 2019) | ~450 k earthquakes (documented) | none (earthquakes + noise) | single-trace records; event grouping Unknown | yes | yes | noise only | ~85 GB (documented) | Unknown (not verified) | excluded: no anthropogenic class |
| SCSN 2011–2020 study set (GJI 236(2):979, 2024) | 37,727 events (documented) | 6,702 blasts (documented) | yes, 337 stations (documented) | yes (documented) | Unknown | Unknown | Unknown | Unknown | public release of the curated set not verified; rebuilding from SCEDC is a large effort |
| Eastern Kentucky (Miao et al., 2020) | 150 events (documented) | 4,218 quarry blasts (documented) | yes, ~14 recordings/event (derived) | Unknown | Unknown | Unknown | Unknown | Unknown | too few earthquakes; availability not verified |
| Self-built: USGS ComCat event types + FDSN waveforms | any | ComCat lists quarry blast / explosion types (documented) | yes | yes | yes | depends on region | unbounded | per data centre | no curated picks; large engineering and reproducibility burden |

**Trade-offs and choice (not a score).** PNW is the only candidate that
simultaneously offers thousands of analyst-labelled anthropogenic events,
event grouping with per-trace station coordinates, three components,
physically distinct non-tectonic sources *recorded by the same network*
(so OOD-ness is not confounded with a change of instruments or data centre),
a peer-reviewed description, an open licence and practical partial access.
CEED offers more stations per event (SCEDC), response-removed amplitudes and a
much larger catalogue, but quarry blasts are rare (≤ 0.5 % of traces) and the
release was being rebuilt at access time, which conflicts with reproducibility.
**No second dataset is combined in Phase 1**: there is no defensible need yet,
and mixing networks would add instrument/processing confounds. CEED quarry
blasts (California) remain a candidate for a later *geographic-generalisation*
test (Phase 10) — to be decided and logged then.

---

## 11C. DATA EXPLORATION FINDINGS

Source: `notebooks/01_data_exploration.ipynb` (executed) and
`scripts/inspect_data.py` outputs. Catalogue statistics use the **full**
metadata; waveform statistics use the exploration subsets only.
Tags: **[O]** observed, **[D]** documented by the authors, **[I]** inferred, **[U]** unknown.

1. **Labels.** [O] PNSN analyst labels (events / traces): `eq` 39,366 / 161,942;
   `px` 6,362 / 16,076; `lf` 265 / 758; `ex` 30 / 63; `su` 24 / 72; `qb` 4 / 21;
   `sn` 2 / 4; `uk` 1 / 3. Every event carries exactly one label. [D] ComCat's
   `earthquake` merges earthquake, slow earthquake and long-period volcanic
   earthquake; its `explosion` merges explosion, shot, probable explosion — and
   [O] ComCat `earthquake` indeed contains `lf`, `su`, `sn`, `uk` events.
   [O] 80 events disagree between ComCat and PNSN on earthquake vs explosion
   (69 ComCat earthquake / PNSN `px`; 11 ComCat explosion / PNSN `eq`).
   Label meanings: `eq` earthquake, `px` probable explosion, `ex` explosion,
   `qb` quarry blast, `sn` sonic boom, `su` surface event [D, AQMS/PNSN docs and
   the dataset paper]; `lf` low-frequency event with volcanic or glacial sources
   [D, PNSN research repository]; `uk` unknown [I from the code name, not verified].
2. **Multi-station coverage.** [O] `(event_id, NET.STA)` is unique, so traces per
   event = stations per event. ComCat events with 1 / 2 / 3 / 4+ stations:
   9,096 / 8,409 / 7,445 / 21,104 (median 3, max 41). Earthquakes:
   6,867 / 6,637 / 6,245 / 19,617; anthropogenic: 2,151 / 1,699 / 1,143 / 1,403.
   Exotic: 4,392 / 506 / 207 / 552 (78 % single-station).
   [I] Station count is a class proxy (explosions have fewer stations) — a
   shortcut risk for graph/set models.
3. **Three components.** [O] Arrays are `[3, T]` in order `ENZ`; the metadata
   keeps only band/instrument codes (EH, BH, HH), not component codes.
   Missing-component patterns (from NaN SNR entries, ComCat traces): none
   125,461 (70.1 %); E+N missing 49,908 (27.9 %, almost all EH short-period);
   Z missing 3,507 (1.96 %; 21 stations, 2,898 of them UW.HDW, 2012–2021);
   other patterns 63. Exotic: none 4,228; E+N 4,885; N 154.
   [O] On the waveform subsets the components that are exactly zero match the
   NaN-SNR pattern in 4,019 / 4,019 ComCat and 1,133 / 1,133 exotic traces, and
   the `trace_missing_channel` flag matches the zero count in 4,019 / 4,019 —
   i.e. missing components are reliably identified from the metadata alone.
   [U] Whether stored N/E were physically north/east or rotated `1`/`2`
   channels mapped to N/E — original component codes are absent.
4. **Sampling and windows.** [O] 100 Hz everywhere. ComCat: 15,001 samples,
   trace start = origin − 50.00 s for every trace; P pick at 51.5–66.4 s
   (5–95 %). Exotic: 18,001 samples, pick-aligned — P at exactly 70.00 s in
   98.7 % of traces (S at 80 s when P is missing).
   [I] Collections are distinguishable by window length and pick position
   alone → a trivial OOD shortcut unless windows are harmonised (§11D).
5. **Amplitudes.** [D] Counts, no instrument-response removal; linear detrend
   applied by the curators; clipped data kept. [O] Median peak |amplitude| on
   the ComCat subset (counts, live components): EH 265, BH 1,052, HH 865
   (5–95 %: EH 40–4,170; BH 177–16,997; HH 164–14,680) — amplitude scales are
   not comparable across instrument types.
6. **Stations.** [O] 642 stations, 9 networks (CC, CN, HW, IU, PB, TA, UO, US,
   UW); 17 ComCat sites (3 exotic) change coordinates over time → coordinates
   must be taken per trace. 94 of 639 ComCat sites (14.7 %) record with more
   than one instrument type over time (BH+HH 47, EH+HH 34, BH+EH 13) and 13 use
   more than one location code → a node's sensor is event-dependent. Exotic recordings over-represent the volcano
   network CC (33.7 % of exotic vs 12.7 % of ComCat traces).
7. **Events.** [O] Origins 2002-01-01 – 2022-12-31; magnitudes −0.28 – 5.1 on
   mixed scales (event counts: ml 31,928, md 14,020, mh 106).
8. **Confounders (label proxies)** [O], earthquake vs anthropogenic (px/ex/qb):
   origin 07:00–17:00 local time 44.4 % vs **97.7 %**; weekday 71.4 % vs 92.0 %;
   median catalogue depth 8.4 km vs **−0.5 km** (at/above sea level);
   median magnitude 1.03 vs 1.54; median stations 3 vs 2; median epicentral
   distance 29.5 km vs 44.2 km; median max-component SNR 15.6 dB vs 13.7 dB;
   vertical-only share 28.4 % vs 22.6 %; EH share 55.6 % vs 49.5 %.
9. **Temporal structure.** [O] Anthropogenic events per year: 36–151 in
   2002–2010, 261 in 2011, 391–616 in 2012–2022; earthquakes 872–2,721 per year
   (2009 peak: Wooded Island swarm [D]). [I] Labelling/cataloguing practice
   changed around 2011–2012 → class-prior drift over time.
10. **Repeat sources.** [O] Anthropogenic events fall in 1,156 cells of 0.1°;
    47 cells with ≥ 20 events hold 48.8 % of them; 663 of the 1,156 cells also
    contain earthquakes. Earthquakes: 2,612 cells, top-10 cells hold 25.2 %.
11. **Station–class balance.** [O] Of 345 stations with ≥ 50 traces, the
    anthropogenic-trace share is 0.9 % / 6.9 % / 34.1 % (10th/50th/90th
    percentile); 15 stations record > 50 % anthropogenic traces, 20 record none.
12. **Regions.** [O] Events (eq / px) by region: WA-west 27,117 / 2,266;
    WA-east 5,460 / 2,007; OR-west 5,416 / 1,892; OR-east 1,373 / 197
    (split at 46 °N and 121 °W). Every region contains both classes.
13. **Waveform inspection.** [O] Representative events were drawn by a seeded
    rule (notebook §8), not hand-picked; multi-station record sections and
    three-component plots are in the notebook. No claim about discriminability
    is made from these plots.

---

## 11D. DATA QUALITY ISSUES

Phase 1 measured these issues; the **Phase 2 status** of each is given after
the table (§11H/§11I have the details).

| # | Issue | Evidence | Scale | Proposed Phase 2 action |
|---|---|---|---|---|
| Q1 | Missing components, zero-filled | [O] NaN SNR + `trace_missing_channel`; zero pattern confirmed on all 5,152 subset traces | 27.9 % vertical-only, 1.96 % missing Z, 0.04 % other | explicit channel-availability mask; decide drop vs keep vertical-only; never treat zeros as signal |
| Q2 | Missing **vertical** at 21 stations | [O] 2,898 traces at UW.HDW | 3,507 traces | inspect waveforms; possible channel-mapping error upstream |
| Q3 | Amplitudes in counts, instrument-dependent | [D] + [O] subset median peaks EH 265 / BH 1,052 / HH 865 counts | all traces | choose normalisation scope (§4) or response removal (needs StationXML) |
| Q4 | Different window length and alignment per collection | [O] 15,001 origin-aligned vs 18,001 pick-aligned | all traces | one common P-aligned window for all collections |
| Q5 | Offsets/steps from triggered + continuous overlap | [D] + [O] `trace_has_offset=1` | 1,534 ComCat traces (0.86 %) | flag; test impact; possibly exclude |
| Q6 | Clipping (kept as-is) | [D]; [O] clipping indicator (≥ 10 samples at the peak value) found in 0 of 4,019 ComCat and 0 of 1,133 exotic subset traces | unknown for the full set (subsets ≈ 2 % / 12 % of traces) | clipping flag in quality checks |
| Q7 | Low SNR | [O] max-component SNR < 0 dB | 4,213 ComCat (2.4 %), 334 exotic traces | SNR threshold decision; report per-SNR performance |
| Q8 | Exotic pick gaps / oddities | [O] P missing 120, S missing 8,607, S = P in 26 traces | exotic | P-or-S alignment rule; review S = P cases |
| Q9 | Label conflicts ComCat vs PNSN | [O] | 80 events (0.17 %) | PNSN label authoritative; exclude from validation/test (Decision 010) |
| Q10 | Non-tectonic labels inside ComCat `earthquake` | [O] `lf` 265, `su` 24, `sn` 2, `uk` 1 events | 292 events | never train on `source_type`; route to OOD/excluded (Decision 011) |
| Q11 | Possible duplicate events (different IDs) | [O] origins ≤ 2 s and ≤ 0.05° apart | 13 pairs | manual review; same split or de-duplicate |
| Q12 | Station coordinates change over time | [O] | 17 ComCat / 3 exotic sites | per-trace coordinates (already in loader) |
| Q13 | Mixed magnitude scales | [O] ml / md / mh | all events | never a model input; stratify analyses carefully |
| Q14 | Upstream metadata cleanup | [D] Nov 2025: 4,970 ComCat samples with undocumented events removed from metadata (waveforms still in HDF5) | 2.7 % | pin metadata snapshot (done: ETag + SHA-256) |
| Q15 | Surface-event label noise | [D, related work] 1.5–8.4 % possibly mislabelled | exotic | report OOD metrics with this caveat |
| Q16 | Horizontal orientation convention | [U] | BH/HH | check StationXML orientations if horizontal direction matters |
| Q17 | Waveform NaN / flat channels / long zero runs | [O] none found: 0 traces with NaN, 0 flat non-zero components, 0 zero-runs ≥ 100 samples on live components (5,152 subset traces) | full set unknown | keep these checks in Phase 2 quality control |

Machine-readable counts: `<data_root>/interim/pnw_quality_report.json`;
per-trace measurements: `<data_root>/interim/pnw_<collection>_waveform_metrics.csv`.

**Phase 2 status:**
Q1 resolved — missing components kept as exact zeros with `channel_mask` (Decision 022).
Q2 investigated — UW.HDW records Z-only 2002–2011, E+N only 2012–2021 (E/N uncorrelated,
r ≈ 0: genuine horizontals, not a duplicated vertical), all three in 2022; handled by Q1's mask;
cause unknown.
Q3 addressed — station-scope std normalisation + common 1–16 Hz band (Decisions 018, 021);
response removal not done (possible ablation).
Q4 resolved — per-station P-aligned 60 s windows for every collection; P at sample 500 verified
in all processed traces (Decision 020).
Q5, Q6 — kept and flagged (`has_offset`, `clipped_components` in traces.csv; 0 clipped on subsets).
Q7 — kept (no SNR threshold); report per-SNR performance from Phase 3.
Q8 — traces without P are dropped (reason `no_p_pick`); S = P cases kept.
Q9 — PNSN label used; 47 conflicts in val/test excluded, 33 in train kept (Decision 023).
Q10 — routing by `configs/data.yaml` label roles; unmapped labels raise (Decision 023).
Q11 — duplicate pairs kept in one split; 0 straddled a boundary.
Q12 — per-trace coordinates (Decision 015). Q13–Q17 unchanged (reporting caveats).

---

## 11E. OOD STRATEGY

**Training classes (proposed, Decision 010):** earthquake = PNSN `eq`;
anthropogenic = PNSN `px` + `ex` + `qb`. Labels always come from
`source_type_pnsn_label`; ComCat `source_type` must never be used for training
labels (it hides OOD events inside `earthquake`).

**OOD sets (proposed, Decision 011) — never used for training, tuning or thresholds:**

| Set | Source | Physical meaning | Events | ≥ 2 st. | ≥ 3 st. | ≥ 4 st. |
|---|---|---|---|---|---|---|
| OOD-A `lf` | ComCat (PNSN label) | low-frequency events, volcanic/glacial | 265 | 192 | 129 | 78 |
| OOD-A `su` | ComCat (PNSN label) | surface events (mass movements) | 24 | 19 | 11 | 6 |
| OOD-A `sn` | ComCat (PNSN label) | sonic boom | 2 | 2 | 0 | 0 |
| OOD-B surface event | Exotic | rockfalls, avalanches, ice/debris | 5,425 | 1,211 | 722 | 530 |
| OOD-B sonic boom | Exotic | atmospheric shock waves | 111 | 38 | 29 | 19 |
| OOD-B thunder | Exotic | atmospheric (thunderquakes) | 120 | 15 | 7 | 3 |
| OOD-B plane crash | Exotic | impact | 1 | 1 | 1 | 0 |
| excluded | ComCat `uk` | unknown | 1 | — | — | — |

**What is realistically possible.** The dataset supports the OOD question
(H5) for *unfamiliar natural non-tectonic and atmospheric sources recorded by
the same network*. It does **not** contain induced, geothermal or mining-collapse
events; the Phase 0 examples of those OOD types cannot be tested with PNW.
No OOD set is manufactured by splitting earthquakes.

**Mandatory controls for OOD experiments (Phase 9):**
1. Harmonise windows first (§11D Q4) — otherwise OOD-B is detectable by window
   length / pick position.
2. Report OOD metrics stratified by station count (1, 2, 3, 4+); multi-station
   OOD sets are small (hundreds of events at best) → bootstrap intervals.
3. Station-identity control: compare OOD scores with in-distribution events
   recorded at the *same* stations (exotic events over-represent CC stations).
4. OOD events never tune thresholds; if an OOD-validation set is required it
   is declared up front (e.g. by year) and kept disjoint from OOD-test.
5. Surface-event label noise (Q15) is reported with every OOD number.

---

## 11F. PROPOSED DATA SPLIT

**S1 is implemented (Phase 2, Decision 023); the actual split counts are in §11I.**
The design below is kept as written in Phase 1. Counts in the S1 table are the
Phase 1 projection (by UTC origin year, before buffer/conflict exclusions).

**Unit:** the event. All traces of an event share a split. Splits are stored as
event-ID lists. Candidate duplicate pairs (Q11) are kept together.

**S1 — primary, temporal (main results):**

| Split | Years | Earthquake | Anthropogenic | Anthropogenic share |
|---|---|---|---|---|
| train | 2002–2018 | 31,332 | 4,491 | 12.5 % |
| validation | 2019–2020 | 3,831 | 903 | 19.1 % |
| test | 2021–2022 | 4,203 | 1,002 | 19.3 % |

Rationale: mimics deployment (train on the past, predict the future);
aftershock sequences and swarms rarely straddle a year boundary — a buffer
(e.g. drop events within 7 days before a boundary) is proposed to remove the
rest; the class-prior shift (12.5 % → 19 %) is real and must be reported, not
hidden (it matters for calibration, H4).
Limitation: repeat quarries and all stations appear in train and test →
report test metrics separately for events at *seen* source sites (0.1° cell
present in train) vs *new* sites.

**S2 — source-site generalisation (robustness, Phase 10):** group events into
spatial source clusters (0.1° cells as a starting point; parameter chosen in
Phase 2), grouped K-fold over clusters stratified by class. Earthquakes and
blasts in the same cell stay together (663 shared cells).

**S3 — geographic hold-out (robustness, Phase 10):** hold out a region
(e.g. WA-east or OR-west; every region has both classes). For geographic
claims, also report a station-disjoint variant (drop held-out-region stations
from training) because S1/S2 share stations.

**Leakage controls (all splits):** event-level unit; no origin time, depth,
magnitude, location or station ID as model inputs (§12.5); label-conflict
events (Q9) excluded from validation/test; normalisation statistics from train
only; OOD events outside all supervised splits. Proposed sanity baselines for
Phase 3: a classifier using *only* station IDs / station count / local hour —
its score quantifies how much a shortcut could explain.

---

## 11G. PHASE 1 DECISIONS

Summary (full entries in §21): **008** dataset = PNW (ComCat + Exotic), single
dataset, no combination; **009** configurable, machine-independent data paths
(flag → env → config), raw data outside Git, Colab/Drive supported;
**010** PNSN label is authoritative; proposed classes earthquake (`eq`) vs
anthropogenic (`px`, `ex`, `qb`); **011** OOD candidates (`lf`, `su`, `sn`,
exotic set) excluded from training, `uk` excluded; **012** proposed split
design (S1 temporal primary, S2 source-site, S3 geographic); **013**
acquisition by metadata download + event-complete byte-range subsets with
provenance; **014** dependencies h5py + pandas (core), notebook extra;
SeisBench/ObsPy not adopted; **015** station node ID = `NET.STA` with per-trace
coordinates, component order kept as stored (`ENZ`); **016**
`SeismicEventDataset` takes metadata + waveform file, opens HDF5 lazily; a
split is a list of event IDs. Decision 003 (dense padded tensors) stays
provisional — observed N ≤ 41, median 3, so dense padding remains practical.

---

## 11H. PREPROCESSING PIPELINE (PHASE 2)

Implemented in `src/seismic_ml/preprocessing/` (`waveform.py`, `normalization.py`),
configured in `configs/data.yaml` → `preprocessing`, justified in
`notebooks/02_waveform_analysis.ipynb` (executed on the exploration subsets).
Evidence tags as in §11C.

**Chain (per station recording, `preprocess_observation`):**

```text
raw trace [3, 15001 | 18001] @100 Hz, counts
  -> quality check (hard fail: non-finite samples, no live component; flags: clipping, flat)
  -> detrend (linear, full trace) -> 2 s cosine taper -> band-pass 1–16 Hz
     (Butterworth order 4, zero-phase, applied to the FULL trace so transients stay away from P)
  -> cut [P - 10 s - 1 s, P + 50 s + 1 s] at 100 Hz        (window must be >= 5 s from trace ends)
  -> resample 100 -> 50 Hz (polyphase) -> trim the 1 s margins  => P exactly at sample 500
  -> missing components forced to exact zero; channel_mask recorded
  -> normalise: divide by the std of the station's live components (one scale per station)
  => float32 [3, 3000], E/N/Z, 50 Hz, window P-10 s .. P+50 s
```

| Parameter | Value | Evidence (notebook 02 unless noted) |
|---|---|---|
| Upper corner | 16 Hz | [O] BH spectra collapse above ~20 Hz (median PSD ratio 18–19.5 Hz / 21–23 Hz = 198 for BH vs 2.1–2.7 for HH/EH); [D] BH upsampled to 100 Hz by the curators → [I] BH recorded at 40 sps; content > ~18 Hz is an instrument fingerprint |
| Lower corner | 1 Hz | [O] median spectral SNR ≈ 1 below 0.5 Hz for all instruments (microseism); EH (1 Hz geophones) roll off below 1 Hz; amplitudes are raw counts, so a common flat band keeps instruments comparable. **Trade-off:** [O] explosions and `lf` events carry relatively more 0.5–2 Hz energy — partly attenuated; 0.5 Hz is a planned ablation |
| Filter type | Butterworth, order 4, zero-phase | no phase delay (P onset stays at the pick); [O, tests] acausal smear before onset is small and symmetric |
| Target rate | 50 Hz | > 2 × 16 Hz; exact decimation by 2; halves the data |
| Window | P − 10 s … P + 50 s, per station | [O] 99 % of S−P ≤ 19.5 s (full ComCat); signal (1–16 Hz energy > 3× noise) ends within P + 50 s for 80 % of eq and 77 % of px traces (subset); 99.957 % of ComCat traces have enough data; per-station P alignment removes the origin- vs pick-aligned collection artefact (§11C.4) |
| Normalisation | std, station scope, live components only | keeps Z/H and P/S ratios within a station (physically meaningful); inter-station amplitudes are raw counts of different instruments (not comparable) → not preserved; std is less sensitive than peak to isolated telemetry spikes [D]; no statistic is fitted across samples (no leakage) |
| Missing components | kept as zeros + `channel_mask` | 28 % of ComCat traces are vertical-only; dropping them would remove most EH stations |
| Hard drops | no P pick; not enough data around P; non-finite samples; no live component | reasons written to `dropped_traces.csv` |
| Soft flags (kept) | `has_offset`, clipping indicator, flat component | written to `traces.csv` |

**Known limitations of the chosen chain:** (1) the coda of large events and of
long emergent sources is truncated (signal ends within P + 50 s for only ~40 %
of surface-event/thunder traces — part of their OOD signature, noted for
Phase 9); (2) relative P times across stations are removed from the waveforms
(kept as `p_time_offset_s` metadata, not used as input); (3) no instrument
response removal — amplitudes remain instrument-dependent across stations.

---

## 11I. S1 SPLIT AND PROCESSED DATA (PHASE 2)

**S1 temporal split (implemented; `src/seismic_ml/data/splits.py`, run by
`scripts/prepare_data.py`; full metadata; every event gets exactly one row):**

| Split | Earthquake | Anthropogenic | Anthropogenic share |
|---|---|---|---|
| train (origin < 2019-01-01) | 31,318 | 4,489 | 12.5 % |
| val (2019–2020) | 3,788 | 879 | 18.8 % |
| test (≥ 2021-01-01) | 4,201 | 977 | 18.9 % |

- `ood`: 5,948 events (ComCat `lf` 265, `su` 24, `sn` 2; exotic surface event 5,425,
  sonic boom 111, thunder 120, plane crash 1).
- `excluded` (with reason): 63 `boundary_buffer` (< 7 days before a boundary),
  47 `label_conflict` (in val/test; 33 conflicting events remain in train with the
  PNSN label), 1 `excluded_label` (`uk`); 0 duplicate pairs straddled a boundary.
- Totals: 45,652 supervised + 5,948 OOD + 111 excluded = 51,711 events (all).
- **[O] Seen vs new source sites in test** (0.1° cell contains a training event of
  the same class): anthropogenic 859 seen / 118 new (88 % seen); earthquake 4,005
  / 196. [I] Test performance on explosions will mostly measure *known* sites —
  Phase 3+ must report the 118 new-site events separately.
- Files: `<data_root>/processed/splits/s1_temporal/events.csv` (columns:
  event_id, collection, label_raw, source_type, class, role, split, reason,
  reference_time, year, origin_time, latitude, longitude, n_stations, site_cell,
  site_seen_in_train, label_conflict) + `summary.json`; read with
  `seismic_ml.data.splits.load_split_table`.

**Processed data (implemented; `src/seismic_ml/data/processed.py`):**
`<data_root>/processed/pnw/<collection>/` holds `waveforms.hdf5`
(`/waveforms` float32 `[n, 3, 3000]`, `/channel_mask` bool `[n, 3]`), `traces.csv`,
`dropped_traces.csv`, `preprocessing.json` (config, input paths relative to the
data root + SHA-256, counts, software versions). Reader: `ProcessedEventDataset`
(→ `SeismicEvent`, label = class name or `None` for OOD) → `collate_events`
(→ `EventBatch` with `station_mask` and `channel_mask`).

| Collection | Input (local subsets) | Kept | Dropped |
|---|---|---|---|
| ComCat | 1,221 events / 4,019 traces | 1,220 events / 4,018 traces | 1 trace `window_out_of_range` |
| Exotic | 732 events / 1,133 traces | 728 events / 1,112 traces | 21 traces `no_p_pick` (4 events lost all traces) |

[O] Verified on the outputs: P at exactly 10.000 s (sample 500) in every window
of both collections; 0 NaN; masked components exactly zero; live-component std = 1.

⚠️ **Only the Phase 1 exploration subsets are processed** (e.g. 872 of the 35,807
train events have local waveforms). The split covers all events; the pipeline runs
unchanged on the full files once they are local. Getting the waveforms of the full
supervised set (178,102 traces: 64.1 GB as full traces, ~31 GB if only a 72 s
window around P is fetched by byte ranges; the server delivered 0.3–0.5 MB/s) is
an **open Phase 3 decision** that needs the user's approval (§17).

---

## 11J. PHASE 2 DECISIONS

Summary (full entries in §21): **017** SciPy added (user-approved);
**018** band-pass 1–16 Hz, Butterworth order 4, zero-phase, on the full trace;
**019** resample to 50 Hz; **020** per-station P-aligned window P − 10 s … P + 50 s
with edge/resampling margins; **021** std normalisation at station scope over
live components; **022** missing components kept as zeros + `channel_mask`
(new optional `EventBatch.channel_mask`); **023** S1 temporal split implemented
(train < 2019, val 2019–2020, test ≥ 2021, 7-day buffer, label-conflict and
duplicate handling, seen/new-site flag); **024** processed-data format +
`ProcessedEventDataset` + `collate_events` (dynamic padding, no truncation,
explicit unlabeled/OOD handling).

---

## 12. Data Leakage Prevention

**This is a major project rule.** Seismic data contain strong non-independence;
a naive random split can inflate every metric in §9.

Seismic events from the same **source / event sequence / station
configuration** must be carefully considered when creating train/test splits.
**Avoid random splitting if it causes information leakage. The final split
strategy must be justified based on the dataset** and recorded in the
Decision Log.

Known risks and rules:

1. **Split by event, never by record.** All station records of one event
   belong to the same split.
2. **Repeating sources.** Quarry blasts recur at the same quarries, producing
   near-identical waveforms; a random event-level split lets a model memorize
   source locations/paths. Consider grouping by source site / spatial cluster.
3. **Event sequences.** Aftershock sequences and swarms are highly correlated;
   consider grouping by sequence or time-blocked splits.
4. **Station configuration.** If a class is recorded mostly by specific
   stations/networks, the model can learn station identity. Inspect
   class balance per station in Phase 1; consider station-held-out tests.
5. **Confounders / shortcuts.** Blasts tend to occur during working hours,
   at shallow depth, small magnitude and known sites. Origin time, location,
   depth and magnitude (`SeismicEvent` metadata) must **not** be model inputs
   unless explicitly studied and logged. Report class-conditional
   distributions of these variables in Phase 1.
6. **Duplicates.** The same event can appear under different IDs (e.g. in
   merged catalogs); deduplicate by origin time/location.
7. **Fitted statistics.** Normalization statistics, thresholds and
   hyperparameters are fitted on training/validation data only. The test set
   is used once, for final evaluation.
8. **OOD sets.** OOD test data are never used for tuning; if a threshold or
   model selection needs OOD data, declare a separate OOD validation set.
9. **Waveform similarity.** Similarity edges are computed within a single
   event only; cross-event comparison is template matching and leaks.
10. **Collection artefacts (Phase 1 finding).** Different collections can differ
    in window length, pick alignment or station network (§11C.4, §11C.6); any
    such artefact is a shortcut and must be removed before comparisons.

Phase 1 measured these risks for PNW in §11C (items 2, 8–12) and turned them
into the split design of §11F.

---

## 13. Reproducibility

Every important experiment must record:

```text
random seed
dataset version
configuration files
model configuration
training configuration
software versions
experiment identifiers
```

Tooling already available (Phase 0, implemented and tested):

- `seismic_ml.utils.reproducibility.set_seed(seed, deterministic=False)`
- `seismic_ml.utils.reproducibility.get_environment_info()` — Python/NumPy/PyTorch/CUDA versions
- `seismic_ml.utils.reproducibility.generate_experiment_id(name)` — sortable unique ID
- `seismic_ml.config.load_config(path)` — resolved config (snapshot it per run)
- `seismic_ml.utils.logging.configure_logging(level, log_file)`

Planned conventions (from Phase 3): each run writes to
`outputs/results/<experiment_id>/` a resolved config snapshot, environment
info, seed, dataset version and git commit (once the repo is under git), and
is summarized in `experiments/` (see `experiments/README.md`).

---

## 14. Development Phases

```text
PHASE 0
Project initialization
    ↓
PHASE 1
Dataset selection + data exploration
    ↓
PHASE 2
Waveform preprocessing
    ↓
PHASE 3
CNN baseline
    ↓
PHASE 4
Time-frequency experiments
    ↓
PHASE 5
Spatial graph construction
    ↓
PHASE 6
GAT
    ↓
PHASE 7
Spatio-temporal GNN
    ↓
PHASE 8
Evidential Deep Learning
    ↓
PHASE 9
OOD + uncertainty evaluation
    ↓
PHASE 10
Ablation + robustness experiments
    ↓
PHASE 11
Final experiments + documentation
```

| Phase | Scope | Main code touched | Exit criteria |
|---|---|---|---|
| 0 | Scaffold, configs, schemas, interfaces, docs | everything (stubs) | tests pass; this file complete ✅ |
| 1 | Survey and select dataset; implement reading into schemas; exploration notebook; draft split + OOD strategy | `paths.py`, `data/` (acquisition, seisbench_format, pnw, inspection, manifest, loader, dataset), `scripts/download_data.py`, `scripts/inspect_data.py`, `notebooks/01`, `utils/visualization.py` | §11 filled; split strategy justified (§11F); decisions logged ✅ |
| 2 | Preprocessing pipeline, batching, **split implementation** | `preprocessing/waveform.py`, `normalization.py`, `data/loader.py::collate_events`, `scripts/prepare_data.py`, split module | common P-aligned window for all collections; missing-component policy; parameters justified and in `data.yaml`; S1 split files; unit tests on synthetic signals ✅ |
| 3 | CNN encoder, Baseline 1, training loop, classification metrics | `features/cnn_encoder.py`, `models/cnn.py`, `training/`, `evaluation/classification.py`, `scripts/train.py`, `scripts/evaluate.py` | Baseline 1 trained on a leakage-safe split, multiple seeds, results recorded |
| 4 | Time-frequency input experiments | `preprocessing/spectrogram.py`, `CNNEncoder` spectrogram path | waveform vs spectrogram comparison recorded |
| 5 | Static spatial + waveform-similarity graphs | `graph/construction.py`, `graph/similarity.py` | graph construction tested; graph statistics inspected |
| 6 | GAT on station graph (Baseline 3) | `models/gat.py`, attention plots | comparison vs Baseline 1 (H1, H2) |
| 7 | Dynamic graph + STGNN (Baseline 5) | `graph/dynamic_graph.py`, `models/stgnn.py` | comparison vs static graph (H3) |
| 8 | EDL head + loss (Baselines 2, 4, Final) | `models/edl.py`, `uncertainty/dirichlet.py`, `training/losses.py` | EDL loss choice logged; invariants tested |
| 9 | Calibration, uncertainty, OOD evaluation | `evaluation/calibration.py`, `evaluation/ood.py`, `uncertainty/metrics.py` | H4, H5 evaluated against softmax references |
| 10 | Ablations (§8) and robustness (§10) | experiment scripts/configs | all ablations run with seeds |
| 11 | Final experiments, figures, documentation | docs, `outputs/` | reproducible final results; claims checked against §6 |

Ordering note: calibration metrics may be pulled forward to Phase 3 to
characterise the softmax baseline — log it in the Decision Log if done.

---

## 15. Phase Completion Rules

Every future phase follows this procedure:

```text
1. Read PROJECT_CONTEXT.md
2. Inspect existing implementation
3. Identify current phase
4. Review previous decisions
5. Implement only the requested phase
6. Run tests
7. Validate outputs
8. Update documentation
9. Update PROJECT_CONTEXT.md
10. Stop
```

**Never silently jump ahead multiple phases.** If the tests fail *before*
you start, report it instead of masking it. When implementing a placeholder,
replace its placeholder test with functional tests.

---

## 16. Repository Map and Code Conventions

### Layout

```text
seismic-event-classification/
├── PROJECT_CONTEXT.md      ← this file (source of truth)
├── README.md               ← public overview
├── CLAUDE.md, AGENTS.md    ← pointers for coding agents → this file
├── pyproject.toml          ← package + dependencies + pytest config
├── configs/                ← config.yaml (entry, storage roots) + data/model/training.yaml
├── data/                   ← DEFAULT data root: raw/ interim/ processed/ (data never committed)
├── notebooks/              ← 01 (Phase 1) and 02 (Phase 2) executed; 03 placeholder
├── scripts/                ← download_data.py, inspect_data.py (Phase 1); prepare_data.py,
│                             run_notebooks.py (Phase 2); train.py, evaluate.py (exit 1: not implemented)
├── src/seismic_ml/
│   ├── config.py           ← YAML loading (implemented)
│   ├── paths.py            ← data/output root resolution: flag > env > config (implemented)
│   ├── placeholders.py     ← NotYetImplementedError convention (implemented)
│   ├── data/               ← schemas, acquisition (HTTP download / byte ranges),
│   │                         seisbench_format (HDF5 layout, subsets, store), pnw (metadata,
│   │                         selection, schema conversion), inspection, manifest,
│   │                         loader (load_events, collate_events), dataset (SeismicEventDataset),
│   │                         splits (S1 temporal), processed (writer + ProcessedEventDataset)
│   │                         — implemented; create_dataloader (stub, Phase 3)
│   ├── preprocessing/      ← waveform + normalization (implemented, Phase 2); spectrogram (stub)
│   ├── features/           ← CNNEncoder (stub)
│   ├── graph/              ← spatial, similarity, dynamic graph (stubs)
│   ├── models/             ← CNNBaseline, GAT, STGNN, EvidentialClassifier (stubs)
│   ├── uncertainty/        ← DirichletOutput container (implemented), functions (stubs)
│   ├── training/           ← Trainer, losses (stubs)
│   ├── evaluation/         ← classification, calibration, OOD metrics (stubs)
│   └── utils/              ← logging, reproducibility, validation (implemented),
│                             visualization (event waveforms + station map implemented; others stubs)
├── tests/                  ← pytest suite (toy PNW-layout fixture + local byte-range HTTP server)
├── experiments/            ← experiment records (none yet)
└── outputs/                ← checkpoints/ logs/ figures/ results/ (empty)
```

### Placeholder convention (Decision 005)

Unimplemented interfaces raise
`seismic_ml.placeholders.NotYetImplementedError(component, phase)` — a
subclass of `NotImplementedError` carrying the planned phase. List what
remains with:

```bash
grep -rn "NotYetImplementedError(" src/
```

Model classes: the constructor stores and validates hyperparameters (no
layers are created); `forward` raises. Functions: the body only raises.

| Phase | Placeholders to implement |
|---|---|
| 1 | ✅ done: `SeismicEventDataset`, `load_events`, `plot_event_waveforms`, `plot_station_map`; ingestion moved to `scripts/download_data.py` |
| 2 | ✅ done: `detrend`, `taper`, `bandpass_filter`, `resample`, `extract_window`, `align_to_p_arrival`, `check_signal_quality`, `preprocess_waveform`, `preprocess_observation`, `normalize_waveform`, `collate_events`, `scripts/prepare_data.py`, S1 split, processed store |
| 3 | `CNNEncoder.forward`, `CNNBaseline.forward`, `cross_entropy_loss`, `Trainer.*`, `create_dataloader`, `compute_classification_metrics`, `confusion_matrix`, `scripts/train.py`, `scripts/evaluate.py` |
| 4 | `compute_spectrogram` |
| 5 | `pairwise_distances_km`, `build_spatial_graph`, `compute_waveform_similarity` |
| 6 | `GAT.forward`, `plot_attention_weights` |
| 7 | `build_dynamic_graph`, `STGNN.forward` |
| 8 | `EvidentialClassifier.forward`, `evidence_to_alpha`, `dirichlet_strength`, `belief_masses`, `uncertainty_mass`, `expected_probabilities`, `dirichlet_from_evidence`, `evidential_loss`, `kl_dirichlet_regularizer`, `annealing_coefficient` |
| 9 | `expected_calibration_error`, `brier_score`, `reliability_curve`, `ood_auroc`, `ood_auprc`, `ood_detection_rate`, `risk_coverage_curve`, `predictive_entropy`, `max_probability`, `dirichlet_mutual_information`, `dissonance`, `plot_reliability_diagram` |

### Tensor and graph conventions (provisional — Decision 003)

| Quantity | Shape | Notes |
|---|---|---|
| Batched waveforms | `[B, N, C, T]` | padded to N stations; Phase 2: C = 3 (E, N, Z), T = 3000 @ 50 Hz, P at sample 500 |
| Station mask | `[B, N]` bool | True = real station |
| Channel mask | `[B, N, C]` bool | True = component holds real data (Phase 2; missing = exact zeros) |
| Coordinates | `[B, N, 2]` | (lat, lon) degrees |
| Station embeddings | `[B, N, D]` | or `[B, N, T', D]` with temporal axis |
| Adjacency | `[B, N, N]` (or `[B, T', N, N]`) | `A[b, i, j] > 0` ⇒ edge j → i (row = receiver); padded rows/cols zero |
| Logits | `[B, K]` | softmax applied in loss / evaluation |
| Dirichlet output | `DirichletOutput` | evidence/alpha/belief/probability `[B, K]`; strength/uncertainty `[B]` |

### Configuration conventions (Decision 004)

- `configs/config.yaml` is the entry point; `includes` attaches `data`, `model`, `training`.
- `null` = not decided. Code must not replace `null` with a guessed default.
- The seed lives only in `config.yaml`.
- Class index order = `data.labels.class_names` order.

### Data access conventions (Phase 1, Decisions 009, 013, 015, 016)

- Storage roots come from `seismic_ml.paths.resolve_paths(config, project_root=..., data_root=...)`:
  flag (`--data-root` / `--output-root`) > env (`SEISMIC_DATA_ROOT` / `SEISMIC_OUTPUT_ROOT`)
  > `configs/config.yaml` > defaults. Never write absolute paths into configs, manifests or
  notebooks (a test scans the repository for machine-specific paths).
- Dataset files are named in `configs/data.yaml` → `dataset.collections.<name>`; the loader
  uses the full waveform file if present in `<raw>/pnw/`, otherwise the exploration subset.
- `trace_name` (e.g. `bucket4$17,:3,:15001`) addresses a trace in both full and subset files.
- Graph node = `NET.STA`; coordinates come from the trace's own metadata row.
- `SeismicEvent.label` = the collection's configured label column (ComCat: PNSN label).
- A split is a list of event IDs passed to `SeismicEventDataset(..., event_ids=...)`.
- Phase 2: model-ready data = `ProcessedEventDataset(<processed>/pnw/<collection>, split_table,
  splits=[...])` → `collate_events(events, class_names, allow_unlabeled=...)` → `EventBatch`.
  The split table (`load_split_table`) is joined at read time, so re-splitting never needs
  re-processing. `SeismicEvent.label` is the class name for train/val/test and `None` for OOD.

### Dependency policy (Decision 002, extended by Decision 014)

- Runtime: `numpy`, `torch`, `pyyaml`, `h5py`, `pandas`, `scipy` (Phase 2). Dev: `pytest`.
  Optional `notebooks` extra: `matplotlib`, `ipykernel`, `nbclient`, `nbformat`.
- Not adopted: SeisBench (its PNW loader downloads whole files; we need byte-range subsets),
  ObsPy (no MiniSEED/StationXML handling needed yet — revisit if response removal is chosen).
- Deferred: scikit-learn (Phase 3 metrics, likely), PyTorch Geometric (Phase 6, only if
  adopted over a dense implementation).

### Environment and testing

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -e ".[dev,notebooks]"   # install torch first per pytorch.org if not present
python -m pytest                # offline: tests use a toy fixture, not the real data
python scripts/download_data.py [--data-root PATH]   # real data (Phase 1 subsets)
python scripts/inspect_data.py  [--data-root PATH]   # manifest + quality report
python scripts/prepare_data.py  [--data-root PATH]   # S1 split + preprocessing (Phase 2)
python scripts/run_notebooks.py notebooks/0*.ipynb   # UTF-8-safe execution (not jupyter-execute)
```

Development machine (Phases 0–1): Windows 11, Python 3.11.9, PyTorch 2.5.1+cu121
(pre-installed system-wide; the local `.venv` was created with
`--system-site-packages` to reuse it), NumPy 2.3.5, PyYAML 6.0.3, pytest 9.1.1,
h5py 3.16.0 (HDF5 2.0.0), pandas 3.0.6, matplotlib 3.11.2, scipy 1.17.1 (Phase 2).
`pyproject.toml` sets `pythonpath = ["src"]`, so tests also run without installation.
The PNW server served ~0.3–0.5 MB/s at access time; the Phase 1 subsets took ~1.5 h.

---

## 17. Open Questions and Known Risks

Answered in Phase 1: dataset (PNW, §11A); class balance 6.2 : 1 and stations
per event (median 3; §11C.2); OOD availability (§11E); split design (§11F).

Answered in Phase 2: common window, missing-component policy, normalisation scope,
filter band, sampling rate, S1 boundary buffer (§11H–11J).

Open questions (to be answered in the indicated phase, then logged):

- **Training-data acquisition (Phase 3, needs user approval):** only 872 of 35,807 train
  events have local waveforms. Options: a larger seeded subset; the full supervised set
  (178,102 traces: 64.1 GB as full traces, ~31 GB as 72 s byte-range windows around P;
  18–60 h at the observed 0.3–0.5 MB/s); or downloading on a faster connection (e.g. Colab).
- How to control the station-count shortcut (e.g. station sub-sampling, per-bin reporting)? (Phase 3/5)
- Minimum stations per event for graph models (currently keep all, min 1)? (Phase 5)
- 0.5 Hz lower corner and longer windows as ablations (Phase 4/10); response removal (Phase 10).
- Source-site clustering parameter for split S2 (Phase 10).
- Dense in-house GAT vs PyTorch Geometric? Handling of variable N (≤ 41)? (Phase 5/6)
- Which EDL loss formulation and evidence activation? (Phase 8)

Known risks:

- **Few stations per event (confirmed, §11C.2):** 20 % of ComCat events have
  one station and only 22 % of anthropogenic events have ≥ 4; graph components
  may have little to exploit — H1–H3 must be evaluated per station-count bin.
- **Shortcuts (confirmed, §11C.8–11):** local time, depth, repeat sites, station
  identity and station count are strong label proxies.
- **Label drift (observed, §11C.9):** anthropogenic labelling rate rose ~2011–2012.
- **Label noise:** catalog event-type labels can be wrong or inconsistent.
- **Shortcut learning:** see §12 (location, time of day, station identity).
- **Small OOD sets:** OOD metrics may have high variance; report intervals.
- **EDL pitfalls:** EDL uncertainty may not reflect epistemic uncertainty
  faithfully; compare with softmax references rather than assuming benefit.
- **Complexity creep:** the full architecture has many parts; each must earn
  its place (§6.4).

---

## 18. RESEARCH HYPOTHESES

> These are **hypotheses, not established facts.** Each must be tested
> experimentally; each may turn out to be false, and that outcome must be
> reported honestly.

| ID | Hypothesis | Planned test |
|---|---|---|
| **H1** | Spatial information between seismic stations improves event classification compared with an isolated waveform classifier. | Baseline 3 vs Baseline 1 (and graph ablation, §8) |
| **H2** | Waveform-informed graph relationships provide useful information beyond geographic distance alone. | geographic-only vs geographic + similarity edges |
| **H3** | Dynamic graph relationships improve representation quality compared with a fixed spatial graph. | Baseline 5 vs Baseline 3 |
| **H4** | EDL produces better calibrated uncertainty than standard softmax classification. | Baseline 2 vs 1, Baseline 4 vs 3, Final vs Baseline 5 on ECE / Brier / reliability |
| **H5** | EDL-based uncertainty provides useful information for detecting unfamiliar seismic events. | OOD AUROC/AUPRC of EDL uncertainty vs softmax references (MSP, entropy) — only if OOD events exist |
| **H6** | Uncertainty increases, and performance degrades gracefully, as stations are removed. | missing-station experiments (§10) |

Status of all hypotheses: **untested** (as of Phase 1).

Data feasibility (Phase 1, not evidence for or against any hypothesis):

- H1–H3: testable on PNW, but only a subset of events has enough stations
  (≥ 3 stations: 65.7 % of earthquakes, 39.8 % of anthropogenic events) —
  report results per station-count bin and control the station-count shortcut.
- H4: testable on S1 (note the class-prior shift between train and test).
- H5: testable for low-frequency, surface, sonic-boom and thunder events only;
  multi-station OOD sets are small (e.g. 129 `lf` and 722 surface events with
  ≥ 3 stations) — wide confidence intervals expected. Not testable for induced
  or geothermal events (absent).
- H6: testable — 21,104 ComCat events have ≥ 4 stations for station-removal experiments.

---

## 19. INSTRUCTIONS FOR FUTURE AI AGENTS

1. Read this file before modifying the project.
2. Do not assume unimplemented components exist (see the placeholder table in §16 and §22).
3. Do not overwrite architectural decisions without justification.
4. Check the current project phase (§22).
5. Work only within the requested phase.
6. Run existing tests before and after modifications (`python -m pytest`).
7. Do not fabricate data or results. Synthetic data are for unit tests only.
8. Do not silently change dataset assumptions.
9. Record important decisions in the Decision Log (§21).
10. Update the project state (§22) after completing work.
11. Preserve backwards compatibility where practical.
12. If a later architectural decision conflicts with an earlier one, document the change in the Decision Log (reference the superseded entry; do not delete it).
13. Do not rewrite functioning modules unnecessarily.
14. Prefer small, verifiable changes over large implementations.
15. Stop when the requested phase is complete.

Additional project-specific rules:

- Treat §12 (leakage) as blocking: no training results before a justified split exists.
- When you implement a placeholder, remove its `NotYetImplementedError`,
  update the placeholder table in §16, and replace the placeholder test.
- Keep `null` config values `null` until a value is justified; never guess.
- Never describe results as "state-of-the-art" or claim improvements without
  the evidence required by §6.
- Keep this file accurate and concise: update sections in place rather than
  appending contradictory text.

---

## 20. References (Starting Points)

Compiled from general knowledge in Phase 0 as a **starting point for the
literature review, not a completed review**. Verify every bibliographic
detail before citing.

- Sensoy, Kaplan & Kandemir (2018). *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS.
- Malinin & Gales (2018). *Predictive Uncertainty Estimation via Prior Networks.* NeurIPS.
- Ulmer, Hardmeier & Frellsen (2023). *Prior and Posterior Networks: A Survey on Evidential Deep Learning Methods for Uncertainty Estimation.* TMLR.
- Jürgens et al. (2024). *Is Epistemic Uncertainty Faithfully Represented by Evidential Deep Learning Methods?* ICML.
- Veličković et al. (2018). *Graph Attention Networks.* ICLR.
- Brody, Alon & Yahav (2022). *How Attentive are Graph Attention Networks?* ICLR.
- Yu, Yin & Zhu (2018). *Spatio-Temporal Graph Convolutional Networks.* IJCAI.
- Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
- Hendrycks & Gimpel (2017). *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.* ICLR.
- Geifman & El-Yaniv (2017). *Selective Classification for Deep Neural Networks.* NeurIPS.
- Jain & Wallace (2019). *Attention is not Explanation.* NAACL.
- van den Ende & Ampuero (2020). *Automated Seismic Source Characterization Using Deep Graph Neural Networks.* GRL.
- Linville, Pankow & Draelos (2019). *Deep Learning Models Augment Analyst Decisions for Event Discrimination.* GRL.
- Woollam et al. (2022). *SeisBench — A Toolbox for Machine Learning in Seismology.* SRL.

Added in Phase 1 (dataset-related; details verified only as far as stated in §11B):

- Ni, Hutko, Skene, Denolle, Malone, Bodin, Hartog & Wright (2023). *Curated Pacific Northwest AI-ready Seismic Dataset.* Seismica 2(1). doi:10.26443/seismica.v2i1.368 — **the dataset used; cite it.**
- *Exploration of Machine Learning Methods to Seismic Event Discrimination in the Pacific Northwest* (2025). arXiv:2510.23795 — related work on the same data (single-station inputs, event-level 80/20 split, surface-event label noise 1.5–8.4 %).
- Suarez & Beroza (2025). arXiv:2511.09805 — cited by the dataset maintainers for the Nov 2025 removal of samples containing undocumented events.
- *California Earthquake Dataset for Machine Learning and Cloud Computing* (2025). arXiv:2502.11500 — CEED (comparison candidate).
- Mousavi, Sheng, Zhu & Beroza (2019). *STanford EArthquake Dataset (STEAD).* IEEE Access (comparison candidate).
- *Deep learning and transfer learning of earthquake and quarry-blast discrimination: applications to southern California and eastern Kentucky* (2024). GJI 236(2):979 (comparison candidate; uses Miao et al., 2020 Kentucky data).
- AQMS Jiggle user guide, event types: aqms-swg.gitlab.io/jiggle-userdocs/eventtypes/ (label codes).

---

## 21. DECISION LOG

Template for new entries (append; never delete — supersede instead):

```text
Decision NNN
Date:
Phase:
Decision:
Reason:
Alternatives considered:
Consequences:
Status: Active | Provisional | Superseded by NNN
```

Future entries are expected for, e.g.: dataset selection, split strategy,
OOD strategy, preprocessing parameters, CNN architecture, graph construction
method, GAT variant/library, STGNN design, EDL loss and evidence activation.

Decisions 001–007 were taken in Phase 0 (engineering); 008–016 in Phase 1
(dataset, data access, labels, OOD, split design); 017–024 in Phase 2
(preprocessing, masks, split implementation, processed data). "Proposed" decisions are
designs whose implementation belongs to a later phase.

```text
Decision 001
Date: 2026-09-18
Phase: 0
Decision: src-layout Python package `seismic_ml` with one module per concern
          and phase-gated interface stubs for all future components.
Reason: Clean imports, testable from day one, lets later phases fill in
        modules without restructuring.
Alternatives considered: flat layout; single prototype script/notebook.
Consequences: Install with `pip install -e .` (pytest also works without it
              via pythonpath = ["src"]).
Status: Active
```

```text
Decision 002
Date: 2026-09-18
Phase: 0
Decision: Minimal runtime dependencies (numpy, torch, pyyaml; pytest for dev).
          SciPy, pandas, Matplotlib, scikit-learn, PyTorch Geometric, ObsPy
          and Jupyter are deferred to the phase that first needs them.
Reason: Phase 0 code imports nothing else; avoids committing to libraries
        (e.g. PyG vs a dense in-house GAT) before the need is demonstrated.
Alternatives considered: declaring the full expected stack up front.
Consequences: Each later phase adds its dependencies to pyproject.toml and
              records them here.
Status: Active
```

```text
Decision 003
Date: 2026-09-18
Phase: 0
Decision: Provisional tensor conventions: padded batches [B, N, C, T] with a
          bool station_mask [B, N]; dense adjacency [B, N, N] with
          A[b, i, j] > 0 meaning an edge from station j to station i.
Reason: Matches the conceptual [batch, stations, channels, time] input and
        the dense A_ij edge formulation; station counts per event are expected
        to be small (tens), where dense tensors are simple and efficient; the
        mask supports missing-station experiments.
Alternatives considered: PyTorch Geometric sparse edge_index batching
                         (better for large/variable graphs).
Consequences: Revisit in Phase 5/6 once station counts and the GAT library are
              known; switching affects graph/ and models/ interfaces.
Status: Provisional
```

```text
Decision 004
Date: 2026-09-18
Phase: 0
Decision: Four YAML files (config.yaml entry + data/model/training) loaded by a
          small PyYAML-based loader into plain dicts; `null` = undecided.
Reason: Transparent, no extra dependency, easy for agents to read and diff.
Alternatives considered: Hydra/OmegaConf (more features, extra dependency).
Consequences: Overrides/sweeps are not built in; migrating to Hydra later is
              possible and should be logged.
Status: Active
```

```text
Decision 005
Date: 2026-09-18
Phase: 0
Decision: Unimplemented interfaces raise NotYetImplementedError(component,
          phase), a NotImplementedError subclass; tests assert this behaviour.
Reason: Makes remaining work greppable, prevents silent use of stubs, and
        records the planned phase in code.
Alternatives considered: bare NotImplementedError; `pass` bodies.
Consequences: Implementing a stub requires updating its placeholder test and
              the table in §16.
Status: Active
```

```text
Decision 006
Date: 2026-09-18
Phase: 0
Decision: Event labels are strings (SeismicEvent.label: str | None) mapped to
          indices via configs/data.yaml `labels.class_names`, not a hard-coded
          Enum.
Reason: The label vocabulary is dataset-dependent and not yet known.
Alternatives considered: Python Enum of classes.
Consequences: Unknown labels must be rejected explicitly when batching.
Status: Active
```

```text
Decision 007
Date: 2026-09-18
Phase: 0
Decision: Separate the shared station encoder (features/cnn_encoder.py::
          CNNEncoder) from the non-graph Baseline 1 classifier
          (models/cnn.py::CNNBaseline).
Reason: The same encoder feeds every model (Baselines 1–5 and Final); keeping
        it separate avoids duplication and keeps comparisons fair.
Alternatives considered: a single CNN module per model.
Consequences: Graph models consume CNNEncoder outputs [B, N, D] / [B, N, T', D].
Status: Active
```

```text
Decision 008
Date: 2026-09-18
Phase: 1
Decision: Use the Curated PNW AI-ready Seismic Dataset (Ni et al., 2023): ComCat
          collection as primary supervised data, Exotic collection (plus the
          non-tectonic PNSN labels inside ComCat) for OOD evaluation. No second
          dataset is combined.
Reason: Only candidate with thousands of analyst-labelled anthropogenic events,
        event grouping with per-trace station coordinates, 3C waveforms,
        non-tectonic sources from the same network, open licence (CC BY 4.0),
        peer-reviewed documentation and practical partial access (§11A–B).
Alternatives considered: CEED (rare blasts, release mid-rebuild), QuakeFlow SC/NC
        (no event-type column, 8–33 GB per year), STEAD (no anthropogenic class),
        SCSN 2011–2020 and eastern-Kentucky study sets (availability unverified),
        self-built ComCat + FDSN (large effort, no curated picks).
Consequences: Few stations per event and strong shortcuts must be handled
              (§11C–F); induced/geothermal OOD cannot be studied; a second
              dataset (e.g. CEED quarry blasts) may be considered for
              geographic generalisation in Phase 10 (new decision needed).
Status: Active
```

```text
Decision 009
Date: 2026-09-18
Phase: 1
Decision: Storage locations are resolved at run time — CLI flag (--data-root /
          --output-root) > environment (SEISMIC_DATA_ROOT / SEISMIC_OUTPUT_ROOT)
          > configs/config.yaml > defaults; relative paths resolve against the
          project root. Raw data stay outside Git; manifests/provenance store
          paths relative to the data root.
Reason: The same code must run locally, on Colab with Google Drive, or on other
        storage without edits; user requirement (no personal machine paths).
Alternatives considered: fixed data/ directory; python-dotenv (.env loading).
Consequences: Every script/notebook resolves paths through seismic_ml.paths; a
              test fails if machine-specific paths are committed.
Status: Active
```

```text
Decision 010
Date: 2026-09-18
Phase: 1
Decision: (Proposed) PNSN analyst label `source_type_pnsn_label` is authoritative.
          Supervised classes: earthquake = eq; anthropogenic = px + ex + qb.
          The 80 events whose ComCat and PNSN labels conflict keep the PNSN
          label but are excluded from validation/test sets.
Reason: Dataset README recommends the PNSN label; ComCat `source_type` merges
        non-tectonic events into "earthquake"; conflicts would make evaluation
        labels unreliable.
Alternatives considered: ComCat source_type (leaks OOD events into training);
        dropping px (analyst-judged) events (would leave 34 anthropogenic events).
Consequences: The anthropogenic class is dominated by "probable explosions"
              whose labels partly rely on location near known quarries —
              location must never be an input (§12.5). Config: data.labels.class_map.
Status: Proposed (used from Phase 2)
```

```text
Decision 011
Date: 2026-09-18
Phase: 1
Decision: (Proposed) OOD candidates never used for training, tuning or
          thresholds: ComCat PNSN labels lf, su, sn (OOD-A) and all exotic
          events — surface event, sonic boom, thunder, plane crash (OOD-B).
          uk (1 event) is excluded entirely. No OOD set is created by splitting
          earthquakes.
Reason: Physically distinct unfamiliar sources recorded by the same network;
        they are absent from the proposed training classes (§11E).
Alternatives considered: holding out one anthropogenic subtype (too few ex/qb
        events); using noise windows as OOD (noise is not an event; possible
        later robustness test).
Consequences: Window harmonisation and station-identity controls are mandatory
              before OOD experiments (§11E); small multi-station OOD sets.
Status: Proposed (used from Phase 2/9)
```

```text
Decision 012
Date: 2026-09-18
Phase: 1
Decision: (Proposed) Split design: event is the unit; S1 temporal primary split
          (train 2002–2018, validation 2019–2020, test 2021–2022, with a
          boundary buffer); S2 grouped K-fold over spatial source clusters;
          S3 geographic region hold-out incl. a station-disjoint variant.
Reason: Measured repeat sites, swarms, labelling drift and station–class
        imbalance make random splits leaky (§11C, §11F).
Alternatives considered: random event-level 80/20 split (used in related work;
        leaks repeat sources); pure spatial split as primary (hides temporal drift).
Consequences: Class prior shifts from 12.5 % (train) to ~19 % (val/test);
              results must be reported for seen vs new source sites.
Status: Proposed (implemented in Phase 2)
```

```text
Decision 013
Date: 2026-09-18
Phase: 1
Decision: Acquire metadata files in full; acquire waveforms as seeded,
          event-complete exploration subsets read from the remote HDF5 files by
          HTTP byte ranges into local files with the original SeisBench layout
          (resumable; unwritten traces occupy no space). Provenance (URL, ETag,
          size, SHA-256, selection parameters) is written to
          <raw>/pnw/PROVENANCE.json.
Reason: The ComCat waveform file is 67 GB; Phase 1 needs representative
        waveforms, not all of them; the server supports byte ranges and the
        HDF5 datasets are contiguous, so exact trace bytes can be fetched.
Alternatives considered: full download (67 GB, not needed yet); mesoPNW/miniPNW
        demo subsets (construction not documented as event-complete); SeisBench
        downloader (whole files only).
Consequences: Later phases can fetch more traces with the same code or place the
              full files in <raw>/pnw/ — the loader prefers full files.
Status: Active
```

```text
Decision 014
Date: 2026-09-18
Phase: 1
Decision: Add h5py and pandas as core dependencies; matplotlib, ipykernel,
          nbclient, nbformat as the optional `notebooks` extra. Do not adopt
          SeisBench or ObsPy.
Reason: HDF5 waveforms and CSV catalogues must be read by the loader; plotting
        and notebook execution are only needed for exploration.
Alternatives considered: SeisBench (convenient, but whole-file downloads and a
        heavy dependency tree); csv module only (clumsy for 179k-row analysis).
Consequences: Revisit ObsPy if Phase 2 chooses instrument-response removal.
Status: Active
```

```text
Decision 015
Date: 2026-09-18
Phase: 1
Decision: Graph-node identity is the site "NET.STA" (unique within every event,
          observed); station coordinates are taken from each trace's own row;
          waveform arrays keep the stored component order E, N, Z (no reordering
          in the loader).
Reason: 17 ComCat sites change coordinates over time; location/channel codes
        vary per event while the site is the physical node.
Alternatives considered: NET.STA.LOC.CHA as node (splits one site into several
        nodes); a static station table (wrong coordinates for moved sites).
Consequences: Location and band codes are kept in StationObservation.metadata.
Status: Active
```

```text
Decision 016
Date: 2026-09-18
Phase: 1
Decision: SeismicEventDataset(metadata, waveform_file, *, label_column,
          event_ids=None, split=None) with lazy per-process HDF5 opening;
          from_config() builds it from configs/data.yaml. A split is a list of
          event IDs. Supersedes the Phase 0 placeholder signature
          (root, split, config).
Reason: The dataset needs the parsed metadata and a waveform file; h5py handles
        cannot be pickled into DataLoader workers; event-ID lists enforce the
        event-level split rule.
Alternatives considered: keeping (root, split, config) and parsing inside.
Consequences: Phase 2 collate/batching builds on SeismicEvent items.
Status: Active
```

```text
Decision 017
Date: 2026-09-19
Phase: 2
Decision: Add SciPy (>= 1.11) as a core dependency (user-approved install, 1.17.1).
Reason: Butterworth design, zero-phase filtering and polyphase resampling are
        standard, tested implementations; hand-written NumPy versions would add risk.
Alternatives considered: NumPy-only implementations; ObsPy (heavier, not needed).
Consequences: pyproject.toml updated; ObsPy still not adopted.
Status: Active
```

```text
Decision 018
Date: 2026-09-19
Phase: 2
Decision: Band-pass 1–16 Hz, Butterworth order 4, zero-phase (sosfiltfilt),
          applied to the full trace after linear detrend and a 2 s cosine taper.
Reason: BH channels contain no energy above ~20 Hz (upsampled from 40 sps) → an
        upper corner below that removes an instrument fingerprint; < 1 Hz is
        microseism-dominated (SNR ~1) and where EH and broadband responses differ
        most; zero-phase keeps P onsets at the pick (§11H, notebook 02).
Alternatives considered: 0.5–16 Hz (keeps low-frequency explosion energy, adds
        instrument differences) — planned ablation; causal filtering (phase delay);
        up to 45 Hz (instrument fingerprint).
Consequences: some 0.5–1 Hz energy of explosions / lf events is attenuated.
Status: Active
```

```text
Decision 019
Date: 2026-09-19
Phase: 2
Decision: Resample 100 -> 50 Hz (polyphase), inside a margin that is trimmed afterwards.
Reason: 50 Hz > 2 x 16 Hz; exact factor 2; halves memory and compute; the margin
        keeps FIR edge effects out of the window and puts P on an exact sample.
Alternatives considered: keep 100 Hz (twice the data, no information above 16 Hz);
        40 Hz (non-integer factor 2.5).
Consequences: T = 3000 samples per 60 s window.
Status: Active
```

```text
Decision 020
Date: 2026-09-19
Phase: 2
Decision: Per-station P-aligned window from P − 10 s to P + 50 s for every
          collection; the window must lie >= 5 s inside the trace; traces without
          a P pick or with too little data are dropped (reasons recorded).
Reason: Contains P and S for ~all traces (99 % S−P ≤ 19.5 s) and the full signal
        for ~80 %; removes the origin- vs pick-aligned collection artefact
        (§11C.4) that would make OOD detection trivial.
Alternatives considered: event-common time axis (keeps moveout, which encodes
        location/depth — label proxies); longer windows (more coda, fewer usable
        traces); S-aligned windows for traces without P (would re-introduce a
        collection-specific alignment).
Consequences: moveout across stations is removed from the waveforms (kept as
              p_time_offset_s metadata); coda of large/long events truncated.
Status: Active
```

```text
Decision 021
Date: 2026-09-19
Phase: 2
Decision: Normalise by the standard deviation of the station's live components
          (one scale per station, per trace window).
Reason: Preserves Z/H and P/S ratios within a station; inter-station amplitudes
        are raw counts of different instruments and not comparable; std is less
        sensitive than peak to telemetry spikes; per-sample statistic → no leakage.
Alternatives considered: channel scope (destroys Z/H ratios); event scope (would
        mix instrument gains); peak normalisation; response removal (needs
        StationXML/ObsPy — possible ablation).
Consequences: absolute amplitude (magnitude/distance proxy) is not in the input;
              the scale is stored per trace (traces.csv norm_scale) for analysis only.
Status: Active
```

```text
Decision 022
Date: 2026-09-19
Phase: 2
Decision: Keep stations with missing components; missing components are exact
          zeros and are marked by a channel mask. EventBatch gains an optional
          channel_mask [B, N, C] (backward compatible; validated against station_mask).
Reason: 28 % of ComCat traces are vertical-only (mostly EH); dropping them would
        remove many stations and bias the station set.
Alternatives considered: drop incomplete stations; impute components.
Consequences: models must consume channel_mask (Phase 3+). Decision 003 remains
              provisional with this extra field.
Status: Active
```

```text
Decision 023
Date: 2026-09-19
Phase: 2
Decision: Implement S1 exactly as designed in Decision 012: event unit; train
          origin < 2019-01-01, val < 2021-01-01, test after; 7-day pre-boundary
          buffer; label conflicts excluded from val/test (kept in train with the
          PNSN label); duplicate candidates never split; OOD labels → split "ood";
          uk → excluded; unmapped labels raise; site_seen_in_train flag for val/test.
Reason: Leakage controls of §12/§11F; every exclusion carries a reason.
Alternatives considered: random event-level split (leaks repeat sources).
Consequences: 88 % of test explosions are at source sites seen in training →
              report new-site results separately (§11I).
Status: Active (supersedes the "Proposed" status of Decisions 010–012 for S1)
```

```text
Decision 024
Date: 2026-09-19
Phase: 2
Decision: Processed data are stored per collection (waveforms.hdf5 + traces.csv +
          dropped_traces.csv + preprocessing.json) independent of the split;
          ProcessedEventDataset joins the split table at read time; collate_events
          pads to the largest event (no truncation), rejects labels outside
          class_names and encodes OOD events as -1 only with allow_unlabeled=True.
Reason: Re-splitting or adding splits (S2/S3) must not require re-processing;
        silent truncation or label mapping would hide data decisions.
Alternatives considered: one file per split; fixed padding N = 41.
Consequences: create_dataloader (Phase 3) wraps ProcessedEventDataset + collate_events.
Status: Active
```

---

## 22. CURRENT PROJECT STATE

> Future agents must update this section after completing each phase.

```text
## CURRENT PROJECT STATE

Current Phase:
Phase 3 — CNN Baseline (COMPLETE, 2026-10-22)

Completed:
Phase 0
- Repository structure, configuration, schemas, module interfaces, documentation,
  config/logging/seeding helpers, tests
Phase 1
- Dataset research/comparison, PNW selected (ComCat + Exotic) — Decision 008
- Configurable data/output locations (flag > env > config; Colab/Drive) — Decision 009
- Metadata + event-complete waveform exploration subsets (1.94 GB) with provenance
- Loader, SeismicEventDataset, exploration notebook 01, manifest, quality report
- Label/OOD/split designs (§11A–11G), Decisions 008–016
Phase 2
- Preprocessing chain (detrend, taper, 1–16 Hz zero-phase band-pass, P-aligned
  60 s window, 50 Hz, station-scope std normalisation, quality checks) with every
  parameter justified from data (§11H, notebook 02) — Decisions 018–021
- Missing-component policy: exact zeros + channel_mask (EventBatch.channel_mask) — Decision 022
- S1 temporal split implemented for all 51,711 events (§11I) — Decision 023
- Processed store + ProcessedEventDataset + collate_events -> EventBatch — Decision 024
- scripts/prepare_data.py (split + preprocessing, --data-root aware)
- Processed outputs for the local subsets verified (P at sample 500 in every window,
  no NaN, masked components zero, unit std)
- SciPy dependency (user-approved) — Decision 017
Phase 3
- DataLoader: `create_dataloader` implemented with seeded batching and `collate_events`.
- Station-level 1D Dilated Residual CNN / TCN encoder: `CNNEncoder` implemented with
  dilations (1, 2, 4, 8), residual connections, dropout, and station-mask zeroing.
- Baseline 1 Event Classifier: `CNNBaseline` with station pooling ("mean", "max",
  "attention") and linear head to logits.
- Losses & Metrics: `cross_entropy_loss`, `compute_classification_metrics`, and `confusion_matrix`.
- Trainer: `Trainer` implemented with AdamW, Cosine Annealing, early stopping,
  and checkpoint saving.
- Runnable Entry Points: `scripts/train.py` and `scripts/evaluate.py`.
- Baseline configuration in `configs/model.yaml` and `configs/training.yaml`.

In Progress:
None

Next Phase:
Phase 4 (Spectrogram experiments) / Phase 5 (Spatial Graph Construction)

Not Yet Implemented:
- spectrogram pipeline (Phase 4)
- graph construction (Phase 5)
- GATv2 (Phase 6)
- STGNN (Phase 7)
- EDL evidential head & Bayes-risk loss (Phase 8)
- OOD evaluation (Phase 9)
- final experiments & ablations (Phase 10-11)

Last verification:
2026-10-22 — `python -m pytest`: 251 passed, 0 failed.

Corrections to earlier phases:
- The Phase 1 notebook 01 as delivered contained double-encoded (mojibake) text in
  all non-ASCII characters (jupyter-execute read UTF-8 as cp1252 on Windows).
  Numbers and figures were unaffected. Fixed in Phase 2: notebook regenerated and
  re-executed with scripts/run_notebooks.py; tests/test_paths.py now fails on
  mojibake in any text file.
(Windows 11, Python 3.11.9, PyTorch 2.5.1+cu121, NumPy 2.3.5, SciPy 1.17.1,
h5py 3.16.0, pandas 3.0.6)

Experimental results so far:
None. Data have been preprocessed and split; no model has been trained or evaluated.
```
