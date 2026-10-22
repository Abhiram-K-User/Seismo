# Architecture and Implementation Specification
## Uncertainty-Aware Spatio-Temporal GNN for Seismic Event Classification

**Purpose:** This document is the implementation brief for the AI coding agent. Read it together with `README.md` and `PROJECT_CONTEXT.md`. It specifies the intended architecture, the changes to the current plan, the order of implementation, and the checks required before claiming success.

**Research reference:** Zhang et al., *Enhanced source discrimination between tectonic earthquakes and quarry blasts via a hybrid CNN–GNN trained with multiple stations*, *Geophysical Journal International* (2026), DOI: `10.1093/gji/ggag322`. The paper is an architectural reference, not a specification to copy unchanged.

---

## 1. Decisions at a glance

1. **Keep the PNW dataset and existing Phase 1/2 work.** Do not replace the dataset with Utah data and do not redo completed preprocessing without evidence that it is unsuitable.
2. **Use the paper's hybrid design as the baseline:** station-level CNN feature extraction, a graph whose nodes are recording stations, GATv2 message passing, attention-based graph pooling, and an event-level classifier.
3. **Use `GATv2Conv`, not a generic/unspecified GAT layer, for the primary graph-attention model.** Keep the layer configurable.
4. **Add explicit temporal modeling** if the project continues to claim a *spatio-temporal* GNN. The reference paper's CNN learns time-frequency patterns from each station's spectrogram, while its graph module primarily aggregates across stations. That alone should not be described as a dedicated temporal GNN.
5. **Add an Evidential Deep Learning (EDL) classification head** as the proposed extension. It outputs non-negative class evidence, parameterizes a Dirichlet distribution, and reports class probabilities plus an uncertainty measure. Do not use Softmax as the final layer of the EDL model.
6. **Build and evaluate baselines before the full model.** Required comparison: CNN baseline, CNN + GATv2 + Softmax (paper-inspired baseline), and temporal CNN + GATv2 + EDL (proposed model).
7. **Do not assume EDL automatically detects every OOD event.** Measure uncertainty on held-out in-distribution data and the project's candidate OOD/exotic sources; report empirical results.
8. **Preserve leakage protections.** Split by event, never by individual station trace. All traces belonging to an event must stay in the same partition.

## 2. What the reference paper does

The paper's core pipeline is:

1. Preprocess station waveforms and convert them to spectrograms.
2. Treat each event as a graph. Each recording station is a node, and its spectrogram is the node input.
3. Connect stations based on geographic distance. The paper uses a 300 km threshold and inverse-distance edge weights, configured so stations recording an event are mutually connected in its experiments.
4. Run the same CNN feature extractor on each station's spectrogram, producing one compact feature vector per station.
5. Apply a `GATv2Conv` layer to propagate information between stations.
6. Apply attention-based graph pooling to produce one representation for the whole event.
7. Use linear layers and Softmax to classify the event as tectonic earthquake or quarry blast.

The paper also uses random subgraph sampling during training to expose the model to different numbers of available stations. It reports event-level train/validation/test partitioning based on event time.

### What to borrow

- A shared CNN encoder for each station observation.
- A graph representation of one event with stations as nodes.
- GATv2 attention/message passing across stations.
- Graph-level attention pooling.
- Variable station-count support.
- An event-level split and careful comparison with a CNN baseline.

### What not to copy blindly

- Do **not** copy the Utah dataset's exact frequency range, sample rate, window length, spectrogram dimensions, or distance threshold without validating them against PNW and the completed preprocessing pipeline.
- Do **not** assume the PNW labels map one-to-one to Utah's binary earthquake/quarry-blast labels.
- Do **not** claim explicit temporal graph learning if the implementation only extracts a single station embedding and performs spatial graph aggregation.
- Do **not** replace EDL with the paper's Softmax head in the proposed model. Softmax is required only for the paper-inspired baseline.
- Do **not** quote the paper's performance as an expected result for PNW. Results must be measured on this project's splits.

## 3. Project objective and task definition

### Primary task

Classify a multi-station seismic event using the observations available from its recording stations.

The current README frames the task as tectonic earthquakes versus anthropogenic seismic events such as explosions/quarry blasts. Keep the primary task aligned with the actual labels supported by the selected PNW subset. Before training, inspect the label mapping and decide explicitly whether the first experiment is:

- **Binary:** earthquake vs. explosion/quarry blast; or
- **Multiclass:** the supported labelled source categories.

Start with the binary task if the existing label quality and project scope make that the most defensible comparison with the reference paper. Record the final mapping in configuration/documentation; do not silently merge or relabel categories. Keep ambiguous, conflicting, and unlabelled events out of supervised training unless a documented policy handles them.

### OOD evaluation

The PNW exotic collection (e.g., surface events, sonic booms, thunder, and other unusual sources) is a **candidate OOD evaluation set**, not automatically a clean OOD benchmark. Verify labels, event grouping, waveform availability, and station coverage first.

For the initial OOD protocol:
- Train the classifier on the selected in-distribution classes only.
- Keep selected exotic/unseen source types out of supervised training.
- Evaluate them separately as candidate OOD events.
- Do not use OOD labels to tune the model repeatedly against the final test set.
- If the data includes source types overlapping the in-distribution categories, document the overlap and exclude ambiguous cases from the OOD claim.

EDL uncertainty is a hypothesis to test, not a guarantee of OOD detection.

## 4. Target architecture

### High-level flow

```text
One event
  └── variable number of station observations
       └── each observation: 3-component waveform + station/event metadata
            ↓
       Existing Phase 2 preprocessing
            ↓
       Shared station encoder (CNN / time-frequency encoder)
            ↓
       Station feature sequence / temporal representation
            ↓
       Event graph (nodes = stations; edges = documented spatial relation)
            ↓
       GATv2 message passing
            ↓
       Attention-based graph pooling
            ↓
       Event-level representation
            ↓
       EDL evidence head
            ↓
       Dirichlet parameters α
            ├── expected class probabilities
            └── uncertainty score
```

The implementation must retain enough temporal information for the temporal module to operate. Do not collapse all time information irreversibly before the temporal module.

### 4.1 Input and batching contract

Use the existing project schemas and `EventBatch` contract wherever possible. Do not create a parallel data representation without a concrete reason.

Each event should expose:
- event identifier and label (when available);
- a variable number of station observations;
- preprocessed three-component waveform windows;
- station identifiers and geographic coordinates, when valid;
- per-component validity/channel masks and any existing quality metadata;
- the event's partition (train/validation/test).

Important rules:
- Missing components must remain distinguishable from genuine zero-valued waveform data. Preserve the existing masks.
- A station with invalid or absent coordinates must not silently create invalid distances. Use a documented fallback (e.g., exclude that edge/observation or use a separately configured topology).
- Handle events with one station and very few stations. Graph code must not assume every graph has the same number of nodes.
- Keep event IDs attached through preprocessing, batching, graph construction, prediction, and metric calculation.
- If an event has no usable station observations, skip it with a logged reason rather than generating a fake graph.

### 4.2 Preprocessing

The current Phase 2 README states:
- 1–16 Hz filtering;
- P-aligned 60-second windows at 50 Hz;
- station-level normalization;
- channel masks;
- a leakage-aware S1 temporal split across 51,711 events.

Treat this as the current data contract. Do not change it simply to match the paper's Utah preprocessing. First inspect `PROJECT_CONTEXT.md`, configuration, preprocessing code, and existing tests.

If experimenting with spectrograms:
- implement them as a configurable feature path or separate experiment, not as an unreviewed change to the completed pipeline;
- use parameters justified by PNW sampling/window properties;
- document tensor dimensions and test them;
- ensure normalization statistics do not use validation/test data.

The reference paper found spectrogram input useful for its Utah task, but that does not prove spectrograms are optimal for this PNW pipeline. Compare raw-waveform and/or time-frequency inputs experimentally where feasible.

### 4.3 Station-level CNN encoder

Implement a **shared-weight** encoder: the same CNN processes every station's input. It must not use station-specific learned parameters merely because a station has a particular ID.

Input options:
- **Initial baseline:** preprocessed waveform tensor with three components and the existing masks.
- **Optional experiment:** a time-frequency representation derived from each station's waveform.

The encoder must:
- accept variable numbers of stations through batching, not by hard-coding a station count;
- handle channel masks correctly;
- output a fixed-size embedding per station;
- make its output dimension configurable;
- support standard training utilities (device placement, gradients, dropout, reproducible initialization);
- have unit tests for shape, finite values, batch behavior, and backward pass.

Do not overdesign the CNN before a baseline runs. Begin with a modest 1D CNN for raw waveforms or a modest 2D CNN for spectrograms, then compare if time permits.

### 4.4 Explicit temporal modeling

The project is titled *Spatio-Temporal GNN*. Implement an explicit temporal component and describe precisely what it models.

Recommended first option: **temporal convolutional network (TCN)** or a **GRU** over time-resolved station features. Choose one for the first complete model; do not implement several temporal architectures at once.

Preferred conceptual options:

**Option A — temporal encoder before graph aggregation**
1. CNN produces a sequence of features over waveform/spectrogram time.
2. TCN/GRU models the temporal evolution for each station.
3. The result is one station embedding (or a compact sequence if the graph layer is designed to preserve time).
4. GATv2 aggregates station-level information.

This is the simplest stable first implementation. Be accurate in the write-up: it is a temporal encoder followed by spatial graph reasoning, not necessarily a graph convolution repeated at every time step.

**Option B — graph message passing at multiple time steps**
1. CNN produces time-indexed features for each station.
2. For each temporal slice/window, build node features and apply GATv2 using the event's station graph.
3. A TCN/GRU aggregates the resulting graph-level sequence.
4. The EDL head classifies the final event representation.

This more directly models time-varying inter-station information but is more complex. Use it only after Option A works, or if the current project design already explicitly requires graph message passing at each time step.

Do not use both options in the first implementation. Document the chosen option and why.

### 4.5 Event graph construction

**Graph semantics:** one graph represents one seismic event; each node represents one station recording that event. Do not make a single graph containing all events or all stations in the dataset.

Initial topology:
- Use station geographic coordinates to calculate pairwise great-circle distance where coordinates are valid.
- Begin with a configurable radius graph or k-nearest-neighbor graph.
- Log node counts, edge counts, disconnected nodes, and coordinate failures.
- Add self-loops if required by the selected graph layer/library, and test the behavior.
- Make edge construction deterministic for a fixed configuration.

Do not blindly set the radius to the paper's 300 km. That value was selected for the Utah network. Choose the initial PNW topology based on station-distance distributions and event coverage, then perform sensitivity analysis. A kNN graph can be a useful comparison if radius graphs create too many disconnected graphs or very dense graphs.

Edge attributes/weights:
- The paper uses inverse geographic distance as an edge weight.
- If using inverse distance, guard against zero distance and define units/normalization.
- Verify that the chosen `GATv2Conv` implementation actually consumes edge attributes/weights as intended. Do not calculate an edge weight and then assume it affects attention if the layer ignores it.
- Keep a no-edge-weight baseline to establish whether the weights help.
- Do not introduce event-to-station distance unless reliable event locations are available and the project explicitly decides to use them.

Graph invariants to test:
- one graph per event;
- every node maps to a real station observation;
- no cross-event edges;
- edge indices stay within the graph/batch;
- batching multiple variable-sized graphs works;
- graph construction is deterministic;
- isolated nodes and one-node graphs do not crash.

### 4.6 GATv2 graph attention layer

Use `GATv2Conv` as the primary graph layer, preferably through PyTorch Geometric if compatible with the project's installed stack.

Responsibilities:
- consume one node embedding per station (or per station/time step for Option B);
- exchange information along the event graph's edges;
- learn feature-dependent attention across neighbors;
- support variable node counts;
- output a configurable embedding dimension.

Implementation requirements:
- make hidden dimension, number of heads, dropout, and number of graph layers configurable;
- handle multi-head output dimensions deliberately (concatenation vs. averaging);
- include normalization and dropout where appropriate;
- keep the first version shallow (e.g., one or two graph-attention layers) and test for over-smoothing/overfitting before adding depth;
- optionally expose attention coefficients for analysis, but do not treat them as a complete explanation of causal station importance;
- test forward/backward passes on toy graphs with different node counts.

The reference paper uses a GATv2 convolution, then normalization/dropout and attention-based graph pooling. Its exact dimensions are a reference only, not mandatory values for PNW.

### 4.7 Graph-level attention pooling

After message passing, reduce the variable number of station embeddings to **one event embedding**.

Use attention-based graph pooling (for example, PyTorch Geometric's `AttentionalAggregation` or a compatible equivalent). The pooling must operate separately per event in a batched graph.

Requirements:
- output shape `[batch_size, event_embedding_dim]`;
- work with different station counts per event;
- not mix nodes from different events;
- handle one-node graphs;
- test that changing the number of nodes does not change output shape.

Mean pooling can be retained as an ablation/baseline. Do not use a station-level classifier followed by majority vote as the primary proposed model: the reference architecture classifies at the graph/event level.

### 4.8 Evidential Deep Learning (EDL) head

The proposed model's final head must output **non-negative evidence**, not direct Softmax probabilities.

For `K` supervised classes:
1. The event representation passes through a linear evidence layer.
2. Apply a non-negative activation such as `Softplus` (recommended for smoother gradients) or `ReLU`.
3. Convert evidence to Dirichlet parameters: `alpha_i = evidence_i + 1`.
4. Compute `S = sum(alpha_i)`.
5. Compute expected class probabilities: `p_i = alpha_i / S`.
6. Compute vacuity uncertainty: `u = K / S`.

Interpretation:
- `p_i` is the expected class probability under the predicted Dirichlet distribution.
- `u` is a simple evidence-based vacuity measure; higher values indicate less total evidence.
- Do not describe `u` as a guaranteed detector of all OOD cases or as a calibrated probability of being OOD.

Training:
- Implement a documented EDL loss suitable for classification (e.g., an evidential classification loss with an explicit evidence/regularization term).
- Define how incorrect-class evidence is penalized and how any annealing schedule is configured.
- Ensure the loss is numerically stable and test it on toy examples.
- Do not simply apply cross-entropy to the expected probabilities and claim that this alone implements the full EDL method unless that is an explicit, justified experimental variant.
- Keep the loss configuration and any annealing schedule in the config file.
- Track both classification metrics and uncertainty metrics during validation.

Required tests:
- evidence is non-negative and finite;
- `alpha_i >= 1`;
- class probabilities are non-negative and sum to 1;
- uncertainty is finite and in its expected range;
- gradients flow through the head and loss;
- class-count changes are supported by configuration.

For the paper-inspired baseline only, use a conventional linear classifier and Softmax. Clearly label that model as the baseline, not the EDL model.

## 5. Training, validation and data leakage

### Split policy

The README says an S1 temporal split has already been built across 51,711 events. Inspect `PROJECT_CONTEXT.md` and the split artifacts to confirm exact boundaries and counts. Preserve the split unless a documented data-quality issue requires rebuilding it.

Strict rules:
- Split by **event ID**, before creating station-level training samples or graph augmentations.
- Every station trace from one event belongs to exactly one partition.
- Never allow a graph/subgraph from a training event into validation or test.
- Fit normalization/statistical transforms only on training data where applicable.
- Tune hyperparameters using validation data only.
- Use the test set once for final reporting after model selection.
- Record the split seed, time boundaries, class counts, and event IDs/manifest provenance.

### Variable station counts and augmentation

First make the model work with naturally occurring station counts. Then, if needed, add training-only station-subgraph sampling inspired by the paper.

If subgraph augmentation is implemented:
- apply it only to training events;
- retain the original event label;
- sample station subsets without creating duplicate/cross-event nodes;
- ensure at least one valid node remains;
- use a reproducible random generator;
- never generate augmented validation/test examples for model selection;
- evaluate robustness on the untouched test events and controlled station-dropout tests.

Do not apply augmentation automatically just because the paper does. Measure whether it improves validation metrics and station-count robustness.

### Optimization

- Use a reproducible seed and record the full configuration for every run.
- Start with Adam or AdamW and a conservative learning rate.
- Use early stopping/model selection based on validation macro F1 or a documented primary metric.
- Track per-class precision, recall, F1, macro F1, confusion matrix, and loss.
- Account for class imbalance using training-only class weights or a documented sampler/loss; do not rebalance validation/test to conceal real class prevalence.
- Save model checkpoint, configuration, label mapping, preprocessing metadata, and metrics together.
- Avoid selecting the best model by test-set performance.

## 6. Required experiment ladder

Implement in this order; keep each model runnable and record results before advancing.

### Experiment 0 — Data/graph smoke tests
- Build graphs from a small number of real processed events.
- Print/report nodes per event, edges per event, valid components and label distribution.
- Visualize a few graph topologies if practical.
- Verify event IDs and split membership.
- Run unit tests and a single batch through the full data path.

### Experiment 1 — CNN baseline
- Use the station encoder and a documented station-to-event aggregation strategy.
- For the initial baseline, a simple masked mean of station embeddings or station-level predictions aggregated to an event prediction is acceptable; document which is used.
- This baseline tests whether the CNN learns useful single-station/local features without graph attention.
- Report event-level metrics and limitations of the aggregation.

### Experiment 2 — Paper-inspired CNN + GATv2 + attention pooling + Softmax
- Use the shared station CNN encoder.
- Construct one station graph per event.
- Apply GATv2 and attention-based graph pooling.
- Use a conventional Softmax classifier.
- This is the closest fair implementation of the 2026 reference paper's core architecture adapted to PNW.
- Do not call it the proposed uncertainty-aware model.

### Experiment 3 — Add explicit temporal modeling
- Add the chosen TCN or GRU as described in Section 4.4.
- Keep other components fixed as far as practical so the contribution of temporal modeling can be measured.
- Compare against Experiment 2.

### Experiment 4 — Replace Softmax with EDL
- Replace the final Softmax classifier with the EDL evidence head and Dirichlet calculations.
- Use an appropriate evidential loss.
- Compare classification performance and uncertainty behavior against the Softmax model.
- Do not expect EDL to necessarily improve accuracy; its key intended contribution is uncertainty estimation.

### Experiment 5 — OOD and uncertainty evaluation
- Evaluate on held-out in-distribution test events.
- Evaluate separately on candidate PNW exotic events excluded from training, after verifying their labels and data quality.
- Report AUROC/AUPRC for OOD-vs-ID using the selected uncertainty score where the evaluation protocol supports these metrics.
- Report uncertainty distributions for correct vs. incorrect ID predictions.
- Consider risk-coverage/selective-prediction curves: how error changes when high-uncertainty predictions are abstained from.
- Compare with a Softmax baseline uncertainty score (e.g., maximum-softmax probability or predictive entropy), using the same data partitions.
- State limitations, including possible class overlap and domain shift.

### Experiment 6 — Ablations and robustness
At minimum, if compute/data permit:
- CNN only vs. CNN + GATv2;
- GATv2 with vs. without distance-derived edge information;
- mean pooling vs. attention pooling;
- without vs. with explicit temporal module;
- Softmax vs. EDL;
- varying station counts / station dropout;
- optional raw waveform vs. spectrogram input.
Report what changed, not only the best score.

## 7. Metrics and interpretation

Classification:
- macro F1 as a primary metric if classes are imbalanced;
- per-class precision, recall, F1;
- confusion matrix;
- accuracy as a supplementary metric, not the only metric.

Uncertainty/OOD:
- AUROC and AUPRC for separating candidate OOD from in-distribution events, if the labels and protocol make the comparison meaningful;
- uncertainty distributions by correct/incorrect predictions;
- risk-coverage or selective accuracy;
- calibration metrics where implemented correctly.

Do not compare a model trained on one label mapping to a model evaluated on a different mapping without explicitly explaining the difference. Do not report metrics from station traces as if they were event-level metrics.

## 8. Implementation and repository requirements

Before editing, inspect:
- `PROJECT_CONTEXT.md`;
- `configs/`;
- schemas and `EventBatch`;
- preprocessing and processed-store code;
- existing model/graph/uncertainty interfaces;
- `tests/` and dependency definitions.

Then implement through the existing package layout:
- `src/seismic_ml/features/`: station CNN/time-frequency encoder;
- `src/seismic_ml/graph/`: per-event topology and graph batching;
- `src/seismic_ml/models/`: GATv2, temporal module, graph pooling, event classifier;
- `src/seismic_ml/uncertainty/`: evidence head, Dirichlet calculations, EDL loss;
- `src/seismic_ml/training/`: train/validation loops, checkpointing, early stopping;
- `src/seismic_ml/evaluation/`: classification and uncertainty/OOD metrics;
- `scripts/train.py` and `scripts/evaluate.py`: real command-line entry points, replacing placeholders when the underlying components are implemented;
- `configs/`: configurable dimensions, graph policy, temporal model, loss, optimizer, metrics and paths.

Respect the current path/config system (`--data-root`, `SEISMIC_DATA_ROOT`, config defaults). Do not hard-code user-specific paths. Raw data must remain outside version control.

Do not rewrite completed Phase 1/2 code wholesale. Make small, tested changes and update documentation if an intentional interface change is necessary.

## 9. Required acceptance criteria

The implementation is not complete until all applicable checks pass:

### Data and graph
- [ ] Existing data pipeline and Phase 1/2 tests still pass.
- [ ] Real processed events can be loaded into `EventBatch`.
- [ ] Graphs are event-local and contain only stations for that event.
- [ ] Variable-size graphs batch and unbatch correctly.
- [ ] Missing components and invalid coordinates are handled explicitly.
- [ ] No event appears in more than one split.
- [ ] Graph construction has tests for zero/one/few/many nodes and disconnected graphs.

### Model
- [ ] CNN outputs one finite, fixed-size embedding per station (or a documented time sequence).
- [ ] Temporal module demonstrably processes time-indexed features.
- [ ] GATv2 forward/backward passes work on batched variable-size graphs.
- [ ] Graph pooling returns one embedding per event without mixing events.
- [ ] Softmax baseline returns valid normalized probabilities.
- [ ] EDL returns finite evidence, alpha, expected probabilities and uncertainty.
- [ ] Gradients flow through all trainable components.
- [ ] CPU toy tests run; GPU use is optional and must not be required for tests.

### Training/evaluation
- [ ] `train.py` can run a short smoke training job from configuration.
- [ ] Checkpoints and experiment metadata are saved.
- [ ] `evaluate.py` loads a checkpoint and emits event-level metrics.
- [ ] EDL uncertainty metrics are computed separately from classification metrics.
- [ ] Candidate OOD evaluation is kept separate from supervised training.
- [ ] Results include the configuration and label mapping used.

### Reproducibility/documentation
- [ ] Tests pass and no placeholder path is mistaken for a completed implementation.
- [ ] README and `PROJECT_CONTEXT.md` status tables are updated accurately.
- [ ] Every reported result is produced by an actual run; no results are fabricated or inferred from the reference paper.
- [ ] Experiments and limitations are recorded.

## 10. Explicit changes to the current README plan

The current README roadmap is broadly sound, but update it as follows:

| Current plan | Required adjustment |
|---|---|
| “CNN / time-frequency encoder” | Keep; first implement a simple working encoder and treat spectrograms as an experiment rather than assuming they are best for PNW. |
| “dynamic seismic graph” | Define explicitly as one graph per event, with station observations as nodes and a configurable, data-justified spatial topology. “Dynamic” should not imply edges vary over time unless that is implemented. |
| “graph attention network” | Specify **GATv2Conv** for the paper-inspired model. |
| “spatio-temporal fusion” | Make temporal modeling explicit (TCN or GRU first); decide whether temporal features are encoded before GATv2 or graph message passing is applied at multiple time steps. |
| “evidential classifier → Dirichlet” | Specify non-negative evidence, `alpha = evidence + 1`, expected probabilities, vacuity uncertainty, and an appropriate EDL loss. |
| Roadmap starts with CNN baseline, then time-frequency experiments, graph construction, GAT, STGNN, EDL | Add a runnable **CNN + GATv2 + attention pooling + Softmax** paper-inspired baseline before the full STGNN+EDL model. Compare stages through controlled experiments. |
| OOD + uncertainty evaluation | Keep, but explicitly compare EDL against Softmax uncertainty baselines and evaluate on candidate exotic events only after label/overlap checks. |
| Robustness experiments | Include station-count/dropout robustness and graph-topology sensitivity, inspired by the paper. |

Suggested revised roadmap:

| Phase | Content |
|---|---|
| 0–2 | Existing setup, dataset inspection, and preprocessing — complete; preserve |
| 3 | Training-scale data loading + CNN baseline |
| 4 | Raw-waveform vs. optional time-frequency feature experiment |
| 5 | Per-event graph construction and graph/batching tests |
| 6 | GATv2 + attention pooling + Softmax paper-inspired baseline |
| 7 | Explicit temporal module and spatio-temporal model |
| 8 | EDL evidence head, Dirichlet calculations and evidential loss |
| 9 | Training, validation, checkpoints and reproducible experiment logging |
| 10 | OOD/uncertainty evaluation and Softmax comparison |
| 11 | Ablations, station/graph robustness, final report and documentation |

## 11. Instructions to the coding agent

1. Read this document, `README.md`, and `PROJECT_CONTEXT.md` before changing code.
2. Inspect the actual repository and current interfaces; do not assume this specification describes code that has not been inspected.
3. Provide a short implementation plan based on the existing code and identify any conflicts with this document before making broad changes.
4. Implement the next incomplete phase, not the entire roadmap in one untested rewrite.
5. Add or update tests with each component. Run the relevant tests and report the actual command and result.
6. Keep configurations, paths, label mappings, graph parameters and random seeds explicit.
7. Do not silently change the Phase 2 data contract or event split.
8. Do not claim an architecture, training run, metric, or OOD result is complete unless it exists in code and has been run.
9. If `PROJECT_CONTEXT.md` or the repository contains a more specific, already-recorded decision, identify the conflict and explain it rather than silently overriding it.
10. At the end of each implementation step, report: files changed, behavior added, tests run, test outcomes, remaining limitations, and the next recommended step.

**Immediate next task:** inspect the repository and `PROJECT_CONTEXT.md`, verify the current model/graph/uncertainty interfaces and data availability, then implement the smallest end-to-end **CNN baseline** and a smoke-training path. Do not jump straight to the complete EDL spatio-temporal model before the data loader and baseline are executable.
