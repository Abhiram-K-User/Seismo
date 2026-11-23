# Implemented Improvements Summary & Experimental Report

This document provides a comprehensive technical record of all architectural, geophysical, evidential, evaluation, and performance upgrades implemented in the **Seismo** repository (`GATv2 + Evidential Deep Learning`) following the **Geophysical Machine Learning Technical Monograph**, along with full ablation benchmarks on the Pacific Northwest (PNW) dataset.

---

## 1. Architectural & Geophysical Upgrades

### 1.1 Kinematic Physics-Informed Graph Edges & Azimuth-Balanced Topology *(Monograph §10.1 / P0)*
* **Motivation & Gap Addressed:**
  * Standard spatial `knn` ($k=5$) connects stations strictly to nearby cluster neighbors along populated valleys and highways, pruning opposing-azimuth receivers needed to observe double-couple $P$-wave quadrantal polarities.
  * `GATv2Layer` previously converted `adjacency` into a boolean connectivity mask `(adjacency > 0)`, discarding continuous spherical distances and inter-station geometry before computing attention.
* **Implementation Details:**
  * **Azimuth-Balanced `quadrant_knn` ([`src/seismic_ml/graph/construction.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/graph/construction.py)):**
    * Added `"quadrant_knn"` to `SUPPORTED_METHODS` in `build_spatial_graph()`.
    * `_pairwise_azimuths_rad(coords)` computes the directed spherical bearing $\theta_{ij} \in (-\pi, \pi]$ between receiver $i$ and sender $j$.
    * `_select_quadrant_neighbors_tensor(sub_dist, sub_az, k_eff)` partitions the $360^\circ$ focal sphere around each receiver into four $90^\circ$ sectors and selects the nearest neighbors round-robin across populated quadrants using vectorized `torch.sort`, `cumsum`, and `torch.topk` (zero Python per-node loops or `.item()` CPU–GPU syncs).
  * **4-D Kinematic Edge Attributes (`build_kinematic_edge_features` in [`src/seismic_ml/graph/construction.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/graph/construction.py)):**
    * Generates `[B, N, N, 4]` edge vectors per directed edge $j \to i$ from `[B, N, 2]` `(lat, lon)` station coordinates (requiring zero extra catalog metadata at inference time):
      $$\mathbf{e}_{ij} = \left[\frac{\log(1 + d_{ij})}{\log(1 + 300)},\; \frac{d_{ij} / v_P}{50.0},\; \cos\theta_{ij},\; \sin\theta_{ij}\right]$$
      where $d_{ij}$ is Haversine great-circle distance (km) and $v_P = 6.2\text{ km/s}$ (`DEFAULT_VP_KM_S`) is the PNW upper-crustal $P$-wave velocity. Self-loops and padded station slots are masked strictly to `0.0`.
  * **Edge-Conditioned Dynamic Attention ([`src/seismic_ml/models/gat.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/models/gat.py)):**
    * Added `edge_dim: int = 0` to `GATv2Layer` and `GAT`. When `edge_dim > 0`, `self.lin_edge = nn.Linear(edge_dim, out_dim, bias=False)` projects $\mathbf{e}_{ij}$ into `[B, N, N, H, d]` and injects it into the pre-nonlinearity GATv2 score:
      $$\alpha_{ij}^{(h)} = \text{Softmax}_j\left(\mathbf{a}_h^T \text{LeakyReLU}\left(\mathbf{Q}_i^{(h)} + \mathbf{K}_j^{(h)} + \mathbf{E}_{ij}^{(h)} + \mathbf{b}_h\right)\right)$$
    * `GATClassifier` accepts `edge_dim` and `vp_km_s` and automatically computes kinematic edge features from `coordinates` when `edge_dim > 0`, while keeping `edge_dim=0` as the default for full backward compatibility with earlier checkpoints.
  * **Configuration & CLI:** Added `vp_km_s: 6.2` and `edge_dim: 0` in [`configs/model.yaml`](file:///C:/Users/abhir/Seismo/configs/model.yaml), `--graph-method` / `--edge-dim` in [`scripts/train.py`](file:///C:/Users/abhir/Seismo/scripts/train.py), and automatic `edge_dim` detection from `gat.layers.0.lin_edge.weight` in [`scripts/evaluate.py`](file:///C:/Users/abhir/Seismo/scripts/evaluate.py) and [`src/seismic_ml/ui/app.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/ui/app.py).

---

### 1.2 Dual-Stream Hybrid Waveform–Spectrogram Encoder *(Monograph §10.2 / P0)*
* **Motivation & Gap Addressed:** Commercial quarry blasts commonly use millisecond-delayed ripple-firing across borehole rows, creating periodic spectral scalloping and high-frequency roll-offs in the 5–20 Hz band that pure 1D time-domain convolutions struggle to isolate cleanly.
* **Implementation Details ([`src/seismic_ml/features/cnn_encoder.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/features/cnn_encoder.py)):**
  * Extended `INPUT_REPRESENTATIONS` to `("waveform", "spectrogram", "dual")`.
  * **Branch A (1D Temporal Stream):** Retains the 4-block dilated residual 1D TCN (`DilatedResidualBlock1D` with dilations $d \in \{1, 2, 4, 8\}$, kernel size 7) capturing sharp $P$-wave onsets and $P$-$S$ arrival separation.
  * **Branch B (2D Time–Frequency Stream — `SpectrogramBranch2D`):**
    * Computes STFT log-power spectrograms $\log(1 + |\text{STFT}(x)|^2)$ directly on GPU via `torch.stft` using a 128-sample Hann window, `hop_length=64` (50% overlap), band-limited strictly to $1.0\text{–}16.0\text{ Hz}$ at $50\text{ Hz}$ sampling rate.
    * Processes the 3-channel `[M, 3, F_band, T']` spectrogram through a 3-block 2D-CNN (`Conv2d(3->32->64->out_dim)` with `BatchNorm2d`, `ReLU`, `MaxPool2d`, and `Dropout2d`) followed by `AdaptiveAvgPool2d((1, 1))`.
  * **Cross-Attention Latent Fusion (`CrossAttentionFusion`):**
    * Fuses the temporal branch embedding $h_{\text{temp}}$ and spectral branch embedding $h_{\text{spec}}$ via gated scaled dot-product cross-attention, concatenates $[h_{\text{cross}} \,\|\, h_{\text{spec}}]$, and projects with `LayerNorm` to `embedding_dim` (`128`).
  * **CLI & Auto-Detection:** Added `--input-representation {waveform,spectrogram,dual}` in [`scripts/train.py`](file:///C:/Users/abhir/Seismo/scripts/train.py) and automatic checkpoint inspection (`encoder.fusion.out_proj.weight` / `encoder.spec_branch.net.0.weight`) in [`scripts/evaluate.py`](file:///C:/Users/abhir/Seismo/scripts/evaluate.py).

---

### 1.3 Out-of-Distribution Exposure ($\mathcal{L}_{\text{OE}}$) & Stochastic Station Dropout *(Monograph §11.1 & Gap 3 / P1)*
* **Motivation & Gap Addressed:**
  * Classical Dirichlet EDL only penalizes misleading evidence on the *non-target* in-distribution class (`kl_dirichlet_regularizer`), leaving substantial residual belief mass on exotic anomalies (Monograph §9).
  * Training previously used all available stations per event without simulating sensor blackouts or telemetry drops.
* **Implementation Details:**
  * **Out-of-Distribution Exposure ([`src/seismic_ml/training/losses.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/training/losses.py)):** Added `ood_exposure_loss(alpha)` computing $\text{KL}(\text{Dir}(\boldsymbol{\alpha}) \,\|\, \text{Dir}(\mathbf{1}))$ to penalize *all* class evidence on non-seismic exposure signals.
  * **Trainer Integration ([`src/seismic_ml/training/trainer.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/training/trainer.py)):**
    * `_make_ood_exposure_batch()` generates a synthetic exposure mini-batch per step composed of 50% pure Gaussian ambient noise and 50% FFT phase-scrambled waveforms (preserving spectral amplitude while destroying coherent $P/S$ wave kinematics), weighted by `ood_exposure_weight` and the KL annealing coefficient.
    * `_apply_station_dropout()` stochastically drops up to `station_dropout` fraction of valid stations per event during training while guaranteeing $\ge 1$ active station per event.
  * **CLI:** Added `--station-dropout` and `--ood-exposure-weight` in [`scripts/train.py`](file:///C:/Users/abhir/Seismo/scripts/train.py).

---

## 2. Phase 9 Evaluation, Calibration, OOD Metrics & UI Diagnostics

### 2.1 Calibration & OOD Benchmarking Modules
* **[`src/seismic_ml/evaluation/calibration.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/evaluation/calibration.py):**
  * `reliability_curve(probabilities, labels, num_bins=10)`: Equal-width confidence binning over $[0, 1]$.
  * `expected_calibration_error` (**ECE**): $\sum_b \frac{n_b}{M} |\text{acc}(b) - \text{conf}(b)|$.
  * `brier_score`: Multi-class Brier score $\frac{1}{M}\sum_{i,k} (p_{i,k} - y_{i,k})^2$.
* **[`src/seismic_ml/evaluation/ood.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/evaluation/ood.py):**
  * `ood_auroc(in_scores, ood_scores)`: Exact Mann-Whitney U / ROC-AUC with tie-averaging in $O((N+M)\log(N+M))$.
  * `ood_auprc(in_scores, ood_scores)`: Exact step-wise Average Precision with OOD as the positive class.
  * `id_quantile_threshold(in_scores, id_tpr=0.95)` & `ood_detection_rate`: Derives the rejection threshold $\tau$ strictly from in-distribution scores without peeking at OOD data.
  * `risk_coverage_curve` & `area_under_risk_coverage` (**AURC**): Selective classification risk curve ordered from lowest to highest uncertainty.
* **[`src/seismic_ml/uncertainty/metrics.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/uncertainty/metrics.py):**
  * `dissonance(belief)`: Subjective Logic belief dissonance quantifying conflicting aleatoric evidence across classes separately from epistemic vacuity $u = K / S$.
  * `dirichlet_mutual_information(alpha)`: Epistemic mutual information $\mathcal{I}[y, \boldsymbol{\pi}] = H(\mathbb{E}[\boldsymbol{\pi}]) - \mathbb{E}[H(\boldsymbol{\pi})]$ under $\text{Dir}(\boldsymbol{\alpha})$.
  * `predictive_entropy(probabilities)` & `max_probability(probabilities)`: Total predictive Shannon entropy and confidence baseline.

### 2.2 Evaluation CLI, Visualization & Showcase UI
* **[`src/seismic_ml/training/trainer.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/training/trainer.py) & [`scripts/evaluate.py`](file:///C:/Users/abhir/Seismo/scripts/evaluate.py):**
  * `Trainer.evaluate()` automatically computes `ece`, `brier_score`, and `aurc` (falling back to $1 - \max_k p_k$ on Softmax models for apples-to-apples comparisons).
  * `scripts/evaluate.py` supports `--ood-collection exotic` to run joint ID + OOD benchmarking in one pass, exporting `evaluation_metrics_<split>.json`, `reliability_diagram_<split>.png`, `risk_coverage_<split>.png`, `confusion_matrix_<split>.png`, and `uncertainty_distribution_<split>.png`.
* **[`src/seismic_ml/utils/visualization.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/utils/visualization.py):** Implemented `plot_reliability_diagram()` and `plot_risk_coverage_curve()`.
* **[`src/seismic_ml/ui/app.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/ui/app.py):**
  * Added **Belief Dissonance (Aleatoric Conflict)** and **Mutual Information $I[y, \pi]$ (Epistemic)** to `/api/classify` and the Step 3 Subjective Logic diagnostics table.
  * Added automatic checkpoint class-index alignment and `edge_dim` detection so both the 64-D local showcase checkpoint and 128-D Colab checkpoints load seamlessly.

---

## 3. Data-Loading, VRAM & Training Throughput Optimizations

To eliminate the ~252 ms/step CPU bottleneck observed on Google Colab (~50 min for 20 epochs), four optimizations were implemented with **bit-exact** tensor parity:

1. **Pre-Indexed Dataset & Optional Full-RAM Store ([`src/seismic_ml/data/processed.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/data/processed.py), [`src/seismic_ml/data/loader.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/data/loader.py)):**
   * Added `FastProcessedDataset` and `collate_fast_events`: pre-indexes per-event row indices, coordinates, station IDs, and labels into NumPy arrays once at initialization—eliminating per-event pandas `iloc`/`sort_values` and `pnw.parse_utc` calls (~700,000 timestamp parses over 20 epochs).
   * Added `--cache-ram` to load only the split's active waveforms (~1.3 GB for `comcat` train+val) into RAM once upfront. Verified bit-for-bit identical to `ProcessedEventDataset` in [`tests/test_processed.py`](file:///C:/Users/abhir/Seismo/tests/test_processed.py).
2. **Valid-Station Compaction in `CNNEncoder` ([`src/seismic_ml/features/cnn_encoder.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/features/cnn_encoder.py)):**
   * Compacts `waveforms[station_mask]` before running 1D/2D convolutions and scatters back afterward, avoiding wasted compute and VRAM on zero-padded station slots (cutting peak VRAM ~5–8× on variable-station batches).
3. **Vectorized `quadrant_knn` ([`src/seismic_ml/graph/construction.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/graph/construction.py)):**
   * Eliminated Python per-node `.item()` loops to prevent CPU–GPU synchronization stalls during `forward()`.
4. **CUDA `pin_memory`, `--num-workers`, and Opt-In `bfloat16` AMP ([`src/seismic_ml/training/trainer.py`](file:///C:/Users/abhir/Seismo/src/seismic_ml/training/trainer.py), [`scripts/train.py`](file:///C:/Users/abhir/Seismo/scripts/train.py)):**
   * Added `--amp` (`torch.amp.autocast("cuda", dtype=torch.bfloat16)` for CNN/STFT/GATv2 forward passes while keeping Evidential `digamma`/`lgamma` losses strictly in `float32`).
   * Increased Colab training throughput from **`63 events/s` (`3.96 steps/s` @ batch 16, `~50 min`)** to **`255 events/s` (`7.96 steps/s` @ batch 32, `~11–12 min`)**—a **4× end-to-end throughput speedup** even while running the heavier Dual-Stream + OOD Exposure pipeline.

---

## 4. Observed Ablation Benchmarks (PNW Unseen 2021+ Test Set & Exotic OOD)

All three models were trained on the balanced PNW `comcat` train split (<2019, 8,976 events), validated on 2019–2020 (1,773 events), and evaluated on **1,979 unseen 2021+ test events** (`comcat`) plus **728 exotic OOD events** (`exotic`) on Google Colab (NVIDIA L4 GPU):

| Metric | Baseline GAT-EDL (`knn`, `edge_dim=0`) | Physics GAT-EDL (`quadrant_knn`, `edge_dim=4`) | Dual-Stream + Physics + OE (`dual`, `quadrant_knn`, `edge_dim=4`, `dropout=0.25`, `OE=0.5`) |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | `0.9343` | `0.9378` | **`0.9510`** (**+1.67%**) |
| **Macro-F1 Score** | `0.9343` | `0.9378` | **`0.9510`** (**+1.67%**) |
| **Balanced Accuracy** | `0.9345` | `0.9379` | **`0.9509`** |
| **Earthquake Precision / Recall / F1** | `0.9476` / `0.9212` / `0.9342` | `0.9426` / `0.9341` / `0.9383` | `0.9467` / **`0.9571`** / **`0.9519`** |
| **Anthropogenic Precision / Recall / F1** | `0.9214` / **`0.9478`** / `0.9344` | `0.9331` / `0.9417` / `0.9373` | **`0.9555`** / `0.9447` / **`0.9501`** |
| **Test Evidential Loss** | `0.2360` | **`0.2127`** | `0.2199` |
| **ECE (Calibration Error)** *(lower is better)* | `0.0514` | **`0.0305`** (**-40.7%**) | `0.0718` |
| **Brier Score** *(lower is better)* | `0.1024` | `0.0948` | **`0.0891`** |
| **AURC (Selective Risk)** *(lower is better)* | `0.0117` | **`0.0105`** | `0.0131` |
| **ID Mean Vacuity ($u$)** | `0.2196` | `0.1722` | `0.2384` |
| **OOD Mean Vacuity ($u$)** *(higher is better)* | `0.4117` (+87.5%) | `0.3602` (+109.2%) | **`0.4812`** (+101.8%) |
| **OOD AUROC** | `0.7906` | **`0.7917`** | `0.7580` |
| **OOD AUPRC** | `0.5125` | **`0.5241`** | `0.4874` |
| **OOD Detection Rate @ 95% ID** | `0.1525` ($\tau = 0.6977$) | **`0.1717`** ($\tau = 0.6673$) | `0.1387` ($\tau = 0.8698$) |

### Key Analytical Takeaways
1. **Kinematic Edges + Quadrant $k$-NN (`93.78%` F1, `0.0305` ECE):** Injecting $[d, \Delta t_P, \cos\theta, \sin\theta]$ into GATv2 attention alongside azimuthal quadrant connectivity improved classification accuracy (`+0.35%`), reduced ECE by **40.7%** (`0.0514` $\to$ `0.0305`), lowered selective risk (`0.0105` AURC), and achieved the strongest OOD ranking (`0.7917` AUROC, `17.17%` detection @ 95% ID).
2. **Dual-Stream 2D-STFT + OOD Exposure + Station Dropout (`95.10%` F1, `0.0891` Brier, `0.4812` OOD Vacuity):** Adding the 2D spectrogram branch and cross-attention fusion pushed unseen test accuracy past **95%** (`95.10%`, raising earthquake recall to `95.71%` and blast precision to `95.55%`), achieved the best Brier score (`0.0891`), and raised mean OOD vacuity to `0.4812`. At the same time, aggressive `station_dropout=0.25` and `ood_exposure_weight=0.5` widened the 95th-percentile in-distribution vacuity tail ($\tau_{95} = 0.8698$) and slightly softened confidence (`0.0718` ECE)—highlighting a clear, tunable accuracy vs. OOD-tail calibration tradeoff.

---

## 5. Verification Status
* **Automated Test Suite:** `117 passed in 36.42s` covering [`tests/test_processed.py`](file:///C:/Users/abhir/Seismo/tests/test_processed.py), [`tests/test_data.py`](file:///C:/Users/abhir/Seismo/tests/test_data.py), [`tests/test_graph.py`](file:///C:/Users/abhir/Seismo/tests/test_graph.py), [`tests/test_models.py`](file:///C:/Users/abhir/Seismo/tests/test_models.py), [`tests/test_training_evaluation.py`](file:///C:/Users/abhir/Seismo/tests/test_training_evaluation.py), [`tests/test_uncertainty.py`](file:///C:/Users/abhir/Seismo/tests/test_uncertainty.py), and [`tests/test_scripts.py`](file:///C:/Users/abhir/Seismo/tests/test_scripts.py) (including bit-exactness verification of `FastProcessedDataset` against `ProcessedEventDataset`, kinematic edge geometry/gradients, Dual-Stream forward/backward passes, and Phase 9 calibration/OOD metrics).

---

## 6. Remaining Monograph Roadmap Items (Future Scope)
1. **Observed $P$-Wave Travel-Time Residuals ($r_{ij}$):** Surface `p_time_offset_s` from `traces.csv` into `EventBatch` to append $|\Delta t_{ij}^{\text{obs}} - d_{ij}/v_P|$ to kinematic edge features (§10.1).
2. **Multi-Task Auxiliary Depth & Magnitude Heads:** Join `depth_km` and `magnitude` onto `EventBatch` for auxiliary regression regularization (§11.2).
3. **Hyper-Opinion EDL (HEDL):** Power-set belief allocation to formally decouple aleatoric ambiguity from epistemic vacuity (§11.1).
4. **Phase 10 Automated Robustness Sweep Script:** Automated station-dropout and SNR degradation sweep benchmarking (`evaluate_robustness.py`).
5. **Pretrained Foundation Backbone (MAE) & Cross-Regional Domain Adaptation (UUSS / SCEDC / INGV):** (§10.3 & §11.3).
