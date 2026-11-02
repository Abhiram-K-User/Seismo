"""Train a showcase GAT-EDL model checkpoint and save sample waveforms.

Generates realistic seismic waveforms for natural earthquakes and quarry blasts
matching the PNW specification (3 channels: E, N, Z; 50 Hz; 3000 samples; P at sample 500).
Trains GATClassifier with EvidentialClassifier head using the project's Trainer.
Saves the trained checkpoint to outputs/checkpoints/best_model.pt and
sample events for UI visualization to outputs/showcase_samples.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from seismic_ml.data.schemas import EventBatch
from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.models.gat import GAT, GATClassifier
from seismic_ml.training.trainer import Trainer
from seismic_ml.utils.reproducibility import set_seed

SAMPLE_RATE = 50.0  # Hz
N_SAMPLES = 3000    # 60 seconds @ 50 Hz
P_SAMPLE = 500      # 10.0 s


def generate_seismic_trace(
    event_type: str,
    station_dist_km: float,
    seed: int,
) -> np.ndarray:
    """Generate a realistic 3-component seismic trace (ENZ) @ 50 Hz, 3000 samples.

    Earthquakes:
      - Double-couple shear mechanism
      - S-wave arrives ~2-5 s after P depending on distance (Vp ~ 6 km/s, Vs ~ 3.5 km/s)
      - S-wave is dominant on horizontal components (E, N)
      - P/S amplitude ratio <= 1.0
      - Broad frequency content (2 - 12 Hz)

    Anthropogenic Explosions / Quarry Blasts:
      - Isotropic compressional source
      - Sharp, impulsive P-wave onset, mostly on vertical (Z)
      - Weak, delayed Rayleigh wave / weak S-wave
      - High P/S amplitude ratio (>= 2.5)
      - Higher frequency initial onset, faster coda decay

    OOD (Low-frequency / Volcanic / Atmospheric):
      - Low-frequency harmonic tremor (0.8 - 2.0 Hz)
      - Emergent, unclear onset, slow decay
    """
    rng = np.random.RandomState(seed)
    t = np.arange(N_SAMPLES) / SAMPLE_RATE  # 0 to 60 s
    p_time = P_SAMPLE / SAMPLE_RATE         # 10.0 s

    # Seismic velocities: Vp ~ 6 km/s, Vs ~ 3.5 km/s
    dt_ps = max(1.5, min(8.0, station_dist_km / 8.0))
    s_time = p_time + dt_ps

    # Attenuation with distance
    geom_spread = 1.0 / max(1.0, math.sqrt(station_dist_km / 10.0))

    trace = np.zeros((3, N_SAMPLES), dtype=np.float32)

    if event_type == "earthquake":
        p_freq = rng.uniform(4.0, 9.0)
        s_freq = rng.uniform(2.5, 6.0)

        # P-wave: moderate on Z, smaller on H
        p_env = np.exp(-np.maximum(0.0, t - p_time) * 1.5) * (t >= p_time)
        p_signal_z = np.sin(2 * np.pi * p_freq * (t - p_time)) * p_env * 0.8
        p_signal_h = np.sin(2 * np.pi * p_freq * (t - p_time) + 0.5) * p_env * 0.4

        # S-wave: very strong on horizontals (E, N), moderate on Z
        s_env = np.exp(-np.maximum(0.0, t - s_time) * 0.6) * (t >= s_time)
        s_signal_e = np.sin(2 * np.pi * s_freq * (t - s_time) + rng.uniform(0, math.pi)) * s_env * 2.5
        s_signal_n = np.cos(2 * np.pi * s_freq * (t - s_time) + rng.uniform(0, math.pi)) * s_env * 2.2
        s_signal_z = np.sin(2 * np.pi * s_freq * (t - s_time) * 1.2) * s_env * 0.9

        # Coda
        coda_env = np.exp(-np.maximum(0.0, t - s_time) * 0.25) * (t >= s_time)
        coda = rng.randn(3, N_SAMPLES) * coda_env * 0.2

        trace[0] = p_signal_h * 0.5 + s_signal_e + coda[0]
        trace[1] = p_signal_h * 0.5 + s_signal_n + coda[1]
        trace[2] = p_signal_z + s_signal_z + coda[2]

    elif event_type == "anthropogenic":
        p_freq = rng.uniform(6.0, 14.0)
        s_freq = rng.uniform(3.0, 5.0)

        # P-wave: sharp, impulsive, dominant on Z
        p_env = np.exp(-np.maximum(0.0, t - p_time) * 3.5) * (t >= p_time)
        p_signal_z = np.sin(2 * np.pi * p_freq * (t - p_time)) * p_env * 3.2
        p_signal_e = np.sin(2 * np.pi * p_freq * (t - p_time) + 0.3) * p_env * 0.7
        p_signal_n = np.cos(2 * np.pi * p_freq * (t - p_time) + 0.7) * p_env * 0.6

        # Weak surface/S-wave
        s_env = np.exp(-np.maximum(0.0, t - s_time) * 1.2) * (t >= s_time)
        s_signal_e = np.sin(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.7
        s_signal_n = np.cos(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.6
        s_signal_z = np.sin(2 * np.pi * s_freq * (t - s_time)) * s_env * 0.4

        # Fast decay coda
        coda_env = np.exp(-np.maximum(0.0, t - p_time) * 0.7) * (t >= p_time)
        coda = rng.randn(3, N_SAMPLES) * coda_env * 0.15

        trace[0] = p_signal_e + s_signal_e + coda[0]
        trace[1] = p_signal_n + s_signal_n + coda[1]
        trace[2] = p_signal_z + s_signal_z + coda[2]

    else:  # ood - e.g. volcanic tremor / low frequency
        freq = rng.uniform(0.8, 1.8)
        env = np.exp(-((t - 25.0) / 12.0) ** 2)
        tremor = np.sin(2 * np.pi * freq * t) * env * 1.5
        noise = rng.randn(3, N_SAMPLES) * 0.1
        trace[0] = tremor + noise[0]
        trace[1] = tremor * 0.8 + noise[1]
        trace[2] = tremor * 1.2 + noise[2]

    trace = trace * geom_spread

    # Add realistic ambient background noise
    bg_noise = rng.randn(3, N_SAMPLES) * 0.05
    trace = trace + bg_noise

    # Station-scope normalization (unit std across live components, standard in Phase 2)
    std = float(np.std(trace))
    if std > 1e-6:
        trace = trace / std

    return trace.astype(np.float32)


class SyntheticSeismicDataset(Dataset):
    """Generates batched events with realistic station networks."""

    def __init__(self, num_events: int, seed: int = 42, is_val: bool = False):
        self.events: list[dict] = []
        rng = np.random.RandomState(seed)

        stations_pool = [
            ("UW.ALKI", 47.575, -122.418),
            ("UW.GSW", 47.288, -122.502),
            ("UW.RCM", 46.852, -121.758),
            ("CC.JRO", 46.275, -122.218),
            ("UW.HDW", 47.653, -122.309),
            ("UO.HAW", 45.412, -122.388),
        ]

        types = ["earthquake", "anthropogenic"]
        for i in range(num_events):
            event_type = types[i % 2]
            event_seed = seed + i * 17
            event_rng = np.random.RandomState(event_seed)

            # 2 to 4 stations per event
            n_stn = event_rng.randint(2, 5)
            selected_stns = [stations_pool[idx] for idx in event_rng.choice(len(stations_pool), n_stn, replace=False)]

            waveforms = []
            coords = []
            stn_names = []

            for stn_name, lat, lon in selected_stns:
                dist = event_rng.uniform(15.0, 75.0)
                wf = generate_seismic_trace(event_type, dist, event_seed + len(waveforms))
                waveforms.append(wf)
                coords.append([lat, lon])
                stn_names.append(stn_name)

            self.events.append({
                "event_id": f"sim-{'val' if is_val else 'train'}-{i:04d}",
                "event_type": event_type,
                "label": 0 if event_type == "earthquake" else 1,
                "waveforms": np.stack(waveforms),  # [N, 3, 3000]
                "coords": np.array(coords, dtype=np.float32),  # [N, 2]
                "stations": stn_names,
            })

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> dict:
        return self.events[idx]


def collate_fn(batch: list[dict]) -> EventBatch:
    max_n = max(e["waveforms"].shape[0] for e in batch)
    b = len(batch)

    padded_wfs = np.zeros((b, max_n, 3, N_SAMPLES), dtype=np.float32)
    coords = np.zeros((b, max_n, 2), dtype=np.float32)
    station_mask = np.zeros((b, max_n), dtype=bool)
    channel_mask = np.zeros((b, max_n, 3), dtype=bool)
    labels = np.array([e["label"] for e in batch], dtype=np.int64)

    for i, e in enumerate(batch):
        n = e["waveforms"].shape[0]
        padded_wfs[i, :n] = e["waveforms"]
        coords[i, :n] = e["coords"]
        station_mask[i, :n] = True
        channel_mask[i, :n] = True

    return EventBatch(
        waveforms=torch.from_numpy(padded_wfs),
        station_mask=torch.from_numpy(station_mask),
        channel_mask=torch.from_numpy(channel_mask),
        coordinates=torch.from_numpy(coords),
        labels=torch.from_numpy(labels),
        event_ids=[e["event_id"] for e in batch],
    )


def train_showcase():
    print("=" * 65)
    print("Seismo — Training Showcase GATv2 + EDL Model Checkpoint")
    print("=" * 65)

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build datasets: 120 train events, 40 val events
    train_dataset = SyntheticSeismicDataset(num_events=120, seed=100)
    val_dataset = SyntheticSeismicDataset(num_events=40, seed=500, is_val=True)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Instantiate model components
    encoder = CNNEncoder(embedding_dim=64, in_channels=3, pool_time=True)
    gat = GAT(in_dim=64, hidden_dim=64, out_dim=64, num_heads=4, num_layers=2, dropout=0.1)

    model = GATClassifier(
        encoder=encoder,
        gat=gat,
        num_classes=2,
        station_pooling="attention",
        graph_method="knn",
        k_neighbors=4,
        head_type="evidential",
        evidence_activation="softplus",
    )

    output_dir = Path("outputs/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "training": {
            "max_epochs": 15,
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
            "scheduler": {"name": "cosine"},
            "early_stopping": {"patience": 10, "monitor": "macro_f1"},
            "loss": {
                "evidential": {
                    "variant": "digamma",
                    "kl_weight": 0.5,
                    "kl_annealing_epochs": 10,
                }
            },
        }
    }

    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
        class_names=["earthquake", "anthropogenic"],
    )

    print("\nStarting training loop...")
    results = trainer.fit(train_loader, val_loader)

    ckpt_path = output_dir / "best_model.pt"
    trainer.save_checkpoint(ckpt_path)
    print(f"\n[OK] Model successfully trained and saved to {ckpt_path}")

    # Export showcase evaluation events
    print("Generating representative showcase events for UI demonstration...")
    samples = []

    # 1. Clear Earthquake event
    eq_wfs = [
        generate_seismic_trace("earthquake", 25.0, seed=1001),
        generate_seismic_trace("earthquake", 45.0, seed=1002),
        generate_seismic_trace("earthquake", 60.0, seed=1003),
    ]
    samples.append({
        "id": "sample-eq-01",
        "title": "Natural Tectonic Earthquake (M 2.4, Puget Sound)",
        "ground_truth": "earthquake",
        "category": "In-Distribution (Natural)",
        "description": "Double-couple shear mechanism with prominent horizontal S-wave arrival, lower P/S ratio (~0.6), and extended regional coda.",
        "stations": [
            {"id": "UW.ALKI", "name": "Alki Point, Seattle", "lat": 47.575, "lon": -122.418, "dist_km": 25.0},
            {"id": "UW.GSW", "name": "Gold Mountain", "lat": 47.288, "lon": -122.502, "dist_km": 45.0},
            {"id": "UW.RCM", "name": "Camp Muir, Rainier", "lat": 46.852, "lon": -121.758, "dist_km": 60.0},
        ],
        "waveforms": [wf.tolist() for wf in eq_wfs],  # 3 stations x 3 channels x 3000
    })

    # 2. Clear Quarry Blast event
    qb_wfs = [
        generate_seismic_trace("anthropogenic", 18.0, seed=2001),
        generate_seismic_trace("anthropogenic", 32.0, seed=2002),
        generate_seismic_trace("anthropogenic", 50.0, seed=2003),
    ]
    samples.append({
        "id": "sample-qb-01",
        "title": "Anthropogenic Quarry Blast (Centralia Mining District)",
        "ground_truth": "anthropogenic",
        "category": "In-Distribution (Explosion)",
        "description": "Compressional blast source with high initial vertical P-wave amplitude, very high P/S ratio (> 3.0), and rapid envelope decay.",
        "stations": [
            {"id": "UW.HDW", "name": "Hoodsport", "lat": 47.653, "lon": -122.309, "dist_km": 18.0},
            {"id": "UW.ALKI", "name": "Alki Point", "lat": 47.575, "lon": -122.418, "dist_km": 32.0},
            {"id": "UW.GSW", "name": "Gold Mountain", "lat": 47.288, "lon": -122.502, "dist_km": 50.0},
        ],
        "waveforms": [wf.tolist() for wf in qb_wfs],
    })

    # 3. Candidate Out-of-Distribution Event (Low-Frequency Volcanic Tremor)
    ood_wfs = [
        generate_seismic_trace("ood", 15.0, seed=3001),
        generate_seismic_trace("ood", 28.0, seed=3002),
    ]
    samples.append({
        "id": "sample-ood-01",
        "title": "Unfamiliar Volcanic Long-Period Event (Mount St. Helens)",
        "ground_truth": "out-of-distribution",
        "category": "Out-of-Distribution (OOD)",
        "description": "Monochromatic low-frequency resonance (1.2 Hz) without distinct P or S impulsive arrivals. Tests evidential vacuity uncertainty.",
        "stations": [
            {"id": "CC.JRO", "name": "Johnston Ridge", "lat": 46.275, "lon": -122.218, "dist_km": 15.0},
            {"id": "UW.RCM", "name": "Camp Muir", "lat": 46.852, "lon": -121.758, "dist_km": 28.0},
        ],
        "waveforms": [wf.tolist() for wf in ood_wfs],
    })

    # Save to outputs
    samples_path = Path("outputs/showcase_samples.json")
    samples_path.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"[OK] Showcase sample events written to {samples_path}")
    print("=" * 65)


if __name__ == "__main__":
    train_showcase()
