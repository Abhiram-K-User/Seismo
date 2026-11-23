"""FastAPI backend and UI server for showcasing the trained Seismic ML Model.

Implements a streamlined, multi-step sequential workflow:
  Step 1: Scenario Selection (22 Curated Events across Earthquakes, Quarry Blasts, OOD + CSV/JSON Upload)
  Step 2: Signal Inspection & Robustness Controls (Interactive Waveform Canvas, Station Array, Telemetry Noise)
  Step 3: Evidential Inference & Diagnostics (Classification, Vacuity Uncertainty, GATv2 Attention, JSON Export)
"""

from __future__ import annotations

import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from seismic_ml.features.cnn_encoder import CNNEncoder
from seismic_ml.models.gat import GAT, GATClassifier
from seismic_ml.training.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "best_model.pt"
SAMPLES_FILE = REPO_ROOT / "outputs" / "showcase_samples.json"

app = FastAPI(title="Seismo — Model Showcase UI", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
state: dict[str, Any] = {
    "model": None,
    "device": "cpu",
    "presets": [],
    "custom_events": {},
    "checkpoint_path": str(DEFAULT_CKPT),
    "param_count": 0,
}


def load_model_and_samples():
    """Load model weights and preset showcase events."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device

    embedding_dim = 64
    k_neighbors = 4
    if DEFAULT_CKPT.exists():
        try:
            ckpt = torch.load(DEFAULT_CKPT, map_location="cpu")
            s = ckpt.get("model_state_dict", ckpt)
            if "classifier.linear.weight" in s:
                embedding_dim = s["classifier.linear.weight"].shape[1]
            elif "encoder.proj.weight" in s:
                embedding_dim = s["encoder.proj.weight"].shape[0]
            print(f"[UI] Detected embedding_dim={embedding_dim} from checkpoint.")
        except Exception as e:
            print(f"[UI] Warning inspecting checkpoint: {e}")

    encoder = CNNEncoder(embedding_dim=embedding_dim, in_channels=3, pool_time=True)
    gat = GAT(in_dim=embedding_dim, hidden_dim=embedding_dim, out_dim=embedding_dim, num_heads=4, num_layers=2, dropout=0.1)
    model = GATClassifier(
        encoder=encoder,
        gat=gat,
        num_classes=2,
        station_pooling="attention",
        graph_method="knn",
        k_neighbors=k_neighbors,
        head_type="evidential",
        evidence_activation="softplus",
    )

    if DEFAULT_CKPT.exists():
        trainer = Trainer(model=model, config={"seed": 42}, device=device)
        trainer.load_checkpoint(DEFAULT_CKPT)
        print(f"[UI] Loaded model checkpoint from {DEFAULT_CKPT} on {device}")
    else:
        print(f"[UI] Warning: Checkpoint {DEFAULT_CKPT} not found. Running with initialized weights.")

    model.to(device)
    model.eval()

    state["model"] = model
    state["param_count"] = sum(p.numel() for p in model.parameters())

    if SAMPLES_FILE.exists():
        state["presets"] = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))
        print(f"[UI] Loaded {len(state['presets'])} showcase sample events.")


# Eager load on module initialization
load_model_and_samples()


@app.get("/api/model_info")
def get_model_info():
    """Return model architecture and runtime details."""
    model = state["model"]
    if model is None:
        return {"status": "Model not loaded"}

    return {
        "model_name": "GATv2 + Evidential Deep Learning (EDL)",
        "encoder": "1D-CNN (4 Residual Dilated Blocks)",
        "gnn": "GATv2 (Dynamic Multi-Head Attention)",
        "head": "Evidential Dirichlet (Subjective Logic)",
        "parameters": state["param_count"],
        "device": state["device"],
        "checkpoint": state["checkpoint_path"],
        "status": "Ready",
    }


@app.get("/api/samples")
def get_samples():
    """Return catalog of available preset events."""
    if SAMPLES_FILE.exists() and len(state["presets"]) < 20:
        state["presets"] = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))

    return [
        {
            "id": s["id"],
            "title": s["title"],
            "category": s["category"],
            "ground_truth": s["ground_truth"],
            "description": s["description"],
            "num_stations": len(s["stations"]),
            "station_names": [st["id"] for st in s["stations"]],
            "stations": s["stations"],
        }
        for s in state["presets"]
    ]


def _format_event_payload(s: dict[str, Any]) -> dict[str, Any]:
    """Helper to return event data with downsampled waveforms for UI canvas."""
    downsampled_wfs = []
    factor = 3  # 3000 / 3 = 1000 points per channel for responsive high-fidelity zooming
    for stn_wf in s["waveforms"]:
        stn_down = []
        for comp in stn_wf:
            comp_arr = np.array(comp, dtype=np.float32)
            reshaped = comp_arr[:3000].reshape(1000, factor).mean(axis=1)
            stn_down.append([round(float(v), 4) for v in reshaped])
        downsampled_wfs.append(stn_down)

    return {
        "id": s["id"],
        "title": s["title"],
        "category": s.get("category", "Custom"),
        "ground_truth": s.get("ground_truth", "unknown"),
        "description": s.get("description", ""),
        "stations": s["stations"],
        "waveforms": downsampled_wfs,
    }


@app.get("/api/sample/{sample_id}")
def get_sample_detail(sample_id: str):
    """Return full sample event data."""
    if sample_id in state["custom_events"]:
        return _format_event_payload(state["custom_events"][sample_id])

    for s in state["presets"]:
        if s["id"] == sample_id:
            return _format_event_payload(s)

    raise HTTPException(status_code=404, detail="Sample event not found")


class UploadPayload(BaseModel):
    file_name: str
    file_type: str  # "json" or "csv"
    content: str


@app.post("/api/upload_event")
def upload_event(payload: UploadPayload):
    """Parse, validate, and register a user-uploaded waveform file (JSON or CSV)."""
    file_name = payload.file_name
    file_type = payload.file_type.lower()
    raw_content = payload.content.strip()

    try:
        if file_type == "json":
            data = json.loads(raw_content)
            if "waveforms" not in data:
                raise ValueError("JSON must contain a 'waveforms' field.")

            raw_wfs = np.array(data["waveforms"], dtype=np.float32)
            if raw_wfs.ndim == 2:
                raw_wfs = np.expand_dims(raw_wfs, axis=0)
            elif raw_wfs.ndim != 3:
                raise ValueError(f"Waveforms must be [3, T] or [N, 3, T], got shape {raw_wfs.shape}")

            n_stations, n_channels, t_samples = raw_wfs.shape
            if n_channels != 3:
                raise ValueError(f"Expected 3 channels (E, N, Z), got {n_channels} channels.")

            if t_samples != 3000:
                old_x = np.linspace(0, 1, t_samples)
                new_x = np.linspace(0, 1, 3000)
                resampled = np.zeros((n_stations, 3, 3000), dtype=np.float32)
                for s in range(n_stations):
                    for c in range(3):
                        resampled[s, c] = np.interp(new_x, old_x, raw_wfs[s, c])
                raw_wfs = resampled

            stations = data.get("stations", [])
            if not stations or len(stations) != n_stations:
                stations = [
                    {
                        "id": f"STA.{i+1:02d}",
                        "name": f"Station {i+1}",
                        "lat": 47.0 + 0.1 * i,
                        "lon": -122.0 - 0.1 * i,
                        "dist_km": round(15.0 + 10.0 * i, 1),
                    }
                    for i in range(n_stations)
                ]

            event_id = f"upload-json-{int(time.time())}"
            event_obj = {
                "id": event_id,
                "title": data.get("title", f"Uploaded Event ({file_name})"),
                "category": "User Uploaded",
                "ground_truth": data.get("ground_truth", "unknown"),
                "description": data.get("description", f"Custom event parsed from {file_name}"),
                "stations": stations,
                "waveforms": raw_wfs.tolist(),
            }
            state["custom_events"][event_id] = event_obj
            return _format_event_payload(event_obj)

        elif file_type == "csv":
            reader = csv.reader(io.StringIO(raw_content))
            header = [h.strip().lower() for h in next(reader, [])]

            col_map = {col: idx for idx, col in enumerate(header)}
            e_col = col_map.get("e", col_map.get("e_channel", -1))
            n_col = col_map.get("n", col_map.get("n_channel", -1))
            z_col = col_map.get("z", col_map.get("z_channel", -1))

            if -1 in (e_col, n_col, z_col):
                if len(header) >= 4:
                    e_col, n_col, z_col = 1, 2, 3
                elif len(header) == 3:
                    e_col, n_col, z_col = 0, 1, 2
                else:
                    raise ValueError("CSV header must specify 'E', 'N', 'Z' components or provide at least 3 numeric columns.")

            parsed_vals = []
            for r in reader:
                if not r or len(r) <= max(e_col, n_col, z_col):
                    continue
                try:
                    vals = [float(r[e_col]), float(r[n_col]), float(r[z_col])]
                    parsed_vals.append(vals)
                except (ValueError, IndexError):
                    continue

            if len(parsed_vals) < 10:
                raise ValueError("CSV contains fewer than 10 valid numeric rows.")

            arr = np.array(parsed_vals, dtype=np.float32).T  # [3, T]
            t_samples = arr.shape[1]

            if t_samples != 3000:
                old_x = np.linspace(0, 1, t_samples)
                new_x = np.linspace(0, 1, 3000)
                resampled = np.zeros((3, 3000), dtype=np.float32)
                for c in range(3):
                    resampled[c] = np.interp(new_x, old_x, arr[c])
                arr = resampled

            std = float(np.std(arr))
            if std > 1e-6:
                arr = arr / std

            event_id = f"upload-csv-{int(time.time())}"
            event_obj = {
                "id": event_id,
                "title": f"Uploaded Waveform ({file_name})",
                "category": "User Uploaded (Single Station)",
                "ground_truth": "unknown",
                "description": f"Imported from {file_name}: 3 channels resampled to 3,000 samples @ 50 Hz.",
                "stations": [
                    {
                        "id": "UPLOAD.STA1",
                        "name": "Local Station",
                        "lat": 47.500,
                        "lon": -122.300,
                        "dist_km": 25.0,
                    }
                ],
                "waveforms": [arr.tolist()],
            }
            state["custom_events"][event_id] = event_obj
            return _format_event_payload(event_obj)

        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported: 'json', 'csv'.")

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/template/csv", response_class=PlainTextResponse)
def get_sample_csv():
    """Return a downloadable template CSV."""
    t = np.arange(3000) / 50.0
    p_time = 10.0
    s_time = 14.5
    p_env = np.exp(-np.maximum(0, t - p_time) * 1.5) * (t >= p_time)
    s_env = np.exp(-np.maximum(0, t - s_time) * 0.7) * (t >= s_time)

    e = np.sin(2 * np.pi * 5.0 * t) * s_env * 2.0 + np.random.randn(3000) * 0.05
    n = np.cos(2 * np.pi * 5.0 * t) * s_env * 1.8 + np.random.randn(3000) * 0.05
    z = np.sin(2 * np.pi * 8.0 * t) * p_env * 1.5 + np.random.randn(3000) * 0.05

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["time_sec", "E", "N", "Z"])
    for i in range(3000):
        writer.writerow([round(t[i], 3), round(float(e[i]), 5), round(float(n[i]), 5), round(float(z[i]), 5)])
    return output.getvalue()


@app.get("/api/template/json")
def get_sample_json():
    """Return a downloadable template JSON."""
    if state["presets"]:
        return state["presets"][0]
    return {"message": "No sample template available"}


class ClassifyRequest(BaseModel):
    sample_id: str
    noise_level: float = 0.0
    active_stations: list[int] | None = None


@app.post("/api/classify")
def classify_event(req: ClassifyRequest):
    """Run model inference on the selected event with optional noise/station dropout."""
    model = state["model"]
    device = state["device"]
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    target_sample = None
    if req.sample_id in state["custom_events"]:
        target_sample = state["custom_events"][req.sample_id]
    else:
        for s in state["presets"]:
            if s["id"] == req.sample_id:
                target_sample = s
                break

    if target_sample is None and state["presets"]:
        target_sample = state["presets"][0]

    if target_sample is None:
        raise HTTPException(status_code=404, detail="No event available for classification")

    raw_wfs = np.array(target_sample["waveforms"], dtype=np.float32)
    stations = target_sample["stations"]
    n_total = len(stations)

    active_indices = req.active_stations if req.active_stations is not None else list(range(n_total))
    if not active_indices:
        active_indices = [0]

    wfs_filtered = raw_wfs[active_indices].copy()
    stations_filtered = [stations[i] for i in active_indices]
    coords_filtered = np.array([[st["lat"], st["lon"]] for st in stations_filtered], dtype=np.float32)

    if req.noise_level > 0.0:
        noise = np.random.randn(*wfs_filtered.shape).astype(np.float32) * req.noise_level
        wfs_filtered = wfs_filtered + noise
        std = np.std(wfs_filtered, axis=(1, 2), keepdims=True)
        std[std < 1e-6] = 1.0
        wfs_filtered = wfs_filtered / std

    wfs_tensor = torch.from_numpy(wfs_filtered).unsqueeze(0).to(device)
    coords_tensor = torch.from_numpy(coords_filtered).unsqueeze(0).to(device)
    station_mask = torch.ones((1, len(stations_filtered)), dtype=torch.bool, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out, attn_weights = model(
            wfs_tensor,
            station_mask=station_mask,
            coordinates=coords_tensor,
            return_attention=True,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    probs = out.expected_probability[0].cpu().numpy()
    evidence = out.evidence[0].cpu().numpy()
    alpha = out.alpha[0].cpu().numpy()
    strength = float(out.strength[0].item())
    uncertainty = float(out.uncertainty[0].item())
    belief = out.belief[0].cpu().numpy()

    # In the trained model checkpoint, index 0 is anthropogenic (blast) and index 1 is earthquake
    p_anth = float(probs[0])
    p_eq = float(probs[1])
    predicted_class = "earthquake" if p_eq > p_anth else "anthropogenic"
    confidence = max(p_eq, p_anth)

    attn_matrix = []
    if attn_weights and len(attn_weights) > 0:
        last_layer_attn = attn_weights[-1][0].mean(dim=0).cpu().numpy()
        attn_matrix = [[round(float(val), 4) for val in row] for row in last_layer_attn]

    return {
        "event_id": target_sample["id"],
        "title": target_sample["title"],
        "ground_truth": target_sample.get("ground_truth", "unknown"),
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100.0, 2),
        "probabilities": {
            "earthquake": round(p_eq, 4),
            "anthropogenic": round(p_anth, 4),
        },
        "evidential": {
            "evidence": {"earthquake": round(float(evidence[1]), 4), "anthropogenic": round(float(evidence[0]), 4)},
            "alpha": [round(float(alpha[1]), 4), round(float(alpha[0]), 4)],
            "strength": round(strength, 4),
            "vacuity_uncertainty": round(uncertainty, 4),
            "belief": {"earthquake": round(float(belief[1]), 4), "anthropogenic": round(float(belief[0]), 4)},
        },
        "stations_used": [st["id"] for st in stations_filtered],
        "attention_matrix": attn_matrix,
        "latency_ms": round(latency_ms, 2),
        "device": device,
    }


# Tectonic Light-Mode UI with Themed Landing Page
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SEISMO — Evidential Multi-Station Seismic Intelligence</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f8f6f2;
      --bg-subtle: #f1ede6;
      --card: #ffffff;
      --card-inner: #faf8f5;
      --border: #e5e0d8;
      --border-strong: #d4cdc2;
      --text: #18181b;
      --text-secondary: #47474f;
      --text-muted: #6c6c75;
      --text-dim: #8f8f99;

      --accent: #d94e28;
      --accent-hover: #bd3e1c;
      --accent-soft: #fef2ee;
      --accent-border: #f8cbbd;

      --eq: #059669;
      --eq-bg: #ecfdf5;
      --eq-border: #a7f3d0;

      --anth: #d97706;
      --anth-bg: #fffbeb;
      --anth-border: #fde68a;

      --ood: #7c3aed;
      --ood-bg: #f5f3ff;
      --ood-border: #ddd6fe;

      --cyan: #0284c7;
      --blue: #2563eb;
      --red: #dc2626;

      --shadow-sm: 0 1px 2px rgba(24, 24, 27, 0.04);
      --shadow-md: 0 6px 24px rgba(24, 24, 27, 0.05), 0 1px 3px rgba(24, 24, 27, 0.03);
      --shadow-lg: 0 16px 40px rgba(24, 24, 27, 0.08), 0 2px 6px rgba(24, 24, 27, 0.04);

      --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { scroll-behavior: smooth; }

    body {
      background-color: var(--bg);
      background-image:
        radial-gradient(circle at 12% 10%, rgba(217, 78, 40, 0.05) 0%, transparent 38%),
        radial-gradient(circle at 88% 85%, rgba(5, 150, 105, 0.05) 0%, transparent 40%),
        linear-gradient(to right, rgba(24, 24, 27, 0.025) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(24, 24, 27, 0.025) 1px, transparent 1px);
      background-size: 100% 100%, 100% 100%, 48px 48px, 48px 48px;
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 14px;
      line-height: 1.6;
      min-height: 100vh;
      -webkit-font-smoothing: antialiased;
      overflow-x: hidden;
    }

    /* Ambient Tectonic Cracks Background Layer */
    .ambient-cracks {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      overflow: hidden;
      opacity: 0.42;
    }

    /* Top Sticky Glass Navigation */
    .top-nav {
      position: sticky;
      top: 0;
      z-index: 50;
      background: rgba(248, 246, 242, 0.85);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border-bottom: 1px solid var(--border);
    }
    .nav-inner {
      max-width: 1320px;
      margin: 0 auto;
      padding: 14px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      flex-wrap: wrap;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      cursor: pointer;
      text-decoration: none;
    }
    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 10px;
      background: linear-gradient(135deg, #18181b 0%, #2e2a27 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 12px rgba(24, 24, 27, 0.16);
      position: relative;
      overflow: hidden;
    }
    .brand-title {
      display: flex;
      flex-direction: column;
      line-height: 1.1;
    }
    .brand-title h1 {
      font-family: var(--font-display);
      font-size: 19px;
      font-weight: 700;
      letter-spacing: 0.8px;
      color: var(--text);
    }
    .brand-title span {
      font-family: var(--font-mono);
      font-size: 10.5px;
      color: var(--accent);
      font-weight: 600;
      letter-spacing: 0.4px;
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .nav-link {
      background: transparent;
      border: none;
      padding: 8px 14px;
      border-radius: 9px;
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      color: var(--text-secondary);
      cursor: pointer;
      transition: all 0.18s ease;
    }
    .nav-link:hover {
      background: rgba(24, 24, 27, 0.05);
      color: var(--text);
    }
    .nav-link.active {
      background: var(--card);
      color: var(--accent);
      box-shadow: var(--shadow-sm);
      border: 1px solid var(--border);
    }

    .meta-bar {
      display: flex;
      align-items: center;
      gap: 14px;
      font-family: var(--font-mono);
      font-size: 11.5px;
      color: var(--text-muted);
      background: var(--card);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 999px;
      box-shadow: var(--shadow-sm);
    }
    .meta-bar b { color: var(--text); font-weight: 600; }
    .status-dot {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--eq);
      margin-right: 6px;
      box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.18);
      animation: pulse-dot 2.4s infinite;
    }
    @keyframes pulse-dot {
      0%, 100% { box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.18); }
      50% { box-shadow: 0 0 0 6px rgba(5, 150, 105, 0.05); }
    }

    /* Main Container */
    .view-container {
      position: relative;
      z-index: 1;
      max-width: 1320px;
      margin: 0 auto;
      padding: 36px 28px 64px;
    }

    .hidden-view { display: none !important; }

    /* ==================== LANDING PAGE ==================== */
    .landing-hero {
      display: grid;
      grid-template-columns: 1.08fr 1fr;
      gap: 48px;
      align-items: center;
      padding: 20px 0 40px;
    }
    @media (max-width: 1024px) {
      .landing-hero { grid-template-columns: 1fr; gap: 36px; }
    }

    .hero-eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 14px;
      border-radius: 999px;
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      color: var(--accent);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.6px;
      text-transform: uppercase;
      margin-bottom: 22px;
    }

    .hero-title {
      font-family: var(--font-display);
      font-size: clamp(36px, 4.8vw, 56px);
      font-weight: 700;
      line-height: 1.06;
      letter-spacing: -1.4px;
      color: var(--text);
      margin-bottom: 20px;
    }
    .hero-title .accent-word {
      color: var(--accent);
      position: relative;
      display: inline-block;
    }
    .hero-title .accent-word::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: -4px;
      height: 6px;
      background-image: url("data:image/svg+xml,%3Csvg width='180' height='8' viewBox='0 0 180 8' fill='none' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M1 5L22 2L44 6L68 1.5L96 5.5L124 2L152 6.5L179 2.5' stroke='%23D94E28' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
      background-repeat: no-repeat;
      background-size: 100% 100%;
    }

    .hero-desc {
      font-size: 16px;
      line-height: 1.7;
      color: var(--text-secondary);
      max-width: 560px;
      margin-bottom: 30px;
    }

    .hero-actions {
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 32px;
    }

    .btn-hero-primary {
      background: var(--text);
      color: #ffffff;
      border: 1px solid var(--text);
      border-radius: 12px;
      padding: 14px 26px;
      font-family: var(--font-sans);
      font-size: 14.5px;
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 8px 22px rgba(24, 24, 27, 0.16);
      transition: all 0.2s ease;
    }
    .btn-hero-primary:hover {
      background: var(--accent);
      border-color: var(--accent);
      transform: translateY(-2px);
      box-shadow: 0 12px 28px rgba(217, 78, 40, 0.28);
    }

    .btn-hero-secondary {
      background: var(--card);
      color: var(--text);
      border: 1px solid var(--border-strong);
      border-radius: 12px;
      padding: 14px 22px;
      font-family: var(--font-sans);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: var(--shadow-sm);
      transition: all 0.2s ease;
    }
    .btn-hero-secondary:hover {
      border-color: var(--text);
      background: #ffffff;
      transform: translateY(-1px);
    }

    .hero-trust-row {
      display: flex;
      gap: 24px;
      flex-wrap: wrap;
      padding-top: 20px;
      border-top: 1px solid var(--border);
    }
    .trust-item {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }
    .trust-val {
      font-family: var(--font-display);
      font-size: 20px;
      font-weight: 700;
      color: var(--text);
    }
    .trust-lbl {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }

    /* Live Hero Seismograph Showcase Card */
    .hero-visual-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 22px;
      padding: 24px;
      box-shadow: var(--shadow-lg);
      position: relative;
      overflow: hidden;
    }
    .hero-visual-card::before {
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 4px;
      background: linear-gradient(90deg, var(--accent), var(--anth), var(--eq));
    }
    .hero-crack-overlay {
      position: absolute;
      top: 0;
      right: 0;
      width: 200px;
      height: 180px;
      pointer-events: none;
      opacity: 0.14;
    }
    .hero-card-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .live-pill {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      padding: 4px 10px;
      border-radius: 999px;
    }
    .live-pulse {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      animation: blink-pulse 1.5s infinite;
    }
    @keyframes blink-pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(1.25); }
    }
    .hero-canvas-frame {
      background: #fcfbf8;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      position: relative;
    }
    canvas#hero-seismo-canvas {
      display: block;
      width: 100%;
      height: 250px;
    }
    .hero-card-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 16px;
    }
    .mini-metric {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
    }
    .mini-metric-lbl {
      font-family: var(--font-mono);
      font-size: 10px;
      text-transform: uppercase;
      color: var(--text-dim);
      margin-bottom: 4px;
    }
    .mini-metric-val {
      font-family: var(--font-mono);
      font-size: 15px;
      font-weight: 700;
      color: var(--text);
    }

    /* Tectonic Crack Section Divider */
    .fault-divider-wrap {
      width: 100%;
      margin: 28px 0 40px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }
    .fault-divider-wrap svg {
      width: 100%;
      height: 28px;
      display: block;
    }

    /* Landing Architecture & Features Grid */
    .section-heading {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 24px;
    }
    .section-kicker {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--accent);
      margin-bottom: 6px;
    }
    .section-h2 {
      font-family: var(--font-display);
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.6px;
      color: var(--text);
    }

    .pillar-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
      margin-bottom: 44px;
    }
    @media (max-width: 960px) {
      .pillar-grid { grid-template-columns: 1fr; }
    }
    .pillar-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 28px;
      box-shadow: var(--shadow-md);
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .pillar-card:hover {
      transform: translateY(-4px);
      box-shadow: var(--shadow-lg);
      border-color: var(--accent-border);
    }
    .pillar-crack {
      position: absolute;
      bottom: -4px;
      right: -4px;
      width: 110px;
      height: 90px;
      opacity: 0.08;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }
    .pillar-card:hover .pillar-crack { opacity: 0.2; }
    .pillar-num {
      font-family: var(--font-mono);
      font-size: 11.5px;
      font-weight: 700;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      display: inline-block;
      padding: 4px 10px;
      border-radius: 8px;
      margin-bottom: 16px;
    }
    .pillar-card h3 {
      font-family: var(--font-display);
      font-size: 19px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 10px;
    }
    .pillar-card p {
      font-size: 13.5px;
      color: var(--text-secondary);
      line-height: 1.65;
    }

    /* Landing Bottom CTA Banner */
    .cta-banner {
      background: linear-gradient(135deg, #ffffff 0%, #fcf7f4 100%);
      border: 1px solid var(--accent-border);
      border-radius: 20px;
      padding: 32px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 24px;
      box-shadow: var(--shadow-md);
      position: relative;
      overflow: hidden;
    }
    .cta-banner h3 {
      font-family: var(--font-display);
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 6px;
    }
    .cta-banner p {
      font-size: 14px;
      color: var(--text-secondary);
      max-width: 620px;
    }

    /* ==================== WORKSPACE (STEPS 1-3) ==================== */
    .workspace-wrap {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* Sequential Step Bar */
    .step-bar {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;
    }
    @media (max-width: 768px) {
      .step-bar { grid-template-columns: 1fr; }
    }
    .step-tab {
      padding: 16px 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      cursor: pointer;
      background: var(--card);
      color: var(--text-muted);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: var(--shadow-sm);
      transition: all 0.18s ease;
      user-select: none;
      position: relative;
      overflow: hidden;
    }
    .step-tab:hover {
      border-color: var(--border-strong);
      color: var(--text);
      transform: translateY(-1px);
    }
    .step-tab.active {
      background: #ffffff;
      color: var(--text);
      font-weight: 700;
      border-color: var(--accent);
      box-shadow: 0 6px 20px rgba(217, 78, 40, 0.10);
    }
    .step-tab.active::after {
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      bottom: 0;
      height: 3px;
      background: var(--accent);
    }
    .step-tab.completed {
      border-color: var(--eq-border);
      background: #fafefd;
      color: var(--text-secondary);
    }
    .step-tab.completed .step-badge {
      background: var(--eq);
      color: #ffffff;
    }
    .step-badge {
      width: 30px;
      height: 30px;
      border-radius: 10px;
      background: var(--bg-subtle);
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-size: 12.5px;
      font-weight: 700;
      flex-shrink: 0;
    }
    .step-tab.active .step-badge {
      background: var(--accent);
      color: #ffffff;
    }
    .step-text-main {
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 700;
    }
    .step-text-sub {
      font-size: 11.5px;
      color: var(--text-dim);
      font-weight: 500;
    }

    /* Page Sections */
    .page-section { display: none; }
    .page-section.active { display: flex; flex-direction: column; gap: 24px; animation: fadeInUp 0.25s ease; }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Unified Card */
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 28px 30px;
      box-shadow: var(--shadow-md);
      display: flex;
      flex-direction: column;
      gap: 22px;
      position: relative;
      overflow: hidden;
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }
    .card-title {
      font-family: var(--font-display);
      font-size: 16px;
      font-weight: 700;
      letter-spacing: -0.2px;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .card-badge {
      font-family: var(--font-mono);
      font-size: 11.5px;
      font-weight: 600;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid var(--accent-border);
      padding: 4px 12px;
      border-radius: 999px;
    }

    /* Sub Tabs & Filter Buttons */
    .pill-group { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill-btn {
      background: var(--card-inner);
      border: 1px solid var(--border);
      color: var(--text-secondary);
      padding: 8px 16px;
      border-radius: 10px;
      cursor: pointer;
      font-family: var(--font-sans);
      font-size: 13px;
      font-weight: 600;
      transition: all 0.16s ease;
    }
    .pill-btn:hover { border-color: var(--border-strong); color: var(--text); background: #ffffff; }
    .pill-btn.active {
      background: var(--text);
      color: #ffffff;
      border-color: var(--text);
      box-shadow: 0 4px 12px rgba(24, 24, 27, 0.14);
    }

    /* Step 1 Search & Filter Toolbar */
    .filter-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 14px;
      background: var(--card-inner);
      padding: 12px 16px;
      border-radius: 14px;
      border: 1px solid var(--border);
    }
    .search-box {
      background: #ffffff;
      border: 1px solid var(--border-strong);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 10px;
      font-family: var(--font-sans);
      font-size: 13px;
      width: 300px;
      max-width: 100%;
      outline: none;
      transition: border-color 0.16s ease, box-shadow 0.16s ease;
    }
    .search-box:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(217, 78, 40, 0.12);
    }

    /* Spacious Catalog Grid */
    .catalog-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
      gap: 16px;
      max-height: 520px;
      overflow-y: auto;
      padding: 4px 6px 8px 2px;
    }
    .catalog-grid::-webkit-scrollbar { width: 6px; }
    .catalog-grid::-webkit-scrollbar-thumb { background: #d4cdc2; border-radius: 99px; }

    .event-card {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      cursor: pointer;
      transition: all 0.18s ease;
      display: flex;
      flex-direction: column;
      gap: 10px;
      position: relative;
      overflow: hidden;
      box-shadow: var(--shadow-sm);
    }
    .event-card:hover {
      border-color: var(--border-strong);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }
    .event-card.active {
      border-color: var(--accent);
      background: #fffdfb;
      box-shadow: 0 8px 24px rgba(217, 78, 40, 0.12);
    }
    .event-card.active::before {
      content: "";
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      width: 4px;
      background: var(--accent);
    }
    .event-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
    .event-card-title {
      font-family: var(--font-display);
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      line-height: 1.35;
    }
    .cat-tag {
      font-family: var(--font-mono);
      font-size: 10px;
      padding: 3px 8px;
      border-radius: 6px;
      font-weight: 700;
      white-space: nowrap;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .cat-tag.eq { background: var(--eq-bg); color: var(--eq); border: 1px solid var(--eq-border); }
    .cat-tag.anth { background: var(--anth-bg); color: var(--anth); border: 1px solid var(--anth-border); }
    .cat-tag.ood { background: var(--ood-bg); color: var(--ood); border: 1px solid var(--ood-border); }

    .event-card-desc { font-size: 12.5px; color: var(--text-muted); line-height: 1.5; }
    .event-card-meta {
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-dim);
      padding-top: 6px;
      border-top: 1px dashed var(--border);
    }

    /* Upload Zone */
    .dropzone {
      border: 2px dashed var(--border-strong);
      border-radius: 16px;
      padding: 48px 28px;
      text-align: center;
      background: var(--card-inner);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
      transition: all 0.18s ease;
    }
    .dropzone:hover {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .dropzone input { display: none; }

    /* Primary & Secondary Action Buttons */
    .btn-primary {
      background: var(--accent);
      color: #ffffff;
      border: 1px solid var(--accent);
      border-radius: 11px;
      padding: 11px 22px;
      font-family: var(--font-sans);
      font-weight: 700;
      font-size: 13.5px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      box-shadow: 0 6px 16px rgba(217, 78, 40, 0.22);
      transition: all 0.18s ease;
    }
    .btn-primary:hover {
      background: var(--accent-hover);
      border-color: var(--accent-hover);
      transform: translateY(-1px);
    }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

    .btn-secondary {
      background: #ffffff;
      color: var(--text);
      border: 1px solid var(--border-strong);
      border-radius: 11px;
      padding: 10px 18px;
      font-family: var(--font-sans);
      font-weight: 600;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.16s ease;
    }
    .btn-secondary:hover {
      background: var(--card-inner);
      border-color: var(--text);
    }

    .nav-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 16px;
      padding-top: 18px;
      border-top: 1px solid var(--border);
    }

    /* Step 2 Layout */
    .step2-layout {
      display: grid;
      grid-template-columns: 1fr 360px;
      gap: 24px;
    }
    @media (max-width: 1060px) {
      .step2-layout { grid-template-columns: 1fr; }
    }

    /* Waveform Canvas Toolbar */
    .canvas-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-bottom: none;
      border-radius: 14px 14px 0 0;
      padding: 10px 16px;
      gap: 10px;
    }
    .canvas-group { display: flex; align-items: center; gap: 6px; font-size: 12px; }

    .ctrl-btn {
      background: #ffffff;
      border: 1px solid var(--border-strong);
      color: var(--text-secondary);
      padding: 5px 11px;
      border-radius: 8px;
      cursor: pointer;
      font-family: var(--font-mono);
      font-size: 11.5px;
      font-weight: 600;
      transition: all 0.14s ease;
    }
    .ctrl-btn.active {
      background: var(--text);
      color: #ffffff;
      border-color: var(--text);
    }
    .ctrl-btn:hover:not(.active) {
      border-color: var(--text);
      color: var(--text);
    }

    .canvas-box {
      position: relative;
      background: #fcfbf8;
      border: 1px solid var(--border);
      border-radius: 0 0 14px 14px;
      overflow: hidden;
      cursor: crosshair;
      box-shadow: inset 0 2px 6px rgba(24, 24, 27, 0.03);
    }
    canvas#waveform-canvas { display: block; width: 100%; height: 390px; }

    /* Light Glass Canvas HUD */
    .canvas-hud {
      position: absolute;
      top: 12px;
      right: 14px;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid var(--border-strong);
      border-radius: 10px;
      padding: 7px 14px;
      font-family: var(--font-mono);
      font-size: 11.5px;
      color: var(--text-muted);
      display: flex;
      gap: 14px;
      pointer-events: none;
      box-shadow: var(--shadow-sm);
    }

    /* Inner Sub-Cards */
    .sub-card {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .sub-card-title {
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }

    /* Step 2 Right Pane: Station Array Table */
    .station-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 290px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .station-row {
      background: #ffffff;
      border: 1px solid var(--border);
      border-radius: 11px;
      padding: 11px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .station-row:hover { border-color: var(--border-strong); background: #fffdfb; }
    .station-row.selected {
      border-color: var(--accent);
      background: var(--accent-soft);
    }
    .station-info { display: flex; flex-direction: column; gap: 3px; }
    .station-code { font-family: var(--font-mono); font-weight: 700; color: var(--text); font-size: 12px; }
    .station-dist { font-size: 11px; color: var(--text-muted); }

    /* Step 3 Grid */
    .step3-layout {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 24px;
    }
    @media (max-width: 960px) {
      .step3-layout { grid-template-columns: 1fr; }
    }

    .verdict-box {
      background: var(--card-inner);
      border: 1.5px solid var(--border);
      border-radius: 16px;
      padding: 22px 24px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      position: relative;
      overflow: hidden;
    }
    .verdict-box.eq {
      background: var(--eq-bg);
      border-color: var(--eq-border);
    }
    .verdict-box.anth {
      background: var(--anth-bg);
      border-color: var(--anth-border);
    }
    .verdict-box.ood {
      background: var(--ood-bg);
      border-color: var(--ood-border);
    }
    .verdict-label {
      font-family: var(--font-display);
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.4px;
      text-transform: uppercase;
      color: var(--text);
    }
    .verdict-box.eq .verdict-label { color: #047857; }
    .verdict-box.anth .verdict-label { color: #b45309; }
    .verdict-box.ood .verdict-label { color: #6d28d9; }

    .prob-bar-track {
      height: 9px;
      background: #eae6de;
      border-radius: 99px;
      overflow: hidden;
      margin-top: 6px;
    }
    .prob-bar-fill { height: 100%; border-radius: 99px; transition: width 0.35s ease; }
    .prob-bar-fill.eq { background: linear-gradient(90deg, #10b981, #059669); }
    .prob-bar-fill.anth { background: linear-gradient(90deg, #f59e0b, #d97706); }

    .gauge-track {
      height: 10px;
      border-radius: 99px;
      background: linear-gradient(to right, #10b981 0%, #f59e0b 50%, #dc2626 100%);
      position: relative;
      margin-top: 6px;
    }
    .gauge-pin {
      position: absolute;
      top: -4px;
      width: 6px;
      height: 18px;
      background: #18181b;
      border: 2px solid #ffffff;
      border-radius: 4px;
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28);
      transform: translateX(-50%);
      transition: left 0.35s ease;
    }

    .diag-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 12px; }
    .diag-table td {
      padding: 9px 6px;
      border-bottom: 1px solid var(--border);
      text-align: left;
      color: var(--text-secondary);
    }
    .diag-table tr:last-child td { border-bottom: none; }
    .diag-table td.num { text-align: right; font-weight: 700; color: var(--text); }

    .attn-matrix { display: grid; gap: 5px; }
    .attn-cell {
      height: 36px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-size: 11.5px;
      font-weight: 700;
      border-radius: 8px;
      border: 1px solid rgba(24, 24, 27, 0.06);
      transition: transform 0.15s ease;
    }
    .attn-cell:hover { transform: scale(1.04); }
  </style>
</head>
<body>

  <!-- Subtle Tectonic Fault Cracks Background Layer -->
  <div class="ambient-cracks" aria-hidden="true">
    <svg width="100%" height="100%" viewBox="0 0 1440 900" fill="none" preserveAspectRatio="xMidYMid slice">
      <!-- Upper left tectonic fissure -->
      <path d="M-20 110 L95 148 L158 132 L245 198 L312 185 L395 252" stroke="#D94E28" stroke-width="1.4" stroke-opacity="0.22"/>
      <path d="M158 132 L196 86 L248 64" stroke="#D94E28" stroke-width="1" stroke-opacity="0.16"/>
      <path d="M245 198 L228 265 L254 318" stroke="#18181B" stroke-width="1" stroke-opacity="0.10"/>
      <!-- Lower right tectonic fault branch -->
      <path d="M1460 760 L1320 712 L1235 745 L1120 670 L1045 695 L950 620" stroke="#18181B" stroke-width="1.3" stroke-opacity="0.14"/>
      <path d="M1235 745 L1190 820 L1145 875" stroke="#D94E28" stroke-width="1.1" stroke-opacity="0.18"/>
      <path d="M1120 670 L1152 605 L1195 572" stroke="#18181B" stroke-width="0.9" stroke-opacity="0.12"/>
    </svg>
  </div>

  <!-- Top Sticky Navigation -->
  <nav class="top-nav">
    <div class="nav-inner">
      <div class="brand" onclick="showLanding()">
        <div class="brand-mark">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <path d="M2 13H6L9 5L13 19L16.5 10L19 13H22" stroke="#F8F6F2" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M13 19L15 22" stroke="#D94E28" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="brand-title">
          <h1>SEISMO</h1>
          <span>GATv2 + EDL SEISMIC AI</span>
        </div>
      </div>

      <div class="nav-links">
        <button class="nav-link active" id="nav-btn-landing" onclick="showLanding()">Overview</button>
        <button class="nav-link" id="nav-btn-workspace" onclick="openWorkspace(1)">Interactive Analyzer</button>
        <button class="nav-link" id="nav-btn-upload" onclick="openWorkspaceUpload()">Upload Waveform</button>
      </div>

      <div class="meta-bar">
        <span><span class="status-dot"></span><b id="hdr-model">Ready</b></span>
        <span>Params: <b id="hdr-params">--</b></span>
        <span>Device: <b id="hdr-device">--</b></span>
      </div>
    </div>
  </nav>

  <!-- ==================== VIEW 1: THEMED LANDING PAGE ==================== -->
  <main class="view-container" id="landing-view">
    <section class="landing-hero">
      <div>
        <div class="hero-eyebrow">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none"><path d="M1 9H4.5L6.5 3L9.5 13L11.5 7.5L13 9H15" stroke="#D94E28" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
          Pacific Northwest Seismic Network &bull; Evidential GNN
        </div>

        <h2 class="hero-title">
          Reading the Earth's <span class="accent-word">Fractures</span> with Certainty.
        </h2>

        <p class="hero-desc">
          SEISMO discriminates natural tectonic earthquakes from anthropogenic quarry blasts across multi-station networks using Dynamic Graph Attention (GATv2) and Subjective Logic Evidential Deep Learning &mdash; automatically flagging exotic out-of-distribution anomalies.
        </p>

        <div class="hero-actions">
          <button class="btn-hero-primary" onclick="openWorkspace(1)">
            Launch Seismic Analyzer
            <svg width="17" height="17" viewBox="0 0 20 20" fill="none"><path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </button>
          <button class="btn-hero-secondary" onclick="openWorkspaceUpload()">
            <svg width="16" height="16" viewBox="0 0 20 20" fill="none"><path d="M10 13V4M10 4L6.5 7.5M10 4L13.5 7.5M3 15.5H17" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
            Test Custom CSV / JSON
          </button>
        </div>

        <div class="hero-trust-row">
          <div class="trust-item">
            <span class="trust-val">93.43%</span>
            <span class="trust-lbl">2021+ Test Accuracy</span>
          </div>
          <div class="trust-item">
            <span class="trust-val">+87.5%</span>
            <span class="trust-lbl">OOD Vacuity Shift</span>
          </div>
          <div class="trust-item">
            <span class="trust-val">13,084</span>
            <span class="trust-lbl">PNW Balanced Events</span>
          </div>
          <div class="trust-item">
            <span class="trust-val">348K</span>
            <span class="trust-lbl">Trained Parameters</span>
          </div>
        </div>
      </div>

      <!-- Right Hero Interactive Live Seismograph Card -->
      <div class="hero-visual-card">
        <svg class="hero-crack-overlay" viewBox="0 0 200 180" fill="none">
          <path d="M200 18 L155 44 L132 34 L94 78 L66 68 L28 112" stroke="#D94E28" stroke-width="2"/>
          <path d="M132 34 L118 8" stroke="#18181B" stroke-width="1.5"/>
          <path d="M94 78 L112 118 L98 148" stroke="#D94E28" stroke-width="1.4"/>
        </svg>

        <div class="hero-card-top">
          <div class="live-pill">
            <span class="live-pulse"></span>
            LIVE SEISMOGRAPH MONITOR
          </div>
          <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">UW.ALKI &bull; 50 Hz &bull; E / N / Z</span>
        </div>

        <div class="hero-canvas-frame">
          <canvas id="hero-seismo-canvas" width="560" height="250"></canvas>
        </div>

        <div class="hero-card-metrics">
          <div class="mini-metric">
            <div class="mini-metric-lbl">In-Domain Vacuity</div>
            <div class="mini-metric-val" style="color: var(--eq);">u = 0.2196</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-lbl">Exotic OOD Vacuity</div>
            <div class="mini-metric-val" style="color: var(--ood);">u = 0.4117</div>
          </div>
          <div class="mini-metric">
            <div class="mini-metric-lbl">Curated Scenarios</div>
            <div class="mini-metric-val" style="color: var(--accent);">22 Ready</div>
          </div>
        </div>
      </div>
    </section>

    <!-- Tectonic Fault-Line Divider -->
    <div class="fault-divider-wrap" aria-hidden="true">
      <svg viewBox="0 0 1200 28" fill="none" preserveAspectRatio="none">
        <path d="M0 14 L240 14 L268 7 L298 20 L330 10 L358 16 L400 14 L760 14 L788 22 L818 6 L848 18 L874 12 L910 14 L1200 14" stroke="#D4CDC2" stroke-width="1.5"/>
        <path d="M268 7 L298 20 L330 10" stroke="#D94E28" stroke-width="2" stroke-linecap="round"/>
        <path d="M788 22 L818 6 L848 18" stroke="#D94E28" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </div>

    <!-- Three-Stage Architecture Highlights -->
    <section>
      <div class="section-heading">
        <div>
          <div class="section-kicker">End-to-End Geophysical Architecture</div>
          <h2 class="section-h2">Built for Multi-Station Wavefield Reasoning</h2>
        </div>
        <span style="font-family: var(--font-mono); font-size: 12px; color: var(--text-muted);">1D TCN &rarr; k-NN GATv2 &rarr; Dirichlet EDL</span>
      </div>

      <div class="pillar-grid">
        <div class="pillar-card">
          <svg class="pillar-crack" viewBox="0 0 110 90" fill="none">
            <path d="M110 12 L76 34 L58 28 L34 62 L10 88" stroke="#D94E28" stroke-width="1.8"/>
            <path d="M58 28 L48 5" stroke="#18181B" stroke-width="1.2"/>
          </svg>
          <span class="pillar-num">STAGE 01 &bull; ENCODER</span>
          <h3>Dilated 1D Residual CNN</h3>
          <p>
            Four residual blocks with exponential dilations (1, 2, 4, 8) extract 128-dimensional embeddings from 3-component (E, N, Z) traces while preserving critical vertical-to-horizontal amplitude ratios.
          </p>
        </div>

        <div class="pillar-card">
          <svg class="pillar-crack" viewBox="0 0 110 90" fill="none">
            <path d="M110 26 L82 40 L68 58 L38 52 L8 84" stroke="#059669" stroke-width="1.8"/>
            <path d="M68 58 L84 86" stroke="#18181B" stroke-width="1.2"/>
          </svg>
          <span class="pillar-num">STAGE 02 &bull; SPATIAL GRAPH</span>
          <h3>Dynamic GATv2 Attention</h3>
          <p>
            Connects active stations via Haversine k-NN graphs with 4 attention heads. Mask-aware message passing prioritizes near-source stations with clear P/S separation and adapts to telemetry dropouts.
          </p>
        </div>

        <div class="pillar-card">
          <svg class="pillar-crack" viewBox="0 0 110 90" fill="none">
            <path d="M110 18 L84 30 L72 48 L44 44 L16 82" stroke="#7C3AED" stroke-width="1.8"/>
            <path d="M44 44 L36 18" stroke="#18181B" stroke-width="1.2"/>
          </svg>
          <span class="pillar-num">STAGE 03 &bull; UNCERTAINTY</span>
          <h3>Evidential Dirichlet Head</h3>
          <p>
            Replaces overconfident softmax outputs with Subjective Logic evidence parameters. Quantifies epistemic vacuity (u = K / S) to flag unfamiliar volcanic tremors, landslides, and sonic booms.
          </p>
        </div>
      </div>

      <div class="cta-banner">
        <div>
          <h3>Ready to inspect waveforms and run live evidential inference?</h3>
          <p>Explore 22 curated Pacific Northwest earthquakes, quarry blasts, and exotic OOD events, simulate sensor noise or station dropouts, or upload your own CSV/JSON trace.</p>
        </div>
        <button class="btn-hero-primary" onclick="openWorkspace(1)">
          Open 3-Step Analyzer &rarr;
        </button>
      </div>
    </section>
  </main>

  <!-- ==================== VIEW 2: 3-STEP INTERACTIVE WORKSPACE ==================== -->
  <main class="view-container hidden-view" id="workspace-view">
    <div class="workspace-wrap">
      <!-- Step Navigation Bar -->
      <div class="step-bar">
        <div class="step-tab active" id="tab-step-1" onclick="goToStep(1)">
          <div class="step-badge">1</div>
          <div>
            <div class="step-text-main">Select Waveform Scenario</div>
            <div class="step-text-sub">22 curated events or CSV/JSON upload</div>
          </div>
        </div>
        <div class="step-tab" id="tab-step-2" onclick="goToStep(2)">
          <div class="step-badge">2</div>
          <div>
            <div class="step-text-main">Waveforms &amp; Network Setup</div>
            <div class="step-text-sub">Inspect E/N/Z traces, dropout &amp; noise</div>
          </div>
        </div>
        <div class="step-tab" id="tab-step-3" onclick="goToStep(3)">
          <div class="step-badge">3</div>
          <div>
            <div class="step-text-main">Evidential Classification</div>
            <div class="step-text-sub">Verdict, vacuity gauge &amp; GATv2 matrix</div>
          </div>
        </div>
      </div>

      <!-- ==================== STEP 1: INPUT SELECTION ==================== -->
      <section class="page-section active" id="page-step-1">
        <div class="card">
          <div class="card-head">
            <div class="card-title">Step 1 &mdash; Select Input Waveform Scenario</div>
            <span class="card-badge" id="catalog-count-badge">Catalog: 22 Events</span>
          </div>

          <!-- Mode Toggle -->
          <div class="pill-group">
            <button class="pill-btn active" id="btn-mode-presets" onclick="switchCatalogMode('presets')">Curated Seismic Events (22)</button>
            <button class="pill-btn" id="btn-mode-upload" onclick="switchCatalogMode('upload')">Upload Custom File (.CSV / .JSON)</button>
          </div>

          <!-- Mode Presets -->
          <div id="section-presets" style="display: flex; flex-direction: column; gap: 16px;">
            <div class="filter-toolbar">
              <div class="pill-group">
                <button class="ctrl-btn active" id="filter-all" onclick="filterCatalog('all')">All (22)</button>
                <button class="ctrl-btn" id="filter-eq" onclick="filterCatalog('Earthquake')">Earthquakes (8)</button>
                <button class="ctrl-btn" id="filter-anth" onclick="filterCatalog('Quarry Blast')">Quarry Blasts (8)</button>
                <button class="ctrl-btn" id="filter-ood" onclick="filterCatalog('Out-of-Distribution')">OOD / Exotic (6)</button>
              </div>
              <input type="text" class="search-box" id="search-box" placeholder="Search events, stations, locations..." oninput="onSearchInput(event)">
            </div>

            <div class="catalog-grid" id="catalog-grid"></div>
          </div>

          <!-- Mode Upload -->
          <div id="section-upload" style="display: none; flex-direction: column; gap: 16px;">
            <div class="dropzone" onclick="document.getElementById('file-picker').click()">
              <div style="width: 48px; height: 48px; border-radius: 14px; background: var(--accent-soft); border: 1px solid var(--accent-border); display: flex; align-items: center; justify-content: center; color: var(--accent); font-size: 22px; font-weight: 700;">&#8682;</div>
              <div style="font-family: var(--font-display); font-weight: 700; font-size: 16px; color: var(--text);">Click to Browse or Drag &amp; Drop Waveform File</div>
              <div style="font-size: 13px; color: var(--text-muted);">
                Supports <b>.csv</b> (columns: E, N, Z or time, E, N, Z) or <b>.json</b> (multi-station event)
              </div>
              <input type="file" id="file-picker" accept=".json,.csv" onchange="onFileUploaded(event)">
            </div>

            <div id="upload-status" style="display: none; background: var(--card-inner); padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border-strong); font-family: var(--font-mono); font-size: 12px;"></div>

            <div style="display: flex; gap: 14px; font-size: 12.5px; color: var(--text-muted); align-items: center; flex-wrap: wrap;">
              <span style="font-weight: 600;">Download Format Templates:</span>
              <a href="/api/template/csv" download="template_waveform.csv" style="color: var(--accent); font-weight: 600; text-decoration: none;">&darr; Sample CSV</a>
              <span>&bull;</span>
              <a href="/api/template/json" download="template_waveform.json" style="color: var(--accent); font-weight: 600; text-decoration: none;">&darr; Sample JSON</a>
            </div>
          </div>

          <!-- Summary & Next Button -->
          <div class="nav-footer">
            <div>
              <div style="font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.5px;">Selected Event</div>
              <div style="font-family: var(--font-display); font-size: 16px; font-weight: 700; color: var(--text);" id="selected-title">Loading...</div>
              <div style="font-size: 12px; color: var(--accent); font-family: var(--font-mono);" id="selected-meta"></div>
            </div>
            <button class="btn-primary" onclick="goToStep(2)">
              Inspect Waveforms &rarr;
            </button>
          </div>
        </div>
      </section>

      <!-- ==================== STEP 2: WAVEFORMS & NETWORK SETUP ==================== -->
      <section class="page-section" id="page-step-2">
        <div class="card">
          <div class="card-head">
            <div class="card-title">Step 2 &mdash; Waveform Inspector &amp; Station Network</div>
            <span class="card-badge" id="step2-badge"></span>
          </div>

          <div class="step2-layout">
            <!-- Left: Waveform Canvas -->
            <div style="display: flex; flex-direction: column;">
              <!-- Station Tabs -->
              <div class="pill-group" id="stn-tabs" style="margin-bottom: 12px;"></div>

              <!-- Canvas Toolbar -->
              <div class="canvas-bar">
                <div class="canvas-group">
                  <span style="color: var(--text-muted); font-weight: 600;">Channels:</span>
                  <button class="ctrl-btn active" id="btn-comp-0" onclick="toggleChannel(0)">E (East)</button>
                  <button class="ctrl-btn active" id="btn-comp-1" onclick="toggleChannel(1)">N (North)</button>
                  <button class="ctrl-btn active" id="btn-comp-2" onclick="toggleChannel(2)">Z (Vert)</button>
                </div>

                <div class="canvas-group">
                  <span style="color: var(--text-muted); font-weight: 600;">Gain:</span>
                  <button class="ctrl-btn active" id="gain-1" onclick="setGain(1.0)">1.0x</button>
                  <button class="ctrl-btn" id="gain-2" onclick="setGain(2.0)">2.0x</button>
                  <button class="ctrl-btn" id="gain-4" onclick="setGain(4.0)">4.0x</button>
                </div>

                <div class="canvas-group">
                  <button class="ctrl-btn" onclick="resetZoom()">&#x21ba; Reset Zoom</button>
                </div>
              </div>

              <!-- Interactive Canvas -->
              <div class="canvas-box" id="canvas-wrapper">
                <canvas id="waveform-canvas" width="800" height="390"></canvas>
                <div class="canvas-hud">
                  <div>Time: <span id="hud-time" style="color: var(--text); font-weight: 700;">--</span></div>
                  <div>E: <span id="hud-e" style="color: #0284c7; font-weight: 700;">--</span></div>
                  <div>N: <span id="hud-n" style="color: #059669; font-weight: 700;">--</span></div>
                  <div>Z: <span id="hud-z" style="color: #d94e28; font-weight: 700;">--</span></div>
                </div>
              </div>

              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 10px;">
                <span style="font-size: 11.5px; color: var(--text-muted); font-family: var(--font-mono);">
                  Drag horizontally to zoom. Hover to inspect amplitudes. Red line marks P-arrival (10.0s).
                </span>
                <span style="font-size: 11.5px; color: var(--accent); font-family: var(--font-mono); font-weight: 600;" id="zoom-indicator">Window: 0.0s &ndash; 60.0s</span>
              </div>

              <div style="font-size: 13px; color: var(--text-secondary); background: var(--card-inner); padding: 14px 16px; border-radius: 12px; border: 1px solid var(--border); border-left: 4px solid var(--accent); margin-top: 14px;" id="event-desc-box"></div>
            </div>

            <!-- Right: Station Array & Robustness -->
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <!-- Station Array Panel -->
              <div class="sub-card">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                  <span class="sub-card-title">Station Array Network</span>
                  <span style="font-size: 11px; font-family: var(--font-mono); color: var(--accent); font-weight: 700;" id="active-stn-badge">-- Active</span>
                </div>
                <div class="station-list" id="station-list"></div>
                <span style="font-size: 11.5px; color: var(--text-muted); line-height: 1.45;">
                  Click a station row to inspect its waveform. Uncheck a station to simulate sensor telemetry dropout.
                </span>
              </div>

              <!-- Robustness: Gaussian Noise Slider -->
              <div class="sub-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <span class="sub-card-title">Injected Gaussian Noise</span>
                  <span id="noise-val" style="font-family: var(--font-mono); color: var(--accent); font-weight: 700; font-size: 13px;">&sigma; = 0.00</span>
                </div>
                <input type="range" id="noise-slider" min="0" max="0.20" step="0.02" value="0" style="width: 100%; accent-color: var(--accent); cursor: pointer;">
                <span style="font-size: 11.5px; color: var(--text-muted); line-height: 1.45;">
                  Injects microseismic noise to stress-test Evidential vacuity uncertainty degradation.
                </span>
              </div>
            </div>
          </div>

          <div class="nav-footer">
            <button class="btn-secondary" onclick="goToStep(1)">&larr; Back to Step 1</button>
            <button class="btn-primary" onclick="goToStep(3)">
              Run Model Classification &rarr;
            </button>
          </div>
        </div>
      </section>

      <!-- ==================== STEP 3: EVIDENTIAL INFERENCE & DIAGNOSTICS ==================== -->
      <section class="page-section" id="page-step-3">
        <div class="card">
          <div class="card-head">
            <div class="card-title">Step 3 &mdash; Model Classification &amp; Evidential Diagnostics</div>
            <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
              <span style="font-size: 12px; color: var(--text-muted); font-family: var(--font-mono);" id="step3-latency">Latency: -- ms</span>
              <button class="btn-secondary" style="padding: 7px 13px; font-size: 12px;" onclick="executeInference()">&#x21ba; Re-Run</button>
              <button class="btn-secondary" style="padding: 7px 13px; font-size: 12px; color: var(--accent); border-color: var(--accent-border);" onclick="exportAuditJSON()">&darr; Export Audit JSON</button>
            </div>
          </div>

          <div class="step3-layout">
            <!-- Left: Classification & Uncertainty -->
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <!-- Primary Verdict -->
              <div class="verdict-box" id="verdict-card">
                <svg style="position: absolute; top: 0; right: 0; width: 130px; height: 100px; opacity: 0.14; pointer-events: none;" viewBox="0 0 130 100" fill="none">
                  <path d="M130 10 L96 32 L78 24 L48 56 L18 82" stroke="currentColor" stroke-width="2"/>
                  <path d="M78 24 L70 4" stroke="currentColor" stroke-width="1.3"/>
                </svg>
                <div style="font-family: var(--font-mono); font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: var(--text-muted);">Classification Verdict</div>
                <div class="verdict-label" id="verdict-label">EVALUATING...</div>
                <div style="font-size: 13px; color: var(--text-secondary);" id="verdict-sub">Computing GATv2 node embeddings...</div>
              </div>

              <!-- Probabilities -->
              <div class="sub-card">
                <span class="sub-card-title">Class Probability Distribution</span>

                <div>
                  <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 12.5px;">
                    <span style="color: var(--text); font-weight: 600;">Natural Tectonic Earthquake</span>
                    <span id="prob-eq-val" style="font-weight: 700; color: var(--eq);">--%</span>
                  </div>
                  <div class="prob-bar-track">
                    <div class="prob-bar-fill eq" id="prob-eq-bar" style="width: 0%;"></div>
                  </div>
                </div>

                <div>
                  <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 12.5px;">
                    <span style="color: var(--text); font-weight: 600;">Anthropogenic Event (Quarry Blast)</span>
                    <span id="prob-anth-val" style="font-weight: 700; color: var(--anth);">--%</span>
                  </div>
                  <div class="prob-bar-track">
                    <div class="prob-bar-fill anth" id="prob-anth-bar" style="width: 0%;"></div>
                  </div>
                </div>
              </div>

              <!-- Vacuity Uncertainty Gauge -->
              <div class="sub-card">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                  <span class="sub-card-title">Epistemic Vacuity Uncertainty (u = K / S)</span>
                  <span id="vacuity-num" style="font-size: 24px; font-weight: 700; font-family: var(--font-mono); color: var(--text);">--</span>
                </div>
                <div class="gauge-track">
                  <div class="gauge-pin" id="vacuity-pin" style="left: 0%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 10.5px; font-family: var(--font-mono); color: var(--text-dim);">
                  <span>0.0 (High Certainty)</span>
                  <span>0.5 (Moderate Risk)</span>
                  <span>1.0 (High Vacuity / OOD)</span>
                </div>
                <div style="font-size: 12.5px; color: var(--text-secondary); line-height: 1.5; margin-top: 4px;" id="vacuity-status-text"></div>
              </div>
            </div>

            <!-- Right: Subjective Logic & GAT Heatmap -->
            <div style="display: flex; flex-direction: column; gap: 20px;">
              <!-- Subjective Logic Table -->
              <div class="sub-card">
                <span class="sub-card-title">Dirichlet Subjective Logic Parameters</span>
                <table class="diag-table">
                  <tbody>
                    <tr><td>Non-negative Evidence (e)</td><td class="num" id="diag-evidence">--</td></tr>
                    <tr><td>Dirichlet Parameters (&alpha; = e + 1)</td><td class="num" id="diag-alpha">--</td></tr>
                    <tr><td>Total Dirichlet Strength (S = &sum; &alpha;)</td><td class="num" id="diag-strength">--</td></tr>
                    <tr><td>Belief Masses (b_k = e_k / S)</td><td class="num" id="diag-belief">--</td></tr>
                    <tr><td>Belief Identity (b_eq + b_anth + u)</td><td class="num" id="diag-identity" style="color: var(--accent);">1.0000</td></tr>
                  </tbody>
                </table>
              </div>

              <!-- GATv2 Attention Matrix -->
              <div class="sub-card">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                  <span class="sub-card-title">GATv2 Spatial Attention Matrix</span>
                  <span style="font-size: 10.5px; font-family: var(--font-mono); color: var(--text-dim);">Target &larr; Source (&alpha;<sub>ij</sub>)</span>
                </div>
                <div class="attn-matrix" id="attn-matrix-box"></div>
                <span style="font-size: 11.5px; color: var(--text-muted);">Inter-station dynamic attention weights computed during graph message passing.</span>
              </div>
            </div>
          </div>

          <div class="nav-footer">
            <button class="btn-secondary" onclick="goToStep(2)">&larr; Back to Step 2: Waveforms</button>
            <button class="btn-secondary" onclick="goToStep(1)">&#x21ba; Select New Scenario</button>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    // View Switching (Landing <-> Workspace)
    function showLanding() {
      document.getElementById('landing-view').classList.remove('hidden-view');
      document.getElementById('workspace-view').classList.add('hidden-view');
      document.getElementById('nav-btn-landing').classList.add('active');
      document.getElementById('nav-btn-workspace').classList.remove('active');
      document.getElementById('nav-btn-upload').classList.remove('active');
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function openWorkspace(step = 1) {
      document.getElementById('landing-view').classList.add('hidden-view');
      document.getElementById('workspace-view').classList.remove('hidden-view');
      document.getElementById('nav-btn-landing').classList.remove('active');
      document.getElementById('nav-btn-workspace').classList.add('active');
      document.getElementById('nav-btn-upload').classList.remove('active');
      switchCatalogMode('presets');
      goToStep(step);
    }

    function openWorkspaceUpload() {
      document.getElementById('landing-view').classList.add('hidden-view');
      document.getElementById('workspace-view').classList.remove('hidden-view');
      document.getElementById('nav-btn-landing').classList.remove('active');
      document.getElementById('nav-btn-workspace').classList.remove('active');
      document.getElementById('nav-btn-upload').classList.add('active');
      goToStep(1);
      switchCatalogMode('upload');
    }

    // Animated Hero Live Seismograph Canvas
    let heroPhase = 0;
    function startHeroSeismoAnimation() {
      const canvas = document.getElementById('hero-seismo-canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      function renderHeroFrame() {
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.parentElement.clientWidth || 540;
        const h = 250;
        if (canvas.width !== Math.floor(w * dpr) || canvas.height !== Math.floor(h * dpr)) {
          canvas.width = Math.floor(w * dpr);
          canvas.height = Math.floor(h * dpr);
        }

        ctx.save();
        ctx.scale(dpr, dpr);

        // Light seismograph chart paper
        ctx.fillStyle = '#fcfbf8';
        ctx.fillRect(0, 0, w, h);

        // Fine vertical/horizontal grid
        ctx.strokeStyle = '#efeae0';
        ctx.lineWidth = 1;
        for (let gx = 0; gx < w; gx += 32) {
          ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
        }
        for (let gy = 0; gy < h; gy += 32) {
          ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
        }

        // 3 Animated seismic channels
        const channels = [
          { y: h * 0.22, color: '#0284c7', label: 'E', amp: 20, freq: 0.085, burstPos: 0.38 },
          { y: h * 0.52, color: '#059669', label: 'N', amp: 26, freq: 0.068, burstPos: 0.52 },
          { y: h * 0.80, color: '#d94e28', label: 'Z', amp: 22, freq: 0.115, burstPos: 0.30 },
        ];

        channels.forEach((ch) => {
          // Baseline
          ctx.strokeStyle = '#e8e2d7';
          ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(0, ch.y); ctx.lineTo(w, ch.y); ctx.stroke();

          // Channel label pill
          ctx.fillStyle = ch.color;
          ctx.font = '700 10px "JetBrains Mono", monospace';
          ctx.fillText(ch.label, 10, ch.y - 12);

          // Waveform trace
          ctx.strokeStyle = ch.color;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          for (let x = 0; x < w; x += 2) {
            const norm = x / w;
            const shift = (norm + heroPhase) % 1.0;
            const d1 = shift - ch.burstPos;
            const env1 = d1 > 0 ? Math.exp(-d1 * 11) * Math.min(1, d1 * 45) : 0;
            const d2 = shift - (ch.burstPos + 0.16);
            const env2 = d2 > 0 ? Math.exp(-d2 * 7) * Math.min(1, d2 * 35) * 1.25 : 0;

            const micro = Math.sin(x * 0.17 + heroPhase * 40) * 1.4;
            const sig = (Math.sin(x * ch.freq + heroPhase * 60) * env1 + Math.cos(x * ch.freq * 0.72 - heroPhase * 45) * env2) * ch.amp;
            const py = ch.y - (micro + sig);
            if (x === 0) ctx.moveTo(x, py);
            else ctx.lineTo(x, py);
          }
          ctx.stroke();
        });

        ctx.restore();
        heroPhase = (heroPhase + 0.0018) % 1.0;
        requestAnimationFrame(renderHeroFrame);
      }
      requestAnimationFrame(renderHeroFrame);
    }

    // State
    let currentStep = 1;
    let catalog = [];
    let filteredCatalog = [];
    let activeFilter = 'all';
    let searchQuery = '';
    let currentEvent = null;
    let selectedStationIndex = 0;
    let excludedStationIndices = new Set();
    let latestInferenceResult = null;

    // Waveform canvas state
    let visibleComponents = [true, true, true]; // E, N, Z
    let waveformGain = 1.0;
    let zoomStart = 0.0;
    let zoomEnd = 60.0;
    let isDragging = false;
    let dragStartX = 0;
    let dragCurrX = 0;

    // Initialization
    async function init() {
      startHeroSeismoAnimation();

      try {
        const info = await (await fetch('/api/model_info')).json();
        document.getElementById('hdr-model').textContent = info.model_name.replace(' (EDL)', '');
        document.getElementById('hdr-params').textContent = Number(info.parameters).toLocaleString();
        document.getElementById('hdr-device').textContent = info.device.toUpperCase();
      } catch (e) {
        console.error('Failed to load model info', e);
      }

      try {
        catalog = await (await fetch('/api/samples')).json();
        filteredCatalog = catalog;
        document.getElementById('catalog-count-badge').textContent = `Catalog: ${catalog.length} Scenarios`;
        renderCatalog();
        if (catalog.length > 0) {
          selectEvent(catalog[0].id);
        }
      } catch (e) {
        console.error('Failed to load catalog', e);
      }

      document.getElementById('noise-slider').addEventListener('input', (e) => {
        document.getElementById('noise-val').textContent = `\u03c3 = ${Number(e.target.value).toFixed(2)}`;
      });

      setupCanvasListeners();
      window.addEventListener('resize', handleCanvasResize);
    }

    // Step Navigation
    function goToStep(step) {
      if (step < 1 || step > 3) return;
      currentStep = step;

      for (let s = 1; s <= 3; s++) {
        const tab = document.getElementById(`tab-step-${s}`);
        const page = document.getElementById(`page-step-${s}`);
        if (s === currentStep) {
          tab.className = 'step-tab active';
          page.className = 'page-section active';
        } else if (s < currentStep) {
          tab.className = 'step-tab completed';
          page.className = 'page-section';
        } else {
          tab.className = 'step-tab';
          page.className = 'page-section';
        }
      }

      if (currentStep === 2) {
        renderStep2();
      } else if (currentStep === 3) {
        executeInference();
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    // Step 1: Catalog Filters & Rendering
    function filterCatalog(category) {
      activeFilter = category;
      ['all', 'eq', 'anth', 'ood'].forEach(k => {
        const btn = document.getElementById(`filter-${k}`);
        if (btn) btn.className = 'ctrl-btn';
      });

      if (category === 'all') document.getElementById('filter-all').className = 'ctrl-btn active';
      else if (category === 'Earthquake') document.getElementById('filter-eq').className = 'ctrl-btn active';
      else if (category === 'Quarry Blast') document.getElementById('filter-anth').className = 'ctrl-btn active';
      else if (category === 'Out-of-Distribution') document.getElementById('filter-ood').className = 'ctrl-btn active';

      applyCatalogFilter();
    }

    function onSearchInput(e) {
      searchQuery = e.target.value.toLowerCase().trim();
      applyCatalogFilter();
    }

    function applyCatalogFilter() {
      filteredCatalog = catalog.filter(item => {
        const matchCat = (activeFilter === 'all') || (item.category === activeFilter);
        const matchQuery = !searchQuery ||
          item.title.toLowerCase().includes(searchQuery) ||
          item.description.toLowerCase().includes(searchQuery) ||
          item.station_names.some(st => st.toLowerCase().includes(searchQuery));
        return matchCat && matchQuery;
      });
      renderCatalog();
    }

    function switchCatalogMode(mode) {
      const btnPresets = document.getElementById('btn-mode-presets');
      const btnUpload = document.getElementById('btn-mode-upload');
      const secPresets = document.getElementById('section-presets');
      const secUpload = document.getElementById('section-upload');

      if (mode === 'presets') {
        btnPresets.className = 'pill-btn active';
        btnUpload.className = 'pill-btn';
        secPresets.style.display = 'flex';
        secUpload.style.display = 'none';
      } else {
        btnPresets.className = 'pill-btn';
        btnUpload.className = 'pill-btn active';
        secPresets.style.display = 'none';
        secUpload.style.display = 'flex';
      }
    }

    function renderCatalog() {
      const grid = document.getElementById('catalog-grid');
      grid.innerHTML = '';

      if (filteredCatalog.length === 0) {
        grid.innerHTML = '<div style="padding: 32px; color: var(--text-dim); text-align: center; grid-column: 1/-1;">No scenarios match filter criteria.</div>';
        return;
      }

      filteredCatalog.forEach(item => {
        const card = document.createElement('div');
        const isSelected = currentEvent && currentEvent.id === item.id;
        card.className = 'event-card' + (isSelected ? ' active' : '');
        card.onclick = () => selectEvent(item.id);

        let catClass = 'eq';
        if (item.category === 'Quarry Blast') catClass = 'anth';
        else if (item.category === 'Out-of-Distribution') catClass = 'ood';

        card.innerHTML = `
          <div class="event-card-head">
            <span class="event-card-title">${item.title}</span>
            <span class="cat-tag ${catClass}">${item.category}</span>
          </div>
          <div class="event-card-desc">${item.description}</div>
          <div class="event-card-meta">&bull; ${item.num_stations} Stations: ${item.station_names.join(', ')}</div>
        `;
        grid.appendChild(card);
      });
    }

    async function selectEvent(id) {
      try {
        const res = await fetch(`/api/sample/${id}`);
        currentEvent = await res.json();
        selectedStationIndex = 0;
        excludedStationIndices.clear();
        resetZoom();
        renderCatalog();
        updateSelectedHeader();
      } catch (err) {
        console.error('Error loading sample', err);
      }
    }

    function updateSelectedHeader() {
      if (!currentEvent) return;
      document.getElementById('selected-title').textContent = currentEvent.title;
      document.getElementById('selected-meta').textContent = `${currentEvent.category} \u2022 ${currentEvent.stations.length} Station(s) \u2022 60.0s @ 50 Hz (3,000 samples)`;
    }

    // Step 1: File Upload
    async function onFileUploaded(e) {
      const file = e.target.files[0];
      if (!file) return;

      const ext = file.name.split('.').pop().toLowerCase();
      const statusEl = document.getElementById('upload-status');
      statusEl.style.display = 'block';
      statusEl.innerHTML = `Reading <b>${file.name}</b>...`;

      const reader = new FileReader();
      reader.onload = async (event) => {
        try {
          const res = await fetch('/api/upload_event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              file_name: file.name,
              file_type: ext,
              content: event.target.result,
            }),
          });
          if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Upload failed');
          }
          currentEvent = await res.json();
          selectedStationIndex = 0;
          excludedStationIndices.clear();
          resetZoom();
          statusEl.innerHTML = `<span style="color: var(--eq); font-weight: 700;">&#10004; Parsed ${file.name}</span> (${currentEvent.stations.length} station(s), 3 components, 3,000 samples @ 50 Hz)`;
          updateSelectedHeader();
        } catch (err) {
          statusEl.innerHTML = `<span style="color: var(--red); font-weight: 700;">&#10008; Error:</span> ${err.message}`;
        }
      };
      reader.readAsText(file);
    }

    // Step 2: Render Waveforms & Stations
    function renderStep2() {
      if (!currentEvent) return;
      document.getElementById('step2-badge').textContent = `Event: ${currentEvent.title}`;
      document.getElementById('event-desc-box').textContent = currentEvent.description || 'Preprocessed 3-component seismic waveform.';

      const tabs = document.getElementById('stn-tabs');
      tabs.innerHTML = '';
      currentEvent.stations.forEach((st, idx) => {
        const btn = document.createElement('button');
        btn.className = 'pill-btn' + (idx === selectedStationIndex ? ' active' : '');
        btn.textContent = `${st.id} (${st.dist_km} km)`;
        btn.onclick = () => {
          selectedStationIndex = idx;
          renderStep2WaveformOnly();
        };
        tabs.appendChild(btn);
      });

      renderStationList();
      handleCanvasResize();
    }

    function renderStep2WaveformOnly() {
      const tabs = document.getElementById('stn-tabs').children;
      for (let i = 0; i < tabs.length; i++) {
        tabs[i].className = 'pill-btn' + (i === selectedStationIndex ? ' active' : '');
      }
      const rows = document.getElementById('station-list').children;
      for (let i = 0; i < rows.length; i++) {
        if (i === selectedStationIndex) rows[i].classList.add('selected');
        else rows[i].classList.remove('selected');
      }
      drawWaveform();
    }

    function renderStationList() {
      const container = document.getElementById('station-list');
      container.innerHTML = '';

      let activeCount = 0;
      currentEvent.stations.forEach((st, idx) => {
        const isIncluded = !excludedStationIndices.has(idx);
        if (isIncluded) activeCount++;

        const row = document.createElement('div');
        row.className = 'station-row' + (idx === selectedStationIndex ? ' selected' : '');

        row.innerHTML = `
          <div class="station-info" onclick="selectStationForWaveform(${idx})">
            <span class="station-code">${st.id} &mdash; ${st.name}</span>
            <span class="station-dist">Distance: ${st.dist_km} km &bull; Coord: ${st.lat}\u00b0N, ${Math.abs(st.lon)}\u00b0W</span>
          </div>
          <input type="checkbox" ${isIncluded ? 'checked' : ''} style="accent-color: var(--accent); cursor: pointer; width: 16px; height: 16px;" onclick="toggleStationActive(event, ${idx})">
        `;
        container.appendChild(row);
      });

      document.getElementById('active-stn-badge').textContent = `${activeCount} / ${currentEvent.stations.length} Active in Graph`;
    }

    function selectStationForWaveform(idx) {
      selectedStationIndex = idx;
      renderStep2WaveformOnly();
    }

    function toggleStationActive(e, idx) {
      e.stopPropagation();
      if (e.target.checked) {
        excludedStationIndices.delete(idx);
      } else {
        if (currentEvent.stations.length - excludedStationIndices.size <= 1) {
          alert('At least 1 station must remain active in the spatial graph.');
          e.target.checked = true;
          return;
        }
        excludedStationIndices.add(idx);
      }
      const activeCount = currentEvent.stations.length - excludedStationIndices.size;
      document.getElementById('active-stn-badge').textContent = `${activeCount} / ${currentEvent.stations.length} Active in Graph`;
    }

    // Interactive Waveform Tools
    function toggleChannel(compIdx) {
      visibleComponents[compIdx] = !visibleComponents[compIdx];
      const btn = document.getElementById(`btn-comp-${compIdx}`);
      btn.className = 'ctrl-btn' + (visibleComponents[compIdx] ? ' active' : '');
      drawWaveform();
    }

    function setGain(gain) {
      waveformGain = gain;
      [1.0, 2.0, 4.0].forEach(g => {
        const btn = document.getElementById(`gain-${g}`);
        if (btn) btn.className = 'ctrl-btn' + (g === gain ? ' active' : '');
      });
      drawWaveform();
    }

    function resetZoom() {
      zoomStart = 0.0;
      zoomEnd = 60.0;
      const el = document.getElementById('zoom-indicator');
      if (el) el.textContent = 'Window: 0.0s \u2013 60.0s';
      drawWaveform();
    }

    // Canvas Mouse Interactions
    function setupCanvasListeners() {
      const canvas = document.getElementById('waveform-canvas');

      canvas.addEventListener('mousemove', (e) => {
        if (!currentEvent) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const width = rect.width;

        const normX = Math.max(0, Math.min(1, mouseX / width));
        const curSec = zoomStart + normX * (zoomEnd - zoomStart);
        document.getElementById('hud-time').textContent = `${curSec.toFixed(2)}s`;

        const stnWaveforms = currentEvent.waveforms[selectedStationIndex];
        if (stnWaveforms) {
          const totalPts = stnWaveforms[0].length;
          const sampleIdx = Math.min(totalPts - 1, Math.max(0, Math.floor((curSec / 60.0) * totalPts)));
          document.getElementById('hud-e').textContent = stnWaveforms[0][sampleIdx] !== undefined ? stnWaveforms[0][sampleIdx].toFixed(3) : '--';
          document.getElementById('hud-n').textContent = stnWaveforms[1][sampleIdx] !== undefined ? stnWaveforms[1][sampleIdx].toFixed(3) : '--';
          document.getElementById('hud-z').textContent = stnWaveforms[2][sampleIdx] !== undefined ? stnWaveforms[2][sampleIdx].toFixed(3) : '--';
        }

        if (isDragging) {
          dragCurrX = mouseX;
        }
        drawWaveform(mouseX);
      });

      canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        isDragging = true;
        dragStartX = e.clientX - rect.left;
        dragCurrX = dragStartX;
      });

      window.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        const canvas = document.getElementById('waveform-canvas');
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;

        const x1 = Math.min(dragStartX, dragCurrX);
        const x2 = Math.max(dragStartX, dragCurrX);

        if (x2 - x1 > 20) {
          const newStart = zoomStart + (x1 / width) * (zoomEnd - zoomStart);
          const newEnd = zoomStart + (x2 / width) * (zoomEnd - zoomStart);

          if (newEnd - newStart >= 1.0) {
            zoomStart = Math.max(0.0, newStart);
            zoomEnd = Math.min(60.0, newEnd);
            document.getElementById('zoom-indicator').textContent = `Zoom: ${zoomStart.toFixed(1)}s \u2013 ${zoomEnd.toFixed(1)}s`;
          }
        }
        drawWaveform();
      });

      canvas.addEventListener('mouseleave', () => {
        if (!isDragging) drawWaveform();
      });
    }

    function handleCanvasResize() {
      const canvas = document.getElementById('waveform-canvas');
      if (!canvas || canvas.parentElement.clientWidth === 0) return;
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.parentElement.clientWidth;
      const height = 390;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      drawWaveform();
    }

    function drawWaveform(crosshairX = null) {
      const canvas = document.getElementById('waveform-canvas');
      if (!canvas || !currentEvent || canvas.width === 0) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Warm Light Seismograph Paper Background
      ctx.fillStyle = '#fcfbf8';
      ctx.fillRect(0, 0, width, height);

      const spanSec = zoomEnd - zoomStart;

      // Time Grid Lines
      ctx.strokeStyle = '#ece7dd';
      ctx.lineWidth = 1;
      const stepSec = spanSec <= 5 ? 0.5 : spanSec <= 15 ? 1.0 : spanSec <= 30 ? 2.0 : 5.0;
      const firstTick = Math.ceil(zoomStart / stepSec) * stepSec;

      for (let sec = firstTick; sec <= zoomEnd; sec += stepSec) {
        const x = ((sec - zoomStart) / spanSec) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.fillStyle = '#71717a';
        ctx.font = '500 10.5px "JetBrains Mono", monospace';
        ctx.fillText(`${sec.toFixed(1)}s`, x + 4, height - 8);
      }

      // 3 Waveform Channels (E, N, Z) in high-contrast light-mode ink colors
      const colors = ['#0284c7', '#059669', '#d94e28'];
      const labels = ['E (East-West)', 'N (North-South)', 'Z (Vertical)'];
      const bandH = height / 3;

      const stnWaveforms = currentEvent.waveforms[selectedStationIndex];
      if (stnWaveforms) {
        stnWaveforms.forEach((compData, cIdx) => {
          if (!visibleComponents[cIdx]) return;
          const centerY = bandH * cIdx + bandH / 2;

          ctx.strokeStyle = '#e6e0d5';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(0, centerY);
          ctx.lineTo(width, centerY);
          ctx.stroke();

          ctx.fillStyle = colors[cIdx];
          ctx.font = '700 10.5px "JetBrains Mono", monospace';
          ctx.fillText(labels[cIdx], 10, bandH * cIdx + 18);

          ctx.strokeStyle = colors[cIdx];
          ctx.lineWidth = 1.35;
          ctx.beginPath();

          const nPts = compData.length;
          const startIdx = Math.floor((zoomStart / 60.0) * nPts);
          const endIdx = Math.ceil((zoomEnd / 60.0) * nPts);
          const ptsInView = Math.max(2, endIdx - startIdx);

          let started = false;
          for (let i = startIdx; i <= endIdx && i < nPts; i++) {
            const x = ((i - startIdx) / (ptsInView - 1)) * width;
            const y = centerY - (compData[i] * waveformGain) * (bandH * 0.38);
            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        });
      }

      // P-Wave Arrival (t = 10.0s)
      if (10.0 >= zoomStart && 10.0 <= zoomEnd) {
        const pX = ((10.0 - zoomStart) / spanSec) * width;
        ctx.strokeStyle = '#dc2626';
        ctx.lineWidth = 1.6;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(pX, 0);
        ctx.lineTo(pX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#dc2626';
        ctx.font = '700 10.5px "JetBrains Mono", monospace';
        ctx.fillText('P (10s)', pX + 5, 20);
      }

      // Drag Zoom Box
      if (isDragging) {
        const x1 = Math.min(dragStartX, dragCurrX);
        const boxW = Math.abs(dragCurrX - dragStartX);
        ctx.fillStyle = 'rgba(217, 78, 40, 0.12)';
        ctx.fillRect(x1, 0, boxW, height);
        ctx.strokeStyle = '#d94e28';
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x1, 0, boxW, height);
      }

      // Hover Crosshair
      if (crosshairX !== null && crosshairX >= 0 && crosshairX <= width) {
        ctx.strokeStyle = 'rgba(24, 24, 27, 0.35)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(crosshairX, 0);
        ctx.lineTo(crosshairX, height);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      ctx.restore();
    }

    // Step 3: Run Model Classification
    async function executeInference() {
      if (!currentEvent) return;
      document.getElementById('verdict-label').textContent = 'RUNNING INFERENCE...';
      document.getElementById('verdict-sub').textContent = 'Executing PyTorch GATv2-EDL forward pass...';

      const noise = parseFloat(document.getElementById('noise-slider').value);
      const activeStations = [];
      currentEvent.stations.forEach((_, idx) => {
        if (!excludedStationIndices.has(idx)) activeStations.push(idx);
      });

      try {
        const res = await fetch('/api/classify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_id: currentEvent.id,
            noise_level: noise,
            active_stations: activeStations,
          }),
        });
        const out = await res.json();
        latestInferenceResult = out;
        renderStep3Results(out);
      } catch (err) {
        console.error('Inference error', err);
        document.getElementById('verdict-label').textContent = 'INFERENCE ERROR';
      }
    }

    function renderStep3Results(out) {
      document.getElementById('step3-latency').textContent = `Latency: ${out.latency_ms} ms (${out.device.toUpperCase()})`;

      const card = document.getElementById('verdict-card');
      const label = document.getElementById('verdict-label');
      const sub = document.getElementById('verdict-sub');

      card.className = 'verdict-box';
      const isOOD = out.ground_truth === 'out-of-distribution';
      const u = out.evidential.vacuity_uncertainty;

      if (isOOD || (u >= 0.45 && out.confidence < 75.0)) {
        card.classList.add('ood');
        label.textContent = 'OUT-OF-DISTRIBUTION ANOMALY';
        sub.innerHTML = `Model detected unfamiliar source mechanics. Automated classification suspended (u = <b>${u.toFixed(4)}</b>).`;
      } else if (out.predicted_class === 'earthquake') {
        card.classList.add('eq');
        label.textContent = 'NATURAL TECTONIC EARTHQUAKE';
        sub.innerHTML = `Confidence: <b>${out.confidence}%</b> &bull; Double-couple shear fault dislocation.`;
      } else {
        card.classList.add('anth');
        label.textContent = 'ANTHROPOGENIC QUARRY BLAST';
        sub.innerHTML = `Confidence: <b>${out.confidence}%</b> &bull; Isotropic compressional detonation with strong surface waves.`;
      }

      // Probabilities
      const pEq = (out.probabilities.earthquake * 100).toFixed(1);
      const pAnth = (out.probabilities.anthropogenic * 100).toFixed(1);

      document.getElementById('prob-eq-val').textContent = `${pEq}%`;
      document.getElementById('prob-anth-val').textContent = `${pAnth}%`;
      document.getElementById('prob-eq-bar').style.width = `${pEq}%`;
      document.getElementById('prob-anth-bar').style.width = `${pAnth}%`;

      // Vacuity Uncertainty
      document.getElementById('vacuity-num').textContent = u.toFixed(4);
      document.getElementById('vacuity-pin').style.left = `${Math.min(100, Math.max(0, u * 100))}%`;

      const statusEl = document.getElementById('vacuity-status-text');
      if (u < 0.30) {
        statusEl.innerHTML = `<span style="color: var(--eq); font-weight: 700;">Low Vacuity (u = ${u.toFixed(3)})</span> &mdash; High Dirichlet evidence (S = ${out.evidential.strength.toFixed(2)}). Prediction is within trained in-distribution envelope.`;
      } else if (u < 0.45) {
        statusEl.innerHTML = `<span style="color: var(--anth); font-weight: 700;">Moderate Vacuity (u = ${u.toFixed(3)})</span> &mdash; Signal partially degraded by noise or reduced station geometry.`;
      } else {
        statusEl.innerHTML = `<span style="color: var(--red); font-weight: 700;">High Epistemic Vacuity (u = ${u.toFixed(3)})</span> &mdash; Evidential safety guard triggered. Waveform does not match known tectonic or quarry templates. Flagged for seismologist audit.`;
      }

      // Subjective Logic Table
      document.getElementById('diag-evidence').textContent = `e_eq: ${out.evidential.evidence.earthquake.toFixed(2)} | e_anth: ${out.evidential.evidence.anthropogenic.toFixed(2)}`;
      document.getElementById('diag-alpha').textContent = `[${out.evidential.alpha[0].toFixed(2)}, ${out.evidential.alpha[1].toFixed(2)}]`;
      document.getElementById('diag-strength').textContent = out.evidential.strength.toFixed(2);
      document.getElementById('diag-belief').textContent = `b_eq: ${(out.evidential.belief.earthquake * 100).toFixed(1)}% | b_anth: ${(out.evidential.belief.anthropogenic * 100).toFixed(1)}%`;

      // GATv2 Attention Matrix
      renderAttentionHeatmap(out.attention_matrix, out.stations_used);
    }

    function renderAttentionHeatmap(matrix, stations) {
      const box = document.getElementById('attn-matrix-box');
      box.innerHTML = '';
      if (!matrix || matrix.length === 0) return;

      const n = matrix.length;
      box.style.gridTemplateColumns = `repeat(${n}, 1fr)`;

      for (let r = 0; r < n; r++) {
        for (let c = 0; c < n; c++) {
          const val = matrix[r][c];
          const cell = document.createElement('div');
          cell.className = 'attn-cell';
          const intensity = Math.min(1.0, val * 2.2);
          cell.style.backgroundColor = `rgba(217, 78, 40, ${Math.max(0.08, intensity)})`;
          cell.style.color = intensity > 0.55 ? '#ffffff' : '#18181b';
          cell.textContent = val.toFixed(2);
          cell.title = `Receiver ${stations[r]} \u2190 Sender ${stations[c]}: Attention Weight ${val}`;
          box.appendChild(cell);
        }
      }
    }

    function exportAuditJSON() {
      if (!latestInferenceResult) {
        alert('No inference result to export.');
        return;
      }
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(latestInferenceResult, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute('href', dataStr);
      downloadAnchor.setAttribute('download', `seismo_audit_${latestInferenceResult.event_id}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    }

    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""
@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the streamlined, polished frontend."""
    return HTML_PAGE


def main():
    print("=" * 60)
    print("Starting Seismo Model Showcase Server (Polished Multi-Step UI)...")
    print("Available at: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("seismic_ml.ui.app:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
