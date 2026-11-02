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

    p_eq = float(probs[0])
    p_anth = float(probs[1])
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
            "evidence": {"earthquake": round(float(evidence[0]), 4), "anthropogenic": round(float(evidence[1]), 4)},
            "alpha": [round(float(alpha[0]), 4), round(float(alpha[1]), 4)],
            "strength": round(strength, 4),
            "vacuity_uncertainty": round(uncertainty, 4),
            "belief": {"earthquake": round(float(belief[0]), 4), "anthropogenic": round(float(belief[1]), 4)},
        },
        "stations_used": [st["id"] for st in stations_filtered],
        "attention_matrix": attn_matrix,
        "latency_ms": round(latency_ms, 2),
        "device": device,
    }


# Polished, Minimalistic UI
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SEISMO — Multi-Station Seismic Classifier</title>
  <style>
    :root {
      --bg: #090d16;
      --card: #0f172a;
      --card-inner: #0b1120;
      --border: #1e293b;
      --border-light: #334155;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --eq: #10b981;
      --eq-bg: rgba(16, 185, 129, 0.12);
      --anth: #f59e0b;
      --anth-bg: rgba(245, 158, 11, 0.12);
      --ood: #a855f7;
      --ood-bg: rgba(168, 85, 247, 0.12);
      --cyan: #06b6d4;
      --blue: #3b82f6;
      --red: #ef4444;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-sans);
      font-size: 13px;
      line-height: 1.5;
      padding: 16px;
      -webkit-font-smoothing: antialiased;
    }
    .container { max-width: 1280px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }

    /* Header */
    header {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 18px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .brand { display: flex; align-items: baseline; gap: 10px; }
    .brand h1 { font-size: 16px; font-weight: 800; letter-spacing: 0.5px; color: #fff; }
    .brand .tag {
      font-family: var(--font-mono);
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
      background: #1e293b;
      color: var(--cyan);
      border: 1px solid var(--border-light);
    }
    .meta-bar {
      display: flex;
      align-items: center;
      gap: 16px;
      font-family: var(--font-mono);
      font-size: 11px;
      color: var(--text-muted);
    }
    .status-dot {
      display: inline-block;
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--eq);
      margin-right: 5px;
    }

    /* Sequential Step Bar */
    .step-bar {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .step-tab {
      padding: 12px 16px;
      display: flex;
      align-items: center;
      gap: 10px;
      cursor: pointer;
      background: var(--card-inner);
      color: var(--text-muted);
      border-right: 1px solid var(--border);
      transition: all 0.15s;
      user-select: none;
    }
    .step-tab:last-child { border-right: none; }
    .step-tab:hover { background: #131c2e; color: var(--text); }
    .step-tab.active {
      background: #172338;
      color: #fff;
      font-weight: 600;
      border-bottom: 2px solid var(--cyan);
    }
    .step-tab.completed .step-badge { background: var(--eq); color: #000; }
    .step-badge {
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: #1e293b;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 700;
    }
    .step-tab.active .step-badge { background: var(--cyan); color: #000; }

    /* Page Sections */
    .page-section { display: none; }
    .page-section.active { display: flex; flex-direction: column; gap: 14px; }

    /* Unified Card */
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }

    /* Sub Tabs & Filter Buttons */
    .pill-group { display: flex; gap: 6px; flex-wrap: wrap; }
    .pill-btn {
      background: var(--card-inner);
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 5px 12px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 11.5px;
      font-weight: 500;
      transition: all 0.15s;
    }
    .pill-btn:hover { border-color: var(--border-light); color: var(--text); }
    .pill-btn.active {
      background: #1e293b;
      color: var(--cyan);
      border-color: var(--cyan);
      font-weight: 600;
    }

    /* Step 1 Search & Catalog Grid */
    .search-box {
      background: var(--card-inner);
      border: 1px solid var(--border);
      color: #fff;
      padding: 6px 12px;
      border-radius: 5px;
      font-size: 12px;
      width: 280px;
      outline: none;
    }
    .search-box:focus { border-color: var(--cyan); }

    .catalog-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
      gap: 10px;
      max-height: 480px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .catalog-grid::-webkit-scrollbar { width: 5px; }
    .catalog-grid::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }

    .event-card {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
      cursor: pointer;
      transition: all 0.15s;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .event-card:hover { border-color: var(--border-light); background: #111a2d; }
    .event-card.active {
      border-color: var(--cyan);
      background: #14243b;
      box-shadow: 0 0 12px rgba(6, 182, 212, 0.12);
    }
    .event-card-head { display: flex; justify-content: space-between; align-items: start; gap: 8px; }
    .event-card-title { font-size: 12.5px; font-weight: 700; color: #fff; line-height: 1.3; }
    .cat-tag {
      font-family: var(--font-mono);
      font-size: 9.5px;
      padding: 2px 6px;
      border-radius: 3px;
      font-weight: 600;
      white-space: nowrap;
    }
    .cat-tag.eq { background: var(--eq-bg); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .cat-tag.anth { background: var(--anth-bg); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .cat-tag.ood { background: var(--ood-bg); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }

    .event-card-desc { font-size: 11.5px; color: var(--text-muted); line-height: 1.35; }
    .event-card-meta { font-family: var(--font-mono); font-size: 10.5px; color: var(--text-dim); }

    /* Upload Zone */
    .dropzone {
      border: 2px dashed var(--border-light);
      border-radius: 6px;
      padding: 32px;
      text-align: center;
      background: var(--card-inner);
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      transition: border-color 0.15s;
    }
    .dropzone:hover { border-color: var(--cyan); }
    .dropzone input { display: none; }

    /* Primary Action Buttons */
    .btn-primary {
      background: #0284c7;
      color: #fff;
      border: 1px solid #38bdf8;
      border-radius: 5px;
      padding: 9px 20px;
      font-weight: 600;
      font-size: 12.5px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.15s;
    }
    .btn-primary:hover { background: #0369a1; }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

    .btn-secondary {
      background: var(--card-inner);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 9px 16px;
      font-weight: 500;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.15s;
    }
    .btn-secondary:hover { background: #1e293b; }

    .nav-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }

    /* Step 2 Layout */
    .step2-layout {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 16px;
    }
    @media (max-width: 1024px) {
      .step2-layout { grid-template-columns: 1fr; }
    }

    /* Waveform Canvas Toolbar */
    .canvas-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      background: #070b13;
      border: 1px solid var(--border);
      border-bottom: none;
      border-radius: 6px 6px 0 0;
      padding: 8px 12px;
      gap: 8px;
    }
    .canvas-group { display: flex; align-items: center; gap: 6px; font-size: 11px; }

    .ctrl-btn {
      background: #131c2e;
      border: 1px solid var(--border);
      color: var(--text-muted);
      padding: 3px 8px;
      border-radius: 4px;
      cursor: pointer;
      font-family: var(--font-mono);
      font-size: 11px;
    }
    .ctrl-btn.active { background: #0284c7; color: #fff; border-color: #38bdf8; }
    .ctrl-btn:hover { color: #fff; }

    .canvas-box {
      position: relative;
      background: #05080f;
      border: 1px solid var(--border);
      border-radius: 0 0 6px 6px;
      overflow: hidden;
      cursor: crosshair;
    }
    canvas#waveform-canvas { display: block; width: 100%; height: 380px; }

    /* Canvas HUD */
    .canvas-hud {
      position: absolute;
      top: 8px;
      right: 12px;
      background: rgba(11, 17, 32, 0.9);
      border: 1px solid var(--border);
      border-radius: 4px;
      padding: 5px 10px;
      font-family: var(--font-mono);
      font-size: 11px;
      display: flex;
      gap: 12px;
      pointer-events: none;
    }

    /* Step 2 Right Pane: Station Array Table */
    .station-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 260px;
      overflow-y: auto;
    }
    .station-row {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 5px;
      padding: 8px 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11.5px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .station-row:hover { background: #131c2e; }
    .station-row.selected {
      border-color: var(--cyan);
      background: #14243b;
    }
    .station-info { display: flex; flex-direction: column; gap: 2px; }
    .station-code { font-family: var(--font-mono); font-weight: 700; color: #fff; }
    .station-dist { font-size: 10.5px; color: var(--text-dim); }

    /* Step 3 Grid */
    .step3-layout {
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      .step3-layout { grid-template-columns: 1fr; }
    }

    .verdict-box {
      background: var(--card-inner);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .verdict-label {
      font-size: 21px;
      font-weight: 800;
      letter-spacing: 0.5px;
      font-family: var(--font-mono);
      text-transform: uppercase;
    }
    .verdict-box.eq .verdict-label { color: var(--eq); }
    .verdict-box.anth .verdict-label { color: var(--anth); }
    .verdict-box.ood .verdict-label { color: var(--ood); }

    .prob-bar-track { height: 7px; background: #1c2738; border-radius: 4px; overflow: hidden; margin-top: 4px; }
    .prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    .prob-bar-fill.eq { background: var(--eq); }
    .prob-bar-fill.anth { background: var(--anth); }

    .gauge-track {
      height: 8px;
      border-radius: 4px;
      background: linear-gradient(to right, #10b981 0%, #f59e0b 50%, #ef4444 100%);
      position: relative;
      margin-top: 4px;
    }
    .gauge-pin {
      position: absolute;
      top: -2px;
      width: 4px;
      height: 12px;
      background: #fff;
      border-radius: 2px;
      box-shadow: 0 0 4px #000;
      transform: translateX(-50%);
      transition: left 0.3s ease;
    }

    .diag-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 11px; }
    .diag-table th, .diag-table td { padding: 6px 8px; border-bottom: 1px solid #1a2436; text-align: left; }
    .diag-table th { color: var(--text-dim); }
    .diag-table td.num { text-align: right; font-weight: 600; color: #fff; }

    .attn-matrix { display: grid; gap: 3px; }
    .attn-cell {
      height: 30px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: var(--font-mono);
      font-size: 10.5px;
      font-weight: 600;
      color: #fff;
      border-radius: 3px;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <h1>SEISMO</h1>
        <span class="tag">GATv2 + EDL Seismic Classifier</span>
      </div>
      <div class="meta-bar">
        <span><span class="status-dot"></span>Model: <b id="hdr-model">Ready</b></span>
        <span>Params: <b id="hdr-params">--</b></span>
        <span>Device: <b id="hdr-device">--</b></span>
      </div>
    </header>

    <!-- Step Navigation Bar -->
    <div class="step-bar">
      <div class="step-tab active" id="tab-step-1" onclick="goToStep(1)">
        <div class="step-badge">1</div>
        <div>1. Select Waveform Scenario</div>
      </div>
      <div class="step-tab" id="tab-step-2" onclick="goToStep(2)">
        <div class="step-badge">2</div>
        <div>2. Waveforms &amp; Network Setup</div>
      </div>
      <div class="step-tab" id="tab-step-3" onclick="goToStep(3)">
        <div class="step-badge">3</div>
        <div>3. Evidential Classification</div>
      </div>
    </div>

    <!-- ==================== STEP 1: INPUT SELECTION ==================== -->
    <section class="page-section active" id="page-step-1">
      <div class="card">
        <div class="card-head">
          <div class="card-title">Step 1 &mdash; Select Input Waveform Scenario</div>
          <span style="font-size: 11px; color: var(--cyan); font-family: var(--font-mono);" id="catalog-count-badge">Catalog: 22 Events</span>
        </div>

        <!-- Mode Toggle -->
        <div class="pill-group">
          <button class="pill-btn active" id="btn-mode-presets" onclick="switchCatalogMode('presets')">Curated Seismic Events (22)</button>
          <button class="pill-btn" id="btn-mode-upload" onclick="switchCatalogMode('upload')">Upload Custom File (.CSV / .JSON)</button>
        </div>

        <!-- Mode Presets -->
        <div id="section-presets" style="display: flex; flex-direction: column; gap: 10px;">
          <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; background: var(--card-inner); padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border);">
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
        <div id="section-upload" style="display: none; flex-direction: column; gap: 12px;">
          <div class="dropzone" onclick="document.getElementById('file-picker').click()">
            <div style="font-size: 26px; color: var(--cyan);">&#8682;</div>
            <div style="font-weight: 600; font-size: 13px;">Click to Browse or Drag &amp; Drop Waveform File</div>
            <div style="font-size: 11px; color: var(--text-muted);">
              Supports <b>.csv</b> (columns: E, N, Z or time, E, N, Z) or <b>.json</b> (multi-station event)
            </div>
            <input type="file" id="file-picker" accept=".json,.csv" onchange="onFileUploaded(event)">
          </div>

          <div id="upload-status" style="display: none; background: var(--card-inner); padding: 10px; border-radius: 4px; border: 1px solid var(--border-light); font-family: var(--font-mono); font-size: 11.5px;"></div>

          <div style="display: flex; gap: 12px; font-size: 11px; color: var(--text-muted); align-items: center;">
            <span>Download Format Templates:</span>
            <a href="/api/template/csv" download="template_waveform.csv" style="color: var(--cyan); text-decoration: none;">&darr; Sample CSV</a>
            <span>&bull;</span>
            <a href="/api/template/json" download="template_waveform.json" style="color: var(--cyan); text-decoration: none;">&darr; Sample JSON</a>
          </div>
        </div>

        <!-- Summary & Next Button -->
        <div class="nav-footer">
          <div>
            <div style="font-size: 10.5px; color: var(--text-dim); text-transform: uppercase;">Selected Event:</div>
            <div style="font-size: 13.5px; font-weight: 700; color: #fff;" id="selected-title">Loading...</div>
            <div style="font-size: 11px; color: var(--cyan); font-family: var(--font-mono);" id="selected-meta"></div>
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
          <span style="font-size: 11px; color: var(--cyan); font-family: var(--font-mono);" id="step2-badge"></span>
        </div>

        <div class="step2-layout">
          <!-- Left: Waveform Canvas -->
          <div style="display: flex; flex-direction: column;">
            <!-- Station Tabs -->
            <div class="pill-group" id="stn-tabs" style="margin-bottom: 8px;"></div>

            <!-- Canvas Toolbar -->
            <div class="canvas-bar">
              <div class="canvas-group">
                <span style="color: var(--text-muted);">Channels:</span>
                <button class="ctrl-btn active" id="btn-comp-0" onclick="toggleChannel(0)">E (East)</button>
                <button class="ctrl-btn active" id="btn-comp-1" onclick="toggleChannel(1)">N (North)</button>
                <button class="ctrl-btn active" id="btn-comp-2" onclick="toggleChannel(2)">Z (Vert)</button>
              </div>

              <div class="canvas-group">
                <span style="color: var(--text-muted);">Gain:</span>
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
              <canvas id="waveform-canvas" width="800" height="380"></canvas>
              <div class="canvas-hud">
                <div>Time: <span id="hud-time" style="color: #fff; font-weight: 700;">--</span></div>
                <div>E: <span id="hud-e" style="color: var(--cyan); font-weight: 700;">--</span></div>
                <div>N: <span id="hud-n" style="color: var(--blue); font-weight: 700;">--</span></div>
                <div>Z: <span id="hud-z" style="color: #fbbf24; font-weight: 700;">--</span></div>
              </div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
              <span style="font-size: 10px; color: var(--text-dim); font-family: var(--font-mono);">
                Drag horizontally to zoom. Hover to inspect sample amplitudes. Red line marks P-arrival (10.0s).
              </span>
              <span style="font-size: 10.5px; color: var(--cyan); font-family: var(--font-mono);" id="zoom-indicator">Window: 0.0s &ndash; 60.0s</span>
            </div>

            <div style="font-size: 11.5px; color: var(--text-muted); background: var(--card-inner); padding: 10px; border-radius: 5px; border-left: 3px solid var(--cyan); margin-top: 10px;" id="event-desc-box"></div>
          </div>

          <!-- Right: Station Array & Robustness -->
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <!-- Station Array Panel -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 11px; font-weight: 700; text-transform: uppercase; color: var(--text-muted);">Station Array Network</span>
                <span style="font-size: 10px; font-family: var(--font-mono); color: var(--cyan);" id="active-stn-badge">-- Active</span>
              </div>
              <div class="station-list" id="station-list"></div>
              <span style="font-size: 10px; color: var(--text-dim);">
                Click a station row to view its waveform. Uncheck the checkbox to simulate sensor telemetry dropout.
              </span>
            </div>

            <!-- Robustness: Gaussian Noise Slider -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 8px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px;">
                <span style="font-weight: 700; text-transform: uppercase; color: var(--text-muted);">Injected Gaussian Noise</span>
                <span id="noise-val" style="font-family: var(--font-mono); color: var(--cyan); font-weight: 700;">&sigma; = 0.00</span>
              </div>
              <input type="range" id="noise-slider" min="0" max="0.20" step="0.02" value="0" style="width: 100%; accent-color: var(--cyan);">
              <span style="font-size: 10px; color: var(--text-dim);">
                Injects microseismic noise to evaluate Evidential vacuity degradation.
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
          <div style="display: flex; gap: 8px; align-items: center;">
            <span style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);" id="step3-latency">Latency: -- ms</span>
            <button class="btn-secondary" style="padding: 5px 10px; font-size: 11px;" onclick="executeInference()">&#x21ba; Re-Run</button>
            <button class="btn-secondary" style="padding: 5px 10px; font-size: 11px; color: var(--cyan);" onclick="exportAuditJSON()">&darr; Export Audit JSON</button>
          </div>
        </div>

        <div class="step3-layout">
          <!-- Left: Classification & Uncertainty -->
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <!-- Primary Verdict -->
            <div class="verdict-box" id="verdict-card">
              <div style="font-size: 10px; text-transform: uppercase; color: var(--text-dim);">Classification Verdict</div>
              <div class="verdict-label" id="verdict-label">EVALUATING...</div>
              <div style="font-size: 11.5px; color: var(--text-muted);" id="verdict-sub">Computing GATv2 node embeddings...</div>
            </div>

            <!-- Probabilities -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 10px;">
              <span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--text-dim);">Class Probability Distribution</span>

              <div>
                <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px;">
                  <span>Natural Tectonic Earthquake</span>
                  <span id="prob-eq-val" style="font-weight: 700; color: var(--eq);">--%</span>
                </div>
                <div class="prob-bar-track">
                  <div class="prob-bar-fill eq" id="prob-eq-bar" style="width: 0%;"></div>
                </div>
              </div>

              <div>
                <div style="display: flex; justify-content: space-between; font-family: var(--font-mono); font-size: 11px;">
                  <span>Anthropogenic Event (Quarry Blast)</span>
                  <span id="prob-anth-val" style="font-weight: 700; color: var(--anth);">--%</span>
                </div>
                <div class="prob-bar-track">
                  <div class="prob-bar-fill anth" id="prob-anth-bar" style="width: 0%;"></div>
                </div>
              </div>
            </div>

            <!-- Vacuity Uncertainty Gauge -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--text-dim);">Epistemic Vacuity Uncertainty (u = K / S)</span>
                <span id="vacuity-num" style="font-size: 20px; font-weight: 700; font-family: var(--font-mono);">--</span>
              </div>
              <div class="gauge-track">
                <div class="gauge-pin" id="vacuity-pin" style="left: 0%;"></div>
              </div>
              <div style="display: flex; justify-content: space-between; font-size: 9.5px; font-family: var(--font-mono); color: var(--text-dim);">
                <span>0.0 (High Certainty)</span>
                <span>0.5 (Moderate Risk)</span>
                <span>1.0 (High Vacuity / OOD)</span>
              </div>
              <div style="font-size: 11px; color: var(--text-muted); line-height: 1.4;" id="vacuity-status-text"></div>
            </div>
          </div>

          <!-- Right: Subjective Logic & GAT Heatmap -->
          <div style="display: flex; flex-direction: column; gap: 14px;">
            <!-- Subjective Logic Table -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 6px;">
              <span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--text-dim);">Dirichlet Subjective Logic Parameters</span>
              <table class="diag-table">
                <tbody>
                  <tr><td>Non-negative Evidence (e)</td><td class="num" id="diag-evidence">--</td></tr>
                  <tr><td>Dirichlet Parameters (&alpha; = e + 1)</td><td class="num" id="diag-alpha">--</td></tr>
                  <tr><td>Total Dirichlet Strength (S = &sum; &alpha;)</td><td class="num" id="diag-strength">--</td></tr>
                  <tr><td>Belief Masses (b_k = e_k / S)</td><td class="num" id="diag-belief">--</td></tr>
                  <tr><td>Belief Identity (b_eq + b_anth + u)</td><td class="num" id="diag-identity" style="color: var(--cyan);">1.0000</td></tr>
                </tbody>
              </table>
            </div>

            <!-- GATv2 Attention Matrix -->
            <div style="background: var(--card-inner); border: 1px solid var(--border); border-radius: 6px; padding: 12px; display: flex; flex-direction: column; gap: 8px;">
              <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <span style="font-size: 10.5px; font-weight: 700; text-transform: uppercase; color: var(--text-dim);">GATv2 Spatial Attention Matrix</span>
                <span style="font-size: 9.5px; font-family: var(--font-mono); color: var(--text-dim);">Target &larr; Source (&alpha;<sub>ij</sub>)</span>
              </div>
              <div class="attn-matrix" id="attn-matrix-box"></div>
              <span style="font-size: 10px; color: var(--text-dim);">Inter-station dynamic attention weights computed during graph message passing.</span>
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

  <script>
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
        const tab = document.getElementById(`tab-step-1`.replace('1', s));
        const page = document.getElementById(`page-step-1`.replace('1', s));
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
        grid.innerHTML = '<div style="padding: 24px; color: var(--text-dim); text-align: center; grid-column: 1/-1;">No scenarios match filter criteria.</div>';
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
          statusEl.innerHTML = `<span style="color: var(--eq);">&#10004; Parsed <b>${file.name}</b>!</span> (${currentEvent.stations.length} station(s), 3 components, 3,000 samples @ 50 Hz)`;
          updateSelectedHeader();
        } catch (err) {
          statusEl.innerHTML = `<span style="color: var(--red);">&#10008; Error:</span> ${err.message}`;
        }
      };
      reader.readAsText(file);
    }

    // Step 2: Render Waveforms & Stations
    function renderStep2() {
      if (!currentEvent) return;
      document.getElementById('step2-badge').textContent = `Event: ${currentEvent.title}`;
      document.getElementById('event-desc-box').textContent = currentEvent.description || 'Preprocessed 3-component seismic waveform.';

      // Station tabs for waveform view
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

      // Station array list with persistent checkboxes
      renderStationList();
      handleCanvasResize();
    }

    function renderStep2WaveformOnly() {
      // Update tab active classes
      const tabs = document.getElementById('stn-tabs').children;
      for (let i = 0; i < tabs.length; i++) {
        tabs[i].className = 'pill-btn' + (i === selectedStationIndex ? ' active' : '');
      }
      // Update station list selection
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
          <input type="checkbox" ${isIncluded ? 'checked' : ''} style="accent-color: var(--cyan); cursor: pointer;" onclick="toggleStationActive(event, ${idx})">
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
      document.getElementById('zoom-indicator').textContent = 'Window: 0.0s \u2013 60.0s';
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
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.parentElement.clientWidth;
      const height = 380;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      drawWaveform();
    }

    function drawWaveform(crosshairX = null) {
      const canvas = document.getElementById('waveform-canvas');
      if (!canvas || !currentEvent) return;
      const ctx = canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      ctx.save();
      ctx.scale(dpr, dpr);

      // Background
      ctx.fillStyle = '#05080f';
      ctx.fillRect(0, 0, width, height);

      const spanSec = zoomEnd - zoomStart;

      // Time Grid Lines
      ctx.strokeStyle = '#121a2a';
      ctx.lineWidth = 1;
      const stepSec = spanSec <= 5 ? 0.5 : spanSec <= 15 ? 1.0 : spanSec <= 30 ? 2.0 : 5.0;
      const firstTick = Math.ceil(zoomStart / stepSec) * stepSec;

      for (let sec = firstTick; sec <= zoomEnd; sec += stepSec) {
        const x = ((sec - zoomStart) / spanSec) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();

        ctx.fillStyle = '#475569';
        ctx.font = '10px monospace';
        ctx.fillText(`${sec.toFixed(1)}s`, x + 3, height - 6);
      }

      // 3 Waveform Channels (E, N, Z)
      const colors = ['#06b6d4', '#3b82f6', '#fbbf24'];
      const labels = ['E (East-West)', 'N (North-South)', 'Z (Vertical)'];
      const bandH = height / 3;

      const stnWaveforms = currentEvent.waveforms[selectedStationIndex];
      if (stnWaveforms) {
        stnWaveforms.forEach((compData, cIdx) => {
          if (!visibleComponents[cIdx]) return;
          const centerY = bandH * cIdx + bandH / 2;

          ctx.strokeStyle = '#0e1626';
          ctx.beginPath();
          ctx.moveTo(0, centerY);
          ctx.lineTo(width, centerY);
          ctx.stroke();

          ctx.fillStyle = '#64748b';
          ctx.font = '10px monospace';
          ctx.fillText(labels[cIdx], 8, bandH * cIdx + 14);

          ctx.strokeStyle = colors[cIdx];
          ctx.lineWidth = 1.2;
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
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.moveTo(pX, 0);
        ctx.lineTo(pX, height);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#ef4444';
        ctx.font = '10px monospace';
        ctx.fillText('P (10s)', pX + 4, 18);
      }

      // Drag Zoom Box
      if (isDragging) {
        const x1 = Math.min(dragStartX, dragCurrX);
        const boxW = Math.abs(dragCurrX - dragStartX);
        ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
        ctx.fillRect(x1, 0, boxW, height);
        ctx.strokeStyle = '#06b6d4';
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, 0, boxW, height);
      }

      // Hover Crosshair
      if (crosshairX !== null && crosshairX >= 0 && crosshairX <= width) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
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
        statusEl.innerHTML = `<span style="color: var(--eq); font-weight: 600;">Low Vacuity (u = ${u.toFixed(3)})</span> &mdash; High Dirichlet evidence (S = ${out.evidential.strength.toFixed(2)}). Prediction is within trained in-distribution envelope.`;
      } else if (u < 0.45) {
        statusEl.innerHTML = `<span style="color: var(--anth); font-weight: 600;">Moderate Vacuity (u = ${u.toFixed(3)})</span> &mdash; Signal partially degraded by noise or reduced station geometry.`;
      } else {
        statusEl.innerHTML = `<span style="color: var(--red); font-weight: 600;">High Epistemic Vacuity (u = ${u.toFixed(3)})</span> &mdash; Evidential safety guard triggered. Waveform does not match known tectonic or quarry templates. Flagged for seismologist audit.`;
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
          const intensity = Math.min(1.0, val * 2.5);
          cell.style.backgroundColor = `rgba(6, 182, 212, ${Math.max(0.08, intensity)})`;
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
