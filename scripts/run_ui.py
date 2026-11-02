"""Launcher script for the Seismo Model Showcase Frontend.

Usage:
    python scripts/run_ui.py [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add src and repo root to pythonpath
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Launch Seismo Model Showcase Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    samples_file = REPO_ROOT / "outputs" / "showcase_samples.json"
    if not samples_file.exists():
        print("[UI] Showcase scenarios file missing. Generating 22 physics-calibrated events...")
        try:
            from scripts.generate_scenarios import generate_all_scenarios
            generate_all_scenarios(samples_file)
        except Exception as e:
            print(f"[UI] Warning: could not auto-generate scenarios: {e}")

    ckpt_file = REPO_ROOT / "outputs" / "checkpoints" / "best_model.pt"
    if not ckpt_file.exists():
        print("[UI] Model checkpoint missing. Training calibrated showcase weights...")
        try:
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "train_showcase.py")], check=True)
        except Exception as e:
            print(f"[UI] Warning: could not auto-train showcase model: {e}")

    print("=" * 65)
    print("SEISMO — Model Showcase Server")
    print(f"Open your browser at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    print("=" * 65)

    uvicorn.run("seismic_ml.ui.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
