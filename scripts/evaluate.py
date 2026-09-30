"""Evaluate a trained model (classification, calibration, uncertainty, OOD).

Placeholder. Classification evaluation is planned for Phase 3; calibration,
uncertainty and OOD evaluation for Phase 9. Running this script only
validates the configuration and exits with status 1.

Usage:
    python scripts/evaluate.py --config configs/config.yaml --checkpoint <path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from seismic_ml.config import load_config
from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.utils.logging import configure_logging

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Entry config file.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint to evaluate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = configure_logging()
    load_config(args.config)  # fail early on a broken configuration
    logger.info("Configuration %s loaded.", args.config)
    logger.error("%s", NotYetImplementedError("scripts/evaluate.py (evaluation pipeline)", phase=3))
    return 1


if __name__ == "__main__":
    sys.exit(main())
