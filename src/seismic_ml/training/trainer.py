"""Training loop interface (placeholder, Phase 3).

No training pipeline exists yet. Once implemented, every run must store the
items listed in PROJECT_CONTEXT.md §13 (seed, dataset version, resolved
config, software versions, experiment ID) next to its results.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from seismic_ml.placeholders import NotYetImplementedError


class Trainer:
    """Train and validate a model.

    Args:
        model: Any model exposing the project's forward conventions.
        config: The resolved configuration (see :func:`seismic_ml.config.load_config`).
        device: Target device.
        output_dir: Directory for checkpoints/logs of this run.

    TODO(Phase 3): implement for the softmax CNN baseline; extend in Phase 8
        for evidential training (loss selection by ``training.loss.name``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        output_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}.")
        self.model = model
        self.config: dict[str, Any] = dict(config)
        self.device = torch.device(device)
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        """Run training; return a history of per-epoch metrics."""
        raise NotYetImplementedError("Trainer.fit", phase=3)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch; return aggregated metrics."""
        raise NotYetImplementedError("Trainer.train_epoch", phase=3)

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on a loader without updating weights."""
        raise NotYetImplementedError("Trainer.evaluate", phase=3)

    def save_checkpoint(self, path: str | Path) -> None:
        raise NotYetImplementedError("Trainer.save_checkpoint", phase=3)

    def load_checkpoint(self, path: str | Path) -> None:
        raise NotYetImplementedError("Trainer.load_checkpoint", phase=3)
