"""Training and evaluation loop (Phase 3).

Trains models using AdamW, cosine scheduling, early stopping, and tracks
classification and loss metrics. Saves model checkpoints and execution metadata.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from seismic_ml.evaluation.classification import compute_classification_metrics, confusion_matrix
from seismic_ml.training.losses import cross_entropy_loss

logger = logging.getLogger("seismic_ml.training")


class Trainer:
    """Train and validate a model.

    Args:
        model: Any model exposing the project's forward conventions.
        config: The resolved configuration.
        device: Target device ("cpu", "cuda", etc.).
        output_dir: Directory for checkpoints/logs of this run.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
        output_dir: str | Path | None = None,
        class_names: Sequence[str] = ("earthquake", "anthropogenic"),
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}.")
        self.model = model
        self.config: dict[str, Any] = dict(config)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names)

        # Setup optimizer and scheduler from training config
        train_cfg = self.config.get("training", {})
        opt_cfg = train_cfg.get("optimizer") or {}
        lr = float(opt_cfg.get("lr") or 1e-3)
        weight_decay = float(opt_cfg.get("weight_decay") or 1e-4)
        opt_name = (opt_cfg.get("name") or "adamw").lower()

        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        max_epochs = train_cfg.get("max_epochs") or 30
        sched_cfg = train_cfg.get("scheduler") or {}
        sched_name = sched_cfg.get("name")
        if sched_name == "cosine":
            self.scheduler: Any = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs
            )
        else:
            self.scheduler = None

        # Class weights for loss
        class_weights_cfg = train_cfg.get("class_weighting")
        if class_weights_cfg is not None:
            self.class_weights = torch.tensor(class_weights_cfg, dtype=torch.float32, device=self.device)
        else:
            self.class_weights = None

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one training epoch; return aggregated metrics."""
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        num_batches = 0

        for batch in loader:
            waveforms = batch.waveforms.to(self.device)
            station_mask = batch.station_mask.to(self.device)
            labels = batch.labels.to(self.device)

            # Filter out unlabeled/OOD samples if present (label == -1)
            valid_mask = labels >= 0
            if not valid_mask.any():
                continue

            waveforms = waveforms[valid_mask]
            station_mask = station_mask[valid_mask]
            labels = labels[valid_mask]

            self.optimizer.zero_grad()
            logits = self.model(waveforms, station_mask=station_mask)
            loss = cross_entropy_loss(logits, labels, class_weights=self.class_weights)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = compute_classification_metrics(
            np.array(all_targets), np.array(all_preds), self.class_names
        )
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate on a loader without updating weights."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                waveforms = batch.waveforms.to(self.device)
                station_mask = batch.station_mask.to(self.device)
                labels = batch.labels.to(self.device)

                valid_mask = labels >= 0
                if not valid_mask.any():
                    continue

                waveforms = waveforms[valid_mask]
                station_mask = station_mask[valid_mask]
                labels = labels[valid_mask]

                logits = self.model(waveforms, station_mask=station_mask)
                loss = cross_entropy_loss(logits, labels, class_weights=self.class_weights)

                total_loss += float(loss.item())
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())
                num_batches += 1

        metrics = compute_classification_metrics(
            np.array(all_targets), np.array(all_preds), self.class_names
        )
        metrics["loss"] = total_loss / max(num_batches, 1)
        cm = confusion_matrix(np.array(all_targets), np.array(all_preds), len(self.class_names))
        metrics["confusion_matrix"] = cm.tolist()
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> dict[str, Any]:
        """Run training across epochs; return history of per-epoch metrics."""
        train_cfg = self.config.get("training", {})
        max_epochs = train_cfg.get("max_epochs") or 30
        early_cfg = train_cfg.get("early_stopping") or {}
        patience = early_cfg.get("patience") or 10
        monitor = early_cfg.get("monitor") or "macro_f1"

        history: list[dict[str, Any]] = []
        best_val_score = -float("inf") if "f1" in monitor or "acc" in monitor else float("inf")
        patience_counter = 0

        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            epoch_data: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_data["val"] = val_metrics
                score = val_metrics.get(monitor, val_metrics.get("macro_f1", 0.0))

                is_better = (score > best_val_score) if ("f1" in monitor or "acc" in monitor) else (score < best_val_score)
                if is_better:
                    best_val_score = score
                    patience_counter = 0
                    if self.output_dir is not None:
                        self.save_checkpoint(self.output_dir / "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

            history.append(epoch_data)

        if self.output_dir is not None:
            self.save_checkpoint(self.output_dir / "last_model.pt")

        return {"history": history, "best_val_score": best_val_score}

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model weights and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
