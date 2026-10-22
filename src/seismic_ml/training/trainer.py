"""Training and evaluation loop (Phase 3).

Trains models using AdamW, cosine scheduling, early stopping, and tracks
classification and loss metrics with interactive progress bars. Saves model
checkpoints, training history, and execution metadata.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from seismic_ml.evaluation.classification import compute_classification_metrics, confusion_matrix
from seismic_ml.training.losses import annealing_coefficient, cross_entropy_loss, evidential_loss
from seismic_ml.uncertainty.dirichlet import DirichletOutput

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

        # EDL loss configuration (Phase 8)
        loss_cfg = train_cfg.get("loss", {})
        edl_cfg = loss_cfg.get("evidential", {}) if isinstance(loss_cfg, dict) else {}
        self.edl_variant = (edl_cfg.get("variant") if edl_cfg else None) or "digamma"
        self.annealing_epochs = int((edl_cfg.get("kl_annealing_epochs") if edl_cfg else None) or 10)
        self.kl_weight = float((edl_cfg.get("kl_weight") if edl_cfg else None) or 1.0)

    def train_epoch(self, loader: DataLoader, *, current_epoch: int = 1, pbar: Any = None) -> dict[str, float]:
        """Run one training epoch; return aggregated metrics."""
        self.model.train()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        all_uncertainties: list[float] = []
        num_batches = 0
        correct_count = 0
        sample_count = 0
        epoch_start = time.time()

        for batch in loader:
            waveforms = batch.waveforms.to(self.device)
            station_mask = batch.station_mask.to(self.device)
            labels = batch.labels.to(self.device)

            coordinates = batch.coordinates.to(self.device) if batch.coordinates is not None else None

            # Filter out unlabeled/OOD samples if present (label == -1)
            valid_mask = labels >= 0
            if not valid_mask.any():
                if pbar is not None:
                    pbar.update(1)
                continue

            waveforms = waveforms[valid_mask]
            station_mask = station_mask[valid_mask]
            labels = labels[valid_mask]
            if coordinates is not None:
                coordinates = coordinates[valid_mask]

            self.optimizer.zero_grad()
            out = self.model(waveforms, station_mask=station_mask, coordinates=coordinates)

            if isinstance(out, DirichletOutput):
                reg_weight = self.kl_weight * annealing_coefficient(current_epoch, self.annealing_epochs)
                loss = evidential_loss(out.alpha, labels, variant=self.edl_variant, regularization_weight=reg_weight)
                preds = torch.argmax(out.expected_probability, dim=-1)
                all_uncertainties.extend(out.uncertainty.detach().cpu().numpy().tolist())
            else:
                loss = cross_entropy_loss(out, labels, class_weights=self.class_weights)
                preds = torch.argmax(out, dim=-1)

            loss.backward()
            self.optimizer.step()

            batch_loss = float(loss.item())
            total_loss += batch_loss
            preds_np = preds.detach().cpu().numpy()
            targets_np = labels.detach().cpu().numpy()

            all_preds.extend(preds_np.tolist())
            all_targets.extend(targets_np.tolist())
            correct_count += int((preds_np == targets_np).sum())
            sample_count += len(targets_np)
            num_batches += 1

            if pbar is not None:
                pbar.update(1)
                step_ms = int(((time.time() - epoch_start) / max(num_batches, 1)) * 1000)
                running_acc = correct_count / max(sample_count, 1)
                running_loss = total_loss / max(num_batches, 1)
                postfix = f"{step_ms}ms/step - accuracy: {running_acc:.4f} - loss: {running_loss:.4f}"
                if all_uncertainties:
                    running_u = float(np.mean(all_uncertainties))
                    postfix += f" - u: {running_u:.4f}"
                pbar.set_postfix_str(postfix)

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = compute_classification_metrics(
            np.array(all_targets), np.array(all_preds), self.class_names
        )
        metrics["loss"] = total_loss / max(num_batches, 1)
        if all_uncertainties:
            metrics["mean_uncertainty"] = float(np.mean(all_uncertainties))
        return metrics

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate on a loader without updating weights."""
        self.model.eval()
        total_loss = 0.0
        all_preds: list[int] = []
        all_targets: list[int] = []
        all_uncertainties: list[float] = []
        all_beliefs: list[list[float]] = []
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                waveforms = batch.waveforms.to(self.device)
                station_mask = batch.station_mask.to(self.device)
                labels = batch.labels.to(self.device)
                coordinates = batch.coordinates.to(self.device) if batch.coordinates is not None else None

                # In-distribution evaluation with labels
                if (labels >= 0).any():
                    valid_mask = labels >= 0
                    eval_waveforms = waveforms[valid_mask]
                    eval_station_mask = station_mask[valid_mask]
                    eval_labels = labels[valid_mask]
                    eval_coords = coordinates[valid_mask] if coordinates is not None else None

                    out = self.model(eval_waveforms, station_mask=eval_station_mask, coordinates=eval_coords)

                    if isinstance(out, DirichletOutput):
                        loss = evidential_loss(out.alpha, eval_labels, variant=self.edl_variant, regularization_weight=0.0)
                        preds = torch.argmax(out.expected_probability, dim=-1)
                        all_uncertainties.extend(out.uncertainty.detach().cpu().numpy().tolist())
                        all_beliefs.extend(out.belief.detach().cpu().numpy().tolist())
                    else:
                        loss = cross_entropy_loss(out, eval_labels, class_weights=self.class_weights)
                        preds = torch.argmax(out, dim=-1)

                    total_loss += float(loss.item())
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_targets.extend(eval_labels.cpu().numpy().tolist())
                    num_batches += 1
                else:
                    # Unlabeled / Out-Of-Distribution evaluation
                    out = self.model(waveforms, station_mask=station_mask, coordinates=coordinates)
                    if isinstance(out, DirichletOutput):
                        preds = torch.argmax(out.expected_probability, dim=-1)
                        all_uncertainties.extend(out.uncertainty.detach().cpu().numpy().tolist())
                        all_beliefs.extend(out.belief.detach().cpu().numpy().tolist())
                    else:
                        preds = torch.argmax(out, dim=-1)
                    all_preds.extend(preds.cpu().numpy().tolist())
                    num_batches += 1

        if all_targets:
            metrics = compute_classification_metrics(
                np.array(all_targets), np.array(all_preds), self.class_names
            )
            metrics["loss"] = total_loss / max(num_batches, 1)
            cm = confusion_matrix(np.array(all_targets), np.array(all_preds), len(self.class_names))
            metrics["confusion_matrix"] = cm.tolist()
        else:
            # For OOD/unlabeled events, classification accuracy/F1 does not apply
            metrics = {
                "num_events": len(all_preds),
            }

        if all_uncertainties:
            metrics["mean_uncertainty"] = float(np.mean(all_uncertainties))
            metrics["uncertainties"] = all_uncertainties
            metrics["beliefs"] = all_beliefs
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

        total_batches = len(train_loader)

        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")
            epoch_start = time.time()

            pbar = tqdm(
                total=total_batches,
                unit="step",
                leave=True,
                dynamic_ncols=True,
                bar_format="{n_fmt}/{total_fmt} ━{bar:20}━ {elapsed} {postfix}",
            )

            train_metrics = self.train_epoch(train_loader, current_epoch=epoch, pbar=pbar)
            pbar.close()

            epoch_data: dict[str, Any] = {"epoch": epoch, "train": train_metrics}

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                epoch_data["val"] = val_metrics
                score = val_metrics.get(monitor, val_metrics.get("macro_f1", 0.0))

                ms_per_step = int(((time.time() - epoch_start) / max(total_batches, 1)) * 1000)
                pbar.set_postfix_str(
                    f"{ms_per_step}ms/step - accuracy: {train_metrics['accuracy']:.4f} - "
                    f"loss: {train_metrics['loss']:.4f} - val_accuracy: {val_metrics['accuracy']:.4f} - "
                    f"val_loss: {val_metrics['loss']:.4f}"
                )

                is_better = (
                    (score > best_val_score)
                    if ("f1" in monitor or "acc" in monitor)
                    else (score < best_val_score)
                )
                if is_better:
                    best_val_score = score
                    patience_counter = 0
                    if self.output_dir is not None:
                        self.save_checkpoint(self.output_dir / "best_model.pt")
                else:
                    patience_counter += 1

            pbar.close()

            if val_loader is not None and patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
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
