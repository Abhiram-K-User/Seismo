"""Training and evaluation interfaces exist and are placeholders."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from seismic_ml.evaluation import (
    brier_score,
    compute_classification_metrics,
    confusion_matrix,
    expected_calibration_error,
    ood_auprc,
    ood_auroc,
    ood_detection_rate,
    reliability_curve,
    risk_coverage_curve,
)
from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.training import (
    Trainer,
    annealing_coefficient,
    cross_entropy_loss,
    evidential_loss,
    kl_dirichlet_regularizer,
)


def _assert_placeholder(call, phase: int) -> None:
    with pytest.raises(NotYetImplementedError) as info:
        call()
    assert info.value.phase == phase


def test_trainer_instantiation_and_save_load(tmp_path) -> None:
    model = nn.Linear(2, 2)
    trainer = Trainer(model, config={"seed": 42}, output_dir=tmp_path)
    assert trainer.device == torch.device("cpu")
    assert trainer.output_dir == tmp_path

    ckpt_path = tmp_path / "model.pt"
    trainer.save_checkpoint(ckpt_path)
    assert ckpt_path.exists()
    trainer.load_checkpoint(ckpt_path)


def test_trainer_requires_module() -> None:
    with pytest.raises(TypeError):
        Trainer(object(), config={})  # type: ignore[arg-type]


def test_cross_entropy_loss() -> None:
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.8]])
    targets = torch.tensor([0, 1])
    loss = cross_entropy_loss(logits, targets)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_losses_are_placeholders() -> None:
    logits = torch.zeros(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    _assert_placeholder(
        lambda: evidential_loss(logits, targets, variant="mse", regularization_weight=0.0), phase=8
    )
    _assert_placeholder(lambda: kl_dirichlet_regularizer(logits, targets), phase=8)
    _assert_placeholder(lambda: annealing_coefficient(1, 10), phase=8)


def test_classification_metrics() -> None:
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0])
    class_names = ["earthquake", "anthropogenic"]

    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    assert metrics["accuracy"] == 0.8
    assert "macro_f1" in metrics
    assert "precision/earthquake" in metrics
    assert "recall/anthropogenic" in metrics

    cm = confusion_matrix(y_true, y_pred, 2)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # true 0, pred 0
    assert cm[0, 1] == 1  # true 0, pred 1
    assert cm[1, 1] == 2  # true 1, pred 1


def test_calibration_and_ood_metrics_are_placeholders() -> None:
    probs = np.full((4, 2), 0.5)
    labels = np.zeros(4, dtype=int)
    scores = np.zeros(4)
    calls = [
        lambda: expected_calibration_error(probs, labels, num_bins=10),
        lambda: brier_score(probs, labels),
        lambda: reliability_curve(probs, labels, num_bins=10),
        lambda: ood_auroc(scores, scores),
        lambda: ood_auprc(scores, scores),
        lambda: ood_detection_rate(scores, threshold=0.5),
        lambda: risk_coverage_curve(scores, labels.astype(bool)),
    ]
    for call in calls:
        _assert_placeholder(call, phase=9)
