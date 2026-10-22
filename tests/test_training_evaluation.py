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


def test_evidential_losses_and_annealing() -> None:
    alpha = torch.tensor([[2.0, 1.0], [1.5, 3.0]])
    targets = torch.tensor([0, 1])
    loss = evidential_loss(alpha, targets, variant="digamma", regularization_weight=0.1)
    assert loss.ndim == 0 and loss.item() > 0

    kl = kl_dirichlet_regularizer(alpha, targets)
    assert kl.ndim == 0 and kl.item() >= 0

    assert annealing_coefficient(5, 10) == 0.5


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
