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


def test_trainer_instantiation_and_placeholders() -> None:
    trainer = Trainer(nn.Linear(2, 2), config={"seed": 42})
    assert trainer.device == torch.device("cpu")
    assert trainer.output_dir is None
    for method in (trainer.fit, trainer.train_epoch, trainer.evaluate):
        _assert_placeholder(lambda m=method: m(None), phase=3)
    _assert_placeholder(lambda: trainer.save_checkpoint("x.pt"), phase=3)
    _assert_placeholder(lambda: trainer.load_checkpoint("x.pt"), phase=3)


def test_trainer_requires_module() -> None:
    with pytest.raises(TypeError):
        Trainer(object(), config={})  # type: ignore[arg-type]


def test_losses_are_placeholders() -> None:
    logits = torch.zeros(4, 2)
    targets = torch.zeros(4, dtype=torch.long)
    _assert_placeholder(lambda: cross_entropy_loss(logits, targets), phase=3)
    _assert_placeholder(
        lambda: evidential_loss(logits, targets, variant="mse", regularization_weight=0.0), phase=8
    )
    _assert_placeholder(lambda: kl_dirichlet_regularizer(logits, targets), phase=8)
    _assert_placeholder(lambda: annealing_coefficient(1, 10), phase=8)


def test_classification_metrics_are_placeholders() -> None:
    y = np.zeros(4, dtype=int)
    _assert_placeholder(
        lambda: compute_classification_metrics(y, y, ["earthquake", "anthropogenic"]), phase=3
    )
    _assert_placeholder(lambda: confusion_matrix(y, y, 2), phase=3)


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
