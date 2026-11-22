"""Training and evaluation interfaces exist and are placeholders."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from seismic_ml.evaluation import (
    area_under_risk_coverage,
    brier_score,
    compute_classification_metrics,
    confusion_matrix,
    expected_calibration_error,
    id_quantile_threshold,
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


def test_calibration_metrics_phase9() -> None:
    # Perfectly confident and accurate predictions -> ECE = 0, Brier = 0
    probs_perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1, 0, 1])

    assert expected_calibration_error(probs_perfect, labels, num_bins=5) == pytest.approx(0.0)
    assert brier_score(probs_perfect, labels) == pytest.approx(0.0)

    rel = reliability_curve(probs_perfect, labels, num_bins=5)
    assert len(rel.bin_confidence) == 5
    assert int(np.sum(rel.bin_counts)) == 4
    assert rel.bin_accuracy[-1] == pytest.approx(1.0)
    assert rel.bin_confidence[-1] == pytest.approx(1.0)

    # Completely wrong with 0.9 confidence -> ECE = 0.9, Brier = (0.9^2 + 0.9^2) = 1.62
    probs_wrong = np.array([[0.1, 0.9], [0.9, 0.1]])
    labels_wrong = np.array([0, 1])
    assert expected_calibration_error(probs_wrong, labels_wrong, num_bins=10) == pytest.approx(0.9)
    assert brier_score(probs_wrong, labels_wrong) == pytest.approx(1.62)


def test_ood_and_selective_prediction_metrics_phase9() -> None:
    # Perfect OOD separation: all OOD scores > all ID scores
    in_scores = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    ood_scores = np.array([0.45, 0.55, 0.65, 0.80])

    assert ood_auroc(in_scores, ood_scores) == pytest.approx(1.0)
    assert ood_auprc(in_scores, ood_scores) == pytest.approx(1.0)

    tau = id_quantile_threshold(in_scores, id_tpr=0.80)
    assert 0.25 <= tau <= 0.30
    assert ood_detection_rate(ood_scores, threshold=tau) == pytest.approx(1.0)

    # Identical distributions -> AUROC = 0.5
    same = np.array([0.2, 0.4, 0.6, 0.8])
    assert ood_auroc(same, same) == pytest.approx(0.5)

    # Selective prediction risk-coverage: low uncertainty samples are correct, highest is wrong
    unc = np.array([0.1, 0.2, 0.3, 0.8])
    correct = np.array([True, True, True, False])
    rc = risk_coverage_curve(unc, correct)
    assert np.allclose(rc.coverage, [0.25, 0.50, 0.75, 1.00])
    assert np.allclose(rc.risk, [0.0, 0.0, 0.0, 0.25])
    aurc = area_under_risk_coverage(rc)
    assert 0.0 < aurc < 0.05
