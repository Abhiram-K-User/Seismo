"""Unit tests for Dirichlet subjective logic and Evidential Deep Learning (Phase 8)."""

from __future__ import annotations

import pytest
import torch

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.training.losses import annealing_coefficient, evidential_loss, kl_dirichlet_regularizer
from seismic_ml.uncertainty import (
    DirichletOutput,
    belief_masses,
    dirichlet_from_evidence,
    dirichlet_mutual_information,
    dirichlet_strength,
    dissonance,
    evidence_to_alpha,
    expected_probabilities,
    max_probability,
    predictive_entropy,
    uncertainty_mass,
)

B, K = 3, 2


def _fields(**overrides) -> dict:
    fields = {
        "evidence": torch.zeros(B, K),
        "alpha": torch.zeros(B, K),
        "strength": torch.zeros(B),
        "belief": torch.zeros(B, K),
        "uncertainty": torch.zeros(B),
        "expected_probability": torch.zeros(B, K),
    }
    fields.update(overrides)
    return fields


def test_dirichlet_output_container() -> None:
    output = DirichletOutput(**_fields())
    assert output.num_classes == K


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence": torch.zeros(B)},
        {"alpha": torch.zeros(B, K + 1)},
        {"belief": torch.zeros(B + 1, K)},
        {"expected_probability": torch.zeros(B, 1)},
        {"strength": torch.zeros(B, 1)},
        {"uncertainty": torch.zeros(B + 1)},
    ],
)
def test_dirichlet_output_rejects_inconsistent_shapes(overrides: dict) -> None:
    with pytest.raises(ValueError):
        DirichletOutput(**_fields(**overrides))


def test_dirichlet_subjective_logic_identity() -> None:
    evidence = torch.tensor([[0.0, 0.0], [5.0, 1.0], [10.0, 20.0]])  # [B, 2]
    out = dirichlet_from_evidence(evidence)

    assert out.alpha.shape == (3, 2)
    assert out.strength.shape == (3,)
    assert out.belief.shape == (3, 2)
    assert out.uncertainty.shape == (3,)
    assert out.expected_probability.shape == (3, 2)

    # Fundamental Subjective Logic Identity: sum_k(b_k) + u == 1.0
    total_belief_and_u = out.belief.sum(dim=-1) + out.uncertainty
    assert torch.allclose(total_belief_and_u, torch.ones_like(total_belief_and_u))

    # Expected probabilities sum to 1.0
    assert torch.allclose(out.expected_probability.sum(dim=-1), torch.ones(3))


def test_zero_evidence_state() -> None:
    # Completely uninformed agent: e = 0
    zero_e = torch.zeros(1, 2)
    out = dirichlet_from_evidence(zero_e)

    assert torch.all(out.alpha == 1.0)
    assert torch.all(out.belief == 0.0)
    assert torch.all(out.uncertainty == 1.0)  # Max vacuity
    assert torch.allclose(out.expected_probability, torch.tensor([[0.5, 0.5]]))  # Uniform probability


def test_high_evidence_state() -> None:
    # Confident agent: high evidence for class 0
    high_e = torch.tensor([[1000.0, 0.0]])
    out = dirichlet_from_evidence(high_e)

    assert out.uncertainty.item() < 0.01  # Very low uncertainty
    assert out.belief[0, 0].item() > 0.99  # Strong belief in class 0
    assert out.expected_probability[0, 0].item() > 0.99


def test_evidence_negative_rejection() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        evidence_to_alpha(torch.tensor([[-1.0, 2.0]]))


@pytest.mark.parametrize("variant", ["digamma", "log", "mse"])
def test_evidential_loss_computation(variant: str) -> None:
    alpha = torch.tensor([[2.0, 1.0], [1.0, 4.0]], requires_grad=True)
    targets = torch.tensor([0, 1])

    loss = evidential_loss(alpha, targets, variant=variant, regularization_weight=0.1)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Gradient backward
    loss.backward()
    assert alpha.grad is not None and torch.all(torch.isfinite(alpha.grad))


def test_kl_dirichlet_regularizer() -> None:
    # Target is class 0
    targets = torch.tensor([0])

    # No misleading evidence on class 1
    alpha_clean = torch.tensor([[5.0, 1.0]])
    kl_clean = kl_dirichlet_regularizer(alpha_clean, targets)

    # High misleading evidence on class 1 (wrong class)
    alpha_misleading = torch.tensor([[5.0, 10.0]])
    kl_misleading = kl_dirichlet_regularizer(alpha_misleading, targets)

    assert kl_misleading.item() > kl_clean.item()


def test_annealing_coefficient() -> None:
    assert annealing_coefficient(0, 10) == 0.0
    assert annealing_coefficient(5, 10) == 0.5
    assert annealing_coefficient(10, 10) == 1.0
    assert annealing_coefficient(15, 10) == 1.0


def test_uncertainty_scores_are_placeholders() -> None:
    x = torch.zeros(B, K)
    for fn in (predictive_entropy, max_probability, dirichlet_mutual_information, dissonance):
        with pytest.raises(NotYetImplementedError) as info:
            fn(x)
        assert info.value.phase == 9
