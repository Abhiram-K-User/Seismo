"""Uncertainty structures exist; EDL computations are not implemented yet."""

from __future__ import annotations

import pytest
import torch

from seismic_ml.placeholders import NotYetImplementedError
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
    # Shape-only placeholders; values are irrelevant to the container.
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


def test_dirichlet_functions_are_placeholders() -> None:
    x = torch.zeros(B, K)
    s = torch.zeros(B)
    calls = [
        lambda: evidence_to_alpha(x),
        lambda: dirichlet_strength(x),
        lambda: belief_masses(x, s),
        lambda: uncertainty_mass(s, K),
        lambda: expected_probabilities(x, s),
        lambda: dirichlet_from_evidence(x),
    ]
    for call in calls:
        with pytest.raises(NotYetImplementedError) as info:
            call()
        assert info.value.phase == 8


def test_uncertainty_scores_are_placeholders() -> None:
    x = torch.zeros(B, K)
    for fn in (predictive_entropy, max_probability, dirichlet_mutual_information, dissonance):
        with pytest.raises(NotYetImplementedError) as info:
            fn(x)
        assert info.value.phase == 9
