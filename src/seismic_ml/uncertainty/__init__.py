"""Dirichlet-based uncertainty structures and uncertainty scores."""

from seismic_ml.uncertainty.dirichlet import (
    DirichletOutput,
    belief_masses,
    dirichlet_from_evidence,
    dirichlet_strength,
    evidence_to_alpha,
    expected_probabilities,
    uncertainty_mass,
)
from seismic_ml.uncertainty.metrics import (
    dirichlet_mutual_information,
    dissonance,
    max_probability,
    predictive_entropy,
)

__all__ = [
    "DirichletOutput",
    "belief_masses",
    "dirichlet_from_evidence",
    "dirichlet_mutual_information",
    "dirichlet_strength",
    "dissonance",
    "evidence_to_alpha",
    "expected_probabilities",
    "max_probability",
    "predictive_entropy",
    "uncertainty_mass",
]
