"""Dirichlet / subjective-logic quantities for Evidential Deep Learning (Stage E, Phase 8).

Definitions for ``K`` classes and non-negative evidence ``e_k >= 0``::

    alpha_k = e_k + 1                 Dirichlet parameters
    S       = sum_k alpha_k           Dirichlet strength
    b_k     = e_k / S                 belief mass of class k
    u       = K / S                   uncertainty mass ("vacuity")
    p_k     = alpha_k / S             expected class probability

These satisfy ``sum_k b_k + u = 1`` (subjective logic identity).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from seismic_ml.utils.validation import check_int


@dataclass(frozen=True, slots=True)
class DirichletOutput:
    """Output of an evidential classifier for a batch of ``B`` events.

    ============================ ========== =================================
    field                        shape      definition
    ============================ ========== =================================
    ``evidence``                 ``[B, K]`` ``e_k >= 0``
    ``alpha``                    ``[B, K]`` ``e_k + 1``
    ``strength``                 ``[B]``    ``S = sum_k alpha_k``
    ``belief``                   ``[B, K]`` ``e_k / S``
    ``uncertainty``              ``[B]``    ``K / S``
    ``expected_probability``     ``[B, K]`` ``alpha_k / S``
    ============================ ========== =================================
    """

    evidence: torch.Tensor
    alpha: torch.Tensor
    strength: torch.Tensor
    belief: torch.Tensor
    uncertainty: torch.Tensor
    expected_probability: torch.Tensor

    def __post_init__(self) -> None:
        if self.evidence.ndim != 2:
            raise ValueError(f"evidence must be [B, K], got {tuple(self.evidence.shape)}.")
        per_class = tuple(self.evidence.shape)
        for name in ("alpha", "belief", "expected_probability"):
            shape = tuple(getattr(self, name).shape)
            if shape != per_class:
                raise ValueError(f"{name} must have shape {per_class}, got {shape}.")
        for name in ("strength", "uncertainty"):
            shape = tuple(getattr(self, name).shape)
            if shape != per_class[:1]:
                raise ValueError(f"{name} must have shape {per_class[:1]}, got {shape}.")

    @property
    def num_classes(self) -> int:
        return self.evidence.shape[1]


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` non-negative evidence -> ``[B, K]`` Dirichlet parameters ``alpha = e + 1``."""
    if evidence.ndim != 2:
        raise ValueError(f"evidence must be [B, K], got {tuple(evidence.shape)}")
    if torch.any(evidence < 0.0):
        raise ValueError("evidence must be non-negative (>= 0)")
    return evidence + 1.0


def dirichlet_strength(alpha: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` alpha -> ``[B]`` Dirichlet strength ``S = sum_k alpha_k``."""
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be [B, K], got {tuple(alpha.shape)}")
    return torch.sum(alpha, dim=-1)


def belief_masses(evidence: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` evidence and ``[B]`` strength -> ``[B, K]`` belief masses ``b = e / S``."""
    if evidence.ndim != 2:
        raise ValueError(f"evidence must be [B, K], got {tuple(evidence.shape)}")
    if strength.ndim != 1 or strength.shape[0] != evidence.shape[0]:
        raise ValueError(f"strength must be [B], matching evidence batch {evidence.shape[0]}")
    s_exp = strength.unsqueeze(-1).clamp(min=1e-8)
    return evidence / s_exp


def uncertainty_mass(strength: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``[B]`` strength -> ``[B]`` uncertainty mass ``u = K / S``."""
    if strength.ndim != 1:
        raise ValueError(f"strength must be [B], got {tuple(strength.shape)}")
    check_int("num_classes", num_classes, minimum=2)
    s_clamped = strength.clamp(min=1e-8)
    return float(num_classes) / s_clamped


def expected_probabilities(alpha: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` alpha and ``[B]`` strength -> ``[B, K]`` expected probabilities ``alpha / S``."""
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be [B, K], got {tuple(alpha.shape)}")
    if strength.ndim != 1 or strength.shape[0] != alpha.shape[0]:
        raise ValueError(f"strength must be [B], matching alpha batch {alpha.shape[0]}")
    s_exp = strength.unsqueeze(-1).clamp(min=1e-8)
    return alpha / s_exp


def dirichlet_from_evidence(evidence: torch.Tensor) -> DirichletOutput:
    """Compute every field of :class:`DirichletOutput` from ``[B, K]`` evidence."""
    alpha = evidence_to_alpha(evidence)
    strength = dirichlet_strength(alpha)
    k = evidence.shape[1]
    belief = belief_masses(evidence, strength)
    uncertainty = uncertainty_mass(strength, k)
    expected_prob = expected_probabilities(alpha, strength)
    return DirichletOutput(
        evidence=evidence,
        alpha=alpha,
        strength=strength,
        belief=belief,
        uncertainty=uncertainty,
        expected_probability=expected_prob,
    )
