"""Dirichlet / subjective-logic quantities for Evidential Deep Learning (Stage E).

Only the *structure* exists in Phase 0: the :class:`DirichletOutput` container
(shape validation only) and function interfaces. The computations are
planned for Phase 8.

Definitions for ``K`` classes and non-negative evidence ``e_k >= 0``::

    alpha_k = e_k + 1                 Dirichlet parameters
    S       = sum_k alpha_k           Dirichlet strength
    b_k     = e_k / S                 belief mass of class k
    u       = K / S                   uncertainty mass ("vacuity")
    p_k     = alpha_k / S             expected class probability

These satisfy ``sum_k b_k + u = 1`` (a useful test invariant for Phase 8).

EDL does NOT guarantee OOD detection: a high ``u`` on unfamiliar events is a
hypothesis (H5) to be tested experimentally, not a property of the method.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from seismic_ml.placeholders import NotYetImplementedError


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

    The container validates shapes only; it does not compute or check values.
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
    """``[B, K]`` evidence -> ``[B, K]`` Dirichlet parameters ``alpha = e + 1``.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("evidence_to_alpha", phase=8)


def dirichlet_strength(alpha: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` alpha -> ``[B]`` strength ``S = sum_k alpha_k``.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("dirichlet_strength", phase=8)


def belief_masses(evidence: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` evidence and ``[B]`` strength -> ``[B, K]`` beliefs ``e / S``.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("belief_masses", phase=8)


def uncertainty_mass(strength: torch.Tensor, num_classes: int) -> torch.Tensor:
    """``[B]`` strength -> ``[B]`` uncertainty ``u = K / S``.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("uncertainty_mass", phase=8)


def expected_probabilities(alpha: torch.Tensor, strength: torch.Tensor) -> torch.Tensor:
    """``[B, K]`` alpha and ``[B]`` strength -> ``[B, K]`` probabilities ``alpha / S``.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("expected_probabilities", phase=8)


def dirichlet_from_evidence(evidence: torch.Tensor) -> DirichletOutput:
    """Compute every field of :class:`DirichletOutput` from ``[B, K]`` evidence.

    TODO(Phase 8): implement by composing the functions above.
    """
    raise NotYetImplementedError("dirichlet_from_evidence", phase=8)
