"""Evidential classification head — Stage E (placeholder).

Planned for Phase 8. The head maps an event embedding to non-negative
evidence ``e = g(f(x)) >= 0`` and from there to the Dirichlet quantities in
:mod:`seismic_ml.uncertainty.dirichlet`. The EDL loss lives in
:mod:`seismic_ml.training.losses` and is not implemented either.
"""

from __future__ import annotations

import torch
from torch import nn

from seismic_ml.placeholders import NotYetImplementedError
from seismic_ml.uncertainty.dirichlet import DirichletOutput
from seismic_ml.utils.validation import check_choice, check_int

EVIDENCE_ACTIVATIONS: tuple[str, ...] = ("relu", "softplus", "exp")


class EvidentialClassifier(nn.Module):
    """Produce Dirichlet parameters (evidence) instead of softmax probabilities.

    Args:
        in_dim: Event embedding size.
        num_classes: Number of known classes ``K`` (>= 2).
        evidence_activation: Non-negative activation ``g`` producing evidence.
            Candidate options: ``"relu"``, ``"softplus"``, ``"exp"`` — to be
            chosen in Phase 8 and recorded in the Decision Log.

    Expected input:
        event_embedding: ``[B, in_dim]``.

    Expected output:
        :class:`~seismic_ml.uncertainty.dirichlet.DirichletOutput` with
        ``alpha = e + 1``, ``S = sum(alpha)``, ``b = e / S``, ``u = K / S``.

    EDL does NOT guarantee OOD detection; its usefulness must be evaluated
    experimentally (hypotheses H4, H5).

    TODO(Phase 8): implement.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        *,
        evidence_activation: str,
    ) -> None:
        super().__init__()
        self.in_dim = check_int("in_dim", in_dim)
        self.num_classes = check_int("num_classes", num_classes, minimum=2)
        self.evidence_activation = check_choice(
            "evidence_activation", evidence_activation, EVIDENCE_ACTIVATIONS
        )

    def forward(self, event_embedding: torch.Tensor) -> DirichletOutput:
        raise NotYetImplementedError("EvidentialClassifier.forward", phase=8)
