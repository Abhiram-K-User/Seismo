"""Evidential classification head — Stage E (Phase 8).

Maps an event embedding to non-negative evidence ``e = g(f(x)) >= 0`` and computes
Dirichlet subjective logic parameters (DirichletOutput).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from seismic_ml.uncertainty.dirichlet import DirichletOutput, dirichlet_from_evidence
from seismic_ml.utils.validation import check_choice, check_int

EVIDENCE_ACTIVATIONS: tuple[str, ...] = ("relu", "softplus", "exp")


class EvidentialClassifier(nn.Module):
    """Produce Dirichlet parameters (evidence) instead of softmax probabilities.

    Args:
        in_dim: Event embedding size.
        num_classes: Number of known classes ``K`` (>= 2).
        evidence_activation: Non-negative activation ``g`` producing evidence.
            Options: ``"relu"``, ``"softplus"``, ``"exp"``.

    Expected input:
        event_embedding: ``[B, in_dim]``.

    Expected output:
        :class:`~seismic_ml.uncertainty.dirichlet.DirichletOutput` with
        ``alpha = e + 1``, ``S = sum(alpha)``, ``b = e / S``, ``u = K / S``.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        *,
        evidence_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.in_dim = check_int("in_dim", in_dim)
        self.num_classes = check_int("num_classes", num_classes, minimum=2)
        self.evidence_activation = check_choice(
            "evidence_activation", evidence_activation, EVIDENCE_ACTIVATIONS
        )

        self.linear = nn.Linear(self.in_dim, self.num_classes)

    def forward(self, event_embedding: torch.Tensor) -> DirichletOutput:
        """Forward pass.

        Args:
            event_embedding: ``[B, in_dim]`` float tensor.

        Returns:
            :class:`~seismic_ml.uncertainty.dirichlet.DirichletOutput`.
        """
        logits = self.linear(event_embedding)
        if self.evidence_activation == "relu":
            evidence = F.relu(logits)
        elif self.evidence_activation == "softplus":
            evidence = F.softplus(logits)
        elif self.evidence_activation == "exp":
            evidence = torch.exp(torch.clamp(logits, max=10.0))
        else:
            raise ValueError(f"Unknown evidence activation: {self.evidence_activation}")

        return dirichlet_from_evidence(evidence)
