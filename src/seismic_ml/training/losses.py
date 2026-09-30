"""Loss function interfaces (placeholders).

The EDL loss is deliberately NOT implemented. Several formulations exist
(Sensoy et al., 2018 propose MSE-, log- and digamma-based variants plus a KL
regularizer with annealing); the choice is made and justified in Phase 8.
"""

from __future__ import annotations

import torch

from seismic_ml.placeholders import NotYetImplementedError


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax cross-entropy for the non-evidential baselines.

    Args:
        logits: ``[B, K]``; targets: ``[B]`` class indices;
        class_weights: optional ``[K]`` weights (class-imbalance decision, Phase 1/3).

    TODO(Phase 3): implement.
    """
    raise NotYetImplementedError("cross_entropy_loss", phase=3)


def evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    *,
    variant: str,
    regularization_weight: float,
) -> torch.Tensor:
    """Evidential classification loss on Dirichlet parameters.

    Args:
        alpha: ``[B, K]`` Dirichlet parameters.
        targets: ``[B]`` class indices.
        variant: Formulation to use (candidates: ``"mse"``, ``"log"``,
            ``"digamma"``) — to be decided in Phase 8.
        regularization_weight: Weight of the KL regularizer at this step
            (e.g. from :func:`annealing_coefficient`).

    TODO(Phase 8): implement after reviewing the EDL literature, including
        known pitfalls of EDL uncertainty (see PROJECT_CONTEXT.md §20).
    """
    raise NotYetImplementedError("evidential_loss", phase=8)


def kl_dirichlet_regularizer(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence of the misleading-evidence Dirichlet from the uniform Dirichlet.

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("kl_dirichlet_regularizer", phase=8)


def annealing_coefficient(epoch: int, annealing_epochs: int) -> float:
    """Schedule for the KL regularizer weight (e.g. ``min(1, epoch / annealing_epochs)``).

    TODO(Phase 8): implement.
    """
    raise NotYetImplementedError("annealing_coefficient", phase=8)
