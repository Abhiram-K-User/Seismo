"""Loss functions for standard cross-entropy and Evidential Deep Learning (Phase 8).

Implements Dirichlet evidential losses (Sensoy et al., NeurIPS 2018) with:
- Data fit terms: "digamma" (expected cross-entropy), "log", and "mse" (expected squared error).
- KL divergence regularizer penalizing misleading evidence on non-target classes.
- Epoch-based annealing schedule.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from seismic_ml.utils.validation import check_choice

SUPPORTED_EDL_VARIANTS: tuple[str, ...] = ("digamma", "log", "mse")


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax cross-entropy for non-evidential models.

    Args:
        logits: ``[B, K]`` float tensor.
        targets: ``[B]`` long tensor of class indices.
        class_weights: Optional ``[K]`` weights tensor.
    """
    return F.cross_entropy(logits, targets, weight=class_weights)


def annealing_coefficient(epoch: int, annealing_epochs: int) -> float:
    """Schedule for the KL regularizer weight: min(1.0, epoch / annealing_epochs).

    Args:
        epoch: Current training epoch (1-indexed or 0-indexed).
        annealing_epochs: Total number of epochs over which to ramp up the regularizer.
    """
    if annealing_epochs <= 0:
        return 1.0
    return min(1.0, float(max(0, epoch)) / float(annealing_epochs))


def kl_dirichlet_regularizer(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence of the misleading-evidence Dirichlet from the uniform Dirichlet.

    Args:
        alpha: ``[B, K]`` Dirichlet parameters.
        targets: ``[B]`` class indices.

    Returns:
        Scalar tensor containing the mean KL divergence.
    """
    b, k = alpha.shape
    device = alpha.device

    y_one_hot = torch.zeros_like(alpha)
    y_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    # Keep uniform prior (1.0) on target class, keep evidence on incorrect classes
    alpha_tilde = y_one_hot + (1.0 - y_one_hot) * alpha
    s_tilde = torch.sum(alpha_tilde, dim=-1, keepdim=True)  # [B, 1]

    # KL(Dir(alpha_tilde) || Dir(1))
    # Term 1: ln(Gamma(S_tilde)) - ln(Gamma(K)) - sum(ln(Gamma(alpha_tilde)))
    term1 = (
        torch.lgamma(s_tilde).squeeze(-1)
        - torch.lgamma(torch.tensor(float(k), device=device))
        - torch.sum(torch.lgamma(alpha_tilde), dim=-1)
    )

    # Term 2: sum((alpha_tilde - 1) * (psi(alpha_tilde) - psi(S_tilde)))
    term2 = torch.sum(
        (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(s_tilde)),
        dim=-1,
    )

    kl = term1 + term2
    return torch.mean(kl)


def evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    *,
    variant: str = "digamma",
    regularization_weight: float = 0.0,
) -> torch.Tensor:
    """Evidential classification loss on Dirichlet parameters.

    Args:
        alpha: ``[B, K]`` Dirichlet parameters.
        targets: ``[B]`` class indices.
        variant: One of ``"digamma"``, ``"log"``, or ``"mse"``.
        regularization_weight: Weight of the KL regularizer at this step.

    Returns:
        Scalar loss tensor.
    """
    check_choice("variant", variant, SUPPORTED_EDL_VARIANTS)

    b, k = alpha.shape
    y_one_hot = torch.zeros_like(alpha)
    y_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

    s = torch.sum(alpha, dim=-1, keepdim=True)  # [B, 1]

    if variant == "digamma":
        # Expected cross-entropy under Dirichlet distribution
        loss_per_class = y_one_hot * (torch.digamma(s) - torch.digamma(alpha))
        data_loss = torch.mean(torch.sum(loss_per_class, dim=-1))

    elif variant == "log":
        # Log-loss on expected probabilities
        prob = alpha / s
        eps = 1e-8
        loss_per_class = - y_one_hot * torch.log(prob.clamp(min=eps))
        data_loss = torch.mean(torch.sum(loss_per_class, dim=-1))

    elif variant == "mse":
        # Expected squared error under Dirichlet
        prob = alpha / s
        err = (y_one_hot - prob) ** 2
        var = (alpha * (s - alpha)) / (s ** 2 * (s + 1.0))
        data_loss = torch.mean(torch.sum(err + var, dim=-1))

    else:
        raise ValueError(f"Unsupported evidential loss variant: {variant}")

    if regularization_weight > 0.0:
        kl_reg = kl_dirichlet_regularizer(alpha, targets)
        return data_loss + regularization_weight * kl_reg

    return data_loss
