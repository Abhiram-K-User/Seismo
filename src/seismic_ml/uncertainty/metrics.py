"""Per-sample uncertainty scores (Phase 9).

Decomposes predictive uncertainty into:
- ``predictive_entropy``: Total Shannon entropy of expected probabilities ``H(p)``.
- ``max_probability``: Maximum class probability (Hendrycks & Gimpel, 2017 confidence baseline).
- ``dirichlet_mutual_information``: Epistemic mutual information ``I[y, pi | x] = H(E[pi]) - E[H(pi)]``
  under ``Dir(alpha)`` (Malinin & Gales, 2018).
- ``dissonance``: Subjective Logic belief dissonance (Josang, 2018; Zhao et al., 2020),
  quantifying conflicting evidence (aleatoric ambiguity) separately from epistemic vacuity ``u = K / S``.
"""

from __future__ import annotations

import torch


def _validate_2d(tensor: torch.Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim != 2:
        raise ValueError(f"{name} must have shape [B, K], got {tuple(tensor.shape)}.")
    if tensor.shape[1] < 2:
        raise ValueError(f"{name} must have K >= 2 classes, got K={tensor.shape[1]}.")


def predictive_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the predictive distribution: ``[B, K]`` -> ``[B]`` (in nats).

    .. math::
        H(p) = -\\sum_{k=1}^{K} p_k \\ln(p_k)
    """
    _validate_2d(probabilities, "probabilities")
    if torch.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")
    p = probabilities.clamp(min=1e-12)
    return -torch.sum(p * torch.log(p), dim=-1)


def max_probability(probabilities: torch.Tensor) -> torch.Tensor:
    """Maximum class probability (confidence): ``[B, K]`` -> ``[B]``."""
    _validate_2d(probabilities, "probabilities")
    if torch.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")
    return torch.max(probabilities, dim=-1).values


def dirichlet_mutual_information(alpha: torch.Tensor) -> torch.Tensor:
    """Epistemic mutual information of a Dirichlet distribution: ``[B, K]`` -> ``[B]``.

    .. math::
        I[y, \\pi] = H(\\mathbb{E}[\\pi]) - \\mathbb{E}[H(\\pi)]
                  = -\\sum_{k=1}^{K} \\frac{\\alpha_k}{S}
                    \\left(\\ln \\frac{\\alpha_k}{S} - \\psi(\\alpha_k + 1) + \\psi(S + 1)\\right)
    """
    _validate_2d(alpha, "alpha")
    if torch.any(alpha <= 0.0):
        raise ValueError("Dirichlet alpha parameters must be strictly positive (> 0).")
    s = torch.sum(alpha, dim=-1, keepdim=True)
    p = alpha / s
    p_clamped = p.clamp(min=1e-12)
    term = torch.log(p_clamped) - torch.digamma(alpha + 1.0) + torch.digamma(s + 1.0)
    mi = -torch.sum(p * term, dim=-1)
    return torch.clamp(mi, min=0.0)


def dissonance(belief: torch.Tensor) -> torch.Tensor:
    """Subjective Logic belief dissonance (conflicting evidence): ``[B, K]`` -> ``[B]``.

    Measures how much non-zero belief masses conflict across classes::

        Bal(b_j, b_k) = 1 - |b_j - b_k| / (b_j + b_k)   if (b_j + b_k) > 0 else 0
        Diss(b)       = sum_k (b_k * sum_{j != k} b_j * Bal(b_j, b_k)) / (sum_{j != k} b_j)
    """
    _validate_2d(belief, "belief")
    if torch.any(belief < 0.0):
        raise ValueError("belief masses must be non-negative.")

    b_batch, k = belief.shape
    # Pairwise sums and differences: [B, K, K]
    bi = belief.unsqueeze(2)  # [B, K, 1]
    bj = belief.unsqueeze(1)  # [B, 1, K]

    pair_sum = bi + bj
    pair_diff = torch.abs(bi - bj)
    balance = torch.where(
        pair_sum > 1e-12,
        1.0 - (pair_diff / pair_sum.clamp(min=1e-12)),
        torch.zeros_like(pair_sum),
    )

    # Zero out diagonal (j == k)
    eye = torch.eye(k, dtype=torch.bool, device=belief.device).unsqueeze(0)
    balance = balance.masked_fill(eye, 0.0)
    bj_off = bj.expand(b_batch, k, k).masked_fill(eye, 0.0)

    # Numerator per k: b_k * sum_{j != k} (b_j * Bal(b_j, b_k))
    weighted_bal = torch.sum(bj_off * balance, dim=-1)  # [B, K]
    denom = torch.sum(bj_off, dim=-1)  # [B, K] = sum_{j != k} b_j

    term_k = torch.where(
        denom > 1e-12,
        belief * weighted_bal / denom.clamp(min=1e-12),
        torch.zeros_like(belief),
    )
    return torch.clamp(torch.sum(term_k, dim=-1), min=0.0, max=1.0)
