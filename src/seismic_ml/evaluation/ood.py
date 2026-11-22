"""OOD detection and selective-prediction metrics (Phase 9).

Score convention: **higher score = more likely OOD / more uncertain**
(e.g. evidential vacuity ``u = K / S``). OOD events are the positive class.

Thresholds must never be tuned on the OOD test set; derive them from
in-distribution validation data via :func:`id_quantile_threshold`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class RiskCoverageCurve(NamedTuple):
    """Selective-prediction curve: keep the most confident fraction of samples.

    Attributes:
        coverage: Fraction of samples retained, ascending in ``(0, 1]``.
        risk: Error rate on the retained samples at each coverage.
    """

    coverage: np.ndarray
    risk: np.ndarray


def _validate_scores(scores: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def ood_auroc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating OOD (positive) from in-distribution (negative) samples.

    Computed exactly via the Mann-Whitney U statistic with tie-averaging:
    the probability that a randomly chosen OOD event has a strictly higher
    uncertainty score than a randomly chosen in-distribution event (plus 0.5
    for ties).
    """
    in_arr = _validate_scores(in_scores, "in_scores")
    ood_arr = _validate_scores(ood_scores, "ood_scores")

    n_in = in_arr.size
    n_ood = ood_arr.size

    # Pairwise comparison: ood[:, None] vs in[None, :]
    # For large arrays, use sorted searchsorted to stay O((N + M) log(N + M))
    in_sorted = np.sort(in_arr)
    strictly_greater = np.searchsorted(in_sorted, ood_arr, side="left")
    greater_or_equal = np.searchsorted(in_sorted, ood_arr, side="right")
    ties = greater_or_equal - strictly_greater

    u_stat = float(np.sum(strictly_greater) + 0.5 * np.sum(ties))
    return u_stat / float(n_in * n_ood)


def ood_auprc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Area under the Precision-Recall curve with OOD as the positive class.

    Uses exact step-wise Average Precision (AP) integration over unique score
    thresholds sorted in descending order::

        AUPRC = sum_k (R_k - R_{k-1}) * P_k
    """
    in_arr = _validate_scores(in_scores, "in_scores")
    ood_arr = _validate_scores(ood_scores, "ood_scores")

    scores = np.concatenate([in_arr, ood_arr])
    labels = np.concatenate([
        np.zeros(in_arr.size, dtype=np.int64),
        np.ones(ood_arr.size, dtype=np.int64),
    ])

    # Sort descending by score
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # Find indices where score value changes (to handle tied scores properly)
    distinct_mask = np.r_[np.diff(sorted_scores) != 0, True]
    threshold_idxs = np.where(distinct_mask)[0]

    tps = np.cumsum(sorted_labels)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)

    precisions = tps / np.maximum(tps + fps, 1e-12)
    recalls = tps / float(ood_arr.size)

    # Prepend (recall=0, precision=1)
    recalls_ext = np.r_[0.0, recalls]
    precisions_ext = np.r_[1.0, precisions]

    d_recall = np.diff(recalls_ext)
    return float(np.sum(d_recall * precisions_ext[1:]))


def id_quantile_threshold(in_scores: np.ndarray, *, id_tpr: float = 0.95) -> float:
    """Select an uncertainty threshold from in-distribution scores at a target ID retention rate.

    Args:
        in_scores: 1D array of in-distribution uncertainty scores (e.g. validation set).
        id_tpr: Target fraction of in-distribution samples kept below the threshold,
            in ``(0, 1)``. Default ``0.95`` (5% false positive OOD flag rate on ID).

    Returns:
        Threshold ``tau`` such that ``quantile(in_scores, id_tpr) == tau``.
    """
    in_arr = _validate_scores(in_scores, "in_scores")
    if not (isinstance(id_tpr, (int, float)) and not isinstance(id_tpr, bool)):
        raise TypeError("id_tpr must be a float in (0, 1).")
    if not (0.0 < float(id_tpr) < 1.0):
        raise ValueError(f"id_tpr must be in (0, 1), got {id_tpr}.")
    return float(np.quantile(in_arr, float(id_tpr)))


def ood_detection_rate(ood_scores: np.ndarray, *, threshold: float) -> float:
    """Fraction of OOD samples flagged as anomalous (``score >= threshold``).

    Args:
        ood_scores: 1D array of OOD uncertainty scores.
        threshold: Rejection threshold ``tau`` derived from in-distribution data
            (see :func:`id_quantile_threshold`).
    """
    ood_arr = _validate_scores(ood_scores, "ood_scores")
    if not (isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and math.isfinite(threshold)):
        raise ValueError(f"threshold must be a finite number, got {threshold}.")
    return float(np.mean(ood_arr >= float(threshold)))


def risk_coverage_curve(uncertainty: np.ndarray, correct: np.ndarray) -> RiskCoverageCurve:
    """Compute the selective-prediction risk-coverage curve.

    Samples are ordered by ascending uncertainty (most confident first). At each
    coverage ``k / M`` (for ``k = 1..M``), risk is the classification error rate
    among the ``k`` retained samples.

    Args:
        uncertainty: ``[M]`` uncertainty scores (lower = more confident).
        correct: ``[M]`` boolean or 0/1 flags indicating whether each prediction is correct.

    Returns:
        :class:`RiskCoverageCurve` with ``coverage`` and ``risk`` arrays of length ``M``.
    """
    unc_arr = _validate_scores(uncertainty, "uncertainty")
    corr_arr = np.asarray(correct)
    if corr_arr.ndim != 1 or corr_arr.size != unc_arr.size:
        raise ValueError(
            f"correct must be 1D of length {unc_arr.size}, got shape {corr_arr.shape}."
        )

    errors = 1.0 - corr_arr.astype(np.float64)
    order = np.argsort(unc_arr, kind="mergesort")
    sorted_errors = errors[order]

    m = unc_arr.size
    counts = np.arange(1, m + 1, dtype=np.float64)
    coverage = counts / float(m)
    risk = np.cumsum(sorted_errors) / counts

    return RiskCoverageCurve(coverage=coverage, risk=risk)


def area_under_risk_coverage(curve: RiskCoverageCurve) -> float:
    """Compute Area Under the Risk-Coverage Curve (AURC). Lower is better."""
    if curve.coverage.size == 0:
        return 0.0
    if curve.coverage.size == 1:
        return float(curve.risk[0])
    trapz_fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz_fn(curve.risk, curve.coverage))
