"""Calibration metrics (Phase 9).

Implements Expected Calibration Error (ECE, equal-width binning), multi-class
Brier score, and reliability diagram bin statistics.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from seismic_ml.utils.validation import check_int


class ReliabilityCurve(NamedTuple):
    """Binned confidence vs accuracy statistics (all arrays have length ``num_bins``)."""

    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_counts: np.ndarray


def _validate_probs_and_labels(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    lbls = np.asarray(labels, dtype=np.int64)

    if probs.ndim != 2:
        raise ValueError(f"probabilities must be 2D [M, K], got shape {probs.shape}.")
    if lbls.ndim != 1:
        raise ValueError(f"labels must be 1D [M], got shape {lbls.shape}.")
    if probs.shape[0] != lbls.shape[0]:
        raise ValueError(
            f"probabilities batch size ({probs.shape[0]}) must match labels length ({lbls.shape[0]})."
        )
    m, k = probs.shape
    if k < 2:
        raise ValueError(f"probabilities must have at least 2 classes, got K={k}.")
    if m > 0:
        if np.any(lbls < 0) or np.any(lbls >= k):
            raise ValueError(f"labels must be in [0, {k - 1}], got min={lbls.min()}, max={lbls.max()}.")
    return probs, lbls


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int = 10,
) -> ReliabilityCurve:
    """Binned statistics for a reliability diagram using equal-width bins over [0, 1].

    Args:
        probabilities: ``[M, K]`` predicted class probabilities.
        labels: ``[M]`` ground-truth integer labels in ``[0, K - 1]``.
        num_bins: Number of equal-width confidence bins in ``[0, 1]``.

    Returns:
        :class:`ReliabilityCurve` containing per-bin mean confidence, mean accuracy,
        and sample counts. Empty bins have confidence and accuracy set to ``0.0``.
    """
    num_bins = check_int("num_bins", num_bins, minimum=1)
    probs, lbls = _validate_probs_and_labels(probabilities, labels)
    m = probs.shape[0]

    bin_confidence = np.zeros(num_bins, dtype=np.float64)
    bin_accuracy = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    if m == 0:
        return ReliabilityCurve(
            bin_confidence=bin_confidence,
            bin_accuracy=bin_accuracy,
            bin_counts=bin_counts,
        )

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == lbls).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for b in range(num_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        if b == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)

        count = int(np.sum(in_bin))
        bin_counts[b] = count
        if count > 0:
            bin_confidence[b] = float(np.mean(confidences[in_bin]))
            bin_accuracy[b] = float(np.mean(accuracies[in_bin]))

    return ReliabilityCurve(
        bin_confidence=bin_confidence,
        bin_accuracy=bin_accuracy,
        bin_counts=bin_counts,
    )


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) of ``[M, K]`` probabilities against ``[M]`` labels.

    Uses equal-width confidence binning over ``[0, 1]``::

        ECE = sum_{b=1}^{B} (n_b / M) * |acc(b) - conf(b)|
    """
    curve = reliability_curve(probabilities, labels, num_bins=num_bins)
    total = int(np.sum(curve.bin_counts))
    if total == 0:
        return 0.0

    weights = curve.bin_counts.astype(np.float64) / float(total)
    gaps = np.abs(curve.bin_accuracy - curve.bin_confidence)
    return float(np.sum(weights * gaps))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score of ``[M, K]`` probabilities against ``[M]`` labels.

    Defined as the mean squared error across samples of the predicted probability
    vector against the one-hot encoded ground-truth target::

        BS = (1 / M) * sum_{i=1}^{M} sum_{k=1}^{K} (p_{i,k} - y_{i,k})^2
    """
    probs, lbls = _validate_probs_and_labels(probabilities, labels)
    m, k = probs.shape
    if m == 0:
        return 0.0

    one_hot = np.zeros((m, k), dtype=np.float64)
    one_hot[np.arange(m), lbls] = 1.0
    sq_err = np.sum((probs - one_hot) ** 2, axis=1)
    return float(np.mean(sq_err))
