"""Calibration metrics (placeholders, Phase 9).

May be pulled forward to Phase 3 to report calibration of the softmax
baseline; record that decision in the Decision Log if done.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from seismic_ml.placeholders import NotYetImplementedError


class ReliabilityCurve(NamedTuple):
    """Binned confidence vs accuracy statistics (all arrays have length ``num_bins``)."""

    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_counts: np.ndarray


def expected_calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int,
) -> float:
    """ECE of ``[M, K]`` predicted probabilities against ``[M]`` labels.

    The binning scheme (equal-width vs equal-mass) must be reported with the value.

    TODO(Phase 9): implement.
    """
    raise NotYetImplementedError("expected_calibration_error", phase=9)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score of ``[M, K]`` probabilities against ``[M]`` labels.

    TODO(Phase 9): implement.
    """
    raise NotYetImplementedError("brier_score", phase=9)


def reliability_curve(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    num_bins: int,
) -> ReliabilityCurve:
    """Binned statistics for a reliability diagram.

    TODO(Phase 9): implement (plotting lives in utils.visualization).
    """
    raise NotYetImplementedError("reliability_curve", phase=9)
