"""OOD detection and selective-prediction metrics (placeholders, Phase 9).

Score convention: **higher score = more likely OOD / more uncertain**.
OOD events are the positive class unless stated otherwise.

Thresholds must never be tuned on the OOD test set; if a threshold is
needed, derive it from in-distribution validation data (or a separately
declared OOD validation set) and document how.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from seismic_ml.placeholders import NotYetImplementedError


class RiskCoverageCurve(NamedTuple):
    """Selective-prediction curve: keep the most confident fraction of samples.

    Attributes:
        coverage: Fraction of samples retained, ascending in ``(0, 1]``.
        risk: Error rate on the retained samples at each coverage.
    """

    coverage: np.ndarray
    risk: np.ndarray


def ood_auroc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """AUROC for separating OOD from in-distribution samples by score.

    TODO(Phase 9): implement.
    """
    raise NotYetImplementedError("ood_auroc", phase=9)


def ood_auprc(in_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """Area under the precision-recall curve with OOD as the positive class.

    TODO(Phase 9): implement.
    """
    raise NotYetImplementedError("ood_auprc", phase=9)


def ood_detection_rate(ood_scores: np.ndarray, *, threshold: float) -> float:
    """Fraction of OOD samples with score ``>= threshold``.

    TODO(Phase 9): implement; document how ``threshold`` was chosen.
    """
    raise NotYetImplementedError("ood_detection_rate", phase=9)


def risk_coverage_curve(uncertainty: np.ndarray, correct: np.ndarray) -> RiskCoverageCurve:
    """Risk-coverage curve from ``[M]`` uncertainty scores and ``[M]`` correctness flags.

    TODO(Phase 9): implement (also report the area under the curve).
    """
    raise NotYetImplementedError("risk_coverage_curve", phase=9)
