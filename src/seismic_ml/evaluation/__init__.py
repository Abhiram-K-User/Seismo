"""Evaluation interfaces: classification, calibration and OOD metrics."""

from seismic_ml.evaluation.calibration import (
    ReliabilityCurve,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from seismic_ml.evaluation.classification import (
    compute_classification_metrics,
    confusion_matrix,
)
from seismic_ml.evaluation.ood import (
    RiskCoverageCurve,
    ood_auprc,
    ood_auroc,
    ood_detection_rate,
    risk_coverage_curve,
)

__all__ = [
    "ReliabilityCurve",
    "RiskCoverageCurve",
    "brier_score",
    "compute_classification_metrics",
    "confusion_matrix",
    "expected_calibration_error",
    "ood_auprc",
    "ood_auroc",
    "ood_detection_rate",
    "reliability_curve",
    "risk_coverage_curve",
]
