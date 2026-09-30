"""Classification metrics (placeholders, Phase 3).

Planned: accuracy, balanced accuracy, per-class precision/recall/F1,
macro-F1 and the confusion matrix. Per-class and macro metrics matter
because the class distribution is unknown and possibly imbalanced.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from seismic_ml.placeholders import NotYetImplementedError


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Compute classification metrics from ``[M]`` true and predicted class indices.

    Returns:
        Flat mapping, e.g. ``{"accuracy": ..., "macro_f1": ..., "recall/earthquake": ...}``.

    TODO(Phase 3): implement (scikit-learn is the likely dependency; add it then).
    """
    raise NotYetImplementedError("compute_classification_metrics", phase=3)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``[K, K]`` counts; rows = true class, columns = predicted class.

    TODO(Phase 3): implement.
    """
    raise NotYetImplementedError("confusion_matrix", phase=3)
