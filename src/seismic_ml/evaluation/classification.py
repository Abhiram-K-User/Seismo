"""Classification metrics (Phase 3).

Computes accuracy, balanced accuracy, per-class precision/recall/F1,
macro-F1 and the confusion matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute confusion matrix of shape ``[num_classes, num_classes]``.

    Rows = ground truth class, columns = predicted class.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Compute classification metrics from ``[M]`` true and predicted class indices.

    Args:
        y_true: Array of true class integers.
        y_pred: Array of predicted class integers.
        class_names: Sequence of class names in index order.

    Returns:
        Flat mapping of metrics: accuracy, macro_f1, balanced_accuracy,
        and per-class precision/recall/f1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    k = len(class_names)

    if len(y_true) == 0:
        metrics: dict[str, float] = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "balanced_accuracy": 0.0,
        }
        for name in class_names:
            metrics[f"precision/{name}"] = 0.0
            metrics[f"recall/{name}"] = 0.0
            metrics[f"f1/{name}"] = 0.0
        return metrics

    acc = float((y_true == y_pred).mean())
    precisions = []
    recalls = []
    f1s = []

    metrics = {"accuracy": acc}

    for idx, name in enumerate(class_names):
        tp = int(np.sum((y_pred == idx) & (y_true == idx)))
        fp = int(np.sum((y_pred == idx) & (y_true != idx)))
        fn = int(np.sum((y_pred != idx) & (y_true == idx)))

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        metrics[f"precision/{name}"] = prec
        metrics[f"recall/{name}"] = rec
        metrics[f"f1/{name}"] = f1

    macro_f1 = float(np.mean(f1s))
    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))

    metrics["macro_f1"] = macro_f1
    metrics["macro_precision"] = macro_prec
    metrics["macro_recall"] = macro_rec
    metrics["balanced_accuracy"] = macro_rec  # Balanced accuracy is macro-averaged recall

    return metrics
