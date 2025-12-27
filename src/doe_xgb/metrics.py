from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


@dataclass(frozen=True)
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    specificity: float


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    """Compute accuracy, precision, recall and specificity for binary classification."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        # fallback for edge cases (should be rare with stratified CV)
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    else:
        tn, fp, fn, tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return FoldMetrics(accuracy=acc, precision=prec, recall=rec, specificity=spec)


def aggregate_fold_metrics(folds: Tuple[FoldMetrics, ...]) -> Dict[str, float]:
    """Aggregate fold metrics (mean)."""
    return {
        "Accuracy_Mean": float(np.mean([f.accuracy for f in folds])),
        "Precision_Mean": float(np.mean([f.precision for f in folds])),
        "Recall_Mean": float(np.mean([f.recall for f in folds])),
        "Specificity_Mean": float(np.mean([f.specificity for f in folds])),
    }
