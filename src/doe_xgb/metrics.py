"""Per-fold classification metrics.

Backward-compatible: the dissertation-era ``compute_binary_metrics`` and
``aggregate_fold_metrics`` keep their exact behavior. The post-dissertation
extension adds:

- :func:`compute_multiclass_metrics` returning F1-macro, balanced
  accuracy, MCC, and (when probabilities are supplied) macro one-vs-rest
  ROC-AUC / PR-AUC, multiclass Brier, and multiclass ECE.
- :func:`compute_classification_metrics` -- a unified dispatcher that
  picks the binary or multiclass path from the number of unique
  classes in ``y_true``.
- :func:`aggregate_metric_dicts` -- aggregates a list of fold metric
  dicts (handles ``np.nan`` cleanly).

Metrics that cannot be computed safely on a given input return
``np.nan`` and add a free-form note to a per-fold ``warnings`` list,
rather than raising.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TaskType = Literal["binary", "multiclass", "auto"]


# ---------------------------------------------------------------------------
# Binary path (backward-compatible)
# ---------------------------------------------------------------------------


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
        # Fallback for edge cases (should be rare with stratified CV).
        tn = cm[0, 0] if cm.shape[0] > 0 and cm.shape[1] > 0 else 0
        fp = cm[0, 1] if cm.shape[0] > 0 and cm.shape[1] > 1 else 0
    else:
        tn, fp, _fn, _tp = cm.ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return FoldMetrics(accuracy=acc, precision=prec, recall=rec, specificity=spec)


def aggregate_fold_metrics(folds: tuple[FoldMetrics, ...]) -> dict[str, float]:
    """Aggregate binary fold metrics (mean) -- dissertation-era keys."""
    return {
        "Accuracy_Mean": float(np.mean([f.accuracy for f in folds])),
        "Precision_Mean": float(np.mean([f.precision for f in folds])),
        "Recall_Mean": float(np.mean([f.recall for f in folds])),
        "Specificity_Mean": float(np.mean([f.specificity for f in folds])),
    }


# ---------------------------------------------------------------------------
# Multiclass path
# ---------------------------------------------------------------------------


def _ece_multiclass(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Multiclass expected calibration error (top-label confidence binning).

    Bins the per-row max probability into ``n_bins`` equal-width bins;
    in each bin, ``|accuracy - mean(confidence)|`` weighted by the bin's
    fraction. Returns NaN when all probabilities are degenerate.
    """
    proba = np.asarray(y_prob, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(y_true):
        return float("nan")
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    correct = (predictions == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    if n == 0:
        return float("nan")
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        w = float(np.sum(mask)) / float(n)
        gap = float(abs(np.mean(correct[mask]) - np.mean(confidences[mask])))
        ece += w * gap
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """Mean squared error between one-hot true labels and predicted probs.

    Sums across classes; matches the standard 'sum-of-squares' multiclass
    Brier score (= sklearn binary Brier when ``n_classes==2`` and
    one-hot encodings match).
    """
    proba = np.asarray(y_prob, dtype=float)
    if proba.ndim != 2 or proba.shape != (len(y_true), n_classes):
        return float("nan")
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), np.asarray(y_true).astype(int)] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Per-fold multiclass classification metrics.

    Returns a dict with the following keys (any may be ``np.nan`` when
    the metric cannot be computed safely):

    - ``accuracy`` (helpful for quick comparisons)
    - ``f1_macro``
    - ``balanced_accuracy``
    - ``mcc``
    - ``roc_auc_ovr_macro`` (requires ``y_prob``)
    - ``pr_auc_ovr_macro`` (requires ``y_prob``)
    - ``brier_multiclass`` (requires ``y_prob``)
    - ``ece_multiclass`` (requires ``y_prob``)

    Always includes a ``warnings`` list of free-form strings recording
    which metrics were skipped and why.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    warnings: list[str] = []

    if labels is None:
        unique = np.unique(np.concatenate([yt, yp]))
        labels = sorted(int(c) for c in unique)
    else:
        labels = list(labels)

    accuracy = float(accuracy_score(yt, yp))
    try:
        f1m = float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0))
    except Exception as e:  # pragma: no cover
        warnings.append(f"f1_macro_failed: {e}")
        f1m = float("nan")
    try:
        bacc = float(balanced_accuracy_score(yt, yp))
    except Exception as e:  # pragma: no cover
        warnings.append(f"balanced_accuracy_failed: {e}")
        bacc = float("nan")
    try:
        mcc = float(matthews_corrcoef(yt, yp))
    except Exception as e:  # pragma: no cover
        warnings.append(f"mcc_failed: {e}")
        mcc = float("nan")

    out: dict[str, Any] = {
        "accuracy": accuracy,
        "f1_macro": f1m,
        "balanced_accuracy": bacc,
        "mcc": mcc,
        "roc_auc_ovr_macro": float("nan"),
        "pr_auc_ovr_macro": float("nan"),
        "brier_multiclass": float("nan"),
        "ece_multiclass": float("nan"),
        "warnings": warnings,
    }

    if y_prob is None:
        warnings.append("y_prob_not_supplied: probability metrics skipped")
        return out

    proba = np.asarray(y_prob, dtype=float)
    if proba.ndim != 2:
        warnings.append("y_prob_must_be_2d: probability metrics skipped")
        return out
    n_classes = len(labels)
    if proba.shape[1] != n_classes:
        warnings.append(
            f"y_prob_class_count_mismatch (got {proba.shape[1]}, expected {n_classes}): "
            "probability metrics skipped"
        )
        return out

    try:
        out["roc_auc_ovr_macro"] = float(
            roc_auc_score(yt, proba, multi_class="ovr", average="macro", labels=labels)
        )
    except Exception as e:
        warnings.append(f"roc_auc_ovr_macro_failed: {e}")

    try:
        # one-hot vs proba
        onehot = np.zeros_like(proba)
        for j, lab in enumerate(labels):
            onehot[:, j] = (yt == lab).astype(float)
        out["pr_auc_ovr_macro"] = float(
            average_precision_score(onehot, proba, average="macro")
        )
    except Exception as e:
        warnings.append(f"pr_auc_ovr_macro_failed: {e}")

    out["brier_multiclass"] = _brier_multiclass(yt, proba, n_classes=n_classes)
    out["ece_multiclass"] = _ece_multiclass(yt, proba)

    return out


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def _infer_task(y_true: np.ndarray) -> TaskType:
    n = int(len(np.unique(np.asarray(y_true))))
    return "binary" if n <= 2 else "multiclass"


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
    task_type: TaskType = "auto",
) -> dict[str, Any]:
    """Unified per-fold classification metrics.

    Returns a flat ``dict`` with the binary keys
    (``accuracy``, ``precision``, ``recall``, ``specificity``) when
    ``task_type`` resolves to ``binary``, or the multiclass keys
    (``accuracy``, ``f1_macro``, ``balanced_accuracy``, ``mcc``,
    ``roc_auc_ovr_macro``, ``pr_auc_ovr_macro``, ``brier_multiclass``,
    ``ece_multiclass``) when it resolves to ``multiclass``. A ``task``
    tag is always present.

    The ``y_prob`` argument is silently ignored on the binary path for
    backward compatibility; pass it only when reporting multiclass
    probability-based metrics.
    """
    resolved = task_type if task_type != "auto" else _infer_task(y_true)
    if resolved == "binary":
        fold = compute_binary_metrics(y_true, y_pred)
        out: dict[str, Any] = {
            "task": "binary",
            "accuracy": fold.accuracy,
            "precision": fold.precision,
            "recall": fold.recall,
            "specificity": fold.specificity,
            "warnings": [],
        }
        return out
    out_m = compute_multiclass_metrics(y_true, y_pred, y_prob=y_prob, labels=labels)
    out_m["task"] = "multiclass"
    return out_m


# ---------------------------------------------------------------------------
# Aggregation across folds (dict-based; supports both binary and multiclass)
# ---------------------------------------------------------------------------


_BINARY_KEY_MAP = {
    "accuracy": "Accuracy_Mean",
    "precision": "Precision_Mean",
    "recall": "Recall_Mean",
    "specificity": "Specificity_Mean",
}

_MULTICLASS_KEY_MAP = {
    "accuracy": "Accuracy_Mean",
    "f1_macro": "F1Macro_Mean",
    "balanced_accuracy": "BalancedAccuracy_Mean",
    "mcc": "MCC_Mean",
    "roc_auc_ovr_macro": "ROCAUC_OVR_Mean",
    "pr_auc_ovr_macro": "PRAUC_OVR_Mean",
    "brier_multiclass": "BrierMC_Mean",
    "ece_multiclass": "ECE_Mean",
}


def aggregate_metric_dicts(folds: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-fold metric dicts into mean keys.

    - Honours both the binary ``FoldMetrics`` shape (post-dispatch dict)
      and the multiclass shape.
    - ``np.nan`` values are skipped via ``np.nanmean`` so a single broken
      fold does not poison the mean.
    """
    folds_list = list(folds)
    if not folds_list:
        return {}
    task = folds_list[0].get("task")
    key_map = _BINARY_KEY_MAP if task == "binary" else _MULTICLASS_KEY_MAP
    out: dict[str, float] = {}
    for src, dst in key_map.items():
        vals = [float(d.get(src, float("nan"))) for d in folds_list]
        arr = np.asarray(vals, dtype=float)
        if np.all(np.isnan(arr)):
            out[dst] = float("nan")
        else:
            out[dst] = float(np.nanmean(arr))
    return out


__all__ = [
    "FoldMetrics",
    "TaskType",
    "aggregate_fold_metrics",
    "aggregate_metric_dicts",
    "compute_binary_metrics",
    "compute_classification_metrics",
    "compute_multiclass_metrics",
]
