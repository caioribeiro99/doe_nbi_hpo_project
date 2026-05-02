from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .config import INT_PARAMS
from .metrics import (
    TaskType,
    aggregate_metric_dicts,
    compute_classification_metrics,
)


def _cast_params(params: dict) -> dict:
    """Round/cast integer hyperparameters consistently."""
    out = dict(params)
    for k in list(out.keys()):
        if k in INT_PARAMS:
            out[k] = int(round(float(out[k])))
        else:
            out[k] = float(out[k])
    return out


@dataclass(frozen=True)
class EvalResult:
    metrics: dict[str, float]
    time_mean_fold: float
    params: dict
    task: Literal["binary", "multiclass"] = "binary"

    def as_dict(self) -> dict[str, float | dict | str]:
        """Flatten evaluation outputs to a single dict.

        This helper exists so downstream modules (benchmarks, scripts) can
        treat CV outputs uniformly.
        """
        out: dict[str, float | dict | str] = dict(self.metrics)
        out["Time_MeanFold"] = float(self.time_mean_fold)
        out["task"] = self.task
        return out


def evaluate_xgb_cv(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    kfold: StratifiedKFold,
    *,
    seed: int,
    n_jobs: int = -1,
    tree_method: str = "hist",
    eval_metric: str = "logloss",
    measure: str = "fit_predict",
    task_type: TaskType = "auto",
) -> EvalResult:
    """Evaluate one hyperparameter set with Stratified K-Fold CV.

    Time is measured as **mean per fold** using ``time.perf_counter()``.

    Backward-compatible: when ``task_type`` resolves to ``binary``, the
    output ``metrics`` carries the dissertation-era keys
    (``Accuracy_Mean / Precision_Mean / Recall_Mean / Specificity_Mean``).
    On a multiclass target the output carries
    ``Accuracy_Mean / F1Macro_Mean / BalancedAccuracy_Mean / MCC_Mean
    / ROCAUC_OVR_Mean / PRAUC_OVR_Mean / BrierMC_Mean / ECE_Mean``
    (NaN where ``y_prob`` is not available or the metric is undefined).

    Parameters
    ----------
    measure:
        ``"fit"`` or ``"fit_predict"``. Default: fit + predict.
    task_type:
        ``"auto"`` (default) infers from the number of unique classes
        in ``y``. Passing ``"binary"`` or ``"multiclass"`` forces the
        path explicitly.
    """
    p = _cast_params(params)

    n_classes_total = int(len(np.unique(np.asarray(y))))
    resolved_task: Literal["binary", "multiclass"]
    if task_type == "auto":
        resolved_task = "binary" if n_classes_total <= 2 else "multiclass"
    else:
        resolved_task = task_type  # type: ignore[assignment]

    folds: list[dict[str, Any]] = []
    fold_times: list[float] = []

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(
            **p,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method=tree_method,
            eval_metric=eval_metric,
            verbosity=0,
        )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob: np.ndarray | None = None
        if resolved_task == "multiclass":
            try:
                y_prob = np.asarray(model.predict_proba(X_test), dtype=float)
            except Exception:  # pragma: no cover - defensive
                y_prob = None
        t1 = time.perf_counter()
        fold_times.append(t1 - t0)

        fold_metrics = compute_classification_metrics(
            y_test,
            y_pred,
            y_prob=y_prob,
            task_type=resolved_task,
        )
        folds.append(fold_metrics)

    fold_warnings = [w for f in folds for w in (f.get("warnings") or [])]
    if fold_warnings:
        warnings.warn(
            f"evaluate_xgb_cv: {len(fold_warnings)} fold-metric warning(s) (first: "
            f"{fold_warnings[0]!r})",
            RuntimeWarning,
            stacklevel=2,
        )

    agg = aggregate_metric_dicts(folds)
    return EvalResult(
        metrics=agg,
        time_mean_fold=float(np.mean(fold_times)),
        params=p,
        task=resolved_task,
    )


# ---------------------------------------------------------------------------
# Article-track guardrail
# ---------------------------------------------------------------------------


_BINARY_FA_DEFAULTS = {
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "Specificity_Mean",
    "Time_MeanFold",
}


class MultiClassNotConfiguredError(RuntimeError):
    """Raised when a multiclass dataset is fed into a pipeline that has
    not been explicitly reconfigured for multiclass response handling.

    Carries hints in ``.required_metric_set`` for the caller to resolve.
    """

    def __init__(self, dataset_id: str, hint: str | None = None) -> None:
        super().__init__(
            f"Dataset {dataset_id!r} is multiclass but the pipeline is using the "
            "binary FA / NBI response defaults. Configure the multiclass response "
            "set ({F1Macro_Mean, BalancedAccuracy_Mean, MCC_Mean, ROCAUC_OVR_Mean, "
            "PRAUC_OVR_Mean, BrierMC_Mean, ECE_Mean, Time_MeanFold}) and re-run."
        )
        self.dataset_id = dataset_id
        self.hint = hint
        self.required_metric_set = (
            "F1Macro_Mean",
            "BalancedAccuracy_Mean",
            "MCC_Mean",
            "ROCAUC_OVR_Mean",
            "PRAUC_OVR_Mean",
            "BrierMC_Mean",
            "ECE_Mean",
            "Time_MeanFold",
        )


def assert_metric_set_compatible_with_task(
    *,
    dataset_id: str,
    task: Literal["binary", "multiclass"],
    fa_metrics: tuple[str, ...] | list[str],
) -> None:
    """Raise if a multiclass task is paired with the dissertation-era
    binary FA / NBI response defaults.

    Call this from the orchestrator before running the FA/NBI stages on
    Dry Bean (or any other multiclass dataset). The check is intentional
    in the article track so that v1 headline tables never silently mix
    binary and multiclass response sets.
    """
    if task == "multiclass":
        used = set(fa_metrics)
        if used <= _BINARY_FA_DEFAULTS:
            raise MultiClassNotConfiguredError(dataset_id)
