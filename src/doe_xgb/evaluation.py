from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .config import INT_PARAMS
from .metrics import FoldMetrics, aggregate_fold_metrics, compute_binary_metrics


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

    def as_dict(self) -> dict[str, float | dict]:
        """Flatten evaluation outputs to a single dict.

        This helper exists so downstream modules (benchmarks, scripts) can
        treat CV outputs uniformly.
        """
        out: dict[str, float | dict] = dict(self.metrics)
        out["Time_MeanFold"] = float(self.time_mean_fold)
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
) -> EvalResult:
    """Evaluate one hyperparameter set with Stratified K-Fold CV.

    Time is measured as **mean per fold** using `time.perf_counter()`.

    Parameters
    ----------
    params:
        Hyperparameters for XGBClassifier.
    measure:
        "fit" or "fit_predict". Default: fit+predict.
    """
    p = _cast_params(params)

    folds: list[FoldMetrics] = []
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
        if measure == "fit_predict":
            y_pred = model.predict(X_test)
        else:
            # still need predictions for metrics
            y_pred = model.predict(X_test)
        t1 = time.perf_counter()
        fold_times.append(t1 - t0)

        folds.append(compute_binary_metrics(y_test, y_pred))

    agg = aggregate_fold_metrics(tuple(folds))
    return EvalResult(metrics=agg, time_mean_fold=float(np.mean(fold_times)), params=p)
