from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import time

from .config import INT_PARAMS
from .metrics import compute_binary_metrics, aggregate_fold_metrics, FoldMetrics


def _cast_params(params: Dict) -> Dict:
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
    metrics: Dict[str, float]
    time_mean_fold: float
    params: Dict


def evaluate_xgb_cv(
    params: Dict,
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
