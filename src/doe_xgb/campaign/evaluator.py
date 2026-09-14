"""The campaign's real evaluation, and the worker setup that runs it in parallel.

One *real evaluation* is one stratified five-fold cross-validation of one
hyperparameter configuration on one dataset partition. It is the unit the budget
counts and the unit the cache keys.

Both cost quantities are persisted on every evaluation: total leaf count, which is
the frozen optimization objective, and measured wall-clock fit-and-predict time,
which is a secondary audit variable. See
``papers/xgboost_hpo_vrfnbi/protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md``.

This module lives under ``src`` rather than beside a script because process
workers are spawned, not forked: the child re-imports by module name, so the
worker entry points have to be importable.
"""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd

PARAMS = ["subsample", "colsample_bytree", "colsample_bylevel",
          "learning_rate", "max_depth", "gamma", "n_estimators"]
INT_PARAMS = frozenset({"max_depth", "n_estimators"})
BOUNDS: dict[str, tuple[float, float]] = {
    "subsample": (0.05, 1.0), "colsample_bytree": (0.05, 1.0),
    "colsample_bylevel": (0.05, 1.0), "learning_rate": (0.01, 0.30),
    "max_depth": (3, 18), "gamma": (0.05, 5.0), "n_estimators": (50, 700),
}

# Protocol section 6.2. Direction and transform are declared, never inferred.
RESPONSES: dict[str, dict[str, Any]] = {
    "Accuracy_Mean":    {"minimize": False, "role": "quality", "transform": "none"},
    "Precision_Mean":   {"minimize": False, "role": "quality", "transform": "none"},
    "Recall_Mean":      {"minimize": False, "role": "quality", "transform": "none"},
    "Specificity_Mean": {"minimize": False, "role": "quality", "transform": "none"},
    "RocAuc_Mean":      {"minimize": False, "role": "quality", "transform": "none"},
    "LogLoss_Mean":     {"minimize": True,  "role": "quality", "transform": "none"},
    "Leaves_Mean":      {"minimize": True,  "role": "cost",    "transform": "log1p"},
}

_STATE: dict[str, Any] = {}


def cast_config(row) -> dict[str, Any]:
    return {k: (int(round(float(row[k]))) if k in INT_PARAMS else float(row[k]))
            for k in PARAMS}


def prepare_dataset(dataset_id: str):
    """Load, one-hot encode any categorical columns, return float arrays."""
    from ..datasets.loaders import load
    d = load(dataset_id)
    X = d.X.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False)
    return X.astype(float).to_numpy(), np.asarray(d.y, dtype=int), len(cat)


def init_worker(dataset_id: str, threads: int, seed: int,
                test_size: float = 0.2) -> None:
    """Per-process setup: thread caps, dataset load, outer split, inner folds.

    The dataset is loaded once per process. Thread environment variables are set
    before the numerical libraries are first used, so that process-level and
    thread-level parallelism do not oversubscribe each other.
    """
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(threads)
    from sklearn.model_selection import StratifiedKFold, train_test_split
    X, y, n_cat = prepare_dataset(dataset_id)
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    _STATE.update(dataset=dataset_id, X=X_tr, y=y_tr, X_holdout=X_ho, y_holdout=y_ho,
                  threads=threads, seed=seed, n_categorical=n_cat,
                  kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed))


def evaluate_config(config: dict[str, Any], *, on_holdout: bool = False) -> dict[str, float]:
    """One real evaluation. Returns every response plus both cost quantities."""
    from sklearn.metrics import (accuracy_score, log_loss, precision_score,
                                 recall_score, roc_auc_score)
    from xgboost import XGBClassifier

    params = {k: (int(config[k]) if k in INT_PARAMS else float(config[k]))
              for k in PARAMS}
    seed, threads = _STATE["seed"], _STATE["threads"]
    if on_holdout:
        splits = [(np.arange(len(_STATE["y"])), None)]
    else:
        splits = list(_STATE["kf"].split(_STATE["X"], _STATE["y"]))

    acc, pre, rec, spe, auc, ll, leaves, times = ([] for _ in range(8))
    for tr, va in splits:
        Xf, yf = _STATE["X"], _STATE["y"]
        Xv, yv = (_STATE["X_holdout"], _STATE["y_holdout"]) if on_holdout else (Xf[va], yf[va])
        model = XGBClassifier(**params, eval_metric="logloss", verbosity=0,
                              tree_method="hist", n_jobs=threads, random_state=seed)
        t0 = time.perf_counter()
        model.fit(Xf[tr], yf[tr])
        prob = model.predict_proba(Xv)[:, 1]
        times.append(time.perf_counter() - t0)
        pred = (prob >= 0.5).astype(int)
        tn = int(((pred == 0) & (yv == 0)).sum())
        fp = int(((pred == 1) & (yv == 0)).sum())
        acc.append(accuracy_score(yv, pred))
        pre.append(precision_score(yv, pred, zero_division=0))
        rec.append(recall_score(yv, pred, zero_division=0))
        spe.append(tn / (tn + fp) if (tn + fp) else 0.0)
        auc.append(roc_auc_score(yv, prob))
        ll.append(log_loss(yv, np.clip(prob, 1e-7, 1 - 1e-7)))
        leaves.append(sum(s.count("leaf=") for s in model.get_booster().get_dump()))

    return {"Accuracy_Mean": float(np.mean(acc)), "Precision_Mean": float(np.mean(pre)),
            "Recall_Mean": float(np.mean(rec)), "Specificity_Mean": float(np.mean(spe)),
            "RocAuc_Mean": float(np.mean(auc)), "LogLoss_Mean": float(np.mean(ll)),
            "Leaves_Mean": float(np.mean(leaves)),
            "Time_MeanFold": float(np.mean(times))}


def evaluate_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pool entry point: takes a labelled configuration, returns labelled results."""
    out = evaluate_config(cast_config(row))
    out["label"] = row.get("label", "")
    return out


def worker_state() -> dict[str, Any]:
    """For tests and diagnostics; never given to an optimizer."""
    return {k: v for k, v in _STATE.items() if k not in ("X", "y", "X_holdout", "y_holdout")}


__all__ = ["PARAMS", "INT_PARAMS", "BOUNDS", "RESPONSES", "cast_config",
           "prepare_dataset", "init_worker", "evaluate_config", "evaluate_row",
           "worker_state"]
