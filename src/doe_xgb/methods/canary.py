"""Canary execution interface for the first executable CC18 adapters.

This module is intentionally small and dependency-light. It is the
contract between the runner skeleton and the four canary adapters
(``default_gbdt``, ``random_search``, ``tpe_optuna``,
``doe_rsm_vrf_true_nbi``). Real benchmark execution still runs through
the existing ``doe_xgb.evaluation`` / ``doe_xgb.metrics`` modules; the
helpers here are just glue that lets an adapter receive ``(X, y)`` and
return a uniform ``CanaryResult``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import compute_classification_metrics


@dataclass
class TaskData:
    """Materialized synthetic or OpenML task ready to feed an estimator."""

    X: np.ndarray
    y: np.ndarray
    task_type: str  # "binary" | "multiclass"
    n_classes: int
    feature_names: tuple[str, ...] = ()


@dataclass
class MethodRunContext:
    """All inputs an adapter's run() needs that are not (X, y)."""

    method_id: str
    algorithm: str  # xgboost | lightgbm | catboost
    replica: int
    seed: int
    n_folds: int = 2
    max_evaluations: int = 5
    output_dir: Path | None = None
    config_path: str | None = None
    task_id: int | None = None
    dataset_name: str = ""

    def derived_seed(self, salt: int = 0) -> int:
        """Replica-stable derived seed for sub-routines (CV split,
        sampler init, etc.)."""
        return int(self.seed) * 10_000 + int(self.replica) * 17 + int(salt)


@dataclass
class CanaryResult:
    """Uniform return value across all canary adapters."""

    method_id: str
    algorithm: str
    replica: int
    task_type: str
    n_folds: int
    fold_metrics: list[dict[str, Any]]  # one dict per fold
    aggregate_metrics: dict[str, float]
    best_config: dict[str, Any] | None = None
    n_configurations_tried: int = 1
    runtime_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "algorithm": self.algorithm,
            "replica": self.replica,
            "task_type": self.task_type,
            "n_folds": self.n_folds,
            "n_configurations_tried": self.n_configurations_tried,
            "runtime_seconds": self.runtime_seconds,
            "best_config": self.best_config,
            "aggregate_metrics": self.aggregate_metrics,
            "notes": self.notes,
            **{k: v for k, v in self.extra.items() if k != "fold_metrics"},
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Reusable CV helper
# ---------------------------------------------------------------------------


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Tiny wrapper around scikit-learn's StratifiedKFold so adapters do
    not import sklearn directly."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def evaluate_fitted_model(
    *, model_fit: Callable[[np.ndarray, np.ndarray], Any],
    model_predict: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray | None]],
    X: np.ndarray, y: np.ndarray, task_type: str, n_folds: int, seed: int,
) -> tuple[list[dict[str, Any]], float]:
    """Run a stratified K-fold CV by calling user-supplied fit/predict
    closures. Returns per-fold metric dicts + total runtime."""
    folds = stratified_folds(y, n_folds=n_folds, seed=seed)
    fold_metrics: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        model = model_fit(Xtr, ytr)
        ypred, yprob = model_predict(model, Xte)
        m = compute_classification_metrics(
            y_true=yte, y_pred=ypred, y_prob=yprob, task_type=task_type,
        )
        m["fold"] = fold_idx
        fold_metrics.append(m)
    return fold_metrics, time.perf_counter() - t0


def aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    """Mean across folds for any numeric metric (excluding the ``fold``
    index, the ``warnings`` list, and the ``task`` tag)."""
    keys: set[str] = set()
    for fm in fold_metrics:
        for k, v in fm.items():
            if isinstance(v, (int, float)) and k != "fold":
                keys.add(k)
    out: dict[str, float] = {}
    for k in sorted(keys):
        vals = [float(fm[k]) for fm in fold_metrics
                if k in fm and isinstance(fm[k], (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------------------
# Synthetic task helpers (used by the canary tests)
# ---------------------------------------------------------------------------


def make_synthetic_binary_task(
    *, n_samples: int = 300, n_features: int = 6, seed: int = 0,
) -> TaskData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    coefs = rng.standard_normal(n_features)
    logits = X @ coefs
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n_samples) < p).astype(int)
    if int(np.unique(y).size) < 2:
        y[:n_samples // 2] = 0
        y[n_samples // 2:] = 1
    return TaskData(
        X=X, y=y, task_type="binary", n_classes=2,
        feature_names=tuple(f"x{i}" for i in range(n_features)),
    )


# ---------------------------------------------------------------------------
# Tiny canary search space (uniform over a small box per algorithm)
# ---------------------------------------------------------------------------


CANARY_SEARCH_SPACE: dict[str, dict[str, tuple[float, float] | tuple[int, int]]] = {
    "xgboost": {
        "learning_rate": (0.05, 0.3),
        "max_depth": (3, 8),
        "subsample": (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
    },
    "lightgbm": {
        "learning_rate": (0.05, 0.3),
        "num_leaves": (15, 63),
        "min_child_samples": (5, 50),
    },
    "catboost": {
        "learning_rate": (0.05, 0.3),
        "depth": (3, 8),
        "l2_leaf_reg": (1.0, 5.0),
    },
}


def sample_canary_config(rng: np.random.Generator, algorithm: str) -> dict[str, Any]:
    space = CANARY_SEARCH_SPACE[algorithm]
    cfg: dict[str, Any] = {}
    for name, (lo, hi) in space.items():
        if isinstance(lo, int) and isinstance(hi, int):
            cfg[name] = int(rng.integers(low=lo, high=hi + 1))
        else:
            cfg[name] = float(rng.uniform(low=lo, high=hi))
    return cfg
