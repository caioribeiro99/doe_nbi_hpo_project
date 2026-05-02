"""Light fit/predict factories for the canary cell.

Each ``make_fit_predict(algorithm, params)`` returns a pair
``(model_fit, model_predict)`` of closures that the canary CV helper
calls per fold. Heavy imports happen inside the factory so the
methods package stays importable even when the GBDT libraries are
missing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def make_fit_predict(
    algorithm: str, params: dict[str, Any], *, seed: int,
) -> tuple[Callable[[np.ndarray, np.ndarray], Any],
           Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray | None]]]:
    if algorithm == "xgboost":
        return _xgb(params, seed)
    if algorithm == "lightgbm":
        return _lgb(params, seed)
    if algorithm == "catboost":
        return _cb(params, seed)
    raise ValueError(f"unknown algorithm: {algorithm!r}")


def _xgb(params: dict[str, Any], seed: int):
    import xgboost as xgb

    def fit(X, y):
        n_classes = int(np.unique(y).size)
        kwargs = dict(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            subsample=float(params.get("subsample", 1.0)),
            colsample_bytree=float(params.get("colsample_bytree", 1.0)),
            random_state=int(seed),
            n_jobs=1,
            verbosity=0,
        )
        if n_classes == 2:
            model = xgb.XGBClassifier(objective="binary:logistic", **kwargs)
        else:
            model = xgb.XGBClassifier(
                objective="multi:softprob", num_class=n_classes, **kwargs,
            )
        model.fit(X, y)
        return model

    def predict(model, X):
        ypred = model.predict(X)
        try:
            yprob = model.predict_proba(X)
        except Exception:
            yprob = None
        return np.asarray(ypred), (np.asarray(yprob) if yprob is not None else None)

    return fit, predict


def _lgb(params: dict[str, Any], seed: int):
    import lightgbm as lgb

    def fit(X, y):
        kwargs = dict(
            n_estimators=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            num_leaves=int(params.get("num_leaves", 31)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            random_state=int(seed),
            n_jobs=1,
            verbose=-1,
        )
        model = lgb.LGBMClassifier(**kwargs)
        model.fit(X, y)
        return model

    def predict(model, X):
        ypred = model.predict(X)
        try:
            yprob = model.predict_proba(X)
        except Exception:
            yprob = None
        return np.asarray(ypred), (np.asarray(yprob) if yprob is not None else None)

    return fit, predict


def _cb(params: dict[str, Any], seed: int):
    import catboost as cb

    def fit(X, y):
        kwargs = dict(
            iterations=int(params.get("n_estimators", 50)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            depth=int(params.get("depth", 6)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_seed=int(seed),
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
        model = cb.CatBoostClassifier(**kwargs)
        model.fit(X, y)
        return model

    def predict(model, X):
        ypred = model.predict(X).ravel()
        try:
            yprob = model.predict_proba(X)
        except Exception:
            yprob = None
        return np.asarray(ypred).astype(int), (np.asarray(yprob) if yprob is not None else None)

    return fit, predict
