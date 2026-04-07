from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import FAIRNESS_DEFAULTS, FAIRNESS_PARAM_NAMES, INT_PARAMS
from .evaluation_fairness import evaluate_fold_fairness, summarize_folds


def _infer_scale_pos_weight(y: np.ndarray) -> float:
    """Infer scale_pos_weight = (neg/pos) from a binary label vector."""
    y = y.astype(int)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def _build_stratify_labels(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Joint stratification labels for (y, protected)."""
    y = y.astype(int)
    g = group.astype(int)
    return (y * 2 + g).astype(int)


def _as_int(v: Any, default: int) -> int:
    try:
        if v is None:
            return int(default)
        vf = float(v)
        if np.isnan(vf):
            return int(default)
        return int(round(vf))
    except Exception:
        return int(default)


def _as_float(v: Any, default: float) -> float:
    try:
        if v is None:
            return float(default)
        vf = float(v)
        if np.isnan(vf):
            return float(default)
        return float(vf)
    except Exception:
        return float(default)


def _xgb_classifier(params: Dict[str, Any], seed: int):
    """Lazy import XGBoost and build an XGBClassifier."""
    import xgboost as xgb  # type: ignore

    fixed = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=params.get("tree_method", "hist"),
        n_estimators=_as_int(params.get("n_estimators"), 300),
        learning_rate=_as_float(params.get("learning_rate"), 0.1),
        max_depth=_as_int(params.get("max_depth"), 5),
        min_child_weight=_as_float(params.get("min_child_weight"), 1.0),
        gamma=_as_float(params.get("gamma"), 0.0),
        subsample=_as_float(params.get("subsample"), 1.0),
        colsample_bytree=_as_float(params.get("colsample_bytree"), 1.0),
        colsample_bylevel=_as_float(params.get("colsample_bylevel"), 1.0),
        reg_alpha=_as_float(params.get("reg_alpha"), 0.0),
        reg_lambda=_as_float(params.get("reg_lambda"), 1.0),
        n_jobs=_as_int(params.get("n_jobs"), -1),
        random_state=int(seed),
    )

    spw = params.get("scale_pos_weight", None)
    if spw is not None:
        spw_f = _as_float(spw, float(FAIRNESS_DEFAULTS["scale_pos_weight"]))
        fixed["scale_pos_weight"] = float(max(0.0, spw_f))

    return xgb.XGBClassifier(**fixed)


def _progress(desc: str, i: int, total: int, every: int) -> None:
    if i == 1 or i == total or (every > 0 and i % every == 0):
        print(f"{desc}: {i}/{total}", flush=True)


def run_doe_fairness(
    design_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    protected: pd.Series,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
    auto_scale_pos_weight: bool = True,
    stratify_by_group: bool = False,
    privileged_value: int = 1,
    desc: str = "DOE runs (fairness)",
    progress_every: int | None = None,
) -> pd.DataFrame:
    """Execute DoE points and return a results dataframe."""
    y_np = y.astype(int).to_numpy()
    prot_np = protected.astype(int).to_numpy()

    strat = _build_stratify_labels(y_np, prot_np) if stratify_by_group else y_np
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

    rows: List[Dict[str, Any]] = []
    total = int(len(design_df))
    if progress_every is None:
        progress_every = max(1, total // 20)

    for i, (_, row) in enumerate(design_df.iterrows(), start=1):
        _progress(desc, i, total, progress_every)

        params: Dict[str, Any] = {}
        for p in FAIRNESS_PARAM_NAMES:
            if p in row.index:
                params[p] = row.get(p)

        params["n_jobs"] = int(n_jobs)
        params["tree_method"] = str(tree_method)

        thr = _as_float(params.get("threshold"), float(FAIRNESS_DEFAULTS["threshold"]))
        thr = float(np.clip(thr, 0.01, 0.99))
        params["threshold"] = thr

        spw_design = None
        if "scale_pos_weight" in params:
            spw_val = _as_float(params.get("scale_pos_weight"), float("nan"))
            if not np.isnan(spw_val):
                spw_design = float(max(0.0, spw_val))

        fold_metrics = []
        fold_times: List[float] = []
        spw_used: List[float] = []

        for fold, (tr, te) in enumerate(skf.split(X, strat), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            pte = protected.iloc[te]

            params_fold = dict(params)
            if spw_design is not None:
                params_fold["scale_pos_weight"] = float(spw_design)
                spw_used.append(float(spw_design))
            elif auto_scale_pos_weight:
                spw_auto = float(_infer_scale_pos_weight(ytr.to_numpy()))
                params_fold["scale_pos_weight"] = float(spw_auto)
                spw_used.append(float(spw_auto))
            else:
                spw_used.append(1.0)

            clf = _xgb_classifier(params_fold, seed=int(seed) + fold)
            t0 = time.perf_counter()
            clf.fit(Xtr, ytr)
            proba = clf.predict_proba(Xte)[:, 1]
            fold_elapsed = float(time.perf_counter() - t0)
            fold_times.append(fold_elapsed)
            yhat = (proba >= thr).astype(int)

            fold_metrics.append(
                evaluate_fold_fairness(
                    y_true=yte.to_numpy(),
                    y_pred=yhat,
                    protected=pte.to_numpy(),
                    privileged_value=privileged_value,
                )
            )

        agg = summarize_folds(fold_metrics)
        out: Dict[str, Any] = {}
        out.update(agg)

        spw_mean = float(np.mean(spw_used)) if spw_used else float(FAIRNESS_DEFAULTS["scale_pos_weight"])
        for p in FAIRNESS_PARAM_NAMES:
            if p == "scale_pos_weight":
                out[p] = float(spw_design) if spw_design is not None else spw_mean
            elif p == "threshold":
                out[p] = float(thr)
            elif p in INT_PARAMS:
                out[p] = _as_int(params.get(p), 0) if p in params else np.nan
            else:
                out[p] = _as_float(params.get(p), float("nan")) if p in params else np.nan

        out["ScalePosWeight_MeanFold"] = float(spw_mean)
        out["auto_scale_pos_weight"] = bool(auto_scale_pos_weight)
        out["stratify_by_group"] = bool(stratify_by_group)
        out["Time_MeanFold"] = float(np.mean(fold_times)) if fold_times else np.nan
        out["Time_TotalCV"] = float(np.sum(fold_times)) if fold_times else np.nan
        out["n_splits"] = int(n_splits)

        rows.append(out)

    _progress(desc, total, total, progress_every)
    return pd.DataFrame(rows)
