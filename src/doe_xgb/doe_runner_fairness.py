from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import PARAM_NAMES
from .evaluation_fairness import evaluate_fold_fairness, summarize_folds


def _infer_scale_pos_weight(y: np.ndarray) -> float:
    y = y.astype(int)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos <= 0:
        return 1.0
    return float(neg / pos)


def _build_stratify_labels(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    # y in {0,1}, group in {0,1}
    y = y.astype(int)
    g = group.astype(int)
    return (y * 2 + g).astype(int)


def _xgb_classifier(params: Dict[str, Any], seed: int):
    """
    Lazily import xgboost only when needed.
    """
    import xgboost as xgb  # type: ignore

    fixed = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=params.get("tree_method", "hist"),
        n_estimators=int(params.get("n_estimators", 300)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 5)),
        min_child_weight=float(params.get("min_child_weight", 1.0)),
        gamma=float(params.get("gamma", 0.0)),
        subsample=float(params.get("subsample", 1.0)),
        colsample_bytree=float(params.get("colsample_bytree", 1.0)),
        colsample_bylevel=float(params.get("colsample_bylevel", 1.0)),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 1.0)),
        n_jobs=int(params.get("n_jobs", -1)),
        random_state=int(seed),
    )

    # optional: scale_pos_weight may be injected per fold
    if "scale_pos_weight" in params and params["scale_pos_weight"] is not None:
        fixed["scale_pos_weight"] = float(params["scale_pos_weight"])

    return xgb.XGBClassifier(**fixed)


def _progress(desc: str, i: int, total: int, every: int) -> None:
    """Minimal progress indicator (no tqdm)."""
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
    """
    Executes DoE points and returns a results dataframe where each row is a configuration
    evaluated by CV. Adds fairness metrics and hyperparameter columns.

    Progress:
      Prints simple progress (no tqdm) every ~5% by default.
    """
    y_np = y.astype(int).to_numpy()
    prot_np = protected.astype(int).to_numpy()

    # Define CV split
    if stratify_by_group:
        strat = _build_stratify_labels(y_np, prot_np)
    else:
        strat = y_np

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))

    rows: list[dict[str, Any]] = []

    total = int(len(design_df))
    if progress_every is None:
        progress_every = max(1, total // 20)  # ~5% updates

    for i, (_, row) in enumerate(design_df.iterrows(), start=1):
        _progress(desc, i, total, progress_every)

        params: Dict[str, Any] = {p: row.get(p) for p in PARAM_NAMES if p in row.index}

        # defaults
        params["n_jobs"] = n_jobs
        params["tree_method"] = tree_method

        fold_metrics = []
        for fold, (tr, te) in enumerate(skf.split(X, strat), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y.iloc[tr], y.iloc[te]
            pte = protected.iloc[te]

            # scale_pos_weight: computed on training fold to deal with imbalance
            if auto_scale_pos_weight:
                params_fold = dict(params)
                params_fold["scale_pos_weight"] = _infer_scale_pos_weight(ytr.to_numpy())
            else:
                params_fold = dict(params)

            clf = _xgb_classifier(params_fold, seed=seed + fold)
            clf.fit(Xtr, ytr)

            yhat = (clf.predict_proba(Xte)[:, 1] >= 0.5).astype(int)
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
        out.update({p: params.get(p) for p in PARAM_NAMES})
        out["auto_scale_pos_weight"] = bool(auto_scale_pos_weight)
        out["stratify_by_group"] = bool(stratify_by_group)

        rows.append(out)

    # final line (guarantee you see "done")
    _progress(desc, total, total, progress_every)
    return pd.DataFrame(rows)
