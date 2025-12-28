from __future__ import annotations

import ast
import time
import itertools
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from tqdm import tqdm

from .config import DEFAULT_BOUNDS, INT_PARAMS, PARAM_NAMES
from .evaluation import evaluate_xgb_cv
from .io_utils import save_csv_ptbr


def _cast_ints(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(params)
    for k in list(out.keys()):
        if k in INT_PARAMS:
            out[k] = int(round(float(out[k])))
        else:
            out[k] = float(out[k])
    return out


def _eval_params(
    params: Dict[str, Any],
    X_np: np.ndarray,
    y_np: np.ndarray,
    kfold: StratifiedKFold,
    seed: int,
    n_jobs: int,
    tree_method: str,
) -> Dict[str, Any]:
    ev = evaluate_xgb_cv(
        params,
        X_np,
        y_np,
        kfold,
        seed=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )

    out: Dict[str, Any] = dict(ev.metrics)
    out["Time_MeanFold"] = float(ev.time_mean_fold)
    out["hyperparameters"] = _cast_ints(ev.params)
    return out


def evaluate_candidate_list(
    candidates: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a list of hyperparameter dicts and return the best by Accuracy_Mean."""
    X_np, y_np = X.to_numpy(), y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    evals: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    best_acc = -np.inf

    for p in tqdm(candidates, desc="Evaluating candidates"):
        res = _eval_params(p, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        evals.append(res)
        if float(res["Accuracy_Mean"]) > best_acc:
            best_acc = float(res["Accuracy_Mean"])
            best = res

    assert best is not None
    return best, evals


def load_nbi_candidates(path: str) -> List[Dict[str, Any]]:
    """Load candidates saved by `save_nbi_candidates` (dict stored as string)."""
    df = pd.read_csv(path, sep=";", decimal=",")
    if "hyperparameters" not in df.columns:
        raise KeyError(f"Expected column 'hyperparameters' in {path}")

    cands: List[Dict[str, Any]] = []
    for s in df["hyperparameters"].tolist():
        if isinstance(s, str):
            cands.append(ast.literal_eval(s))
        elif isinstance(s, dict):
            cands.append(s)
        else:
            raise TypeError(f"Unsupported hyperparameters type: {type(s)}")
    return cands


# -----------------------------
# Benchmarks (fixed budgets)
# -----------------------------

def random_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int = 40,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Random Search baseline with fixed budget."""
    param_dist = {
        "subsample": np.linspace(bounds["subsample"][0], bounds["subsample"][1], 10),
        "colsample_bytree": np.linspace(bounds["colsample_bytree"][0], bounds["colsample_bytree"][1], 10),
        "colsample_bylevel": np.linspace(bounds["colsample_bylevel"][0], bounds["colsample_bylevel"][1], 10),
        "learning_rate": np.linspace(bounds["learning_rate"][0], bounds["learning_rate"][1], 10),
        "max_depth": np.arange(int(bounds["max_depth"][0]), int(bounds["max_depth"][1]) + 1),
        "gamma": np.linspace(bounds["gamma"][0], bounds["gamma"][1], 10),
        "n_estimators": np.linspace(bounds["n_estimators"][0], bounds["n_estimators"][1], 10, dtype=int),
    }

    samples = list(ParameterSampler(param_dist, n_iter=budget, random_state=seed))

    t0 = time.perf_counter()
    best, _ = evaluate_candidate_list(samples, X, y, seed=seed, n_splits=n_splits, n_jobs=n_jobs, tree_method=tree_method)
    opt_time = time.perf_counter() - t0

    best["method"] = "random_search"
    best["budget"] = int(budget)
    best["Optimization_Time_Seconds"] = float(opt_time)
    best["Total_Time_Seconds"] = float(opt_time + float(best.get("Time_MeanFold", 0.0)))
    return best


def coarse_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int = 40,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Coarse Grid Search baseline bounded by budget.

    We build a **3-level grid** for *all* hyperparameters: {center, low, high},
    and evaluate at most `budget` points (deterministic order, center-first).
    This keeps the method "grid-like" while supporting budgets > 27.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")

    # Build 3 levels per parameter: center-first ordering
    levels: Dict[str, List[float]] = {}
    for p, (lo, hi) in bounds.items():
        mid = (float(lo) + float(hi)) / 2.0
        # center-first improves typical performance vs starting at all-lows
        if p in INT_PARAMS:
            vals = [int(round(mid)), int(round(lo)), int(round(hi))]
        else:
            vals = [float(mid), float(lo), float(hi)]
        # Remove duplicates if bounds collapse
        uniq: List[float] = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        levels[p] = uniq

    # Deterministic cartesian product in PARAM_NAMES order
    candidates: List[Dict[str, Any]] = []
    for combo in itertools.product(*[levels[p] for p in PARAM_NAMES]):
        cand: Dict[str, Any] = {p: v for p, v in zip(PARAM_NAMES, combo)}
        candidates.append(cand)
        if len(candidates) >= budget:
            break

    t0 = time.perf_counter()
    best, _ = evaluate_candidate_list(
        candidates, X, y, seed=seed, n_splits=n_splits, n_jobs=n_jobs, tree_method=tree_method
    )
    opt_time = time.perf_counter() - t0

    best["method"] = "coarse_grid_search"
    best["budget"] = int(len(candidates))
    best["Optimization_Time_Seconds"] = float(opt_time)
    best["Total_Time_Seconds"] = float(opt_time + float(best.get("Time_MeanFold", 0.0)))
    return best

def bayes_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int = 40,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Bayesian Optimization baseline (scikit-optimize BayesSearchCV) with fixed budget."""
    try:
        from skopt import BayesSearchCV
    except ImportError as e:
        raise ImportError("scikit-optimize is required for bayes_search. Install scikit-optimize.") from e

    from xgboost import XGBClassifier

    X_np, y_np = X.to_numpy(), y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    search_spaces = {
        "subsample": (DEFAULT_BOUNDS["subsample"][0], DEFAULT_BOUNDS["subsample"][1]),
        "colsample_bytree": (DEFAULT_BOUNDS["colsample_bytree"][0], DEFAULT_BOUNDS["colsample_bytree"][1]),
        "colsample_bylevel": (DEFAULT_BOUNDS["colsample_bylevel"][0], DEFAULT_BOUNDS["colsample_bylevel"][1]),
        "learning_rate": (DEFAULT_BOUNDS["learning_rate"][0], DEFAULT_BOUNDS["learning_rate"][1]),
        "max_depth": (int(DEFAULT_BOUNDS["max_depth"][0]), int(DEFAULT_BOUNDS["max_depth"][1])),
        "gamma": (DEFAULT_BOUNDS["gamma"][0], DEFAULT_BOUNDS["gamma"][1]),
        "n_estimators": (int(DEFAULT_BOUNDS["n_estimators"][0]), int(DEFAULT_BOUNDS["n_estimators"][1])),
    }

    estimator = XGBClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric="logloss",
        verbosity=0,
    )

    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=budget,
        cv=kfold,
        scoring="accuracy",
        n_jobs=1,  # avoid nested parallelism; XGBoost uses n_jobs
        random_state=seed,
    )

    t0 = time.perf_counter()
    opt.fit(X_np, y_np)
    opt_time = time.perf_counter() - t0

    # ---- Pylance-safe access to best_params_ ----
    best_params_raw = cast(Any, getattr(opt, "best_params_", None))
    if best_params_raw is None:
        raise RuntimeError("BayesSearchCV did not expose best_params_ after fit().")
    best_params = _cast_ints(best_params_raw)

    best_eval = _eval_params(best_params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)

    best_eval["method"] = "bayes_search"
    best_eval["budget"] = int(budget)
    best_eval["Optimization_Time_Seconds"] = float(opt_time)
    best_eval["Total_Time_Seconds"] = float(opt_time + float(best_eval.get("Time_MeanFold", 0.0)))
    return best_eval


def hyperopt_tpe(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int = 40,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Hyperopt TPE baseline with fixed budget."""
    try:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    except ImportError as e:
        raise ImportError("hyperopt is required for hyperopt_tpe. Install hyperopt.") from e

    X_np, y_np = X.to_numpy(), y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(params: Dict[str, Any]) -> Dict[str, Any]:
        params = _cast_ints(params)
        ev = _eval_params(params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        return {"loss": -float(ev["Accuracy_Mean"]), "status": STATUS_OK, "eval": ev}

    space = {
        "subsample": hp.uniform("subsample", DEFAULT_BOUNDS["subsample"][0], DEFAULT_BOUNDS["subsample"][1]),
        "colsample_bytree": hp.uniform("colsample_bytree", DEFAULT_BOUNDS["colsample_bytree"][0], DEFAULT_BOUNDS["colsample_bytree"][1]),
        "colsample_bylevel": hp.uniform("colsample_bylevel", DEFAULT_BOUNDS["colsample_bylevel"][0], DEFAULT_BOUNDS["colsample_bylevel"][1]),
        "learning_rate": hp.uniform("learning_rate", DEFAULT_BOUNDS["learning_rate"][0], DEFAULT_BOUNDS["learning_rate"][1]),
        "max_depth": hp.quniform("max_depth", DEFAULT_BOUNDS["max_depth"][0], DEFAULT_BOUNDS["max_depth"][1], 1),
        "gamma": hp.uniform("gamma", DEFAULT_BOUNDS["gamma"][0], DEFAULT_BOUNDS["gamma"][1]),
        "n_estimators": hp.quniform("n_estimators", DEFAULT_BOUNDS["n_estimators"][0], DEFAULT_BOUNDS["n_estimators"][1], 1),
    }

    trials = Trials()

    t0 = time.perf_counter()
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=budget,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )
    opt_time = time.perf_counter() - t0

    best_trial = min(trials.results, key=lambda r: r["loss"])
    best_eval = cast(Dict[str, Any], best_trial["eval"])

    best_eval["method"] = "hyperopt_tpe"
    best_eval["budget"] = int(budget)
    best_eval["Optimization_Time_Seconds"] = float(opt_time)
    best_eval["Total_Time_Seconds"] = float(opt_time + float(best_eval.get("Time_MeanFold", 0.0)))
    return best_eval


def save_benchmark_summary(rows: List[Dict[str, Any]], path: str) -> None:
    df = pd.DataFrame(rows)
    save_csv_ptbr(df, path)
