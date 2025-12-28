from __future__ import annotations

import ast
import itertools
import time
from math import ceil
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from tqdm import tqdm

from .config import DEFAULT_BOUNDS, INT_PARAMS, PARAM_NAMES
from .evaluation import evaluate_xgb_cv
from .io_utils import save_csv_ptbr


def _cast_ints(params: Dict[str, Any]) -> Dict[str, float | int]:
    """Cast integer hyperparameters to int (keeps others as float)."""
    out: Dict[str, float | int] = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


def _eval_params(
    params: Dict[str, float | int],
    X_np: np.ndarray,
    y_np: np.ndarray,
    kfold: StratifiedKFold,
    seed: int,
    n_jobs: int,
    tree_method: str,
) -> Dict[str, Any]:
    # evaluate_xgb_cv expects (params, X, y, kfold, ...)
    ev = evaluate_xgb_cv(
        params,
        X_np,
        y_np,
        kfold,
        seed=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
    )
    out = cast(Dict[str, Any], ev.as_dict())
    out["hyperparameters"] = dict(params)
    return out


def evaluate_candidate_list(
    candidates: List[Dict[str, float]],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
    desc: str = "Evaluating candidates",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a list of hyperparameter candidates.

    Returns:
      - best_row: chosen by highest Accuracy_Mean
      - all_rows: one row per candidate (includes CV metrics + timing)
    """
    if len(candidates) < 1:
        raise ValueError("candidates list is empty")

    X_np = X.to_numpy()
    y_np = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] | None = None
    best_acc = -np.inf

    for params in tqdm(candidates, desc=desc, dynamic_ncols=True):
        casted = _cast_ints(params)
        row = _eval_params(casted, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        all_rows.append(row)

        acc = float(row.get("Accuracy_Mean", float("-inf")))
        if acc > best_acc:
            best_acc = acc
            best_row = row

    if best_row is None:
        raise RuntimeError("No best row found (unexpected)")
    return best_row, all_rows


def load_nbi_candidates(path: str) -> List[Dict[str, float]]:
    """Load NBI candidates from CSV.

    Supports:
      1) Explicit param columns (PARAM_NAMES)
      2) Single 'hyperparameters' column containing a dict (or stringified dict)
    """
    df = pd.read_csv(path, sep=";", decimal=",")

    # Format 1
    if all(p in df.columns for p in PARAM_NAMES):
        out: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            out.append({p: float(row[p]) for p in PARAM_NAMES})
        return out

    # Format 2
    if "hyperparameters" in df.columns:
        out2: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            cell = row["hyperparameters"]
            hp: Dict[str, Any] = {}
            if isinstance(cell, dict):
                hp = cell
            elif isinstance(cell, str) and cell.strip():
                try:
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, dict):
                        hp = parsed
                except Exception:
                    hp = {}

            ok = True
            cand: Dict[str, float] = {}
            for p in PARAM_NAMES:
                if p not in hp:
                    ok = False
                    break
                try:
                    cand[p] = float(hp[p])
                except Exception:
                    ok = False
                    break
            if ok:
                out2.append(cand)
        return out2

    raise KeyError(f"Invalid NBI candidates file format: {path}")


def coarse_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Coarse grid search over DEFAULT_BOUNDS.

    Chooses k levels per parameter so that k^d >= budget, then samples `budget` points.
    """
    if budget < 1:
        raise ValueError("budget must be >= 1")

    d = len(PARAM_NAMES)
    k = max(3, int(ceil(budget ** (1.0 / d))))

    levels: Dict[str, List[float]] = {}
    for p in PARAM_NAMES:
        lo, hi = DEFAULT_BOUNDS[p]
        if p in INT_PARAMS:
            raw = np.linspace(lo, hi, num=k)
            ints = [int(round(v)) for v in raw]
            uniq: List[int] = []
            for iv in ints:
                iv = int(min(max(iv, int(round(lo))), int(round(hi))))
                if iv not in uniq:
                    uniq.append(iv)
            if len(uniq) < 2:
                uniq = [int(round(lo)), int(round(hi))]
            levels[p] = [float(v) for v in uniq]
        else:
            levels[p] = [float(v) for v in np.linspace(lo, hi, num=k)]

    grid = list(itertools.product(*[levels[p] for p in PARAM_NAMES]))
    candidates_all: List[Dict[str, float]] = [{p: float(v) for p, v in zip(PARAM_NAMES, combo)} for combo in grid]

    if len(candidates_all) <= budget:
        candidates = candidates_all
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(candidates_all), size=budget, replace=False)
        candidates = [candidates_all[int(i)] for i in idx]

    t0 = time.perf_counter()
    best_eval, _ = evaluate_candidate_list(
        candidates,
        X,
        y,
        seed=seed,
        n_splits=n_splits,
        n_jobs=n_jobs,
        tree_method=tree_method,
        desc=f"coarse_grid_search ({len(candidates)})",
    )
    opt_time = time.perf_counter() - t0

    best_eval["method"] = "coarse_grid_search"
    best_eval["budget"] = int(len(candidates))
    best_eval["Optimization_Time_Seconds"] = float(opt_time)
    best_eval["Total_Time_Seconds"] = float(opt_time + float(best_eval.get("Time_MeanFold", 0.0)))
    return best_eval


def random_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Random search within DEFAULT_BOUNDS."""
    if budget < 1:
        raise ValueError("budget must be >= 1")

    rng = np.random.default_rng(seed)
    param_grid: Dict[str, List[float]] = {}
    for p in PARAM_NAMES:
        lo, hi = DEFAULT_BOUNDS[p]
        if p in INT_PARAMS:
            vals = rng.integers(int(lo), int(hi) + 1, size=max(10, budget), dtype=np.int64)
            param_grid[p] = [float(v) for v in vals]
        else:
            vals = rng.uniform(lo, hi, size=max(10, budget)).astype(float)
            param_grid[p] = [float(v) for v in vals]

    sampler = list(ParameterSampler(param_grid, n_iter=budget, random_state=seed))
    candidates: List[Dict[str, float]] = [{p: float(s[p]) for p in PARAM_NAMES} for s in sampler]

    t0 = time.perf_counter()
    best_eval, _ = evaluate_candidate_list(
        candidates,
        X,
        y,
        seed=seed,
        n_splits=n_splits,
        n_jobs=n_jobs,
        tree_method=tree_method,
        desc=f"random_search ({len(candidates)})",
    )
    opt_time = time.perf_counter() - t0

    best_eval["method"] = "random_search"
    best_eval["budget"] = int(len(candidates))
    best_eval["Optimization_Time_Seconds"] = float(opt_time)
    best_eval["Total_Time_Seconds"] = float(opt_time + float(best_eval.get("Time_MeanFold", 0.0)))
    return best_eval


def bayes_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Bayesian optimization via scikit-optimize BayesSearchCV."""
    if budget < 1:
        raise ValueError("budget must be >= 1")

    from skopt import BayesSearchCV
    from skopt.space import Integer, Real
    from xgboost import XGBClassifier

    # ✅ FIX: BayesSearchCV expects a dict mapping param_name -> Dimension
    search_spaces: Dict[str, Any] = {}
    for p in PARAM_NAMES:
        lo, hi = DEFAULT_BOUNDS[p]
        if p in INT_PARAMS:
            search_spaces[p] = Integer(int(lo), int(hi))
        else:
            search_spaces[p] = Real(float(lo), float(hi))

    estimator = XGBClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric="logloss",
        verbosity=0,
    )

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    opt = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        n_iter=budget,
        cv=kfold,
        scoring="accuracy",
        n_jobs=1,  # avoid nested parallelism; XGBoost uses n_jobs
        random_state=seed,
        verbose=0,
    )

    X_np = X.to_numpy()
    y_np = y.to_numpy()
    t0 = time.perf_counter()
    opt.fit(X_np, y_np)
    opt_time = time.perf_counter() - t0

    best_params_raw = cast(Dict[str, Any], getattr(opt, "best_params_"))
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
    budget: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict[str, Any]:
    """Hyperopt TPE search."""
    if budget < 1:
        raise ValueError("budget must be >= 1")

    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    space: Dict[str, Any] = {}
    for p in PARAM_NAMES:
        lo, hi = DEFAULT_BOUNDS[p]
        if p in INT_PARAMS:
            space[p] = hp.quniform(p, int(lo), int(hi), q=1)
        else:
            space[p] = hp.uniform(p, float(lo), float(hi))

    X_np = X.to_numpy()
    y_np = y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(params: Dict[str, Any]) -> Dict[str, Any]:
        casted = _cast_ints(params)
        ev = evaluate_xgb_cv(casted, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        acc = ev.metrics.get("Accuracy_Mean")
        if acc is None:
            raise KeyError("Accuracy_Mean not found in EvalResult.metrics")
        return {"loss": -float(acc), "status": STATUS_OK}

    trials = Trials()
    t0 = time.perf_counter()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=int(budget),
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=True,
    )
    opt_time = time.perf_counter() - t0

    best_params = _cast_ints(cast(Dict[str, Any], best))
    best_eval = _eval_params(best_params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)

    best_eval["method"] = "hyperopt_tpe"
    best_eval["budget"] = int(budget)
    best_eval["Optimization_Time_Seconds"] = float(opt_time)
    best_eval["Total_Time_Seconds"] = float(opt_time + float(best_eval.get("Time_MeanFold", 0.0)))
    return best_eval


def save_benchmark_summary(rows: List[Dict[str, Any]], path: str) -> None:
    """Save a one-row-per-method benchmark summary as pt-BR friendly CSV."""
    if not rows:
        raise ValueError("rows is empty")
    df = pd.DataFrame(rows)
    save_csv_ptbr(df, path)
