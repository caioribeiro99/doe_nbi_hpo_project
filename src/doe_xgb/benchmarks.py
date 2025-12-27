from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from tqdm import tqdm

from .config import DEFAULT_BOUNDS, PARAM_NAMES, INT_PARAMS
from .evaluation import evaluate_xgb_cv
from .io_utils import save_csv_ptbr


def _cast_ints(params: Dict) -> Dict:
    out = dict(params)
    for k in list(out.keys()):
        if k in INT_PARAMS:
            out[k] = int(round(float(out[k])))
        else:
            out[k] = float(out[k])
    return out


def _eval_params(params: Dict, X_np: np.ndarray, y_np: np.ndarray, kfold: StratifiedKFold, seed: int, n_jobs: int, tree_method: str) -> Dict:
    ev = evaluate_xgb_cv(params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
    out = dict(ev.metrics)
    out["Time_MeanFold"] = ev.time_mean_fold
    out["hyperparameters"] = _cast_ints(ev.params)
    return out


def evaluate_candidate_list(
    candidates: List[Dict],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Tuple[Dict, List[Dict]]:
    """Evaluate a list of hyperparameter dicts and return the best by Accuracy_Mean."""
    X_np, y_np = X.to_numpy(), y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    evals = []
    best = None
    best_acc = -np.inf
    for p in tqdm(candidates, desc="Evaluating candidates"):
        res = _eval_params(p, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        evals.append(res)
        if res["Accuracy_Mean"] > best_acc:
            best_acc = res["Accuracy_Mean"]
            best = res
    assert best is not None
    return best, evals


def load_nbi_candidates(path: str) -> List[Dict]:
    """Load candidates saved by `save_nbi_candidates` (dict stored as string)."""
    df = pd.read_csv(path, sep=';', decimal=',')
    if "hyperparameters" not in df.columns:
        raise KeyError(f"Expected column 'hyperparameters' in {path}")
    cands = []
    for s in df["hyperparameters"].tolist():
        if isinstance(s, str):
            cands.append(ast.literal_eval(s))
        elif isinstance(s, dict):
            cands.append(s)
        else:
            raise TypeError(f"Unsupported hyperparameters type: {type(s)}")
    return cands


# -----------------------
# Benchmark optimizers
# -----------------------
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
) -> Dict:
    """Random search with a fixed evaluation budget."""
    rng = np.random.default_rng(seed)

    # build discrete sampling distributions (as in notebook)
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
    best, _ = evaluate_candidate_list(samples, X, y, seed=seed, n_splits=n_splits, n_jobs=n_jobs, tree_method=tree_method)
    best["method"] = "random_search"
    best["budget"] = budget
    return best


def coarse_grid_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    budget: int = 40,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> Dict:
    """A small coarse grid bounded by `budget` evaluations.

    For fairness, we keep the grid <= budget by varying only 3 key factors
    (learning_rate, max_depth, n_estimators) and fixing the others at center.
    """
    # centers
    def center(p):
        lo, hi = DEFAULT_BOUNDS[p]
        return (lo + hi) / 2.0

    grid = {
        "learning_rate": [0.01, center("learning_rate"), 0.30],
        "max_depth": [int(DEFAULT_BOUNDS["max_depth"][0]), int(center("max_depth")), int(DEFAULT_BOUNDS["max_depth"][1])],
        "n_estimators": [int(DEFAULT_BOUNDS["n_estimators"][0]), int(center("n_estimators")), int(DEFAULT_BOUNDS["n_estimators"][1])],
    }

    base_params = {
        "subsample": center("subsample"),
        "colsample_bytree": center("colsample_bytree"),
        "colsample_bylevel": center("colsample_bylevel"),
        "gamma": center("gamma"),
    }

    candidates = []
    for lr in grid["learning_rate"]:
        for md in grid["max_depth"]:
            for ne in grid["n_estimators"]:
                p = dict(base_params)
                p.update({"learning_rate": lr, "max_depth": md, "n_estimators": ne})
                candidates.append(p)

    # cap to budget if needed
    candidates = candidates[:budget]

    best, _ = evaluate_candidate_list(candidates, X, y, seed=seed, n_splits=n_splits, n_jobs=n_jobs, tree_method=tree_method)
    best["method"] = "coarse_grid_search"
    best["budget"] = len(candidates)
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
) -> Dict:
    """Bayesian optimization using scikit-optimize (BayesSearchCV)."""
    try:
        from skopt import BayesSearchCV
    except ImportError as e:
        raise ImportError("scikit-optimize is required for bayes_search. Install scikit-optimize.") from e

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

    # Avoid nested parallelism: let XGB use all cores, BayesSearchCV itself sequential (n_jobs=1)
    opt = BayesSearchCV(
        estimator=None,  # set below
        search_spaces=search_spaces,
        n_iter=budget,
        cv=kfold,
        scoring='accuracy',
        n_jobs=1,
        random_state=seed,
    )

    from xgboost import XGBClassifier
    opt.estimator = XGBClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        tree_method=tree_method,
        eval_metric='logloss',
        verbosity=0,
    )

    opt.fit(X_np, y_np)

    best_params = opt.best_params_
    best_params = _cast_ints(best_params)

    best_eval = _eval_params(best_params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
    best_eval["method"] = "bayes_search"
    best_eval["budget"] = budget
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
) -> Dict:
    """Hyperopt TPE with a fixed evaluation budget."""
    try:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    except ImportError as e:
        raise ImportError("hyperopt is required for hyperopt_tpe. Install hyperopt.") from e

    X_np, y_np = X.to_numpy(), y.to_numpy()
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(params):
        params = _cast_ints(params)
        ev = _eval_params(params, X_np, y_np, kfold, seed=seed, n_jobs=n_jobs, tree_method=tree_method)
        return {'loss': -ev['Accuracy_Mean'], 'status': STATUS_OK, 'eval': ev}

    space = {
        'subsample': hp.uniform('subsample', DEFAULT_BOUNDS['subsample'][0], DEFAULT_BOUNDS['subsample'][1]),
        'colsample_bytree': hp.uniform('colsample_bytree', DEFAULT_BOUNDS['colsample_bytree'][0], DEFAULT_BOUNDS['colsample_bytree'][1]),
        'colsample_bylevel': hp.uniform('colsample_bylevel', DEFAULT_BOUNDS['colsample_bylevel'][0], DEFAULT_BOUNDS['colsample_bylevel'][1]),
        'learning_rate': hp.uniform('learning_rate', DEFAULT_BOUNDS['learning_rate'][0], DEFAULT_BOUNDS['learning_rate'][1]),
        'max_depth': hp.quniform('max_depth', DEFAULT_BOUNDS['max_depth'][0], DEFAULT_BOUNDS['max_depth'][1], 1),
        'gamma': hp.uniform('gamma', DEFAULT_BOUNDS['gamma'][0], DEFAULT_BOUNDS['gamma'][1]),
        'n_estimators': hp.quniform('n_estimators', DEFAULT_BOUNDS['n_estimators'][0], DEFAULT_BOUNDS['n_estimators'][1], 1),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=budget,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )

    # recover best eval from trials
    best_trial = min(trials.results, key=lambda r: r['loss'])
    best_eval = best_trial['eval']
    best_eval["method"] = "hyperopt_tpe"
    best_eval["budget"] = budget
    return best_eval


def save_benchmark_summary(rows: List[Dict], path: str) -> None:
    df = pd.DataFrame(rows)
    save_csv_ptbr(df, path)
