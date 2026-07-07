"""Base-model zoo, OOF probability matrices and the runtime microbenchmark.

Fixed, lightweight, documented configurations (no hyperparameter search —
pre-registered scope decision). The zoo has M=5 heterogeneous components;
the 3rd slot is kNN OR ExtraTrees, decided by the microbenchmark + the
twin-collinearity concern (ExtraTrees ~ RandomForest weakens Scheffé
coefficient identifiability; kNN is preferred when runtime allows).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CLIP_EPS = 1e-3


def make_model(name: str, random_state: int = 42):
    """Factory of fixed lightweight configurations (documented in the report)."""
    if name == "lr":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
            ]
        )
    if name == "gnb":
        return GaussianNB()
    if name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=100, algorithm="brute", n_jobs=-1)),
            ]
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=random_state
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=random_state
        )
    if name == "xgb":
        return XGBClassifier(
            tree_method="hist",
            n_estimators=400,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        )
    raise ValueError(f"unknown model {name!r}")


def zoo_names(third_slot: str = "knn") -> list[str]:
    if third_slot not in ("knn", "et"):
        raise ValueError("third_slot must be 'knn' or 'et'")
    return ["lr", "gnb", third_slot, "rf", "xgb"]


def clip_probs(p: np.ndarray, eps: float = CLIP_EPS) -> np.ndarray:
    """Clip COMPONENT probabilities (never the blend) so log-loss stays finite
    at kNN/GNB vertices while p_ens remains linear in w."""
    return np.clip(p, eps, 1.0 - eps)


@dataclass
class OOFResult:
    model_names: list[str]
    # P[repeat] has shape (n_train, M): clipped positive-class OOF probabilities
    P: list[np.ndarray]
    fold_ids: list[np.ndarray]
    fit_seconds: dict[str, float]
    predict_seconds: dict[str, float]
    n_splits: int
    n_repeats: int
    random_state: int


def generate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    third_slot: str = "knn",
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = 42,
    eps: float = CLIP_EPS,
) -> OOFResult:
    """Out-of-fold probability matrices via RepeatedStratifiedKFold (5×2 = 10 rounds).

    Each model is fitted once per fold inside its own Pipeline (all
    data-dependent transforms fold-fitted). Returns one (n, M) matrix per
    repeat; every weight vector is later evaluated on these matrices at
    ~zero cost.
    """
    names = zoo_names(third_slot)
    Xv = X.to_numpy(dtype=np.float32)
    yv = y.to_numpy()
    n = len(yv)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    P = [np.full((n, len(names)), np.nan, dtype=np.float32) for _ in range(n_repeats)]
    fold_ids = [np.full(n, -1, dtype=np.int16) for _ in range(n_repeats)]
    fit_s = dict.fromkeys(names, 0.0)
    pred_s = dict.fromkeys(names, 0.0)

    for split_idx, (tr, te) in enumerate(rskf.split(Xv, yv)):
        repeat = split_idx // n_splits
        fold = split_idx % n_splits
        fold_ids[repeat][te] = fold
        for j, name in enumerate(names):
            model = make_model(name, random_state=random_state + repeat)
            t0 = time.perf_counter()
            model.fit(Xv[tr], yv[tr])
            t1 = time.perf_counter()
            proba = model.predict_proba(Xv[te])[:, 1]
            t2 = time.perf_counter()
            P[repeat][te, j] = clip_probs(proba, eps)
            fit_s[name] += t1 - t0
            pred_s[name] += t2 - t1

    for repeat in range(n_repeats):
        if np.isnan(P[repeat]).any():
            raise RuntimeError(f"OOF matrix for repeat {repeat} has unfilled entries")
    return OOFResult(
        model_names=names,
        P=P,
        fold_ids=fold_ids,
        fit_seconds=fit_s,
        predict_seconds=pred_s,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )


def fit_full_and_predict_holdout(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    *,
    third_slot: str = "knn",
    random_state: int = 42,
    eps: float = CLIP_EPS,
) -> tuple[np.ndarray, dict[str, float]]:
    """Refit each base model once on the full training side and predict the
    holdout -> Q (n_holdout, M), clipped."""
    names = zoo_names(third_slot)
    Q = np.empty((len(X_te), len(names)), dtype=np.float32)
    seconds: dict[str, float] = {}
    Xtr = X_tr.to_numpy(dtype=np.float32)
    Xte = X_te.to_numpy(dtype=np.float32)
    ytr = y_tr.to_numpy()
    for j, name in enumerate(names):
        model = make_model(name, random_state=random_state)
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        Q[:, j] = clip_probs(model.predict_proba(Xte)[:, 1], eps)
        seconds[name] = time.perf_counter() - t0
    return Q, seconds


# ---------------------------------------------------------------------------
# Runtime microbenchmark and execution-mode decision (pre-registered)
# ---------------------------------------------------------------------------


def microbenchmark(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_bench: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Time one fit+predict per candidate model on a small stratified sample."""
    if len(X) > n_bench:
        Xb, _, yb, _ = train_test_split(
            X, y, train_size=n_bench, stratify=y, random_state=random_state
        )
    else:
        Xb, yb = X, y
    Xf, Xp, yf, _ = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=random_state)
    rows = []
    for name in ["lr", "gnb", "knn", "et", "rf", "xgb"]:
        model = make_model(name, random_state=random_state)
        t0 = time.perf_counter()
        model.fit(Xf.to_numpy(dtype=np.float32), yf.to_numpy())
        t1 = time.perf_counter()
        model.predict_proba(Xp.to_numpy(dtype=np.float32))
        t2 = time.perf_counter()
        rows.append(
            {
                "model": name,
                "n_fit": len(Xf),
                "n_predict": len(Xp),
                "fit_seconds": t1 - t0,
                "predict_seconds": t2 - t1,
            }
        )
    return pd.DataFrame(rows)


def estimate_total_minutes(
    bench: pd.DataFrame,
    *,
    n_rows: int,
    third_slot: str,
    n_splits: int = 5,
    n_repeats: int = 2,
    safety: float = 1.5,
) -> float:
    """Conservative wall-clock estimate (minutes) for the full protocol at n_rows.

    Fit costs scale ~linearly with fold-train size; kNN predict scales with
    (reference × query). Includes the 10 OOF rounds plus the final refit on
    the full 80% side and holdout prediction.
    """
    b = bench.set_index("model")
    names = zoo_names(third_slot)
    fold_train = 0.8 * n_rows * (1 - 1 / n_splits)
    fold_pred = 0.8 * n_rows / n_splits
    full_train = 0.8 * n_rows
    holdout = 0.2 * n_rows
    total = 0.0
    for name in names:
        fit0 = float(b.loc[name, "fit_seconds"])
        pred0 = float(b.loc[name, "predict_seconds"])
        nf0 = float(b.loc[name, "n_fit"])
        np0 = float(b.loc[name, "n_predict"])
        fit_fold = fit0 * fold_train / nf0
        fit_full = fit0 * full_train / nf0
        if name == "knn":  # predict cost ~ reference × query
            pred_fold = pred0 * (fold_train / nf0) * (fold_pred / np0)
            pred_hold = pred0 * (full_train / nf0) * (holdout / np0)
        else:
            pred_fold = pred0 * fold_pred / np0
            pred_hold = pred0 * holdout / np0
        total += n_splits * n_repeats * (fit_fold + pred_fold) + fit_full + pred_hold
    return safety * total / 60.0


def decide_execution(
    bench: pd.DataFrame,
    *,
    requested_mode: str,
    n_available: int,
    max_runtime_minutes: float = 120.0,
) -> dict:
    """Pre-registered mode/zoo decision with a logical hard stop.

    Tries the requested mode first; if the conservative estimate exceeds the
    budget, downgrades the mode (full_optional -> final_2h -> fast) and, as a
    last resort, swaps kNN for ExtraTrees. Raises if nothing fits.
    """
    from mixens.data import SAMPLE_CAPS  # local import to avoid cycle

    order = ["full_optional", "final_2h", "fast"]
    start = order.index(requested_mode) if requested_mode in order else len(order) - 1
    tried = []
    for third in ("knn", "et"):
        for mode in order[start:]:
            cap = SAMPLE_CAPS[mode]
            n_rows = min(n_available, cap) if cap else n_available
            est = estimate_total_minutes(bench, n_rows=n_rows, third_slot=third)
            tried.append({"mode": mode, "third_slot": third, "n_rows": n_rows,
                          "estimated_minutes": round(est, 1)})
            if est <= max_runtime_minutes:
                return {
                    "mode": mode,
                    "third_slot": third,
                    "n_rows": n_rows,
                    "estimated_minutes": round(est, 1),
                    "max_runtime_minutes": max_runtime_minutes,
                    "downgraded": mode != requested_mode or third != "knn",
                    "attempts": tried,
                }
    raise RuntimeError(
        f"no execution mode fits within {max_runtime_minutes} min; attempts: {tried}"
    )


__all__ = [
    "CLIP_EPS",
    "OOFResult",
    "clip_probs",
    "decide_execution",
    "estimate_total_minutes",
    "fit_full_and_predict_holdout",
    "generate_oof",
    "make_model",
    "microbenchmark",
    "zoo_names",
]
