"""Evaluate ensemble weight vectors and baselines on OOF/holdout matrices.

Everything here operates on precomputed probability matrices P (n, M):
evaluating a weight vector is a matrix-vector product, so the mixture
design and all baselines share the exact same folds and cost ~nothing.
The threshold for F1/balanced accuracy is selected on the OOF predictions
of each method (never on the holdout) — pre-registered policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


def blend(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """p_ens = P @ w (linear in w; components already clipped)."""
    return P @ np.asarray(w, dtype=P.dtype)


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid: int = 199) -> float:
    """Threshold maximizing F1 on the given (OOF) predictions."""
    ts = np.linspace(0.005, 0.995, grid)
    f1s = [f1_score(y, p >= t, zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f1s))])


def score_probs(
    y: np.ndarray,
    p: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Common metric block for one probability vector."""
    out = {
        "roc_auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }
    if threshold is not None:
        yhat = p >= threshold
        out["f1"] = float(f1_score(y, yhat, zero_division=0))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y, yhat))
        out["threshold"] = float(threshold)
    return out


def evaluate_design(
    W: np.ndarray,
    P_list: list[np.ndarray],
    y: np.ndarray,
) -> pd.DataFrame:
    """Evaluate each weight vector on each OOF repeat (AUC, log-loss, Brier)."""
    rows = []
    for r, P in enumerate(P_list):
        for i, w in enumerate(W):
            p = blend(P, w)
            rows.append(
                {
                    "point": i,
                    "repeat": r,
                    **{f"w{j+1}": float(w[j]) for j in range(W.shape[1])},
                    **score_probs(y, p),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baselines (all on the same OOF matrices / holdout matrix)
# ---------------------------------------------------------------------------


def corrected_repeated_cv_ttest(
    diffs: np.ndarray,
    *,
    rho: float = 0.25,
) -> dict[str, float]:
    """Corrected resampled t-test for repeated k-fold CV (Nadeau & Bengio, 2003;
    Bouckaert & Frank, 2004). ``diffs`` = per-fold metric differences between
    two methods evaluated on the SAME folds (here J = 5 folds × 2 repeats = 10).
    ``rho`` = n_test/n_train = (1/K)/(1-1/K) = 0.25 for K=5. The correction
    accounts for the overlap of training sets across folds; with a single
    dataset this remains an approximation and is reported as such.
    """
    from scipy import stats

    d = np.asarray(diffs, dtype=float)
    J = d.size
    dbar = float(d.mean())
    s2 = float(d.var(ddof=1))
    if s2 == 0.0:
        return {"t": np.inf if dbar != 0 else 0.0, "p_value": 0.0 if dbar != 0 else 1.0,
                "mean_diff": dbar, "df": J - 1}
    t = dbar / np.sqrt(s2 * (1.0 / J + rho))
    p = float(2.0 * stats.t.sf(abs(t), df=J - 1))
    return {"t": float(t), "p_value": p, "mean_diff": dbar, "df": J - 1}


def per_fold_metric(
    p: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Metric per OOF fold for one probability vector (one CV repeat)."""
    out = []
    for f in np.unique(fold_ids[fold_ids >= 0]):
        m = fold_ids == f
        if metric == "roc_auc":
            out.append(roc_auc_score(y[m], p[m]))
        elif metric == "log_loss":
            out.append(log_loss(y[m], p[m]))
        elif metric == "brier":
            out.append(brier_score_loss(y[m], p[m]))
        else:
            raise ValueError(metric)
    return np.asarray(out, dtype=float)


def best_single_index(P_list: list[np.ndarray], y: np.ndarray) -> int:
    """Best individual model by mean OOF ROC-AUC (primary metric)."""
    M = P_list[0].shape[1]
    aucs = [
        float(np.mean([roc_auc_score(y, P[:, j]) for P in P_list])) for j in range(M)
    ]
    return int(np.argmax(aucs))


def fit_stacking(P_train: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Logistic-regression stacker on the OOF probability matrix."""
    stk = LogisticRegression(max_iter=2000)
    stk.fit(P_train, y)
    return stk


def uniform_weights(M: int) -> np.ndarray:
    return np.full(M, 1.0 / M)


def vertex_weights(M: int, j: int) -> np.ndarray:
    w = np.zeros(M)
    w[j] = 1.0
    return w


__all__ = [
    "best_f1_threshold",
    "best_single_index",
    "blend",
    "evaluate_design",
    "fit_stacking",
    "score_probs",
    "uniform_weights",
    "vertex_weights",
]
