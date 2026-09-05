"""Fast, vectorized metrics for evaluating many ensemble-weight vectors.

Everything operates on a cached probability matrix ``P`` (n, M) and labels
``y`` (n,). A weight vector gives the blend ``p = P @ w``; evaluating tens
of thousands of weight vectors needs rank-based AUC and batched log-loss
rather than per-vector sklearn calls (the AUC on 160k rows costs ~10 ms
with one argsort, ~50 ms with ``roc_auc_score``).

All functions are numerically cross-checked against sklearn in the tests.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

CLIP_EPS = 1e-15


def rank_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic with average ranks for ties."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(p, method="average")
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fast_auc_no_ties(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC assuming no ties (single argsort; ~5x faster than rankdata).

    Blends of continuous probabilities essentially never tie except at
    vertices of the simplex; use :func:`rank_auc` when ties matter.
    """
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    order = np.argsort(p, kind="stable")
    ranks = np.empty(y.size, dtype=np.float64)
    ranks[order] = np.arange(1, y.size + 1, dtype=np.float64)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def log_loss_vec(y: np.ndarray, p: np.ndarray, eps: float = CLIP_EPS) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def brier_vec(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y, dtype=np.float64)) ** 2))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """PR-AUC as average precision (sklearn's step-wise definition, no ties assumed)."""
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(p), kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    k = np.arange(1, ys.size + 1)
    precision = tp / k
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    return float(precision[ys].sum() / n_pos)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y).astype(bool)
    yhat = np.asarray(p) >= threshold
    tp = int(np.sum(yhat & y)); fp = int(np.sum(yhat & ~y))
    fn = int(np.sum(~yhat & y)); tn = int(np.sum(~yhat & ~y))
    tpr = tp / max(tp + fn, 1); tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    return {"f1": float(f1), "balanced_accuracy": float(0.5 * (tpr + tnr)), "threshold": float(threshold)}


def best_f1_threshold(y: np.ndarray, p: np.ndarray, grid: int = 199) -> float:
    ts = np.linspace(0.005, 0.995, grid)
    y = np.asarray(y).astype(bool)
    order = np.argsort(-np.asarray(p), kind="stable")
    ps = np.asarray(p)[order]; ys = y[order]
    tp_cum = np.cumsum(ys); n_pos = int(y.sum())
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        k = int(np.searchsorted(-ps, -t, side="right"))  # number predicted positive
        tp = tp_cum[k - 1] if k > 0 else 0
        prec = tp / max(k, 1); rec = tp / max(n_pos, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# ---------------------------------------------------------------------------
# Weight-vector descriptors
# ---------------------------------------------------------------------------


def n_eff(W: np.ndarray) -> np.ndarray:
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    return 1.0 / np.sum(W ** 2, axis=1)


def entropy(W: np.ndarray) -> np.ndarray:
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(W > 0, W * np.log(W), 0.0)
    return -t.sum(axis=1)


def support_mask(W: np.ndarray, eps: float) -> np.ndarray:
    return np.atleast_2d(np.asarray(W, dtype=np.float64)) > eps


def weighted_cost(W: np.ndarray, costs: np.ndarray) -> np.ndarray:
    return np.atleast_2d(np.asarray(W, dtype=np.float64)) @ np.asarray(costs, dtype=np.float64)


def support_cost(W: np.ndarray, costs: np.ndarray, eps: float) -> np.ndarray:
    return support_mask(W, eps).astype(np.float64) @ np.asarray(costs, dtype=np.float64)


# ---------------------------------------------------------------------------
# Batched evaluation of many weight vectors on cached probabilities
# ---------------------------------------------------------------------------


def evaluate_weights(
    P: np.ndarray,
    y: np.ndarray,
    W: np.ndarray,
    *,
    costs: np.ndarray | None = None,
    support_eps: float = 1e-3,
    chunk: int = 256,
    with_pr_auc: bool = True,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Evaluate every row of ``W`` (N, M) on ``P`` (n, M): AUC, log-loss,
    Brier, PR-AUC plus weight descriptors and costs. Returns a dict of
    (N,) arrays. Uses argsort-based AUC (no-ties assumption)."""
    P = np.asarray(P, dtype=np.float32)
    y = np.asarray(y)
    yb = y.astype(bool)
    yf = y.astype(np.float64)
    W = np.atleast_2d(np.asarray(W, dtype=np.float64))
    N = W.shape[0]

    def _chunk(lo: int, hi: int):
        Wc = W[lo:hi]
        Pc = (P @ Wc.T.astype(np.float32)).astype(np.float64)  # (n, b)
        Pc = np.clip(Pc, CLIP_EPS, 1.0 - CLIP_EPS)
        ll = -np.mean(yf[:, None] * np.log(Pc) + (1.0 - yf[:, None]) * np.log1p(-Pc), axis=0)
        br = np.mean((Pc - yf[:, None]) ** 2, axis=0)
        auc = np.empty(hi - lo); ap = np.empty(hi - lo)
        for j in range(hi - lo):
            col = Pc[:, j]
            auc[j] = fast_auc_no_ties(yb, col)
            ap[j] = average_precision(yb, col) if with_pr_auc else np.nan
        return ll, br, auc, ap

    bounds = [(lo, min(lo + chunk, N)) for lo in range(0, N, chunk)]
    if n_jobs != 1 and len(bounds) > 1:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_chunk)(lo, hi) for lo, hi in bounds)
    else:
        parts = [_chunk(lo, hi) for lo, hi in bounds]
    out = {
        "log_loss": np.concatenate([p[0] for p in parts]),
        "brier": np.concatenate([p[1] for p in parts]),
        "roc_auc": np.concatenate([p[2] for p in parts]),
        "pr_auc": np.concatenate([p[3] for p in parts]),
        "n_eff": n_eff(W),
        "entropy": entropy(W),
        "n_support": support_mask(W, support_eps).sum(axis=1).astype(np.float64),
    }
    if costs is not None:
        out["cost_weighted"] = weighted_cost(W, costs)
        out["cost_support"] = support_cost(W, costs, support_eps)
    return out


__all__ = [
    "average_precision",
    "best_f1_threshold",
    "brier_vec",
    "entropy",
    "evaluate_weights",
    "fast_auc_no_ties",
    "log_loss_vec",
    "n_eff",
    "rank_auc",
    "support_cost",
    "support_mask",
    "threshold_metrics",
    "weighted_cost",
]
