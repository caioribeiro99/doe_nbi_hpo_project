"""Weight optimization on the simplex: direct (SLSQP) and via the metamodel.

Direct optimization of log-loss over the OOF matrix is a CONVEX problem in w
(composition of a convex loss with a linear map), so SLSQP with the sum-to-one
constraint finds the global optimum — it is the honesty benchmark against
which the Scheffé metamodel's optimum is validated (pre-registered framing:
the metamodel is an interpretable statistical summary, not a computational
shortcut). AUC is not smooth in w, so its direct optimum is approximated by
a dense Dirichlet scan.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

from mixens.mixture_design import sample_dirichlet, validate_weights
from mixens.scheffe import MixtureScheffeModel


def _multistart_points(M: int, n_starts: int, random_state: int) -> np.ndarray:
    starts = [np.full(M, 1.0 / M)]
    starts.extend(sample_dirichlet(M, n_starts - 1, random_state=random_state))
    return np.vstack(starts)


def minimize_on_simplex(
    fn: Callable[[np.ndarray], float],
    M: int,
    *,
    n_starts: int = 10,
    random_state: int = 42,
) -> np.ndarray:
    """SLSQP with sum(w)=1, w>=0, multi-start; returns the best w found."""
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    best_w, best_v = None, np.inf          # converged (res.success) solutions
    feas_w, feas_v = None, np.inf          # best feasible iterate regardless of the success flag
    for w0 in _multistart_points(M, n_starts, random_state):
        res = minimize(fn, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 500, "ftol": 1e-10})
        w = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum()
        v = float(fn(w))
        if not np.isfinite(v):
            continue
        if res.success and v < best_v:
            best_v, best_w = v, w
        if v < feas_v:
            feas_v, feas_w = v, w
    # Exact safeguard for objectives whose optimum lies on a vertex (e.g. linear
    # Scheffé surfaces or their scalarizations), where SLSQP reports
    # "positive directional derivative" at the degenerate corner: evaluate the
    # vertices and the centroid explicitly.
    for w in np.vstack([np.eye(M), np.full((1, M), 1.0 / M)]):
        v = float(fn(w))
        if np.isfinite(v) and v < feas_v:
            feas_v, feas_w = v, w
    if best_w is None or feas_v < best_v - 1e-12:
        best_w = feas_w
    if best_w is None:
        raise RuntimeError("SLSQP failed from all starts")
    w = np.clip(best_w, 0.0, 1.0)
    w = w / w.sum()
    validate_weights(w[None, :])
    return w


def direct_logloss_optimum(P: np.ndarray, y: np.ndarray, **kw) -> np.ndarray:
    """Global optimum of OOF log-loss over the simplex (convex problem)."""
    return minimize_on_simplex(lambda w: log_loss(y, P @ w), P.shape[1], **kw)


def metamodel_optimum(
    model: MixtureScheffeModel,
    *,
    maximize: bool = False,
    **kw,
) -> np.ndarray:
    """Optimum of the fitted Scheffé polynomial over the simplex."""
    sign = -1.0 if maximize else 1.0
    M = len(model.component_names)
    return minimize_on_simplex(lambda w: sign * model.predict_weights(w[None, :])[0], M, **kw)


def dirichlet_scan_auc(
    P: np.ndarray,
    y: np.ndarray,
    *,
    n: int = 10_000,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Approximate direct AUC optimum by dense uniform sampling of the simplex
    (AUC is neither smooth nor convex in w). Includes vertices and centroid."""
    M = P.shape[1]
    W = np.vstack(
        [
            np.eye(M),
            np.full((1, M), 1.0 / M),
            sample_dirichlet(M, n, random_state=random_state),
        ]
    )
    best_w, best_auc = None, -np.inf
    for w in W:
        auc = roc_auc_score(y, P @ w)
        if auc > best_auc:
            best_auc, best_w = float(auc), w
    return best_w, best_auc


__all__ = [
    "direct_logloss_optimum",
    "dirichlet_scan_auc",
    "metamodel_optimum",
    "minimize_on_simplex",
]
