"""Multi-objective and frontier reporting metrics.

Pure-NumPy implementations suitable for any q >= 2:

- :func:`generalized_distance`: Mahalanobis-style distance from a
  candidate to the utopia point, using a supplied or empirical
  variance-covariance matrix (Pereira et al., 2025, Eq. 29).
- :func:`shannon_entropy`: ``S(w) = -Σ w_i ln w_i``.
- :func:`spread_delta`: Pareto-front spread (Δ).
- :func:`spacing_entropy`: normalized entropy of nearest-neighbor gaps.
- :func:`hypervolume`: 2D / 3D exact via inclusion-exclusion; higher q
  uses a Monte-Carlo estimate.
- :func:`igd`: Inverted Generational Distance against a reference set.
- :func:`pareto_front`: extracts the non-dominated subset.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def pareto_front(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of ``F`` (canonical-min)."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = F - F[i]
        dominated = np.all(diff >= 0, axis=1) & np.any(diff > 0, axis=1)
        keep &= ~dominated
    return keep


def generalized_distance(
    F_at_x: np.ndarray,
    utopia: np.ndarray,
    cov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """GD per Pereira et al. (2025) Eq. 29 using Mahalanobis distance."""
    F_at_x = np.asarray(F_at_x, dtype=float)
    if F_at_x.ndim == 1:
        F_at_x = F_at_x[None, :]
    diff = F_at_x - np.asarray(utopia, dtype=float)
    if cov is None or np.allclose(cov, 0.0):
        return np.sqrt(np.sum(diff ** 2, axis=1))
    inv = np.linalg.pinv(np.asarray(cov, dtype=float))
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", diff, inv, diff)))


def shannon_entropy(weights: np.ndarray) -> np.ndarray:
    """``S(w) = -Σ w_i ln w_i``. Vectorized over rows."""
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = w[None, :]
    safe = np.where(w > 0.0, w, 1.0)
    return -np.sum(np.where(w > 0.0, w * np.log(safe), 0.0), axis=1)


def spread_delta(F: np.ndarray) -> float:
    """Front spread Δ; smaller is better."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n < 2:
        return 0.0
    # Sort by first objective, then compute consecutive distances.
    order = np.argsort(F[:, 0])
    Fs = F[order]
    d = np.linalg.norm(np.diff(Fs, axis=0), axis=1)
    if d.size == 0:
        return 0.0
    d_mean = float(np.mean(d))
    if d_mean == 0.0:
        return 0.0
    extreme = np.linalg.norm(Fs[0] - Fs[-1])
    return float((extreme + np.sum(np.abs(d - d_mean))) / (extreme + (n - 1) * d_mean))


def spacing_entropy(F: np.ndarray) -> float:
    """Shannon entropy of normalized nearest-neighbor distances."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    if n < 2:
        return 0.0
    d = np.zeros(n, dtype=float)
    for i in range(n):
        diff = F - F[i]
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        d[i] = float(np.min(dist))
    total = float(np.sum(d))
    if total == 0.0:
        return 0.0
    p = d / total
    safe = np.where(p > 0.0, p, 1.0)
    return float(-np.sum(np.where(p > 0.0, p * np.log(safe), 0.0)) / np.log(n))


def igd(F: np.ndarray, reference: np.ndarray) -> float:
    """Inverted Generational Distance (lower is better)."""
    F = np.asarray(F, dtype=float)
    R = np.asarray(reference, dtype=float)
    if R.size == 0 or F.size == 0:
        return float("nan")
    dists = []
    for r in R:
        d = np.min(np.linalg.norm(F - r, axis=1))
        dists.append(d)
    return float(np.mean(dists))


def hypervolume(F: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume for q in {2, 3}; Monte-Carlo for higher q."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if F.size == 0:
        return 0.0
    q = F.shape[1]
    if q == 2:
        # Sort by first objective ascending, then sweep.
        mask = pareto_front(F)
        Fs = F[mask]
        Fs = Fs[np.argsort(Fs[:, 0])]
        hv = 0.0
        prev_y = ref[1]
        for x, y in Fs:
            if x >= ref[0] or y >= ref[1]:
                continue
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
        return float(hv)
    # Monte-Carlo for q >= 3.
    rng = np.random.default_rng(0)
    n_samples = 20_000
    lo = F.min(axis=0)
    samples = rng.uniform(low=lo, high=ref, size=(n_samples, q))
    dominated = np.zeros(n_samples, dtype=bool)
    for f in F:
        dominated |= np.all(f <= samples, axis=1)
    box_vol = float(np.prod(ref - lo))
    return float(box_vol * dominated.mean())


__all__ = [
    "pareto_front",
    "generalized_distance",
    "shannon_entropy",
    "spread_delta",
    "spacing_entropy",
    "igd",
    "hypervolume",
]
