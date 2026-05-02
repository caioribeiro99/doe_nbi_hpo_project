"""Simplex weight generation for N-objective NBI / mixture DOE.

Implements the Scheffé Simplex-Lattice {q, m} (uniform proportions
``i/m`` for ``i = 0, …, m`` summing to 1), simplex-centroid, extreme-vertex
designs, and validation of externally supplied weight arrays.

All functions are dimension-agnostic (work for arbitrary q >= 2). They
only exist to *generate* (q,)-shaped weight vectors; they do not assume
any particular q.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterable, List, Optional

import numpy as np


def simplex_lattice_count(q: int, m: int) -> int:
    """Number of points on the Simplex-Lattice {q, m}.

    Equals C(q + m - 1, m).
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")
    return comb(q + m - 1, m)


def generate_simplex_lattice(q: int, m: int) -> np.ndarray:
    """All weight vectors w in R^q with w_i in {0, 1/m, …, 1} and sum(w) = 1.

    Returns
    -------
    np.ndarray of shape (N, q), N = C(q+m-1, m), dtype=float64.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    if m < 1:
        raise ValueError("m must be >= 1")

    rows: List[List[int]] = []

    def _recurse(remaining: int, slots: int, prefix: List[int]) -> None:
        if slots == 1:
            rows.append(prefix + [remaining])
            return
        for k in range(remaining + 1):
            _recurse(remaining - k, slots - 1, prefix + [k])

    _recurse(m, q, [])
    arr = np.asarray(rows, dtype=float) / float(m)
    # Sanity: every row sums to 1 (within float tolerance).
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-12):
        raise RuntimeError(f"simplex-lattice rows do not sum to 1; got max abs deviation {np.max(np.abs(sums - 1.0))}")
    return arr


def generate_simplex_centroid(q: int, depth: Optional[int] = None) -> np.ndarray:
    """Simplex-centroid design.

    For each subset of size ``r in {1, …, depth}``, places a single
    point with equal weight ``1/r`` on those ``r`` components and 0
    elsewhere. Default ``depth = q`` (full centroid).
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    if depth is None:
        depth = q
    if not (1 <= depth <= q):
        raise ValueError("depth must be in [1, q]")
    rows: List[np.ndarray] = []
    for r in range(1, depth + 1):
        for subset in combinations(range(q), r):
            v = np.zeros(q, dtype=float)
            for j in subset:
                v[j] = 1.0 / r
            rows.append(v)
    return np.vstack(rows)


def generate_extreme_vertices(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    n_grid: int = 5,
) -> np.ndarray:
    """Extreme-vertex points of a constrained simplex.

    For mixture problems with component-wise lower/upper bounds, this
    enumerates a coarse grid on the box and projects each point onto the
    simplex by normalization, then retains those satisfying the bounds.

    This is intentionally simple: when full McLean-Anderson design is
    needed, the user should supply weights externally via
    :func:`validate_custom_weights`.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or lower.ndim != 1:
        raise ValueError("lower and upper must be 1-D arrays of the same shape")
    if np.any(lower > upper):
        raise ValueError("lower must be elementwise <= upper")
    q = lower.size
    if q < 2:
        raise ValueError("need at least two components")

    grid = [np.linspace(lower[i], upper[i], n_grid) for i in range(q)]
    candidates: List[np.ndarray] = []
    mesh = np.array(np.meshgrid(*grid, indexing="ij")).reshape(q, -1).T
    for row in mesh:
        s = row.sum()
        if s <= 0.0:
            continue
        w = row / s
        if np.all(w >= lower) and np.all(w <= upper):
            candidates.append(w)
    if not candidates:
        raise ValueError("no feasible extreme vertices found; tighten bounds or increase n_grid")
    arr = np.vstack(candidates)
    # Deduplicate up to numerical tolerance.
    rounded = np.round(arr, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def validate_weights(weights: np.ndarray, *, atol: float = 1e-9) -> None:
    """Raise if weights are not a valid (N, q) simplex array.

    Constraints: shape is (N, q) with N >= 1 and q >= 2;
    every row sums to 1 within ``atol``; every entry is non-negative.
    """
    arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"weights must be 2-D; got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"weights shape must be (N>=1, q>=2); got {arr.shape}")
    if np.any(arr < -atol):
        raise ValueError("weights must be non-negative")
    sums = arr.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=atol):
        worst = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"each row must sum to 1; worst |sum-1| = {worst:.3e}")


def custom_weights(weights: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(list(weights), dtype=float)
    validate_weights(arr)
    return arr


__all__ = [
    "simplex_lattice_count",
    "generate_simplex_lattice",
    "generate_simplex_centroid",
    "generate_extreme_vertices",
    "validate_weights",
    "custom_weights",
]
