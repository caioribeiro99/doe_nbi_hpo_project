"""Simplex weight generation for mixture designs over ensemble weights.

Ported from doe_nbi_hpo_project, branch ``repo-publication-readiness``
(commit 0465466), file ``src/doe_xgb/simplex.py``. Original author:
Caio Tertuliano Ribeiro (MIT License). Adapted for the PCO213 final
project: added Dirichlet sampling, axial points and the study's
composite design builder. The functions are dimension-agnostic
(work for arbitrary q >= 2) and only *generate* (q,)-shaped weight
vectors summing to 1.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from math import comb

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

    rows: list[list[int]] = []

    def _recurse(remaining: int, slots: int, prefix: list[int]) -> None:
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
        raise RuntimeError(
            f"simplex-lattice rows do not sum to 1; got max abs deviation {np.max(np.abs(sums - 1.0))}"
        )
    return arr


def generate_simplex_centroid(q: int, depth: int | None = None) -> np.ndarray:
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
    rows: list[np.ndarray] = []
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
    candidates: list[np.ndarray] = []
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
    rounded = np.round(arr, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    return arr[np.sort(idx)]


def validate_weights(weights: np.ndarray, *, atol: float = 1e-9) -> None:
    """Raise if weights are not a valid (N, q) simplex array."""
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


# ---------------------------------------------------------------------------
# Additions for the PCO213 study (not in the ported original)
# ---------------------------------------------------------------------------


def generate_axial_points(q: int) -> np.ndarray:
    """Axial (interior) check points, one per component.

    Midway between the overall centroid and each vertex (Cornell, 2002):
    ``x_i = (q + 1) / (2q)`` for the axial component, ``x_j = 1 / (2q)``
    for the others. These add interior support that the {q, 2} lattice
    (vertices + edge midpoints only) lacks.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    pts = np.full((q, q), 1.0 / (2.0 * q), dtype=float)
    np.fill_diagonal(pts, (q + 1.0) / (2.0 * q))
    return pts


def sample_dirichlet(
    q: int,
    n: int,
    *,
    alpha: float = 1.0,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Uniform (alpha=1) Dirichlet sample of n weight vectors on the simplex."""
    if q < 2:
        raise ValueError("q must be >= 2")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    arr = rng.dirichlet(np.full(q, float(alpha)), size=n)
    validate_weights(arr)
    return arr


def build_study_design(q: int) -> np.ndarray:
    """The study's composite mixture design.

    Simplex-lattice {q, 2} (vertices + edge midpoints) + overall centroid
    + q axial points, deduplicated. For q = 5: 15 + 1 + 5 = 21 runs, which
    supports the Scheffé quadratic (15 terms, 6 residual df).
    """
    parts = [
        generate_simplex_lattice(q, 2),
        np.full((1, q), 1.0 / q, dtype=float),
        generate_axial_points(q),
    ]
    arr = np.vstack(parts)
    rounded = np.round(arr, 12)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(idx)]
    validate_weights(arr)
    return arr


__all__ = [
    "simplex_lattice_count",
    "generate_simplex_lattice",
    "generate_simplex_centroid",
    "generate_extreme_vertices",
    "generate_axial_points",
    "sample_dirichlet",
    "build_study_design",
    "validate_weights",
    "custom_weights",
]
