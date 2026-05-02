"""Design-matrix diagnostics: rank, condition number, estimability, etc."""

from __future__ import annotations

import numpy as np


def matrix_diagnostics(X: np.ndarray) -> dict[str, float]:
    """Compute rank, condition number, and basic shape diagnostics."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError("X must be 2-D")
    n, p = arr.shape
    rank = int(np.linalg.matrix_rank(arr))
    s = np.linalg.svd(arr, compute_uv=False)
    if s.size == 0 or s[-1] == 0.0:
        cond = float("inf")
    else:
        cond = float(s[0] / s[-1])
    return {
        "n_runs": int(n),
        "n_columns": int(p),
        "rank": rank,
        "rank_deficient": bool(rank < min(n, p)),
        "condition_number": cond,
        "max_singular": float(s[0]) if s.size else 0.0,
        "min_singular": float(s[-1]) if s.size else 0.0,
    }


def design_coverage(coded: np.ndarray) -> dict[str, float]:
    """Per-factor coverage of the coded box [-1, +1]."""
    arr = np.asarray(coded, dtype=float)
    if arr.ndim != 2:
        raise ValueError("coded must be 2-D")
    return {
        "min_per_factor": [float(v) for v in arr.min(axis=0)],
        "max_per_factor": [float(v) for v in arr.max(axis=0)],
        "covers_lower_bound": bool(np.all(arr.min(axis=0) <= -1.0 + 1e-9)),
        "covers_upper_bound": bool(np.all(arr.max(axis=0) >= 1.0 - 1e-9)),
    }


__all__ = ["matrix_diagnostics", "design_coverage"]
