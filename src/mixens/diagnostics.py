"""Design-matrix diagnostics (rank, condition number).

Ported from doe_nbi_hpo_project, branch ``repo-publication-readiness``
(commit 0465466), file ``src/doe_xgb/design/diagnostics.py``. Original
author: Caio Tertuliano Ribeiro (MIT License). The box-coded coverage
helper was dropped (not applicable to simplex designs).
"""

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


__all__ = ["matrix_diagnostics"]
