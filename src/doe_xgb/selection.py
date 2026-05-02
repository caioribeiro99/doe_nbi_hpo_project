"""MCDM rules for picking a final candidate from a Pareto set.

All rules accept a (N, q) array of canonicalized objective values
(minimization-convention) and return ``(index, info)`` with the chosen
row plus a diagnostics dict.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


class SelectionRule(str, Enum):
    MAX_QUALITY = "max_quality"
    MAX_ACCURACY = "max_accuracy"
    DISTANCE_TO_UTOPIA = "distance_to_utopia"
    KNEE = "knee"
    UTILITY = "utility"
    LEXICOGRAPHIC = "lexicographic"
    TOPSIS = "topsis"


def _minmax_normalize(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    lo = F.min(axis=0)
    hi = F.max(axis=0)
    den = np.where(hi - lo == 0.0, 1.0, hi - lo)
    return (F - lo) / den


def select(
    F: np.ndarray,
    rule: SelectionRule,
    *,
    weights: Optional[Sequence[float]] = None,
    utopia: Optional[Sequence[float]] = None,
    quality_index: int = 0,
) -> Tuple[int, Dict[str, object]]:
    """Pick a row of ``F`` (canonical-min) according to ``rule``.

    Parameters
    ----------
    F:
        (N, q) canonicalized objectives (lower = better).
    rule:
        Selection strategy.
    weights:
        For UTILITY / TOPSIS: per-objective weights (length q).
    utopia:
        For DISTANCE_TO_UTOPIA: (q,). If None, the per-column min is used.
    quality_index:
        For MAX_QUALITY / MAX_ACCURACY: which column to maximize
        (canonical-min, so the *minimum* value is the "best quality").
    """

    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2-D (N, q)")
    n, q = F.shape

    if rule in (SelectionRule.MAX_QUALITY, SelectionRule.MAX_ACCURACY):
        # Canonical-min => "lower is better" => "max quality" = argmin column.
        idx = int(np.argmin(F[:, quality_index]))
        return idx, {"rule": rule.value, "quality_index": quality_index}

    if rule is SelectionRule.DISTANCE_TO_UTOPIA:
        Fn = _minmax_normalize(F)
        if utopia is None:
            uto_n = np.zeros(q, dtype=float)
        else:
            # Normalize the supplied utopia using the same min/max as F.
            uto = np.asarray(utopia, dtype=float)
            lo = F.min(axis=0)
            hi = F.max(axis=0)
            den = np.where(hi - lo == 0.0, 1.0, hi - lo)
            uto_n = (uto - lo) / den
        d2 = np.sum((Fn - uto_n) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, {"rule": rule.value, "distance": float(np.sqrt(d2[idx]))}

    if rule is SelectionRule.UTILITY:
        if weights is None:
            w = np.ones(q, dtype=float) / q
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (q,):
                raise ValueError(f"weights must have length {q}")
            w = w / float(w.sum())
        Fn = _minmax_normalize(F)
        u = (Fn * w).sum(axis=1)  # canonical-min: lower is better
        idx = int(np.argmin(u))
        return idx, {"rule": rule.value, "weights": w.tolist()}

    if rule is SelectionRule.KNEE:
        Fn = _minmax_normalize(F)
        n_rows, q_cols = Fn.shape
        # Compute the q "axis extremes": the row that minimizes each
        # column j. These define a (q-1)-dimensional hyperplane in
        # normalized objective space. The knee is the row farthest
        # from that hyperplane (largest perpendicular distance).
        axis_extremes = np.array(
            [Fn[int(np.argmin(Fn[:, j]))] for j in range(q_cols)],
            dtype=float,
        )
        if q_cols == 2:
            a, b = axis_extremes[0], axis_extremes[1]
            line = b - a
            line_n = float(np.linalg.norm(line))
            if line_n == 0.0:
                return (
                    int(np.argmin(np.linalg.norm(Fn, axis=1))),
                    {"rule": rule.value, "knee_distance": 0.0},
                )
            # 2D normal vector.
            normal = np.array([-line[1], line[0]], dtype=float) / line_n
            d = np.abs((Fn - a) @ normal)
        else:
            anchor = axis_extremes[0]
            diffs = axis_extremes[1:] - anchor
            # SVD: smallest right-singular vector spans the normal direction.
            _, _, vh = np.linalg.svd(diffs)
            normal = vh[-1]
            d = np.abs((Fn - anchor) @ normal)
        idx = int(np.argmax(d))
        return idx, {"rule": rule.value, "knee_distance": float(d[idx])}

    if rule is SelectionRule.LEXICOGRAPHIC:
        # Sort by every column in turn; first row wins.
        order = np.lexsort([F[:, i] for i in reversed(range(q))])
        return int(order[0]), {"rule": rule.value}

    if rule is SelectionRule.TOPSIS:
        if weights is None:
            w = np.ones(q, dtype=float) / q
        else:
            w = np.asarray(weights, dtype=float)
            w = w / float(w.sum())
        # Vector normalize.
        norms = np.linalg.norm(F, axis=0)
        norms = np.where(norms == 0.0, 1.0, norms)
        V = (F / norms) * w
        ideal = V.min(axis=0)
        anti = V.max(axis=0)
        d_ideal = np.linalg.norm(V - ideal, axis=1)
        d_anti = np.linalg.norm(V - anti, axis=1)
        denom = d_ideal + d_anti
        denom = np.where(denom == 0.0, 1.0, denom)
        score = d_anti / denom  # higher = closer to ideal
        idx = int(np.argmax(score))
        return idx, {"rule": rule.value, "score": float(score[idx])}

    raise ValueError(f"unknown selection rule: {rule}")


__all__ = ["SelectionRule", "select"]
