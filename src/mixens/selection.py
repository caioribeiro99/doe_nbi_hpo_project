"""Pareto utilities and MCDM selection rules for the postwork NBI study.

The ``SelectionRule``/``select`` block is ported from doe_nbi_hpo_project @
origin/repo-publication-readiness:0465466, file ``src/doe_xgb/selection.py``,
adapted for PCO213 postwork (original author: Caio Tertuliano Ribeiro, MIT
License; numpy-only, no ``doe_xgb`` dependency). The Pareto helpers
(``pareto_filter``, ``normalize_objectives``, ``spacing_metric``,
``generational_distance``, ``inverted_generational_distance``) are new.

Convention everywhere: objectives in an (N, q) array, MINIMIZATION
(lower = better). Wrap "maximize AUC" as -AUC before calling.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Pareto helpers (PCO213 postwork additions)
# ---------------------------------------------------------------------------


def pareto_filter(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows of F (minimization convention).

    Row i is dominated if some row j is <= in every objective and < in at
    least one. Duplicate rows are kept (both non-dominated).
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2-D (N, q)")
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominates_i.any():
            mask[i] = False
    return mask


def normalize_objectives(
    F: np.ndarray,
    utopia: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
) -> np.ndarray:
    """Rescale objectives so utopia -> 0 and nadir -> 1 (per column).

    Defaults to the column-wise min/max of F. Zero-span columns map to 0.
    """
    F = np.asarray(F, dtype=float)
    lo = np.asarray(utopia, dtype=float) if utopia is not None else F.min(axis=0)
    hi = np.asarray(nadir, dtype=float) if nadir is not None else F.max(axis=0)
    span = hi - lo
    span = np.where(np.abs(span) < 1e-15, 1.0, span)
    return (F - lo) / span


def spacing_metric(front: np.ndarray) -> float:
    """Schott's spacing: std of nearest-neighbour distances along a front.

    0 = perfectly even spacing; needs >= 3 points (returns nan otherwise).
    """
    front = np.asarray(front, dtype=float)
    n = front.shape[0]
    if n < 3:
        return float("nan")
    d = np.full(n, np.inf)
    for i in range(n):
        diff = np.abs(front - front[i]).sum(axis=1)  # L1, Schott's convention
        diff[i] = np.inf
        d[i] = diff.min()
    return float(np.std(d, ddof=1))


def generational_distance(front: np.ndarray, reference: np.ndarray) -> float:
    """GD: mean Euclidean distance from each front point to the nearest
    reference point (how CLOSE the obtained front is to the true one)."""
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    d = np.array([np.linalg.norm(reference - p, axis=1).min() for p in front])
    return float(d.mean())


def inverted_generational_distance(front: np.ndarray, reference: np.ndarray) -> float:
    """IGD: mean distance from each reference point to the nearest front
    point (how well the obtained front COVERS the true one)."""
    return generational_distance(reference, front)


# ---------------------------------------------------------------------------
# MCDM selection rules (ported)
# ---------------------------------------------------------------------------


class SelectionRule(str, Enum):
    MAX_QUALITY = "max_quality"
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
    weights: Sequence[float] | None = None,
    utopia: Sequence[float] | None = None,
    quality_index: int = 0,
) -> tuple[int, dict[str, object]]:
    """Pick a row of ``F`` (canonical-min) according to ``rule``."""
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("F must be 2-D (N, q)")
    _, q = F.shape

    if rule is SelectionRule.MAX_QUALITY:
        idx = int(np.argmin(F[:, quality_index]))
        return idx, {"rule": rule.value, "quality_index": quality_index}

    if rule is SelectionRule.DISTANCE_TO_UTOPIA:
        Fn = _minmax_normalize(F)
        if utopia is None:
            uto_n = np.zeros(q, dtype=float)
        else:
            uto = np.asarray(utopia, dtype=float)
            lo = F.min(axis=0)
            hi = F.max(axis=0)
            den = np.where(hi - lo == 0.0, 1.0, hi - lo)
            uto_n = (uto - lo) / den
        d2 = np.sum((Fn - uto_n) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return idx, {"rule": rule.value, "distance": float(np.sqrt(d2[idx]))}

    if rule is SelectionRule.UTILITY:
        w = (np.ones(q) / q) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != (q,):
            raise ValueError(f"weights must have length {q}")
        w = w / float(w.sum())
        Fn = _minmax_normalize(F)
        idx = int(np.argmin((Fn * w).sum(axis=1)))
        return idx, {"rule": rule.value, "weights": w.tolist()}

    if rule is SelectionRule.KNEE:
        Fn = _minmax_normalize(F)
        _, q_cols = Fn.shape
        axis_extremes = np.array(
            [Fn[int(np.argmin(Fn[:, j]))] for j in range(q_cols)], dtype=float
        )
        if q_cols == 2:
            a, b = axis_extremes[0], axis_extremes[1]
            line = b - a
            line_n = float(np.linalg.norm(line))
            if line_n == 0.0:
                return (int(np.argmin(np.linalg.norm(Fn, axis=1))),
                        {"rule": rule.value, "knee_distance": 0.0})
            normal = np.array([-line[1], line[0]], dtype=float) / line_n
            d = np.abs((Fn - a) @ normal)
        else:
            anchor = axis_extremes[0]
            diffs = axis_extremes[1:] - anchor
            _, _, vh = np.linalg.svd(diffs)
            normal = vh[-1]
            d = np.abs((Fn - anchor) @ normal)
        idx = int(np.argmax(d))
        return idx, {"rule": rule.value, "knee_distance": float(d[idx])}

    if rule is SelectionRule.LEXICOGRAPHIC:
        order = np.lexsort([F[:, i] for i in reversed(range(q))])
        return int(order[0]), {"rule": rule.value}

    if rule is SelectionRule.TOPSIS:
        w = (np.ones(q) / q) if weights is None else np.asarray(weights, dtype=float)
        w = w / float(w.sum())
        norms = np.linalg.norm(F, axis=0)
        norms = np.where(norms == 0.0, 1.0, norms)
        V = (F / norms) * w
        ideal = V.min(axis=0)
        anti = V.max(axis=0)
        d_ideal = np.linalg.norm(V - ideal, axis=1)
        d_anti = np.linalg.norm(V - anti, axis=1)
        denom = d_ideal + d_anti
        denom = np.where(denom == 0.0, 1.0, denom)
        score = d_anti / denom
        idx = int(np.argmax(score))
        return idx, {"rule": rule.value, "score": float(score[idx])}

    raise ValueError(f"unknown selection rule: {rule}")


__all__ = [
    "SelectionRule",
    "generational_distance",
    "inverted_generational_distance",
    "normalize_objectives",
    "pareto_filter",
    "select",
    "spacing_metric",
]
