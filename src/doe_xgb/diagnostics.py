"""Frontier diagnostics that gate the conditional MBPA post-optimization.

A flat / compressed / weakly informative Pareto front is the signal the
EAAI 2025 article calls out as motivating post-optimization. We compute
a battery of diagnostics on the candidate set and emit a boolean
trigger flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .reporting import pareto_front


@dataclass(frozen=True)
class PostOptTriggerThresholds:
    min_normalized_obj_range: float = 0.05
    min_avg_pairwise_dist: float = 0.05
    min_unique_nondominated: int = 5
    max_weight_concentration_top1: float = 0.7
    min_curvature_score: float = 0.05
    min_spread: float = 0.5


@dataclass(frozen=True)
class FrontierDiagnostics:
    normalized_ranges: np.ndarray
    avg_pairwise_distance: float
    unique_nondominated: int
    weight_concentration: float
    curvature_score: float
    spread: float
    triggers: Dict[str, bool] = field(default_factory=dict)
    summary: Dict[str, object] = field(default_factory=dict)


def _avg_pairwise_distance(F: np.ndarray) -> float:
    n = F.shape[0]
    if n < 2:
        return 0.0
    diffs = F[:, None, :] - F[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.mean(dist[iu]))


def _curvature_score(F: np.ndarray) -> float:
    """Crude curvature proxy: 1 - linearity (R^2 of axis-line fit)."""
    F = np.asarray(F, dtype=float)
    n, q = F.shape
    if n < 3 or q < 2:
        return 0.0
    # Fit a line through min and max points along the first column.
    order = np.argsort(F[:, 0])
    Fs = F[order]
    a, b = Fs[0], Fs[-1]
    line_dir = b - a
    line_n = np.linalg.norm(line_dir)
    if line_n == 0.0:
        return 0.0
    # Perpendicular distance of each point to the line.
    diff = F - a
    proj = (diff @ line_dir) / (line_n ** 2)
    closest = a + proj[:, None] * line_dir
    perp = np.linalg.norm(F - closest, axis=1)
    rng = np.maximum(F.max(axis=0) - F.min(axis=0), 1e-12)
    return float(np.mean(perp / np.linalg.norm(rng)))


def _weight_concentration_top1(weights: np.ndarray) -> float:
    """Maximum across rows of max(weight) -- 1.0 if any vertex weight."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    return float(np.max(w.max(axis=1)))


def evaluate_frontier(
    F: np.ndarray,
    weights: np.ndarray,
    thresholds: PostOptTriggerThresholds,
) -> FrontierDiagnostics:
    """Compute frontier diagnostics and trigger flags on canonical objectives."""
    F = np.asarray(F, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if F.ndim != 2 or weights.ndim != 2:
        raise ValueError("F and weights must be 2-D")

    rng_per_col = F.max(axis=0) - F.min(axis=0)
    overall = float(np.max(np.abs(F).max(axis=0)) + 1e-12)
    normalized_ranges = rng_per_col / overall

    avg_dist = _avg_pairwise_distance(F)
    nd_mask = pareto_front(F)
    unique_nd = int(np.sum(nd_mask))
    weight_conc = _weight_concentration_top1(weights)
    curvature = _curvature_score(F)

    # Spread (delta) reused from reporting:
    from .reporting import spread_delta

    spread = spread_delta(F)

    triggers = {
        "low_obj_range": bool(np.any(normalized_ranges < thresholds.min_normalized_obj_range)),
        "low_avg_pairwise_dist": bool(avg_dist < thresholds.min_avg_pairwise_dist),
        "few_unique_nondominated": bool(unique_nd < thresholds.min_unique_nondominated),
        "high_weight_concentration": bool(weight_conc > thresholds.max_weight_concentration_top1),
        "low_curvature": bool(curvature < thresholds.min_curvature_score),
        "low_spread": bool(spread < thresholds.min_spread),
    }
    triggers["any"] = bool(any(triggers.values()))

    summary = {
        "normalized_ranges": normalized_ranges.tolist(),
        "avg_pairwise_distance": avg_dist,
        "unique_nondominated": unique_nd,
        "weight_concentration_top1": weight_conc,
        "curvature_score": curvature,
        "spread": spread,
        "thresholds": thresholds.__dict__,
    }
    return FrontierDiagnostics(
        normalized_ranges=normalized_ranges,
        avg_pairwise_distance=avg_dist,
        unique_nondominated=unique_nd,
        weight_concentration=weight_conc,
        curvature_score=curvature,
        spread=spread,
        triggers=triggers,
        summary=summary,
    )


def should_post_optimize(diag: FrontierDiagnostics) -> bool:
    """Trigger MBPA whenever any of the gates fired."""
    return bool(diag.triggers.get("any", False))


__all__ = [
    "PostOptTriggerThresholds",
    "FrontierDiagnostics",
    "evaluate_frontier",
    "should_post_optimize",
]
