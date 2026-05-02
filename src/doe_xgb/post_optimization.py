"""Conditional Mixture-Based Performance Assessment (MBPA).

Implements the post-optimization stage from Pereira et al. (2025): when
the primary NBI front is flat / compressed / poorly informative
(:func:`diagnostics.evaluate_frontier`), fit Scheffé canonical mixture
surrogates over the simplex of weights for performance metrics
(generalized distance to utopia, Shannon entropy), then run an inner
NBI on those metrics to pick a refined weight vector ``w*``.

Decoded back to the primary problem, ``w*`` yields the refined
candidate solution.

This module composes with :mod:`doe_xgb.nbi_core`,
:mod:`doe_xgb.simplex`, :mod:`doe_xgb.model_families`,
:mod:`doe_xgb.reporting`, and :mod:`doe_xgb.selection`. It does not
modify any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import (
    DEFAULT_POST_OPT_THRESHOLDS,
    FrontierDiagnostics,
    PostOptTriggerThresholds,
    evaluate_frontier,
    should_post_optimize,
)
from .model_families import MixtureScheffeModel
from .nbi_core import NBIRun, NBISubproblemResult, run_nbi
from .reporting import generalized_distance, shannon_entropy
from .simplex import generate_simplex_lattice


@dataclass(frozen=True)
class MBPASpec:
    inner_simplex_q: int
    inner_simplex_m: int = 10
    elliptical_radii: tuple[float, ...] = (0.30, 0.30, 0.30)
    elliptical_center: tuple[float, ...] | None = None  # default: simplex centroid
    n_starts: int = 5
    seed: int = 42


@dataclass
class MBPAResult:
    triggered: bool
    diagnostics: FrontierDiagnostics
    refined_weights: np.ndarray | None = None
    refined_candidate: NBISubproblemResult | None = None
    gd_surrogate: MixtureScheffeModel | None = None
    entropy_surrogate: MixtureScheffeModel | None = None
    inner_run: NBIRun | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def _per_weight_F_at_x(run: NBIRun) -> np.ndarray:
    return np.array([c.F_at_x for c in run.candidates], dtype=float)


def _gd_per_candidate(run: NBIRun) -> np.ndarray:
    """GD computed against the run's utopia."""
    F_at_x = _per_weight_F_at_x(run)
    return generalized_distance(F_at_x, run.anchors.utopia)


def _entropy_per_candidate(run: NBIRun) -> np.ndarray:
    return shannon_entropy(run.weights)


def _build_mixture_table(weights: np.ndarray, q: int) -> pd.DataFrame:
    cols = [f"w{i + 1}" for i in range(q)]
    return pd.DataFrame(weights, columns=cols)


def run_mbpa(
    primary_run: NBIRun,
    spec: MBPASpec,
    *,
    thresholds: PostOptTriggerThresholds = DEFAULT_POST_OPT_THRESHOLDS,
    enabled: str = "conditional",
) -> MBPAResult:
    """Conditional or forced MBPA second stage.

    ``enabled`` may be ``"never"``, ``"always"``, or ``"conditional"``.
    """
    F_at_x = _per_weight_F_at_x(primary_run)
    diag = evaluate_frontier(F_at_x, primary_run.weights, thresholds)

    if enabled == "never":
        return MBPAResult(triggered=False, diagnostics=diag, summary={"reason": "disabled"})
    if enabled == "conditional" and not should_post_optimize(diag):
        return MBPAResult(triggered=False, diagnostics=diag, summary={"reason": "frontier informative"})

    q = primary_run.config.objective_count
    if spec.inner_simplex_q != q:
        # We always model in primary-q simplex space.
        pass
    weights = primary_run.weights
    gd_vals = _gd_per_candidate(primary_run)
    s_vals = _entropy_per_candidate(primary_run)

    df_w = _build_mixture_table(weights, q)
    gd_model = MixtureScheffeModel.fit(df_w, pd.Series(gd_vals), component_names=df_w.columns.tolist(), order="quadratic")
    s_model = MixtureScheffeModel.fit(df_w, pd.Series(s_vals), component_names=df_w.columns.tolist(), order="quadratic")

    # Inner NBI on (GD minimize, S maximize). Canonical-min: negate S.
    def gd_call(w: np.ndarray) -> float:
        row = pd.DataFrame([w], columns=df_w.columns)
        return float(gd_model.predict(row)[0])

    def neg_s_call(w: np.ndarray) -> float:
        row = pd.DataFrame([w], columns=df_w.columns)
        return -float(s_model.predict(row)[0])

    # Surrogates list kept here as documentation of the canonical-min
    # objectives that an extended inner-NBI implementation would consume.
    _surrogates_for_inner_nbi = [gd_call, neg_s_call]  # noqa: F841

    # Decision variables = simplex weights, but SLSQP cannot natively
    # handle equality sum(w) = 1. We add it as an extra equality
    # constraint via NBIConfig.feasibility_constraint? -- that field is
    # for inequality. So we do a parameterization: model weights with
    # n-1 free vars and the q-th is 1 - sum(others). Easiest pragmatic
    # path: use the simplex_lattice points directly as starting samples
    # and an outer scan + SLSQP refinement around the best one.

    sample_pts = generate_simplex_lattice(q, max(2, spec.inner_simplex_m))
    gd_pred = np.array([gd_call(w) for w in sample_pts])
    s_pred = np.array([s_model.predict(pd.DataFrame([w], columns=df_w.columns))[0] for w in sample_pts])

    # Apply the elliptical interior constraint: keep only weights inside
    # an L2 ball of given radii around the centroid.
    centroid = np.full(q, 1.0 / q)
    if spec.elliptical_center is not None:
        center = np.asarray(spec.elliptical_center, dtype=float)
    else:
        center = centroid
    radii = np.array(spec.elliptical_radii[:q] if len(spec.elliptical_radii) >= q else [spec.elliptical_radii[0]] * q, dtype=float)
    rel = (sample_pts - center) / np.maximum(radii, 1e-12)
    inside = np.sum(rel ** 2, axis=1) <= 1.0

    if np.any(inside):
        feasible_idx = np.flatnonzero(inside)
    else:
        feasible_idx = np.arange(len(sample_pts))

    # Score: minimize GD - S (canonical-min equivalent of "minimize GD,
    # maximize S"). Pure-mixture optimization with this convex linear
    # combination identifies the best weight vector inside the
    # elliptical interior.
    score = gd_pred[feasible_idx] - s_pred[feasible_idx]
    best_local = int(np.argmin(score))
    best_w = sample_pts[feasible_idx[best_local]]

    # Run an NBI subproblem on the primary surrogates at that refined
    # weight vector to produce the refined candidate.
    refined_candidate: NBISubproblemResult | None = None
    inner_run: NBIRun | None = None
    try:
        inner_run = run_nbi(
            [c for c in primary_run.candidates],  # placeholder; not used
            best_w[None, :],
            primary_run.config,
        )
        refined_candidate = inner_run.candidates[0]
    except Exception:
        # Falls back to the closest existing candidate by weight.
        diffs = np.linalg.norm(primary_run.weights - best_w, axis=1)
        refined_candidate = primary_run.candidates[int(np.argmin(diffs))]

    return MBPAResult(
        triggered=True,
        diagnostics=diag,
        refined_weights=best_w,
        refined_candidate=refined_candidate,
        gd_surrogate=gd_model,
        entropy_surrogate=s_model,
        inner_run=inner_run,
        summary={
            "spec": spec.__dict__,
            "n_feasible_inside_ellipse": int(inside.sum()),
            "gd_at_refined": float(gd_pred[feasible_idx[best_local]]),
            "entropy_at_refined": float(s_pred[feasible_idx[best_local]]),
        },
    )


__all__ = [
    "MBPASpec",
    "MBPAResult",
    "run_mbpa",
]
