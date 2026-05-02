"""Methodology tests for the conditional MBPA layer."""

from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.diagnostics import (
    PostOptTriggerThresholds,
    evaluate_frontier,
    should_post_optimize,
)
from doe_xgb.nbi_core import NBIConfig, run_nbi
from doe_xgb.simplex import generate_simplex_lattice


@pytest.mark.methodology
def test_flat_frontier_triggers_post_optimization() -> None:
    # Synthetic: all "candidates" near the same point in obj space.
    F = np.tile(np.array([0.5, 0.5]), (15, 1)) + np.random.default_rng(0).normal(0, 0.001, size=(15, 2))
    weights = generate_simplex_lattice(2, 14)
    diag = evaluate_frontier(F, weights, PostOptTriggerThresholds())
    assert should_post_optimize(diag) is True
    assert diag.triggers["any"] is True


@pytest.mark.methodology
def test_informative_frontier_skips_post_optimization() -> None:
    # A well-spread non-degenerate front (linear from utopia to nadir).
    F = np.column_stack([np.linspace(0.0, 1.0, 11), np.linspace(1.0, 0.0, 11)])
    weights = generate_simplex_lattice(2, 10)
    thresh = PostOptTriggerThresholds(
        min_normalized_obj_range=0.01,
        min_avg_pairwise_dist=0.01,
        min_unique_nondominated=2,
        max_weight_concentration_top1=1.01,  # vertices are allowed in this test
        min_curvature_score=0.0,
        min_spread=0.0,
    )
    diag = evaluate_frontier(F, weights, thresh)
    assert should_post_optimize(diag) is False


@pytest.mark.methodology
def test_post_optimization_runs_on_synthetic_quadratics() -> None:
    """Smoke: MBPA returns a refined candidate when forced on a 3-objective run."""
    from doe_xgb.post_optimization import MBPASpec, run_mbpa

    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(targets[i]) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=0, maxiter=300)
    weights = generate_simplex_lattice(q, 4)
    primary_run = run_nbi(surrogates, weights, cfg)
    spec = MBPASpec(inner_simplex_q=q, inner_simplex_m=6, elliptical_radii=(0.5, 0.5, 0.5))
    res = run_mbpa(primary_run, spec, enabled="always")
    assert res.triggered is True
    assert res.refined_weights is not None
    np.testing.assert_allclose(float(np.sum(res.refined_weights)), 1.0, atol=1e-9)
    assert (res.refined_weights >= 0).all()
    assert res.gd_surrogate is not None
    assert res.entropy_surrogate is not None


@pytest.mark.methodology
def test_post_optimization_disabled_returns_no_refinement() -> None:
    from doe_xgb.post_optimization import MBPASpec, run_mbpa

    q = 2

    def make_obj(t):
        return lambda x: float(np.dot(x - t, x - t))

    surrogates = [make_obj(np.array([0.0, 0.0])), make_obj(np.array([1.0, 0.0]))]
    bounds = np.array([[-0.5, 1.5], [-0.5, 0.5]])
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=0)
    weights = generate_simplex_lattice(q, 5)
    run = run_nbi(surrogates, weights, cfg)
    res = run_mbpa(run, MBPASpec(inner_simplex_q=q, inner_simplex_m=6), enabled="never")
    assert res.triggered is False
    assert res.refined_weights is None


@pytest.mark.methodology
def test_refined_weights_remain_on_simplex() -> None:
    from doe_xgb.post_optimization import MBPASpec, run_mbpa

    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)

    surrogates = [lambda x, t=targets[i]: float(np.dot(x - t, x - t)) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=1)
    weights = generate_simplex_lattice(q, 4)
    primary = run_nbi(surrogates, weights, cfg)
    spec = MBPASpec(inner_simplex_q=q, inner_simplex_m=6, elliptical_radii=(0.7, 0.7, 0.7))
    res = run_mbpa(primary, spec, enabled="always")
    assert res.refined_weights is not None
    np.testing.assert_allclose(float(np.sum(res.refined_weights)), 1.0, atol=1e-9)
    assert (res.refined_weights >= 0).all()
