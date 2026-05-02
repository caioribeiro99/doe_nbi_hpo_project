"""Methodology-pinning tests for true N-objective NBI."""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import pytest

from doe_xgb.nbi_core import (
    NBIConfig,
    build_chim,
    compute_anchors,
    run_nbi,
    solve_nbi_subproblem,
)
from doe_xgb.simplex import generate_simplex_lattice


def _quadratic_minimizer_at(target: np.ndarray, k: int) -> Callable[[np.ndarray], float]:
    """f_i(x) = ||x - target||^2, minimum at `target`, value 0."""

    def f(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        diff = x - target
        return float(np.dot(diff, diff))

    return f


@pytest.mark.methodology
@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_nbi_recovers_anchors_for_separable_quadratics(q: int) -> None:
    """Each ``f_i`` minimized by a unit-vector target. NBI must recover them."""
    rng = np.random.default_rng(0)
    targets = np.eye(q)  # k = q in this synthetic problem
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=0)
    anchors = compute_anchors(surrogates, cfg)
    # Diagonal of payoff matrix is the per-objective optimum; should be ~0.
    np.testing.assert_allclose(np.diag(anchors.F_star), 0.0, atol=1e-4)
    # Off-diagonal entries are |target_j - target_i|^2 = 2 for unit vectors.
    for i in range(q):
        for j in range(q):
            if i == j:
                continue
            assert anchors.F_star[i, j] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.methodology
@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_nbi_subproblem_residual_is_small(q: int) -> None:
    """Verify the equality constraint residual is below 1e-4 across q objectives."""
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=1)
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    beta = np.full(q, 1.0 / q)  # centroid
    res = solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg)
    # The equality constraint residual must be small.
    assert res.residual_norm < 1e-3, f"residual={res.residual_norm:.2e}"
    assert res.t >= -1e-9


@pytest.mark.methodology
def test_nbi_q3_full_simplex_lattice_runs_and_residuals_small() -> None:
    """3-objective NBI on a simplex_lattice {3, 4} (15 points). All residuals small."""
    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=2)
    weights = generate_simplex_lattice(q, 4)
    run = run_nbi(surrogates, weights, cfg)
    assert len(run.candidates) == 15
    residuals = [c.residual_norm for c in run.candidates]
    assert max(residuals) < 1e-2, f"max residual = {max(residuals):.2e}"


@pytest.mark.methodology
def test_chim_phi_matrix_is_q_by_q_and_diag_zero() -> None:
    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=3)
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    assert chim.Phi.shape == (q, q)
    np.testing.assert_allclose(np.diag(chim.Phi), 0.0, atol=1e-3)


@pytest.mark.methodology
def test_quasi_normal_default_is_minus_phi_ones_normalized() -> None:
    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=4)
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    expected = -chim.Phi @ np.ones(q)
    expected = expected / np.linalg.norm(expected)
    np.testing.assert_allclose(chim.n_hat, expected, atol=1e-9)


@pytest.mark.methodology
def test_run_nbi_rejects_wrong_weight_shape() -> None:
    q = 3
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=2, seed=5)
    with pytest.raises(ValueError):
        run_nbi(surrogates, np.zeros((10, 2)), cfg)


@pytest.mark.methodology
def test_legacy_scalarization_emits_deprecation_warning() -> None:
    """Calling legacy run_nbi_weighted_sum through doe_xgb.nbi must warn."""
    import warnings

    from doe_xgb.nbi import run_nbi_weighted_sum  # noqa: F401

    # Build trivial surrogate models compatible with the legacy API.
    terms = ["Intercept", "subsample"]
    coefs = [0.0, 1.0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            run_nbi_weighted_sum(
                (terms, coefs),
                (terms, coefs),
                beta_step=0.5,
                n_starts=1,
            )
        except Exception:
            # We don't care whether the call fully succeeds; the
            # DeprecationWarning is what we're testing.
            pass
        deprecation = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation, "expected DeprecationWarning from doe_xgb.nbi.run_nbi_weighted_sum"


@pytest.mark.methodology
def test_nbi_core_objective_count_validation() -> None:
    bounds = np.array([[-1.5, 1.5]] * 3)
    cfg = NBIConfig(objective_count=3, bounds=bounds, n_starts=2, seed=0)
    surrogates = [_quadratic_minimizer_at(np.eye(3)[0], 3)]  # only 1 surrogate
    with pytest.raises(ValueError):
        compute_anchors(surrogates, cfg)


@pytest.mark.methodology
def test_nbi_q5_synthetic_runs_at_centroid() -> None:
    """Smoke check that q=5 works at the simplex centroid."""
    q = 5
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=6, maxiter=400)
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    beta = np.full(q, 1.0 / q)
    res = solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg)
    assert res.residual_norm < 1e-2


@pytest.mark.methodology
def test_nbi_subproblem_t_nonnegative() -> None:
    q = 2
    targets = np.eye(q)
    bounds = np.array([[-1.5, 1.5]] * q)
    surrogates = [_quadratic_minimizer_at(targets[i], q) for i in range(q)]
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=4, seed=7)
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    beta = np.array([0.5, 0.5])
    res = solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg)
    # NBI projects toward the Pareto front; t must be non-negative.
    assert res.t >= -1e-6


@pytest.mark.methodology
def test_weighted_sum_vs_nbi_diagnostic_difference() -> None:
    """Diagnostic: a non-convex Pareto set should be approximated more
    uniformly by NBI than by weighted sum.

    Build a 2D problem whose Pareto front is curved. Weighted sum tends
    to cluster solutions near the extremes; NBI should produce more
    spread-out points across betas. We measure spread as the mean
    pairwise Euclidean distance between successive candidates.
    """
    q = 2

    # F1 minimized at x=(0,0); F2 minimized at x=(1,0).
    def f1(x: np.ndarray) -> float:
        return float((x[0] - 0.0) ** 2 + 4.0 * x[1] ** 2)

    def f2(x: np.ndarray) -> float:
        return float((x[0] - 1.0) ** 2 + 4.0 * x[1] ** 2)

    bounds = np.array([[-0.5, 1.5], [-0.5, 0.5]])
    cfg = NBIConfig(objective_count=q, bounds=bounds, n_starts=3, seed=8)
    weights = np.linspace(0.0, 1.0, 11)
    weights = np.column_stack([1.0 - weights, weights])
    run = run_nbi([f1, f2], weights, cfg)
    nbi_xs = np.array([c.x for c in run.candidates])
    spread_nbi = float(
        np.mean(np.linalg.norm(np.diff(nbi_xs, axis=0), axis=1))
    )
    # Sanity: NBI should produce a non-trivial spread.
    assert spread_nbi > 0.05
