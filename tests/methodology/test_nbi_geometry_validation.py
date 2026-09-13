"""Geometry validation for the NBI core.

The existing tests in ``test_nbi_core.py`` exercise separable convex quadratics.
On a convex Pareto front weighted-sum scalarization and NBI agree, so those tests
cannot distinguish a correct NBI from a weighted sum wearing its name. The tests
here use a problem with a provably **non-convex** Pareto front, where the two
methods must disagree, and check the NBI output against a closed-form solution.

The test problem, on ``x in [0, 1]``:

    f1(x) = x
    f2(x) = 1 - x^2

Every ``x`` is Pareto optimal, and the front is the curve ``f2 = 1 - f1^2``, whose
second derivative is ``-2``: the attainable region is non-convex.

*Weighted sum.* Minimizing ``w*f1 + (1-w)*f2`` has second derivative
``-2(1-w) < 0`` for ``w < 1``, so every interior stationary point is a maximum and
the minimum is attained at an endpoint. Weighted sum therefore recovers only the
two anchors, whatever grid of weights it is given.

*NBI.* The anchors are ``x=0`` and ``x=1``, so ``utopia = (0, 0)`` and
``Phi = [[0, 1], [1, 0]]``. The subproblem constrains ``f1 - f2`` to equal
``beta_2 - beta_1``, giving ``x^2 + x - 1 - (beta_1 - beta_2) = 0`` and hence

    x*(beta) = (-1 + sqrt(5 - 4*(beta_1 - beta_2))) / 2

which is an interior point for every interior ``beta``. NBI must reproduce it.
"""
from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.nbi_core import NBIConfig, build_chim, compute_anchors, run_nbi

BOUNDS = np.array([[0.0, 1.0]])


def f1(x: np.ndarray) -> float:
    return float(x[0])


def f2(x: np.ndarray) -> float:
    return float(1.0 - x[0] ** 2)


def analytic_x(beta: np.ndarray) -> float:
    """Closed-form NBI solution for the concave test front."""
    return float((-1.0 + np.sqrt(5.0 - 4.0 * (beta[0] - beta[1]))) / 2.0)


def simplex_weights(n: int) -> np.ndarray:
    w = np.linspace(0.0, 1.0, n)
    return np.column_stack([1.0 - w, w])


# ---------------------------------------------------------------------------
# The front really is non-convex, and weighted sum really cannot reach it
# ---------------------------------------------------------------------------


def test_test_problem_front_is_non_convex() -> None:
    """The chord between two front points lies strictly inside the front's hull."""
    xs = np.linspace(0.0, 1.0, 101)
    front = np.column_stack([xs, 1.0 - xs**2])
    a, b = front[0], front[-1]
    mid_chord = 0.5 * (a + b)
    # The front point with the same f1 as the chord midpoint.
    on_front = np.array([mid_chord[0], 1.0 - mid_chord[0] ** 2])
    # Non-convex (concave) front: the front bulges AWAY from the utopia point,
    # so the front point is dominated by the chord, not the other way round.
    assert on_front[1] > mid_chord[1] + 1e-6


@pytest.mark.parametrize("w", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_weighted_sum_recovers_only_anchors_on_non_convex_front(w: float) -> None:
    """Every interior weight sends weighted-sum scalarization to an endpoint."""
    xs = np.linspace(0.0, 1.0, 2001)
    vals = w * xs + (1.0 - w) * (1.0 - xs**2)
    x_hat = float(xs[int(np.argmin(vals))])
    assert min(x_hat, 1.0 - x_hat) < 1e-6, (
        f"weighted sum at w={w} returned interior x={x_hat}; the front is concave, "
        "so it should return an anchor"
    )


# ---------------------------------------------------------------------------
# NBI geometry
# ---------------------------------------------------------------------------


def test_anchors_and_payoff_matrix_on_non_convex_front() -> None:
    cfg = NBIConfig(objective_count=2, bounds=BOUNDS, n_starts=5, seed=1)
    anchors = compute_anchors([f1, f2], cfg)
    np.testing.assert_allclose(anchors.utopia, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(anchors.F_star, [[0.0, 1.0], [1.0, 0.0]], atol=1e-6)
    chim = build_chim(anchors, cfg)
    # Phi has a zero diagonal by construction, and the quasi-normal is the
    # normalized negative row sum.
    np.testing.assert_allclose(np.diag(chim.Phi), [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(chim.n_hat, [-1 / np.sqrt(2), -1 / np.sqrt(2)], atol=1e-6)


def test_nbi_recovers_the_closed_form_non_convex_front() -> None:
    """The decisive test: interior beta must give the analytic interior solution."""
    cfg = NBIConfig(objective_count=2, bounds=BOUNDS, n_starts=5, seed=1)
    weights = simplex_weights(9)
    run = run_nbi([f1, f2], weights, cfg)
    for cand in run.candidates:
        want = analytic_x(cand.beta)
        assert cand.success, f"subproblem failed at beta={cand.beta}"
        assert cand.residual_norm < 1e-6, (
            f"equality constraint violated at beta={cand.beta}: "
            f"residual {cand.residual_norm:.2e}"
        )
        assert abs(cand.x[0] - want) < 1e-4, (
            f"beta={cand.beta}: NBI returned x={cand.x[0]:.6f}, analytic {want:.6f}"
        )


def test_nbi_reaches_interior_points_weighted_sum_cannot() -> None:
    """At least the interior weights must land strictly between the anchors."""
    cfg = NBIConfig(objective_count=2, bounds=BOUNDS, n_starts=5, seed=1)
    run = run_nbi([f1, f2], simplex_weights(9), cfg)
    xs = np.array([c.x[0] for c in run.candidates])
    interior = xs[(xs > 1e-3) & (xs < 1 - 1e-3)]
    assert len(interior) >= 7, f"only {len(interior)} interior points: {xs}"


def test_t_is_negative_on_a_concave_front() -> None:
    """A concave front sits on the far side of the CHIM, so t must go negative.

    This is the property the earlier ``t >= 0`` bound made unreachable.
    """
    cfg = NBIConfig(objective_count=2, bounds=BOUNDS, n_starts=5, seed=1)
    run = run_nbi([f1, f2], simplex_weights(9), cfg)
    interior = [c for c in run.candidates if 1e-3 < c.x[0] < 1 - 1e-3]
    assert interior, "no interior candidates to inspect"
    assert all(c.t < 0 for c in interior), [float(c.t) for c in interior]


def test_restricting_t_to_nonnegative_breaks_the_non_convex_case() -> None:
    """Guards the reason the default changed: the restricted solver cannot do this."""
    cfg = NBIConfig(objective_count=2, bounds=BOUNDS, n_starts=5, seed=1,
                    restrict_t_nonnegative=True)
    run = run_nbi([f1, f2], simplex_weights(9), cfg)
    interior = [c for c in run.candidates if 1e-6 < c.beta[0] < 1 - 1e-6]
    assert not any(c.success and c.residual_norm < 1e-6 for c in interior), (
        "the t >= 0 solver unexpectedly certified an interior subproblem; if this "
        "now passes, the non-negativity restriction is no longer the binding issue"
    )


# ---------------------------------------------------------------------------
# Three objectives, still non-convex
# ---------------------------------------------------------------------------


def test_nbi_q3_non_convex_front_is_feasible_and_interior() -> None:
    """A concave three-objective front on the unit simplex of decision variables."""

    def g(i: int):
        def _g(x: np.ndarray) -> float:
            # Each objective is minimized at a different vertex; the 1 - x^2 shape
            # makes the attainable set non-convex in every pairwise projection.
            return float(1.0 - x[i] ** 2)
        return _g

    bounds = np.array([[0.0, 1.0]] * 3)

    def on_simplex(x: np.ndarray) -> np.ndarray:
        # sum(x) <= 1 keeps the objectives in genuine conflict.
        return np.array([1.0 - float(np.sum(x))])

    cfg = NBIConfig(objective_count=3, bounds=bounds, n_starts=6, seed=3,
                    feasibility_constraint=on_simplex, maxiter=800)
    weights = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
    run = run_nbi([g(0), g(1), g(2)], weights, cfg)
    solved = [c for c in run.candidates if c.success and c.residual_norm < 1e-4]
    assert len(solved) == len(weights), [
        (c.beta.tolist(), c.success, float(c.residual_norm)) for c in run.candidates
    ]
    for c in solved:
        assert np.sum(c.x) <= 1.0 + 1e-6
