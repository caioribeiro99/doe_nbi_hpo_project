"""Tests for the NBI core ported to the ensemble-weight simplex (mixens.nbi)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mixens.mixture_design import validate_weights
from mixens.nbi import (
    beta_lattice,
    lift_simplex,
    linear_cost,
    run_nbi_on_simplex,
    simplex_nbi_config,
)

M = 3  # components in the analytical tests

# Analytical bi-objective problem on the 3-simplex:
#   f1(w) = ||w - e1||^2  (minimum at vertex e1)
#   f2(w) = ||w - e2||^2  (minimum at vertex e2)
# Both convex; the Pareto set is the edge between e1 and e2 (w3 = 0).
E1 = np.array([1.0, 0.0, 0.0])
E2 = np.array([0.0, 1.0, 0.0])


def f1(w: np.ndarray) -> float:
    return float(np.sum((w - E1) ** 2))


def f2(w: np.ndarray) -> float:
    return float(np.sum((w - E2) ** 2))


@pytest.fixture(scope="module")
def nbi_result() -> dict:
    return run_nbi_on_simplex([f1, f2], M, n_points=9, n_starts=6, seed=0)


def test_lift_simplex_reconstructs_last_weight() -> None:
    w = lift_simplex(np.array([0.2, 0.3]))
    np.testing.assert_allclose(w, [0.2, 0.3, 0.5])
    validate_weights(w[None, :])


def test_lift_simplex_rejects_gross_violation() -> None:
    with pytest.raises(ValueError):
        lift_simplex(np.array([0.9, 0.9]))  # w_3 = -0.8


def test_beta_lattice_respects_budget() -> None:
    b = beta_lattice(2, 9)
    assert b.shape[1] == 2
    assert 2 <= b.shape[0] <= 9
    np.testing.assert_allclose(b.sum(axis=1), 1.0)
    b3 = beta_lattice(3, 66)
    assert b3.shape == (66, 3)  # {3,10} lattice fits exactly


def test_simplex_config_feasibility_encodes_last_weight() -> None:
    cfg = simplex_nbi_config(5, 3)
    assert cfg.bounds.shape == (4, 2)
    assert cfg.feasibility_constraint(np.array([0.2, 0.2, 0.2, 0.2]))[0] == pytest.approx(0.2)
    assert cfg.feasibility_constraint(np.array([0.5, 0.5, 0.1, 0.0]))[0] == pytest.approx(-0.1)


def test_candidates_live_on_the_simplex(nbi_result: dict) -> None:
    for c in nbi_result["candidates"]:
        w = np.asarray(c["w"])
        assert w.shape == (M,)
        assert (w >= -1e-9).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-8)


def test_anchors_recover_the_vertices(nbi_result: dict) -> None:
    anchors = np.asarray(nbi_result["anchors_w"])
    assert anchors.shape == (2, M)
    np.testing.assert_allclose(anchors[0], E1, atol=1e-4)
    np.testing.assert_allclose(anchors[1], E2, atol=1e-4)


def test_frontier_points_lie_between_the_extremes(nbi_result: dict) -> None:
    """The true Pareto set is the e1-e2 edge: w3 ~ 0 and w1+w2 ~ 1, with
    candidates spanning the segment between both anchors."""
    ws = np.asarray([c["w"] for c in nbi_result["candidates"] if c["success"]])
    assert len(ws) >= 5
    assert np.all(ws[:, 2] < 0.05)                     # on (or near) the edge
    f1_vals = np.array([f1(w) for w in ws])
    f2_vals = np.array([f2(w) for w in ws])
    assert f1_vals.min() < 0.1 and f1_vals.max() > 1.5  # spans toward both anchors
    assert f2_vals.min() < 0.1 and f2_vals.max() > 1.5
    order = np.argsort(f1_vals)                         # monotone trade-off
    assert np.all(np.diff(f2_vals[order]) <= 1e-6)


def test_payoff_and_normalization_metadata(nbi_result: dict) -> None:
    payoff = np.asarray(nbi_result["payoff_raw"])
    assert payoff.shape == (2, 2)
    # diagonal = utopia (each objective at its own minimizer ~ 0)
    assert np.allclose(np.diag(payoff), 0.0, atol=1e-3)
    assert nbi_result["normalized"] is True


def test_linear_cost() -> None:
    w = np.array([0.5, 0.3, 0.2])
    c = np.array([10.0, 1.0, 100.0])
    assert linear_cost(w, c) == pytest.approx(0.5 * 10 + 0.3 * 1 + 0.2 * 100)
    with pytest.raises(ValueError):
        linear_cost(w, c[:2])


def test_mixens_does_not_import_doe_xgb() -> None:
    """mixens must be self-contained: no import statement may reference
    doe_xgb (mentions in attribution docstrings/comments are allowed)."""
    import ast

    src = Path(__file__).resolve().parents[2] / "src" / "mixens"
    offenders = []
    for p in sorted(src.glob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any("doe_xgb" in n for n in names):
                offenders.append(f"{p.name}:{node.lineno}")
    assert offenders == [], f"mixens must not import doe_xgb: {offenders}"


# --- regression: failed subproblems must be reported, never crash the run ---


def test_project_free_vars_handles_feasible_and_infeasible_points() -> None:
    from mixens.nbi import project_free_vars

    w, feasible = project_free_vars(np.array([0.2, 0.3]))
    assert feasible
    np.testing.assert_allclose(w, [0.2, 0.3, 0.5])
    # sum(z) > 1: a failed subproblem's free variables; projected onto w_3 = 0
    w, feasible = project_free_vars(np.array([0.9, 0.9]))
    assert not feasible
    np.testing.assert_allclose(w, [0.5, 0.5, 0.0])
    validate_weights(w[None, :])


def test_run_nbi_on_simplex_survives_degenerate_vertex_anchor() -> None:
    """One anchor at a box corner (cheapest vertex of a linear cost) and the
    other in the interior: the beta-vertex subproblem at the corner is a
    degenerate KKT point where SLSQP may fail from every start (observed with
    the Santander log-loss x cost pair). The run must still return valid
    simplex weights and flag any unconverged subproblem as success=False."""
    costs = np.array([1.0, 10.0, 20.0, 30.0, 40.0])
    centre = np.full(5, 0.2)

    def cost(w: np.ndarray) -> float:
        return float(w @ costs)

    def quality(w: np.ndarray) -> float:  # convex, interior minimum
        return float(np.sum((w - centre) ** 2))

    res = run_nbi_on_simplex([quality, cost], 5, n_points=8, n_starts=4, seed=1)
    ws = np.asarray([c["w"] for c in res["candidates"]])
    validate_weights(ws)
    assert ws.shape == (8, 5)
    for c in res["candidates"]:
        if c["success"]:
            assert c["residual_norm"] < 1e-3
    assert sum(c["success"] for c in res["candidates"]) >= 4
