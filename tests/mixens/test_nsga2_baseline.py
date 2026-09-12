"""Tests for the evaluation-matched NSGA-II baseline.

Covers the simplex repair (non-negativity, sum-to-one, idempotence, edge and vertex
inputs, degenerate rows), determinism under seed, the budget-to-generations rule, and
an end-to-end smoke run on a synthetic problem with a known trade-off.
"""
from __future__ import annotations

import numpy as np
import pytest

from mixens.nsga2_baseline import (
    POP_SIZE,
    SEED_BASE,
    SEED_OFFSET,
    n_gen_for_budget,
    repair_simplex,
    run_nsga2,
    sample_simplex,
)

pymoo = pytest.importorskip("pymoo")


# ----------------------------------------------------------------------------- repair
def test_repair_produces_valid_compositions():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))  # includes negatives and rows summing to <= 0
    W = repair_simplex(X)
    assert W.shape == X.shape
    assert (W >= 0).all(), "repair must yield non-negative weights"
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-12), "repair must yield sum-to-one weights"


def test_repair_is_idempotent():
    rng = np.random.default_rng(1)
    W1 = repair_simplex(rng.normal(size=(50, 5)))
    assert np.allclose(W1, repair_simplex(W1), atol=1e-15)


def test_repair_preserves_points_already_on_the_simplex():
    W = np.array([[0.2, 0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.0, 0.0, 0.0]])
    assert np.allclose(repair_simplex(W), W)


def test_repair_handles_vertices_and_edges():
    vertices = np.eye(5)
    assert np.allclose(repair_simplex(vertices), vertices)
    edge = np.array([[0.7, 0.3, 0.0, 0.0, 0.0]])
    assert np.allclose(repair_simplex(edge), edge)


def test_repair_degenerate_row_falls_back_to_uniform():
    X = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -2.0, -3.0, -4.0, -5.0]])
    W = repair_simplex(X)
    assert np.allclose(W, 0.2)


def test_repair_scales_rather_than_truncates():
    # a row that is already non-negative is rescaled, keeping its direction
    X = np.array([[2.0, 1.0, 1.0, 0.0, 0.0]])
    W = repair_simplex(X)
    assert np.allclose(W, [[0.5, 0.25, 0.25, 0.0, 0.0]])


# --------------------------------------------------------------------------- sampling
def test_sampling_is_on_the_simplex_and_deterministic():
    a = sample_simplex(40, 5, np.random.default_rng(123))
    b = sample_simplex(40, 5, np.random.default_rng(123))
    assert np.allclose(a, b), "same seed must give the same initial population"
    assert (a >= 0).all() and np.allclose(a.sum(axis=1), 1.0)
    c = sample_simplex(40, 5, np.random.default_rng(124))
    assert not np.allclose(a, c), "different seeds must give different populations"


# ----------------------------------------------------------------------------- budget
@pytest.mark.parametrize("target,expected_evals", [(6600, 6600), (424413, 424380), (1000, 990)])
def test_budget_rule_lands_within_one_generation(target, expected_evals):
    n_gen = n_gen_for_budget(target, POP_SIZE)
    assert POP_SIZE * n_gen == expected_evals
    assert abs(POP_SIZE * n_gen - target) <= POP_SIZE


def test_budget_rule_never_returns_zero_generations():
    assert n_gen_for_budget(1, POP_SIZE) == 1
    assert n_gen_for_budget(66, POP_SIZE) == 1
    assert n_gen_for_budget(0, POP_SIZE) == 1


def test_seed_policy_is_documented_and_distinct_per_replication():
    seeds = {SEED_BASE + r + SEED_OFFSET for r in range(30)}
    assert len(seeds) == 30
    # must not collide with any offset used by the frozen benchmark (max 12000)
    assert min(seeds) - SEED_BASE > 12000


# -------------------------------------------------------------------- end-to-end smoke
def _synthetic(n=400, m=5, seed=0):
    """A tiny cached-OOF stand-in with a genuine accuracy/calibration/cost trade-off."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    P = np.clip(rng.normal(loc=0.25 + 0.5 * y[:, None], scale=0.22, size=(n, m)), 1e-6, 1 - 1e-6)
    P[:, 2] = np.clip(P[:, 2] * 0.5 + 0.25, 1e-6, 1 - 1e-6)  # a poorly calibrated member
    costs = np.array([1.0, 1.0, 50.0, 4.0, 1.0])
    return P, y, costs


def test_smoke_run_returns_a_valid_nondominated_set():
    P, y, costs = _synthetic()
    res = run_nsga2(P, y, costs, dataset="synthetic", rep=0, target_evals=66 * 12)
    assert res.actual_evals == res.pop_size * res.n_gen
    assert abs(res.actual_evals - res.target_evals) <= res.pop_size
    for W in (res.W_final, res.W_nd):
        assert (W >= 0).all() and np.allclose(W.sum(axis=1), 1.0, atol=1e-10)
    assert len(res.W_nd) >= 1
    assert res.F_nd.shape[1] == 3
    # the returned set must actually be non-dominated among itself
    F = res.F_nd
    for i in range(len(F)):
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        assert not dominated.any(), "returned set contains a dominated point"


def test_run_is_deterministic_under_seed():
    P, y, costs = _synthetic()
    a = run_nsga2(P, y, costs, dataset="synthetic", rep=3, target_evals=66 * 8)
    b = run_nsga2(P, y, costs, dataset="synthetic", rep=3, target_evals=66 * 8)
    assert a.seed == b.seed
    assert np.allclose(np.sort(a.F_nd, axis=0), np.sort(b.F_nd, axis=0))


def test_objectives_match_the_nbi_c_definition():
    """f1 must be -ROC-AUC, f2 log-loss, f3 weighted cost, on the real cached objectives."""
    from mixens.fastmetrics import evaluate_weights

    P, y, costs = _synthetic()
    res = run_nsga2(P, y, costs, dataset="synthetic", rep=0, target_evals=66 * 6)
    direct = evaluate_weights(P, y, res.W_nd, costs=costs, support_eps=1e-3,
                              chunk=len(res.W_nd), with_pr_auc=False, n_jobs=1)
    assert np.allclose(res.F_nd[:, 0], -direct["roc_auc"], atol=1e-10)
    assert np.allclose(res.F_nd[:, 1], direct["log_loss"], atol=1e-10)
    assert np.allclose(res.F_nd[:, 2], direct["cost_weighted"], atol=1e-10)
