"""Tests for the benchmark additions: fast metrics, Pareto tools, the 66-run
design, NBI anchor override / warm start, Scheffé order comparison."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from mixens import fastmetrics as fm
from mixens import pareto_tools as pt
from mixens.mixture_design import build_benchmark_design, validate_weights
from mixens.nbi import run_nbi_on_simplex
from mixens.scheffe import compare_orders
from mixens.selection import pareto_filter


@pytest.fixture(scope="module")
def toy_probs():
    rng = np.random.default_rng(0)
    n, M = 5000, 5
    y = (rng.random(n) < 0.2).astype(int)
    base = rng.random((n, M))
    P = np.clip(0.5 * base + 0.5 * y[:, None] * rng.random((n, M)), 1e-3, 1 - 1e-3).astype(np.float32)
    return P, y


def test_fast_metrics_match_sklearn(toy_probs):
    P, y = toy_probs
    w = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    p = P.astype(np.float64) @ w
    assert fm.rank_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert fm.fast_auc_no_ties(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-9)
    assert fm.log_loss_vec(y, p) == pytest.approx(log_loss(y, p), abs=1e-12)
    assert fm.brier_vec(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    assert fm.average_precision(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-9)


def test_evaluate_weights_batch_matches_single(toy_probs):
    P, y = toy_probs
    W = np.random.default_rng(1).dirichlet(np.ones(5), size=7)
    costs = np.array([1.0, 2.0, 50.0, 5.0, 3.0])
    out = fm.evaluate_weights(P, y, W, costs=costs, support_eps=1e-3, chunk=3, n_jobs=2)
    for i, w in enumerate(W):
        p = P.astype(np.float64) @ w
        assert out["roc_auc"][i] == pytest.approx(roc_auc_score(y, p), abs=1e-6)
        assert out["log_loss"][i] == pytest.approx(log_loss(y, p), abs=1e-6)
        assert out["cost_weighted"][i] == pytest.approx(w @ costs)
        assert out["cost_support"][i] == pytest.approx(costs[w > 1e-3].sum())
        assert out["n_eff"][i] == pytest.approx(1 / np.sum(w ** 2))


def test_fast_pareto_mask_matches_reference_filter():
    rng = np.random.default_rng(2)
    F = rng.random((400, 3))
    np.testing.assert_array_equal(pt.fast_pareto_mask(F), pareto_filter(F))
    F2 = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [3.0, 3.0]])
    np.testing.assert_array_equal(pt.fast_pareto_mask(F2), [True, True, True, True, False])


def test_hypervolume_known_values():
    assert pt.hypervolume(np.array([[0.0, 0.0]]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert pt.hypervolume(np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([1.0, 1.0])) == pytest.approx(0.75)
    assert pt.hypervolume(np.array([[0.0, 0.0, 0.0]]), np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)
    # two 3-D points: union of two boxes
    F = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.5]])
    assert pt.hypervolume(F, np.ones(3)) == pytest.approx(0.5 + 0.25 - 0.5 * 0.5 * 0.5)


def test_igd_plus_and_coverage():
    ref = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert pt.igd_plus(ref, ref) == pytest.approx(0.0)
    worse = ref + 0.1
    assert pt.igd_plus(worse, ref) == pytest.approx(np.sqrt(2) * 0.1)
    better = ref - 0.1
    assert pt.igd_plus(better, ref) == pytest.approx(0.0)  # dominating points incur no IGD+ penalty
    assert pt.coverage(better, ref) == pytest.approx(1.0)
    assert pt.coverage(ref, better) == pytest.approx(0.0)
    assert pt.joint_nondominated_fraction(worse, ref) == pytest.approx(0.0)


def test_benchmark_design_has_66_valid_runs():
    W = build_benchmark_design(5)
    validate_weights(W)
    assert W.shape == (66, 5)
    assert len(np.unique(np.round(W, 10), axis=0)) == 66
    assert (np.count_nonzero(W, axis=1) == 5).sum() >= 16  # centroid + 5 axial + 10 interior midpoints


def test_nbi_supplied_anchors_and_vertex_warm_start():
    e1 = np.array([1.0, 0, 0]); e2 = np.array([0, 1.0, 0])
    f1 = lambda w: float(np.sum((w - e1) ** 2)); f2 = lambda w: float(np.sum((w - e2) ** 2))
    res = run_nbi_on_simplex([f1, f2], 3, n_points=7, n_starts=4, seed=0, anchors_w=[e1, e2])
    assert res["anchors_source"] == "supplied"
    np.testing.assert_allclose(res["anchors_w"], [e1, e2], atol=1e-12)
    cands = res["candidates"]
    assert cands[0]["message"].startswith("vertex beta") and cands[-1]["message"].startswith("vertex beta")
    assert all(c["success"] for c in cands)
    ws = np.array([c["w"] for c in cands]); validate_weights(ws)
    assert np.all(ws[:, 2] < 0.05)


def test_nbi_fd_eps_on_piecewise_constant_objective():
    """A step-like objective defeats the default 1e-8 finite differences; a larger
    FD step lets SLSQP make progress (this is the metamodel-free NBI setting)."""
    e1 = np.array([1.0, 0, 0]); e2 = np.array([0, 1.0, 0])
    f1 = lambda w: float(np.round(np.sum((w - e1) ** 2), 2))  # plateaus of width ~0.01
    f2 = lambda w: float(np.sum((w - e2) ** 2))
    res = run_nbi_on_simplex([f1, f2], 3, n_points=5, n_starts=2, seed=0, anchors_w=[e1, e2], fd_eps=1e-2)
    assert res["fd_eps"] == 1e-2
    assert sum(c["success"] for c in res["candidates"]) >= 3


def test_compare_orders_selects_parsimonious_model():
    rng = np.random.default_rng(3)
    W = build_benchmark_design(5)
    comp = [f"w{i}" for i in range(5)]
    beta = np.array([1.0, 2.0, 0.5, 1.5, 3.0])
    truth = lambda X: X @ beta + 0.8 * X[:, 0] * X[:, 1]  # quadratic truth
    y_d = truth(W)
    W_val = rng.dirichlet(np.ones(5), size=40); y_v = truth(W_val)
    out = compare_orders(W, y_d, W_val, y_v, component_names=comp)
    assert out["orders"]["quadratic"]["estimable"] and out["orders"]["special_cubic"]["estimable"]
    assert out["selected_order"] == "quadratic"
    assert out["orders"]["quadratic"]["external"]["rmse"] < 1e-8
