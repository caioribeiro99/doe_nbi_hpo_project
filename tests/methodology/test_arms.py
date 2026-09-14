"""The four arms must differ in exactly the ways the contrasts name, and no others.

The contrasts are only interpretable if each changes one thing. These tests pin
the "one thing" for each, and pin the historical arm's two deliberate departures
so that they cannot be silently repaired.
"""
from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.campaign.arms import (N_WEIGHTS, Realizer, empirical_reference,
                                   historical_weights, run_nbi_arm, run_ws_s,
                                   surrogate_reference, symmetric_weights)
from doe_xgb.nbi_core import NBIConfig

PARAMS = ["a", "b"]
BOUNDS = {"a": (0.0, 1.0), "b": (50.0, 700.0)}
INTS = ["b"]


def realizer() -> Realizer:
    return Realizer(PARAMS, BOUNDS, INTS)


# A concave Pareto front in coded coordinates: minimizing f1 and f2 over x0 in
# [-1, 1]. Weighted sum can only reach the endpoints; NBI must reach the interior.
def _u(x):          # coded -> [0, 1]
    return (np.clip(x[0], -1.0, 1.0) + 1.0) / 2.0


def f1(x) -> float:
    return float(_u(x))


def f2(x) -> float:
    return float(1.0 - _u(x) ** 2)


def cfg(**kw) -> NBIConfig:
    base = dict(objective_count=2, bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                n_starts=5, seed=3)
    base.update(kw)
    return NBIConfig(**base)


# ------------------------------------------------------------- the weight grids

def test_historical_grid_omits_the_pure_quality_vertex() -> None:
    """A property of the historical method, recorded rather than repaired."""
    g = historical_weights()
    assert len(g) == N_WEIGHTS
    assert not any(np.allclose(w, [1.0, 0.0]) for w in g)
    assert any(np.allclose(w, [0.0, 1.0]) for w in g)
    assert g[:, 0].max() == pytest.approx(0.95)


def test_shared_grid_is_symmetric_and_matches_the_historical_cardinality() -> None:
    g = symmetric_weights()
    assert len(g) == N_WEIGHTS == len(historical_weights())
    assert any(np.allclose(w, [1.0, 0.0]) for w in g)
    assert any(np.allclose(w, [0.0, 1.0]) for w in g)
    np.testing.assert_allclose(g.sum(axis=1), 1.0)


# ------------------------------------------------------------ the geometry pair

def test_ws_s_and_nbi_s_share_the_same_reference_object() -> None:
    """The geometry contrast is only clean if the reference is identical."""
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    ws = run_ws_s([f1, f2], c, anchors, realizer())
    nbi = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer())
    assert ws.diagnostics["utopia"] == nbi.diagnostics["utopia"]
    assert ws.diagnostics["nadir_used"] == anchors.pseudo_nadir.tolist(), (
        "the protocol fixes the pseudo-nadir as WS-S's second reference end"
    )
    assert ws.diagnostics["n_weights"] == nbi.diagnostics["n_weights"]


def test_the_geometry_contrast_is_detectable_on_a_concave_front() -> None:
    """If WS-S and NBI-S agreed here, the primary contrast could measure nothing."""
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    ws = run_ws_s([f1, f2], c, anchors, realizer())
    nbi = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer())

    def interior(run):
        u = np.array([_u(k.x_continuous) for k in run.candidates])
        return int(((u > 1e-3) & (u < 1 - 1e-3)).sum())

    assert interior(ws) <= 2, (
        f"weighted sum returned {interior(ws)} interior points on a concave front"
    )
    assert interior(nbi) >= 15, f"NBI returned only {interior(nbi)} interior points"


def test_nbi_records_negative_t_and_certifies_on_a_concave_front() -> None:
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    run = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer())
    assert run.diagnostics["restrict_t_nonnegative"] is False
    assert run.diagnostics["t_range"][0] < 0, "a concave front needs t < 0"
    assert run.diagnostics["certified_fraction"] >= 0.9
    assert run.diagnostics["max_equality_residual"] < 1e-5


# ------------------------------------------------------------ the anchor pair

def test_nbi_r_uses_the_supplied_anchors_not_the_surrogate_ones() -> None:
    """The anchor-provenance contrast changes the payoff matrix and nothing else."""
    c = cfg()
    s_anchors, s_chim = surrogate_reference([f1, f2], c)
    # anchors displaced from the surrogate optima, as a real search would be
    x_star = np.array([[-0.8, 0.0], [0.85, 0.0]])
    F_star = np.array([[f1(x_star[0]) - 0.02, f1(x_star[1])],
                       [f2(x_star[0]), f2(x_star[1]) - 0.03]])
    r_anchors, r_chim = empirical_reference(x_star, F_star, c)

    assert not np.allclose(s_anchors.utopia, r_anchors.utopia)
    assert "empirical anchor search" in r_anchors.diagnostics["source"]

    s_run = run_nbi_arm("NBI-S", [f1, f2], c, s_anchors, s_chim, realizer())
    r_run = run_nbi_arm("NBI-R", [f1, f2], c, r_anchors, r_chim, realizer())
    assert s_run.diagnostics["payoff_matrix"] != r_run.diagnostics["payoff_matrix"]
    assert s_run.diagnostics["scalarization"] == r_run.diagnostics["scalarization"]
    assert s_run.diagnostics["n_weights"] == r_run.diagnostics["n_weights"]


def test_empirical_anchors_are_never_called_optima() -> None:
    c = cfg()
    a, _ = empirical_reference(np.zeros((2, 2)), np.eye(2), c)
    text = a.diagnostics["source"].lower()
    assert "not certified optima" in text
    assert "true optim" not in text


# ------------------------------------------------------------- the realization

def test_realization_is_deterministic() -> None:
    r = realizer()
    x = np.array([0.137, -0.42])
    first = r.realize(x)
    for _ in range(5):
        cfg_i, coded_i = r.realize(x)
        assert cfg_i == first[0]
        np.testing.assert_array_equal(coded_i, first[1])


def test_realization_rounds_integers_and_keeps_them_in_bounds() -> None:
    r = realizer()
    for x0 in np.linspace(-1, 1, 41):
        cfg_i, _ = r.realize(np.array([0.0, x0]))
        assert isinstance(cfg_i["b"], int)
        assert BOUNDS["b"][0] <= cfg_i["b"] <= BOUNDS["b"][1]
        assert isinstance(cfg_i["a"], float)


def test_realization_displacement_is_recorded_and_can_be_nonzero() -> None:
    """NBI certifies the continuous point; the learner runs the rounded one."""
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    run = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer())
    for k in run.candidates:
        assert k.realization_displacement == pytest.approx(
            float(np.linalg.norm(k.f_surrogate_realized - k.f_surrogate_continuous)))
        assert k.x_continuous.shape == k.x_realized_coded.shape
    assert "max_realization_displacement" in run.diagnostics


def test_candidate_record_keeps_continuous_and_realized_apart() -> None:
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    d = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer()).candidates[5].as_dict()
    for key in ("x_continuous", "f_surrogate_continuous", "config_realized",
                "x_realized_coded", "f_surrogate_realized",
                "realization_displacement", "t", "equality_residual"):
        assert key in d, key


# ----------------------------------------------------------------- degeneracy

def test_coincident_anchors_are_reported_not_repaired() -> None:
    c = cfg()
    x_star = np.array([[0.3, 0.0], [0.3, 0.0]])          # the same point twice
    F_star = np.array([[0.5, 0.5], [0.5, 0.5]])
    anchors, chim = empirical_reference(x_star, F_star, c)
    run = run_nbi_arm("NBI-R", [f1, f2], c, anchors, chim, realizer())
    assert run.diagnostics["anchors_coincide"] is True
    assert run.diagnostics["payoff_rank"] < 2
    assert run.diagnostics["payoff_condition_number"] is None


def test_every_arm_reports_the_dominated_share_of_its_returned_set() -> None:
    """Certification is feasibility, not Pareto optimality, and the arms differ.

    Weighted-sum minimizers are weakly Pareto optimal by construction; NBI
    subproblem solutions need not be. If only one arm reported this, the asymmetry
    would look like a difference in approximation quality.
    """
    c = cfg()
    anchors, chim = surrogate_reference([f1, f2], c)
    ws = run_ws_s([f1, f2], c, anchors, realizer())
    nbi = run_nbi_arm("NBI-S", [f1, f2], c, anchors, chim, realizer())
    for run in (ws, nbi):
        assert "dominated_share_of_returned_set" in run.diagnostics, run.arm
        share = run.diagnostics["dominated_share_of_returned_set"]
        assert 0.0 <= share <= 1.0
    assert "dominated_share_among_certified" in nbi.diagnostics


def test_igd_plus_is_weakly_pareto_compliant_and_igd_is_not_the_protocol_indicator() -> None:
    from doe_xgb.reporting import igd_plus

    R = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    A = np.array([[0.1, 1.0], [0.6, 0.6], [1.0, 0.1]])
    B = A + 0.2                                   # every point dominated by A's
    assert igd_plus(A, R) < igd_plus(B, R)
    # a set containing the reference scores zero
    assert igd_plus(R, R) == pytest.approx(0.0, abs=1e-12)


def test_historical_arm_loads_the_frozen_tree_not_the_article_rewrite() -> None:
    """The one arm whose job is bit-faithful reproduction must call the tag's code.

    The article track imports `doe_xgb`, so putting the frozen tree on sys.path
    cannot change what `doe_xgb.nbi` resolves to. The frozen tree is therefore
    extracted under its own package name and both are importable side by side.
    """
    import importlib
    import sys

    from doe_xgb.campaign.arms import FROZEN_PACKAGE, _ensure_frozen_tree

    src = _ensure_frozen_tree()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    frozen = importlib.import_module(f"{FROZEN_PACKAGE}.nbi")
    assert str(src) in str(frozen.__file__)
    assert hasattr(frozen, "run_nbi_weighted_sum")
    # the weighted-sum solver is the only optimizer the dissertation shipped
    assert not hasattr(frozen, "run_nbi")
    # and the article-track NBI is a different module, still importable
    import doe_xgb.nbi_core as article
    assert "nbi_core" in article.__file__
    assert hasattr(article, "solve_nbi_subproblem")


def test_the_frozen_tree_is_repository_local_not_a_scratch_path() -> None:
    from pathlib import Path

    from doe_xgb.campaign.arms import _ensure_frozen_tree

    src = _ensure_frozen_tree()
    repo = Path(__file__).resolve().parents[2]
    assert repo in src.parents, f"{src} is outside the repository"
    assert "/tmp" not in str(src)
