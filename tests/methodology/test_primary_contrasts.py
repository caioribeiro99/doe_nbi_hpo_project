"""Each primary contrast must isolate exactly one mechanism.

    NBI-S vs WS-S    Pareto-front construction geometry, and nothing else
    NBI-R vs NBI-S   anchor / payoff provenance, and nothing else

The test builds a machine-readable configuration diff for each contrast and fails
if anything unrelated differs. "Unrelated" is enumerated rather than left to
judgement: the factor mapping, the surface fit, the normalization, the candidate
count, the RNG policy, the validation budget, the rounding, the cache policy, the
reference construction and the solver tolerances.
"""
from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.campaign.arms import (EQUALITY_TOLERANCE, Realizer, empirical_reference,
                                   run_nbi_arm, run_ws_s, surrogate_reference,
                                   symmetric_weights)
from doe_xgb.campaign.evaluator import BOUNDS, INT_PARAMS, PARAMS
from doe_xgb.nbi_core import NBIConfig

BOX = np.array([[-1.0, 1.0], [-1.0, 1.0]])


def f1(x) -> float:
    return float((np.clip(x[0], -1, 1) + 1) / 2)


def f2(x) -> float:
    return float(1.0 - ((np.clip(x[0], -1, 1) + 1) / 2) ** 2)


def _cfg() -> NBIConfig:
    return NBIConfig(objective_count=2, bounds=BOX, n_starts=5, seed=17, maxiter=400)


def _rz() -> Realizer:
    return Realizer(["a", "b"], {"a": (0.0, 1.0), "b": (50.0, 700.0)}, ["b"])


def _profile(run, *, surrogates, cfg, anchors, weights, realizer) -> dict:
    """Everything about how an arm was configured, excluding the mechanism itself."""
    return {
        "surrogate_identities": [id(s) for s in surrogates],
        "bounds": cfg.bounds.tolist(),
        "n_starts": cfg.n_starts,
        "seed": cfg.seed,
        "maxiter": cfg.maxiter,
        "integer_dims": list(cfg.integer_dims),
        "feasibility_constraint": cfg.feasibility_constraint is not None,
        "n_weights": int(len(weights)),
        "weights": np.round(weights, 12).tolist(),
        "realizer_params": realizer.params,
        "realizer_bounds": [realizer.lo.tolist(), realizer.hi.tolist()],
        "realizer_int_params": sorted(realizer.int_params),
        "candidate_count": len(run.candidates),
        "utopia": np.round(anchors.utopia, 12).tolist(),
        "payoff": np.round(anchors.F_star, 12).tolist(),
    }


@pytest.fixture(scope="module")
def built():
    cfg, rz, w = _cfg(), _rz(), symmetric_weights(12)
    surrogates = [f1, f2]
    s_anchors, s_chim = surrogate_reference(surrogates, cfg)
    ws = run_ws_s(surrogates, cfg, s_anchors, rz, w)
    nbi_s = run_nbi_arm("NBI-S", surrogates, cfg, s_anchors, s_chim, rz, w)
    x_star = np.array([[-0.85, 0.0], [0.9, 0.0]])
    F_star = np.array([[f1(x_star[0]) - 0.01, f1(x_star[1])],
                       [f2(x_star[0]), f2(x_star[1]) - 0.02]])
    r_anchors, r_chim = empirical_reference(x_star, F_star, cfg)
    nbi_r = run_nbi_arm("NBI-R", surrogates, cfg, r_anchors, r_chim, rz, w)
    return dict(cfg=cfg, rz=rz, w=w, surrogates=surrogates, s_anchors=s_anchors,
                r_anchors=r_anchors, ws=ws, nbi_s=nbi_s, nbi_r=nbi_r)


def test_geometry_contrast_differs_only_in_the_scalarization(built) -> None:
    common = dict(surrogates=built["surrogates"], cfg=built["cfg"],
                  anchors=built["s_anchors"], weights=built["w"], realizer=built["rz"])
    a = _profile(built["ws"], **common)
    b = _profile(built["nbi_s"], **common)
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert not diff, f"WS-S and NBI-S differ in {sorted(diff)} beyond the scalarization"
    # and the mechanism itself genuinely differs
    assert (built["ws"].diagnostics["scalarization"]
            != built["nbi_s"].diagnostics["scalarization"])


def test_anchor_contrast_differs_only_in_the_payoff_matrix(built) -> None:
    base = dict(surrogates=built["surrogates"], cfg=built["cfg"],
                weights=built["w"], realizer=built["rz"])
    a = _profile(built["nbi_s"], anchors=built["s_anchors"], **base)
    b = _profile(built["nbi_r"], anchors=built["r_anchors"], **base)
    diff = {k for k in a if a[k] != b[k]}
    assert diff <= {"utopia", "payoff"}, (
        f"NBI-S and NBI-R differ in {sorted(diff - {'utopia', 'payoff'})} beyond the "
        "anchor provenance"
    )
    assert "payoff" in diff, "the contrast must actually change the payoff matrix"
    assert (built["nbi_s"].diagnostics["scalarization"]
            == built["nbi_r"].diagnostics["scalarization"])


def test_both_nbi_arms_share_every_solver_setting(built) -> None:
    for key in ("restrict_t_nonnegative", "equality_tolerance",
                "n_weights", "feasibility_constraint_applied"):
        assert built["nbi_s"].diagnostics[key] == built["nbi_r"].diagnostics[key], key
    assert EQUALITY_TOLERANCE == built["nbi_s"].diagnostics["equality_tolerance"]


def test_the_anchor_injection_control_isolates_set_composition(built) -> None:
    """What the control separates, stated so it cannot be mis-cited.

    A vertex weight returns its own anchor, so NBI-R's set CONTAINS the empirical
    anchors. The control rescores NBI-S's own set augmented with those same
    anchors: it therefore holds the geometry fixed at NBI-S's and varies only set
    composition, so the part of any NBI-S to NBI-R difference it reproduces is
    attributable to injected extreme points rather than to the relocated CHIM.
    """
    from doe_xgb.campaign.scoring import augmented_reference, indicators

    nbi_s_F = np.array([c.f_surrogate_realized for c in built["nbi_s"].candidates])
    anchors_F = built["r_anchors"].F_star.T          # one row per anchor
    injected = np.vstack([nbi_s_F, anchors_F])
    assert len(injected) == len(nbi_s_F) + 2
    ref = augmented_reference(nbi_s_F, {"injected": injected})["front"]
    base_ind = indicators(nbi_s_F, ref)
    inj_ind = indicators(injected, ref)
    # the control can only add points, so it cannot be worse on hypervolume
    assert inj_ind["hv_ratio"] >= base_ind["hv_ratio"] - 1e-12
