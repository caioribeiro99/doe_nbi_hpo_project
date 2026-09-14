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
    """Each arm is configured from its OWN objects, constructed independently.

    An earlier version of this fixture built one `common` dict and passed it to
    both _profile calls, so 13 of the 14 profile fields were the same objects read
    twice and only candidate_count came off an arm. The test could not fail for any
    reason except a candidate-count difference: it asserted that a dict equals
    itself. Building each arm's configuration separately makes the comparison an
    assertion about VALUES that two independent constructions agree on.

    The surrogates are deliberately the same objects, because "the same fitted
    surfaces" is part of what the contrast holds fixed, and identity is the
    strongest way to state it.
    """
    surrogates = [f1, f2]                      # shared ON PURPOSE; asserted by identity

    cfg_ws, rz_ws, w_ws = _cfg(), _rz(), symmetric_weights(12)
    cfg_nbi_s, rz_nbi_s, w_nbi_s = _cfg(), _rz(), symmetric_weights(12)
    cfg_nbi_r, rz_nbi_r, w_nbi_r = _cfg(), _rz(), symmetric_weights(12)

    s_anchors_ws, _ = surrogate_reference(surrogates, cfg_ws)
    s_anchors, s_chim = surrogate_reference(surrogates, cfg_nbi_s)

    ws = run_ws_s(surrogates, cfg_ws, s_anchors_ws, rz_ws, w_ws)
    nbi_s = run_nbi_arm("NBI-S", surrogates, cfg_nbi_s, s_anchors, s_chim, rz_nbi_s, w_nbi_s)

    x_star = np.array([[-0.85, 0.0], [0.9, 0.0]])
    F_star = np.array([[f1(x_star[0]) - 0.01, f1(x_star[1])],
                       [f2(x_star[0]), f2(x_star[1]) - 0.02]])
    r_anchors, r_chim = empirical_reference(x_star, F_star, cfg_nbi_r)
    nbi_r = run_nbi_arm("NBI-R", surrogates, cfg_nbi_r, r_anchors, r_chim, rz_nbi_r, w_nbi_r)

    return dict(
        surrogates=surrogates,
        ws=dict(run=ws, cfg=cfg_ws, rz=rz_ws, w=w_ws, anchors=s_anchors_ws),
        nbi_s=dict(run=nbi_s, cfg=cfg_nbi_s, rz=rz_nbi_s, w=w_nbi_s, anchors=s_anchors),
        nbi_r=dict(run=nbi_r, cfg=cfg_nbi_r, rz=rz_nbi_r, w=w_nbi_r, anchors=r_anchors),
        s_chim=s_chim, r_chim=r_chim, x_star=x_star, F_star=F_star)


def _profile_arm(entry, surrogates) -> dict:
    """Profile an arm from the objects THAT arm was built with."""
    return _profile(entry["run"], surrogates=surrogates, cfg=entry["cfg"],
                    anchors=entry["anchors"], weights=entry["w"],
                    realizer=entry["rz"])


def test_geometry_contrast_differs_only_in_the_scalarization(built) -> None:
    a = _profile_arm(built["ws"], built["surrogates"])
    b = _profile_arm(built["nbi_s"], built["surrogates"])
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    assert not diff, f"WS-S and NBI-S differ in {sorted(diff)} beyond the scalarization"
    # and the mechanism itself genuinely differs
    assert (built["ws"]["run"].diagnostics["scalarization"]
            != built["nbi_s"]["run"].diagnostics["scalarization"])


@pytest.mark.parametrize("field,value", [("n_starts", 8), ("maxiter", 500), ("seed", 18)])
def test_the_geometry_contrast_check_can_fail(built, field, value) -> None:
    """A mutation the contrast must catch, so the check above is not vacuous.

    NBIConfig is frozen, so the perturbed configuration is constructed with
    dataclasses.replace rather than mutated in place.
    """
    import dataclasses

    entry = dict(built["nbi_s"])
    entry["cfg"] = dataclasses.replace(entry["cfg"], **{field: value})
    a = _profile_arm(built["ws"], built["surrogates"])
    b = _profile_arm(entry, built["surrogates"])
    assert a != b, f"perturbing {field} did not change the profile"
    assert a[field] != b[field]


def test_the_contrast_catches_a_different_weight_grid(built) -> None:
    import copy
    entry = copy.deepcopy(built["nbi_s"])
    entry["w"] = symmetric_weights(11)
    a = _profile_arm(built["ws"], built["surrogates"])
    b = _profile_arm(entry, built["surrogates"])
    assert a["n_weights"] != b["n_weights"]
    assert a != b


def test_the_contrast_catches_a_different_realizer(built) -> None:
    import copy
    entry = copy.deepcopy(built["nbi_s"])
    entry["rz"] = Realizer(["a", "b"], {"a": (0.0, 1.0), "b": (50.0, 900.0)}, ["b"])
    a = _profile_arm(built["ws"], built["surrogates"])
    b = _profile_arm(entry, built["surrogates"])
    assert a["realizer_bounds"] != b["realizer_bounds"]
    assert a != b


def test_the_two_arms_really_share_the_surrogate_objects(built) -> None:
    """Held fixed by identity, not merely by equal values."""
    a = _profile_arm(built["ws"], built["surrogates"])
    b = _profile_arm(built["nbi_s"], built["surrogates"])
    assert a["surrogate_identities"] == b["surrogate_identities"]


def test_anchor_contrast_differs_only_in_the_payoff_matrix(built) -> None:
    # Each arm profiled from its OWN configuration objects, so equality here is an
    # assertion about values two independent constructions agree on rather than
    # about one dict compared with itself.
    a = _profile_arm(built["nbi_s"], built["surrogates"])
    b = _profile_arm(built["nbi_r"], built["surrogates"])
    diff = {k for k in a if a[k] != b[k]}
    assert diff <= {"utopia", "payoff"}, (
        f"NBI-S and NBI-R differ in {sorted(diff - {'utopia', 'payoff'})} beyond the "
        "anchor provenance"
    )
    assert "payoff" in diff, "the contrast must actually change the payoff matrix"
    assert (built["nbi_s"]["run"].diagnostics["scalarization"]
            == built["nbi_r"]["run"].diagnostics["scalarization"])


def test_both_nbi_arms_share_every_solver_setting(built) -> None:
    for key in ("restrict_t_nonnegative", "equality_tolerance",
                "n_weights", "feasibility_constraint_applied"):
        assert built["nbi_s"]["run"].diagnostics[key] == built["nbi_r"]["run"].diagnostics[key], key
    assert EQUALITY_TOLERANCE == built["nbi_s"]["run"].diagnostics["equality_tolerance"]


def test_the_anchor_injection_control_isolates_set_composition(built) -> None:
    """What the control separates, stated so it cannot be mis-cited.

    A vertex weight returns its own anchor, so NBI-R's set CONTAINS the empirical
    anchors. The control rescores NBI-S's own set augmented with those same
    anchors: it therefore holds the geometry fixed at NBI-S's and varies only set
    composition, so the part of any NBI-S to NBI-R difference it reproduces is
    attributable to injected extreme points rather than to the relocated CHIM.
    """
    from doe_xgb.campaign.scoring import augmented_reference, indicators

    nbi_s_F = np.array([c.f_surrogate_realized for c in built["nbi_s"]["run"].candidates])
    anchors_F = built["nbi_r"]["anchors"].F_star.T          # one row per anchor
    injected = np.vstack([nbi_s_F, anchors_F])
    assert len(injected) == len(nbi_s_F) + 2
    ref = augmented_reference(nbi_s_F, {"injected": injected})["front"]
    base_ind = indicators(nbi_s_F, ref)
    inj_ind = indicators(injected, ref)
    # the control can only add points, so it cannot be worse on hypervolume
    assert inj_ind["hv_ratio"] >= base_ind["hv_ratio"] - 1e-12
