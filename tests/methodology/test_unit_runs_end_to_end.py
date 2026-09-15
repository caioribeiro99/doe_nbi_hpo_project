"""One full unit, all twenty stages, against a stub evaluator.

Four defects reached the pre-freeze engineering smoke and every one of them died
inside run_unit, between stage six and stage eighteen, after real XGBoost
evaluations had already been spent:

    stage  6  _run_historical had three mutually inconsistent signatures
    stage 13  a 64-bit BLAKE2b seed handed to GaussianProcessRegressor
    stage 17  the seed ledger stored inside the methods namespace
    (silent)  the core reference built from 88 design rows instead of 288

312 tests passed throughout. Nothing ran the pipeline, so nothing could catch a
defect that lives between components rather than inside one.

This runs the whole unit with XGBoost replaced by a cheap deterministic surface, so
it takes seconds and needs no data. It asserts EXECUTION and ACCOUNTING only:
stages complete, artifacts exist and are finite, the logical ledger equals the
declared registry, and resume is a no-op. It deliberately asserts nothing about
which arm performed better -- that is the campaign's job, and inspecting it here
would breach the confirmatory blindness the protocol is frozen under.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from doe_xgb.campaign import evaluator as EV
from doe_xgb.campaign import runner as R


# ---------------------------------------------------------------------------
# a deterministic stand-in for the real learner
# ---------------------------------------------------------------------------

def _stub_evaluate(config, *, on_holdout: bool = False) -> dict[str, float]:
    """Smooth, bounded, conflicting responses. Deterministic in the config.

    Quality improves with depth and estimators; leaf count grows with both, so the
    two objectives genuinely trade off and the arms have a front to find.
    """
    x = np.array([float(config[p]) for p in EV.PARAMS], dtype=float)
    lo = np.array([EV.BOUNDS[p][0] for p in EV.PARAMS], dtype=float)
    hi = np.array([EV.BOUNDS[p][1] for p in EV.PARAMS], dtype=float)
    u = (x - lo) / np.where(hi - lo == 0, 1.0, hi - lo)          # in [0, 1]
    shift = 0.03 if on_holdout else 0.0                          # a different partition

    base = 0.70 + 0.22 * (0.6 * u[4] + 0.4 * u[6]) - 0.05 * u[5] - shift
    acc = float(np.clip(base + 0.01 * math.sin(6.0 * u[0]), 0.05, 0.99))
    pre = float(np.clip(base - 0.02 + 0.02 * u[1], 0.05, 0.99))
    rec = float(np.clip(base + 0.01 - 0.03 * u[2], 0.05, 0.99))
    spe = float(np.clip(0.95 - 0.25 * (0.6 * u[4] + 0.4 * u[6]), 0.05, 0.99))
    auc = float(np.clip(base + 0.05, 0.05, 0.999))
    ll = float(np.clip(1.2 - 1.0 * base + 0.05 * u[3], 0.01, 5.0))
    leaves = float(4.0 + 600.0 * u[4] * (0.3 + 0.7 * u[6]) + 20.0 * u[0])
    return {"Accuracy_Mean": acc, "Precision_Mean": pre, "Recall_Mean": rec,
            "Specificity_Mean": spe, "RocAuc_Mean": auc, "LogLoss_Mean": ll,
            "Leaves_Mean": leaves, "Time_MeanFold": 0.01 + 0.002 * leaves}


@pytest.fixture(scope="module")
def unit(tmp_path_factory):
    root = tmp_path_factory.mktemp("stub_unit")
    import doe_xgb.campaign.runner as runner_mod
    real_eval, real_init = runner_mod.evaluate_config, runner_mod.init_worker
    runner_mod.evaluate_config = _stub_evaluate
    runner_mod.init_worker = lambda *a, **k: None
    try:
        result = R.run_unit("spambase", 0, root, threads=1)
    finally:
        runner_mod.evaluate_config, runner_mod.init_worker = real_eval, real_init
    return {"result": result, "dir": root / "spambase" / "rep_00", "root": root}


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def test_the_unit_completes(unit):
    assert unit["result"]["dataset"] == "spambase"
    assert unit["result"]["replication"] == 0


@pytest.mark.parametrize("stage", R.STAGES)
def test_every_declared_stage_completed(unit, stage):
    assert stage in unit["result"]["stages_complete"], f"{stage} did not complete"


def test_no_methodological_failure_was_recorded(unit):
    assert not (unit["dir"] / "methodological_failure.json").exists()


@pytest.mark.parametrize("stage", R.STAGES)
def test_every_stage_artifact_is_present_and_complete(unit, stage):
    p = unit["dir"] / f"{stage}.json"
    assert p.exists(), f"{stage}.json was never written"
    assert json.loads(p.read_text())["_complete"] is True


def _finite(obj, path="") -> list[str]:
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            bad += _finite(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2000]):
            bad += _finite(v, f"{path}[{i}]")
    elif isinstance(obj, float) and not math.isfinite(obj):
        bad.append(path)
    return bad


@pytest.mark.parametrize("stage", R.STAGES)
def test_no_stage_artifact_contains_a_non_finite_number(unit, stage):
    bad = _finite(json.loads((unit["dir"] / f"{stage}.json").read_text()))
    assert not bad, f"{stage}.json holds non-finite values at {bad[:8]}"


# ---------------------------------------------------------------------------
# the registry of executed entities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("arm,stage", [
    ("HISTORICAL-WS-asrun", "historical_ws_asrun"),
    ("HISTORICAL-WS", "historical_ws"),
    ("WS-S", "ws_s"),
    ("NBI-S", "nbi_s"),
    ("NBI-R", "nbi_r"),
])
def test_each_arm_ran_under_its_own_identifier(unit, arm, stage):
    payload = json.loads((unit["dir"] / f"{stage}.json").read_text())
    assert payload["arm"] == arm, (
        f"stage {stage} recorded arm {payload['arm']!r}, not {arm!r}")
    assert len(payload["candidates"]) > 0


def test_the_two_historical_arms_are_distinguishable(unit):
    a = json.loads((unit["dir"] / "historical_ws_asrun.json").read_text())["arm"]
    b = json.loads((unit["dir"] / "historical_ws.json").read_text())["arm"]
    assert a != b, "the reproduction and the normalization control share an identifier"


def test_every_comparator_ran(unit):
    methods = json.loads((unit["dir"] / "direct_baselines.json").read_text())["methods"]
    assert set(methods) == set(R.SCORED_BASELINES)


def test_the_baseline_namespace_holds_only_methods(unit):
    payload = json.loads((unit["dir"] / "direct_baselines.json").read_text())
    R._baseline_methods(payload)                    # raises if polluted


# ---------------------------------------------------------------------------
# accounting -- the check the tautological one could never make
# ---------------------------------------------------------------------------

def test_the_measured_ledger_reconciles_against_the_registry(unit):
    rec = R.reconcile_unit_accounting(unit["result"]["accounting"], replication=0)
    assert rec["reconciles"], (
        f"charged {rec['total_charged']} against {rec['total_declared']} declared; "
        f"mismatched={rec['mismatched_stages']} "
        f"undeclared={rec['charged_but_never_declared']} "
        f"never_charged={rec['declared_but_never_charged']}")


def test_the_measured_total_equals_the_published_unit_budget(unit):
    """Replication 0 additionally carries the unmatched NSGA-II context run."""
    unmatched = R.NSGA2_POP * R.NSGA2_GEN * R.NSGA2_UNMATCHED_MULTIPLIER
    assert (unit["result"]["accounting"]["logical_evaluations_total"]
            == R.unit_budget(2)["total_logical"] + unmatched)


def test_no_evaluation_happened_outside_the_ledger(unit):
    """Every real evaluation the unit performed is attributed to some method."""
    import sqlite3
    db = sqlite3.connect(str(unit["dir"] / "evaluations.sqlite"))
    physical = db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
    attributed = db.execute("SELECT COUNT(DISTINCT key) FROM requests").fetchone()[0]
    db.close()
    assert physical == attributed, (
        f"{physical - attributed} evaluations exist that no method ever requested")


def test_the_holdout_stage_is_charged(unit):
    by_stage = unit["result"]["accounting"]["by_stage"]
    assert by_stage["holdout_audit"]["logical"] == 5


def test_the_anchor_injection_control_is_charged(unit):
    per = {m["method"]: m for m in unit["result"]["accounting"]["per_method"]}
    assert per["anchor_injection_control"]["logical_evaluations"] == 2


# ---------------------------------------------------------------------------
# the science-bearing invariants this suite is allowed to inspect
# ---------------------------------------------------------------------------

def test_the_core_reference_contains_the_design_and_the_anchor_search(unit):
    core = json.loads((unit["dir"] / "reference_core.json").read_text())
    assert core["n_design_rows"] == 88
    assert core["n_anchor_rows"] == 2 * R.B_ANCHOR_PER_OBJECTIVE
    assert core["n_points"] == 88 + 2 * R.B_ANCHOR_PER_OBJECTIVE


def test_the_factor_model_was_applied_not_fitted(unit):
    fmj = json.loads((unit["dir"] / "factor_model.json").read_text())
    assert fmj["fitted_at_campaign_time"] is False
    assert fmj["source"] == "frozen reference model, APPLIED"
    assert "per_replication_refit_sensitivity" in fmj


def test_the_refit_sensitivity_is_reported(unit):
    s = json.loads((unit["dir"] / "factor_model.json").read_text())[
        "per_replication_refit_sensitivity"]
    assert len(s["tucker_congruence"]) == 3
    assert all(math.isfinite(v) for v in s["tucker_congruence"])


def test_single_objective_comparators_are_absent_from_the_augmented_reference(unit):
    aug = json.loads((unit["dir"] / "augmented_reference.json").read_text())
    share = aug.get("self_grading_share_of_front", {})
    for m in R.SINGLE_OBJECTIVE:
        assert m not in share, f"{m} contributed to the shared reference"


def test_the_external_set_is_audit_only(unit):
    ext = json.loads((unit["dir"] / "external_validation.json").read_text())
    assert len(ext["rows"]) == 78
    assert "audit-only" in ext["note"]


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

def test_resuming_a_completed_unit_changes_nothing(unit):
    """Every stage is checkpointed, so a rerun must do no work and alter no total."""
    import doe_xgb.campaign.runner as runner_mod
    before = json.loads((unit["dir"] / "metrics.json").read_text())

    calls = {"n": 0}

    def counting(cfg, *, on_holdout: bool = False):
        calls["n"] += 1
        return _stub_evaluate(cfg, on_holdout=on_holdout)

    real_eval, real_init = runner_mod.evaluate_config, runner_mod.init_worker
    runner_mod.evaluate_config = counting
    runner_mod.init_worker = lambda *a, **k: None
    try:
        again = R.run_unit("spambase", 0, unit["root"], threads=1)
    finally:
        runner_mod.evaluate_config, runner_mod.init_worker = real_eval, real_init

    assert calls["n"] == 0, f"a completed unit re-evaluated {calls['n']} configurations"
    assert set(again["stages_complete"]) == set(R.STAGES)
    assert set(before["stages_complete"]) == set(R.STAGES)


# ---------------------------------------------------------------------------
# the holdout is read last -- proved dynamically, not by reading the source
# ---------------------------------------------------------------------------

def test_no_holdout_evaluation_happens_before_the_holdout_stage(unit):
    """The holdout partition must be untouched until the confirmation stage.

    The dry run checks this statically. A static check on this property is a proxy
    -- the previous one asserted that the text near the call mentioned the stage
    name, and it broke when the compute function moved while the call site did not.
    This is the dynamic version: every holdout evaluation is keyed under
    fold_id="holdout", and the request log records the order, so the ordering can be
    read off the database the unit actually produced.
    """
    import sqlite3
    db = sqlite3.connect(str(unit["dir"] / "evaluations.sqlite"))
    rows = db.execute(
        "SELECT r.id, r.method, r.stage, e.fold_id FROM requests r "
        "LEFT JOIN evaluations e ON e.key = r.key ORDER BY r.id").fetchall()
    db.close()

    holdout_ids = [i for i, _m, _s, fold in rows if fold == "holdout"]
    assert holdout_ids, "no holdout evaluation was recorded at all"

    first_holdout = min(holdout_ids)
    before = {(m, s) for i, m, s, _f in rows if i < first_holdout}
    assert all(s != "holdout_audit" for _m, s in before)
    # every holdout request belongs to the holdout stage and to nothing else
    for i, m, s, fold in rows:
        if fold == "holdout":
            assert s == "holdout_audit" and m == "holdout_confirmation", (
                f"a holdout evaluation was charged to {m}/{s}")


def test_the_holdout_measurement_is_keyed_apart_from_the_internal_one(unit):
    """The same configuration measured two ways must not collide in the cache."""
    import sqlite3
    db = sqlite3.connect(str(unit["dir"] / "evaluations.sqlite"))
    folds = {f for (f,) in db.execute("SELECT DISTINCT fold_id FROM evaluations")}
    dupes = db.execute(
        "SELECT config, COUNT(DISTINCT fold_id) c FROM evaluations "
        "GROUP BY config HAVING c > 1").fetchall()
    db.close()
    assert "holdout" in folds and "all" in folds
    assert dupes, ("no configuration was measured both internally and on the "
                   "holdout, so this test cannot show they are keyed apart")


# ---------------------------------------------------------------------------
# contrast isolation AS THE RUNNER WIRED IT, not as the arm API allows
#
# tests/methodology/test_primary_contrasts.py exercises the arm functions directly
# and now does so non-vacuously, but it constructs the arms itself. It therefore
# proves that the API CAN isolate one mechanism, not that run_unit DID. The
# contrast could be broken by the runner passing a different anchor set, weight
# grid or config to one arm, and that test would still pass. These read the
# artifacts a real unit wrote.
# ---------------------------------------------------------------------------

def _diag(unit, stage):
    return json.loads((unit["dir"] / f"{stage}.json").read_text())["diagnostics"]


def test_the_geometry_contrast_shares_its_reference_as_the_runner_ran_it(unit):
    """WS-S and NBI-S must have been given the SAME surrogate reference."""
    ws, nbi_s = _diag(unit, "ws_s"), _diag(unit, "nbi_s")
    assert ws["utopia"] == nbi_s["utopia"], (
        "WS-S and NBI-S were run against different utopia points, so the geometry "
        "contrast also varies the reference")
    assert ws["n_weights"] == nbi_s["n_weights"]
    assert ws["scalarization"] != nbi_s["scalarization"], \
        "the contrast does not actually change the scalarization"


def test_the_anchor_contrast_changes_only_the_payoff_as_the_runner_ran_it(unit):
    """NBI-S and NBI-R must differ in anchor provenance and in nothing else."""
    s, r = _diag(unit, "nbi_s"), _diag(unit, "nbi_r")
    assert s["anchor_source"] != r["anchor_source"], \
        "the anchor contrast did not change anchor provenance"
    assert s["payoff_matrix"] != r["payoff_matrix"], \
        "the anchor contrast did not change the payoff matrix"
    # quasi_normal is NOT a setting: it is the Das and Dennis direction
    # n_hat = -Phi.1 / ||Phi.1||, derived from the payoff matrix. It must move when
    # the payoff matrix moves, and holding it fixed would mean the contrast had not
    # actually relocated the CHIM. It is checked below as a derived quantity.
    for key in ("scalarization", "n_weights", "equality_tolerance",
                "restrict_t_nonnegative", "feasibility_constraint_applied"):
        assert s[key] == r[key], (
            f"NBI-S and NBI-R differ in {key!r} as the runner ran them, which is "
            f"beyond anchor/payoff provenance")


def test_the_quasi_normal_is_derived_from_each_arms_own_payoff_matrix(unit):
    """n_hat = -Phi.1 / ||Phi.1||, recomputed from what each arm recorded.

    This is the mechanism the NBI-S to NBI-R contrast varies. Asserting the two arms
    share a quasi-normal would assert the contrast does nothing; asserting each
    derives its own from its own payoff matrix is the real invariant.
    """
    for stage in ("nbi_s", "nbi_r"):
        d = _diag(unit, stage)
        Phi = np.asarray(d["payoff_matrix"], dtype=float)
        got = np.asarray(d["quasi_normal"], dtype=float)
        want = -(Phi @ np.ones(Phi.shape[1]))
        want = want / np.linalg.norm(want)
        assert np.allclose(got, want, atol=1e-9), (
            f"{stage}: recorded quasi-normal {got} is not -Phi.1/||Phi.1|| = {want}")
    assert not np.allclose(np.asarray(_diag(unit, "nbi_s")["quasi_normal"]),
                           np.asarray(_diag(unit, "nbi_r")["quasi_normal"])), (
        "NBI-S and NBI-R share a quasi-normal, so relocating the anchors did not "
        "relocate the CHIM and the contrast varies nothing")


def test_both_historical_arms_used_the_same_weight_grid_shape_as_declared(unit):
    """HISTORICAL-WS-asrun keeps the dissertation's asymmetric grid; the shared-
    specification arm uses the symmetric one. That difference is the point."""
    asrun = _diag(unit, "historical_ws_asrun")
    shared = _diag(unit, "historical_ws")
    assert asrun["weight_grid_is_symmetric"] is False, \
        "the as-run arm lost the dissertation's missing pure-quality vertex"
    assert shared.get("weight_grid_symmetric") is True, \
        "the shared-specification arm is not on the symmetric grid"


def test_the_shared_specification_arm_differs_from_ws_s_only_in_normalization(unit):
    """HISTORICAL-WS vs WS-S is the normalization contrast and nothing else."""
    ws, hist = _diag(unit, "ws_s"), _diag(unit, "historical_ws")
    assert ws["scalarization"] == hist["scalarization"]
    assert ws["n_weights"] == hist["n_weights"]
    assert ws["reference_construction"] != hist["reference_construction"], \
        "the normalization contrast did not change the reference construction"


# ---------------------------------------------------------------------------
# the contrast fingerprint, as a live runner invariant
# ---------------------------------------------------------------------------

def test_the_three_surrogate_reference_arms_share_a_contrast_fingerprint(unit):
    """WS-S, HISTORICAL-WS and NBI-S must differ only in their declared mechanism."""
    fps = {}
    for stage in ("ws_s", "historical_ws", "nbi_s"):
        payload = json.loads((unit["dir"] / f"{stage}.json").read_text())
        assert "contrast_fingerprint" in payload, f"{stage} recorded no fingerprint"
        fps[stage] = payload["contrast_fingerprint"]
    for key in ("solver", "realizer", "weights", "surrogate_object_ids"):
        vals = {s: fp[key] for s, fp in fps.items()}
        assert len(set(map(json.dumps, vals.values()))) == 1, (
            f"the three surrogate-reference arms differ in {key!r}: {vals}")


def test_nbi_r_differs_from_nbi_s_only_in_the_reference(unit):
    s = json.loads((unit["dir"] / "nbi_s.json").read_text())["contrast_fingerprint"]
    r = json.loads((unit["dir"] / "nbi_r.json").read_text())["contrast_fingerprint"]
    for key in ("solver", "realizer", "weights", "surrogate_object_ids"):
        assert s[key] == r[key], f"NBI-S and NBI-R differ in {key!r} beyond the anchors"
    assert s["reference_utopia"] != r["reference_utopia"], (
        "NBI-S and NBI-R share a reference utopia, so the anchor contrast varies "
        "nothing")


def test_the_contrast_guard_rejects_a_mismatched_arm():
    """The guard must fail on the mutation it exists to catch."""
    from doe_xgb.campaign.runner import MethodologicalFailure, _require_same
    base = {"solver": [10, 7, 500], "realizer": ["a"], "weights": [[1.0, 0.0]],
            "surrogate_object_ids": [1, 2], "reference_utopia": [0.0, 0.0]}
    _require_same(base, dict(base), arm="NBI-S")                       # identical: ok
    with pytest.raises(MethodologicalFailure, match="configured differently"):
        _require_same(base, {**base, "solver": [15, 7, 500]}, arm="NBI-S")
    with pytest.raises(MethodologicalFailure, match="weights"):
        _require_same(base, {**base, "weights": [[0.5, 0.5]]}, arm="NBI-S")
    # the permitted difference for the anchor contrast
    _require_same(base, {**base, "reference_utopia": [1.0, 1.0]},
                  arm="NBI-R", except_keys=("reference_utopia",))


def test_the_primary_reference_is_named(unit):
    """Amendment 21: one reference carries the primary family, named before launch."""
    m = json.loads((unit["dir"] / "metrics_by_method.json").read_text())
    assert m["primary_reference"] == "core"
    assert m["primary_indicator"] == "hv_ratio"
    assert m["augmented_is_a_declared_sensitivity"] is True
    for name, block in m["methods"].items():
        assert block["primary_reference"] == "core", f"{name} names no primary reference"


def test_both_references_are_still_computed_for_every_scored_method(unit):
    """Naming a primary must not delete the sensitivity."""
    m = json.loads((unit["dir"] / "metrics_by_method.json").read_text())
    assert set(m["references_computed"]) == {"core", "augmented"}
    for name, block in m["methods"].items():
        assert "hv_ratio" in block["core"], f"{name} has no core hv_ratio"
        assert "hv_ratio" in block["augmented"], f"{name} has no augmented hv_ratio"


def test_single_objective_comparators_are_scored_on_endpoints_only(unit):
    m = json.loads((unit["dir"] / "metrics_by_method.json").read_text())
    for name in R.SINGLE_OBJECTIVE:
        assert name not in m["methods"], f"{name} entered the front-indicator table"
        assert name in m["single_objective_endpoints"]


def test_the_anchor_injection_control_is_scored(unit):
    m = json.loads((unit["dir"] / "metrics_by_method.json").read_text())
    assert "nbi_s_plus_anchors" in m["methods"], \
        "the mandatory anchor-injection control produced no indicator row"
