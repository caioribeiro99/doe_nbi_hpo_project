"""One factor model per dataset, APPLIED to every replication. Protocol 7.2.

The protocol froze this and gave the reason:

    if the model is refit per replication, the objective is not the same variable
    in every pair, so 30 paired indicator values do not live in one objective
    space, and no normalized indicator is invariant to that.

The runner called ``fit_factor_model(design_df)`` at the top of every unit anyway,
so for 120 units the two objectives were 120 slightly different pairs of variables
while the analysis plan treated them as 30 paired draws from one. That does not
crash and no test caught it: every indicator is computed consistently WITHIN a
unit, and only the paired inference across units is invalid. Nothing in the
artifacts would have looked wrong.

These tests pin the frozen model, its application, and the sensitivity 7.2 requires.
"""
from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.factor_model import (REFERENCE_MODEL_DIR, FrozenFactorModel,
                                           fit_factor_model,
                                           load_reference_factor_model,
                                           tucker_congruence)
from doe_xgb.campaign.runner import DATASETS

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNNER = REPO / "src" / "doe_xgb" / "campaign" / "runner.py"
PILOT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"


@pytest.mark.parametrize("dataset", DATASETS)
def test_a_frozen_model_is_committed_for_every_dataset(dataset):
    assert (REFERENCE_MODEL_DIR / f"{dataset}.json").exists()


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_frozen_model_was_fitted_on_the_design_rows_alone(dataset):
    """The reference set is the 88 design rows and NOTHING else.

    It was the 88 design rows plus their 78-point complement, per an earlier reading
    of EXPERIMENT_PROTOCOL.md 7.2. That complement is the audit-only external
    validation construction -- 64 of its 78 coded points identical to
    design.external_validation_set(), the other 14 the same axial runs differing only
    by integer rounding of max_depth. Fitting the objective definition on it made the
    surrogate gate validate a surface against data that had helped define its target.
    """
    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    ref = d["reference_set"]
    assert ref["n_design"] == 88
    assert ref["n_complement"] == 0
    assert ref["n_total"] == 88
    assert ref["external_validation_rows_used"] == 0
    assert ref["files"] == [f"{dataset}_design.csv"]
    assert "APPLIED to every replication" in d["fitted_on"]
    assert d["no_arm_result_in_input"] is True


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_committed_model_is_reproducible_from_its_declared_inputs(dataset):
    """The artifact must be exactly what its stated reference set produces."""
    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    rebuilt = fit_factor_model(design)
    frozen = load_reference_factor_model(dataset)
    assert np.allclose(frozen.rotated_loadings, rebuilt.rotated_loadings, atol=1e-12)
    assert np.allclose(frozen.quality_weights, rebuilt.quality_weights, atol=1e-12)
    assert frozen.cost_index == rebuilt.cost_index
    assert tuple(frozen.quality_indices) == tuple(rebuilt.quality_indices)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_model_round_trips_exactly(dataset):
    frozen = load_reference_factor_model(dataset)
    again = FrozenFactorModel.from_dict(json.loads(json.dumps(frozen.as_dict())))
    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    assert np.array_equal(frozen.objectives(design), again.objectives(design))


@pytest.mark.parametrize("dataset", DATASETS)
def test_every_replication_measures_the_same_objective(dataset):
    """The property 7.2 is for: the objective is one variable across replications.

    Two different design samples must be scored by the SAME transform, so the
    objective values a unit produces depend on the configurations it measured and
    not on which rows happened to be in that replication's fit.
    """
    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    a = design.sample(frac=0.8, random_state=1)
    b = design.sample(frac=0.8, random_state=2)
    fm = load_reference_factor_model(dataset)

    # Same rows, scored via two separately loaded models -> identical.
    shared = design.head(20)
    assert np.array_equal(fm.objectives(shared),
                          load_reference_factor_model(dataset).objectives(shared))

    # A per-replication REFIT would have made these disagree; the frozen model
    # cannot, because it never sees a or b.
    fa, fb = fit_factor_model(a), fit_factor_model(b)
    refit_gap = np.abs(fa.objectives(shared) - fb.objectives(shared)).max()
    frozen_gap = np.abs(fm.objectives(shared) - fm.objectives(shared)).max()
    assert frozen_gap == 0.0
    assert refit_gap > 0.0, "the refit path should differ; otherwise this proves nothing"


def test_the_runner_applies_and_does_not_fit_the_objective_model():
    """Static guard: the objective model must be loaded, not fitted, at campaign time."""
    src = RUNNER.read_text()
    tree = ast.parse(src)
    fit_calls, load_calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "fit_factor_model":
                fit_calls.append(node.lineno)
            if node.func.id == "load_reference_factor_model":
                load_calls.append(node.lineno)
    assert load_calls, "the runner never loads the frozen reference model"

    # fit_factor_model may appear ONLY for the reported sensitivity refit, which is
    # never assigned to the model the objectives come from.
    for lineno in fit_calls:
        line = src.splitlines()[lineno - 1]
        assert "refit" in line, (
            f"runner.py:{lineno} fits a factor model outside the reported "
            f"sensitivity: {line.strip()!r}. The objective model is frozen per "
            f"dataset by EXPERIMENT_PROTOCOL.md 7.2.")
    assert "fm = load_reference_factor_model(dataset)" in src
    assert "fm = fit_factor_model(" not in src


def test_tucker_congruence_is_one_against_itself():
    fm = load_reference_factor_model("magic")
    assert np.allclose(tucker_congruence(fm.rotated_loadings, fm.rotated_loadings), 1.0)


def test_tucker_congruence_survives_column_permutation_and_sign_flip():
    """Varimax fixes neither factor order nor orientation; congruence must not care."""
    fm = load_reference_factor_model("adult")
    L = fm.rotated_loadings
    perm = L[:, [2, 0, 1]] * np.array([1.0, -1.0, 1.0])
    phi = tucker_congruence(L, perm)
    assert np.allclose(np.abs(phi), 1.0, atol=1e-12)


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_frozen_composite_is_not_inverted(dataset):
    """The orientation invariant, checked once per dataset on the frozen artifact."""
    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    assert d["diagnostics_on_reference_set"]["composite_alignment_with_raw_quality"] > 0


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_factors_are_orthogonal_on_the_reference_set(dataset):
    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    assert d["diagnostics_on_reference_set"][
        "max_abs_offdiagonal_factor_correlation"] < 1e-8


@pytest.mark.parametrize("dataset", DATASETS)
def test_quality_and_cost_are_pearson_orthogonal_by_construction(dataset):
    """Recorded because it is the true mechanism behind the conflict-sign divergence.

    The quality composite is a weighted sum of rotated quality factors and the cost
    objective is another rotated factor from the same orthogonal basis, so their
    linear correlation is zero by construction. Amendment 19 originally attributed
    the divergence to specificity; that diagnosis was false.
    """
    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    r = d["diagnostics_on_reference_set"]["objective_conflict_latent_pearson"]
    assert abs(r) < 1e-10, f"expected ~0 by construction, got {r}"


def test_specificity_removal_does_not_explain_the_divergence():
    """The falsification of Amendment 19's original diagnosis, kept as a test."""
    from doe_xgb.campaign.factor_model import RESPONSES, apply_transforms
    from scipy.stats import spearmanr

    for dataset, disagrees in (("spambase", True), ("adult", True)):
        design = pd.read_csv(PILOT / f"{dataset}_design.csv")
        fm = fit_factor_model(design)
        t = fm.transform(design)
        M = apply_transforms(design)
        names = list(RESPONSES)
        idx = [i for i, n in enumerate(names)
               if RESPONSES[n]["role"] == "quality" and "Specificity" not in n]
        ref = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1)
                               for i in idx]).mean(axis=1)
        latent = spearmanr(t["quality"], t["cost"]).statistic
        raw_no_spec = spearmanr(ref, t["cost"]).statistic
        if disagrees:
            assert np.sign(latent) != np.sign(raw_no_spec), (
                f"{dataset}: removing specificity reconciled the sign, which would "
                f"support the withdrawn diagnosis")


# ---------------------------------------------------------------------------
# the leakage rule the reference set must obey (amendment 23)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", DATASETS)
def test_the_reference_set_contains_no_audit_only_row(dataset):
    """The objective definition must not be fitted on the external validation set.

    It was, for the life of the 166-point reference set: that set's second half is
    the audit-only external construction -- 64 of 78 coded points identical to
    design.external_validation_set(), the other 14 the same axial runs differing only
    by integer rounding of max_depth. The consequence is circular rather than merely
    procedural: the surrogate gate scores a fitted surface's prediction of the
    objective ON the external set, so if those responses helped define the objective,
    the gate validates a surface against data that informed its own target.
    """
    import numpy as np
    from doe_xgb.campaign.design import external_validation_set, to_coded
    from doe_xgb.campaign.evaluator import PARAMS

    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    assert d["reference_set"]["n_complement"] == 0
    assert d["reference_set"]["external_validation_rows_used"] == 0

    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    ext = external_validation_set()
    fitted = {tuple(r) for r in np.round(to_coded(design[list(PARAMS)]), 6).tolist()}
    audit = {tuple(r) for r in np.round(to_coded(ext[list(PARAMS)]), 6).tolist()}
    shared = fitted & audit
    assert not shared, (
        f"{dataset}: {len(shared)} audit-only coded points are inside the factor "
        f"model's fitting sample")


def test_the_complement_really_is_the_audit_only_construction():
    """The premise of amendment 23, asserted so the reasoning stays checkable."""
    import numpy as np
    from doe_xgb.campaign.design import external_validation_set, to_coded
    from doe_xgb.campaign.evaluator import PARAMS

    comp = pd.read_csv(PILOT / "magic_validation_complement.csv")
    ext = external_validation_set()
    assert len(comp) == len(ext) == 78
    c = {tuple(r) for r in np.round(to_coded(comp[list(PARAMS)]), 6).tolist()}
    e = {tuple(r) for r in np.round(to_coded(ext[list(PARAMS)]), 6).tolist()}
    assert len(c & e) >= 64, (
        f"only {len(c & e)} shared points; amendment 23's premise that the "
        f"complement IS the audit-only construction needs rechecking")


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_gate_scores_a_surface_whose_target_the_external_set_did_not_define(dataset):
    """The circularity amendment 23 removes, stated as an end-to-end property."""
    d = json.loads((REFERENCE_MODEL_DIR / f"{dataset}.json").read_text())
    assert d["reference_set"]["files"] == [f"{dataset}_design.csv"], (
        "the objective definition is fitted on something other than the design rows, "
        "so the surrogate gate may be validating against data that defined its target")
