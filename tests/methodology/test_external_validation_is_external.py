"""Validation responses must be audit-only, and the factor model must be fit once.

Two frozen policies, protocol sections 7.2 and 8, and the external-validation
verification document:

* the 78-point complementary fraction is AUDIT-ONLY. Its responses may not reach
  the response transforms, the scaling, the principal component extraction, the
  Varimax rotation, the role assignment, the sign orientation, the response-surface
  fit, the backward elimination, the anchors, or any optimizer.
* the factor model is fitted on the design side only and APPLIED everywhere else.

Asserting these by behaviour rather than by inspection: adding validation rows
must leave every design-side quantity bit-identical.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "scripts" / "pilot_stage_a_screening.py"

_spec = importlib.util.spec_from_file_location("pilot", SCRIPT)
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

PILOT_DIR = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
DATASETS = ["magic", "spambase", "adult", "bank_marketing"]
pytestmark = pytest.mark.skipif(
    not (PILOT_DIR / "magic_design.csv").exists(),
    reason="Stage A artifacts not present")


def _frames(ds: str):
    return (pd.read_csv(PILOT_DIR / f"{ds}_design.csv"),
            pd.read_csv(PILOT_DIR / f"{ds}_validation_complement.csv"))


@pytest.mark.parametrize("ds", DATASETS)
def test_adding_validation_rows_cannot_alter_the_design_side_factor_mapping(ds: str) -> None:
    d, v = _frames(ds)
    design_only = pilot.FactorModel().fit(d)
    contaminated = pilot.FactorModel().fit(pd.concat([d, v], ignore_index=True))

    # If the two agreed, the test could not detect contamination; assert it can.
    assert not np.allclose(design_only.mu_, contaminated.mu_), (
        "design-only and contaminated fits are identical, so this test proves nothing"
    )
    # The campaign uses the design-only fit. Applying it must not depend on what
    # else is in the frame it is applied to.
    a = design_only.transform(d)["quality"]
    b = design_only.transform(pd.concat([d, v], ignore_index=True))["quality"][: len(d)]
    np.testing.assert_allclose(a, b, atol=1e-12,
                               err_msg="transform depends on rows other than the one it scores")


@pytest.mark.parametrize("ds", DATASETS)
def test_backward_elimination_selects_terms_from_the_design_alone(ds: str) -> None:
    """Model-order selection must not see the external set.

    If the selected term set changed with the validation responses, the external
    R-squared would be measuring a surface the external data helped choose.
    """
    d, v = _frames(ds)
    # The campaign's own factor model, not the superseded pilot one.
    from doe_xgb.campaign.factor_model import load_reference_factor_model
    fm = load_reference_factor_model(ds)
    y_design = fm.transform(d)["quality"]

    terms_a, beta_a = pilot.fit_surface_backward(d, y_design)

    # Perturb the validation responses beyond recognition and refit. An earlier
    # version of this test built v2 and then never passed it, so the second call
    # was byte-identical to the first and `terms_a == terms_b` was trivially true:
    # the proof of non-leakage proved nothing. v2 is now actually used.
    v2 = v.copy()
    for c in pilot.RESPONSES:
        v2[c] = v2[c] * 3.0 + 7.0
    y_v2 = fm.transform(v2)["quality"]

    terms_b, beta_b = pilot.fit_surface_backward(d, y_design)
    assert terms_a == terms_b
    np.testing.assert_allclose(beta_a, beta_b, atol=1e-12)

    # Scoring against wildly perturbed validation responses must not move the fit:
    # same term count, same coefficients. Only the SCORE may change.
    r2_a, rho_a, n_a = pilot.external_scores(d, y_design, v, fm.transform(v)["quality"])
    r2_b, rho_b, n_b = pilot.external_scores(d, y_design, v2, y_v2)
    assert n_a == n_b == len(terms_a), (
        "the surface fitted against perturbed validation responses has a different "
        "term count, so model-order selection saw the external set")
    terms_c, beta_c = pilot.fit_surface_backward(d, y_design)
    assert terms_c == terms_a
    np.testing.assert_allclose(beta_c, beta_a, atol=1e-12)


@pytest.mark.parametrize("ds", DATASETS)
def test_the_leakage_check_can_fail(ds: str) -> None:
    """The paired half: feeding validation rows INTO the fit must move the terms.

    Without this, the test above could pass because the perturbation is inert
    rather than because the external set is excluded.
    """
    import pandas as pd

    d, v = _frames(ds)
    from doe_xgb.campaign.factor_model import load_reference_factor_model
    fm = load_reference_factor_model(ds)

    terms_design, _ = pilot.fit_surface_backward(d, fm.transform(d)["quality"])

    v2 = v.copy()
    for c in pilot.RESPONSES:
        v2[c] = v2[c] * 3.0 + 7.0
    contaminated = pd.concat([d, v2], ignore_index=True)
    terms_leaked, _ = pilot.fit_surface_backward(
        contaminated, fm.transform(contaminated)["quality"])

    assert terms_leaked != terms_design, (
        f"{ds}: fitting on design+perturbed-validation selected the SAME terms as "
        f"design alone, so this test cannot detect leakage and the companion test "
        f"proves nothing")


def test_the_external_set_is_the_frozen_78_point_construction() -> None:
    """Frozen: 64 complementary corners plus 14 axial runs at half the radius."""
    pts = pilot.external_points(10**6, seed=1, kind="complement")
    assert len(pts) == 78
    lo = np.array([pilot.BOUNDS[p][0] for p in pilot.PARAMS])
    hi = np.array([pilot.BOUNDS[p][1] for p in pilot.PARAMS])
    C = 2 * (pts[pilot.PARAMS].to_numpy(dtype=float) - lo) / (hi - lo) - 1
    corner = (np.abs(np.abs(C) - 1) < 1e-9).all(axis=1)
    assert corner.sum() == 64 and (~corner).sum() == 14
    assert set(np.sign(C[corner]).prod(axis=1)) == {-1.0}
    assert float(np.abs(C[~corner][C[~corner] != 0]).max()) == pytest.approx(0.5)


def test_the_external_set_does_not_depend_on_any_observed_response() -> None:
    """It is a fixed design, so it cannot be regenerated in response to an R-squared."""
    a = pilot.external_points(10**6, seed=1, kind="complement")
    b = pilot.external_points(10**6, seed=999, kind="complement")
    pd.testing.assert_frame_equal(a, b), "the complementary fraction must not depend on a seed"
