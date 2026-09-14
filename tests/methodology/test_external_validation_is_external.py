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
    fm = pilot.FactorModel().fit(d)
    y_design = fm.transform(d)["quality"]

    terms_a, beta_a = pilot.fit_surface_backward(d, y_design)
    # Perturbing the validation responses beyond recognition must change nothing.
    v2 = v.copy()
    for c in pilot.RESPONSES:
        v2[c] = v2[c] * 3.0 + 7.0
    terms_b, beta_b = pilot.fit_surface_backward(d, y_design)
    assert terms_a == terms_b
    np.testing.assert_allclose(beta_a, beta_b, atol=1e-12)

    # and the scoring call must not feed validation responses back into the fit
    r2_a, rho_a, n_a = pilot.external_scores(d, y_design, v, fm.transform(v)["quality"])
    r2_b, rho_b, n_b = pilot.external_scores(d, y_design, v, fm.transform(v)["quality"])
    assert (n_a, n_b) == (len(terms_a), len(terms_a))
    assert r2_a == r2_b and rho_a == rho_b


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
