"""Guard the factor-stage decisions of the Paper-2 protocol.

`papers/xgboost_hpo_vrfnbi/protocol/EXPERIMENT_PROTOCOL.md` section 6.3 fixes four
things that the PCA/Varimax audit showed cannot be left to a default:

  * objectives are canonicalized to **minimization**, declared per objective;
  * the reported loadings are **scaled** (eigenvectors times the square root of the
    eigenvalues), not the eigenvector matrix;
  * Varimax is applied to those scaled loadings;
  * the quality composite is weighted by **explained-variance share**, with the
    unweighted mean retained as the pre-registered sensitivity.

These are protocol commitments, so they are asserted here rather than trusted to a
comment. A change that breaks one of them should fail a test, not pass review.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "scripts" / "pilot_stage_a_screening.py"


def _load():
    spec = importlib.util.spec_from_file_location("pilot_stage_a", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pilot = pytest.importorskip("numpy") and _load()


def _frame(n=60, seed=0) -> pd.DataFrame:
    """A design-shaped frame in which quality and cost genuinely conflict."""
    rng = np.random.default_rng(seed)
    latent_q = rng.normal(size=n)
    latent_c = 0.7 * latent_q + 0.7 * rng.normal(size=n)
    return pd.DataFrame({
        "Accuracy_Mean": 0.8 + 0.05 * latent_q + 0.005 * rng.normal(size=n),
        "Precision_Mean": 0.7 + 0.05 * latent_q + 0.005 * rng.normal(size=n),
        "Recall_Mean": 0.6 + 0.04 * latent_q + 0.01 * rng.normal(size=n),
        "Specificity_Mean": 0.9 + 0.03 * latent_q + 0.005 * rng.normal(size=n),
        "RocAuc_Mean": 0.85 + 0.05 * latent_q + 0.004 * rng.normal(size=n),
        "LogLoss_Mean": 0.5 - 0.05 * latent_q + 0.01 * rng.normal(size=n),
        "Leaves_Mean": 5000 + 2000 * latent_c + 100 * rng.normal(size=n),
    })


def test_every_response_declares_a_direction() -> None:
    """No direction is inferred from a loading sign (audit finding, and D3)."""
    for name, spec in pilot.RESPONSES.items():
        assert "minimize" in spec, f"{name} does not declare a direction"
        assert isinstance(spec["minimize"], bool)
    assert pilot.RESPONSES["LogLoss_Mean"]["minimize"] is True
    assert pilot.RESPONSES["Leaves_Mean"]["minimize"] is True
    assert pilot.RESPONSES["Accuracy_Mean"]["minimize"] is False


def test_loadings_are_scaled_not_eigenvectors() -> None:
    """The reported loadings must NOT have unit-norm columns.

    This is the defect the PCA/Varimax audit found in the frozen dissertation code.
    """
    out = pilot.factor_stage(_frame())
    norms = np.linalg.norm(out["loadings"].to_numpy(), axis=0)
    assert not np.allclose(norms, 1.0, atol=1e-6), (
        f"loading columns have norms {norms}; unit norms mean eigenvectors were "
        "reported instead of loadings"
    )
    # A Varimax rotation is orthogonal, so it preserves the total sum of squares.
    total = float((out["loadings"].to_numpy() ** 2).sum())
    assert total > 1.0 + 1e-6


def test_quality_weighting_is_by_explained_variance_not_equal() -> None:
    out = pilot.factor_stage(_frame())
    w = np.asarray(out["quality_weights"], dtype=float)
    assert np.isclose(w.sum(), 1.0)
    assert not np.allclose(w, 1.0 / len(w), atol=1e-3), (
        f"weights {w} are equal; the protocol fixes explained-variance weighting"
    )
    shares = np.asarray(out["explained_variance_share"], dtype=float)
    assert np.all(np.diff(shares) <= 1e-12), "shares must be in descending order"


def test_equal_weighting_is_retained_as_the_sensitivity() -> None:
    out = pilot.factor_stage(_frame())
    assert "quality_equal" in out
    assert out["quality_equal"].shape == out["quality"].shape
    # The two must be genuinely different objects, or the sensitivity is vacuous.
    assert not np.allclose(out["quality"], out["quality_equal"])


def test_cost_factor_is_identified_by_the_declared_cost_response() -> None:
    out = pilot.factor_stage(_frame())
    L = out["loadings"]
    cost_row = L.loc["Leaves_Mean"].abs()
    assert int(np.argmax(cost_row.to_numpy())) + 1 == out["cost_factor"]


def test_front_curvature_is_zero_on_a_straight_front() -> None:
    q = np.linspace(0.0, 1.0, 25)
    c = 1.0 - q                       # exactly linear trade-off
    assert pilot.front_curvature(q, c) == pytest.approx(0.0, abs=1e-9)


def test_front_curvature_is_positive_on_a_curved_front() -> None:
    q = np.linspace(0.0, 1.0, 25)
    c = 1.0 - q**2                    # concave
    assert pilot.front_curvature(q, c) > 0.05


def test_latin_hypercube_validation_points_lie_in_the_declared_box() -> None:
    pts = pilot.lhs_points(50, seed=1)
    assert list(pts.columns) == pilot.PARAMS
    for p in pilot.PARAMS:
        lo, hi = pilot.BOUNDS[p]
        assert pts[p].min() >= lo - 1e-9 and pts[p].max() <= hi + 1e-9
    # A held-out set that repeats a design row is not held out.
    assert pts.round(6).duplicated().sum() == 0


def test_factor_model_is_fitted_once_and_applied_not_refitted() -> None:
    """Refitting on a second sample is not the same as applying the first fit.

    A screening that refits the factor stage on the held-out set compares a surface
    against targets in a different coordinate system: different standardization,
    different rotation, different sign orientation. The two must therefore be
    measurably different, or the distinction the API draws is decorative.
    """
    a, b = _frame(n=80, seed=1), _frame(n=80, seed=2)
    m = pilot.FactorModel().fit(a)

    # Applying the fitted model reproduces its own fit exactly.
    np.testing.assert_allclose(m.transform(a)["quality"],
                               pilot.factor_stage(a)["quality"], atol=1e-12)

    applied = m.transform(b)["quality"]          # correct: a's model, applied to b
    refitted = pilot.factor_stage(b)["quality"]  # the bug: b gets its own model
    assert not np.allclose(applied, refitted, atol=1e-6), (
        "applying and refitting gave identical composites, so this test cannot "
        "detect the coordinate-system error it exists to catch"
    )


def test_transform_uses_the_fitted_standardization() -> None:
    """Shifting the held-out responses must move the transformed scores.

    If transform re-standardized its input, a uniform shift would cancel and the
    held-out points would be scored on their own scale rather than the design's.
    """
    a = _frame(n=80, seed=3)
    m = pilot.FactorModel().fit(a)
    b = a.copy()
    b["Leaves_Mean"] = b["Leaves_Mean"] + 5000.0
    assert not np.allclose(m.transform(a)["cost"], m.transform(b)["cost"], atol=1e-6)


def test_factor_signs_are_oriented_deterministically() -> None:
    """Each factor's dominant response must load positively.

    A principal component's sign is arbitrary and Varimax does not fix it, so without
    an explicit orientation rule the quality composite's sign is arbitrary and its
    measured conflict with the cost factor can come out either way.
    """
    out = pilot.factor_stage(_frame(n=90, seed=7))
    L = out["loadings"].to_numpy()
    for j in range(L.shape[1]):
        dominant = int(np.argmax(np.abs(L[:, j])))
        assert L[dominant, j] > 0, (
            f"factor {j+1}'s dominant loading is negative; the orientation rule did "
            "not run"
        )


def test_measured_conflict_sign_is_stable_under_a_benign_respecification() -> None:
    """Changing a transform must not flip the sign of the objective conflict.

    Every response is canonicalized to minimization, so a positive correlation between
    the quality composite and the cost factor means the objectives agree and a negative
    one means they conflict. That reading has to survive a change that does not alter
    which configurations are good.
    """
    df = _frame(n=90, seed=11)
    from scipy.stats import spearmanr
    signs = []
    for transform in ("none", "log1p"):
        original = pilot.RESPONSES["LogLoss_Mean"]["transform"]
        pilot.RESPONSES["LogLoss_Mean"]["transform"] = transform
        try:
            out = pilot.factor_stage(df)
            signs.append(np.sign(spearmanr(out["quality"], out["cost"]).statistic))
        finally:
            pilot.RESPONSES["LogLoss_Mean"]["transform"] = original
    assert signs[0] == signs[1], f"conflict sign flipped between transforms: {signs}"
