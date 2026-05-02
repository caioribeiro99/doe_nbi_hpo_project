"""Unit tests for DesignProvider, encoding/decoding, and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from doe_xgb.design import (
    DesignKind,
    DesignProvider,
    DesignSpec,
    FactorMeta,
)
from doe_xgb.design.coding import decode, encode

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DESIGN = REPO_ROOT / "data" / "design" / "hyperparameter_design.csv"


FACTORS = (
    FactorMeta("subsample", 0.05, 1.00, "float"),
    FactorMeta("colsample_bytree", 0.05, 1.00, "float"),
    FactorMeta("colsample_bylevel", 0.05, 1.00, "float"),
    FactorMeta("learning_rate", 0.01, 0.30, "float"),
    FactorMeta("max_depth", 3.0, 18.0, "int"),
    FactorMeta("gamma", 0.05, 5.00, "float"),
    FactorMeta("n_estimators", 50.0, 700.0, "int"),
)


def test_encode_decode_round_trip() -> None:
    lo = [f.lo for f in FACTORS]
    hi = [f.hi for f in FACTORS]
    raw = np.array([[0.1, 0.2, 0.3, 0.05, 5.0, 1.0, 200.0]])
    coded = encode(raw, lo, hi)
    back = decode(coded, lo, hi)
    np.testing.assert_allclose(back, raw, atol=1e-10)


def test_encode_maps_bounds_to_pm1() -> None:
    lo = np.array([0.0, 10.0])
    hi = np.array([1.0, 20.0])
    coded_lo = encode(np.array([[0.0, 10.0]]), lo, hi)
    coded_hi = encode(np.array([[1.0, 20.0]]), lo, hi)
    np.testing.assert_allclose(coded_lo, np.array([[-1.0, -1.0]]))
    np.testing.assert_allclose(coded_hi, np.array([[+1.0, +1.0]]))


def test_full_factorial_3_factors_no_centers() -> None:
    spec = DesignSpec(
        kind=DesignKind.FULL_FACTORIAL,
        factors=(FactorMeta("a", -1.0, 1.0), FactorMeta("b", -1.0, 1.0), FactorMeta("c", -1.0, 1.0)),
        n_center=0,
    )
    artifact = DesignProvider.build(spec)
    assert artifact.matrix_coded.shape == (8, 3)
    assert set(np.unique(artifact.matrix_coded.values).tolist()) == {-1.0, 1.0}


def test_ccd_face_centered_run_count() -> None:
    spec = DesignSpec(
        kind=DesignKind.CCD_FACE_CENTERED,
        factors=(FactorMeta("a", 0.0, 1.0), FactorMeta("b", 0.0, 1.0), FactorMeta("c", 0.0, 1.0)),
        n_center=4,
    )
    artifact = DesignProvider.build(spec)
    # 2^3 corners + 2*3 axial + 4 centers = 18
    assert artifact.matrix_coded.shape == (18, 3)
    coverage = artifact.diagnostics["coverage"]
    assert coverage["covers_lower_bound"] is True
    assert coverage["covers_upper_bound"] is True


def test_ccd_circumscribed_alpha_is_outside_box() -> None:
    spec = DesignSpec(
        kind=DesignKind.CCD_CIRCUMSCRIBED,
        factors=(FactorMeta("a", 0.0, 1.0), FactorMeta("b", 0.0, 1.0)),
        n_center=2,
    )
    artifact = DesignProvider.build(spec)
    assert artifact.metadata["alpha"] > 1.0


def test_box_behnken_runs() -> None:
    spec = DesignSpec(
        kind=DesignKind.BOX_BEHNKEN,
        factors=tuple(FactorMeta(f"x{i}", 0.0, 1.0) for i in range(3)),
        n_center=3,
    )
    artifact = DesignProvider.build(spec)
    # k=3: 3 pairs * 4 sign combos = 12 + 3 centers = 15
    assert artifact.matrix_coded.shape == (15, 3)


def test_lhs_returns_correct_shape_and_box() -> None:
    spec = DesignSpec(
        kind=DesignKind.LATIN_HYPERCUBE,
        factors=tuple(FactorMeta(f"x{i}", 0.0, 1.0) for i in range(3)),
        seed=0,
    )
    artifact = DesignProvider.build(spec)
    arr = artifact.matrix_coded.values
    assert arr.shape[1] == 3
    assert arr.min() >= -1.0 - 1e-9
    assert arr.max() <= 1.0 + 1e-9


def test_simplex_lattice_through_provider() -> None:
    spec = DesignSpec(kind=DesignKind.SIMPLEX_LATTICE, simplex_q=3, simplex_m=10)
    artifact = DesignProvider.build(spec)
    assert artifact.matrix_coded.shape == (66, 3)
    np.testing.assert_allclose(artifact.matrix_coded.sum(axis=1).values, 1.0, atol=1e-12)


def test_d_optimal_intentionally_not_implemented() -> None:
    spec = DesignSpec(
        kind=DesignKind.D_OPTIMAL,
        factors=(FactorMeta("a", 0.0, 1.0),),
    )
    with pytest.raises(NotImplementedError):
        DesignProvider.build(spec)


@pytest.mark.skipif(not EXTERNAL_DESIGN.exists(), reason="canonical design CSV not present")
def test_external_minitab_design_loads() -> None:
    spec = DesignSpec(
        kind=DesignKind.EXTERNAL_CSV,
        factors=FACTORS,
        external_path=EXTERNAL_DESIGN,
    )
    artifact = DesignProvider.build(spec)
    assert artifact.matrix_uncoded.shape[1] == 7
    assert artifact.matrix_coded.shape == artifact.matrix_uncoded.shape
    coded_min = artifact.matrix_coded.min().min()
    coded_max = artifact.matrix_coded.max().max()
    assert coded_min == pytest.approx(-1.0, abs=1e-9)
    assert coded_max == pytest.approx(+1.0, abs=1e-9)
    assert "sha256" in artifact.metadata
    assert artifact.metadata["n_runs"] == 88


def test_validate_for_model_rejects_mixture_on_ccd() -> None:
    spec = DesignSpec(
        kind=DesignKind.CCD_FACE_CENTERED,
        factors=(FactorMeta("a", 0.0, 1.0), FactorMeta("b", 0.0, 1.0)),
    )
    artifact = DesignProvider.build(spec)
    rep = DesignProvider.validate_for_model(artifact, family="mixture_scheffe", order="quadratic")
    assert rep.ok is False
    assert any("mixture_scheffe" in e for e in rep.errors)


def test_validate_for_model_warns_on_lhs_with_quadratic() -> None:
    spec = DesignSpec(
        kind=DesignKind.LATIN_HYPERCUBE,
        factors=(FactorMeta("a", 0.0, 1.0), FactorMeta("b", 0.0, 1.0)),
        seed=0,
    )
    artifact = DesignProvider.build(spec)
    rep = DesignProvider.validate_for_model(artifact, family="process_quadratic", order="quadratic")
    assert rep.ok is True
    assert any("LHS" in w or "Sobol" in w or "Space-filling" in w for w in rep.warnings)


def test_validate_for_model_rejects_process_quadratic_on_simplex() -> None:
    spec = DesignSpec(kind=DesignKind.SIMPLEX_LATTICE, simplex_q=3, simplex_m=4)
    artifact = DesignProvider.build(spec)
    rep = DesignProvider.validate_for_model(artifact, family="process_quadratic", order="quadratic")
    assert rep.ok is False
