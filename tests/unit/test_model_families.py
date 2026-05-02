"""Unit tests for ProcessQuadraticRSM and MixtureScheffeModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.design import DesignKind, DesignProvider, DesignSpec, FactorMeta
from doe_xgb.model_families import (
    BackwardEliminationSpec,
    MixtureScheffeModel,
    ProcessQuadraticRSM,
    SurrogateSpec,
    make_surrogate,
    select_default_family,
)
from doe_xgb.simplex import generate_simplex_lattice


def _ccd_artifact(k: int = 3) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    spec = DesignSpec(
        kind=DesignKind.CCD_FACE_CENTERED,
        factors=tuple(FactorMeta(f"x{i}", 0.0, 1.0) for i in range(k)),
        n_center=4,
    )
    artifact = DesignProvider.build(spec)
    return artifact.matrix_coded, artifact.matrix_uncoded, [f"x{i}" for i in range(k)]


def test_process_quadratic_recovers_known_polynomial() -> None:
    coded, _, names = _ccd_artifact(k=3)
    # True model: y = 2 + 3*x0 - 1*x1 + 4*x0*x1 + 2*x2^2
    rng = np.random.default_rng(0)
    y = (
        2.0
        + 3.0 * coded["x0"]
        - 1.0 * coded["x1"]
        + 4.0 * coded["x0"] * coded["x1"]
        + 2.0 * coded["x2"] ** 2
        + rng.normal(0.0, 1e-6, size=len(coded))
    )
    model = ProcessQuadraticRSM.fit(
        coded,
        pd.Series(y),
        factor_names=names,
        order="quadratic",
        backward=None,  # do not eliminate; we want exact recovery
    )
    coefs = dict(zip(model.terms, model.coefficients, strict=True))
    assert coefs["Intercept"] == pytest.approx(2.0, abs=1e-3)
    assert coefs["x0"] == pytest.approx(3.0, abs=1e-3)
    assert coefs["x1"] == pytest.approx(-1.0, abs=1e-3)
    assert coefs["x0*x1"] == pytest.approx(4.0, abs=1e-3)
    assert coefs["x2*x2"] == pytest.approx(2.0, abs=1e-3)


def test_process_quadratic_predict_matches_design_rows() -> None:
    coded, _, names = _ccd_artifact(k=3)
    rng = np.random.default_rng(1)
    y = pd.Series(coded["x0"] + 0.5 * coded["x1"] + rng.normal(0, 1e-9, size=len(coded)))
    model = ProcessQuadraticRSM.fit(coded, y, factor_names=names, order="quadratic", backward=None)
    pred = model.predict(coded)
    assert pred.shape == (len(coded),)
    # Should be very close to y for low-noise data.
    assert np.max(np.abs(pred - y.values)) < 1e-3


def test_process_quadratic_backward_elimination_drops_irrelevant_terms() -> None:
    coded, _, names = _ccd_artifact(k=3)
    rng = np.random.default_rng(2)
    # Only x0 matters. Other terms should fall out of a strong-hierarchy
    # backward elimination at alpha=0.05.
    y = pd.Series(2.0 * coded["x0"] + rng.normal(0, 0.05, size=len(coded)))
    model = ProcessQuadraticRSM.fit(
        coded,
        y,
        factor_names=names,
        order="quadratic",
        backward=BackwardEliminationSpec(alpha=0.05, enforce_hierarchy=True),
    )
    # Intercept and x0 should always survive.
    assert "Intercept" in model.terms
    assert "x0" in model.terms
    # At least one of the irrelevant-only terms should be dropped.
    assert len(model.terms) < 1 + len(names) + len(names) + (len(names) * (len(names) - 1)) // 2


def test_mixture_scheffe_quadratic_fits_known_simplex_polynomial() -> None:
    pts = generate_simplex_lattice(3, 8)
    df = pd.DataFrame(pts, columns=["w1", "w2", "w3"])
    rng = np.random.default_rng(3)
    # Scheffé quadratic: y = b1*w1 + b2*w2 + b3*w3 + b12*w1*w2
    y = pd.Series(
        1.0 * df["w1"]
        + 2.0 * df["w2"]
        + 3.0 * df["w3"]
        + 4.0 * df["w1"] * df["w2"]
        + rng.normal(0, 1e-6, size=len(df))
    )
    model = MixtureScheffeModel.fit(df, y, component_names=["w1", "w2", "w3"], order="quadratic")
    coefs = dict(zip(model.terms, model.coefficients, strict=True))
    assert coefs["w1"] == pytest.approx(1.0, abs=1e-3)
    assert coefs["w2"] == pytest.approx(2.0, abs=1e-3)
    assert coefs["w3"] == pytest.approx(3.0, abs=1e-3)
    assert coefs["w1*w2"] == pytest.approx(4.0, abs=1e-3)


def test_mixture_scheffe_predict_round_trip() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=["w1", "w2", "w3"])
    y = pd.Series(0.5 * df["w1"] + 1.5 * df["w2"] + 2.5 * df["w3"])
    model = MixtureScheffeModel.fit(df, y, component_names=["w1", "w2", "w3"], order="linear")
    pred = model.predict(df)
    assert pred.shape == (len(df),)
    assert np.max(np.abs(pred - y.values)) < 1e-9


def test_make_surrogate_dispatches_correctly() -> None:
    coded, _, names = _ccd_artifact(k=2)
    y = pd.Series(coded["x0"] + coded["x1"])
    spec_q = SurrogateSpec(family="process_quadratic", order="quadratic", backward_elimination=None)
    surr_q = make_surrogate(spec_q, coded, y, factor_names=names)
    assert isinstance(surr_q, ProcessQuadraticRSM)

    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=["w1", "w2", "w3"])
    y2 = pd.Series(df["w1"] + df["w2"] + df["w3"])
    spec_m = SurrogateSpec(family="mixture_scheffe", order="quadratic", backward_elimination=None)
    surr_m = make_surrogate(spec_m, df, y2, factor_names=["w1", "w2", "w3"])
    assert isinstance(surr_m, MixtureScheffeModel)


def test_select_default_family_picks_mixture_for_simplex() -> None:
    spec = DesignSpec(kind=DesignKind.SIMPLEX_LATTICE, simplex_q=3, simplex_m=4)
    artifact = DesignProvider.build(spec)
    assert select_default_family(artifact).family == "mixture_scheffe"


def test_select_default_family_picks_quadratic_for_ccd() -> None:
    spec = DesignSpec(
        kind=DesignKind.CCD_FACE_CENTERED,
        factors=(FactorMeta("a", 0.0, 1.0), FactorMeta("b", 0.0, 1.0)),
    )
    artifact = DesignProvider.build(spec)
    assert select_default_family(artifact).family == "process_quadratic"


def test_make_surrogate_rejects_unknown_family() -> None:
    coded, _, names = _ccd_artifact(k=2)
    y = pd.Series(coded["x0"])
    with pytest.raises(ValueError):
        make_surrogate(
            SurrogateSpec(family="black_box"),  # type: ignore[arg-type]
            coded,
            y,
            factor_names=names,
        )


def test_mixture_scheffe_special_cubic_basis_size() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=["w1", "w2", "w3"])
    y = pd.Series(np.zeros(len(df)))
    model = MixtureScheffeModel.fit(df, y, component_names=["w1", "w2", "w3"], order="special_cubic")
    # Linear (3) + cross (3) + triple (1) = 7 terms.
    assert len(model.terms) == 7


def test_process_quadratic_includes_only_main_when_order_linear() -> None:
    coded, _, names = _ccd_artifact(k=3)
    y = pd.Series(coded["x0"] + 0.5 * coded["x1"])
    model = ProcessQuadraticRSM.fit(
        coded, y, factor_names=names, order="linear", backward=None
    )
    # Intercept + 3 main effects = 4 terms.
    assert len(model.terms) == 4
