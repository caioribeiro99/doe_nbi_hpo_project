"""Unit tests for the Scheffé metamodel (mixens.scheffe).

Adapted from doe_nbi_hpo_project tests/unit/test_model_families.py
(mixture parts; branch repo-publication-readiness, commit 0465466),
plus tests for the PCO213 additions (external validation, coefficient
summaries, weight-array prediction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mixens.diagnostics import matrix_diagnostics
from mixens.mixture_design import build_study_design, generate_simplex_lattice, sample_dirichlet
from mixens.scheffe import (
    MixtureScheffeModel,
    external_validation,
    summarize_coefficients,
)

W3 = ["w1", "w2", "w3"]


def test_mixture_scheffe_quadratic_fits_known_simplex_polynomial() -> None:
    pts = generate_simplex_lattice(3, 8)
    df = pd.DataFrame(pts, columns=W3)
    rng = np.random.default_rng(3)
    y = pd.Series(
        1.0 * df["w1"]
        + 2.0 * df["w2"]
        + 3.0 * df["w3"]
        + 4.0 * df["w1"] * df["w2"]
        + rng.normal(0, 1e-6, size=len(df))
    )
    model = MixtureScheffeModel.fit(df, y, component_names=W3, order="quadratic")
    coefs = dict(zip(model.terms, model.coefficients, strict=True))
    assert coefs["w1"] == pytest.approx(1.0, abs=1e-3)
    assert coefs["w2"] == pytest.approx(2.0, abs=1e-3)
    assert coefs["w3"] == pytest.approx(3.0, abs=1e-3)
    assert coefs["w1*w2"] == pytest.approx(4.0, abs=1e-3)


def test_mixture_scheffe_predict_round_trip() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=W3)
    y = pd.Series(0.5 * df["w1"] + 1.5 * df["w2"] + 2.5 * df["w3"])
    model = MixtureScheffeModel.fit(df, y, component_names=W3, order="linear")
    pred = model.predict(df)
    assert pred.shape == (len(df),)
    assert np.max(np.abs(pred - y.values)) < 1e-9


def test_mixture_scheffe_special_cubic_basis_size() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=W3)
    y = pd.Series(np.zeros(len(df)))
    model = MixtureScheffeModel.fit(df, y, component_names=W3, order="special_cubic")
    # Linear (3) + cross (3) + triple (1) = 7 terms.
    assert len(model.terms) == 7


def test_mixture_scheffe_quadratic_q5_has_15_terms_on_study_design() -> None:
    d = build_study_design(5)
    names = [f"w{i}" for i in range(1, 6)]
    df = pd.DataFrame(d, columns=names)
    y = pd.Series(np.linspace(0.0, 1.0, len(df)))
    model = MixtureScheffeModel.fit(df, y, component_names=names, order="quadratic")
    # Linear (5) + cross C(5,2)=10 → 15 terms; 21 runs → 6 residual df.
    assert len(model.terms) == 15
    assert model.fit_report.n_obs == 21
    diag = matrix_diagnostics(np.column_stack([df[n] for n in names]))
    assert diag["rank"] == 5


# ------------------------- PCO213 additions -------------------------


def _quadratic_truth(w: np.ndarray) -> np.ndarray:
    # y = 1*w1 + 2*w2 + 3*w3 + 4*w1*w2 (exactly quadratic in the simplex)
    return 1.0 * w[:, 0] + 2.0 * w[:, 1] + 3.0 * w[:, 2] + 4.0 * w[:, 0] * w[:, 1]


def test_predict_weights_matches_predict() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=W3)
    y = pd.Series(_quadratic_truth(pts))
    model = MixtureScheffeModel.fit(df, y, component_names=W3, order="quadratic")
    np.testing.assert_allclose(model.predict_weights(pts), model.predict(df))


def test_external_validation_exact_for_quadratic_truth() -> None:
    train = generate_simplex_lattice(3, 8)
    df = pd.DataFrame(train, columns=W3)
    model = MixtureScheffeModel.fit(
        df, pd.Series(_quadratic_truth(train)), component_names=W3, order="quadratic"
    )
    val = sample_dirichlet(3, 30, random_state=7)
    report = external_validation(model, val, _quadratic_truth(val))
    assert report["rmse"] < 1e-8
    assert report["n_val"] == 30
    assert report["rmse_relative_to_range"] < 1e-6


def test_summarize_coefficients_across_repeats() -> None:
    names = W3
    models = []
    for seed in (0, 1, 2):
        pts = generate_simplex_lattice(3, 8)
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(pts, columns=names)
        y = pd.Series(_quadratic_truth(pts) + rng.normal(0, 1e-3, size=len(pts)))
        models.append(MixtureScheffeModel.fit(df, y, component_names=names, order="quadratic"))
    table = summarize_coefficients(models)
    assert set(table.columns) == {"term", "mean", "min", "max"}
    assert len(table) == 6  # 3 linear + 3 cross
    row = table.set_index("term").loc["w1*w2"]
    assert row["min"] <= 4.0 <= row["max"] + 1e-2


def test_summarize_coefficients_rejects_mismatched_terms() -> None:
    pts = generate_simplex_lattice(3, 6)
    df = pd.DataFrame(pts, columns=W3)
    y = pd.Series(np.zeros(len(df)))
    m_lin = MixtureScheffeModel.fit(df, y, component_names=W3, order="linear")
    m_quad = MixtureScheffeModel.fit(df, y, component_names=W3, order="quadratic")
    with pytest.raises(ValueError):
        summarize_coefficients([m_lin, m_quad])
