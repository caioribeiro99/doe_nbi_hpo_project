"""Unit tests for the simplex weight generators (mixens.mixture_design).

Adapted from doe_nbi_hpo_project tests/unit/test_simplex.py (branch
repo-publication-readiness, commit 0465466), plus tests for the PCO213
additions (axial points, Dirichlet sampling, composite study design).
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from mixens.mixture_design import (
    build_study_design,
    custom_weights,
    generate_axial_points,
    generate_extreme_vertices,
    generate_simplex_centroid,
    generate_simplex_lattice,
    sample_dirichlet,
    simplex_lattice_count,
    validate_weights,
)


@pytest.mark.parametrize(
    "q,m",
    [(2, 2), (2, 10), (2, 50), (3, 2), (3, 10), (4, 5), (5, 4), (7, 3)],
)
def test_simplex_lattice_count_matches_combinatorial(q: int, m: int) -> None:
    assert simplex_lattice_count(q, m) == comb(q + m - 1, m)


@pytest.mark.parametrize("q,m", [(2, 50), (3, 10), (4, 5), (5, 4)])
def test_simplex_lattice_shape_and_sum(q: int, m: int) -> None:
    w = generate_simplex_lattice(q, m)
    assert w.shape == (simplex_lattice_count(q, m), q)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert (w >= 0.0).all()


def test_simplex_lattice_q5_m2_study_design_base() -> None:
    """{5, 2} must yield 15 points: 5 vertices + 10 edge midpoints."""
    w = generate_simplex_lattice(5, 2)
    assert w.shape == (15, 5)


def test_simplex_lattice_q1_rejected() -> None:
    with pytest.raises(ValueError):
        generate_simplex_lattice(1, 5)
    with pytest.raises(ValueError):
        simplex_lattice_count(1, 5)


def test_simplex_lattice_invalid_m_rejected() -> None:
    with pytest.raises(ValueError):
        generate_simplex_lattice(3, 0)


def test_simplex_lattice_uniqueness() -> None:
    w = generate_simplex_lattice(4, 5)
    assert len(np.unique(np.round(w, 12), axis=0)) == w.shape[0]


@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_simplex_centroid_includes_vertices(q: int) -> None:
    pts = generate_simplex_centroid(q)
    for j in range(q):
        v = np.zeros(q)
        v[j] = 1.0
        assert any(np.allclose(row, v, atol=1e-12) for row in pts)


def test_validate_weights_accepts_valid() -> None:
    validate_weights(np.array([[0.5, 0.5], [0.2, 0.8]]))


def test_validate_weights_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        validate_weights(np.array([[0.5, 0.4], [0.2, 0.8]]))  # row-sum != 1
    with pytest.raises(ValueError):
        validate_weights(np.array([[1.5, -0.5]]))  # negative entry
    with pytest.raises(ValueError):
        validate_weights(np.array([0.5, 0.5]))  # 1-D


def test_custom_weights_round_trip() -> None:
    w = custom_weights([[0.1, 0.9], [0.5, 0.5]])
    assert w.shape == (2, 2)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_extreme_vertices_finds_some_feasible_points() -> None:
    pts = generate_extreme_vertices(np.array([0.1, 0.1, 0.1]), np.array([0.7, 0.7, 0.7]), n_grid=5)
    assert pts.shape[1] == 3
    np.testing.assert_allclose(pts.sum(axis=1), 1.0, atol=1e-12)


# ------------------------- PCO213 additions -------------------------


@pytest.mark.parametrize("q", [3, 4, 5])
def test_axial_points_on_simplex_with_correct_values(q: int) -> None:
    pts = generate_axial_points(q)
    assert pts.shape == (q, q)
    np.testing.assert_allclose(pts.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.diag(pts), (q + 1.0) / (2.0 * q))


def test_sample_dirichlet_valid_and_reproducible() -> None:
    a = sample_dirichlet(5, 25, random_state=123)
    b = sample_dirichlet(5, 25, random_state=123)
    assert a.shape == (25, 5)
    np.testing.assert_allclose(a, b)
    validate_weights(a)


def test_build_study_design_q5_has_21_unique_points() -> None:
    d = build_study_design(5)
    assert d.shape == (21, 5)
    validate_weights(d)
    # Contains the uniform-voting centroid and all 5 vertices.
    assert any(np.allclose(row, 0.2) for row in d)
    for j in range(5):
        v = np.zeros(5)
        v[j] = 1.0
        assert any(np.allclose(row, v, atol=1e-12) for row in d)
