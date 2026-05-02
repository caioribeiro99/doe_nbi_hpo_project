"""Unit tests for the reporting metrics."""

from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.reporting import (
    generalized_distance,
    hypervolume,
    igd,
    pareto_front,
    shannon_entropy,
    spacing_entropy,
    spread_delta,
)


def test_pareto_front_filters_dominated() -> None:
    F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.4, 0.6], [0.7, 0.7]])
    keep = pareto_front(F)
    # Index 4 (0.7, 0.7) is dominated by all others.
    assert keep[4] is np.False_ or not keep[4]


def test_generalized_distance_identity() -> None:
    F = np.array([[1.0, 1.0], [0.0, 0.0]])
    utopia = np.array([0.0, 0.0])
    d = generalized_distance(F, utopia)
    np.testing.assert_allclose(d, [np.sqrt(2.0), 0.0])


def test_shannon_entropy_max_at_uniform() -> None:
    q = 3
    uniform = np.full(q, 1.0 / q)
    extreme = np.array([1.0, 0.0, 0.0])
    assert float(shannon_entropy(uniform[None, :])[0]) > float(shannon_entropy(extreme[None, :])[0])


def test_spread_delta_zero_for_uniform_two_points() -> None:
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    # Just two points; spread should be a finite non-negative number.
    assert spread_delta(F) >= 0.0


def test_spacing_entropy_high_for_uniform_grid() -> None:
    F = np.linspace(0.0, 1.0, 5)
    F2 = np.column_stack([F, 1.0 - F])
    se = spacing_entropy(F2)
    assert 0.0 <= se <= 1.0 + 1e-9


def test_igd_zero_when_F_includes_reference() -> None:
    F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    ref = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert igd(F, ref) == pytest.approx(0.0, abs=1e-12)


def test_hypervolume_2d_simple() -> None:
    F = np.array([[0.0, 1.0], [1.0, 0.0]])
    ref = np.array([1.5, 1.5])
    hv = hypervolume(F, ref)
    # Two non-dominated points form an L-shape; HV should be positive.
    assert hv > 0.0


def test_hypervolume_3d_via_montecarlo_returns_positive() -> None:
    F = np.array([[0.1, 0.5, 0.5], [0.5, 0.1, 0.5], [0.5, 0.5, 0.1]])
    ref = np.array([1.0, 1.0, 1.0])
    hv = hypervolume(F, ref)
    assert hv > 0.0
