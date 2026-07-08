"""Tests for Pareto helpers and MCDM selection rules (mixens.selection)."""

from __future__ import annotations

import numpy as np
import pytest

from mixens.selection import (
    SelectionRule,
    generational_distance,
    inverted_generational_distance,
    normalize_objectives,
    pareto_filter,
    select,
    spacing_metric,
)


def test_pareto_filter_removes_dominated() -> None:
    F = np.array([
        [1.0, 5.0],   # non-dominated (best f1)
        [2.0, 3.0],   # non-dominated
        [3.0, 1.0],   # non-dominated (best f2)
        [3.0, 3.0],   # dominated by [2,3]
        [4.0, 2.0],   # dominated by [3,1]
    ])
    mask = pareto_filter(F)
    np.testing.assert_array_equal(mask, [True, True, True, False, False])


def test_pareto_filter_keeps_duplicates_and_single_point() -> None:
    F = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    mask = pareto_filter(F)
    np.testing.assert_array_equal(mask, [True, True, False])
    assert pareto_filter(np.array([[3.0, 7.0]]))[0]


def test_normalize_objectives_maps_utopia_to_zero_nadir_to_one() -> None:
    F = np.array([[0.9, 0.2], [0.8, 0.5]])
    Fn = normalize_objectives(F, utopia=np.array([0.8, 0.2]), nadir=np.array([0.9, 0.5]))
    np.testing.assert_allclose(Fn, [[1.0, 0.0], [0.0, 1.0]])
    # default: column-wise min/max
    Fn2 = normalize_objectives(F)
    assert Fn2.min() == 0.0 and Fn2.max() == 1.0


def test_spacing_metric_prefers_even_fronts() -> None:
    even = np.column_stack([np.linspace(0, 1, 9), np.linspace(1, 0, 9)])
    uneven = even.copy()
    uneven[4] = [0.02, 0.97]  # clump one point near an extreme
    assert spacing_metric(even) == pytest.approx(0.0, abs=1e-12)
    assert spacing_metric(uneven) > spacing_metric(even)
    assert np.isnan(spacing_metric(even[:2]))


def test_gd_igd_zero_when_fronts_match_and_positive_otherwise() -> None:
    ref = np.column_stack([np.linspace(0, 1, 11), np.linspace(1, 0, 11)])
    assert generational_distance(ref, ref) == pytest.approx(0.0)
    assert inverted_generational_distance(ref, ref) == pytest.approx(0.0)
    shifted = ref + 0.1
    assert generational_distance(shifted, ref) > 0.0
    # a sparse front covers the reference worse than a dense identical one
    sparse = ref[::5]
    assert inverted_generational_distance(sparse, ref) > 0.0


def test_knee_selects_the_corner_of_an_L_front() -> None:
    # L-shaped front: knee at (0.1, 0.1)
    F = np.array([[0.0, 1.0], [0.02, 0.5], [0.1, 0.1], [0.5, 0.02], [1.0, 0.0]])
    idx, info = select(F, SelectionRule.KNEE)
    assert idx == 2
    assert info["rule"] == "knee"


def test_topsis_and_utility_pick_balanced_points() -> None:
    F = np.array([[0.0, 1.0], [0.4, 0.4], [1.0, 0.0]])
    idx_t, _ = select(F, SelectionRule.TOPSIS)
    idx_u, _ = select(F, SelectionRule.UTILITY)
    assert idx_t == 1
    assert idx_u == 1


def test_max_quality_and_lexicographic() -> None:
    F = np.array([[0.3, 9.0], [0.1, 5.0], [0.1, 2.0]])
    idx_q, _ = select(F, SelectionRule.MAX_QUALITY, quality_index=0)
    assert idx_q in (1, 2)  # argmin of column 0
    idx_l, _ = select(F, SelectionRule.LEXICOGRAPHIC)
    assert idx_l == 2  # ties on col 0 broken by col 1
