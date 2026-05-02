"""Unit tests for the selection rules."""

from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.selection import SelectionRule, select


def test_distance_to_utopia_picks_closest_to_origin() -> None:
    F = np.array([[0.1, 0.9], [0.9, 0.1], [0.05, 0.05], [0.5, 0.5]])
    idx, info = select(F, SelectionRule.DISTANCE_TO_UTOPIA)
    assert idx == 2
    assert info["rule"] == "distance_to_utopia"


def test_max_quality_minimizes_first_canonical_column() -> None:
    F = np.array([[0.4, 0.5], [0.1, 0.9], [0.9, 0.1]])
    idx, info = select(F, SelectionRule.MAX_QUALITY, quality_index=0)
    # canonical-min: argmin of column 0 wins.
    assert idx == 1


def test_utility_uses_weights() -> None:
    F = np.array([[0.1, 0.9], [0.9, 0.1]])
    idx_q, _ = select(F, SelectionRule.UTILITY, weights=[1.0, 0.0])
    assert idx_q == 0
    idx_c, _ = select(F, SelectionRule.UTILITY, weights=[0.0, 1.0])
    assert idx_c == 1


def test_utility_rejects_wrong_weight_length() -> None:
    F = np.array([[0.1, 0.9], [0.9, 0.1]])
    with pytest.raises(ValueError):
        select(F, SelectionRule.UTILITY, weights=[1.0, 1.0, 1.0])


def test_knee_picks_a_curved_corner() -> None:
    # Curved 2D Pareto: linear front + one knee point.
    F = np.array(
        [
            [0.0, 1.0],   # extreme
            [1.0, 0.0],   # extreme
            [0.5, 0.5],   # straight-line midpoint
            [0.2, 0.2],   # knee (closer to utopia)
        ]
    )
    idx, info = select(F, SelectionRule.KNEE)
    assert idx == 3
    assert info["rule"] == "knee"


def test_lexicographic_orders_by_columns() -> None:
    F = np.array([[0.5, 0.5], [0.5, 0.4], [0.6, 0.0]])
    idx, _ = select(F, SelectionRule.LEXICOGRAPHIC)
    assert idx == 1  # tied on col 0 -> tiebreak col 1


def test_topsis_returns_valid_index() -> None:
    F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    idx, info = select(F, SelectionRule.TOPSIS, weights=[0.5, 0.5])
    assert 0 <= idx < len(F)
    assert "score" in info


def test_select_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        select(np.array([1.0, 2.0]), SelectionRule.MAX_QUALITY)
