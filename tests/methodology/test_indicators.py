"""IGD+ must be IGD+, verified against a hand computation and against pymoo.

The adversarial review found code named IGD+ computing ordinary IGD. The protocol
names IGD+ among its indicators, so a value under that label that is not IGD+ is a
mislabelled primary quantity. These tests close that completely:

* a hand-computable case with the arithmetic written out;
* agreement with pymoo's reference implementation on random problems;
* zero on identical fronts;
* the dominance-sensitive cases where IGD and IGD+ deliberately disagree.
"""
from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.reporting import dominance_filter, dominated_fraction, hypervolume, igd, igd_plus

pymoo = pytest.importorskip("pymoo")
from pymoo.indicators.igd_plus import IGDPlus          # noqa: E402
from pymoo.indicators.hv import HV                      # noqa: E402


def test_igd_plus_hand_computation() -> None:
    """One reference point, one solution, arithmetic written out.

    Reference z = (1, 1). Solution a = (2, 0.5): worse than z in the first
    component by 1, better in the second. IGD+ counts only the worse components,
    so d+ = sqrt(1^2 + 0^2) = 1. Plain IGD uses the full Euclidean distance,
    sqrt(1^2 + 0.5^2) = 1.11803.
    """
    R = np.array([[1.0, 1.0]])
    A = np.array([[2.0, 0.5]])
    assert igd_plus(A, R) == pytest.approx(1.0, abs=1e-12)
    assert igd(A, R) == pytest.approx(np.sqrt(1.25), abs=1e-12)


def test_igd_plus_is_zero_when_the_set_contains_the_reference() -> None:
    R = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    assert igd_plus(R, R) == pytest.approx(0.0, abs=1e-12)


def test_igd_plus_is_zero_for_a_set_that_dominates_the_reference_everywhere() -> None:
    """The defining property: being better than the reference costs nothing."""
    R = np.array([[1.0, 1.0], [2.0, 0.5]])
    A = R - 0.3                                   # strictly better in both components
    assert igd_plus(A, R) == pytest.approx(0.0, abs=1e-12)
    assert igd(A, R) > 0.0, "plain IGD penalizes a dominating set; that is why it is not used"


@pytest.mark.parametrize("seed", range(8))
def test_igd_plus_agrees_with_pymoo(seed: int) -> None:
    rng = np.random.default_rng(seed)
    q = int(rng.integers(2, 4))
    R = rng.random((rng.integers(5, 25), q))
    A = rng.random((rng.integers(3, 20), q))
    assert igd_plus(A, R) == pytest.approx(float(IGDPlus(R)(A)), rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_exact_two_objective_hypervolume_agrees_with_pymoo(seed: int) -> None:
    """The primary indicator, checked against a reference implementation."""
    rng = np.random.default_rng(100 + seed)
    F = rng.random((rng.integers(4, 30), 2))
    ref = np.array([1.1, 1.1])
    assert hypervolume(F, ref) == pytest.approx(float(HV(ref_point=ref)(F)), rel=1e-9, abs=1e-12)


def test_igd_and_igd_plus_diverge_exactly_where_they_should() -> None:
    """A set that dominates part of the reference is where the two disagree."""
    R = np.array([[0.5, 0.5], [0.2, 0.9], [0.9, 0.2]])
    A = np.array([[0.3, 0.3]])                    # dominates the first reference point
    assert igd_plus(A, R) < igd(A, R)


def test_dominance_filter_and_dominated_fraction() -> None:
    F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0], [0.9, 0.9], [0.6, 0.6]])
    keep = dominance_filter(F)
    assert sorted(keep.tolist()) == [0, 1, 2]
    assert dominated_fraction(F) == pytest.approx(2 / 5)
    assert dominated_fraction(F[keep]) == pytest.approx(0.0)


def test_dominance_filter_agrees_with_pymoo_nondominated_sorting() -> None:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    rng = np.random.default_rng(7)
    for _ in range(5):
        F = rng.random((30, 2))
        mine = set(dominance_filter(F).tolist())
        theirs = set(NonDominatedSorting().do(F, only_non_dominated_front=True).tolist())
        assert mine == theirs
