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


# --------------------------------------------------------------------------
# V9 oracle suite, re-run after every change to the indicator path
# --------------------------------------------------------------------------


def _rand_front(rng, n, q):
    return rng.random((n, q))


@pytest.mark.parametrize("seed", range(6))
def test_every_indicator_is_permutation_invariant(seed: int) -> None:
    """Reordering rows must not change any reported quantity."""
    from doe_xgb.campaign.scoring import indicators

    rng = np.random.default_rng(seed)
    F, R = _rand_front(rng, 18, 2), _rand_front(rng, 25, 2)
    a = indicators(F, R)
    perm = rng.permutation(len(F))
    b = indicators(F[perm], R[rng.permutation(len(R))])
    for k in a:
        if isinstance(a[k], float) and np.isnan(a[k]):
            assert np.isnan(b[k])
        else:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12), k


@pytest.mark.parametrize("seed", range(4))
def test_duplicated_points_do_not_change_the_front_or_hypervolume(seed: int) -> None:
    from doe_xgb.campaign.scoring import indicators

    rng = np.random.default_rng(100 + seed)
    F, R = _rand_front(rng, 12, 2), _rand_front(rng, 20, 2)
    a = indicators(F, R)
    b = indicators(np.vstack([F, F[:5]]), R)
    assert a["hv_ratio"] == pytest.approx(b["hv_ratio"], rel=1e-9)
    assert a["igd_plus"] == pytest.approx(b["igd_plus"], rel=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_adding_dominated_points_does_not_change_hypervolume_or_igd_plus(seed: int) -> None:
    """Contamination must be filtered out, since the indicators score the front."""
    from doe_xgb.campaign.scoring import indicators

    rng = np.random.default_rng(200 + seed)
    F, R = _rand_front(rng, 12, 2), _rand_front(rng, 20, 2)
    dominated = F.max(axis=0) + rng.random((6, 2)) * 0.1 + 0.05
    a, b = indicators(F, R), indicators(np.vstack([F, dominated]), R)
    assert a["hv_ratio"] == pytest.approx(b["hv_ratio"], rel=1e-9)
    assert a["igd_plus"] == pytest.approx(b["igd_plus"], rel=1e-9)


def test_generational_distance_is_zero_for_a_subset_of_the_reference() -> None:
    from doe_xgb.campaign.scoring import indicators

    R = np.array([[0.0, 1.0], [0.3, 0.6], [0.6, 0.3], [1.0, 0.0]])
    out = indicators(R[[1, 2]], R)
    assert out["gd"] == pytest.approx(0.0, abs=1e-12)


def test_the_joint_fraction_counts_the_methods_surviving_points() -> None:
    from doe_xgb.campaign.scoring import indicators

    R = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    on_front = np.array([[0.2, 0.7]])
    dominated = np.array([[0.9, 0.9]])
    assert indicators(on_front, np.vstack([R, on_front]))["joint_nondominated_fraction"] \
        == pytest.approx(1.0)
    assert indicators(dominated, R)["joint_nondominated_fraction"] == pytest.approx(0.0)


def test_hypervolume_ratio_is_one_when_the_set_is_the_reference() -> None:
    from doe_xgb.campaign.scoring import indicators

    R = np.array([[0.0, 1.0], [0.4, 0.5], [1.0, 0.0]])
    assert indicators(R, R)["hv_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert indicators(R, R)["igd_plus"] == pytest.approx(0.0, abs=1e-12)


def test_indicators_consume_the_dominance_filtered_set() -> None:
    """The protocol says the front; the code must not score the raw set."""
    from doe_xgb.campaign.scoring import indicators

    R = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    F = np.array([[0.2, 0.7], [0.8, 0.95]])       # the second is dominated by the first
    assert indicators(F, R)["n_front"] == 1
