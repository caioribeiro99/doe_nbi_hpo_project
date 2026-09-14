"""A gate failure must change nothing about execution.

Protocol section 7.1 makes the surrogate reliability gate diagnostic rather than
adaptive: all four arms run at every replication whatever the gate says, because
what surrogate-assisted Pareto construction does on an unreliable surface is one of
the study's questions. An adaptive gate would delete the answer.

These tests use a deliberately terrible surrogate -- one that cannot pass any
reasonable external check -- and assert that every arm still produces its full
candidate set from it.
"""
from __future__ import annotations

import numpy as np
import pytest

from doe_xgb.campaign.arms import (N_WEIGHTS, Realizer, empirical_reference,
                                   run_nbi_arm, run_ws_s, surrogate_reference)
from doe_xgb.nbi_core import NBIConfig

BOUNDS = {"a": (0.0, 1.0), "b": (50.0, 700.0)}
PARAMS, INTS = ["a", "b"], ["b"]


def realizer() -> Realizer:
    return Realizer(PARAMS, BOUNDS, INTS)


def cfg() -> NBIConfig:
    return NBIConfig(objective_count=2,
                     bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
                     n_starts=4, seed=11)


# A surface with almost no signal: a tiny linear term on one coordinate only.
# This is the shape backward elimination produced on Spambase's second quality
# factor, which scored external R2 0.080 with rank correlation 0.252.
def flat1(x) -> float:
    return float(1e-4 * x[0])


def flat2(x) -> float:
    return float(-1e-4 * x[0] + 1e-9 * x[1])


ARMS = ["WS-S", "NBI-S", "NBI-R"]


def _run_all():
    c = cfg()
    s_anchors, s_chim = surrogate_reference([flat1, flat2], c)
    out = {
        "WS-S": run_ws_s([flat1, flat2], c, s_anchors, realizer()),
        "NBI-S": run_nbi_arm("NBI-S", [flat1, flat2], c, s_anchors, s_chim, realizer()),
    }
    x_star = np.array([[-0.9, 0.1], [0.9, -0.1]])
    F_star = np.array([[flat1(x_star[0]), flat1(x_star[1])],
                       [flat2(x_star[0]), flat2(x_star[1])]])
    r_anchors, r_chim = empirical_reference(x_star, F_star, c)
    out["NBI-R"] = run_nbi_arm("NBI-R", [flat1, flat2], c, r_anchors, r_chim, realizer())
    return out


@pytest.mark.parametrize("arm", ARMS)
def test_a_gate_failing_surrogate_still_produces_a_full_candidate_set(arm: str) -> None:
    run = _run_all()[arm]
    assert len(run.candidates) == N_WEIGHTS, (
        f"{arm} returned {len(run.candidates)} candidates on an unreliable surface; "
        "the gate must not suppress an arm"
    )


@pytest.mark.parametrize("arm", ARMS)
def test_every_candidate_is_realizable_even_from_an_unreliable_surface(arm: str) -> None:
    for k in _run_all()[arm].candidates:
        assert set(k.config_realized) == set(PARAMS)
        assert isinstance(k.config_realized["b"], int)
        assert BOUNDS["b"][0] <= k.config_realized["b"] <= BOUNDS["b"][1]
        assert np.all(np.isfinite(k.x_continuous))


def test_the_arms_expose_no_gate_input_at_all() -> None:
    """The strongest form of the guarantee: the arms cannot see gate status.

    An arm that never receives the gate result cannot adapt to it, whatever a
    future edit to the runner does.
    """
    import inspect

    from doe_xgb.campaign import arms

    import re

    # "surrogate" contains "gate", so match whole words rather than substrings.
    forbidden = re.compile(r"(?:^|_)(?:gate|gated|reliability|reliable|passed|pass)(?:$|_)")
    for fn in (arms.run_ws_s, arms.run_nbi_arm, arms.run_historical_ws):
        names = set(inspect.signature(fn).parameters)
        leaked = {n for n in names if forbidden.search(n.lower())}
        assert not leaked, f"{fn.__name__} takes {leaked}; an arm must not see gate status"
    # and the guard itself must be capable of catching a leak
    assert forbidden.search("gate_pass")
    assert forbidden.search("reliability_gate")
    assert not forbidden.search("surrogates")


def test_degenerate_geometry_is_reported_rather_than_repaired() -> None:
    """A rank-deficient payoff matrix is a methodological failure, not a gate failure.

    The two must not be conflated: one is an uninformative surface, the other is an
    absence of a usable construction.
    """
    c = cfg()
    x_star = np.array([[0.2, 0.0], [0.2, 0.0]])          # coincident anchors
    F_star = np.array([[0.5, 0.5], [0.5, 0.5]])
    anchors, chim = empirical_reference(x_star, F_star, c)
    run = run_nbi_arm("NBI-R", [flat1, flat2], c, anchors, chim, realizer())
    assert run.diagnostics["anchors_coincide"] is True
    assert run.diagnostics["payoff_rank"] < 2
    assert run.diagnostics["payoff_condition_number"] is None
    assert len(run.candidates) == N_WEIGHTS, "the run still completes and is recorded"
