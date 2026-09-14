"""Real-objective revalidation, the two references, and the indicator conventions.

The distinctions these tests pin are the ones that would be easiest to lose:
an emitted candidate is not a realized one, a realized one is not necessarily
Pareto optimal, and the reference a method is graded against must be the same for
every method and must never be called a true Pareto front.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.scoring import (HV_REFERENCE_COORDINATE, augmented_reference,
                                      canonical_key, indicators, reference_core,
                                      revalidate)
from doe_xgb.campaign.evaluator import PARAMS, RESPONSES


class _View:
    """A stand-in evaluator that counts how many real evaluations it performed."""

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def evaluate(self, cfg, *, fold_id="all"):
        self.calls += 1
        return self.fn(cfg)


def _responses(cfg) -> dict:
    base = float(cfg["learning_rate"]) + float(cfg["max_depth"]) / 100.0
    return {"Accuracy_Mean": 0.8 + 0.01 * base, "Precision_Mean": 0.7 + 0.01 * base,
            "Recall_Mean": 0.6 + 0.01 * base, "Specificity_Mean": 0.9 - 0.01 * base,
            "RocAuc_Mean": 0.85 + 0.01 * base, "LogLoss_Mean": 0.5 - 0.05 * base,
            "Leaves_Mean": 100.0 * float(cfg["n_estimators"]),
            "Time_MeanFold": 0.1}


def _cfg(**kw) -> dict:
    base = {"subsample": 0.5, "colsample_bytree": 0.5, "colsample_bylevel": 0.5,
            "learning_rate": 0.1, "max_depth": 6, "gamma": 1.0, "n_estimators": 200}
    base.update(kw)
    return base


def _objectives(df: pd.DataFrame) -> np.ndarray:
    if not len(df):
        return np.zeros((0, 2))
    return np.column_stack([-df["Accuracy_Mean"].to_numpy(),
                            np.log1p(df["Leaves_Mean"].to_numpy())])


# --------------------------------------------------------------- realization

def test_configurations_that_realize_to_the_same_learner_are_one_configuration() -> None:
    a = canonical_key(_cfg(max_depth=6, n_estimators=200))
    b = canonical_key(_cfg(max_depth=6.0, n_estimators=200.0))
    assert a == b


def test_every_emitted_candidate_is_charged_while_the_fit_happens_once(tmp_path) -> None:
    """The budget is what a method requested; the cache decides what is computed.

    Deduplicating before requesting charged an arm the number of UNIQUE realized
    configurations instead of its declared budget, and the bias was systematic: a
    weighted sum's minimizers cluster at the anchors and collapse under rounding far
    more often than NBI's spread subproblem solutions, so the arm most understated
    was the geometry control in the study's own primary contrast.
    """
    from doe_xgb.campaign.evaluation_cache import EvaluationCache

    fits: list[dict] = []

    def compute(cfg):
        fits.append(cfg)
        return _responses(cfg)

    cache = EvaluationCache(tmp_path / "e.sqlite", dataset="magic",
                            split_id="rep_00", seed=1)
    view = cache.view("ws_s_revalidation", compute)
    emitted = [_cfg(n_estimators=200), _cfg(n_estimators=200), _cfg(n_estimators=400)]
    r = revalidate(emitted, view, _objectives)

    assert r["emitted"] == 3
    assert r["unique_realizable"] == 2, "duplicates realize to one learner"
    assert r["real_valid"] == 2
    assert len(fits) == 2, "the duplicate must not be fitted twice"
    assert cache.ledger("ws_s_revalidation").logical == 3, (
        "every emitted candidate must be charged, or an arm that emits duplicates "
        "appears to have spent less than its declared budget"
    )
    cache.close()


def test_twenty_identical_candidates_are_still_charged_twenty(tmp_path) -> None:
    """The worst case, which is exactly the weighted sum's characteristic behaviour."""
    from doe_xgb.campaign.evaluation_cache import EvaluationCache

    fits: list[dict] = []

    def compute(cfg):
        fits.append(cfg)
        return _responses(cfg)

    cache = EvaluationCache(tmp_path / "e.sqlite", dataset="magic",
                            split_id="rep_00", seed=1)
    view = cache.view("arm", compute)
    r = revalidate([_cfg(n_estimators=200)] * 20, view, _objectives)
    assert r["emitted"] == 20 and r["unique_realizable"] == 1
    assert len(fits) == 1
    assert cache.ledger("arm").logical == 20
    cache.close()


def test_revalidation_drops_non_finite_results_and_counts_them() -> None:
    def broken(cfg):
        out = _responses(cfg)
        if int(cfg["n_estimators"]) == 400:
            out["LogLoss_Mean"] = float("nan")
        return out

    r = revalidate([_cfg(n_estimators=200), _cfg(n_estimators=400)],
                   _View(broken), _objectives)
    assert r["unique_realizable"] == 2
    assert r["real_valid"] == 1


def test_dominated_fraction_after_revalidation_is_reported() -> None:
    """A returned set may contain points its own members dominate. That is a result."""
    emitted = [_cfg(n_estimators=100, learning_rate=0.3),      # good and small
               _cfg(n_estimators=700, learning_rate=0.01)]     # bad and large
    r = revalidate(emitted, _View(_responses), _objectives)
    assert "dominated_fraction_after_revalidation" in r
    assert 0.0 <= r["dominated_fraction_after_revalidation"] <= 1.0


# ---------------------------------------------------------------- references

def test_the_core_reference_contains_nothing_a_method_returned() -> None:
    design = pd.DataFrame([_responses(_cfg(n_estimators=n)) for n in (100, 300, 700)])
    core = reference_core([design], _objectives)
    assert core["n_points"] == 3
    # the note must forbid the phrase, which means it necessarily contains it
    assert "never called a true Pareto front" in core["note"]
    assert "method-independent" in core["note"]


def test_the_augmented_reference_quantifies_self_grading() -> None:
    core = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    per = {"NBI-S": np.array([[0.4, 0.4]]), "WS-S": np.array([[0.95, 0.95]])}
    aug = augmented_reference(core, per)
    share = aug["self_grading_share_of_front"]
    assert set(share) == {"core", "NBI-S", "WS-S"}
    assert share["NBI-S"] > 0, "a method contributing a front point is partly self-graded"
    assert abs(sum(share.values()) - 1.0) < 1e-9
    assert "never called a true Pareto front" in aug["note"]
    assert "self-graded" in aug["note"]


def test_every_method_is_scored_against_the_same_reference() -> None:
    core = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    per = {"A": np.array([[0.4, 0.6]]), "B": np.array([[0.9, 0.9]])}
    front = augmented_reference(core, per)["front"]
    ia, ib = indicators(per["A"], front), indicators(per["B"], front)
    # A is on the joint front, B is dominated by the core point (0.5, 0.5)
    assert ia["joint_nondominated_fraction"] == pytest.approx(1.0)
    assert ib["joint_nondominated_fraction"] == pytest.approx(0.0)
    assert ia["hv_ratio"] > ib["hv_ratio"]


def test_joint_fraction_uses_a_mask_not_the_index_array() -> None:
    """The bug this guards: dominance_filter returns indices, not a boolean mask.

    Averaging the index array silently produced a plausible-looking number.
    """
    core = np.array([[0.0, 1.0], [1.0, 0.0]])
    on_front = np.array([[0.2, 0.8]])
    front = augmented_reference(core, {"A": on_front})["front"]
    assert indicators(on_front, front)["joint_nondominated_fraction"] == pytest.approx(1.0)


def test_hypervolume_reference_point_is_fixed_before_any_front_is_seen() -> None:
    assert HV_REFERENCE_COORDINATE == 1.1


def test_indicators_return_nan_rather_than_a_number_on_an_empty_set() -> None:
    out = indicators(np.zeros((0, 2)), np.array([[0.0, 1.0]]))
    assert all(np.isnan(v) for k, v in out.items() if k != "n_front")


def test_a_surrogate_returns_a_python_float_not_a_length_one_array() -> None:
    """Every arm calls the surrogate through float(); numpy 2 refuses size-1 arrays."""
    import numpy as np

    from doe_xgb.campaign.design import make_surrogate, surface_terms

    terms = surface_terms()
    beta = np.arange(len(terms), dtype=float)
    f = make_surrogate(terms, beta)
    v = f(np.zeros(7))
    assert isinstance(v, float) and not isinstance(v, np.ndarray)
    assert float(v) == v


def test_no_campaign_module_calls_the_removed_ndarray_ptp() -> None:
    """numpy 2 removed ndarray.ptp; np.ptp(array, axis=) is the replacement.

    It bit twice, in two different files, because it only fires on the code path
    that runs it. A grep is cheaper than a third occurrence.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for d in (repo / "src" / "doe_xgb", repo / "scripts",
              repo / "papers" / "xgboost_hpo_vrfnbi" / "scripts"):
        for f in d.rglob("*.py"):
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if re.search(r"(?<!np)\.ptp\s*\(", line):
                    offenders.append(f"{f.relative_to(repo)}:{i}")
    assert not offenders, f"ndarray.ptp used at {offenders}; use np.ptp(x, axis=...)"
