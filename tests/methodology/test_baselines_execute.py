"""Every comparator must actually run, with the seeds the campaign will hand it.

Twice now a stage has died on an integration detail invisible to unit tests, deep
into a unit and after expensive real evaluations: first ``_run_historical`` could
not be called at all, then ``bayesian_optimization`` passed a 64-bit BLAKE2b seed
to ``GaussianProcessRegressor``, whose ``random_state`` is validated against
``[0, 2**32 - 1]``. Both were found by running the pipeline, at stage six and stage
thirteen of twenty. In the confirmatory campaign that is hours of real evaluations
before the traceback, times 120 units.

This exercises each comparator against a stub view -- no XGBoost, no data, so it
runs in under a second -- with REAL derived seeds for real (dataset, replication)
coordinates. It asserts only that each one executes, returns the shape the runner
expects, and spends its budget. It makes no claim about which comparator is any
good; that is the campaign's job, and inspecting it here would break the
confirmatory blindness this protocol is frozen under.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign import baselines as B
from doe_xgb.campaign.evaluator import PARAMS
from doe_xgb.campaign.runner import DATASETS
from doe_xgb.campaign.seeding import derive_seed


class StubView:
    """Stands in for a cache MethodView: deterministic, cheap, and it counts."""

    def __init__(self):
        self.calls = 0

    def evaluate(self, cfg: dict) -> dict:
        self.calls += 1
        x = np.array([float(cfg[p]) for p in PARAMS])
        # Two smooth, conflicting responses. Their values are irrelevant; they only
        # have to be finite and to vary with the configuration.
        return {"quality": float(np.sum(np.sin(x))),
                "cost": float(np.sum(np.cos(x * 0.5)))}


def _to_objectives(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["quality"].to_numpy(float),
                            df["cost"].to_numpy(float)])


BUDGET = 24


def _seed(method: str, dataset: str = "magic", rep: int = 0) -> int:
    return derive_seed(dataset, rep, method)


@pytest.mark.parametrize("dataset", DATASETS)
def test_grid_runs_and_spends_its_budget(dataset):
    v = StubView()
    out = B.coarse_grid(v, BUDGET, _seed("grid", dataset))
    assert len(out) == BUDGET and v.calls == BUDGET
    assert np.isfinite(out["quality"]).all()


@pytest.mark.parametrize("dataset", DATASETS)
def test_random_runs_and_spends_its_budget(dataset):
    v = StubView()
    out = B.random_search(v, BUDGET, _seed("random", dataset))
    assert len(out) == BUDGET and v.calls == BUDGET


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("obj", [0, 1])
def test_bayesian_optimization_runs(dataset, obj):
    """The exact stage that died on the 64-bit random_state."""
    v = StubView()
    method = "bayes_quality" if obj == 0 else "bayes_cost"
    out = B.bayesian_optimization(v, BUDGET, _seed(method, dataset), _to_objectives, obj)
    assert len(out) == BUDGET and v.calls == BUDGET
    assert np.isfinite(_to_objectives(out)).all()


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("obj", [0, 1])
def test_tpe_runs(dataset, obj):
    v = StubView()
    method = "tpe_quality" if obj == 0 else "tpe_cost"
    out = B.tpe(v, BUDGET, _seed(method, dataset), _to_objectives, obj)
    assert len(out) == BUDGET and v.calls == BUDGET


@pytest.mark.parametrize("dataset", DATASETS)
def test_nsga2_runs(dataset):
    v = StubView()
    out = B.nsga2(v, 8, 3, _seed("nsga2", dataset), _to_objectives)
    assert len(out) > 0 and v.calls > 0
    assert set(PARAMS).issubset(out.columns)


def test_every_comparator_runs_at_the_last_replication_too():
    """Seeds differ per replication; a wide one must not appear only at r = 29."""
    v = StubView()
    B.bayesian_optimization(v, BUDGET, _seed("bayes_quality", "bank_marketing", 29),
                            _to_objectives, 0)
    B.tpe(StubView(), BUDGET, _seed("tpe_cost", "adult", 29), _to_objectives, 1)
    B.nsga2(StubView(), 8, 3, _seed("nsga2", "spambase", 29), _to_objectives)


def test_comparators_do_not_share_a_candidate_stream():
    """The aliasing the seeding module exists to prevent, checked end to end."""
    from doe_xgb.campaign.seeding import candidate_hash

    runs = {
        "grid": B.coarse_grid(StubView(), BUDGET, _seed("grid")),
        "random": B.random_search(StubView(), BUDGET, _seed("random")),
        "bayes_quality": B.bayesian_optimization(
            StubView(), BUDGET, _seed("bayes_quality"), _to_objectives, 0),
        "tpe_quality": B.tpe(
            StubView(), BUDGET, _seed("tpe_quality"), _to_objectives, 0),
    }
    digests = {m: candidate_hash(df[list(PARAMS)].to_dict("records"))
               for m, df in runs.items()}
    assert len(set(digests.values())) == len(digests), (
        f"comparators share a candidate stream: {digests}")
