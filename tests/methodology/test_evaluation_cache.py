"""The cache must save wall clock without distorting the scientific budget.

Two properties matter and both are easy to break:

* every request a method makes is charged to that method, whether or not the
  physical fit was already done, so caching cannot make one optimizer look
  cheaper than another;
* a method can reach only the evaluations it requested itself, so nothing leaks
  into a Bayesian surrogate, a Parzen history, an evolutionary population, an
  anchor search or a weighted-sum search.
"""
from __future__ import annotations

import pytest

from doe_xgb.campaign.evaluation_cache import (EvaluationCache, canonical_config,
                                               evaluation_key)


@pytest.fixture()
def cache(tmp_path):
    c = EvaluationCache(tmp_path / "evals.sqlite", dataset="magic",
                        split_id="rep_00", seed=42)
    yield c
    c.close()


def _counter():
    """A compute function that records how many real fits it performed."""
    calls: list[dict] = []

    def compute(cfg):
        calls.append(cfg)
        return {"leaves": 100.0 + len(calls), "time": 0.5, "acc": 0.9}

    return compute, calls


CFG_A = {"max_depth": 6, "n_estimators": 200, "learning_rate": 0.1}
CFG_B = {"max_depth": 9, "n_estimators": 400, "learning_rate": 0.05}


# --------------------------------------------------------------- normalization

def test_canonical_config_rounds_integers_and_stabilizes_floats() -> None:
    a = canonical_config({"max_depth": 5.5, "n_estimators": 137.4, "gamma": 0.1})
    b = canonical_config({"n_estimators": 137, "gamma": 0.1 + 1e-15, "max_depth": 6})
    assert a == b, "configurations that name the same learner must share a key"


def test_key_changes_with_every_pinned_component() -> None:
    base = dict(dataset="magic", split_id="rep_00", fold_id="all",
                config=CFG_A, seed=42)
    k = evaluation_key(**base)
    for field, value in [("dataset", "adult"), ("split_id", "rep_01"),
                         ("fold_id", "fold_2"), ("seed", 43)]:
        assert evaluation_key(**{**base, field: value}) != k, f"{field} not in the key"
    assert evaluation_key(**base, protocol_version="other") != k
    assert evaluation_key(**{**base, "config": CFG_B}) != k


# ------------------------------------------------------------------ accounting

def test_a_cache_hit_is_still_charged_to_the_requesting_method(cache) -> None:
    compute, calls = _counter()
    ws = cache.view("WS-S", compute)
    nbi = cache.view("NBI-S", compute)

    ws.evaluate(CFG_A)
    nbi.evaluate(CFG_A)          # same configuration, already computed

    assert len(calls) == 1, "the physical fit should have happened once"
    assert cache.ledger("WS-S").logical == 1
    assert cache.ledger("NBI-S").logical == 1, (
        "the second method was served from cache but must still be charged; "
        "otherwise caching makes it look cheaper than it is"
    )
    acc = cache.accounting()
    assert acc["logical_evaluations_total"] == 2
    assert acc["unique_physical_fits"] == 1
    assert acc["cache_hit_rate"] == pytest.approx(0.5)


def test_repeated_requests_by_one_method_are_charged_each_time(cache) -> None:
    compute, calls = _counter()
    v = cache.view("NBI-R", compute)
    for _ in range(3):
        v.evaluate(CFG_A)
    assert len(calls) == 1
    led = cache.ledger("NBI-R")
    assert led.logical == 3, "a method's own repetition is its own cost"
    assert led.unique_logical == 1


def test_physical_fits_count_distinct_evaluations_only(cache) -> None:
    compute, calls = _counter()
    a, b = cache.view("A", compute), cache.view("B", compute)
    a.evaluate(CFG_A); a.evaluate(CFG_B); b.evaluate(CFG_A); b.evaluate(CFG_B)
    assert cache.physical_fits() == 2
    assert len(calls) == 2
    assert cache.accounting()["logical_evaluations_total"] == 4


# -------------------------------------------------------------------- isolation

def test_a_method_sees_only_its_own_history(cache) -> None:
    compute, _ = _counter()
    a, b = cache.view("BO", compute), cache.view("TPE", compute)
    a.evaluate(CFG_A)
    b.evaluate(CFG_B)
    assert [h["config"] for h in a.history()] == [canonical_config(CFG_A)]
    assert [h["config"] for h in b.history()] == [canonical_config(CFG_B)]


def test_a_method_view_exposes_no_route_to_other_methods(cache) -> None:
    """An optimizer given this object must not be able to enumerate the table."""
    compute, _ = _counter()
    v = cache.view("NSGA-II", compute)
    public = {n for n in dir(v) if not n.startswith("_")}
    assert public == {"evaluate", "history", "logical_evaluations", "method",
                      "stage"}, public
    with pytest.raises(AttributeError):
        v.cache                       # no handle on the shared store
    with pytest.raises(AttributeError):
        v.anything_else = 1           # __slots__, so no smuggling state through it


def test_cache_hit_returns_the_same_result_not_a_recomputation(cache) -> None:
    compute, calls = _counter()
    a, b = cache.view("A", compute), cache.view("B", compute)
    first = a.evaluate(CFG_A)
    second = b.evaluate(CFG_A)
    assert first == second
    assert len(calls) == 1


# ----------------------------------------------------------------- persistence

def test_the_cache_survives_a_restart(tmp_path) -> None:
    compute, calls = _counter()
    c1 = EvaluationCache(tmp_path / "e.sqlite", dataset="magic", split_id="rep_00", seed=1)
    c1.view("A", compute).evaluate(CFG_A)
    c1.close()

    c2 = EvaluationCache(tmp_path / "e.sqlite", dataset="magic", split_id="rep_00", seed=1)
    r = c2.view("B", compute).evaluate(CFG_A)
    assert len(calls) == 1, "a resumed replication must not recompute a completed fit"
    assert r["leaves"] == 101.0
    assert c2.ledger("B").logical == 1, "and the resumed method is still charged"
    c2.close()


def test_a_different_replication_does_not_reuse_another_replications_fits(tmp_path) -> None:
    compute, calls = _counter()
    for rep in ("rep_00", "rep_01"):
        c = EvaluationCache(tmp_path / f"{rep}.sqlite", dataset="magic",
                            split_id=rep, seed=1)
        c.view("A", compute).evaluate(CFG_A)
        c.close()
    assert len(calls) == 2, "replications have different splits, so different evaluations"


def test_the_logical_ledger_survives_a_resume(tmp_path) -> None:
    """The scientific budget must not be lost when a unit resumes from checkpoints.

    Held only in memory, a resumed unit reported zero logical evaluations beside
    thousands of physical fits, which is exactly backwards: the logical figure is
    the one that enters a fairness comparison.
    """
    compute, calls = _counter()
    path = tmp_path / "e.sqlite"

    c1 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c1.view("NBI-S", compute).evaluate(CFG_A)
    c1.view("NBI-S", compute).evaluate(CFG_B)
    c1.view("WS-S", compute).evaluate(CFG_A)          # a hit, still charged
    before = c1.accounting()
    c1.close()

    c2 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    after = c2.accounting()
    c2.close()

    assert after["logical_evaluations_total"] == before["logical_evaluations_total"] == 3
    assert after["unique_physical_fits"] == before["unique_physical_fits"] == 2
    per = {m["method"]: m for m in after["per_method"]}
    assert per["NBI-S"]["logical_evaluations"] == 2
    assert per["WS-S"]["logical_evaluations"] == 1
    assert per["WS-S"]["served_from_cache"] == 1
    assert len(calls) == 2


def test_re_entering_a_stage_replaces_its_charge_rather_than_adding_to_it(tmp_path) -> None:
    """Re-entry means the stage is being run again from the start.

    This test previously asserted the opposite -- that a resumed cache ADDS to the
    previous count -- which is exactly the double-charge. Stage completion is
    checkpointed at stage granularity, so the runner re-enters a stage only when it
    did not finish; everything the killed attempt charged is superseded, not
    accumulated. A stage that DID complete is never re-entered, and
    ``test_the_logical_ledger_survives_a_resume`` covers that case: there the
    resumed process opens no view and the original count stands.
    """
    compute, _ = _counter()
    path = tmp_path / "e.sqlite"
    c1 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c1.view("A", compute, stage="direct_search").evaluate(CFG_A)
    c1.close()

    c2 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c2.view("A", compute, stage="direct_search").evaluate(CFG_A)
    assert c2.ledger("A").logical == 1, "a re-entered stage must not double-charge"
    c2.close()

    # The superseded attempt is retained for audit, not deleted.
    import sqlite3
    db = sqlite3.connect(str(path))
    assert db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 2
    assert {a for (a,) in db.execute("SELECT DISTINCT attempt FROM requests")} == {1, 2}
    db.close()


def test_a_resumed_process_still_charges_work_it_has_not_done_before(tmp_path) -> None:
    """Rewinding a re-entered stage must not suppress genuinely new requests."""
    compute, _ = _counter()
    path = tmp_path / "e.sqlite"
    c1 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c1.view("A", compute, stage="direct_search").evaluate(CFG_A)
    c1.close()

    c2 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    v = c2.view("A", compute, stage="direct_search")
    v.evaluate(CFG_A)
    v.evaluate(CFG_B)                                   # new work in the new attempt
    assert c2.ledger("A").logical == 2
    c2.close()


def test_a_resumed_unit_reports_the_same_budget_as_an_uninterrupted_one(tmp_path) -> None:
    """The invariant that actually matters, stated directly.

    The logical ledger is the scientific budget and the only figure entering a
    fairness comparison. Before the attempt column, killing a process part-way
    through one stage and resuming took that stage's charge from 386 to 772 while
    the physical fit count stayed correct -- so the corruption was invisible in the
    cache statistics, and it was asymmetric across the comparison because it landed
    only on whichever method happened to be interrupted.
    """
    compute, _ = _counter()
    configs = [{**CFG_A, "n_estimators": 50 + 10 * i} for i in range(12)]

    # (a) uninterrupted
    a_path = tmp_path / "uninterrupted.sqlite"
    ca = EvaluationCache(a_path, dataset="magic", split_id="rep_00", seed=1)
    va = ca.view("grid", compute, stage="direct_search")
    for cfg in configs:
        va.evaluate(cfg)
    uninterrupted = ca.accounting()
    ca.close()

    # (b) killed after 5 of 12, then resumed and re-run from the start
    b_path = tmp_path / "interrupted.sqlite"
    cb = EvaluationCache(b_path, dataset="magic", split_id="rep_00", seed=1)
    vb = cb.view("grid", compute, stage="direct_search")
    for cfg in configs[:5]:
        vb.evaluate(cfg)
    cb.close()                                           # the kill

    cb2 = EvaluationCache(b_path, dataset="magic", split_id="rep_00", seed=1)
    vb2 = cb2.view("grid", compute, stage="direct_search")
    for cfg in configs:                                  # the stage re-runs entire
        vb2.evaluate(cfg)
    resumed = cb2.accounting()
    cb2.close()

    assert resumed["logical_evaluations_total"] == uninterrupted["logical_evaluations_total"] == 12
    assert resumed["by_stage"]["direct_search"]["logical"] == 12
    per_r = {m["method"]: m for m in resumed["per_method"]}
    per_u = {m["method"]: m for m in uninterrupted["per_method"]}
    assert per_r["grid"]["logical_evaluations"] == per_u["grid"]["logical_evaluations"] == 12
    assert per_r["grid"]["unique_logical_evaluations"] == 12


def test_two_interruptions_still_charge_once(tmp_path) -> None:
    compute, _ = _counter()
    path = tmp_path / "e.sqlite"
    configs = [{**CFG_A, "n_estimators": 50 + 10 * i} for i in range(6)]
    for stop in (2, 4):
        c = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
        v = c.view("random", compute, stage="direct_search")
        for cfg in configs[:stop]:
            v.evaluate(cfg)
        c.close()
    c = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    v = c.view("random", compute, stage="direct_search")
    for cfg in configs:
        v.evaluate(cfg)
    assert c.ledger("random").logical == 6
    c.close()


def test_an_interruption_does_not_disturb_another_methods_ledger(tmp_path) -> None:
    """Re-entering one stage must rewind that stage only."""
    compute, _ = _counter()
    path = tmp_path / "e.sqlite"
    c1 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c1.view("grid", compute, stage="direct_search").evaluate(CFG_A)
    c1.view("nbi_s_revalidation", compute, stage="candidate_validation").evaluate(CFG_B)
    c1.close()

    c2 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c2.view("grid", compute, stage="direct_search").evaluate(CFG_A)   # re-entered
    acc = c2.accounting()
    per = {m["method"]: m for m in acc["per_method"]}
    assert per["grid"]["logical_evaluations"] == 1
    assert per["nbi_s_revalidation"]["logical_evaluations"] == 1, \
        "an untouched method's ledger was disturbed by another stage's re-entry"
    c2.close()


def test_the_ledger_is_reported_per_method_and_per_stage(tmp_path) -> None:
    """The budget table must break down by stage, not only by method."""
    compute, _ = _counter()
    cache = EvaluationCache(tmp_path / "e.sqlite", dataset="magic",
                            split_id="rep_00", seed=1)
    cache.view("NBI-R", compute, stage="anchor").evaluate(CFG_A)
    cache.view("NBI-R", compute, stage="anchor").evaluate(CFG_B)
    cache.view("NBI-R", compute, stage="candidate_validation").evaluate(CFG_A)
    acc = cache.accounting()
    per = {m["method"]: m for m in acc["per_method"]}["NBI-R"]
    assert per["logical_evaluations"] == 3
    assert per["by_stage"]["anchor"]["logical"] == 2
    assert per["by_stage"]["candidate_validation"]["logical"] == 1
    assert acc["by_stage"]["anchor"]["logical"] == 2
    cache.close()


def test_the_stage_ledger_survives_a_resume(tmp_path) -> None:
    compute, _ = _counter()
    path = tmp_path / "e.sqlite"
    c1 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    c1.view("WS-S", compute, stage="candidate_validation").evaluate(CFG_A)
    c1.view("WS-S", compute, stage="candidate_validation").evaluate(CFG_B)
    before = c1.accounting()
    c1.close()
    c2 = EvaluationCache(path, dataset="magic", split_id="rep_00", seed=1)
    assert c2.accounting()["by_stage"] == before["by_stage"]
    c2.close()


def test_twenty_collapsing_candidates_cost_twenty_logical_and_one_physical(tmp_path) -> None:
    """The exact case the review named: no method gains budget from rounding collapse."""
    calls: list[dict] = []

    def compute(cfg):
        calls.append(cfg)
        return {"leaves": 1.0}

    cache = EvaluationCache(tmp_path / "e.sqlite", dataset="magic",
                            split_id="rep_00", seed=1)
    for method in ("HISTORICAL-WS", "WS-S", "NBI-S", "NBI-R"):
        v = cache.view(method, compute, stage="candidate_validation")
        for _ in range(20):
            v.evaluate(CFG_A)                # twenty continuous solutions, one learner
        led = cache.ledger(method)
        assert led.by_stage["candidate_validation"]["logical"] == 20, method
    assert len(calls) == 1, "one physical fit for all eighty requests"
    assert cache.physical_fits() == 1
    assert cache.accounting()["logical_evaluations_total"] == 80
    cache.close()
