"""The direct-baselines methods namespace must contain methods and nothing else.

Found by the pre-freeze smoke at stage eighteen of twenty. The seed ledger was
written into the methods dict as well as at the top level, and both the
augmented-reference and the metrics stage iterate that dict and score every entry
they find. It raised ``KeyError: 'rows'``.

The crash is not the interesting part. Had the stray entry carried a ``rows`` key,
it would have been converted to objective values and folded into the AUGMENTED
REFERENCE -- the set every arm's IGD+ and hypervolume are measured against -- with
no error at all. The reference would have been contaminated by metadata, every
indicator in the paper computed against it, and nothing would have looked wrong.

So this tests the namespace invariant, not the crash.
"""
from __future__ import annotations

import pytest

from doe_xgb.campaign.runner import (SCORED_BASELINES, SINGLE_OBJECTIVE,
                                     MethodologicalFailure, _baseline_methods)


def _payload(methods):
    return {"methods": methods, "budget": 386}


def _full(extra=None):
    m = {name: {"n_rows": 3, "rows": [{}, {}, {}]} for name in SCORED_BASELINES}
    if extra:
        m.update(extra)
    return m


def test_a_complete_namespace_passes():
    out = _baseline_methods(_payload(_full()))
    assert set(out) == set(SCORED_BASELINES)


def test_the_seed_ledger_in_the_methods_namespace_is_rejected():
    """The exact defect: metadata stored beside the methods."""
    stray = {"_seed_ledger": {"grid": {"derived_method_seed": 1}}}
    with pytest.raises(MethodologicalFailure, match="non-method entries"):
        _baseline_methods(_payload(_full(stray)))


def test_a_stray_entry_that_would_have_scored_silently_is_rejected():
    """The dangerous shape: metadata carrying rows, which would NOT have crashed."""
    stray = {"_provenance": {"n_rows": 2, "rows": [{"a": 1.0}, {"a": 2.0}]}}
    with pytest.raises(MethodologicalFailure, match="non-method entries"):
        _baseline_methods(_payload(_full(stray)))


def test_a_missing_comparator_is_rejected():
    """A reference built from a subset of comparators is not the declared reference."""
    partial = _full()
    partial.pop("nsga2")
    with pytest.raises(MethodologicalFailure, match="did not produce"):
        _baseline_methods(_payload(partial))


@pytest.mark.parametrize("name", SCORED_BASELINES)
def test_each_comparator_individually_is_required(name):
    partial = _full()
    partial.pop(name)
    with pytest.raises(MethodologicalFailure):
        _baseline_methods(_payload(partial))


def test_single_objective_comparators_are_registered_baselines():
    """They run as comparators but are excluded from front indicators downstream."""
    assert set(SINGLE_OBJECTIVE).issubset(set(SCORED_BASELINES))


def test_the_registry_has_no_duplicates():
    assert len(set(SCORED_BASELINES)) == len(SCORED_BASELINES)


def test_the_runner_reads_baselines_only_through_the_guard():
    """No read site may bypass the namespace validation."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[2] / "src" / "doe_xgb"
           / "campaign" / "runner.py").read_text()
    assert 'ck.load("direct_baselines")["methods"]' not in src, (
        "a read site bypasses _baseline_methods and would score a polluted namespace")
    assert src.count("_baseline_methods(ck.load(") == 2
