"""Figures published in the protocol documents must match what the code computes.

`protocol/budget_accounting.md` carried a campaign total of 285,120 evaluations and
111 hours long after the anchor-injection control, the as-run historical arm, the
unmatched NSGA-II run and the Stage B throughput recalibration had changed all
three numbers. Nothing detected it, because a figure typed into a Markdown table
has no relationship to the code that will actually spend the budget.

Every number asserted here is derived from ``campaign_budget()`` and searched for
in the document, so the document cannot drift from the runner's own registry.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from doe_xgb.campaign.runner import (DATASETS, N_REPLICATIONS, campaign_budget,
                                     logical_budget_plan, unit_budget)

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
BUDGET_DOC = PAPER / "protocol" / "budget_accounting.md"


def _text() -> str:
    # Thousands separators are how the document writes them; normalize so a search
    # for 395520 finds "395,520".
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", BUDGET_DOC.read_text())


@pytest.mark.parametrize("key", ["campaign_total_logical",
                                 "campaign_audit_only_logical",
                                 "campaign_solution_producing_logical",
                                 "total_logical",
                                 "audit_only_logical",
                                 "solution_producing_logical",
                                 "unmatched_nsga2_logical"])
def test_published_totals_match_the_registry(key):
    value = campaign_budget(2)[key]
    assert str(value) in _text(), (
        f"budget_accounting.md does not state {key} = {value}; the document has "
        f"drifted from campaign_budget()")


def test_published_per_arm_standalone_costs_match():
    text = _text()
    for arm, cost in logical_budget_plan(2)["B_total_solution_per_arm"].items():
        assert arm in text, f"{arm} is never named in budget_accounting.md"
        assert str(cost) in text, f"{arm}'s standalone cost {cost} is not published"


def test_both_historical_arms_appear_as_separate_rows():
    """The defect: one ambiguous row at 108 for two arms costing 108 and 186."""
    text = _text()
    assert "HISTORICAL-WS-asrun" in text
    # The shared-specification arm must be named somewhere NOT as part of the
    # as-run identifier.
    assert re.search(r"HISTORICAL-WS(?!-asrun)", text), \
        "only the as-run arm is named; the shared-specification arm is missing"


def test_the_superseded_projection_is_marked_as_superseded():
    """Retained for visibility, but it must not read as current."""
    text = _text()
    if "285120" in text:
        window = text[max(0, text.index("285120") - 600): text.index("285120") + 200]
        assert re.search(r"supersed|predated|earlier|no longer|retained", window, re.I), \
            "the old 285,120 projection appears without being marked as superseded"


def test_unit_budget_totals_are_internally_consistent():
    u = unit_budget(2)
    assert u["total_logical"] == u["solution_producing_logical"] + u["audit_only_logical"]
    assert u["total_logical"] == sum(u["by_stage"].values())


def test_the_reconciliation_check_can_actually_fail():
    """The old `reconciles` flag was `X == X` over the same table and could not fail.

    It was published by the dry run as a pre-launch proof while the runner charged
    840 evaluations per campaign that no ledger recorded.
    """
    from doe_xgb.campaign.runner import method_stage_ledger, reconcile_unit_accounting

    def accounting_from(table):
        per = {}
        for method, stages in table.items():
            per.setdefault(method, {"method": method, "by_stage": {}})
            for stage, n in stages.items():
                per[method]["by_stage"][stage] = {"logical": n}
        return {"per_method": list(per.values())}

    table = method_stage_ledger(2)

    # A unit that charged exactly what is declared reconciles.
    assert reconcile_unit_accounting(accounting_from(table))["reconciles"]

    # A method charged but never declared -- the shape of the holdout-confirmation
    # defect, where real evaluations were spent by something the registry did not
    # know about. (holdout_confirmation is now declared, so this uses a stray name.)
    extra = accounting_from(table)
    extra["per_method"].append(
        {"method": "undeclared_side_stage", "by_stage": {"holdout_audit": {"logical": 5}}})
    r = reconcile_unit_accounting(extra)
    assert not r["reconciles"]
    assert r["difference"] == 5
    assert r["charged_but_never_declared"]

    # A stage declared but never charged -- the anchor-injection bypass.
    missing = {k: v for k, v in table.items() if k != "anchor_injection_control"}
    r = reconcile_unit_accounting(accounting_from(missing))
    assert not r["reconciles"]
    assert r["declared_but_never_charged"]

    # A stage charged the wrong amount.
    wrong = {k: dict(v) for k, v in table.items()}
    wrong["design"] = {"design": 87}
    r = reconcile_unit_accounting(accounting_from(wrong))
    assert not r["reconciles"]
    assert r["mismatched_stages"]


def test_campaign_total_is_the_product_it_claims_to_be():
    cb = campaign_budget(2)
    units = len(DATASETS) * N_REPLICATIONS
    assert cb["units"] == units
    assert (cb["campaign_total_logical"]
            == unit_budget(2)["total_logical"] * units + cb["unmatched_nsga2_logical"])
    assert (cb["campaign_total_logical"]
            == cb["campaign_solution_producing_logical"] + cb["campaign_audit_only_logical"])


def test_the_open_item_register_does_not_record_the_panel_as_closed():
    """It was closed on a screening verdict that has since been withdrawn.

    A freeze cannot ship with the panel decision recorded as settled by evidence the
    same repository marks superseded three documents away.
    """
    import pathlib
    proto = (pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
             / "protocol" / "EXPERIMENT_PROTOCOL.md").read_text()
    row = next((l for l in proto.splitlines()
                if l.startswith("| 7 ") and "panel" in l.lower()), None)
    assert row is not None, "the dataset-panel register row is gone"
    assert "closed" not in row.lower() or "REOPENED" in row, (
        f"the panel is still recorded as closed: {row[:160]}")
    assert "blocks the v3 freeze" in row


@pytest.mark.parametrize("doc", ["STAGE_B_THROUGHPUT.md", "THREE_OBJECTIVE_DECISION.md"])
def test_every_document_publishing_a_campaign_total_publishes_the_current_one(doc):
    """A stale subtotal is as wrong as a stale total, and harder to notice.

    STAGE_B_THROUGHPUT.md published 380,160 campaign evaluations against the
    registry's 380,760, and 380,160 + 15,360 reproduces the superseded 395,520 as a
    hidden subtotal -- so the document silently disagreed with the planner while
    every headline figure in it looked right.
    """
    import pathlib
    import re
    from doe_xgb.campaign.runner import campaign_budget, unit_budget
    text = (pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
            / doc).read_text()
    flat = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    cb, ub = campaign_budget(2), unit_budget(2)
    assert str(cb["campaign_total_logical"]) in flat, f"{doc} does not state the current total"
    # the superseded totals must not appear as current figures
    for stale in ("395520", "380160"):
        if stale in flat:
            i = flat.index(stale)
            window = flat[max(0, i - 400):i + 200]
            assert re.search(r"supersed|earlier|as first published|withdrawn", window, re.I), \
                f"{doc} states the superseded {stale} without marking it as superseded"


def test_the_in_unit_subtotal_and_the_unmatched_run_add_to_the_total():
    """The arithmetic that the stale subtotal broke."""
    from doe_xgb.campaign.runner import campaign_budget, unit_budget
    cb, ub = campaign_budget(2), unit_budget(2)
    assert ub["total_logical"] * cb["units"] + cb["unmatched_nsga2_logical"] \
        == cb["campaign_total_logical"]
