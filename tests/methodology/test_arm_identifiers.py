"""Every executed entity carries a globally unique identifier.

The defect this pins down: the bit-faithful historical reproduction and the
shared-specification normalization control both labelled themselves
``HISTORICAL-WS``. The runner kept them apart by checkpoint stage key, so nothing
failed, but every candidate record in both carried the same ``arm`` field. The
protocol says of exactly these two that "the two are never mixed in one table" --
and with one identifier between them, mixing them is undetectable rather than
forbidden. ``protocol/EXPERIMENT_PROTOCOL.md`` and
``protocol/OBJECTIVE_COUNT_DECISION.md`` had named the as-run arm
``HISTORICAL-WS-asrun`` all along; the code was out of conformance with them.

Identifier hygiene is not cosmetic here. Every downstream aggregation the analysis
will perform -- per-arm indicator tables, provenance exports, the method registry
in the freeze report -- groups by these strings.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from doe_xgb.campaign.runner import STAGES, SINGLE_OBJECTIVE, method_stage_ledger

ARMS = pathlib.Path(__file__).resolve().parents[2] / "src" / "doe_xgb" / "campaign" / "arms.py"

# Every scientific entity the campaign executes, by exact identifier. This list is
# the registry: adding an arm without adding it here fails the conformance test
# below, and adding it here without executing it fails the stage test.
REGISTRY = {
    # surrogate-assisted Pareto-construction arms
    "HISTORICAL-WS-asrun": "historical_ws_asrun",
    "HISTORICAL-WS": "historical_ws",
    "WS-S": "ws_s",
    "NBI-S": "nbi_s",
    "NBI-R": "nbi_r",
}


def test_registry_identifiers_are_unique():
    assert len(set(REGISTRY)) == len(REGISTRY)
    assert len(set(REGISTRY.values())) == len(REGISTRY)


def test_every_registry_arm_has_a_runner_stage():
    for arm, stage in REGISTRY.items():
        assert stage in STAGES, f"{arm} claims stage {stage!r}, which the runner never runs"


def test_every_registry_arm_is_revalidated():
    ledger = method_stage_ledger(2)
    for arm, stage in REGISTRY.items():
        key = f"{stage}_revalidation"
        assert key in ledger, f"{arm} is never revalidated on the real learner"
        assert ledger[key]["candidate_validation"] > 0


def test_the_two_historical_entities_are_distinct():
    """The reproduction and the normalization control must never share a label."""
    assert REGISTRY["HISTORICAL-WS-asrun"] != REGISTRY["HISTORICAL-WS"]
    assert "HISTORICAL-WS-asrun" != "HISTORICAL-WS"


def _string_literals_passed_as_arm_labels() -> set[str]:
    """Labels that ``arms.py`` stamps onto ArmRun objects and candidate records."""
    tree = ast.parse(ARMS.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in {"ArmRun", "_record"}:
            continue
        if node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            out.add(node.args[0].value)
    return out


def test_hardcoded_arm_labels_are_registered():
    """A literal label in arms.py must be one the registry knows about."""
    literals = _string_literals_passed_as_arm_labels()
    assert literals, "found no hard-coded arm labels; the AST walk is not working"
    unknown = literals - set(REGISTRY)
    assert not unknown, (
        f"arms.py stamps unregistered identifier(s) {sorted(unknown)} onto results. "
        f"Registered: {sorted(REGISTRY)}")


def test_the_asrun_reproduction_is_labelled_asrun():
    """Regression: run_historical_ws once labelled its output 'HISTORICAL-WS'."""
    src = ARMS.read_text()
    body = src[src.index("def run_historical_ws"):]
    body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
    assert '"HISTORICAL-WS-asrun"' in body
    assert '_record("HISTORICAL-WS"' not in body
    assert 'ArmRun("HISTORICAL-WS"' not in body


def test_single_objective_comparators_are_not_arms():
    """They are excluded from front indicators and must never appear as arms."""
    assert not (set(SINGLE_OBJECTIVE) & set(REGISTRY.values()))
    for m in SINGLE_OBJECTIVE:
        assert m not in STAGES
