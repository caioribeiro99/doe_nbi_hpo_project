"""The two author-level specification decisions, pinned so they cannot drift.

DECISION 1 — screening. Criterion 1's latent operationalization is withdrawn and
replaced by the CANONICAL raw-response measurement; criterion 2 is unchanged and
Spambase fails it, making it a pre-specified boundary control rather than a
replacement case.

DECISION 2 — the primary reference is the method-independent CORE; the augmented
reference is a mandatory sensitivity and never an alternate primary test.

Both were taken before any confirmatory arm-level result existed. These tests bind
the code, the committed screening artifact and the protocol prose to each other, so
none of the three can move without the others.
"""
from __future__ import annotations

import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.factor_model import load_reference_factor_model, raw_conflict
from doe_xgb.campaign.runner import (BOUNDARY_CONTROLS, DATASET_ROLES, DATASETS,
                                     N_REPLICATIONS, PRIMARY_GEOMETRY_PANEL,
                                     PRIMARY_INDICATOR, PRIMARY_REFERENCE)

REPO = pathlib.Path(__file__).resolve().parents[2]
PAPER = REPO / "papers" / "xgboost_hpo_vrfnbi"
SCREENING = PAPER / "audits" / "final_panel_screening.json"
SCRIPT = PAPER / "scripts" / "final_panel_screening.py"
PROTO = PAPER / "protocol" / "EXPERIMENT_PROTOCOL.md"
PILOT = PAPER / "audits" / "pilot_stage_a"


def _screening() -> dict:
    return json.loads(SCREENING.read_text())


# ---------------------------------------------------------------- decision 1A

def test_criterion_1_uses_the_canonical_raw_conflict_implementation():
    """It must be IMPORTED, not reimplemented. A reimplementation got it wrong once."""
    tree = ast.parse(SCRIPT.read_text())
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert "raw_conflict" in imported, \
        "final_panel_screening.py does not import the canonical raw_conflict"
    src = SCRIPT.read_text()
    assert "def raw_conflict" not in src, "raw_conflict is reimplemented locally"
    assert _screening()["datasets"][0]["detail"]["criterion_1"]["implementation"] == \
        "doe_xgb.campaign.factor_model.raw_conflict"


@pytest.mark.parametrize("dataset", DATASETS)
def test_criterion_1_values_are_reproducible_from_the_canonical_path(dataset):
    """Recomputed here, independently of the artifact."""
    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    want = float(raw_conflict(design))
    row = next(r for r in _screening()["datasets"] if r["dataset"] == dataset)
    assert abs(row["criterion_1_raw_conflict_value"] - want) < 5e-4


def test_criterion_1_measures_against_the_raw_leaf_count_not_the_latent_cost():
    """The exact defect a previous reimplementation introduced."""
    from doe_xgb.campaign import factor_model
    src = pathlib.Path(factor_model.__file__).read_text()
    body = src[src.index("def raw_conflict"):]
    body = body[:body.index("\n\n\n")] if "\n\n\n" in body else body
    assert 'names.index("Leaves_Mean")' in body, \
        "raw_conflict no longer correlates against the raw Leaves_Mean response"
    assert "transform(" not in body, \
        "raw_conflict passes through the factor stage; it must not"


@pytest.mark.parametrize("dataset", DATASETS)
def test_criterion_1_passes_on_every_dataset(dataset):
    row = next(r for r in _screening()["datasets"] if r["dataset"] == dataset)
    assert row["criterion_1_pass"] is True
    assert row["criterion_1_raw_conflict_value"] < 0
    assert row["detail"]["criterion_1"]["ci_excludes_zero"] is True


def test_the_latent_operationalization_is_not_computed_for_criterion_1():
    """Withdrawn means absent, not merely unused."""
    a = _screening()
    assert "latent" in a["criterion_1_operationalization"].lower()
    assert "WITHDRAWN" in a["criterion_1_operationalization"]
    for row in a["datasets"]:
        assert "latent" not in json.dumps(row["detail"]["criterion_1"]).lower()


# ---------------------------------------------------------------- decision 1B

@pytest.mark.parametrize("dataset,role", [
    ("magic", "primary_geometry_confirmatory"),
    ("adult", "primary_geometry_confirmatory"),
    ("bank_marketing", "primary_geometry_confirmatory"),
    ("spambase", "boundary_geometry_control"),
])
def test_the_dataset_roles_are_exactly_as_screened(dataset, role):
    assert DATASET_ROLES[dataset] == role
    row = next(r for r in _screening()["datasets"] if r["dataset"] == dataset)
    assert row["final_role"] == role


def test_spambase_genuinely_fails_criterion_2_under_the_unchanged_rule():
    """Recomputed from the design rows, not read from the artifact."""
    from doe_xgb.reporting import pareto_front
    design = pd.read_csv(PILOT / "spambase_design.csv")
    F = load_reference_factor_model("spambase").objectives(design)
    assert int(pareto_front(F).sum()) == 2, \
        "Spambase's design-row front is no longer two points; the role rule must be rechecked"
    row = next(r for r in _screening()["datasets"] if r["dataset"] == "spambase")
    assert row["criterion_2_pass"] is False
    assert row["criterion_2_curvature"] is None


@pytest.mark.parametrize("dataset", ["magic", "adult", "bank_marketing"])
def test_the_primary_panel_datasets_genuinely_pass_criterion_2(dataset):
    from doe_xgb.reporting import pareto_front
    design = pd.read_csv(PILOT / f"{dataset}_design.csv")
    F = load_reference_factor_model(dataset).objectives(design)
    assert int(pareto_front(F).sum()) >= 3
    row = next(r for r in _screening()["datasets"] if r["dataset"] == dataset)
    assert row["criterion_2_pass"] is True and row["criterion_2_curvature"] > 0


def test_every_dataset_including_the_boundary_control_is_executed():
    """The boundary control is excluded from an inferential family, not from the run."""
    a = _screening()
    assert set(a["all_datasets_executed"]) == set(DATASETS)
    assert "spambase" in a["all_datasets_executed"]
    assert len(DATASETS) * N_REPLICATIONS == 120


def test_the_unit_split_is_ninety_primary_and_thirty_boundary():
    assert len(PRIMARY_GEOMETRY_PANEL) * N_REPLICATIONS == 90
    assert len(BOUNDARY_CONTROLS) * N_REPLICATIONS == 30


def test_no_replacement_dataset_was_selected():
    assert _screening()["replacement_dataset_selected"] is False


def test_the_roles_partition_the_panel():
    assert not set(PRIMARY_GEOMETRY_PANEL) & set(BOUNDARY_CONTROLS)
    assert set(PRIMARY_GEOMETRY_PANEL) | set(BOUNDARY_CONTROLS) == set(DATASETS)


def test_criterion_2_threshold_was_not_weakened():
    """The rule text must still demand a detectable non-linearity."""
    rule = _screening()["datasets"][0]["detail"]["criterion_2"]["rule"]
    assert "detectably non-linear" in rule
    spam = next(r for r in _screening()["datasets"] if r["dataset"] == "spambase")
    assert "fewer than three points" in spam["detail"]["criterion_2"]["why"]


# ---------------------------------------------------------------- decision 2

def test_the_primary_reference_is_the_core():
    assert PRIMARY_REFERENCE == "core"
    assert PRIMARY_INDICATOR == "hv_ratio"


def _proto_flat() -> str:
    """Line breaks are formatting; assertions are about content."""
    import re
    return re.sub(r"\s+", " ", PROTO.read_text())


def test_the_protocol_names_the_core_as_primary_and_augmented_as_sensitivity():
    text = _proto_flat()
    assert "The primary reference is the CORE reference" in text
    assert "mandatory sensitivity" in text
    assert "is a sensitivity and never an alternate primary test" in text


def test_igd_plus_is_not_silently_demoted():
    """It was never co-primary; the protocol must say so rather than imply a demotion."""
    text = _proto_flat()
    assert "IGD⁺ was never co-primary and is not demoted here" in text
    assert "MF17" in text


def test_the_plan_forbids_choosing_a_reference_after_the_fact():
    text = _proto_flat()
    assert "Neither may be chosen after the fact" in text
    assert "disagreement is itself a result" in text


def test_dataset_is_the_generalization_unit_and_there_is_no_pooling():
    text = _proto_flat()
    assert "no pooling" in text.lower()
    assert "90-replication" in text


def test_the_boundary_control_interpretation_is_fixed_in_advance():
    text = _proto_flat()
    assert "is **not** evidence against the geometry mechanism" in text
    assert "after** campaign completion" in text
