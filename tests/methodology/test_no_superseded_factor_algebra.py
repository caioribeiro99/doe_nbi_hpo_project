"""No script that produces current evidence may use the withdrawn factor algebra.

The Stage-B power analysis -- the study's frozen statement of what R = 30 can
resolve -- was computed by importing pilot_stage_a_screening.py and calling its
FactorModel, which still rotates raw component scores and weights quality axes by
unrotated eigenvalue shares. Only the Nadeau-Bengio arithmetic downstream had been
corrected. Correcting an estimator while leaving its input on withdrawn algebra is
the failure this guards against, and it survived a full verification pass because
nothing connected the two.

The pilot module is NOT banned wholesale: its design-geometry helpers (PARAMS,
BOUNDS, INTS, external_points) are unaffected by the factor defects and are still
the right source for the Stage A design. Only the factor model is off limits.
"""
from __future__ import annotations

import pathlib
import re

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi" / "scripts"

# Scripts allowed to use the superseded FactorModel, each with the reason. Every
# entry is a script whose job is to reproduce a historical artifact exactly.
HISTORICAL = {
    "pilot_stage_a_screening.py": "defines it; it IS the Stage A record",
    "objective_count_evidence.py": "reproduces the superseded objective-count "
                                   "evidence; revalidated by "
                                   "objective_count_revalidation.py",
}

FACTOR_USE = re.compile(r"\bpilot\.FactorModel\b|\bpilot\.apply_transforms\b")


def _live_scripts():
    return sorted(p for p in SCRIPTS.glob("*.py") if p.name not in HISTORICAL)


@pytest.mark.parametrize("path", _live_scripts(), ids=lambda p: p.name)
def test_no_live_script_uses_the_superseded_factor_model(path):
    hits = [f"{path.name}:{i}" for i, line in enumerate(path.read_text().splitlines(), 1)
            if FACTOR_USE.search(line)]
    assert not hits, (
        f"{path.name} computes with the withdrawn factor algebra at {hits}. Use "
        f"doe_xgb.campaign.factor_model, or add the script to HISTORICAL with a reason.")


def test_the_superseded_class_says_so():
    src = (SCRIPTS / "pilot_stage_a_screening.py").read_text()
    head = src[:src.index("class FactorModel")]
    assert "SUPERSEDED" in head[-2000:]
    assert "doe_xgb.campaign.factor_model" in head[-2000:]


def test_the_superseded_evidence_script_says_so():
    src = (SCRIPTS / "objective_count_evidence.py").read_text()
    assert "SUPERSEDED by objective_count_revalidation.py" in src[:2500]


def test_the_stage_b_sensitivity_is_a_live_script_and_is_clean():
    """The specific regression: this script must never go back to the pilot model."""
    src = (SCRIPTS / "stage_b_statistical_sensitivity.py").read_text()
    assert "load_reference_factor_model" in src
    assert not FACTOR_USE.search(src)


def test_the_design_geometry_helpers_remain_available():
    """The ban is on the factor model only; the design helpers are not defective."""
    src = (SCRIPTS / "verify_external_validation_design.py").read_text()
    assert "pilot.external_points" in src, (
        "the external-set verification no longer uses the committed design geometry; "
        "if that is intended this test should be updated deliberately")
