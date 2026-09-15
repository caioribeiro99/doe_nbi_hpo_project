"""The frozen power analysis in print must equal the audit JSON it came from.

STAGE_B_STATISTICAL_SENSITIVITY.md is the study's pre-campaign statement of what
R = 30 can resolve, and it is the document a reader checks a published effect
against. It had drifted from its own audit artifact in two ways at once: it
published a MAGIC detectable difference of 0.0197 where the JSON carried 0.01734,
and it stated the Nadeau-Bengio inflation as "3.31x at rho = 0.25" -- the
equicorrelation variant, which corresponds to a 75/25 split rather than this
protocol's 80/20, where the correct figure is sqrt(8.5) = 2.9155.

Underneath both, the JSON itself had been generated from the superseded factor
algebra, because the sensitivity script imported pilot_stage_a_screening.py rather
than the campaign's own factor model. Only the arithmetic had been corrected; the
input had not.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
DOC = PAPER / "STAGE_B_STATISTICAL_SENSITIVITY.md"
AUDIT = PAPER / "audits" / "stage_b_sensitivity.json"
SCRIPT = PAPER / "scripts" / "stage_b_statistical_sensitivity.py"

DATASETS = ("magic", "spambase", "adult", "bank_marketing")
INDICATORS = ("hv_ratio", "igd_plus")


def _audit() -> dict:
    return json.loads(AUDIT.read_text())


def _doc() -> str:
    return DOC.read_text()


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("indicator", INDICATORS)
def test_every_published_figure_is_in_the_audit(dataset, indicator):
    b = _audit()["datasets"][dataset][indicator]
    text = _doc()
    for key in ("paired_sd_proxy",
                "minimum_detectable_paired_difference_80pct",
                "expected_ci95_half_width",
                "mde_corrected_test_train_0.25"):
        assert f"{b[key]:.4f}" in text, (
            f"{dataset}/{indicator} {key} = {b[key]:.4f} is not published in "
            f"STAGE_B_STATISTICAL_SENSITIVITY.md")


def test_the_superseded_magic_figure_is_gone():
    """0.0197 is the 75/25 equicorrelation variant and must not appear."""
    assert "0.0197" not in _doc()


def test_the_inflation_is_stated_as_sqrt_8_5():
    text = _doc()
    assert "2.9155" in text
    assert not re.search(r"inflates by \*\*3\.31", text), \
        "the superseded 3.31x inflation is stated as current"


def test_the_correction_is_labelled_by_ratio_not_by_rho():
    """rho and n_test/n_train are equal only when rho = n_test/(n_test+n_train)."""
    text = _doc()
    assert "n_test/n_train = 0.25" in text
    header = next(l for l in text.splitlines() if l.startswith("| Dataset | indicator"))
    assert "n_test/n_train" in header, \
        f"the table column still labels the correction by rho: {header}"


def test_the_draw_count_in_print_matches_the_audit():
    draws = _audit()["draws"]
    m = re.search(r"(\d+) resampling draws per dataset", _doc())
    assert m and int(m.group(1)) == draws


def test_every_audit_inflation_factor_is_the_nadeau_bengio_value():
    """SE_corr/SE_naive = sqrt((1/R + n_test/n_train) / (1/R)), derived here."""
    import math
    a = _audit()
    R = a["replications"]
    for dataset in DATASETS:
        for indicator in INDICATORS:
            b = a["datasets"][dataset][indicator]
            for key, ratio in (("se_inflation_test_train_0.25", 0.25),
                               ("se_inflation_test_train_0.1111", 0.1111),
                               ("se_inflation_test_train_0.4286", 0.4286)):
                expected = math.sqrt((1.0 / R + ratio) / (1.0 / R))
                assert abs(b[key] - expected) < 5e-4, (
                    f"{dataset}/{indicator} {key}: {b[key]} vs derived {expected}")
            assert abs(b["se_inflation_test_train_0.25"] - math.sqrt(8.5)) < 1e-3


def test_the_sensitivity_uses_the_campaigns_own_factor_model():
    """Its input must be the algebra the campaign uses, not the superseded pilot code."""
    src = SCRIPT.read_text()
    assert "load_reference_factor_model" in src
    assert "pilot.FactorModel" not in src, \
        "the power analysis is computed on the superseded pilot factor algebra"
    assert "pilot_stage_a_screening.py" not in src.split("# The objective definition")[-1] \
        or "superseded" in src


# ---------------------------------------------------------------------------
# the NARRATIVE too, not only the table
#
# The table was regenerated from the audit and the prose beneath it was not, so
# four narrative lines went on publishing the superseded-algebra ranges (0.6-2.4
# and 2.0-8.0 percentage points) directly under a corrected table, and two of them
# still labelled the correction by rho. A document-conformance test that checks
# only the machine-shaped part of a document is a test that the prose can drift
# past.
# ---------------------------------------------------------------------------

def test_the_narrative_ranges_match_the_audit():
    a = _audit()["datasets"]
    hv = [b["hv_ratio"] for b in a.values()]
    lo_u = min(v["minimum_detectable_paired_difference_80pct"] for v in hv)
    hi_u = max(v["minimum_detectable_paired_difference_80pct"] for v in hv)
    lo_c = min(v["mde_corrected_test_train_0.25"] for v in hv)
    hi_c = max(v["mde_corrected_test_train_0.25"] for v in hv)
    text = _doc()
    for pp in (lo_u * 100, hi_u * 100, lo_c * 100, hi_c * 100):
        assert f"{pp:.1f}" in text, f"the narrative does not state {pp:.1f} percentage points"
    for raw in (lo_c, hi_c):
        assert f"{raw:.3f}" in text, f"the narrative does not state {raw:.3f}"


def test_the_superseded_narrative_ranges_are_gone():
    """The exact strings the prose published from the withdrawn algebra."""
    text = _doc()
    for stale in ("0.6 and 2.4", "2.0 and 8.0", "0.020 to 0.080", "0.007 to 0.026"):
        head = text.split("an earlier version")[0] if "an earlier version" in text else text
        assert stale not in head, f"the narrative still asserts the superseded {stale!r}"


def test_no_line_labels_the_correction_by_rho_except_the_withdrawal_note():
    """rho and n_test/n_train coincide only at a 75/25 split, which this is not."""
    import re
    for i, line in enumerate(_doc().splitlines(), 1):
        if re.search(r"(rho|ρ)\s*=\s*0\.25", line):
            assert "equicorrelation" in line, (
                f"line {i} labels the Nadeau-Bengio correction by rho = 0.25 without "
                f"flagging it as the superseded equicorrelation reading: {line.strip()[:120]}")
