"""Every screening figure in print must equal the artifact it claims to come from.

This class of defect has now occurred four times: an audit artifact is regenerated
and one or more documents keep publishing the superseded numbers, each citing the
artifact that now contradicts them. The worst instance was amendment 23's own
"what changed numerically" row, whose entire function was to certify that refitting
the factor models did not disturb the panel -- and which certified it using the
PRE-refit figures. MAGIC's non-dominated set had gone from 8 points to 7.

Prose cannot be trusted to track an artifact by discipline, so it is checked. Every
document that publishes a per-dataset screening figure is parsed here and compared
against audits/final_panel_screening.json.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
SCREENING = PAPER / "audits" / "final_panel_screening.json"

# Documents that publish per-dataset screening figures. Adding a document here is
# the deliberate act of putting it under this guard.
PUBLISHING_DOCS = [
    PAPER / "protocol" / "EXPERIMENT_PROTOCOL.md",
    PAPER / "protocol" / "dataset_selection.md",
    PAPER / "PROTOCOL_AMENDMENTS.md",
]

DISPLAY = {"magic": "MAGIC", "spambase": "Spambase",
           "adult": "Adult", "bank_marketing": "Bank Marketing"}


def _screening() -> dict:
    return {r["dataset"]: r for r in json.loads(SCREENING.read_text())["datasets"]}


@pytest.mark.parametrize("doc", PUBLISHING_DOCS, ids=lambda p: p.name)
def test_every_published_front_size_matches_the_artifact(doc):
    """A row naming a dataset and a front size must state the artifact's front size."""
    data = _screening()
    for line in doc.read_text().splitlines():
        if not line.startswith("|"):
            continue
        m = re.search(r"(\d+)\s+(?:front )?points?\b", line)
        if not m:
            continue
        named = [k for k, disp in DISPLAY.items()
                 if re.search(rf"\|\s*\**{re.escape(disp)}\**\s*\|", line)]
        if len(named) != 1:
            continue
        want = data[named[0]]["criterion_2_front_size"]
        assert int(m.group(1)) == want, (
            f"{doc.name}: row for {DISPLAY[named[0]]} publishes "
            f"{m.group(1)} front points; the artifact says {want}")


@pytest.mark.parametrize("doc", PUBLISHING_DOCS, ids=lambda p: p.name)
def test_every_published_curvature_matches_the_artifact(doc):
    data = _screening()
    for line in doc.read_text().splitlines():
        if not line.startswith("|") or "curvature" not in line.lower():
            continue
        named = [k for k, disp in DISPLAY.items()
                 if re.search(rf"\|\s*\**{re.escape(disp)}\**\s*\|", line)]
        if len(named) != 1:
            continue
        want = data[named[0]]["criterion_2_curvature"]
        if want is None:
            assert "undefined" in line.lower(), (
                f"{doc.name}: {DISPLAY[named[0]]} has no defined curvature but the "
                f"row does not say so")
            continue
        found = [float(x) for x in re.findall(r"0\.\d{3,4}", line)]
        assert any(abs(v - want) < 5e-4 for v in found), (
            f"{doc.name}: row for {DISPLAY[named[0]]} publishes curvature {found}; "
            f"the artifact says {want:.4f}")


@pytest.mark.parametrize("dataset", list(DISPLAY))
def test_the_superseded_pre_refit_figures_appear_nowhere_as_current(dataset):
    """The exact values amendment 23 moved, which three documents kept printing."""
    stale = {"magic": ["8 front points", "8 points, 0.166", "0.1662"],
             "adult": ["curvature 0.211", "6 points, 0.211", "0.2108"],
             "bank_marketing": ["curvature 0.214", "6 points, 0.214", "0.2140"],
             "spambase": []}[dataset]
    for doc in PUBLISHING_DOCS:
        text = doc.read_text()
        for token in stale:
            if token not in text:
                continue
            i = text.index(token)
            window = text[max(0, i - 500):i + 200]
            assert re.search(r"earlier version|supersed|pre-refit|withdrawn|An earlier",
                             window, re.I), (
                f"{doc.name} states the superseded {token!r} without marking it as "
                f"superseded")


def test_the_amendment_certifying_the_refit_states_the_moved_figures():
    """Amendment 23's certification row must reflect what actually moved."""
    text = (PAPER / "PROTOCOL_AMENDMENTS.md").read_text()
    data = _screening()
    row = next(l for l in text.splitlines() if "What changed numerically" in l)
    assert str(data["magic"]["criterion_2_front_size"]) in row
    assert "robust to that movement" in row, (
        "the amendment claims the refit changed nothing rather than that the roles "
        "were robust to what it changed")


def test_no_code_module_hardcodes_a_screening_figure():
    """Hardcoded figures in docstrings were silently falsified by the refit."""
    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "doe_xgb" / "campaign"
    for path in src.glob("*.py"):
        text = path.read_text()
        assert "166-point" not in text, (
            f"{path.name} still describes the withdrawn 166-point reference set")
        assert "0.14 to 0.17" not in text, (
            f"{path.name} hardcodes a measured percentile that the refit moved; "
            f"point at audits/latent_conflict_stability.json instead")
