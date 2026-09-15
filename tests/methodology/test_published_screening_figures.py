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


# ---------------------------------------------------------------------------
# PROSE-FORM FIGURES
#
# The two checks above resolve a dataset by requiring its name to occupy a table
# cell of its own. An independent reviewer showed that this certifies coverage it
# does not deliver: they rewrote a live amendment row to "99, 96 and 96 front points
# and curvature 0.999, 0.911, 0.914" -- absurd values, in a document listed in
# PUBLISHING_DOCS -- and the guard passed, because the figures sit in prose INSIDE a
# cell rather than in cells of their own. Two real defects had survived the same way.
#
# These checks read multi-dataset prose positionally: "MAGIC, Adult and Bank
# Marketing ... with A, B and C front points and curvature X, Y, Z" must state the
# artifact's values in that order, or be explicitly marked as a superseded figure.
# ---------------------------------------------------------------------------

_PROSE_ORDER = ("magic", "adult", "bank_marketing")


def _lines_naming_the_primary_panel_in_order(text: str):
    """Lines that name MAGIC, Adult and Bank Marketing in that order, on one line."""
    pat = re.compile(r"MAGIC.{0,80}?Adult.{0,80}?Bank Marketing", re.S)
    for line in text.splitlines():
        if pat.search(line):
            yield line


# THE RULE, and it is deliberately not "the line mentions the word superseded".
#
# A first version of these checks exempted any line containing a marker word. That
# is disarmed by the very notes this repository writes: a row corrected to the
# current figures AND carrying "(an earlier version said X)" contains a marker, so
# the whole line went unguarded — and two adversarial injections of absurd values
# into exactly such a row passed. The marker was doing the opposite of its job.
#
# The rule instead: a line that makes a prose claim about the panel's screening
# figures MUST state the artifact's current values. Superseded values may appear
# ALONGSIDE them, which is how a correction records what it corrected, but never
# INSTEAD of them.


@pytest.mark.parametrize("doc", PUBLISHING_DOCS, ids=lambda p: p.name)
def test_prose_front_sizes_state_the_current_values(doc):
    data = _screening()
    want = [data[d]["criterion_2_front_size"] for d in _PROSE_ORDER]
    wanted = re.compile(rf"{want[0]},\s*{want[1]}\s+and\s+{want[2]}\s+front points")
    for line in _lines_naming_the_primary_panel_in_order(doc.read_text()):
        claims = re.findall(r"(\d+),\s*(\d+)\s+and\s+(\d+)\s+front points", line)
        if not claims:
            continue
        assert wanted.search(line), (
            f"{doc.name}: a prose claim about the panel states front sizes "
            f"{[list(map(int, c)) for c in claims]} but never states the artifact's "
            f"{want}. A superseded figure may be cited alongside the current one, "
            f"never instead of it.")


@pytest.mark.parametrize("doc", PUBLISHING_DOCS, ids=lambda p: p.name)
def test_prose_curvatures_state_the_current_values(doc):
    data = _screening()
    want = [data[d]["criterion_2_curvature"] for d in _PROSE_ORDER]
    wanted = re.compile(
        rf"curvature\s+{want[0]:.3f},\s*{want[1]:.3f},\s*{want[2]:.3f}")
    for line in _lines_naming_the_primary_panel_in_order(doc.read_text()):
        claims = re.findall(r"curvature\s+(0\.\d+),\s*(0\.\d+),\s*(0\.\d+)", line)
        if not claims:
            continue
        assert wanted.search(line), (
            f"{doc.name}: a prose claim states curvatures {claims} but never states "
            f"the artifact's {[round(w, 3) for w in want]}")


@pytest.mark.parametrize("doc", PUBLISHING_DOCS, ids=lambda p: p.name)
def test_no_document_prints_a_measured_percentile_from_the_stability_artifact(doc):
    """Inline percentiles belong in the artifact; the refit falsified every one.

    Amendment 19's row once asserted, in its own text, that "the measurements are now
    in audits/latent_conflict_stability.json rather than in prose" while printing six
    groups of them. Every one moved when amendment 23 refitted the models. The ban is
    outright rather than marker-exempt, because a marker elsewhere in a long row was
    what let the stale figures survive.
    """
    stability = json.loads(
        (PAPER / "audits" / "latent_conflict_stability.json").read_text())["datasets"]
    text = doc.read_text()
    for token in ("0.14 to 0.17", "P(ρ > 0) = 53%"):
        assert token not in text, (
            f"{doc.name} prints the withdrawn percentile {token!r}. These figures are "
            f"maintained in audits/latent_conflict_stability.json and must not be "
            f"duplicated in prose, where a refit silently falsifies them.")
    for ds, block in stability.items():
        p95 = f"{block['abs_pearson_at_n88']['p95']:.4f}"
        assert text.count(p95) == 0, (
            f"{doc.name} duplicates {ds}'s measured p95 {p95} inline")
