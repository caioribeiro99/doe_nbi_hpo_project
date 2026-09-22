"""S12.2 must not restate a universal the evidence does not support.

The per-dataset table in S12.2 is medians, and medians hide the per-unit exceptions.
The supplement previously concluded from that table that solver health was "identical on
every dataset" — the exact universal that CONFIRMATORY_CLAIMS_AND_EVIDENCE.md, recording
verifier C7, says "must not be stated as a universal". The manuscript's own closing
sentence agrees: comparable, not identical.

These tests pin the corrected wording and, more usefully, tie every campaign-level number
in it back to the manuscript, so the supplement cannot drift away from the paper.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = "papers/xgboost_hpo_vrfnbi"
TAG = "paper2-manuscript-v5"


def at_tag(rel: str) -> str:
    return subprocess.run(["git", "show", f"{TAG}:{PAPER}/{rel}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def s122() -> str:
    sup = (ROOT / PAPER / "manuscript" / "SUPPLEMENT.md").read_text()
    return sup[sup.index("### S12.2"):sup.index("### S12.3")]


def test_the_withdrawn_universal_is_gone(s122: str) -> None:
    for banned in ("Identical on every dataset", "identical on every dataset"):
        assert banned not in s122
    assert "comparable, but not identical" in s122, \
        "S12.2 must say comparable rather than identical"


def test_s122_states_the_exceptions_the_table_hides(s122: str) -> None:
    """A median of 1.000 is not the same as 1.000 everywhere."""
    for figure in ("240 arm-units", "0.900", "0.950", "6.6e-10", "6.9e-1",
                   "20 distinct realized configurations"):
        assert figure in s122, f"S12.2 no longer states {figure}"


def test_every_campaign_figure_matches_the_manuscript(s122: str) -> None:
    """The supplement may not invent or drift from the paper's audited values."""
    man = at_tag("manuscript/MANUSCRIPT.md")
    passage = man[man.index("Solver health is"):]
    passage = " ".join(passage[:passage.index("\n\n")].split())
    assert "240 arm-units" in passage
    for token, in_manuscript in (("0.900", "0.900"), ("0.950", "0.950"),
                                 ("6.6e-10", r"6.6\times10^{-10}"),
                                 ("6.9e-1", r"6.9\times10^{-1}"),
                                 ("20 distinct", "20 distinct")):
        assert token in s122
        assert in_manuscript in passage, \
            f"the manuscript no longer carries {in_manuscript}; S12.2 would be stale"


def test_the_per_dataset_table_is_unchanged(s122: str) -> None:
    """The correction is to the sentence only. The medians are scientific content."""
    rows = re.findall(r"^\| (MAGIC|Spambase|Adult|Bank Marketing) \|(.+)\|$", s122, re.M)
    assert len(rows) == 4
    expected = {"MAGIC": "8.6e-10", "Spambase": "5.9e-10",
                "Adult": "4.4e-10", "Bank Marketing": "6.1e-10"}
    for name, body in rows:
        assert "1.000 / 1.000" in body, f"{name} certified median changed"
        assert "20 / 20" in body, f"{name} distinct-config median changed"
        assert expected[name] in body, f"{name} residual median changed"
