"""The claim blacklist must be enforced by the test suite, not only at build time.

`COST_OBJECTIVE_CLAIM_BOUNDARY.md` asserts that the terminology rules are "enforced
by scripts/check_claim_blacklist.py". That was false for two reasons: the scanner
looked only for *.tex and a compiled PDF, of which there were none, so it exited
having scanned nothing; and it was never run automatically. Both are fixed here.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "scripts" / "check_claim_blacklist.py"
PAPER = REPO / "papers" / "xgboost_hpo_vrfnbi"


def _load():
    spec = importlib.util.spec_from_file_location("blacklist", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bl = _load()


def test_the_workspace_contains_no_blacklisted_claim() -> None:
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_scanner_actually_scans_something() -> None:
    """The failure mode that made the documented control fictional."""
    text, source = bl.load_text(None, PAPER)
    assert len(text.split()) > 5_000, f"scanned only {len(text.split())} words from {source}"
    assert ".md" in source or ".tex" in source


def test_exemptions_are_named_not_detected() -> None:
    """A phrase in a paragraph must not be able to exempt a document.

    Exemption is by file name in a list a reader can audit, so adding one is a
    visible edit rather than a sentence an author can drop anywhere.
    """
    assert bl.QUOTING_DOCUMENTS, "no exemptions declared"
    for name, reason in bl.QUOTING_DOCUMENTS.items():
        assert (PAPER / name).exists(), f"exempt file {name} does not exist"
        assert len(reason) > 10, f"exemption of {name} has no stated reason"


@pytest.mark.parametrize(
    "claim",
    [
        "We optimize three objectives because NBI's simplex geometry rewards it.",
        "At three objectives NBI shows a clear advantage over the weighted sum.",
        "The non-dominated set grew tenfold, which demonstrates real structure.",
        "Three objectives recover the suppressed precision trade-off.",
        "The surrogate is adequate on all four datasets.",
        "Stage A confirms that the surrogate is reliable.",
        "The surrogate is reliable.",
        "This study was pre-registered.",
        "We minimize training time as the second objective.",
        "Our cost objective is training time.",
        "The third objective conflicts with quality across the panel.",
    ],
)
def test_each_new_rule_catches_its_claim(tmp_path, claim: str) -> None:
    (tmp_path / "m.tex").write_text(claim)
    text, _ = bl.load_text(None, tmp_path)
    import re
    hits = [why for _, pat, why in bl.BLACKLIST
            if re.search(pat, re.sub(r"\s+", " ", text), re.I)]
    assert hits, f"no rule caught: {claim!r}"


@pytest.mark.parametrize(
    "sentence",
    [
        "Whether NBI gives more uniformly spaced fronts here is measured in Section 5.",
        "The protocol must not claim that the surrogate is reliable.",
        "All four candidate datasets passed the pre-campaign screening under the revised procedure.",
        "Leaf count is a deterministic model-complexity proxy tracking measured wall clock at 0.86.",
        "Tag v1 records what was pre-registered before any measurement.",
    ],
)
def test_legitimate_sentences_are_not_flagged(tmp_path, sentence: str) -> None:
    (tmp_path / "m.tex").write_text(sentence)
    text, _ = bl.load_text(None, tmp_path)
    import re
    flat = re.sub(r"\s+", " ", text)
    for _, pat, why in bl.BLACKLIST:
        for mm in re.finditer(pat, flat, re.I):
            run_up = flat[max(0, mm.start() - 60):mm.start()]
            if bl.HEDGE.search(run_up):
                continue
            if bl.PROHIBITION.search(flat[max(0, mm.start() - 140):mm.start()]):
                continue
            pytest.fail(f"false positive ({why}) on: {sentence!r}")
