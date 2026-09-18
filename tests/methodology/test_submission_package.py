"""The submission package must say exactly what the frozen master says.

A submission package is a second copy of the manuscript's declarations, and a second
copy is a second source of truth. This project has already lost a funding number to a
statement that was pinned but never checked against its source, so every block in
submission/DECLARATIONS.md is compared here against the manuscript at
paper2-manuscript-v5 -- not against a transcription of it.

It also asserts the package does not quietly acquire metadata nobody verified. The
target journal, the ORCIDs and the co-author affiliations are absent from the record;
they must stay absent from the package until an author supplies them.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = "papers/xgboost_hpo_vrfnbi"
TAG = "paper2-manuscript-v5"
SUB = ROOT / PAPER / "submission"


def at_tag(rel: str) -> str:
    return subprocess.run(["git", "show", f"{TAG}:{PAPER}/{rel}"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def master() -> str:
    return at_tag("manuscript/MANUSCRIPT.md")


@pytest.fixture(scope="module")
def declarations() -> str:
    if not (SUB / "DECLARATIONS.md").exists():
        pytest.skip("submission package not built on this branch")
    return (SUB / "DECLARATIONS.md").read_text()


@pytest.mark.parametrize("label", [
    "Funding.", "Acknowledgements.", "Competing interests.",
    "Data and code availability.", "Keywords:"])
def test_declaration_block_is_verbatim_from_the_master(label, master, declarations) -> None:
    start = master.index(f"**{label}**")
    text = master[start:master.index("\n\n", start)].strip()
    assert text in declarations, f"{label} does not match {TAG} verbatim"


def test_credit_statement_is_verbatim_from_the_master(master, declarations) -> None:
    start = master.index("**Author contributions (CRediT).**")
    text = master[start:master.index("\n\n---", start)].strip()
    assert text in declarations


def test_title_and_abstract_are_verbatim_from_the_master(master, declarations) -> None:
    title = master.splitlines()[0].lstrip("# ").strip()
    abstract = master[master.index("## Abstract") + 11:master.index("**Keywords:**")].strip()
    assert title in declarations
    assert abstract in declarations


def test_package_carries_no_unverified_metadata(declarations) -> None:
    """No journal name, no ORCID, no invented affiliation may appear in the package."""
    whole = " ".join(f.read_text() for f in SUB.glob("*.md"))
    assert not re.search(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", whole), \
        "an ORCID iD appears in the package; none is recorded anywhere in this repository"
    for invented in ("Finance Code", "140663", "FAPESP"):
        assert invented not in whole


def test_unfilled_fields_are_declared_not_guessed() -> None:
    meta = (SUB / "SUBMISSION_METADATA.md").read_text()
    for field in ("Target journal", "ORCID iDs", "Corresponding author",
                  "Affiliation, Matheus Costa Pereira",
                  "Affiliation, Anderson Paulo de Paiva"):
        assert field in meta, f"{field} must be listed as AUTHOR INPUT REQUIRED"
    assert "AUTHOR INPUT REQUIRED" in meta


def test_manifest_checksums_match_the_shipped_pdfs() -> None:
    import hashlib
    manifest = (SUB / "PACKAGE_MANIFEST.md").read_text()
    pdfs = sorted((ROOT / PAPER / "manuscript").glob("Paper2_*_FINAL_v5.pdf"))
    assert pdfs, "no v5 PDFs present"
    for f in pdfs:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        assert digest in manifest, f"{f.name} checksum is stale in the manifest"
