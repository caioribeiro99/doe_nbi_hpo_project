"""The ASOC package must match the frozen master, and must not invent author metadata.

ORCID iDs are the sharpest hazard here: a wrong one silently attaches a stranger's
publication record to this paper, and nothing in the manuscript would look wrong. They
are therefore re-resolved against the ORCID public API rather than trusted from a
comment. The network check skips cleanly offline; the local checks always run.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = "papers/xgboost_hpo_vrfnbi"
TAG = "paper2-manuscript-v5"
SUB = ROOT / PAPER / "submission"

ORCIDS = {
    "Caio Tertuliano Ribeiro": "0009-0006-7748-1449",
    "Matheus Costa Pereira": "0009-0007-2011-9235",
    "Anderson Paulo de Paiva": "0000-0002-8199-411X",
}


def at_tag(rel: str) -> str:
    return subprocess.run(["git", "show", f"{TAG}:{PAPER}/{rel}"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout


@pytest.fixture(scope="module")
def title_page() -> str:
    p = SUB / "ASOC_TITLE_PAGE.md"
    if not p.exists():
        pytest.skip("ASOC package not built on this branch")
    return p.read_text()


@pytest.mark.parametrize("name,orcid", ORCIDS.items())
def test_orcid_resolves_to_the_named_person(name: str, orcid: str) -> None:
    """Re-resolve against the ORCID registry. A transposed digit must fail here."""
    try:
        req = urllib.request.Request(f"https://pub.orcid.org/v3.0/{orcid}/person",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=25) as r:
            person = json.load(r)
    except Exception as exc:                       # offline, rate-limited, blocked
        pytest.skip(f"ORCID API unreachable: {exc}")
    reg = person.get("name") or {}
    got = " ".join(x for x in [(reg.get("given-names") or {}).get("value"),
                               (reg.get("family-name") or {}).get("value")] if x)
    assert got.strip().lower() == name.lower(), \
        f"{orcid} is registered to {got!r}, not {name!r}"


@pytest.mark.parametrize("name,orcid", ORCIDS.items())
def test_title_page_carries_the_verified_orcid(name: str, orcid: str, title_page) -> None:
    assert name in title_page
    assert orcid in title_page


def test_no_orcid_outside_the_verified_set(title_page) -> None:
    found = set(re.findall(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", title_page))
    assert found == set(ORCIDS.values()), f"unexpected ORCID-shaped strings: {found}"


def test_unverified_metadata_is_marked_not_guessed(title_page) -> None:
    """de Paiva's affiliation, the institutional email and the postal address."""
    assert title_page.count("**AUTHOR INPUT REQUIRED**") >= 3
    assert "caio.tertu99@gmail.com" in title_page, \
        "the one address on record should be shown, labelled as personal"


@pytest.mark.parametrize("label", ["Funding.", "Acknowledgements.", "Competing interests."])
def test_declaration_is_verbatim_from_the_master(label: str, title_page) -> None:
    master = at_tag("manuscript/MANUSCRIPT.md")
    start = master.index(f"**{label}**")
    assert master[start:master.index("\n\n", start)].strip() in title_page


def test_credit_is_verbatim_from_the_master(title_page) -> None:
    master = at_tag("manuscript/MANUSCRIPT.md")
    s = master.index("**Author contributions (CRediT).**")
    assert master[s:master.index("\n\n---", s)].strip() in title_page


def test_highlights_respect_the_character_limit() -> None:
    p = SUB / "ASOC_HIGHLIGHTS.md"
    if not p.exists():
        pytest.skip("not built")
    bullets = [m.group(1) for m in re.finditer(r"^- (.+?)  \*\(\d+ characters\)\*$",
                                               p.read_text(), re.M)]
    assert 3 <= len(bullets) <= 5, f"{len(bullets)} highlights; the journal allows 3 to 5"
    for b in bullets:
        assert len(b) <= 85, f"{len(b)} characters: {b!r}"


def test_keywords_respect_the_journal_limit() -> None:
    man = (ROOT / PAPER / "manuscript" / "MANUSCRIPT.md").read_text()
    kw = man[man.index("**Keywords:**"):man.index("\n\n", man.index("**Keywords:**"))]
    keys = [" ".join(k.split()) for k in kw.replace("**Keywords:**", "").split(";")]
    assert len(keys) <= 7, f"{len(keys)} keywords; Applied Soft Computing allows 1 to 7"
    assert not [k for k in keys if " of " in k or " and " in k]


def test_graphical_abstract_meets_the_size_rule() -> None:
    png = SUB / "ASOC_graphical_abstract.png"
    if not png.exists():
        pytest.skip("not built")
    Image = pytest.importorskip("PIL.Image", reason="Pillow not installed")
    w, h = Image.open(png).size
    assert w >= 1328 and h >= 531
    assert abs(w / h - 1328 / 531) < 0.02, "must be proportional to the stated minimum"


def test_graphical_abstract_states_only_v5_numbers() -> None:
    """Every median it prints must appear in the frozen abstract."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ga", ROOT / PAPER / "scripts" / "build_graphical_abstract.py")
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    master = at_tag("manuscript/MANUSCRIPT.md")
    abstract = master[master.index("## Abstract"):master.index("**Keywords:**")]
    # the figure renders the two mechanism contrasts only; the specification contrast
    # is loaded but printed as prose ("no detectable change"), not as a number
    rendered = ("WS-S -> NBI-S", "NBI-S -> NBI-R")
    data = ga.load()
    assert set(rendered) <= set(data)
    for contrast in rendered:
        for ds, v in data[contrast].items():
            assert ("%+.3f" % v) in abstract, \
                f"{contrast} on {ds} renders {v:+.3f}, which is not in the v5 abstract"


# The files the publisher actually receives. The two audit records are internal
# venue-assessment notes whose whole purpose is to compare the two journals, so naming
# EAAI in them is correct; naming it in a deliverable would not be.
DELIVERABLES = ("ASOC_TITLE_PAGE.md", "ASOC_HIGHLIGHTS.md", "ASOC_COVER_LETTER.md")
AUDIT_RECORDS = ("ASOC_SCOPE_AUDIT.md", "ASOC_REQUIREMENTS.md")


def test_no_eaai_language_leaked_into_a_deliverable() -> None:
    for name in DELIVERABLES:
        f = SUB / name
        if not f.exists():
            pytest.skip(f"{name} not built")
        text = f.read_text()
        for bad in ("Engineering Applications of Artificial Intelligence", "EAAI",
                    "application in engineering"):
            assert bad not in text, f"{name} mentions {bad}"


def test_every_asoc_file_is_either_a_deliverable_or_a_declared_audit_record() -> None:
    """Stops a new file from quietly inheriting the audit exemption."""
    present = {f.name for f in SUB.glob("ASOC_*.md")}
    assert present <= set(DELIVERABLES) | set(AUDIT_RECORDS),         f"undeclared ASOC file(s): {sorted(present - set(DELIVERABLES) - set(AUDIT_RECORDS))}"
