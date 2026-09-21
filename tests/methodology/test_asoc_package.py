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


def test_generative_ai_declaration_uses_the_current_section_title() -> None:
    """The 2024 capture said "writing process"; Elsevier's current title differs."""
    man = (ROOT / PAPER / "manuscript" / "MANUSCRIPT.md").read_text()
    current = ("## Declaration of generative AI and AI-assisted technologies in the "
               "manuscript preparation process")
    assert current in man, "the declaration must use the current Elsevier section title"
    assert "in the writing process" not in man, "stale 2024 section title"
    i = man.index(current)
    assert man.index("## References") > i, "the declaration must precede the references"
    assert "Anthropic Claude" in man and "OpenAI ChatGPT" in man
    for research_ai in ("XGBoost", "NSGA-II"):
        seg = man[i:man.index("## References")]
        assert research_ai not in seg, \
            f"{research_ai} is research methodology and must not appear in the AI declaration"


def test_vitae_respect_the_word_limit_and_are_unconfirmed() -> None:
    p = SUB / "ASOC_VITAE.md"
    if not p.exists():
        pytest.skip("not built")
    text = p.read_text()
    counts = [int(m) for m in re.findall(r"\((\d+) words; limit 100\)", text)]
    assert len(counts) == 3, f"expected three biographies, found {len(counts)}"
    assert all(c <= 100 for c in counts), counts
    assert text.count("**AUTHOR CONFIRMATION REQUIRED**") >= 3
    assert text.count("PHOTO_PLACEHOLDER") == 3, "one photo placeholder per author"


def test_doi_is_a_placeholder_not_an_invented_identifier() -> None:
    p = SUB / "ASOC_DATA_CODE_AVAILABILITY.md"
    if not p.exists():
        pytest.skip("not built")
    text = p.read_text()
    assert "[ZENODO_DOI_PLACEHOLDER]" in text
    assert not re.search(r"10\.5281/zenodo\.\d+", text), "an invented Zenodo DOI is present"


def test_every_mutable_rule_carries_a_verification_status() -> None:
    p = SUB / "ASOC_SUBMISSION_CHECKLIST.md"
    if not p.exists():
        pytest.skip("not built")
    rows = [l for l in p.read_text().splitlines()
            if l.startswith("| ") and "---" not in l and "Requirement" not in l]
    assert rows, "checklist has no rows"
    for r in rows:
        assert "VERIFIED LIVE" in r or "ARCHIVE" in r, f"row without a status: {r[:70]}"


def test_no_orcid_outside_the_verified_set(title_page) -> None:
    found = set(re.findall(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b", title_page))
    assert found == set(ORCIDS.values()), f"unexpected ORCID-shaped strings: {found}"


CORRESPONDING_EMAIL = "caio.tertu@hotmail.com"
SUPERSEDED_EMAIL = "caio.tertu99@gmail.com"


def test_no_title_page_field_remains_unresolved(title_page) -> None:
    """Every field is now supplied. Asserted exactly, not as a lower bound, so that
    marking a new field unresolved without saying so also fails."""
    assert title_page.count("**AUTHOR INPUT REQUIRED**") == 0
    assert f"Email: `{CORRESPONDING_EMAIL}`" in title_page
    assert "author-confirmed" in title_page
    assert "Av. BPS 1303" in title_page, "the verified postal address must be present"
    for name in ORCIDS:
        assert "Institute of Production Engineering and Management (IEPG)" in title_page
    assert "does **not** require an institutional address" in title_page, \
        "the package must not impose a requirement the journal does not"


def test_no_institutional_address_was_substituted(title_page) -> None:
    """The author chose a personal address; no @unifei.edu.br may be invented for them."""
    import re as _re
    for m in _re.finditer(r"[\w.+-]+@[\w.-]+", title_page):
        addr = m.group(0).rstrip(".,;:)")   # trailing sentence punctuation
        # de Paiva's SIGAA address may appear as affiliation evidence, never as the
        # corresponding address
        assert addr in (CORRESPONDING_EMAIL, "andersonppaiva@unifei.edu.br"), \
            f"unexpected address on the title page: {addr}"
    i = title_page.index("### Corresponding author")
    assert "unifei.edu.br" not in title_page[i:], \
        "no institutional address may stand as the corresponding address"


def test_superseded_email_is_gone_from_every_deliverable() -> None:
    for name in DELIVERABLES:
        f = SUB / name
        if f.exists():
            assert SUPERSEDED_EMAIL not in f.read_text(), \
                f"{name} still carries the superseded address"


def test_cover_letter_carries_the_confirmed_contact() -> None:
    f = SUB / "ASOC_COVER_LETTER.md"
    if not f.exists():
        pytest.skip("not built")
    assert CORRESPONDING_EMAIL in f.read_text()


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
DELIVERABLES = ("ASOC_TITLE_PAGE.md", "ASOC_HIGHLIGHTS.md", "ASOC_COVER_LETTER.md",
                "ASOC_VITAE.md", "ASOC_DATA_CODE_AVAILABILITY.md")
AUDIT_RECORDS = ("ASOC_SCOPE_AUDIT.md", "ASOC_REQUIREMENTS.md",
                 "ASOC_SUBMISSION_CHECKLIST.md", "ASOC_PACKAGE_MANIFEST.md")


# A biography may state where an author has PUBLISHED — that is publication record, not
# venue framing. What must never appear is EAAI's own scope language or any phrasing that
# addresses EAAI as the target. The two are separated rather than exempting a whole file.
EAAI_FRAMING = ("application in engineering", "real-world engineering application",
                "EAAI", "we submit", "Engineering Applications of Artificial "
                "Intelligence for consideration")
VENUE_CONTEXT = ("published in", "author of", "co-author of", "appeared in")


def test_no_eaai_framing_leaked_into_a_deliverable() -> None:
    for name in DELIVERABLES:
        f = SUB / name
        if not f.exists():
            continue
        text = " ".join(f.read_text().split())
        for bad in EAAI_FRAMING:
            assert bad not in text, f"{name} carries EAAI framing: {bad!r}"


def test_eaai_is_named_only_as_a_publication_venue() -> None:
    """Where the journal name appears at all, it must be a past publication."""
    name = "Engineering Applications of Artificial Intelligence"
    for f in (SUB / d for d in DELIVERABLES):
        if not f.exists():
            continue
        text = " ".join(f.read_text().split())
        for i in range(len(text)):
            i = text.find(name, i)
            if i < 0:
                break
            # look back to the start of the sentence: a list of venues can put
            # "published in" well before the name it governs
            start = max((text.rfind(p, 0, i) for p in (". ", "! ", "? ")), default=-1)
            sentence = text[start + 1:i]
            assert any(v in sentence for v in VENUE_CONTEXT), (
                f"{f.name} names the journal outside a publication context: "
                f"...{sentence[-60:]}[{name}]")
            i += len(name)


def test_every_asoc_file_is_either_a_deliverable_or_a_declared_audit_record() -> None:
    """Stops a new file from quietly inheriting the audit exemption."""
    present = {f.name for f in SUB.glob("ASOC_*.md")}
    assert present <= set(DELIVERABLES) | set(AUDIT_RECORDS),         f"undeclared ASOC file(s): {sorted(present - set(DELIVERABLES) - set(AUDIT_RECORDS))}"
