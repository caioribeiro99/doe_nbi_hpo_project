"""The declarations, and the evidence standing behind each element of them.

Pinning a string proves only that nobody edited it. It does not establish that the
string is true, and an earlier version of this file did exactly that: it fixed CNPq
process 140663/2026-6 verbatim and passed, while no source anywhere established that
number. The number was withdrawn in v5 and is asserted absent below.

Two properties are enforced, and the second is the one that was missing.

  BOUNDED   the funding and acknowledgements paragraphs are compared by EQUALITY over a
            delimited region, not by containment. A containment check cannot see text
            appended after the pinned sentences, so an extra unevidenced funder passed
            the whole suite.
  TWO-WAY   every funder and identifier NAMED IN THE MANUSCRIPT must appear in EVIDENCE,
            not merely the reverse. A one-way whitelist only checks that declared
            elements are present; it is blind to undeclared ones, which is precisely the
            failure v5 exists to correct.

Evidence classes, which are not interchangeable:

  REPOSITORY  corroborated by a file in this repository that ASSERTS the fact. The test
              reads that file. An unresolved TODO is not an assertion and does not
              qualify -- treating one as corroboration would repeat the provenance error
              this file documents.
  AUTHOR      attested by the authors from a source outside this repository (the
              master's thesis acknowledgements). The test pins the wording. It does not,
              and cannot, claim independent verification.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = ROOT / "papers" / "xgboost_hpo_vrfnbi"
MAN = (PAPER / "manuscript" / "MANUSCRIPT.md").read_text()
SUP = (PAPER / "manuscript" / "SUPPLEMENT.md").read_text()

FUNDING = (
    "**Funding.** The master's research underlying this study received support from the\n"
    "Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG), project "
    "BPD-01045-22,\nand from the Coordenação de Aperfeiçoamento de Pessoal de Nível "
    "Superior (CAPES).\nAnderson Paulo de Paiva acknowledges research support from the "
    "Conselho Nacional de\nDesenvolvimento Científico e Tecnológico (CNPq), process "
    "312844/2023-9."
)
ACKNOWLEDGEMENTS = (
    "**Acknowledgements.** The authors acknowledge the Universidade Federal de Itajubá\n"
    "(UNIFEI) for institutional support."
)

# Every funding body and identifier the manuscript is permitted to name, each with the
# evidence class and source that stands behind it.
EVIDENCE: dict[str, tuple[str, str, str]] = {
    "FAPEMIG": ("AUTHOR", "master's thesis acknowledgements", ""),
    "CAPES": ("AUTHOR", "master's thesis acknowledgements, no grant number given", ""),
    "CNPq": ("AUTHOR", "de Paiva funding record / predecessor acknowledgement", ""),
    "UNIFEI": ("REPOSITORY", "article/main.tex", "UNIFEI"),
}
IDENTIFIERS = {
    "BPD-01045-22": ("AUTHOR", "master's thesis acknowledgements"),
    "312844/2023-9": ("AUTHOR", "de Paiva funding record"),
}
# article/main.tex names BPD-01045-22 only inside an unresolved \todoblock, conditionally
# ("collaboration if applicable"). That is an instruction to a future author, not an
# assertion, so FAPEMIG is classed AUTHOR above and not REPOSITORY. This records the
# mention so the distinction is deliberate and visible rather than an omission.
NOT_CORROBORATION = {"BPD-01045-22": "article/main.tex \\todoblock, conditional"}

WITHDRAWN = {
    "140663/2026-6": "withdrawn in v5: no source established this CNPq process",
    "140663": "withdrawn in v5: no source established this CNPq process",
    "Finance Code 001": "never evidenced for this manuscript",
    "Finance Code": "never evidenced for this manuscript",
}
LEDGERS = {"papers/xgboost_hpo_vrfnbi/scripts/audit_scientific_invariance.py"}

ROLES = {
    "Caio Tertuliano Ribeiro": [
        "Conceptualization", "Methodology", "Software", "Validation", "Formal analysis",
        "Investigation", "Data curation", "Project administration",
        "Writing – original draft", "Visualization"],
    "Matheus Costa Pereira": ["Methodology", "Writing – review & editing"],
    "Anderson Paulo de Paiva": [
        "Conceptualization", "Supervision", "Funding acquisition",
        "Writing – review & editing"],
}

ACRONYM = re.compile(r"\b([A-Z][A-Za-z]{2,}(?:[A-Z]|q))\b")
GRANT = re.compile(r"\b(?:[A-Z]{2,}-\d[\d-]*|\d{4,}/\d{4}-\d)\b")


def declarations_region() -> str:
    """Everything between Competing interests and Author contributions.

    Delimiting the region is what makes the check bounded: a funder appended as a new
    paragraph after Acknowledgements lands inside it and fails equality below.
    """
    start = MAN.index("**Competing interests.**")
    return MAN[MAN.index("\n\n", start) + 2:MAN.index("**Author contributions (CRediT).**")]


def test_declarations_region_is_exactly_funding_then_acknowledgements() -> None:
    assert declarations_region().strip() == (FUNDING + "\n\n" + ACKNOWLEDGEMENTS), (
        "the funding/acknowledgements region does not match the author's text exactly; "
        "text was added, removed or altered")


def test_funding_is_scoped_to_the_underlying_research() -> None:
    """FAPEMIG and CAPES supported the master's research, not the new campaign."""
    assert "The master's research underlying this study received support" in MAN


@pytest.mark.parametrize("name", sorted(EVIDENCE))
def test_each_declared_funder_has_its_evidence(name: str) -> None:
    kind, source, needle = EVIDENCE[name]
    assert kind in ("REPOSITORY", "AUTHOR")
    if kind == "REPOSITORY":
        assert needle in (ROOT / source).read_text(), (
            f"{name} is declared REPOSITORY-corroborated by {source}, which no longer "
            f"contains {needle!r}")


def test_no_funder_is_named_without_evidence() -> None:
    """The reverse direction: everything named must be declared.

    A one-way whitelist let an unevidenced funder sit in the funding statement while all
    tests passed. This walks the manuscript's own text back to EVIDENCE.
    """
    named = {a for a in ACRONYM.findall(declarations_region())}
    undeclared = named - set(EVIDENCE) - {"CRediT"}
    assert not undeclared, f"funding bodies named with no evidence entry: {sorted(undeclared)}"


def test_no_identifier_is_named_without_evidence() -> None:
    found = set(GRANT.findall(declarations_region()))
    undeclared = found - set(IDENTIFIERS)
    assert not undeclared, f"grant identifiers with no evidence entry: {sorted(undeclared)}"


@pytest.mark.parametrize("ident", sorted(NOT_CORROBORATION))
def test_weak_mentions_are_not_counted_as_corroboration(ident: str) -> None:
    """Whatever article/main.tex says about BPD-01045-22, it is a TODO, not a claim."""
    line = next(l for l in (ROOT / "article" / "main.tex").read_text().splitlines()
                if ident in l)
    assert "todoblock" in line or "if applicable" in line or "Mention" in line, (
        f"{ident} now appears in article/main.tex outside a TODO; if it has become an "
        "assertion, reclassify FAPEMIG as REPOSITORY deliberately rather than by drift")
    assert EVIDENCE["FAPEMIG"][0] == "AUTHOR"


@pytest.mark.parametrize("value,reason", sorted(WITHDRAWN.items()))
def test_withdrawn_value_appears_nowhere_in_paper_2(value: str, reason: str) -> None:
    hits = []
    for path in PAPER.rglob("*"):
        if not path.is_file() or "vendor" in path.parts:
            continue
        if path.suffix.lower() in {".pdf", ".png", ".html"}:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if value not in text:
            continue
        rel = str(path.relative_to(ROOT))
        if rel in LEDGERS:
            bad = [n for n, line in enumerate(text.splitlines(), 1)
                   if value in line and "withdrawn" not in line.lower()]
            if bad:
                hits.append(f"{rel}: lines {bad} name it without declaring it withdrawn")
            continue
        hits.append(rel)
    assert not hits, f"{value!r} ({reason}) still present in: {hits}"


def _pdf_text(stem: str) -> str:
    pymupdf = pytest.importorskip("pymupdf")
    pdfs = sorted((PAPER / "manuscript").glob(f"Paper2_{stem}_FINAL_v*.pdf"))
    if not pdfs:
        pytest.skip("PDF not built in this working tree")
    return " ".join("".join(p.get_text() for p in pymupdf.open(pdfs[-1])).split())


def test_rendered_pdf_carries_the_declarations() -> None:
    """Nothing previously tied the shipped PDF's declarations to the manuscript."""
    flat = _pdf_text("MANUSCRIPT")
    for block in (FUNDING, ACKNOWLEDGEMENTS):
        expect = " ".join(block.replace("**", "").split())
        assert expect in flat, f"the PDF does not render: {expect[:70]}..."
    for author, roles in ROLES.items():
        assert author in flat
        for role in roles:
            assert role in flat, f"the PDF does not render {author}'s role {role}"


def test_withdrawn_value_absent_from_the_rendered_pdfs() -> None:
    for stem in ("MANUSCRIPT", "SUPPLEMENT"):
        flat = _pdf_text(stem)
        for value in ("140663", "Finance Code"):
            assert value not in flat, f"the {stem} PDF still renders {value!r}"


def test_exactly_the_confirmed_authors_are_credited() -> None:
    """A fourth author could previously be added and credited with anything."""
    block = MAN[MAN.index("**Author contributions (CRediT).**"):
                MAN.index("The CRediT role *Resources* is not assigned.")]
    named = set(re.findall(r"^\*([^*]+)\* —", block, re.M))
    assert named == set(ROLES), f"credited authors are {sorted(named)}"


@pytest.mark.parametrize("author,roles", ROLES.items())
def test_credit_roles_are_exactly_as_confirmed(author: str, roles: list[str]) -> None:
    start = MAN.index("*%s* —" % author)
    entry = MAN[start:MAN.index(".\n", start) + 1]
    flat = " ".join(entry.split())
    stated = {r.strip(" .*") for r in flat.split("—", 1)[1].split(";")}
    assert stated == set(roles), (
        f"{author}: extra {sorted(stated - set(roles))}, "
        f"missing {sorted(set(roles) - stated)}")


def test_resources_is_deliberately_unassigned() -> None:
    assert "The CRediT role *Resources* is not assigned." in MAN
    for author in ROLES:
        start = MAN.index("*%s* —" % author)
        assert "Resources" not in MAN[start:MAN.index(".\n", start)]


@pytest.mark.parametrize("marker", ("AUTHOR ACTION REQUIRED", "AUTHOR REVIEW"))
def test_no_author_action_marker_survives(marker: str) -> None:
    assert marker not in MAN and marker not in SUP
