"""The declarations are author-supplied and no artifact can check them.

Every other number in the manuscript traces to a committed artifact, so
``audit_manuscript_numbers`` verifies it. A grant number has no such backing: a digit
could change and every audit would still pass. The statements the authors supplied are
therefore pinned verbatim here, together with the roles they confirmed, so any drift is
a test failure rather than a silent corruption of someone's funding record.
"""
from __future__ import annotations

import pathlib

import pytest

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
MAN = (PAPER / "manuscript" / "MANUSCRIPT.md").read_text()

FUNDING = (
    "**Funding.** Caio Tertuliano Ribeiro acknowledges doctoral scholarship support "
    "from the\nConselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq), "
    "process\n140663/2026-6. Anderson Paulo de Paiva acknowledges research support "
    "from CNPq, process\n312844/2023-9."
)
ACKNOWLEDGEMENTS = (
    "**Acknowledgements.** The authors acknowledge the Universidade Federal de Itajubá\n"
    "(UNIFEI) for institutional support."
)
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


def test_funding_statement_is_verbatim() -> None:
    assert FUNDING in MAN, "the funding statement no longer matches the author's text"


def test_acknowledgements_are_verbatim() -> None:
    assert ACKNOWLEDGEMENTS in MAN


def test_no_unauthorised_funder_is_named() -> None:
    """CAPES and FAPEMIG were explicitly excluded; neither may reappear."""
    for banned in ("CAPES", "FAPEMIG", "Finance Code", "BPD-01045"):
        assert banned not in MAN, f"{banned} must not appear in this manuscript"


@pytest.mark.parametrize("author,roles", ROLES.items())
def test_credit_roles_are_exactly_as_confirmed(author: str, roles: list[str]) -> None:
    start = MAN.index("*%s* —" % author)
    entry = MAN[start:MAN.index(".\n", start) + 1]
    flat = " ".join(entry.split())
    for role in roles:
        assert role in flat, f"{author} is missing the confirmed role {role}"
    # nothing beyond the confirmed set
    stated = {r.strip(" .*") for r in flat.split("—", 1)[1].split(";")}
    assert stated == set(roles), f"{author} states {sorted(stated - set(roles))} extra"


def test_resources_is_deliberately_unassigned() -> None:
    assert "The CRediT role *Resources* is not assigned." in MAN
    for author in ROLES:
        start = MAN.index("*%s* —" % author)
        assert "Resources" not in MAN[start:MAN.index(".\n", start)]


@pytest.mark.parametrize("marker", ("AUTHOR ACTION REQUIRED", "AUTHOR REVIEW"))
def test_no_author_action_marker_survives(marker: str) -> None:
    sup = (PAPER / "manuscript" / "SUPPLEMENT.md").read_text()
    assert marker not in MAN and marker not in sup
