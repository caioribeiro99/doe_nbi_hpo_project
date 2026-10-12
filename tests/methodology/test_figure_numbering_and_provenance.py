"""Figure numbers, figure files and the stated package provenance must agree with what exists.

The LaTeX build numbers its floats in the order it places them, while the text and the
captions carry fixed numbers. A figure first cited out of order was therefore printed under
one number and cited under another, and nothing failed. The supplement likewise named a tag
that had not been created as its "final package tag". Both are checked here.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
MAN = PAPER / "manuscript"
FIG = MAN / "figures"
PACKAGE_TAG = "paper2-manuscript-v6"


def _split() -> tuple[str, str]:
    text = (MAN / "MANUSCRIPT.md").read_text()
    body, captions = text.split("\n## Figure captions\n", 1)
    return body, captions


def _defined() -> list[int]:
    return [int(n) for n in re.findall(r"^\*\*Figure (\d+)\.", _split()[1], re.M)]


def test_figures_are_numbered_in_the_order_of_first_citation() -> None:
    body = _split()[0]
    first: list[int] = []
    for m in re.finditer(r"\bFigure\s+(\d+)\b", body):
        if int(m.group(1)) not in first:
            first.append(int(m.group(1)))
    defined = _defined()
    assert defined == list(range(1, len(defined) + 1)), defined
    assert first == defined, f"first cited in the order {first}"


def test_each_figure_number_has_exactly_one_png_and_one_vector_file() -> None:
    defined = _defined()
    pngs = sorted(int(p.name[3]) for p in FIG.glob("fig[0-9]_*.png"))
    vectors = sorted(int(p.stem.split("_")[1]) for p in FIG.glob("Figure_*.pdf"))
    assert pngs == defined, f"PNG numbers {pngs}"
    assert vectors == defined, f"vector figure numbers {vectors}"


def test_latex_places_each_figure_file_in_number_order() -> None:
    tex = (MAN / "latex" / "manuscript.tex").read_text()
    placed = [int(n) for n in re.findall(r"\\includegraphics\[[^\]]*\]\{Figure_(\d+)\.pdf\}", tex)]
    assert placed == _defined(), f"floats placed in the order {placed}"
    caps = (MAN / "latex" / "figure_captions.tex").read_text()
    assert [int(n) for n in re.findall(r"\\textbf\{Figure (\d+)\.\}", caps)] == _defined()


def test_supplement_names_the_package_tag_only_when_it_is_real() -> None:
    sup = (MAN / "SUPPLEMENT.md").read_text()
    final = re.search(r"\|\s*final package tag\s*\|\s*`([^`]+)`", sup)
    planned = re.search(r"\|\s*package tag\s*\|\s*not yet created; planned name\s*`([^`]+)`", sup)
    exists = subprocess.run(["git", "rev-parse", "--verify", "--quiet", PACKAGE_TAG + "^{commit}"],
                            cwd=PAPER, capture_output=True).returncode == 0
    releasing = os.environ.get("PAPER2_PACKAGE_RELEASE") == PACKAGE_TAG
    if exists or releasing:
        assert final and final.group(1) == PACKAGE_TAG and not planned
    else:
        assert planned and planned.group(1) == PACKAGE_TAG, "the draft must name the tag as planned"
        assert not final, f"{PACKAGE_TAG} does not exist but the supplement calls it the final package tag"
        assert "The final package commit and tag are recorded in S17" not in sup
