"""The rendered manuscript must not leak LaTeX, and must not misstate the maths.

The manuscript source keeps inline LaTeX so a LaTeX submission path stays open; the
HTML/PDF path has no MathJax, so ``demath`` substitutes Unicode. Two distinct failures
are pinned here. The first is cosmetic: a command the converter does not know reaches
the reader as literal backslash text. The second is not: flattening a nested
construct can print an expression that means something other than what was written.
"""
from __future__ import annotations

import pathlib
import re
import sys

import pytest

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
sys.path.insert(0, str(PAPER / "scripts"))

from demath import demath  # noqa: E402

SOURCES = ("MANUSCRIPT.md", "SUPPLEMENT.md")
LATEX = re.compile(r"\\[a-zA-Z]{2,12}")


@pytest.mark.parametrize("name", SOURCES)
def test_no_latex_survives_conversion(name: str) -> None:
    text = demath((PAPER / "manuscript" / name).read_text())
    leaked = sorted(set(LATEX.findall(text)))
    assert not leaked, f"{name} renders literal LaTeX: {leaked}"


@pytest.mark.parametrize("name", SOURCES)
def test_no_stray_math_delimiters(name: str) -> None:
    text = demath((PAPER / "manuscript" / name).read_text())
    assert "$" not in text, f"{name} leaves unpaired $ delimiters"


def test_sqrt_of_a_fraction_keeps_its_scope() -> None:
    """The Nadeau-Bengio inflation factor, the one place this actually bites.

    ``\\sqrt{\\frac{a}{b}}`` flattened to ``√a/b`` reads as ``(√a)/b`` and evaluates to
    15.97 rather than the 2.9155 the manuscript states two terms later.
    """
    out = demath(r"$\sqrt{\frac{1/30 + 0.25}{1/30}}$")
    assert out == "√((1/30 + 0.25)/(1/30))"


def test_nested_frac_arguments_resolve() -> None:
    assert demath(r"$\frac{\mathrm{SE}_{\text{corr}}}{\mathrm{SE}_{\text{naive}}}$") \
        == "(SE_corr)/(SE_naive)"


def test_quasi_normal_keeps_its_accent() -> None:
    """The prose names the quasi-normal n-hat; bare ``n`` would rename the symbol."""
    assert demath(r"$\hat{n}$") == "n̂"


def test_simple_radicand_is_not_over_parenthesised() -> None:
    assert demath(r"$\sqrt{8.5}$") == "√8.5"


def test_unbalanced_braces_do_not_hang() -> None:
    """A malformed source should degrade, not spin: the scanner must terminate."""
    assert demath(r"$\frac{a}{b$") is not None
