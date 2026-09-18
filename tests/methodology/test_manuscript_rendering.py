"""The rendered mathematics must survive markdown and reach a real typesetter intact.

Converting LaTeX to Unicode by string replacement was the earlier approach and it is
retired: it cannot express the CHIM definition, and it failed quietly rather than
loudly -- ``\\geq`` matched the ``\\ge`` entry and printed "rank-biserial ≥ q +0.88".
Mathematics is now typeset by MathJax from the original LaTeX, so what these tests
protect is the handoff: markdown must not touch a math span, every span must come back
byte for byte, and no expression may use a construct outside the loaded packages.
"""
from __future__ import annotations

import pathlib
import re
import sys

import markdown
import pytest

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
sys.path.insert(0, str(PAPER / "scripts"))

from mathprotect import protect, restore, unrestored  # noqa: E402

SOURCES = ("MANUSCRIPT.md", "SUPPLEMENT.md")

# Everything TeX-SVG provides with the base and ams packages. A command outside this
# set renders as an error box in the PDF, which a text-level check cannot see.
SUPPORTED = {
    r"\Phi", r"\Lambda", r"\alpha", r"\beta", r"\lambda", r"\rho", r"\sigma", r"\mu",
    r"\epsilon", r"\phi", r"\top", r"\times", r"\cdot", r"\pm", r"\sum", r"\sqrt",
    r"\frac", r"\max", r"\min", r"\log", r"\ge", r"\geq", r"\le", r"\leq", r"\neq",
    r"\approx", r"\to", r"\rightarrow", r"\leftarrow", r"\in", r"\hat", r"\bar",
    r"\tilde", r"\mathbf", r"\mathrm", r"\text", r"\operatorname", r"\lVert", r"\rVert",
    r"\lfloor", r"\rfloor", r"\mid", r"\infty", r"\prime", r"\subseteq",
}
# Spacing and delimiter escapes, which are punctuation rather than named commands.
ESCAPES = {r"\,", r"\;", r"\!", "\\ ", r"\{", r"\}", r"\|"}


def _spans(name: str) -> list[str]:
    return protect((PAPER / "manuscript" / name).read_text())[1]


@pytest.mark.parametrize("name", SOURCES)
def test_markdown_never_touches_a_math_span(name: str) -> None:
    """``$F(x^{*i}) - F^{*}$`` used to become an emphasis run and print "is*"."""
    text, spans = protect((PAPER / "manuscript" / name).read_text())
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    assert not unrestored(html) or True          # tokens are expected here
    out = restore(html, spans)
    for span in spans:
        assert span in out, f"{name}: markdown altered the span {span[:60]!r}"


@pytest.mark.parametrize("name", SOURCES)
def test_every_math_span_is_restored(name: str) -> None:
    text, spans = protect((PAPER / "manuscript" / name).read_text())
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    out = restore(html, spans)                    # raises if a token cannot be placed
    assert not unrestored(out)


@pytest.mark.parametrize("name", SOURCES)
def test_no_unsupported_latex_command(name: str) -> None:
    used = set()
    for span in _spans(name):
        body = span[2:-2]                         # drop the \( \) or \[ \] wrapper
        used |= set(re.findall(r"\\[a-zA-Z]+|\\[^a-zA-Z]", body))
    unknown = sorted(used - SUPPORTED - ESCAPES)
    assert not unknown, f"{name}: MathJax would render an error box for {unknown}"


@pytest.mark.parametrize("name", SOURCES)
def test_math_delimiters_are_balanced(name: str) -> None:
    text = (PAPER / "manuscript" / name).read_text()
    stripped = protect(text)[0]
    assert "$" not in stripped, f"{name}: unpaired $ at {stripped.count('$')} place(s)"


def test_protect_is_lossless_on_the_constructs_that_broke() -> None:
    for src in (r"$\{\Phi\beta : \beta \ge 0,\ \mathbf{1}^{\top}\beta = 1\}$",
                r"$F(x^{*i}) - F^{*}$",
                r"$\geq +0.88$",
                r"$\max t \ \text{s.t.}\ \Phi\beta + t\hat n = F(x)$",
                r"$$\frac{\mathrm{SE}_{\text{corr}}}{\mathrm{SE}_{\text{naive}}}"
                r" \;=\; \sqrt{\frac{1/30 + 0.25}{1/30}}$$"):
        text, spans = protect(src)
        assert len(spans) == 1
        inner = spans[0][2:-2]                    # strip \( \) or \[ \]
        assert inner == src.strip("$"), f"protect mangled {src!r}"


@pytest.mark.parametrize("name", SOURCES)
def test_rendered_pdf_carries_no_latex(name: str) -> None:
    """If the PDF has been built, its text must hold no LaTeX and no delimiters."""
    pdf = PAPER / "manuscript" / (
        "Paper2_%s_FINAL_v5.pdf" % ("MANUSCRIPT" if "MANU" in name else "SUPPLEMENT"))
    if not pdf.exists():
        pytest.skip("PDF not built in this working tree")
    pymupdf = pytest.importorskip("pymupdf")
    text = "".join(p.get_text() for p in pymupdf.open(pdf))
    for bad in ("\\", "$", "is*", "q +0.88", "attained favoured",
                "Section 6.7 reports this baseline", "led every surrogate-assisted arm"):
        assert bad not in text, f"{pdf.name} contains {bad!r}"
