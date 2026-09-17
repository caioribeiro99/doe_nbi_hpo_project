"""Hold math out of markdown's reach, then hand it to a real typesetter.

Two independent things were corrupting the rendered mathematics.

The first is markdown. It runs over the whole document before anything looks at
``$...$``, so the ``*`` in ``$F(x^{*i}) - F^{*}$`` pairs with the next ``*`` on the line
and becomes an emphasis run: the PDF printed "is*" where a superscript belonged. The
same happens to ``_`` in subscripts. Escaping is not a fix, because the escape then
reaches the typesetter. The fix is to lift every math span out before markdown sees it
and put it back afterwards, byte for byte.

The second was converting LaTeX to Unicode by string replacement. That approach cannot
represent ``\\{\\Phi\\beta : \\beta \\ge 0,\\ \\mathbf{1}^{\\top}\\beta = 1\\}`` and fails
quietly rather than loudly -- ``\\geq`` matched the ``\\ge`` entry and left a stray "q"
behind. The mathematics is valid LaTeX, so it is typeset as LaTeX.
"""
from __future__ import annotations

import re

# Display math first: a $$...$$ span would otherwise be seen as two empty $...$ spans.
DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<!\$)\$([^$]+?)\$(?!\$)")

# A sentinel markdown will not touch: no punctuation it assigns meaning to, and no
# character that could begin an emphasis, a link, a code span or an entity.
_TOKEN = "zZmathspanZz%04dzZ"
_TOKEN_RE = re.compile(r"zZmathspanZz(\d{4})zZ")


def protect(text: str) -> tuple[str, list[str]]:
    """Replace every math span with an opaque token. Returns the text and the spans."""
    spans: list[str] = []

    def take(m: re.Match, wrapper: str) -> str:
        spans.append(wrapper % m.group(1))
        return _TOKEN % (len(spans) - 1)

    text = DISPLAY.sub(lambda m: take(m, r"\[%s\]"), text)
    text = INLINE.sub(lambda m: take(m, r"\(%s\)"), text)
    return text, spans


def restore(html: str, spans: list[str]) -> str:
    """Put the math back, unmodified, after markdown has run."""
    def put(m: re.Match) -> str:
        return spans[int(m.group(1))]
    out = _TOKEN_RE.sub(put, html)
    left = _TOKEN_RE.findall(out)
    if left:
        raise AssertionError(f"math tokens survived restoration: {left[:5]}")
    return out


def unrestored(html: str) -> list[str]:
    """Any token still present is a math span markdown swallowed or mangled."""
    return _TOKEN_RE.findall(html)
