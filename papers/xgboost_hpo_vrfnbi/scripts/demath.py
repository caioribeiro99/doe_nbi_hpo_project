"""Convert the manuscript's inline LaTeX to Unicode for HTML/PDF rendering.

The rendering path has no MathJax, so inline LaTeX would print as literal command
text. The manuscript SOURCE keeps LaTeX, so a LaTeX submission path stays available;
only the rendered PDF is substituted.

Literal string replacement, not regex: every entry below is a fixed LaTeX command and
regex escaping buys nothing here except a class of bugs.
"""
from __future__ import annotations

import re

# Order matters: longer commands first, so \\rightarrow is not eaten by \\to.
MATH: list[tuple[str, str]] = [
    (r"\operatorname{diag}", "diag"), (r"\mathrm{diag}", "diag"),
    (r"\text{test}", "test"), (r"\text{train}", "train"),
    (r"\text{cost}", "cost"), (r"\text{quality}", "quality"),
    (r"\text{corr}", "corr"), (r"\text{naive}", "naive"),
    (r"\rightarrow", " \u2192 "), (r"\leftarrow", " \u2190 "),
    (r"\lfloor", "\u230a"), (r"\rfloor", "\u230b"),
    (r"\lVert", "\u2016"), (r"\rVert", "\u2016"),
    (r"\Lambda", "\u039b"), (r"\lambda", "\u03bb"),
    (r"\Phi", "\u03a6"), (r"\phi", "\u03c6"),
    (r"\beta", "\u03b2"), (r"\sigma", "\u03c3"), (r"\rho", "\u03c1"),
    (r"\alpha", "\u03b1"), (r"\mu", "\u03bc"), (r"\epsilon", "\u03b5"),
    (r"\approx", " \u2248 "), (r"\times", "\u00d7"),
    (r"\ge", " \u2265 "), (r"\le", " \u2264 "), (r"\neq", " \u2260 "),
    (r"\pm", "\u00b1"), (r"\cdot", "\u00b7"), (r"\sum", "\u2211"),
    (r"\sqrt", "\u221a"), (r"\to", " \u2192 "),
    (r"\max", "max"), (r"\min", "min"), (r"\log", "log"),
    (r"\hat{n}", "n\u0302"), (r"\tilde{f}", "f\u0303"),
    (r"\mathbf", ""), (r"\mathrm", ""), (r"\hat", ""), (r"\bar", ""),
    (r"\text", ""), (r"\,", " "), (r"\;", " "), (r"\!", ""),
    (r"\tilde", "~"), (r"\in", " \u2208 "), (r"\leftarrow", " \u2190 "),
    (r"\subseteq", " \u2286 "), (r"\infty", "\u221e"), (r"\prime", "'"),
]

_SUP = str.maketrans("0123456789+-n", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076"
                                      "\u2077\u2078\u2079\u207a\u207b\u207f")


def _group(body: str, i: int) -> tuple[str, int]:
    """Read the brace group starting at body[i] == '{'; return its contents and the
    index just past its closing brace. A depth counter handles any nesting, which a
    regex cannot: \\frac{\\mathrm{SE}_{\\text{corr}}}{...} is three levels deep."""
    depth, j = 0, i
    while j < len(body):
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
            if depth == 0:
                return body[i + 1:j], j + 1
        j += 1
    return body[i + 1:], len(body)      # unbalanced source: take the rest


def _frac(body: str) -> str:
    """\\frac{a}{b} -> (a)/(b), innermost-last so nested fractions resolve."""
    while (k := body.find("\\frac")) != -1:
        i = k + len("\\frac")
        if i >= len(body) or body[i] != "{":
            break                        # not a fraction we can parse; leave it
        num, i = _group(body, i)
        if i >= len(body) or body[i] != "{":
            break
        den, i = _group(body, i)
        body = body[:k] + f"({_frac(num)})/({_frac(den)})" + body[i:]
    return body


def _sqrt(body: str) -> str:
    """\\sqrt{a} -> \u221aa, parenthesised when the radicand is not a single atom.

    Without the parentheses \\sqrt{\\frac{a}{b}} flattens to \u221aa/b, which a reader
    parses as (\u221aa)/b \u2014 a different number from the one the manuscript states.
    """
    while (k := body.find("\\sqrt")) != -1:
        i = k + len("\\sqrt")
        if i >= len(body) or body[i] != "{":
            break                        # brace-less use falls through to MATH
        arg, i = _group(body, i)
        arg = _sqrt(_frac(arg))
        atom = re.fullmatch(r"[\w.\\]+(?:_\{?\w+\}?)?", arg) is not None
        body = body[:k] + ("\u221a" + arg if atom else "\u221a(" + arg + ")") + body[i:]
    return body


def _supers(body: str) -> str:
    """x^{2} and x^2 -> superscript where the exponent is simple."""
    def one(m: re.Match) -> str:
        e = m.group(1) or m.group(2)
        return e.translate(_SUP) if all(c in "0123456789+-n" for c in e) else f"^{e}"
    return re.sub(r"\^\{([^{}]+)\}|\^(\w)", one, body)


def _convert(body: str) -> str:
    body = _sqrt(_frac(body))
    for pat, rep in MATH:
        body = body.replace(pat, rep)
    body = _supers(body)
    body = re.sub(r"_\{([^{}]+)\}", r"_\1", body)
    return body.replace("{", "").replace("}", "")


def demath(text: str) -> str:
    """Strip $...$ and $$...$$ delimiters, converting the contents to Unicode."""
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert(m.group(1)), text, flags=re.S)
    # inline math may wrap across a single line break in the markdown source
    return re.sub(r"\$([^$]{1,200}?)\$",
                  lambda m: _convert(m.group(1).replace("\n", " ")), text)
