#!/usr/bin/env python
"""Render the manuscript and supplement to PDF via HTML and headless Chrome.

Mathematics is typeset by MathJax, vendored under ``vendor/mathjax`` and inlined into
the page so the build needs no network and reproduces offline. The earlier path
converted LaTeX to Unicode by string replacement, which cannot express the CHIM
definition or a nested fraction and failed silently rather than loudly. Math spans are
lifted out before markdown runs -- see ``mathprotect`` for why -- and MathJax is given
the LaTeX unmodified.
"""
from __future__ import annotations
import base64, pathlib, re, subprocess, sys
import markdown
from mathprotect import protect, restore, unrestored

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
PAPER = pathlib.Path(__file__).resolve().parents[1]
MAN = PAPER/"manuscript"
FIG = MAN/"figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MATHJAX = PAPER/"vendor"/"mathjax"/"tex-svg.js"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 0.6pt solid #bbb;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 10.5pt; font-style: italic; font-weight: 650;
     margin: 10pt 0 3pt; page-break-after: avoid; }
p { margin: 0 0 7pt; text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 7pt 0;
        page-break-inside: avoid; }
th, td { border: 0.4pt solid #bbb; padding: 2.5pt 4pt; text-align: left;
         vertical-align: top; word-break: normal; }
th { background: #f0f2f5; font-weight: 600; }
td:nth-child(n+3) { text-align: right; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.5pt;
       background: #f4f5f7; padding: 0.5pt 2pt; border-radius: 2pt; }
img { max-width: 100%; display: block; margin: 8pt auto; page-break-inside: avoid; }
blockquote { border-left: 2pt solid #ccc; margin: 7pt 0; padding: 2pt 0 2pt 9pt;
             color: #333; }
strong { font-weight: 650; }
h4 + p, h3 + p, h2 + p { margin-top: 0; }
mjx-container { font-size: 100% !important; }
mjx-container[display="true"] { margin: 9pt 0 !important; page-break-inside: avoid; }
mjx-container svg { vertical-align: baseline; }
"""

# MathJax is told the delimiters mathprotect emits. Local font caching keeps every
# expression self-contained: a global cache puts shared glyph definitions in one element
# at the end of the document, and <use> references into it are fragile once the print
# path paginates.
MATHJAX_CONFIG = """
window.MathJax = {
  tex: { inlineMath: [['\\\\(', '\\\\)']], displayMath: [['\\\\[', '\\\\]']],
         packages: {'[+]': ['ams']} },
  svg: { fontCache: 'local', scale: 1.0 },
  options: { enableMenu: false },
  startup: { typeset: true }
};
"""


def render(md_path: pathlib.Path, out_pdf: pathlib.Path, title: str) -> None:
    text, spans = protect(md_path.read_text())

    # inline every figure the text names, as a data URI
    def fig_for(n: str):
        hits = sorted(FIG.glob(f"fig{n}_*.png"))
        return hits[0] if hits else None
    for n in sorted({m for m in re.findall(r"\*\*Figure (\d)\.", text)}):
        f = fig_for(n)
        if not f:
            continue
        b64 = base64.b64encode(f.read_bytes()).decode()
        text = text.replace(f"**Figure {n}.",
                            f'<img src="data:image/png;base64,{b64}"/>\n\n**Figure {n}.', 1)

    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    swallowed = unrestored(html)
    html = restore(html, spans)
    if swallowed and len(swallowed) != len(spans):
        pass                                   # restore() raises on anything it cannot place
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
           f"<style>{CSS}</style>"
           f"<script>{MATHJAX_CONFIG}</script>"
           f"<script>{MATHJAX.read_text()}</script>"
           f"</head><body>{html}</body></html>")
    tmp = out_pdf.with_suffix(".html")
    tmp.write_text(doc)
    # --virtual-time-budget lets MathJax finish typesetting before the page is printed;
    # without it Chrome prints the raw \( ... \) source.
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    "--virtual-time-budget=60000", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={out_pdf}", tmp.as_uri()],
                   check=True, capture_output=True, timeout=300)
    print(f"  {out_pdf.name}  {out_pdf.stat().st_size/1024:.0f} KB  "
          f"({len(spans)} math spans)")


def main() -> int:
    render(MAN/"MANUSCRIPT.md", MAN/"Paper2_MANUSCRIPT_FINAL_v4.pdf",
           "Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance")
    render(MAN/"SUPPLEMENT.md", MAN/"Paper2_SUPPLEMENT_FINAL_v4.pdf",
           "Supplementary material")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
