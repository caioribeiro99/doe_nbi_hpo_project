#!/usr/bin/env python
"""Render the manuscript and supplement to PDF via HTML and headless Chrome."""
from __future__ import annotations
import base64, pathlib, re, subprocess, sys
import markdown

PAPER = pathlib.Path(__file__).resolve().parents[1]
MAN = PAPER/"manuscript"
FIG = MAN/"figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Times New Roman", Georgia, serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; max-width: 100%; }
h1 { font-size: 16pt; line-height: 1.25; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 16pt 0 5pt; border-bottom: 0.6pt solid #bbb;
     padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
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
"""


def render(md_path: pathlib.Path, out_pdf: pathlib.Path, title: str) -> None:
    text = md_path.read_text()
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
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
           f"<style>{CSS}</style></head><body>{html}</body></html>")
    tmp = out_pdf.with_suffix(".html")
    tmp.write_text(doc)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", tmp.as_uri()],
                   check=True, capture_output=True, timeout=180)
    print(f"  {out_pdf.name}  {out_pdf.stat().st_size/1024:.0f} KB")


def main() -> int:
    render(MAN/"MANUSCRIPT.md", MAN/"Paper2_MANUSCRIPT_FINAL_v2.pdf",
           "Separating Scalarization Specification, Pareto Geometry, and Anchor Provenance")
    render(MAN/"SUPPLEMENT.md", MAN/"Paper2_SUPPLEMENT_FINAL_v2.pdf",
           "Supplementary material")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
