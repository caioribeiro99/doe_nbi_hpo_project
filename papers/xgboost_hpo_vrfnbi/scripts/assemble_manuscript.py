#!/usr/bin/env python
"""Assemble the manuscript from its sections, in the frozen order."""
from __future__ import annotations
import pathlib, re, sys
PAPER = pathlib.Path(__file__).resolve().parents[1]
SEC = PAPER / "manuscript" / "sections"
OUT = PAPER / "manuscript" / "MANUSCRIPT.md"
ORDER = ["00_title_abstract.md", "01_introduction.md", "02_related_work.md",
         "03_historical_reconstruction.md", "04_research_questions.md",
         "05a_methods_design.md", "05b_methods_arms.md",
         "06a_results_integrity.md", "06b_results_contrasts.md",
         "06c_results_boundary_baselines.md", "07_discussion.md",
         "08_limitations.md", "09_reproducibility.md", "10_conclusion.md",
         "11_references_declarations.md",
         "12_figure_captions.md"]

def main() -> int:
    missing = [f for f in ORDER if not (SEC / f).exists()]
    parts = []
    for f in ORDER:
        p = SEC / f
        if p.exists():
            parts.append(p.read_text().rstrip() + "\n")
    text = "\n\n".join(parts)
    OUT.write_text(text)
    words = len(text.split())
    figs = sorted(set(re.findall(r"\[FIG:([A-Za-z0-9_\-]+)\]", text)))
    tabs = sorted(set(re.findall(r"\[TAB:([A-Za-z0-9_\-]+)\]", text)))
    print(f"assembled {len(parts)} sections -> {OUT}")
    print(f"  words {words:,}   ~pages at 600 w/p: {words/600:.1f}")
    if missing:
        print(f"  MISSING: {missing}")
    print(f"  figure placeholders ({len(figs)}): {figs}")
    print(f"  table placeholders  ({len(tabs)}): {tabs}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
