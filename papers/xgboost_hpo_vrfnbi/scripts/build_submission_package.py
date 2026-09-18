"""Assemble the journal submission package by EXTRACTION from paper2-manuscript-v5.

Nothing here is retyped. Every declaration, the title, the abstract and the keywords are
read out of the manuscript at the frozen tag with ``git show``, so the submission package
cannot drift from the master even by a character. A retyped statement is a second source
of truth, and this project has already lost a funding number to exactly that.

The package is deliberately incomplete where the record is. Fields that no committed file
establishes -- the target journal, ORCIDs, co-author affiliations -- are emitted as
explicit AUTHOR INPUT REQUIRED rows rather than guessed. Inventing them is the failure
mode that cost v4 its freeze.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
PAPER = "papers/xgboost_hpo_vrfnbi"
MASTER_TAG = "paper2-manuscript-v5"
OUT = REPO / PAPER / "submission"


def at_tag(rel: str) -> str:
    return subprocess.run(["git", "show", f"{MASTER_TAG}:{PAPER}/{rel}"],
                          cwd=REPO, capture_output=True, text=True, check=True).stdout


def block(text: str, label: str) -> str:
    """One bold-labelled declaration paragraph, verbatim, terminator included."""
    start = text.index(f"**{label}**")
    end = text.index("\n\n", start)
    return text[start:end].strip()


def credit(text: str) -> str:
    start = text.index("**Author contributions (CRediT).**")
    end = text.index("\n\n---", start)
    return text[start:end].strip()


# What the committed record establishes, and what it does not. A row with source None is
# not a gap in this script -- it is a gap in the record, and it is reported as one.
VERIFIED: list[tuple[str, str, str]] = [
    ("First author", "Caio Tertuliano Ribeiro", "CITATION.cff; manuscript CRediT"),
    ("First author affiliation", "Federal University of Itajubá (UNIFEI)", "CITATION.cff"),
    ("First author email", "caio.tertu99@gmail.com",
     "CITATION.cff, pyproject.toml (personal address, not institutional)"),
    ("Second author", "Matheus Costa Pereira", "manuscript CRediT; Pereira et al. (2025)"),
    ("Third author", "Anderson Paulo de Paiva", "manuscript CRediT; research lineage"),
    ("Code repository", "https://github.com/caioribeiro99/doe_nbi_hpo_project", "CITATION.cff"),
    ("Code licence", "MIT", "CITATION.cff, LICENSE"),
    ("Master tag", MASTER_TAG, "this repository"),
]

REQUIRED: list[tuple[str, str]] = [
    ("Target journal", "No committed file designates one. Engineering Applications of "
     "Artificial Intelligence and Applied Soft Computing appear in "
     "protocol/NSGA2_DECISION.md only as reviewer communities used to justify including "
     "an NSGA-II baseline -- that is not a venue decision. Guidelines, template, "
     "highlights and graphical-abstract requirements all depend on this."),
    ("ORCID iDs", "Zero ORCIDs exist anywhere in this repository, for any author. "
     "An ORCID cannot be inferred from a name."),
    ("Affiliation, Matheus Costa Pereira", "Not recorded in any committed file. "
     "Inferring UNIFEI from the research lineage would be invention."),
    ("Affiliation, Anderson Paulo de Paiva", "Not recorded in any committed file."),
    ("Department / institute and postal address", "Not recorded for any author; most "
     "publishers require the full postal affiliation."),
    ("Corresponding author", "article/main.tex designates the first author for PAPER 1 "
     "with a personal gmail address. Nothing designates a corresponding author for "
     "Paper 2, and most publishers expect an institutional address."),
]


def main() -> int:
    man = at_tag("manuscript/MANUSCRIPT.md")
    OUT.mkdir(exist_ok=True)

    title = man.splitlines()[0].lstrip("# ").strip()
    abstract = man[man.index("## Abstract") + len("## Abstract"):man.index("**Keywords:**")].strip()
    keywords = block(man, "Keywords:")

    parts = {
        "Funding": block(man, "Funding."),
        "Acknowledgements": block(man, "Acknowledgements."),
        "Competing interests": block(man, "Competing interests."),
        "Data and code availability": block(man, "Data and code availability."),
        "Author contributions (CRediT)": credit(man),
    }

    w = [f"# Submission declarations — extracted from `{MASTER_TAG}`\n",
         "Every block below is reproduced verbatim from the manuscript at the frozen tag",
         "by `scripts/build_submission_package.py`. None is retyped. If a publisher form",
         "requires one of these fields, paste it from here rather than from memory.\n",
         f"## Title\n\n{title}\n",
         f"## Abstract\n\n{abstract}\n",
         f"## {keywords}\n"]
    for name, text in parts.items():
        w.append(f"## {name}\n\n{text}\n")
    (OUT / "DECLARATIONS.md").write_text("\n".join(w))

    m = ["# Submission metadata inventory\n",
         f"Master: `{MASTER_TAG}`. This file separates what the committed record",
         "establishes from what it does not. Nothing in the second table has been",
         "guessed.\n", "## Established by the record\n",
         "| field | value | source |", "|---|---|---|"]
    for f, v, s in VERIFIED:
        m.append(f"| {f} | {v} | {s} |")
    m += ["\n## AUTHOR INPUT REQUIRED — not established by any committed file\n",
          "| field | why it cannot be filled here |", "|---|---|"]
    for f, why in REQUIRED:
        m.append(f"| **{f}** | {why} |")
    m.append("\nThe submission cannot be completed while any row above is open. The "
             "target journal blocks the most: guidelines, template, section order, "
             "reference style, highlights and graphical abstract are all journal-specific.")
    (OUT / "SUBMISSION_METADATA.md").write_text("\n".join(m))

    # package manifest with checksums, so what is uploaded is what was frozen
    man_dir = REPO / PAPER / "manuscript"
    rows = []
    for f in sorted(man_dir.glob("Paper2_*_FINAL_v5.pdf")):
        import pymupdf
        rows.append((f.name, pymupdf.open(f).page_count, f.stat().st_size,
                     hashlib.sha256(f.read_bytes()).hexdigest()))
    p = ["# Package manifest\n", "| file | pages | bytes | sha256 |", "|---|---|---|---|"]
    for n, pages, size, h in rows:
        p.append(f"| `{n}` | {pages} | {size:,} | `{h}` |")
    p.append(f"\nFigures: 6 PNG, embedded in the manuscript PDF. Tables: 5, typeset in "
             f"the text. Manuscript body: {len(man.split()):,} words including references.")
    (OUT / "PACKAGE_MANIFEST.md").write_text("\n".join(p))

    print(f"wrote {OUT.relative_to(REPO)}/")
    for f in sorted(OUT.glob("*.md")):
        print(f"  {f.name:<28} {len(f.read_text().split()):>5} words")
    print(f"  extracted {len(parts)} declaration blocks verbatim from {MASTER_TAG}")
    print(f"  AUTHOR INPUT REQUIRED rows: {len(REQUIRED)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
