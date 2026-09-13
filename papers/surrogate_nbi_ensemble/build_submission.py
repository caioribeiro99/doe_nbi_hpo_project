#!/usr/bin/env python
"""Assemble submission_eaai/ from the live manuscript sources.

The package was previously copied by hand, which let the frozen PDF and the live
sources drift apart. This rebuilds it from one command so the tagged submission and
the repository agree by construction.

Produces, under submission_eaai/:
    manuscript.pdf              compiled main text
    supplementary.pdf           compiled supplementary material
    main.tex, sections/, tables/, figures/, references.bib
    supplementary/              supplementary source tree
    highlights.txt, cover_letter.md and the editorial declarations

Usage:  python build_submission.py [--skip-compile]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission_eaai"
BUILD = Path("/tmp/texout")

# Editorial documents that live only in the submission package and are edited by hand.
DECLARATIONS = [
    "highlights.txt",
    "cover_letter.md",
    "author_contributions.md",
    "conflict_of_interest.md",
    "data_availability.md",
    "code_availability.md",
    "response_to_pre_submission_review.md",
    "self_overlap_assessment.md",
]


def compile_pdf(tex: Path) -> Path:
    """Compile with tectonic and return the produced PDF."""
    r = subprocess.run(["tectonic", "-o", str(BUILD), tex.name],
                       cwd=tex.parent, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"tectonic failed on {tex}")
    bad = [ln for ln in r.stderr.splitlines() if "Overfull" in ln or "Undefined" in ln]
    print(f"  compiled {tex.name}: {len(bad)} overfull/undefined warnings")
    return BUILD / (tex.stem + ".pdf")


def sync_dir(src: Path, dst: Path, suffixes: tuple[str, ...]) -> int:
    """Mirror src into dst for the given suffixes, deleting files src no longer has."""
    dst.mkdir(parents=True, exist_ok=True)
    wanted = {p.name for p in src.iterdir() if p.suffix in suffixes}
    for p in dst.iterdir():
        if p.is_file() and p.name not in wanted:
            p.unlink()
    for name in sorted(wanted):
        shutil.copy2(src / name, dst / name)
    return len(wanted)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    print("building submission_eaai/")

    if not args.skip_compile:
        main_pdf = compile_pdf(HERE / "main.tex")
        supp_pdf = compile_pdf(HERE / "supplementary" / "supplementary.tex")
    else:
        main_pdf, supp_pdf = BUILD / "main.pdf", BUILD / "supplementary.pdf"

    shutil.copy2(main_pdf, OUT / "manuscript.pdf")
    shutil.copy2(supp_pdf, OUT / "supplementary.pdf")
    print(f"  manuscript.pdf    {(OUT / 'manuscript.pdf').stat().st_size / 1024:,.0f} KiB")
    print(f"  supplementary.pdf {(OUT / 'supplementary.pdf').stat().st_size / 1024:,.0f} KiB")

    shutil.copy2(HERE / "main.tex", OUT / "main.tex")
    shutil.copy2(HERE / "references.bib", OUT / "references.bib")
    n_sec = sync_dir(HERE / "sections", OUT / "sections", (".tex",))
    # .csv alongside .tex: the per-replication edge-condition data and the derived
    # compute accounting support Tables 5 and 7 and travel with the submission.
    n_tab = sync_dir(HERE / "tables", OUT / "tables", (".tex", ".csv"))
    n_fig = sync_dir(HERE / "figures", OUT / "figures", (".pdf", ".png"))
    print(f"  main source: {n_sec} sections, {n_tab} tables, {n_fig} figures")

    sup_out = OUT / "supplementary"
    sup_out.mkdir(exist_ok=True)
    shutil.copy2(HERE / "supplementary" / "supplementary.tex", sup_out / "supplementary.tex")
    s_tab = sync_dir(HERE / "supplementary" / "tables", sup_out / "tables", (".tex",))
    s_fig = sync_dir(HERE / "supplementary" / "figures", sup_out / "figures", (".pdf", ".png"))
    print(f"  supplementary source: {s_tab} tables, {s_fig} figures")

    missing = [d for d in DECLARATIONS if not (OUT / d).exists()]
    if missing:
        raise SystemExit(f"missing editorial document(s): {missing}")
    print(f"  editorial documents present: {len(DECLARATIONS)}")

    # reproducibility_statement.tex mirrors the manuscript section; keep them identical
    shutil.copy2(HERE / "sections" / "09_reproducibility.tex", OUT / "reproducibility_statement.tex")
    print("  reproducibility_statement.tex synced from sections/09_reproducibility.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
