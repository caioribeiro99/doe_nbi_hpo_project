"""Assemble the Applied Soft Computing submission package.

Declarations, title and abstract are EXTRACTED from the manuscript at
paper2-manuscript-v5, never retyped. Author metadata is written only where an
authoritative source was checked; anything unverified is emitted as an explicit
AUTHOR INPUT REQUIRED marker rather than guessed.

ORCID iDs below were each resolved against the ORCID public API, and two of the three
were independently corroborated by Crossref on a co-authored paper. The verification is
re-run by tests/methodology/test_asoc_package.py, not taken on trust from this comment.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
PAPER = "papers/xgboost_hpo_vrfnbi"
TAG = "paper2-manuscript-v5"
OUT = REPO / PAPER / "submission"

MARK = "**AUTHOR INPUT REQUIRED**"

# (name, orcid, verification, affiliation, affiliation source)
AUTHORS = [
    ("Caio Tertuliano Ribeiro", "0009-0006-7748-1449",
     "ORCID public API: registered name 'Caio Tertuliano Ribeiro' — exact match",
     "Universidade Federal de Itajubá, Itajubá, Brazil",
     "ORCID employment record (department recorded there as 'NOMATI' — confirm whether "
     "this is the unit to print)"),
    ("Matheus Costa Pereira", "0009-0007-2011-9235",
     "ORCID public API: 'Matheus Costa Pereira'; independently corroborated by Crossref "
     "on doi:10.1016/j.engappai.2025.112510",
     "Federal University of Itajubá, Itajubá, Brazil",
     "ORCID employment record: Industrial Engineering Institute"),
    ("Anderson Paulo de Paiva", "0000-0002-8199-411X",
     "ORCID public API: 'Anderson Paulo de Paiva' (credit name 'Paiva, A. P.'); "
     "independently corroborated by Crossref on doi:10.1016/j.engappai.2025.112510",
     MARK, "no public employment record on ORCID; not recorded in this repository"),
]

HIGHLIGHTS = [
    "Scalarization, front geometry and anchor provenance separated in one pipeline",
    "Normal Boundary Intersection raised hypervolume on all three primary datasets",
    "Pre-specified real anchors gave no benefit and degraded two of three fronts",
    "A frozen-budget coarse grid led on the paired statistic on all four datasets",
    "Protocol frozen before any comparative result existed; 30 replications per set",
]


def at_tag(rel: str) -> str:
    return subprocess.run(["git", "show", f"{TAG}:{PAPER}/{rel}"], cwd=REPO,
                          capture_output=True, text=True, check=True).stdout


def block(text: str, label: str) -> str:
    start = text.index(f"**{label}**")
    return text[start:text.index("\n\n", start)].strip()


def main() -> int:
    man = at_tag("manuscript/MANUSCRIPT.md")
    live = (REPO / PAPER / "manuscript" / "MANUSCRIPT.md").read_text()
    title = man.splitlines()[0].lstrip("# ").strip()
    abstract = man[man.index("## Abstract") + 11:man.index("**Keywords:**")].strip()
    keywords = block(live, "Keywords:")          # the ASOC-compliant seven
    credit = man[man.index("**Author contributions (CRediT).**"):man.index("\n\n---", man.index("**Author contributions (CRediT).**"))].strip()

    OUT.mkdir(exist_ok=True)

    # ---- title page (ASOC is single anonymized: authors stay visible) -------------
    t = ["# Title page\n", f"## Title\n\n{title}\n", "## Authors\n"]
    for i, (name, orcid, ver, aff, affsrc) in enumerate(AUTHORS, 1):
        t.append(f"**{i}. {name}**  \n"
                 f"ORCID: `{orcid}` — {ver}  \n"
                 f"Affiliation: {aff}  \n"
                 f"*(affiliation source: {affsrc})*\n")
    t += ["## Corresponding author\n",
          "Caio Tertuliano Ribeiro.\n",
          f"Email: {MARK} — the institutional address has not been verified. The only "
          "address in the project record is `caio.tertu99@gmail.com` (CITATION.cff, "
          "pyproject.toml), which is personal.\n",
          f"Full postal address and telephone: {MARK} — the submission checklist "
          "requires \"full contact details (email address, full postal address and "
          "phone numbers)\"; none is recorded in this repository.\n",
          f"## Keywords\n\n{keywords}\n",
          block(man, "Competing interests.") + "\n",
          block(man, "Funding.") + "\n",
          block(man, "Acknowledgements.") + "\n",
          credit + "\n"]
    (OUT / "ASOC_TITLE_PAGE.md").write_text("\n".join(t))

    # ---- highlights ---------------------------------------------------------------
    over = [h for h in HIGHLIGHTS if len(h) > 85]
    h = ["# Highlights\n",
         "Applied Soft Computing: optional, 3-5 bullets, maximum 85 characters each "
         "including spaces. Upload as a separate editable file.\n"]
    for x in HIGHLIGHTS:
        h.append(f"- {x}  *({len(x)} characters)*")
    (OUT / "ASOC_HIGHLIGHTS.md").write_text("\n".join(h) + "\n")

    # ---- cover letter -------------------------------------------------------------
    c = f"""# Cover letter

Dear Editor,

We submit *{title}* for consideration as a research article in Applied Soft Computing.

The paper separates three choices that surrogate-assisted multiobjective hyperparameter
optimization pipelines normally change together: the scalarization specification, the
geometry by which a Pareto front is constructed, and the provenance of the anchors and
payoff matrix that geometry uses. Because a pipeline revision typically moves all three
at once, a reported gain cannot be attributed to any one of them. We hold the surrogate
model, scaling, decision space, candidate realization and budget fixed and vary one
mechanism at a time, under a protocol frozen before any comparative result existed, with
30 replicated outer partitions per dataset and every returned candidate re-evaluated on
the real learner.

Relevance to Applied Soft Computing: the work sits in two areas the journal names
explicitly — Machine and Deep Learning, since the optimized learner is a gradient-boosted
decision-tree model, and Multi-objective Optimization, which is the object of the study.
Evolutionary computing enters as an NSGA-II comparator rather than as the contribution,
and we state that plainly rather than overclaiming it.

We also report results that do not favour our own method line. The anchor-provenance
mechanism gave no benefit and degraded the front on two of three datasets, and a
frozen-budget coarse grid led the surrogate-assisted arms on the paired statistic on all
four datasets. Both are reported in full, with the budget asymmetry that qualifies the
second stated alongside it.

The protocol, the analysis code and every committed artifact are version controlled and
tagged, and the manuscript states the tags. A prospectively designated boundary dataset
that did not resolve is reported as unresolved rather than omitted.

The work is original, is not under consideration elsewhere, and has not been published
previously. There is no prior conference version.

Yours sincerely,

Caio Tertuliano Ribeiro, on behalf of all authors
"""
    (OUT / "ASOC_COVER_LETTER.md").write_text(c)

    print("wrote:")
    for f in sorted(OUT.glob("ASOC_*")):
        print(f"  {f.name}")
    print(f"  highlights over 85 characters: {over or 'none'}")
    print(f"  AUTHOR INPUT REQUIRED markers on the title page: "
          f"{(OUT / 'ASOC_TITLE_PAGE.md').read_text().count(MARK)}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
