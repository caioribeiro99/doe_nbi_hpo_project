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
CONFIRM = "**AUTHOR CONFIRMATION REQUIRED**"

# Author-confirmed 2026-09-21. The journal requires only "the email address of each
# author" -- it does not ask for an institutional address -- so this is the address the
# author chose, not a fallback. No @unifei.edu.br address was guessed or substituted.
CORRESPONDING_EMAIL = "caio.tertu@hotmail.com"

# One institutional address, supplied and verified by the author from UNIFEI SIGAA.
IEPG = ("Institute of Production Engineering and Management (IEPG), "
        "Federal University of Itajubá (UNIFEI), Av. BPS 1303, Pinheirinho, "
        "Itajubá, MG 37500-903, Brazil")
# Expansion taken from the predecessor publication's own affiliation string, not coined
# here: Pereira et al. (2025), doi:10.1016/j.engappai.2025.112510.
NOMATI = ("Nucleus of Manufacturing Optimization and Innovation Technology (NOMATI), "
          "Itajubá, MG, Brazil")

# (name, orcid, orcid evidence, [affiliations], affiliation evidence, vitae draft)
AUTHORS = [
    ("Caio Tertuliano Ribeiro", "0009-0006-7748-1449",
     "ORCID public API: registered name 'Caio Tertuliano Ribeiro' — exact match",
     [IEPG, NOMATI],
     "IEPG is the author's stated primary academic affiliation; ORCID records "
     "Universidade Federal de Itajubá with NOMATI as the unit. NOMATI is carried as a "
     "second research-group affiliation because the predecessor publication "
     "(doi:10.1016/j.engappai.2025.112510) lists exactly this pair for this author.",
     "Caio Tertuliano Ribeiro holds a master's degree from the Institute of Production "
     "Engineering and Management at the Federal University of Itajubá, where his "
     "dissertation applied design of experiments and Normal Boundary Intersection to "
     "hyperparameter optimization of gradient-boosted decision trees. He is a member of "
     "the Nucleus of Manufacturing Optimization and Innovation Technology (NOMATI) and a "
     "co-author of work on hybrid multivariate Normal Boundary Intersection published in "
     "Engineering Applications of Artificial Intelligence. His research concerns "
     "replicated, protocol-frozen evaluation of multiobjective optimization pipelines for "
     "machine learning."),
    ("Matheus Costa Pereira", "0009-0007-2011-9235",
     "ORCID public API: 'Matheus Costa Pereira'; independently corroborated by Crossref "
     "on doi:10.1016/j.engappai.2025.112510",
     [IEPG, NOMATI],
     "ORCID employment record gives the Industrial Engineering Institute at the Federal "
     "University of Itajubá, the English rendering of IEPG; the predecessor publication "
     "lists UNIFEI with NOMATI for this author. A 2026 publication also lists a second "
     "affiliation at the University of Melbourne; it is NOT carried here, because "
     "Elsevier asks for the affiliation where the work was performed and that has not "
     "been established for this manuscript.",
     "Matheus Costa Pereira is a researcher at the Institute of Production Engineering "
     "and Management, Federal University of Itajubá, and a member of the Nucleus of "
     "Manufacturing Optimization and Innovation Technology (NOMATI). He is the first "
     "author of work on a hybrid multivariate Normal Boundary Intersection approach with "
     "post-optimization assisted by mixture design of experiments, published in "
     "Engineering Applications of Artificial Intelligence. His research interests include "
     "multiobjective optimization, response surface methodology and multivariate "
     "statistical methods for engineering and machine-learning problems."),
    ("Anderson Paulo de Paiva", "0000-0002-8199-411X",
     "ORCID public API: 'Anderson Paulo de Paiva' (credit name 'Paiva, A. P.'; Scopus "
     "and ResearcherID present); independently corroborated by Crossref on "
     "doi:10.1016/j.engappai.2025.112510",
     [IEPG, NOMATI],
     "Verified by the author against UNIFEI SIGAA: Instituto de Engenharia de Produção e "
     "Gestão, andersonppaiva@unifei.edu.br. The predecessor publication lists UNIFEI with "
     "NOMATI for this author.",
     "Anderson Paulo de Paiva is a professor at the Institute of Production Engineering "
     "and Management, Federal University of Itajubá, and a member of the Nucleus of "
     "Manufacturing Optimization and Innovation Technology (NOMATI). He is the senior "
     "author of the Normal Boundary Intersection and variance-reduction-factor "
     "methodological line, including work published in The International Journal of "
     "Advanced Manufacturing Technology, the Journal of Cleaner Production and "
     "Engineering Applications of Artificial Intelligence. His research concerns "
     "multiobjective optimization, design of experiments and multivariate statistics "
     "applied to manufacturing and process engineering."),
]

# Elsevier generative-AI policy, page last updated June 2026, retrieved live 2026-09-21.
# Section title and template are the publisher's current wording, not the 2024 capture,
# which said "writing process". The policy explicitly does NOT cover AI used in research
# methodology -- XGBoost, NSGA-II, Bayesian optimization and TPE belong in the Methods and
# get no declaration.
AI_DECLARATION = (
    "During the preparation of this work, the authors used Anthropic Claude and OpenAI "
    "ChatGPT in order to assist with manuscript drafting, language refinement, "
    "consistency checking and editorial review. After using these tools, the authors "
    "reviewed and edited the content as needed and take full responsibility for the "
    "content of the published article.\n\n"
    "This declaration concerns manuscript preparation only. The study's computational "
    "methods are described in the Methods section, and every reported result derives "
    "from the committed, version-controlled artifacts referenced there.")

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
    affs, order = [], {}
    for _, _, _, alist, _, _ in AUTHORS:
        for a in alist:
            if a not in order:
                order[a] = len(order) + 1
                affs.append(a)
    t = ["# Title page — Applied Soft Computing\n", f"## Title\n\n{title}\n",
         "## Authors\n"]
    for i, (name, orcid, ver, alist, affsrc, _) in enumerate(AUTHORS, 1):
        marks = ",".join(str(order[a]) for a in alist)
        star = " *(corresponding author)*" if i == 1 else ""
        t.append(f"**{name}**^{marks}^{star}  \n"
                 f"ORCID: `{orcid}`  \n"
                 f"*ORCID evidence: {ver}*  \n"
                 f"*Affiliation evidence: {affsrc}*\n")
    t.append("### Affiliations\n")
    for a in affs:
        t.append(f"^{order[a]}^ {a}\n")
    t += ["### Corresponding author\n",
          "**Caio Tertuliano Ribeiro**\n",
          f"Email: `{CORRESPONDING_EMAIL}` — author-confirmed.\n",
          "The journal does **not** require an institutional address. Its only stated "
          "rule is to provide \"the email address of each author\" and to keep the "
          "corresponding author's \"email address and contact details … up to date\", "
          "so this address satisfies it. No institutional address was guessed or "
          "substituted.\n",
          f"Postal address: {IEPG}\n",
          "Telephone: the 2024 submission checklist asks for \"full contact details "
          "(email address, full postal address and phone numbers)\". That rule is "
          "**ARCHIVE-ONLY** — the live Editorial Manager site for this journal states "
          "\"Site under development. Do not use for live manuscript submission\", so no "
          "live form could be inspected. Supply a number only if the live portal asks; "
          "the package is not blocked on it.\n",
          f"## Keywords\n\n{keywords}\n",
          block(man, "Competing interests.") + "\n",
          block(man, "Funding.") + "\n",
          block(man, "Acknowledgements.") + "\n",
          credit + "\n",
          "## Declaration of generative AI and AI-assisted technologies in the "
          "manuscript preparation process\n",
          AI_DECLARATION + "\n"]
    (OUT / "ASOC_TITLE_PAGE.md").write_text("\n".join(t))

    # ---- vitae -------------------------------------------------------------------
    v = ["# Vitae\n",
         "Applied Soft Computing asks for \"a short (maximum 100 words) biography of "
         "each author\" and \"a passport-type photograph as a separate figure\", "
         "provided in an editable format. **ARCHIVE-ONLY (2024-06-30) — recheck at "
         "submission.**\n",
         f"Every biography below is drafted from verified academic record only and is "
         f"marked {CONFIRM} until its author approves the wording.\n"]
    for name, orcid, _, alist, _, bio in AUTHORS:
        v.append(f"## {name}  \nORCID `{orcid}` — {CONFIRM}\n")
        v.append(f"{bio}\n")
        v.append(f"*({len(bio.split())} words; limit 100)*\n")
        v.append(f"**Photograph:** `PHOTO_PLACEHOLDER_{name.split()[0].lower()}.jpg` — "
                 "passport-type photograph to be supplied by the author. None has been "
                 "sourced or generated.\n")
    (OUT / "ASOC_VITAE.md").write_text("\n".join(v))

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
Corresponding author — {CORRESPONDING_EMAIL}
Institute of Production Engineering and Management (IEPG), Federal University of
Itajubá (UNIFEI), Av. BPS 1303, Pinheirinho, Itajubá, MG 37500-903, Brazil
"""
    (OUT / "ASOC_COVER_LETTER.md").write_text(c)

    # ---- checklist, with live-vs-archive status on every mutable rule -------------
    LIVE = "**VERIFIED LIVE 2026-09-21**"
    ARCH = "**ARCHIVE-ONLY (2024-06-30) — RECHECK AT SUBMISSION**"
    rows = [
        ("Generative-AI declaration: section title, template, placement",
         LIVE, "Elsevier policy page, last updated June 2026, fetched directly",
         "Done — in the manuscript before the references, and on the title page"),
        ("Generative AI not an author; not used for graphical-abstract artwork",
         LIVE, "same page", "Complies — abstract figure is matplotlib from committed data"),
        ("AI in research methodology needs no declaration",
         LIVE, "same page", "XGBoost, NSGA-II, BO, TPE described in Methods only"),
        ("Journal identity, ISSN, hybrid-subscription, submission route",
         LIVE, "core.submit.elsevier.com/core/v1/journals/ASOC returned 200 live",
         "ASOC, 1568-4946, HYBRID_SUBSCRIPTION, redirectToEm=false"),
        ("Editorial Manager is NOT the live submission route",
         LIVE, "editorialmanager.com/asoc: \"Site under development. Do not use for "
         "live manuscript submission\"", "No live form could be inspected"),
        ("Aims and scope", "**ARCHIVE 2026-02-03**", "journal home capture",
         "Scope gate accepted: DEFENSIBLE FIT WITH EDITORIAL RISK"),
        ("Keywords 1 to 7", ARCH, "guide capture", "Done — reduced 8 to 7"),
        ("Graphical abstract required; min 531 x 1328 px", ARCH, "guide capture",
         "Done — 1527 x 610 px, proportional"),
        ("Highlights optional; 3-5 bullets, max 85 characters", ARCH, "guide capture",
         "Done — 5 bullets, all within limit"),
        ("Single anonymized review; no blinded file", ARCH, "guide capture",
         "Authors visible on the title page; no anonymized manuscript prepared"),
        ("Vitae: 100-word biography and passport photograph per author", ARCH,
         "guide capture", "Biographies drafted; photographs are author-supplied"),
        ("Corresponding author contact details incl. phone numbers", ARCH,
         "submission checklist in the guide capture",
         f"Email `{CORRESPONDING_EMAIL}` and postal address supplied; phone only if "
         "the live portal asks"),
        ("No institutional-email requirement", ARCH, "guide capture: the rule is only "
         "\"the email address of each author\"", "Any valid address satisfies it"),
        ("Reference formatting flexible at submission", ARCH, "guide capture",
         "Author-date, consistent — compliant"),
        ("Research data Option C: deposit, cite and link", ARCH, "guide capture",
         "Statement drafted with a DOI placeholder; deposit pending"),
        ("No page limit, no file-size limit, no abstract word limit, no acronym ban",
         ARCH, "guide capture — searched exhaustively, none stated",
         "32 pp, 1.4 MB, 257-word abstract all unconstrained"),
    ]
    c = ["# Applied Soft Computing — submission checklist\n",
         "Every mutable journal rule carries its verification status. An archived rule is "
         "never silently promoted to a live one.\n",
         "| Requirement | Status | Evidence | This package |", "|---|---|---|---|"]
    for r in rows:
        c.append("| %s | %s | %s | %s |" % r)
    c.append("\n## Why some rules could not be verified live\n")
    c.append("ScienceDirect and elsevier.com return HTTP 403 to every automated route, "
             "this session's web-search budget is exhausted, and the Internet Archive "
             "holds no capture of this journal's Guide for Authors newer than "
             "2024-06-30 — re-checked on 2026-09-21 and still none. The live Editorial "
             "Manager site states it is not to be used for submission, so no live form "
             "exists to read. Elsevier's own policy pages ARE reachable, which is why "
             "the generative-AI rules could be verified live and the journal-specific "
             "formatting rules could not.\n")
    (OUT / "ASOC_SUBMISSION_CHECKLIST.md").write_text("\n".join(c))

    # ---- manifest ------------------------------------------------------------------
    import hashlib
    man_dir = REPO / PAPER / "manuscript"
    files = sorted(list(OUT.glob("ASOC_*")) +
                   list(man_dir.glob("Paper2_*_FINAL_v5.pdf")))
    mf = ["# Package manifest\n",
          "| file | bytes | sha256 |", "|---|---|---|"]
    for f in files:
        mf.append("| `%s` | %s | `%s` |" % (f.name, format(f.stat().st_size, ","),
                                            hashlib.sha256(f.read_bytes()).hexdigest()[:32]))
    mf.append("\nUpload roles: manuscript PDF as the main document; supplement as "
              "supplementary material; graphical abstract, highlights, title page, vitae "
              "and cover letter as separate files.")
    (OUT / "ASOC_PACKAGE_MANIFEST.md").write_text("\n".join(mf))

    print("wrote:")
    for f in sorted(OUT.glob("ASOC_*")):
        print(f"  {f.name}")
    print(f"  highlights over 85 characters: {over or 'none'}")
    tp = (OUT / "ASOC_TITLE_PAGE.md").read_text()
    print(f"  corresponding author: {CORRESPONDING_EMAIL}")
    print(f"  AUTHOR INPUT REQUIRED markers on the title page: {tp.count(MARK)}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
