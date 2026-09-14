#!/usr/bin/env python
"""Fail if a blacklisted claim appears in the manuscript.

`novelty_matrix.md` lists claims this paper must not make. Paper 1 learned that a
checklist does not hold: a blacklisted phrasing was reintroduced by a later editing
pass and survived into the conclusion, and only a scan caught it. This script is
that scan, meant to run in the build rather than beside it.

It reads the compiled PDF if one is given, otherwise every .tex under the paper
directory, normalizes ligatures and whitespace, and reports each match with its
context.

Exit status is 1 if anything matched, so a build can gate on it.

Usage:
    python check_claim_blacklist.py [--pdf PATH] [--tex-dir DIR]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent

LIG = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
       "—": "-", "–": "-", "’": "'", "“": '"', "”": '"'}

# Each entry: (blacklist item in novelty_matrix.md, regex, why it is forbidden).
# Patterns are deliberately narrow: they target the claim, not the topic, so that
# discussing prior work does not trip them.
BLACKLIST: list[tuple[int, str, str]] = [
    (1, r"\bwe\s+(?:introduce|propose|present)\b[^.]{0,80}\b(?:normal boundary intersection|NBI)\b",
     "NBI is Das and Dennis (1998)"),
    (1, r"\bwe\s+(?:introduce|propose|present)\b[^.]{0,80}\bvarimax\b",
     "VRF objectives are the group's own prior work"),
    (2, r"\b(?:we are the first|first to|for the first time)\b[^.]{0,90}"
        r"\b(?:design of experiments|response surface|hyperparameter)\b",
     "DoE/RSM for hyperparameter tuning is Lujan-Moreno et al. (2018)"),
    (2, r"\bfirst\b[^.]{0,60}\bresponse surface\b[^.]{0,40}\bXGBoost\b",
     "Vasquez-Ramos et al. (2025) precedes this"),
    (3, r"\bwe\s+(?:introduce|propose)\b[^.]{0,80}\bmulti-?objective hyperparameter\b",
     "multiobjective HPO is an established field"),
    (4, r"\b(?:our|the)\s+contribution\s+is\b[^.]{0,60}\bapply(?:ing)?\b[^.]{0,50}"
        r"\b(?:NBI|normal boundary intersection)\b[^.]{0,40}\bhyperparameter\b",
     "applying an established method to a new domain is not the contribution"),
    (5, r"\bdissertation(?:'s)?\s+results?\s+(?:are|were)\s+(?:invalid|wrong|incorrect)\b",
     "the results are a weighted-sum method's results; only the name was wrong"),
    (6, r"\btheir\s+code\s+implements\b[^.]{0,40}\b(?:canonical\s+)?(?:NBI|normal boundary intersection)\b",
     "no implementation other than the two audited here has been examined"),
    (7, r"\bwe\s+(?:discover|discovered|reveal|revealed|uncover|uncovered)\b[^.]{0,80}"
        r"\bweighted[- ]sum\b",
     "the discrepancy was recorded by the author before the audit; it was confirmed, not discovered"),
    (8, r"\b(?:scandal|misconduct|concealed|cover[- ]?up|deliberately hidden|fraud)\b",
     "the discrepancy must not be framed as concealment"),
    (9, r"\bNBI\b[^.]{0,50}\b(?:produces|gives|yields)\b[^.]{0,30}\bmore uniformly spaced\b",
     "only claimable if measured here and found to hold"),
    (10, r"\b(?:no|not any)\s+(?:prior\s+)?work\s+(?:exists|has\s+ever)\b",
     "a bounded search establishes non-location, not absence"),
    # --- added after the objective-count adversarial review -------------------
    (11, r"\b(?:three|3)[- ]objectives?\b[^.]{0,90}\b(?:because|so that|in order to)\b"
         r"[^.]{0,60}\b(?:NBI|normal boundary intersection|simplex|CHIM)\b",
     "objective count is justified by structure in the data and the decision rule, "
     "never by the method it favours"),
    (12, r"\b(?:at|with)\s+(?:three|3)\s+objectives?\b[^.]{0,70}\bNBI\b[^.]{0,50}"
         r"\b(?:advantage|superior\w*|outperform\w*|wins?|better\s+spread)\b",
     "the simplex is where an NBI advantage is hypothesised, not where it is established"),
    (13, r"\bnon-?dominated\b[^.]{0,70}\b(?:grew|grows|growth|increased?|expand\w*)\b"
         r"[^.]{0,70}\b(?:shows?|demonstrat\w+|prov\w+|confirms?|establish\w+)\b",
     "adding any coordinate weakly enlarges a non-dominated set; the figure is "
     "claimable only as an excess over a declared null"),
    (14, r"\b(?:recover\w*|reveal\w*|uncover\w*|restor\w*)\b[^.]{0,60}"
         r"\b(?:suppressed|hidden|masked|collapsed)\b[^.]{0,50}"
         r"\b(?:trade-?off|axis|structure|front)\b",
     "recovery language asserts as established what only a null-referenced "
     "comparison supports, and it is false on one panel dataset"),
    (15, r"\bsurrogate\w*\b[^.]{0,80}"
         r"\b(?:reliable|adequate|validated|accurate|passes?\s+the\s+gate)\b[^.]{0,60}"
         r"\b(?:all|every|each|four|both)\s+(?:datasets?|objectives?|factors?)\b",
     "false at both objective counts: Spambase's quality surface already fails the gate"),
    (16, r"\b(?:pilot|stage\s*A)\b[^.]{0,70}"
         r"\b(?:confirms?|confirmed|establish\w+|demonstrat\w+|shows?|proves?)\b[^.]{0,60}"
         r"\b(?:surrogate|reliab\w+|valid\w+|fidelity)\b",
     "Stage A is one partition per dataset and is a measurement-validation pilot, "
     "not confirmatory evidence"),
    (16, r"\bthe\s+surrogate\s+is\s+reliable\b",
     "same; the permitted register names the screening, not the surrogate"),
    (17, r"\b(?:this|the)\s+(?:study|protocol|campaign|experiment|amendment)\s+"
         r"(?:was|is|were)\s+pre-?registered\b",
     "only what tag v1 froze before any measurement may be called pre-registered"),
    (17, r"\bprotocol\s+v(?:2|3)\b[^.]{0,50}\bpre-?registered\b",
     "v2 and v3 are prospectively specified and pilot-amended, not pre-registered"),
    (18, r"\b(?:optimiz\w+|minimiz\w+|target\w*)\s+(?:for\s+)?(?:the\s+)?"
         r"(?:training|computational|wall-?clock)\s+(?:time|cost)\b",
     "the optimized objective is a deterministic model-complexity proxy, not time"),
    (18, r"\b(?:our|the|second)\s+(?:cost\s+)?objective\s+(?:is|was)\s+(?:the\s+)?"
         r"(?:training|computational|wall-?clock)\s+(?:time|cost)\b",
     "same; measured time is a secondary audit variable"),
    (19, r"\b(?:the\s+)?(?:third\s+objective|second\s+quality\s+factor|objective\s+2)\b"
         r"[^.]{0,80}\b(?:across|on\s+all|every\s+dataset|the\s+panel|generally|consistently)\b",
     "the quality axes exchange roles across the panel, so objective 2 does not name "
     "the same quantity on any two datasets"),
]


# A claim stated as something to be tested is not a claim. Reporting a blacklisted
# phrase that the paper is about to measure, or attribute to someone else, is allowed;
# asserting it is not. This guard looks at the run-up to the match.
# A document that states a prohibition has to quote it. Matches whose run-up carries
# prohibition language are therefore not violations. This is narrow on purpose: it
# looks only at the immediately preceding text, so an author cannot license a claim
# by mentioning the word "forbidden" earlier in the paragraph.
PROHIBITION = re.compile(
    r"\b(?:must\s+not|may\s+not|never|forbidden|not\s+permitted|prohibit\w*|blacklist\w*|"
    r"do\s+not\s+(?:write|say|use|claim)|is\s+not:|are\s+not:|incorrect[,:]|wrong[,:]|"
    r"That\s+(?:this|the)\s+(?:work|study|paper)|forbids?|rules?\s+out)\b", re.I)

HEDGE = re.compile(
    r"\b(?:whether|if|tested|we\s+test|is\s+measured|was\s+measured|not\s+assumed|"
    r"do(?:es)?\s+not\s+claim|claims?\s+no|question\s+(?:of|whether)|ask(?:s|ed)?\s+whether)\b",
    re.I)


# Directories whose prose is subject to the claim rules. The workspace documents are
# included deliberately: a terminology rule that binds only the manuscript is a rule
# that is broken everywhere it is decided.
SCAN_SUFFIXES = (".tex", ".md")

# Documents whose job is to STATE the prohibitions necessarily quote them. They are
# listed here by name rather than detected, so that the exemption is auditable: a
# reader can see exactly which files are exempt and why, and adding one is a visible
# edit to this script rather than a phrase an author can drop into any paragraph.
QUOTING_DOCUMENTS = {
    "novelty_matrix.md": "contains the claim blacklist itself",
    "research_lineage.md": "contains the permitted/not-permitted statement pair",
    "PROTOCOL_AMENDMENTS.md": "quotes the terminology rules it establishes",
    "protocol/Q3_AMENDMENT_REVIEW.md": "quotes the claims it refuses",
    "protocol/COST_OBJECTIVE_CLAIM_BOUNDARY.md": "states the cost-terminology rule",
    "audits/METHODOLOGICAL_IDENTITY_AUDIT.md": "states what may and may not be written",
    "audits/PCA_VARIMAX_IDENTITY_AUDIT.md": "states what may and may not be written",
    "README.md": "restates the workspace rules, including the permitted/forbidden pair",
}


def load_text(pdf: Path | None, tex_dir: Path) -> tuple[str, str]:
    """Prefer the compiled manuscript; otherwise scan the workspace's own prose.

    The earlier version looked only for *.tex, of which there are none until the
    manuscript exists, so it exited 0 having scanned nothing while
    COST_OBJECTIVE_CLAIM_BOUNDARY.md advertised it as an active control.
    """
    if pdf and pdf.exists():
        import pymupdf
        doc = pymupdf.open(pdf)
        return "".join(p.get_text() for p in doc), str(pdf)
    files = sorted(f for s in SCAN_SUFFIXES for f in tex_dir.rglob(f"*{s}")
                   if "audits/pilot_stage_a" not in str(f)
                   and str(f.relative_to(tex_dir)) not in QUOTING_DOCUMENTS)
    if not files:
        return "", str(tex_dir)
    parts = [f"\n<<<FILE {f.relative_to(tex_dir)}>>>\n" + f.read_text(errors="replace")
             for f in files]
    exempt = sum(1 for n in QUOTING_DOCUMENTS if (tex_dir / n).exists())
    return ("\n".join(parts),
            f"{len(files)} .tex/.md files under {tex_dir} "
            f"({exempt} rule-stating documents exempt by name)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--tex-dir", type=Path, default=PAPER)
    args = ap.parse_args()

    raw, source = load_text(args.pdf, args.tex_dir)
    if not raw.strip():
        print(f"nothing to scan at {source}; the manuscript does not exist yet")
        return 0
    for k, v in LIG.items():
        raw = raw.replace(k, v)
    text = re.sub(r"\s+", " ", raw)

    hits = []
    for item, pattern, why in BLACKLIST:
        for m in re.finditer(pattern, text, re.I):
            ctx = text[max(0, m.start() - 100):m.end() + 100]
            run_up = text[max(0, m.start() - 60):m.start()]
            if HEDGE.search(run_up):
                continue    # the sentence poses the claim as a question, not an assertion
            if PROHIBITION.search(text[max(0, m.start() - 140):m.start()]):
                continue    # the sentence forbids the claim rather than making it
            hits.append((item, why, m.group(0), ctx))

    print(f"scanned {source}: {len(text.split())} words against "
          f"{len(BLACKLIST)} blacklist patterns")
    if not hits:
        print("PASS: no blacklisted claim found")
        return 0
    print(f"FAIL: {len(hits)} blacklisted claim(s)\n")
    for item, why, matched, ctx in hits:
        print(f"  blacklist item {item}: {why}")
        print(f"    matched: {matched!r}")
        print(f"    context: ...{ctx}...\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
