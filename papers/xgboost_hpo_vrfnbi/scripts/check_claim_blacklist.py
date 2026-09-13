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
]


# A claim stated as something to be tested is not a claim. Reporting a blacklisted
# phrase that the paper is about to measure, or attribute to someone else, is allowed;
# asserting it is not. This guard looks at the run-up to the match.
HEDGE = re.compile(
    r"\b(?:whether|if|tested|we\s+test|is\s+measured|was\s+measured|not\s+assumed|"
    r"do(?:es)?\s+not\s+claim|claims?\s+no|question\s+(?:of|whether)|ask(?:s|ed)?\s+whether)\b",
    re.I)


def load_text(pdf: Path | None, tex_dir: Path) -> tuple[str, str]:
    if pdf and pdf.exists():
        import pymupdf
        doc = pymupdf.open(pdf)
        return "".join(p.get_text() for p in doc), str(pdf)
    parts = [p.read_text(errors="replace") for p in sorted(tex_dir.rglob("*.tex"))]
    if not parts:
        return "", str(tex_dir)
    return "\n".join(parts), f"{len(parts)} .tex files under {tex_dir}"


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
            if HEDGE.search(text[max(0, m.start() - 60):m.start()]):
                continue    # the sentence poses the claim as a question, not an assertion
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
