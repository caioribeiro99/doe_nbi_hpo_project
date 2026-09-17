#!/usr/bin/env python
"""Known-regression phrase audit, on unwrapped text.

A line-based scan misses a negation split across a markdown line wrap -- "It is not\\nthe
true Pareto front" reads as an assertion to a line scanner. Text is normalized to one
line per paragraph before matching, and the classifying window is the sentence.
"""
from __future__ import annotations
import pathlib, re, sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
DOCS = {
    "MANUSCRIPT": PAPER/"manuscript"/"MANUSCRIPT.md",
    "SUPPLEMENT": PAPER/"manuscript"/"SUPPLEMENT.md",
    "CLAIMS":     PAPER/"CONFIRMATORY_CLAIMS_AND_EVIDENCE.md",
}
PATTERNS = ["leads every surrogate-assisted arm", "every surrogate-assisted arm on every dataset",
            "evaluation-matched direct search", "411,480", "Spambase null",
            "no geometry effect on Spambase", "true Pareto",
            "fraction of optimal hypervolume", "significant on all three", "fc10a84",
            "Figure 3, Figure 3", "To be completed by the authors",
            "higher median CORE-relative hypervolume ratio than every",
            # v4: the grid universal in its remaining forms, the broken prose the v3
            # substitutions left behind, and the cross-reference the renumbering staled.
            "led every surrogate-assisted arm",
            "attained higher median core-relative hypervolume than every",
            "attained favoured",
            "favoured the coarse grid on all four datasets in median paired difference, and exceeded every",
            "Section 6.7 reports this baseline",
            "geometry result in Section 6.2",
            "a coarse grid favoured the coarse grid"]
# A forbidden phrase is acceptable only inside a denial, a prohibition, or a record of
# something withdrawn.
OK = re.compile(r"\b(?:not|never|no |wrong|must not|do not|does not|prohibited|forbidden|"
                r"withdrawn|false|earlier version|is not|cannot|superseded|pre-refit)\b", re.I)


def sentences(text: str):
    flat = re.sub(r"[ \t]*\n(?!\n)", " ", text)     # unwrap, keep paragraph breaks
    for para in flat.split("\n"):
        for sent in re.split(r"(?<=[.!?])\s+", para):
            if sent.strip():
                yield sent.strip()


def main() -> int:
    invalid = 0
    for pat in PATTERNS:
        hits = []
        for name, path in DOCS.items():
            if not path.exists():
                continue
            for sent in sentences(path.read_text()):
                if pat in sent:
                    cls = "HISTORICAL/WARNING" if OK.search(sent) else "INVALID"
                    hits.append((name, cls, sent[:95]))
        if not hits:
            print(f"  VALID (absent)       {pat!r}")
        for name, cls, sent in hits:
            print(f"  {cls:19}  {pat!r} [{name}] {sent}")
            if cls == "INVALID":
                invalid += 1
    print(f"\n{'PASS: no invalid occurrences' if not invalid else f'FAIL: {invalid} invalid'}")
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
