"""The v3 pass is editorial and provenance only: no scientific content may move.

Rather than trust a reading of the diff, this compares the working tree against the
frozen ``paper2-manuscript-v2`` tag and asserts that the scientific surface is
bit-identical: every numeric value, every citation key, every dataset-role statement.

Editorial work does change some numbers -- section numbers in headings and
cross-references, and the provenance SHAs the supplement now names. Those are
declared here individually, with the reason, so an undeclared numeric movement fails
the audit rather than hiding among them.
"""
from __future__ import annotations

import collections
import pathlib
import re
import subprocess
import sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
# The chain is checked from the first frozen manuscript by default. A later base can be
# given on the command line -- the v5 pass is a funding-provenance correction and must
# also show a zero scientific diff against v4 specifically.
BASE = sys.argv[1] if len(sys.argv) > 1 else "paper2-manuscript-v2"
DOCS = ("MANUSCRIPT.md", "SUPPLEMENT.md")
# The supplement carries tables and provenance and cites no works in prose.
# Stating that here keeps the citation check from passing vacuously on it.
CITES_WORKS = {"MANUSCRIPT.md": True, "SUPPLEMENT.md": False}

# Numeric tokens whose movement is editorial, each with the reason it is allowed.
DECLARED: dict[str, str] = {
    "3.4": "numbering: the second 3.4 became 3.5",
    "3.5": "numbering: the second 3.4 became 3.5",
    "6.3.1": "numbering: the second 6.3.1 became 6.3.2",
    "6.3.2": "numbering: the second 6.3.1 became 6.3.2",
    "3": "prose repair: 'Figure 3, Figure 3.' became '(Figure 3)'",
    "13": "prose repair: the duplicated '13 of the 20 cells' clause was de-duplicated",
    "1.0192": "grid universal withdrawn: the Spambase counter-example, already in "
              "Table 5, is now stated in the prose that previously overclaimed",
    "1.0812": "grid universal withdrawn: the Spambase counter-example, already in "
              "Table 5, is now stated in the prose that previously overclaimed",
    "2016": "CRediT block cites Costa et al. (2016) by year",
    "2021": "CRediT block cites Luz et al. (2021) by year",
    "2022": "CRediT block cites Streitenberger et al. (2022) by year",
    "2025": "CRediT block cites Pereira et al. (2025) by year",
    "2026": "CRediT block cites de Azevedo et al. (2026) by year",
    "5": "two causes: the missing parent heading '5. Methods' was supplied, the "
         "numbering 5.1-5.15 having presupposed it; and Figure 5, captioned but "
         "never pointed at from the text, is now cited from 6.5",
    "6": "the missing parent heading '6. Results' was supplied; the numbering "
         "6.1-6.9 presupposed it and no file defined it",
    "4": "Table 4 alone carried two numbered captions; the title above the table was "
         "removed and its sign convention folded into the caption below",
    "30": "same removal: the duplicate Table 4 title also stated R = 30",
    "6.2": "cross-reference corrected: the geometry result is 6.3, not 6.2 "
           "(6.2 is the specification/normalization reconstruction); wrong since v1",
    "6.3": "cross-reference corrected: the geometry result is 6.3, not 6.2",
    "6.7": "cross-reference corrected: the grid baseline is reported in 6.8, not 6.7, "
           "after the section renumbering",
    "6.8": "cross-reference corrected: the grid baseline is reported in 6.8",
    # Author-supplied funding figures. No campaign artifact can verify a grant number,
    # so audit_manuscript_numbers cannot see these; the exact text is pinned instead by
    # tests/methodology/test_declarations.py.
    "386": "'evaluation-matched' replaced by the explicit frozen comparator budget",
    "384": "the NSGA-II shortfall is now disclosed as '384 against 386'",
}

NUM = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
CITE = re.compile(
    r"\b((?:[A-Z][A-Za-z\u00c0-\u024f-]+"
    r"(?: and [A-Z][A-Za-z\u00c0-\u024f-]+| et al\.)?))[ ,]+\(?((?:19|20)\d{2})\)?")
# The four datasets and the role vocabulary that assigns them.
ROLE = re.compile(
    r"\b(Adult|Bank ?Marketing|MAGIC(?: Gamma Telescope)?|Spambase)\b"
    r"[^.]{0,160}?\b(boundary geometry control|primary geometry panel|primary "
    r"inferential family|holdout|control)\b",
    re.I)


def at_tag(name: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BASE}:papers/xgboost_hpo_vrfnbi/manuscript/{name}"],
        cwd=PAPER.parents[1], capture_output=True, text=True, check=True).stdout


# The funding and acknowledgements paragraphs are excluded from the numeric surface and
# checked instead by tests/methodology/test_declarations.py, which pins them by exact
# equality and enforces a two-way funder check. Declaring their grant digits here was
# worse than useless: "22" and "9" are common tokens, and whitelisting them blinded this
# audit to every future movement of those values anywhere in the manuscript.
FUNDING_BLOCK = re.compile(
    r"\*\*Funding\.\*\*.*?(?=\*\*Author contributions)", re.S)


# A git object name is provenance metadata, not a scientific number: its digits are
# incidental and change at every freeze. Declaring them was worse than excluding them --
# the declaration whitelists whatever digits the current hash happens to contain, and
# goes stale the moment the hash does. audit_package_tag.py owns provenance correctness.
COMMIT_HASH = re.compile(r"`[0-9a-f]{7,40}`")


def without_declarations(text: str) -> str:
    return COMMIT_HASH.sub("``", FUNDING_BLOCK.sub("", text))


def numbers(text: str) -> collections.Counter:
    return collections.Counter(m.group(0).replace(",", "")
                               for m in NUM.finditer(without_declarations(text)))


def main() -> int:
    failures: list[str] = []
    for name in DOCS:
        old, new = at_tag(name), (PAPER / "manuscript" / name).read_text()

        # 1. numeric surface
        o, n = numbers(old), numbers(new)
        moved = {k: (o.get(k, 0), n.get(k, 0)) for k in set(o) | set(n)
                 if o.get(k, 0) != n.get(k, 0)}
        undeclared = {k: v for k, v in moved.items()
                      if k not in DECLARED and k.replace(".", "") not in DECLARED}
        print(f"{name}: {sum(n.values())} numeric tokens, "
              f"{len(moved)} moved ({len(moved) - len(undeclared)} declared)")
        for k, (a, b) in sorted(moved.items()):
            tag = DECLARED.get(k) or DECLARED.get(k.replace(".", ""))
            print(f"    {k:>12}  {a} -> {b}   {tag or '** UNDECLARED **'}")
        if undeclared:
            failures.append(f"{name}: undeclared numeric movement {sorted(undeclared)}")

        # 2. citation keys
        def keys(t):
            return {(a.strip(), y) for a, y in CITE.findall(t)}
        ko, kn = keys(old), keys(new)
        if bool(ko) is not CITES_WORKS[name]:
            failures.append(
                f"{name}: expected {'some' if CITES_WORKS[name] else 'no'} prose "
                f"citations at {BASE}, found {len(ko)} -- the pattern or the "
                "expectation is wrong, and the check would not be meaningful")
        if ko != kn:
            added, dropped = sorted(kn - ko), sorted(ko - kn)
            failures.append(f"{name}: works cited changed -- "
                            f"added {added}, dropped {dropped}")
        print(f"    works cited: {len(kn)} distinct, unchanged: {ko == kn}")

        # 3. dataset role assignments
        ro = collections.Counter((a.lower(), b.lower()) for a, b in ROLE.findall(old))
        rn = collections.Counter((a.lower(), b.lower()) for a, b in ROLE.findall(new))
        if ro != rn:
            failures.append(f"{name}: dataset role statements changed: "
                            f"{sorted((collections.Counter(ro) - rn).items())} -> "
                            f"{sorted((collections.Counter(rn) - ro).items())}")
        print(f"    dataset-role statements: {sum(rn.values())}, "
              f"unchanged: {ro == rn}")

    print()
    if failures:
        print("FAIL: the editorial pass moved scientific content")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: scientific surface identical to " + BASE +
          " (numbers, citations, dataset roles)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
