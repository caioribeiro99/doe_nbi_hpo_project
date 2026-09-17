"""Every internal pointer must resolve, and no markdown marker may reach the reader.

Renumbering a section is an editorial edit that silently breaks any cross-reference
aimed at the old number: nothing errors, the PDF builds, and the citation simply points
somewhere else. The v3 pass renumbered two sections, so this resolves every section,
table, figure and supplement reference against the headings and captions that exist.

It also catches markdown control characters stranded mid-line, where they render as
literal punctuation instead of as structure.
"""
from __future__ import annotations

import collections
import pathlib
import re
import sys

PAPER = pathlib.Path(__file__).resolve().parents[1] / "manuscript"

SEC_HEAD = re.compile(r"^#{2,6}\s+(\d+(?:\.\d+)*)\.?\s", re.M)
SUP_HEAD = re.compile(r"^#{2,6}\s+(S\d+)\b", re.M)
SEC_REF = re.compile(r"§\s*(\d+(?:\.\d+)*)")
SUP_REF = re.compile(r"\bSupplement\s+(S\d+)\b")
TAB_REF = re.compile(r"\bTable\s+(\d+)\b")
FIG_REF = re.compile(r"\bFigure\s+(\d+)\b")
TAB_DEF = re.compile(r"^\*\*Table\s+(\d+)\.", re.M)
# Captions are gathered in a "Figure captions" section and read
# **Figure 1. The arm lattice.** -- bold spans the whole caption.
FIG_DEF = re.compile(r"^\*\*Figure\s+(\d+)\.", re.M)

# A markdown marker is structural only at the start of a line. Mid-line it is literal.
# A block marker is structural only at the start of its line, so the test is simply
# whether anything precedes it. Anchoring on sentence-ending punctuation instead was
# the first attempt and it missed "**Funding.** > **AUTHOR...", where the character
# before the marker is the bold close, not the full stop.
STRANDED = re.compile(r"(?<=\S)[ \t]+(>|\#{1,6})[ \t]+(?=\*{0,2}[A-Z])")


# A marker must either open a parenthetical or be governed by a preposition. Anything
# else is "...in the normalization reference alone Figure 1." -- the parentheses were
# lost somewhere in assembly, and the sentence no longer parses.
MARKER = re.compile(r"(?<!\*\*)\b((?:Table|Figure)s?\s+\d+|Supplement\s+S\d+(?:\.\d+)?)")
GOVERNED = ("(", "see", "in", "and", "as", "from", "of", ";", ",", ":")


def malformed(name: str, text: str) -> list[str]:
    bad = []
    for m in MARKER.finditer(text):
        before = text[max(0, m.start() - 60):m.start()].replace("\n", " ").rstrip()
        if before and not before.endswith(GOVERNED):
            bad.append("L%d %s" % (text[:m.start()].count("\n") + 1, m.group(0)))
    print(f"    {name:<26} malformed references: {bad or 'none'}")
    return [f"{name}: reference marker not parenthesised or governed: {bad}"] if bad else []


def report(label: str, refs: set[str], defined: set[str]) -> list[str]:
    missing = sorted(refs - defined)
    print(f"    {label:<26} {len(refs):>3} referenced, "
          f"{len(defined):>3} defined, unresolved: {missing or 'none'}")
    return [f"{label}: unresolved {missing}"] if missing else []


def _regenerates_identically(script: str, target: pathlib.Path, label: str) -> list[str]:
    """Run a generator and check it reproduces its output byte for byte.

    Both documents are generated -- MANUSCRIPT.md from sections/, SUPPLEMENT.md from
    build_supplement.py -- so an edit made to either output is reverted the next time
    anyone regenerates, silently and without an error. Three v3 fixes were found living
    only in a generated file, so this is checked rather than assumed.
    """
    import contextlib, difflib, importlib.util, io
    spec = importlib.util.spec_from_file_location(
        label, pathlib.Path(__file__).with_name(script))
    mod = importlib.util.module_from_spec(spec)
    before = target.read_text()
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
        mod.main()
    after = target.read_text()
    if before != after:
        target.write_text(before)              # leave the tree exactly as found
        d = [l for l in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                             lineterm="", n=0) if l[:1] in "+-"][:6]
        print(f"    {label:<26} OUT OF SYNC, e.g. {d}")
        return [f"{target.name} differs from its generator ({script}): edits made to "
                "the generated file will be lost on the next regeneration"]
    print(f"    {label:<26} in sync")
    return []


def sections_are_in_sync() -> list[str]:
    """MANUSCRIPT.md is generated from sections/; an edit made to it directly is
    reverted the next time anyone assembles, silently and without an error. Three v3
    fixes were found living only in the generated file, so this is checked, not assumed.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "assemble", pathlib.Path(__file__).with_name("assemble_manuscript.py"))
    mod = importlib.util.module_from_spec(spec)
    target = PAPER / "MANUSCRIPT.md"
    before = target.read_text()
    spec.loader.exec_module(mod)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main()
    after = target.read_text()
    if before != after:
        target.write_text(before)          # leave the tree exactly as found
        import difflib
        d = [l for l in difflib.unified_diff(before.splitlines(), after.splitlines(),
                                             lineterm="", n=0) if l[:1] in "+-"][:6]
        print(f"    sections/ sync             OUT OF SYNC, e.g. {d}")
        return ["MANUSCRIPT.md differs from assemble(sections/): edits made to the "
                "generated file will be lost on the next assembly"]
    print("    sections/ sync             in sync")
    return []


def main() -> int:
    man = (PAPER / "MANUSCRIPT.md").read_text()
    sup = (PAPER / "SUPPLEMENT.md").read_text()
    failures: list[str] = []

    print("MANUSCRIPT.md")
    sections = set(SEC_HEAD.findall(man))
    dup = [s for s, c in collections.Counter(SEC_HEAD.findall(man)).items() if c > 1]
    if dup:
        failures.append(f"duplicate section numbers: {dup}")
    print(f"    section numbers            {len(sections)} unique, duplicates: {dup or 'none'}")
    failures += report("section references", set(SEC_REF.findall(man)), sections)
    failures += report("table references", set(TAB_REF.findall(man)), set(TAB_DEF.findall(man)))
    failures += report("figure references", set(FIG_REF.findall(man)), set(FIG_DEF.findall(man)))
    failures += report("supplement references", set(SUP_REF.findall(man)), set(SUP_HEAD.findall(sup)))

    failures += sections_are_in_sync()
    failures += _regenerates_identically(
        "build_supplement.py", PAPER / "SUPPLEMENT.md", "SUPPLEMENT.md generator")

    # A caption defines a float; an orphan is one nothing points at, and a float with
    # two numbered captions shows the reader the same number twice. Both are invisible
    # to a resolution check, which only asks whether a pointer has a target.
    for kind, defn in (("table", TAB_DEF), ("figure", FIG_DEF)):
        counts = collections.Counter(defn.findall(man))
        twice = sorted(k for k, c in counts.items() if c > 1)
        body = re.sub(r"^\*\*(?:Table|Figure)\s+\d+\..*$", "", man, flags=re.M)
        cited = set((TAB_REF if kind == "table" else FIG_REF).findall(body))
        orphan = sorted(set(counts) - cited, key=int)
        print(f"    {kind + 's':<26} {len(counts)} captioned, "
              f"duplicate captions: {twice or 'none'}, never cited: {orphan or 'none'}")
        if twice:
            failures.append(f"{kind} with more than one numbered caption: {twice}")
        if orphan:
            failures.append(f"{kind} captioned but never cited in the text: {orphan}")

    print("SUPPLEMENT.md")
    sup_secs = set(SUP_HEAD.findall(sup))
    print(f"    supplement sections        {len(sup_secs)} defined")

    for name, text in (("MANUSCRIPT.md", man), ("SUPPLEMENT.md", sup)):
        failures += malformed(name, text)
        stranded = [m.group(0).strip() for m in STRANDED.finditer(text)]
        print(f"    {name:<26} stranded markdown markers: {stranded or 'none'}")
        if stranded:
            failures.append(f"{name}: markdown marker stranded mid-line {stranded}")

    print()
    if failures:
        print("FAIL: cross-reference audit")
        for f in failures:
            print("  -", f)
        return 1
    print("PASS: every internal reference resolves; no stranded markdown markers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
