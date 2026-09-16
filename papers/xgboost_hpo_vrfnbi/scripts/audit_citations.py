#!/usr/bin/env python
"""Citation audit: every in-text citation resolves, no placeholders, no duplicates."""
from __future__ import annotations
import pathlib, re, sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
SEC = PAPER / "manuscript" / "sections"
BIB = PAPER / "manuscript" / "references.bib"
SUP = PAPER / "manuscript" / "SUPPLEMENT.md"

# surname(s) -> bib key, for author-year citations in the prose
SURNAME = {
    "Das and Dennis": "DasDennis1998",
    "Lujan-Moreno": "LujanMoreno2018",
    "Vasquez-Ramos": "VasquezRamos2025",
    "Ishibuchi": "Ishibuchi2015",
    "Karl": "Karl2023",
    "Morales-Hernández": "MoralesHernandez2023",
    "Costa": "Costa2016",
    "Luz": "Luz2021",
    "Streitenberger": "Streitenberger2022",
    "Pereira": "Pereira2025",
    "de Azevedo": "Azevedo2026",
    "Azevedo": "Azevedo2026",
    "Nadeau and Bengio": "NadeauBengio2003",
    "Bergstra and Bengio": "BergstraBengio2012",
    "Eggensperger": "Eggensperger2021",
    "Pfisterer": "Pfisterer2022",
    "Guerrero-Viu": "GuerreroViu2021",
    "Ribeiro": "Ribeiro2026Dissertation",
}


def main() -> int:
    bib = BIB.read_text()
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    body = "\n".join(p.read_text() for p in sorted(SEC.glob("*.md"))
                     if not p.name.startswith("11_references"))
    body += "\n" + (SUP.read_text() if SUP.exists() else "")
    refs = (SEC / "11_references_declarations.md").read_text()

    fail = 0
    # 1. no placeholders anywhere
    n_ph = (body + refs).count("complete before submission")
    print(f"placeholder markers: {n_ph}")
    if n_ph:
        fail = 1

    # 2. every cited surname resolves to a bib key
    cited, unresolved = set(), []
    for sur, key in SURNAME.items():
        if re.search(rf"{re.escape(sur)}", body):
            cited.add(key)
            if key not in keys:
                unresolved.append((sur, key))
    print(f"in-text citation groups resolved: {len(cited)}   unresolved: {len(unresolved)}")
    for s, k in unresolved:
        print(f"  UNRESOLVED {s} -> {k}")
        fail = 1

    # 3. bibliography entries never cited
    # the dissertation and the companion manuscript are cited in Declarations
    cited |= {"Ribeiro2026Dissertation", "RibeiroCompanion2026"} if (
        "MSc dissertation" in refs or "companion" in refs.lower()) else set()
    never = sorted(keys - cited)
    print(f"bib entries never cited in body: {len(never)}  {never}")

    # 4. duplicate DOIs / duplicate works
    dois = [d.lower() for d in re.findall(r"doi\s*=\s*\{([^}]+)\}", bib)]
    dup = {d for d in dois if dois.count(d) > 1}
    print(f"duplicate DOIs: {len(dup)}  {sorted(dup)}")
    if dup:
        fail = 1
    titles = [t.lower().replace("\n", " ") for t in re.findall(r"title\s*=\s*\{(.+?)\},\n", bib, re.S)]
    norm = [re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t)) for t in titles]
    dupt = {t for t in norm if norm.count(t) > 1}
    print(f"duplicate titles: {len(dupt)}")
    if dupt:
        fail = 1

    # 5. every record carries a resolvable identifier
    recs = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)
    noid = [k for k, b in recs
            if "doi" not in b and "url" not in b and "eprint" not in b
            and "mastersthesis" not in k.lower() and k not in
            ("Ribeiro2026Dissertation", "RibeiroCompanion2026")]
    print(f"records without DOI/URL/eprint (excluding unpublished): {len(noid)} {noid}")
    if noid:
        fail = 1

    # 6. accented names preserved
    for name in ("Morales-Hern{\\'a}ndez", "Rom{\\~a}o", "P{\\'e}rez-Cisneros",
                 "Ces{\\'a}rio", "Itajub{\\'a}"):
        if name not in bib:
            print(f"  ACCENT MISSING: {name}")
            fail = 1
    print("accented author/institution names: preserved" if not fail else "")

    print("\nPASS: citation audit clean" if not fail else "\nFAIL: citation audit")
    return fail


if __name__ == "__main__":
    raise SystemExit(main())
