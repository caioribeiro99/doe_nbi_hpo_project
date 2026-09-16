#!/usr/bin/env python
"""Final static audit: cross-references, regression guards, consistency, disclosure."""
from __future__ import annotations
import json, pathlib, re, sys

PAPER = pathlib.Path(__file__).resolve().parents[1]
M = (PAPER/"manuscript"/"MANUSCRIPT.md").read_text()
S = (PAPER/"manuscript"/"SUPPLEMENT.md").read_text()
FIG = PAPER/"manuscript"/"figures"
A = PAPER/"analysis"
fails: list[str] = []

def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)

print("3. CROSS-REFERENCE AUDIT")
check("no unresolved [FIG:] tags", "[FIG:" not in M and "[FIG:" not in S)
check("no unresolved [TAB:] tags", "[TAB:" not in M and "[TAB:" not in S)
figs = sorted(set(re.findall(r"\bFigure (\d)\b", M)))
have = sorted(p.stem[3] for p in FIG.glob("fig?_*.png"))
check("every cited Figure N has a file", set(figs) <= set(have), f"cited {figs}, files {have}")
check("every figure has a caption", all(f"**Figure {n}." in M for n in figs))
tabs = sorted(set(re.findall(r"\bTable (\d)\b", M)))
check("every cited Table N has a caption", all(f"**Table {n}." in M for n in tabs),
      f"cited {tabs}")
supp = sorted(set(re.findall(r"Supplement (S\d+)(?:\.\d+)?", M)))
missing = [s for s in supp if f"## {s}." not in S]
check("every Supplement Sn reference exists", not missing, f"referenced {supp}")

print("\n4. REGRESSION GUARDS on the nineteen review fixes")
prim = json.loads((A/"primary_analysis.json").read_text())
def nb(ds):
    for r in prim["primary_family"]["core"][ds]:
        if r["contrast"] == "WS-S -> NBI-S":
            return r["nadeau_bengio"]["p"]
check("Nadeau-Bengio values appear in the manuscript",
      all(f"{nb(d):.4f}" in M for d in ("magic","adult","bank_marketing")))
check("Nadeau-Bengio NOT buried in the supplement only",
      "0.0148" in M.split("## 7.")[0] or "corrected test" in M.split("## 8.")[0])
check("no unqualified 'significant on all three' claim",
      not re.search(r"significant(?:ly)? on all three", M, re.I))
check("grid false universal absent",
      "every surrogate-assisted arm on every dataset" not in M)
check("'evaluation-matched' not used for the grid",
      "evaluation-matched coarse grid" not in M and "evaluation-matched grid" not in M)
check("grid budget asymmetry stated", "386" in M and "186" in M)
check("64-corner / CORE overlap disclosed",
      "64 factorial" in M and ("CORE reference" in M or "core reference" in M.lower()))
check("67% random padding disclosed", "67%" in M)
check("historical objective substitution disclosed",
      "wall-clock" in M and "leaf count" in M)
check("Spambase not a demonstrated null",
      "did not resolve" in M and "no geometry effect on Spambase" not in M)
check("consistent-with / does-not-confirm preserved",
      "consistent with" in M.lower() and "does not confirm" in M.lower())
check("joint-ND non-significance disclosed",
      "joint non-dominated fraction" in M and "non-significant" in M)
check("CORE never called a true Pareto front",
      not re.search(r"CORE[^.]{0,40}is the true Pareto front", M, re.I))
check("hv ratio unboundedness stated", "27.8" in M or "above 1" in M)
check("140 cells / 70 blocks stated", "140" in M and "70" in M)
# 411,480 may appear ONLY inside an explicit warning against it, never as a total
_bad = [l for l in M.splitlines() if "411,480" in l
        and not re.search(r"wrong|not added|never|double-count", l, re.I)]
check("budget never double-counted", not _bad, f"{len(_bad)} bare occurrences")
check("396,120 stated", "396,120" in M)

print("\n5. SELF-OVERLAP AND 6. THESIS DISCLOSURE")
check("dissertation disclosed", "MSc dissertation" in M or "master's dissertation" in M)
check("dissertation cited in references", "Ribeiro, C. T. (2026)" in M)
check("companion manuscript disclosed", "companion" in M.lower())
check("companion instruments attributed",
      "gate" in M.lower() and "anchor-injection control" in M)
check("Pereira et al. relationship stated", "Pereira" in M and "co-author" in M.lower())
check("overlap statement present", "Overlap statement" in M or "overlap" in M.lower())
check("historical reconstruction neutral",
      not re.search(r"\b(exposé|scandal|blunder|fraud)\b", M, re.I))

print("\n8. TABLE WIDTH")
wide = [(i, len(l)) for i, l in enumerate(M.splitlines(), 1)
        if l.startswith("|") and len(l) > 200]
check("no table row exceeds 200 characters", not wide,
      f"{len(wide)} wide rows" if wide else "")

print("\n9. MANUSCRIPT / SUPPLEMENT CONSISTENCY")
for tag, commit in (("xgboost-hpo-protocol-v3","9b15ba7"),
                    ("xgboost-hpo-confirmatory-results-v1","a4f20ed")):
    check(f"{tag} commit agrees in both documents",
          (commit in M) and (commit in S), commit)
check("both documents state 396,120", "396,120" in M and "396,120" in S)
check("both state the 288-point core", "288" in M and "288" in S)

print(f"\n{'PASS: final audit clean' if not fails else 'FAIL: ' + str(fails)}")
return_code = 1 if fails else 0
sys.exit(return_code)
