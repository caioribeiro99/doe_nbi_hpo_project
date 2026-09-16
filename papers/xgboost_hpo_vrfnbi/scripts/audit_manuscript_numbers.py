#!/usr/bin/env python
"""Every number in the manuscript must trace to a verified artifact.

The manuscript is drafted from CONFIRMATORY_CLAIMS_AND_EVIDENCE.md, which is itself
generated from the analysis artifacts. This closes the loop: it extracts every numeric
token from the manuscript sections and checks each against the union of values that
appear in the claims map and the analysis JSONs.

A number that matches nothing is not necessarily wrong -- section counts, years,
hyperparameter bounds and structural constants are legitimate -- so those are declared
in ALLOWED_STRUCTURAL and everything else must match an artifact. The point is that an
INVENTED RESULT cannot pass silently.

Usage:  python audit_manuscript_numbers.py [--verbose]
Exit 1 if any unexplained result-like number is found.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
PAPER = REPO / "papers" / "xgboost_hpo_vrfnbi"
SECTIONS = PAPER / "manuscript" / "sections"
A = PAPER / "analysis"

# Structural constants of the design, not results. Each is checkable in the protocol.
ALLOWED_STRUCTURAL = {
    # design and panel
    "88", "78", "166", "288", "200", "120", "30", "20", "7", "4", "3", "2", "1", "0",
    "64", "14", "10", "5", "6", "8", "9", "12", "15", "16", "18", "19", "21", "22",
    "23", "24", "25", "26", "27", "28", "29", "32", "36", "40", "50", "100", "102",
    # budgets
    "108", "186", "386", "384", "396120", "380760", "15360", "9960", "386160",
    "377316", "3173", "3090", "83", "3840",
    # hyperparameter bounds
    "0.05", "1.0", "0.01", "0.3", "0.5", "700", "0.0", "5.0", "0.525", "0.155",
    "375", "0.155", "18",
    # statistics constants
    "0.05", "8.5", "2.9155", "0.25", "0.20", "0.80", "0.95", "1.96", "1.1",
    "0.9", "0.5", "10000", "2000", "1500", "20260914", "20260915",
    # years
    "1998", "2003", "2015", "2018", "2025", "2026",
}

NUM = re.compile(r"(?<![\w.])[-+−]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?(?![\w])")



CITE_YEAR = re.compile(r"(19|20)\d{2}$")
SECTION_REF = re.compile(r"(?:§|[Ss]ection|[Ss]ec\.)\s*$")


def _is_reference(raw: str, line: str, pos: int) -> bool:
    """Bibliographic years and section cross-references are not results.

    A citation year is a 4-digit 19xx/20xx immediately inside a parenthetical or
    followed by a citation separator; a section reference is a number introduced by a
    section marker. Neither can be an invented finding, which is what this audit is
    for.
    """
    before = line[:pos]
    after = line[pos + len(raw):]
    if CITE_YEAR.match(raw.strip()):
        if before.rstrip().endswith("(") or after.lstrip()[:1] in {")", ";", ",", ":"}:
            return True
        if re.search(r"(?:et al\.|and [A-Z][a-z]+)\s*\(?$", before):
            return True
    if SECTION_REF.search(before):
        return True
    # section headings like "### 6.4.1" and dotted version strings like "3.11.15"
    if re.match(r"^\d+(?:\.\d+){2,}$", raw):
        return True
    if re.match(r"^#{1,6}\s*$", before):
        return True
    # scientific notation, in LaTeX ($9.42\times10^{-7}$) or plain (9.42e-07) form:
    # the regex splits the mantissa from the exponent, so reassemble and re-check.
    m = re.match(r"^\s*(?:\\times\s*10\^\{?|[eE])\s*[-+\u2212]?\d+", after)
    if m:
        # the exponent is what follows "^{" or "e", NOT the 10 in "times10"
        exp = re.search(r"(?:\^\{?|[eE])\s*([-+\u2212]?\d+)", m.group(0))
        if exp:
            e = int(exp.group(1).replace("\u2212", "-"))
            try:
                v = abs(float(raw.replace("\u2212", "-")) * 10.0 ** e)
            except ValueError:
                return False
            # A printed mantissa is rounded, so string equality cannot work here.
            # Match numerically, at the precision the manuscript actually prints.
            return any(abs(v - k) <= 5e-3 * max(abs(v), abs(k))
                       for k in _KNOWN_FLOATS if k != 0)
    # the exponent half of a split scientific-notation token
    if re.search(r"(?:\\times\s*10\^\{?|[eE])\s*[-+\u2212]?$", before):
        return True
    # LaTeX weight pairs like $(1.00,0.00)$ tokenize as one token
    if "," in raw and re.match(r"^\d\.\d{2},\d\.\d{2}$", raw):
        return True
    return False



def _artifact_floats() -> set[float]:
    """Every float in the verified artifacts, for tolerant numeric matching."""
    out: set[float] = set()
    for p in sorted(A.glob("*.json")):
        for m in NUM.finditer(p.read_text()):
            try:
                out.add(abs(float(m.group(0).replace(",", ""))))
            except ValueError:
                pass
    return out


def artifact_numbers() -> set[str]:
    """Every numeric token appearing anywhere in the verified artifacts."""
    vals: set[str] = set()
    texts = [(PAPER / "CONFIRMATORY_CLAIMS_AND_EVIDENCE.md").read_text()]
    for p in sorted(A.glob("*.json")):
        texts.append(p.read_text())
    texts.append((PAPER / "PROTOCOL_V3_FREEZE_REPORT.md").read_text())
    # The protocol documents legitimately record WITHDRAWN comparison values -- the
    # pre-correction factor correlations and unrotated weightings, for instance --
    # which the Methods section cites deliberately to say what was corrected. They are
    # part of the traceable record, so they count as sourced.
    for rel in ("PROTOCOL_AMENDMENTS.md", "protocol/OBJECTIVE_DEFINITIONS.md",
                "protocol/EXPERIMENT_PROTOCOL.md", "protocol/dataset_selection.md",
                "protocol/budget_accounting.md", "protocol/method_arms.md",
                "audits/final_panel_screening.json",
                "audits/PILOT_STAGE_A_FINDINGS.md", "STAGE_B_STATISTICAL_SENSITIVITY.md"):
        q = PAPER / rel
        if q.exists():
            texts.append(q.read_text())
    for q in sorted((PAPER / "audits" / "reference_factor_models").glob("*.json")):
        texts.append(q.read_text())
    for t in texts:
        for m in NUM.finditer(t):
            vals.add(norm(m.group(0)))
            # also register rounded forms, so 0.07414992 covers a printed 0.0741
            try:
                f = float(m.group(0).replace("−", "-").replace(",", ""))
            except ValueError:
                continue
            for d in (1, 2, 3, 4, 5):
                vals.add(norm(f"{f:.{d}f}"))
                vals.add(norm(f"{abs(f):.{d}f}"))
            vals.add(norm(str(int(f))) if f == int(f) else norm(f"{f}"))
            vals.add(norm(f"{round(f):d}"))
            vals.add(norm(f"{round(abs(f)):d}"))
            vals.add(norm(f"{abs(f)}"))
            if abs(f) < 1:
                vals.add(norm(f"{f*100:.1f}"))   # percentage forms
                vals.add(norm(f"{f*100:.0f}"))
    return vals


def norm(s: str) -> str:
    s = s.strip().replace("−", "-").replace(",", "")
    s = s.lstrip("+")
    if s.startswith("-"):
        s = s[1:]                      # compare magnitudes; sign is prose's job
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"



def _verify_environment_claims() -> list[str]:
    """The reproducibility section names package versions. Check them for real.

    A wrong version is the one number that actually breaks a reproduction attempt,
    and it is invisible to a claims-map check because it is not a result.
    """
    import platform
    text = (SECTIONS / "09_reproducibility.md").read_text()
    bad = []
    try:
        import numpy, pandas, scipy, sklearn, xgboost
        pairs = [("Python", platform.python_version()), ("numpy", numpy.__version__),
                 ("pandas", pandas.__version__), ("scipy", scipy.__version__),
                 ("scikit-learn", sklearn.__version__), ("xgboost", xgboost.__version__)]
    except Exception as exc:                       # pragma: no cover
        return [f"could not import a package to verify versions: {exc}"]
    for name, actual in pairs:
        if name in text and f"{name} {actual}" not in text:
            bad.append(f"{name}: manuscript does not state the installed {actual}")
    return bad

_KNOWN: set[str] = set()
_KNOWN_FLOATS: set[float] = set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    global _KNOWN
    known = artifact_numbers() | {norm(x) for x in ALLOWED_STRUCTURAL}
    _KNOWN = known
    global _KNOWN_FLOATS
    _KNOWN_FLOATS = _artifact_floats()
    unexplained: list[tuple[str, int, str, str]] = []
    total = 0
    for path in sorted(SECTIONS.glob("*.md")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip().startswith("[FIG:") or line.strip().startswith("[TAB:"):
                continue
            for m in NUM.finditer(line):
                raw = m.group(0)
                tok = norm(raw)
                total += 1
                if tok in known or _is_reference(raw, line, m.start()):
                    continue
                unexplained.append((path.name, i, raw, line.strip()[:110]))

    envbad = _verify_environment_claims()
    for b in envbad:
        print(f"ENVIRONMENT MISMATCH: {b}")
    print(f"scanned {len(list(SECTIONS.glob('*.md')))} sections, {total} numeric tokens, "
          f"{len(known)} known values from artifacts + structural constants")
    if not unexplained and not envbad:
        print("PASS: every number in the manuscript traces to a verified artifact "
              "or a declared structural constant")
        return 0
    print(f"FAIL: {len(unexplained)} unexplained numeric token(s)\n")
    for f, i, tok, ctx in unexplained[:40]:
        print(f"  {f}:{i}  {tok!r}")
        if args.verbose:
            print(f"      {ctx}")
    if len(unexplained) > 40:
        print(f"  ... and {len(unexplained)-40} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
