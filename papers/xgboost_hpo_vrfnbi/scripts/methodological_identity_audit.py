#!/usr/bin/env python
"""Assert, from the frozen source, which optimizer the dissertation actually ran.

Extracts tag ``v0.1.0-dissertation`` to a scratch directory and checks the solver
module for the structural elements of Normal Boundary Intersection. Writes a JSON
record so the claim in METHODOLOGICAL_IDENTITY_AUDIT.md is reproducible rather
than asserted.

No historical artifact is modified.

Usage:  python methodological_identity_audit.py [--scratch DIR]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits"
DISS_TAG = "v0.1.0-dissertation"

# Structural elements of NBI. Each is a family of spellings that any faithful
# implementation would have to use somewhere.
NBI_ELEMENTS = {
    "payoff_matrix": [r"\bphi\b", r"payoff", r"F_star", r"f_star"],
    "chim": [r"\bchim\b", r"convex hull of individual minima"],
    "quasi_normal": [r"n_hat", r"quasi[_ ]normal", r"normal direction"],
    "anchors": [r"\banchor", r"individual minim", r"per-objective minimiz"],
    "max_t_subproblem": [r"\bmax\s+t\b", r"maximize\s+t\b", r"\bt_val\b", r"\[k\]\s*#?\s*t\b"],
}


def extract(scratch: Path) -> Path:
    scratch.mkdir(parents=True, exist_ok=True)
    tar = subprocess.run(["git", "archive", DISS_TAG, "src/doe_xgb", "scripts"],
                         cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(scratch)], input=tar.stdout, check=True)
    return scratch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/tmp/diss_frozen_ident")
    args = ap.parse_args()

    root = extract(Path(args.scratch))
    nbi_src = (root / "src" / "doe_xgb" / "nbi.py").read_text()
    run_src = (root / "scripts" / "run_nbi.py").read_text()

    tree = ast.parse(nbi_src)
    solvers = [n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and "nbi" in n.name.lower()]

    low = nbi_src.lower()
    elements = {
        name: {"patterns": pats,
               "found": sorted({m for p in pats for m in re.findall(p, low)})}
        for name, pats in NBI_ELEMENTS.items()
    }
    absent = [k for k, v in elements.items() if not v["found"]]

    # The scalarization signature: a dot product of a weight vector with normalized
    # predictions, negated for a minimizer.
    ws_signature = re.search(r"return\s+-float\(\s*np\.dot\(\s*betas_arr\s*,\s*norm\s*\)\s*\)", nbi_src)
    minmax = re.search(r"norm\s*=\s*\(preds\s*-\s*nadir\)\s*/\s*denom", nbi_src)

    # Where the reference box comes from, in the driver script.
    # Match to end of line: the assignment contains nested parentheses, so a
    # non-greedy bracket class stops inside float(... .max() and misses the call.
    utopia_line = re.search(r"^\s*utopia\s*=.*$", run_src, re.M)
    nadir_line = re.search(r"^\s*nadir\s*=.*$", run_src, re.M)
    observed_extremes = bool(utopia_line and ".max()" in utopia_line.group(0)
                             and nadir_line and ".min()" in nadir_line.group(0))

    report = {
        "dissertation_tag": DISS_TAG,
        "dissertation_commit": subprocess.run(
            ["git", "rev-parse", DISS_TAG + "^{commit}"], cwd=REPO,
            capture_output=True, text=True).stdout.strip(),
        "solver_functions_in_nbi_module": solvers,
        "nbi_structural_elements": elements,
        "nbi_elements_absent": absent,
        "weighted_sum_objective_found": bool(ws_signature),
        "weighted_sum_objective_source": ws_signature.group(0) if ws_signature else None,
        "minmax_normalization_found": bool(minmax),
        "reference_box_from_observed_extremes": observed_extremes,
        "reference_box_source": {
            "utopia": utopia_line.group(0).strip() if utopia_line else None,
            "nadir": nadir_line.group(0).strip() if nadir_line else None,
        },
        "verdict": (
            "The frozen solver is normalized weighted-sum scalarization over a weight grid, "
            "with a normalization box built from component-wise observed extremes. None of the "
            f"{len(absent)} checked NBI structural elements is present."
            if absent and ws_signature else
            "INCONCLUSIVE -- re-read the source by hand."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "methodological_identity.json").write_text(json.dumps(report, indent=2))

    print(f"solver functions found: {solvers}")
    print(f"NBI structural elements absent: {absent}")
    print(f"weighted-sum objective present: {bool(ws_signature)}")
    print(f"reference box from observed extremes: {observed_extremes}")
    print(f"\n{report['verdict']}")
    ok = bool(absent) and bool(ws_signature) and observed_extremes
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
