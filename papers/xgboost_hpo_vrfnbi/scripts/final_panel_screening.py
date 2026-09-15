#!/usr/bin/env python
"""The authoritative pre-campaign panel screening, and the dataset-role map.

This is the single artifact the protocol, the runner and the statistical plan all
read the panel from. It supersedes `panel_rescreen_corrected.json`, which remains as
the record of the intermediate recomputation.

CRITERION 1 uses the CANONICAL `raw_conflict()` from `doe_xgb.campaign.factor_model`
-- the equally weighted standardized quality badness against the RAW complexity
response `Leaves_Mean`, with no factor stage on either side. It is imported, not
reimplemented: an earlier script reimplemented it and silently correlated raw quality
against the LATENT cost factor instead, producing a third quantity that was neither
criterion.

NO NUMERIC THRESHOLD IS INVENTED HERE. `protocol/dataset_selection.md` states the
four criteria qualitatively and never fixed numbers for them, so each is
operationalized the minimum way a boolean requires, and the operationalization is
recorded in the artifact beside the value:

  1. "clearly negative"                    -> value < 0 AND its bootstrap 95%
                                              interval excludes zero
  2. "detectably non-linear"               -> at least three non-dominated design
                                              rows (curvature is undefined on two)
                                              AND curvature > 0
  3. "strictly between the trivial and     -> 0 < external R2 < 1
      the perfect"
  4. "large enough that cost differences   -> Leaves_Mean is a deterministic
      exceed run-to-run variation"            model-complexity count at fixed
                                              config and seed, so its run-to-run
                                              variation is zero; any range > 1
                                              satisfies this

PRE-CONFIRMATORY SCREENING EVIDENCE. Computed before any arm has executed. Roles are
assigned from these measurements and are NEVER revised from optimizer outcomes.

Usage:
    python final_panel_screening.py [--check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
REPO = PAPER.parent.parent
sys.path.insert(0, str(REPO / "src"))

from doe_xgb.campaign.design import external_scores          # noqa: E402
from doe_xgb.campaign.factor_model import (load_reference_factor_model,  # noqa: E402
                                           raw_conflict)
from doe_xgb.reporting import pareto_front                   # noqa: E402

PILOT = PAPER / "audits" / "pilot_stage_a"
OUT = PAPER / "audits" / "final_panel_screening.json"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")
BOOTSTRAP = 2000
SEED = 20260914


def criterion_1(design: pd.DataFrame, rng) -> dict:
    """Canonical raw-response conflict. Imported, never reimplemented."""
    value = float(raw_conflict(design))
    n = len(design)
    boot = np.array([raw_conflict(design.iloc[rng.integers(0, n, n)])
                     for _ in range(BOOTSTRAP)], dtype=float)
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {"value": round(value, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "ci_excludes_zero": bool(hi < 0.0 or lo > 0.0),
            "pass": bool(value < 0.0 and hi < 0.0),
            "implementation": "doe_xgb.campaign.factor_model.raw_conflict",
            "measures": ("equally weighted standardized quality badness against the "
                         "RAW complexity response Leaves_Mean; no factor stage on "
                         "either side"),
            "rule": "clearly negative (dataset_selection.md, amendment 20)"}


def criterion_2(F: np.ndarray) -> dict:
    """Curvature of the non-dominated set of design rows."""
    mask = pareto_front(F)
    P = F[mask]
    P = P[np.argsort(P[:, 0])]
    n_front = int(mask.sum())
    if n_front < 3:
        return {"front_size": n_front, "curvature": None, "pass": False,
                "rule": "detectably non-linear (dataset_selection.md)",
                "why": ("a non-dominated set of fewer than three points has no "
                        "interior and no defined curvature: every scalarization "
                        "returns the same extreme points, so there is no interior "
                        "front geometry for the primary contrast to separate")}
    a, b = P[0], P[-1]
    v = b - a
    L = float(np.linalg.norm(v))
    signed = (v[0] * (P[:, 1] - a[1]) - v[1] * (P[:, 0] - a[0])) / L
    curv = float(np.abs(signed).max() / L)
    return {"front_size": n_front, "curvature": round(curv, 4),
            "pass": bool(curv > 0.0),
            "rule": "detectably non-linear (dataset_selection.md)",
            "why": "deviation of the non-dominated set from its own endpoint chord"}


def criterion_3(design, complement, fm) -> dict:
    Yd, Ye = fm.objectives(design), fm.objectives(complement)
    per = {}
    for j, obj in enumerate(("quality", "cost")):
        sc = external_scores(design, Yd[:, j], complement, Ye[:, j])
        r2 = float(sc["external_r2"])
        per[obj] = {"external_r2": round(r2, 4),
                    "external_spearman": round(float(sc["external_spearman"]), 4),
                    "surface_terms": int(sc["terms"]),
                    "pass": bool(0.0 < r2 < 1.0),
                    "meets_frozen_reliability_gate":
                        bool(r2 >= 0.5 and float(sc["external_spearman"]) >= 0.9)}
    return {"per_objective": per,
            "value": min(per[o]["external_r2"] for o in per),
            "pass": all(per[o]["pass"] for o in per),
            "rule": "strictly between the trivial and the perfect (0 < R2 < 1)"}


def criterion_4(frame: pd.DataFrame) -> dict:
    leaves = frame["Leaves_Mean"].to_numpy(float)
    ratio = float(leaves.max() / max(leaves.min(), 1e-9))
    return {"value": round(ratio, 1), "min": float(leaves.min()),
            "max": float(leaves.max()), "pass": bool(ratio > 1.0),
            "rule": ("large enough that cost differences exceed run-to-run variation; "
                     "Leaves_Mean is a deterministic complexity count at fixed config "
                     "and seed, so its run-to-run variation is zero")}


def screen(ds: str, rng) -> dict:
    design = pd.read_csv(PILOT / f"{ds}_design.csv")
    complement = pd.read_csv(PILOT / f"{ds}_validation_complement.csv")
    both = pd.concat([design, complement], ignore_index=True)
    fm = load_reference_factor_model(ds)
    F = fm.objectives(design)

    c1 = criterion_1(design, rng)
    c2 = criterion_2(F)
    c3 = criterion_3(design, complement, fm)
    c4 = criterion_4(both)

    # THE ROLE RULE, applied mechanically. Criterion 2 is the geometry criterion:
    # a dataset whose design-row front has no interior cannot exhibit the
    # weighted-sum-versus-NBI geometry effect, so it cannot carry the primary
    # mechanistic claim. It is executed in full and reported as a boundary control.
    role = ("primary_geometry_confirmatory" if c2["pass"]
            else "boundary_geometry_control")
    return {"dataset": ds,
            "criterion_1_raw_conflict_value": c1["value"],
            "criterion_1_pass": c1["pass"],
            "criterion_2_front_size": c2["front_size"],
            "criterion_2_curvature": c2["curvature"],
            "criterion_2_pass": c2["pass"],
            "criterion_3_value": c3["value"],
            "criterion_3_pass": c3["pass"],
            "criterion_4_value": c4["value"],
            "criterion_4_pass": c4["pass"],
            "final_role": role,
            "detail": {"criterion_1": c1, "criterion_2": c2,
                       "criterion_3": c3, "criterion_4": c4}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows = [screen(ds, rng) for ds in DATASETS]
    payload = {
        "purpose": "the authoritative pre-campaign panel screening and dataset-role map",
        "status": "PRE-CONFIRMATORY SCREENING EVIDENCE; no arm has executed",
        "criterion_1_operationalization": (
            "canonical raw_conflict(): equally weighted standardized quality badness "
            "against the RAW Leaves_Mean complexity response. The latent "
            "quality-factor versus latent cost-factor rank correlation is WITHDRAWN "
            "(amendment 20) and is not computed here."),
        "role_rule": (
            "criterion 2 is the geometry criterion. A dataset meeting it is "
            "primary_geometry_confirmatory; one failing it is boundary_geometry_control, "
            "executed in full under the identical pipeline and excluded ONLY from the "
            "primary inferential family. Roles are assigned from these pre-campaign "
            "measurements and are never revised from optimizer outcomes."),
        "bootstrap_resamples": BOOTSTRAP, "seed": SEED,
        "datasets": rows,
        "primary_geometry_panel": [r["dataset"] for r in rows
                                   if r["final_role"] == "primary_geometry_confirmatory"],
        "boundary_controls": [r["dataset"] for r in rows
                              if r["final_role"] == "boundary_geometry_control"],
        "all_datasets_executed": [r["dataset"] for r in rows],
        "replacement_dataset_selected": False,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if a.check:
        if not OUT.exists() or OUT.read_text() != text:
            print("MISMATCH: committed screening differs from a rebuild")
            return 1
    else:
        OUT.write_text(text)

    print(f"{'dataset':16} {'C1 raw':>8} {'C1':>4} {'C2 |F|':>7} {'C2 curv':>8} {'C2':>4} "
          f"{'C3 min R2':>10} {'C3':>4} {'C4 range':>9} {'C4':>4}  role")
    for r in rows:
        cv = "n/a" if r["criterion_2_curvature"] is None else f"{r['criterion_2_curvature']:.4f}"
        print(f"{r['dataset']:16} {r['criterion_1_raw_conflict_value']:8.4f} "
              f"{str(r['criterion_1_pass']):>4} {r['criterion_2_front_size']:7} {cv:>8} "
              f"{str(r['criterion_2_pass']):>4} {r['criterion_3_value']:10.4f} "
              f"{str(r['criterion_3_pass']):>4} {r['criterion_4_value']:9.1f} "
              f"{str(r['criterion_4_pass']):>4}  {r['final_role']}")
    print(f"\nprimary geometry panel : {payload['primary_geometry_panel']}")
    print(f"boundary controls      : {payload['boundary_controls']}")
    print(f"all executed           : {payload['all_datasets_executed']}")
    print(f"\n{'checked' if a.check else 'wrote ' + str(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
