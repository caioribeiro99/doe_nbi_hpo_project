#!/usr/bin/env python
"""Which criterion of the surrogate gate failed, per unit and objective (Section 6.7).

The frozen gate (protocol section 7; doe_xgb.campaign.design.gate_pass) passes a fitted
surface when its external R^2 >= 0.5 AND its external Spearman rho >= 0.9 on the 78
audit-only external-validation rows. The campaign stored both statistics and the gate
flag for each of the 120 units and both objectives (external_validation.json);
analysis/secondary_analysis.json reports only the pass counts. This script breaks every
gate decision down by criterion:

  * per unit and objective: external R^2, external Spearman rho, the gate flag and
    which criterion failed (none, spearman_only, r2_only, both);
  * over all 240 unit-by-objective cells: the R^2 range and the failure tally;
  * per dataset and objective: pass counts, the failure tally, and the median, minimum
    and maximum of R^2 and rho (the composite-quality median rho is the "quality"
    objective's spearman median).

Provenance label:
post hoc

Every stored flag is checked against the frozen rule applied to the stored statistics;
the script stops on any disagreement.

Inputs (read only): experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/
external_validation.json, rep_00 to rep_29; and the gate thresholds from
src/doe_xgb/campaign/design.py.
Output: papers/xgboost_hpo_vrfnbi/analysis/gate_criterion_breakdown.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:  python papers/xgboost_hpo_vrfnbi/scripts/gate_criterion_breakdown.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.design import GATE_R2, GATE_SPEARMAN, gate_pass  # noqa: E402
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis" / "gate_criterion_breakdown.json"
LABEL = "post hoc"
OBJECTIVES = ("quality", "cost")
KINDS = ("none", "spearman_only", "r2_only", "both")


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def failed_criterion(r2: float, rho: float) -> str:
    bad_r2, bad_rho = not r2 >= GATE_R2, not rho >= GATE_SPEARMAN
    return {(False, False): "none", (False, True): "spearman_only",
            (True, False): "r2_only", (True, True): "both"}[(bad_r2, bad_rho)]


def tally(cells: list[dict]) -> dict:
    return {k: sum(c["failed_criterion"] == k for c in cells) for k in KINDS if k != "none"}


def span(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"median": float(np.median(x)), "min": float(np.min(x)), "max": float(np.max(x))}


def main() -> int:
    units = []
    for ds in DATASETS:
        for rep in range(N_REPLICATIONS):
            ev = json.loads((ROOT / ds / f"rep_{rep:02d}" / "external_validation.json")
                            .read_text())
            if tuple(ev["objectives"]) != OBJECTIVES:
                raise SystemExit(f"{ds} rep {rep}: objectives {ev['objectives']}")
            for j, obj in enumerate(OBJECTIVES):
                s = ev["scores"][j]
                r2, rho = float(s["external_r2"]), float(s["external_spearman"])
                if not (np.isfinite(r2) and np.isfinite(rho)):
                    raise SystemExit(f"{ds} rep {rep} {obj}: non-finite gate statistic")
                ok = gate_pass(s)
                if ok != bool(ev["gate_pass"][j]):
                    raise SystemExit(f"{ds} rep {rep} {obj}: stored flag disagrees with "
                                     f"the frozen rule")
                kind = failed_criterion(r2, rho)
                if (kind == "none") != ok:
                    raise SystemExit(f"{ds} rep {rep} {obj}: criterion bookkeeping")
                units.append({"dataset": ds, "replication": rep, "objective": obj,
                              "external_r2": r2, "external_spearman": rho,
                              "gate_pass": ok, "failed_criterion": kind})

    datasets = {}
    for ds in DATASETS:
        blk = {}
        for obj in OBJECTIVES:
            cells = [c for c in units if c["dataset"] == ds and c["objective"] == obj]
            blk[obj] = {"of": len(cells),
                        "passes": sum(c["gate_pass"] for c in cells),
                        "failures": sum(not c["gate_pass"] for c in cells),
                        "failures_by_criterion": tally(cells),
                        "external_r2": span([c["external_r2"] for c in cells]),
                        "external_spearman": span([c["external_spearman"] for c in cells])}
        datasets[ds] = blk
    r2_all = [c["external_r2"] for c in units]
    overall = {"n_units": len({(c["dataset"], c["replication"]) for c in units}),
               "n_cells": len(units),
               "external_r2_min": float(min(r2_all)),
               "external_r2_max": float(max(r2_all)),
               "n_cells_r2_at_or_above_threshold": sum(r >= GATE_R2 for r in r2_all),
               "n_cells_spearman_at_or_above_threshold":
                   sum(c["external_spearman"] >= GATE_SPEARMAN for c in units),
               "n_failures": sum(not c["gate_pass"] for c in units),
               "failures_by_criterion": tally(units),
               "failures_by_objective": {o: tally([c for c in units if c["objective"] == o])
                                         for o in OBJECTIVES}}
    out = {
        "script": rel(Path(__file__).resolve()),
        "label": LABEL,
        "inputs": ["experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/"
                   "external_validation.json"],
        "gate": {"rule": "external R^2 >= r2_threshold and external Spearman rho >= "
                         "spearman_threshold (doe_xgb.campaign.design.gate_pass)",
                 "r2_threshold": GATE_R2, "spearman_threshold": GATE_SPEARMAN},
        "dataset_order": list(DATASETS),
        "overall": overall,
        "datasets": datasets,
        "units": units,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print(f"GATE CRITERION BREAKDOWN, {overall['n_cells']} unit-by-objective cells\n")
    print(f"  external R^2 range {overall['external_r2_min']:.3f} to "
          f"{overall['external_r2_max']:.3f}; at or above {GATE_R2} in "
          f"{overall['n_cells_r2_at_or_above_threshold']} of {overall['n_cells']}")
    print(f"  failures {overall['n_failures']}: {overall['failures_by_criterion']}\n")
    for ds in DATASETS:
        q, c = datasets[ds]["quality"], datasets[ds]["cost"]
        print(f"  {ds:15s} quality {q['passes']:2d}/{q['of']} (median rho "
              f"{q['external_spearman']['median']:.3f})  cost {c['passes']:2d}/{c['of']} "
              f"(median rho {c['external_spearman']['median']:.3f})")
    print(f"\nwrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
