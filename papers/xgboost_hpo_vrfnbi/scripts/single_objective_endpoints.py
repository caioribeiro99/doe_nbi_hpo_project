#!/usr/bin/env python
"""Single-objective endpoints of the Bayesian-optimization and TPE runs (Supplement S13.1).

The frozen protocol (EXPERIMENT_PROTOCOL.md section 3, resolution of A4) declares that
the four single-objective comparators (BAYES-QUALITY, BAYES-COST, TPE-QUALITY,
TPE-COST) produce single-objective optima, not fronts, and are reported on their own
endpoints only. Every unit's metrics stage stored those endpoints
(metrics_by_method.json, "single_objective_endpoints"); no analysis summarized them.
This script does, per dataset over the 30 replications:

  * best_median, best_q1, best_q3: median and quartiles of the run's endpoint, the
    best value of its own objective (f_quality or f_cost, both minimized) among its
    real evaluations (386 per run);
  * core_best_median: median over replications of the best value of the same
    objective among that replication's 288 CORE rows (88 design rows and 200
    anchor-search rows), to which no compared method contributes. Context only.

Provenance label:
retrospective (declared in the frozen protocol, section 3; computed after the campaign from stored outputs); the CORE context column is post hoc

Before summarizing, every stored endpoint is re-derived from the run's stored rows with
the frozen reference factor model, and every CORE minimum is checked against the stored
CORE front; the script stops on any disagreement. Quartiles are numpy's default linear
interpolation over the 30 per-replication values.

Inputs (read only), per dataset and replication rep_00 to rep_29 under
experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/:
  metrics_by_method.json, direct_baselines.json, design.json, empirical_anchors.json,
  reference_core.json; and papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/.
Output: papers/xgboost_hpo_vrfnbi/analysis/single_objective_endpoints.json

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:  python papers/xgboost_hpo_vrfnbi/scripts/single_objective_endpoints.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign.factor_model import load_reference_factor_model  # noqa: E402
from doe_xgb.campaign.runner import (DATASETS, N_REPLICATIONS,  # noqa: E402
                                     SINGLE_OBJECTIVE, logical_budget_plan)

ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "analysis" / "single_objective_endpoints.json"
LABEL = ("retrospective (declared in the frozen protocol, section 3; computed after the "
         "campaign from stored outputs); the CORE context column is post hoc")
# Stored rows and the stored endpoint went through the same frozen arithmetic, so they
# agree exactly; the tolerance only absorbs last-bit differences of a different BLAS.
TOL = 1e-12


def rel(p: Path) -> str:
    return p.relative_to(REPO).as_posix()


def main() -> int:
    plan = logical_budget_plan(2)
    budget = plan["comparator_budget"]
    n_anchor = 2 * plan["B_anchor_per_objective"]
    n_core = plan["B_design"] + n_anchor          # 88 design + 200 anchor-search rows
    datasets, max_dev_endpoint, max_dev_core = {}, 0.0, 0.0
    for ds in DATASETS:
        fm = load_reference_factor_model(ds)
        best = {k: [] for k in SINGLE_OBJECTIVE}
        core_best = {0: [], 1: []}
        for rep in range(N_REPLICATIONS):
            u = ROOT / ds / f"rep_{rep:02d}"
            load = lambda name: json.loads((u / name).read_text())  # noqa: E731
            endpoints = load("metrics_by_method.json")["single_objective_endpoints"]
            methods = load("direct_baselines.json")["methods"]
            for k in SINGLE_OBJECTIVE:
                ep = endpoints[k]
                j = int(ep["objective_index"])
                # the runner's own rule: quality runs on index 0, cost runs on index 1
                if j != (0 if k.endswith("quality") else 1):
                    raise SystemExit(f"{ds} rep {rep} {k}: objective index {j}")
                F = fm.objectives(pd.DataFrame(methods[k]["rows"]))
                if len(F) != budget or int(ep["n_evaluations"]) != budget:
                    raise SystemExit(f"{ds} rep {rep} {k}: {len(F)} rows, "
                                     f"{ep['n_evaluations']} stored, budget {budget}")
                dev = abs(float(F[:, j].min()) - float(ep["best_value"]))
                max_dev_endpoint = max(max_dev_endpoint, dev)
                if not dev <= TOL:
                    raise SystemExit(f"{ds} rep {rep} {k}: stored endpoint differs by {dev}")
                best[k].append(float(ep["best_value"]))
            core = load("reference_core.json")
            design_rows = load("design.json")["rows"]
            anchor_rows = load("empirical_anchors.json")["rows"]
            n_d, n_a = len(design_rows), len(anchor_rows)
            if ((n_d, n_a, n_d + n_a) != (plan["B_design"], n_anchor, n_core)
                    or (core["n_design_rows"], core["n_anchor_rows"],
                        core["n_points"]) != (n_d, n_a, n_d + n_a)):
                raise SystemExit(f"{ds} rep {rep}: CORE composition does not match")
            Fc = fm.objectives(pd.DataFrame(design_rows + anchor_rows))
            front = np.asarray(core["front"], dtype=float)
            for j in (0, 1):
                v = float(Fc[:, j].min())
                # a per-objective minimum of the 288 rows lies on their front
                dev = abs(v - float(front[:, j].min()))
                max_dev_core = max(max_dev_core, dev)
                if not dev <= TOL:
                    raise SystemExit(f"{ds} rep {rep}: CORE minimum differs by {dev}")
                core_best[j].append(v)
        blk = {}
        for k in SINGLE_OBJECTIVE:
            b = np.asarray(best[k], dtype=float)
            c = np.asarray(core_best[0 if k.endswith("quality") else 1], dtype=float)
            if not (np.isfinite(b).all() and np.isfinite(c).all()):
                raise SystemExit(f"{ds} {k}: non-finite endpoint")
            blk[k] = {"n_replications": int(len(b)),
                      "best_median": float(np.median(b)),
                      "best_q1": float(np.percentile(b, 25)),
                      "best_q3": float(np.percentile(b, 75)),
                      "core_best_median": float(np.median(c))}
        datasets[ds] = blk

    unit = "experiments/xgboost_hpo_vrfnbi_confirmatory/<dataset>/rep_XX/"
    out = {
        "script": rel(Path(__file__).resolve()),
        "label": LABEL,
        "inputs": [unit + n for n in ("metrics_by_method.json", "direct_baselines.json",
                                      "design.json", "empirical_anchors.json",
                                      "reference_core.json")]
                  + ["papers/xgboost_hpo_vrfnbi/audits/reference_factor_models/<dataset>.json"],
        "definitions": {
            "best": ("the run's stored endpoint: the minimum of its own objective "
                     "(index 0 f_quality, index 1 f_cost; both minimized) over its "
                     f"{budget} real evaluations"),
            "best_median, best_q1, best_q3": ("median and 25th/75th percentiles over "
                                              "replications, numpy linear interpolation"),
            "core_best_median": ("post hoc context: median over replications of the "
                                 "minimum of the same objective over that replication's "
                                 "288 CORE rows (88 design + 200 anchor-search rows); no "
                                 "compared method contributes to CORE"),
            "objective_map": "frozen reference factor model per dataset, applied"},
        "checks": {
            "evaluations_per_run": budget,
            "max_abs_difference_stored_vs_recomputed_endpoint": max_dev_endpoint,
            "max_abs_difference_core_minimum_vs_stored_core_front": max_dev_core},
        "datasets": datasets,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    print("SINGLE-OBJECTIVE ENDPOINTS: median (q1 to q3) | CORE best median\n")
    for ds in DATASETS:
        for k in SINGLE_OBJECTIVE:
            v = datasets[ds][k]
            print(f"  {ds:15s} {k:14s} {v['best_median']:+.3f} ({v['best_q1']:+.3f} to "
                  f"{v['best_q3']:+.3f}) | {v['core_best_median']:+.3f}")
    print(f"\nmax |stored - recomputed endpoint| = {max_dev_endpoint:.1e}; "
          f"max |CORE minimum - stored front minimum| = {max_dev_core:.1e}")
    print(f"wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
