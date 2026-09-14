#!/usr/bin/env python
"""Stage B calibration: measure the campaign's real workload, not a sample of it.

The throughput sweep uses 17 pre-specified configurations to compare execution
layouts. Comparing layouts only needs a fixed workload; *projecting the campaign*
needs a representative one. The campaign's per-replication workload is dominated by
two sets that are known exactly and are not samples of anything:

    the 88-run design, and the 78-run external validation set.

This runs both, per dataset, at the chosen layout, and reports the per-evaluation
cost the projection should actually use.

No arm is run. Nothing here compares methods.

Usage:
  python stage_b_calibration.py --workers 14 --threads 1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"

_spec = importlib.util.spec_from_file_location("pilot", HERE / "pilot_stage_a_screening.py")
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)

import sys
sys.path.insert(0, str(REPO / "src"))
from doe_xgb.campaign import evaluator            # importable: workers are spawned
from doe_xgb.campaign.sampling import Sampler

PARAMS, INTS = pilot.PARAMS, pilot.INTS


def workload() -> pd.DataFrame:
    """Exactly what a replication evaluates before any optimizer runs."""
    d = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    d.columns = [str(c).strip().strip('"') for c in d.columns]
    design = d[PARAMS].copy()
    design["label"] = [f"design_{i}" for i in range(len(design))]
    ext = pilot.external_points(10**6, seed=1, kind="complement")[PARAMS].copy()
    ext["label"] = [f"external_{i}" for i in range(len(ext))]
    return pd.concat([design, ext], ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["magic", "spambase", "adult", "bank_marketing"])
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()

    work = workload()
    rows = work.to_dict("records")
    print(f"Stage B calibration: {len(work)} real evaluations per dataset "
          f"(88 design + 78 external) at {args.workers}w x {args.threads}t\n")
    print(f"{'dataset':16s}{'evals':>7}{'wall s':>9}{'s/eval':>9}{'evals/s':>9}"
          f"{'peakGB':>8}{'swapMB':>8}")

    report = {"layout": {"workers": args.workers, "threads": args.threads},
              "workload": {"design_runs": 88, "external_runs": len(work) - 88,
                           "total": len(work)},
              "datasets": {}}
    for ds in args.datasets:
        with Sampler() as smp:
            t0 = time.perf_counter()
            if args.workers == 1:
                evaluator.init_worker(ds, args.threads, args.seed)
                _ = [evaluator.evaluate_row(r) for r in rows]
            else:
                ctx = mp.get_context("spawn")
                with ctx.Pool(args.workers, initializer=evaluator.init_worker,
                              initargs=(ds, args.threads, args.seed)) as pool:
                    _ = pool.map(evaluator.evaluate_row, rows, chunksize=1)
            wall = time.perf_counter() - t0
        s = smp.summary()
        report["datasets"][ds] = {
            "evaluations": len(rows), "wall_seconds": round(wall, 1),
            "seconds_per_evaluation": round(wall / len(rows), 4),
            "evaluations_per_second": round(len(rows) / wall, 3),
            "system": s}
        print(f"{ds:16s}{len(rows):7d}{wall:9.1f}{wall/len(rows):9.4f}"
              f"{len(rows)/wall:9.3f}{s.get('peak_rss_gb',0):8.2f}"
              f"{s.get('swap_used_delta_mb',0):8.1f}")

    per = [v["seconds_per_evaluation"] for v in report["datasets"].values()]
    mean_s = float(np.mean(per))
    report["mean_seconds_per_evaluation"] = round(mean_s, 4)
    report["panel_mean_evaluations_per_second"] = round(1.0 / mean_s, 3)

    # campaign projection, per protocol/budget_accounting.md
    R, D, design_n, valid_n, cand, anchor_per_obj, n_comp = 30, 4, 88, 78, 20, 100, 5
    proj = {}
    for q in (2, 3):
        arms_total = design_n + valid_n + 4 * cand + anchor_per_obj * q
        arm_max = design_n + valid_n + anchor_per_obj * q + cand
        comp_total = n_comp * arm_max
        per_rep = arms_total + comp_total
        evals = per_rep * R * D
        unmatched = 10 * arm_max * D          # one replication per dataset
        proj[f"q{q}"] = {
            "B_total_solution": {"HISTORICAL-WS": design_n + cand,
                                 "WS-S": design_n + valid_n + cand,
                                 "NBI-S": design_n + valid_n + cand,
                                 "NBI-R": arm_max},
            "comparator_budget": arm_max,
            "evaluations_per_replication_per_dataset": per_rep,
            "campaign_evaluations": evals,
            "campaign_hours": round(evals * mean_s / 3600, 2),
            "campaign_days": round(evals * mean_s / 86400, 3),
            "unmatched_nsga2_evaluations": unmatched,
            "unmatched_nsga2_hours": round(unmatched * mean_s / 3600, 2),
            "total_days_including_unmatched":
                round((evals + unmatched) * mean_s / 86400, 3),
        }
    report["projection"] = proj
    print(f"\nmean {mean_s:.4f} s/evaluation across the panel "
          f"({1/mean_s:.2f} evaluations/s)\n")
    for q in (2, 3):
        p = proj[f"q{q}"]
        print(f"  q={q}: {p['campaign_evaluations']:,} evals -> "
              f"{p['campaign_hours']:.1f} h ({p['campaign_days']:.2f} days); "
              f"with the unmatched NSGA-II run {p['total_days_including_unmatched']:.2f} days")

    (OUT / "audits").mkdir(parents=True, exist_ok=True)
    (OUT / "audits" / "stage_b_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'audits'/'stage_b_calibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
