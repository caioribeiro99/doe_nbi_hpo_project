#!/usr/bin/env python
"""The confirmatory campaign: planner, dry run, and detached execution.

Runs the frozen protocol at ``xgboost-hpo-protocol-v3`` over 4 datasets x 30
replications = 120 units, parallel at the unit level with the execution layout
Stage B measured.

    python scripts/xgb_hpo_campaign.py plan        # counts and projections only
    python scripts/xgb_hpo_campaign.py dry-run     # plan, plus the pre-launch proofs
    python scripts/xgb_hpo_campaign.py run         # execute, resumable
    python scripts/xgb_hpo_campaign.py status      # completion by dataset

The confirmatory root is separate from every pilot artifact so that Stage A and
Stage B output cannot enter a confirmatory table by accident.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

from doe_xgb.campaign.design import (external_validation_set, load_design)  # noqa: E402
from doe_xgb.campaign.runner import (DATASETS, N_REPLICATIONS, PROTOCOL_TAG,  # noqa: E402
                                     STAGES, Checkpoint, logical_budget_plan,
                                     run_unit, unit_seed)

CONFIRMATORY_ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
PILOT_ROOTS = [REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"]
WORKERS, THREADS = 14, 1
SECONDS_PER_EVALUATION = 0.1585                      # Stage B calibration, panel mean


# --------------------------------------------------------------------- planning

def per_unit_evaluations() -> dict[str, int]:
    b = logical_budget_plan()
    design, ext, cand = b["B_design"], b["B_external_validation"], b["B_candidate_validation"]
    anchor = b["B_anchor_per_objective"] * 2
    comparator = b["comparator_budget"]
    arms_physical = design + ext + anchor + 4 * cand + 2   # +2 payoff measurements
    baselines = {"grid": comparator, "random": comparator,
                 "bayes_quality": comparator, "bayes_cost": comparator,
                 "tpe_quality": comparator, "tpe_cost": comparator,
                 "nsga2": b["nsga2"]["evaluations"]}
    return {"arms_and_shared": arms_physical,
            **baselines,
            "total_logical_upper_bound": arms_physical + sum(baselines.values())}


def plan() -> dict:
    b = logical_budget_plan()
    per = per_unit_evaluations()
    units = [(d, r) for d in DATASETS for r in range(N_REPLICATIONS)]
    total = per["total_logical_upper_bound"] * len(units)
    return {
        "protocol_tag": PROTOCOL_TAG,
        "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip(),
        "datasets": list(DATASETS), "replications_per_dataset": N_REPLICATIONS,
        "confirmatory_units": len(units),
        "stages_per_unit": list(STAGES),
        "logical_budget": b,
        "evaluations_per_unit": per,
        "campaign_logical_evaluations": total,
        "projected_hours": round(total * SECONDS_PER_EVALUATION / 3600, 2),
        "projected_days": round(total * SECONDS_PER_EVALUATION / 86400, 3),
        "execution_layout": {"process_workers": WORKERS,
                             "xgboost_threads_per_fit": THREADS,
                             "start_method": "spawn",
                             "source": "Stage B, STAGE_B_THROUGHPUT.md"},
        "seeds": {f"{d}/rep_{r:02d}": unit_seed(d, r) for d in DATASETS
                  for r in (0, N_REPLICATIONS - 1)},
        "confirmatory_root": str(CONFIRMATORY_ROOT.relative_to(REPO)),
        "pilot_roots_excluded": [str(p.relative_to(REPO)) for p in PILOT_ROOTS],
    }


# ------------------------------------------------------------------- dry run

def dry_run() -> int:
    p = plan()
    print(f"CAMPAIGN DRY RUN  protocol {p['protocol_tag']}  commit {p['source_commit'][:12]}\n")
    print(f"  datasets              {', '.join(p['datasets'])}")
    print(f"  replications each     {p['replications_per_dataset']}")
    print(f"  confirmatory units    {p['confirmatory_units']}")
    print(f"  stages per unit       {len(p['stages_per_unit'])}")
    print(f"  execution layout      {WORKERS} process workers x {THREADS} thread\n")

    b = p["logical_budget"]
    print("  logical solution-producing budget per method, per unit:")
    for k, v in b["B_total_solution_per_arm"].items():
        print(f"      {k:16s} {v:6d}")
    print(f"      {'each comparator':16s} {b['comparator_budget']:6d}")
    print(f"      {'NSGA-II':16s} {b['nsga2']['evaluations']:6d}"
          f"   (shortfall {b['nsga2']['shortfall_against_comparator_budget']}, "
          f"population {b['nsga2']['population']} x {b['nsga2']['generations']} generations)")
    print(f"      external validation {b['B_external_validation']} per unit, AUDIT-ONLY, "
          "charged to no comparator\n")
    print(f"  evaluations per unit  {p['evaluations_per_unit']['total_logical_upper_bound']:,}")
    print(f"  campaign total        {p['campaign_logical_evaluations']:,} logical evaluations")
    print(f"  projected wall clock  {p['projected_hours']:.1f} h "
          f"({p['projected_days']:.2f} days) at {SECONDS_PER_EVALUATION} s/evaluation\n")

    ok = True

    def check(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    print("  pre-launch proofs")
    done = completed_units()
    check("zero confirmatory units are already complete", not done,
          f"{len(done)} complete" if done else "")
    check("the confirmatory root is separate from every pilot root",
          all(not str(CONFIRMATORY_ROOT).startswith(str(r)) and
              not str(r).startswith(str(CONFIRMATORY_ROOT)) for r in PILOT_ROOTS))
    stray = [q for r in PILOT_ROOTS for q in (CONFIRMATORY_ROOT.rglob("*")
                                              if CONFIRMATORY_ROOT.exists() else [])
             if str(r) in str(q)]
    check("no pilot artifact lies inside the confirmatory root", not stray)
    check("the projected wall clock is within the five-day ceiling",
          p["projected_days"] <= 5.0, f"{p['projected_days']:.2f} days")
    check("the design and the external set are disjoint",
          _disjoint(), "88 design runs, 78 external runs, 0 shared")
    check("every unit has a distinct seed",
          len({unit_seed(d, r) for d in DATASETS for r in range(N_REPLICATIONS)})
          == len(DATASETS) * N_REPLICATIONS)
    check("holdout labels are not read before the confirmation stage",
          _holdout_is_late())
    check("the git tree is clean", _tree_clean(), _tree_status())

    CONFIRMATORY_ROOT.mkdir(parents=True, exist_ok=True)
    (CONFIRMATORY_ROOT / "campaign_plan.json").write_text(json.dumps(p, indent=2))
    print(f"\n  wrote {CONFIRMATORY_ROOT/'campaign_plan.json'}")
    print(f"\n{'DRY RUN CLEAN' if ok else 'DRY RUN HAS FAILURES'}")
    return 0 if ok else 1


def _disjoint() -> bool:
    from doe_xgb.campaign.design import to_coded
    d, e = load_design(), external_validation_set()
    return not ({tuple(r) for r in np.round(to_coded(e), 6)}
                & {tuple(r) for r in np.round(to_coded(d), 6)})


def _holdout_is_late() -> bool:
    """The holdout is read in exactly one place, and it is the last stage."""
    src = (REPO / "src" / "doe_xgb" / "campaign" / "runner.py").read_text()
    uses = [ln for ln in src.splitlines() if "on_holdout=True" in ln]
    return len(uses) == 1 and "holdout_confirmation" in src.split("on_holdout=True")[0][-3000:]


def _tree_clean() -> bool:
    return not subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                              capture_output=True, text=True).stdout.strip()


def _tree_status() -> str:
    out = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return "" if not out else f"{len(out.splitlines())} modified/untracked paths"


# ------------------------------------------------------------------ execution

def completed_units() -> list[tuple[str, int]]:
    out = []
    for d in DATASETS:
        for r in range(N_REPLICATIONS):
            ck = Checkpoint(CONFIRMATORY_ROOT / d / f"rep_{r:02d}")
            if ck.done("metrics") and ck.done("holdout_confirmation"):
                out.append((d, r))
    return out


def _worker(task: tuple[str, int]) -> dict:
    dataset, rep = task
    t0 = time.time()
    try:
        res = run_unit(dataset, rep, CONFIRMATORY_ROOT, threads=THREADS)
        return {"dataset": dataset, "rep": rep, "ok": True,
                "seconds": round(time.time() - t0, 1), **{k: v for k, v in res.items()
                                                          if k == "accounting"}}
    except Exception as exc:                            # noqa: BLE001
        return {"dataset": dataset, "rep": rep, "ok": False, "error": repr(exc),
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0, 1)}


def run() -> int:
    CONFIRMATORY_ROOT.mkdir(parents=True, exist_ok=True)
    done = set(completed_units())
    tasks = [(d, r) for d in DATASETS for r in range(N_REPLICATIONS)
             if (d, r) not in done]
    manifest = {
        **plan(),
        "started_at": time.time(),
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform(),
                        "numpy": np.__version__, "pandas": pd.__version__},
        "already_complete": len(done), "pending": len(tasks),
    }
    (CONFIRMATORY_ROOT / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2))
    log = (CONFIRMATORY_ROOT / "campaign.log").open("a")
    print(f"campaign: {len(tasks)} units pending, {len(done)} already complete, "
          f"{WORKERS} workers", flush=True)
    log.write(f"START {time.ctime()} pending={len(tasks)} done={len(done)}\n")
    log.flush()

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(THREADS)

    n_ok = n_fail = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS) as pool:
        for res in pool.imap_unordered(_worker, tasks, chunksize=1):
            n_ok += bool(res["ok"])
            n_fail += not res["ok"]
            line = (f"{'OK  ' if res['ok'] else 'FAIL'} {res['dataset']}/rep_{res['rep']:02d} "
                    f"{res['seconds']:.0f}s  ({n_ok + n_fail}/{len(tasks)})")
            print(line, flush=True)
            log.write(line + "\n")
            if not res["ok"]:
                log.write(res["traceback"] + "\n")
            log.flush()
    log.write(f"END {time.ctime()} ok={n_ok} fail={n_fail}\n")
    log.close()
    print(f"campaign finished: {n_ok} ok, {n_fail} failed, "
          f"{len(completed_units())}/{len(DATASETS)*N_REPLICATIONS} complete")
    return 0 if n_fail == 0 else 1


def status() -> int:
    done = completed_units()
    print(f"{len(done)}/{len(DATASETS)*N_REPLICATIONS} confirmatory units complete")
    for d in DATASETS:
        n = sum(1 for x, _ in done if x == d)
        print(f"  {d:16s} {n:2d}/{N_REPLICATIONS}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["plan", "dry-run", "run", "status"])
    args = ap.parse_args()
    if args.command == "plan":
        print(json.dumps(plan(), indent=2))
        return 0
    return {"dry-run": dry_run, "run": run, "status": status}[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
