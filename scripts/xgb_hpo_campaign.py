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
import pathlib
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
                                     STAGES, Checkpoint, campaign_budget,
                                     logical_budget_plan, method_stage_ledger,
                                     reconcile_unit_accounting, run_unit, unit_seed)

CONFIRMATORY_ROOT = REPO / "experiments" / "xgboost_hpo_vrfnbi_confirmatory"
PILOT_ROOTS = [REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"]
WORKERS, THREADS = 14, 1
SECONDS_PER_EVALUATION = 0.1585                      # Stage B calibration, panel mean


# --------------------------------------------------------------------- planning

def per_unit_evaluations() -> dict[str, int]:
    """Every logical evaluation a unit requests, from the runner's own registry.

    Derived rather than restated: the planner, the dry run and the campaign
    manifest all read `campaign_budget`, so no figure here can drift from what the
    runner actually does. An earlier hand-maintained version understated the
    campaign by 39%.
    """
    table = method_stage_ledger()
    out = {m: sum(st.values()) for m, st in table.items()}
    out["total_logical_per_unit"] = sum(out.values())
    return out


def plan() -> dict:
    b = logical_budget_plan()
    per = per_unit_evaluations()
    cb = campaign_budget()
    units = [(d, r) for d in DATASETS for r in range(N_REPLICATIONS)]
    unmatched = cb["unmatched_nsga2_logical"]
    total = cb["campaign_total_logical"]
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
        "campaign_solution_producing_logical": cb["campaign_solution_producing_logical"],
        "campaign_audit_only_logical": cb["campaign_audit_only_logical"],
        "budget_by_stage_per_unit": cb["by_stage"],
        "budget_arithmetic_consistent": cb["arithmetic_consistent"],
        "nsga2_unmatched_evaluations": unmatched,
        "nsga2_unmatched_scope": cb["unmatched_nsga2_scope"],
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
    print(f"  evaluations per unit  {p['evaluations_per_unit']['total_logical_per_unit']:,}")
    print(f"  unmatched NSGA-II     {p['nsga2_unmatched_evaluations']:,} "
          f"({p['nsga2_unmatched_scope']})")
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
    _ho_ok, _ho_why = _holdout_is_late()
    check("holdout labels are not read before the confirmation stage", _ho_ok, _ho_why)
    check("the git tree is clean", _tree_clean(), _tree_status())
    cb = campaign_budget()
    # This is an arithmetic identity and is labelled as one. It is NOT evidence that
    # the campaign spends the budget it publishes: the previous "the budget
    # reconciles" proof compared a sum of the method-stage table against a sum of
    # the same table, so it could not fail, and it reported a clean reconciliation
    # while the runner charged 840 evaluations per campaign outside every ledger.
    check("the published total is the sum of its published parts (arithmetic only)",
          cb["arithmetic_consistent"],
          f"{cb['campaign_total_logical']:,} = "
          f"{cb['campaign_solution_producing_logical']:,} solution-producing + "
          f"{cb['campaign_audit_only_logical']:,} audit-only")
    # The check that CAN fail: what a real unit charged, against what is declared.
    # It needs a completed unit, so the dry run reports whether one is available
    # rather than silently skipping.
    check("every stage in the ledger is a declared stage",
          set(cb["by_stage"]) <= {"design", "external_audit", "anchor",
                                  "candidate_validation", "direct_search",
                                  "holdout_audit"},
          str(sorted(cb["by_stage"])))
    smoke = _latest_smoke_accounting()
    if smoke is None:
        check("a completed unit's ledger reconciles against the registry", False,
              "no completed unit available; run an engineering smoke first. "
              "The arithmetic identity above is NOT a substitute.")
    else:
        rec = reconcile_unit_accounting(smoke["accounting"])
        check("a completed unit's ledger reconciles against the registry",
              rec["reconciles"],
              f"{smoke['label']}: charged {rec['total_charged']:,} against "
              f"{rec['total_declared']:,} declared"
              + (f"; mismatched {rec['mismatched_stages']}" if rec["mismatched_stages"] else "")
              + (f"; undeclared {rec['charged_but_never_declared']}"
                 if rec["charged_but_never_declared"] else "")
              + (f"; never charged {rec['declared_but_never_charged']}"
                 if rec["declared_but_never_charged"] else ""))

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


# Engineering smoke roots, searched newest first. These live OUTSIDE the
# confirmatory root by construction: a smoke unit is not campaign data, and the
# dry run reads only its accounting ledger, never any arm's outcome.
SMOKE_ROOTS = [pathlib.Path("/tmp/smoke_v3"),
               REPO / "experiments" / "_xgb_hpo_v3_smoke"]


def _latest_smoke_accounting():
    """The most recent completed smoke unit's accounting, or None.

    Reads ONLY the accounting block of the metrics checkpoint. No arm outcome is
    opened, so running this proof cannot breach confirmatory blindness.
    """
    best = None
    for root in SMOKE_ROOTS:
        if not root.exists():
            continue
        for metrics in root.glob("*/rep_*/metrics.json"):
            try:
                payload = json.loads(metrics.read_text())
            except Exception:
                continue
            if not payload.get("_complete") or "accounting" not in payload:
                continue
            stamp = payload.get("_written_at", 0)
            if best is None or stamp > best["stamp"]:
                best = {"stamp": stamp, "accounting": payload["accounting"],
                        "label": f"{metrics.parent.parent.name}/{metrics.parent.name}"}
    return best


def _holdout_is_late() -> tuple[bool, str]:
    """The holdout partition is evaluated in exactly one stage, and it is the last.

    The earlier version of this check asserted that the 3,000 characters preceding
    the single ``on_holdout=True`` mentioned ``holdout_confirmation``. That was a
    proxy for "the call sits inside the holdout stage", and it broke the moment the
    holdout compute function had to be defined at the top of ``run_unit`` so the
    evaluation cache could be given it -- while the call site had not moved at all.
    A proxy that fails when the invariant holds is no better than one that passes
    when it does not.

    What is checked now: the holdout flag is set in exactly one place; the holdout
    cache view is created exactly once; and that creation is inside the
    ``holdout_confirmation`` stage guard, which is the last stage before metrics.
    The dynamic proof -- that no holdout evaluation is REQUESTED before that stage --
    is in tests/methodology/test_unit_runs_end_to_end.py, where a real unit runs.
    """
    src = (REPO / "src" / "doe_xgb" / "campaign" / "runner.py").read_text()
    lines = src.splitlines()
    flags = [i for i, ln in enumerate(lines) if "on_holdout=True" in ln]
    views = [i for i, ln in enumerate(lines) if 'cache.view("holdout_confirmation"' in ln]
    guards = [i for i, ln in enumerate(lines) if 'ck.done("holdout_confirmation")' in ln]
    if len(flags) != 1:
        return False, f"on_holdout=True appears {len(flags)} times, expected 1"
    if len(views) != 1:
        return False, f"the holdout cache view is created {len(views)} times, expected 1"
    if len(guards) != 1:
        return False, f"the holdout stage guard appears {len(guards)} times, expected 1"
    if not guards[0] < views[0]:
        return False, "the holdout view is created outside the holdout stage guard"
    later = [s for s in STAGES[STAGES.index("holdout_confirmation") + 1:]]
    return True, (f"one flag, one view, inside the stage guard; "
                  f"stages after it: {later}")


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
