#!/usr/bin/env python
"""Execution facts of the evaluation-matched NSGA-II baseline: worker pool, pooled runs,
the pilot run executed before the pool, and the pool's wall clock.

Provenance label: retrospective

scripts/pco213_run_nsga2_baseline.py plans one run per dataset x replication of the budget
table, skips every run whose nsga2_summary.json already exists, executes the rest in a
ProcessPoolExecutor with --workers processes and writes nsga2_manifest.json for the runs it
executed. A run whose stored summary exists but which is absent from that manifest was
therefore executed before the pool (the pilot of the runner's docstring); the pool's plan
line in nsga2.log gives the number of runs still pending when the pool started. The pool's
wall clock is the manifest's wall_clock_hours (perf_counter over the pool), cross-checked
against its created_at/finished_at stamps and the log's final line; the pilot's own run time
(its summary's seconds) lies outside it.

Inputs (read-only):
  experiments/pco213_postwork_nsga2/nsga2_manifest.json and nsga2.log (gitignored run outputs)
  experiments/pco213_postwork_nsga2/<dataset>/rep_XX/nsga2_summary.json (gitignored)
  reports/pco213_postwork_benchmark/tables/nbi_runs.csv (committed; the runner's budget table,
      NBI-C rows, defines the planned dataset x replication runs)
Output:
  reports/pco213_postwork_benchmark/nsga2/nsga2_execution_summary.json
      (timestamps are copied from the stored manifest, not generated)

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:
    python scripts/pco213_nsga2_execution_summary.py
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

SCRIPT = "scripts/pco213_nsga2_execution_summary.py"
LABEL = "retrospective"
NSGA = Path("experiments") / "pco213_postwork_nsga2"
IN_MANIFEST = NSGA / "nsga2_manifest.json"
IN_LOG = NSGA / "nsga2.log"
IN_BUDGET = Path("reports") / "pco213_postwork_benchmark" / "tables" / "nbi_runs.csv"
OUT_JSON = Path("reports") / "pco213_postwork_benchmark" / "nsga2" / "nsga2_execution_summary.json"
STAMP = "%Y-%m-%dT%H:%M:%S%z"   # time.strftime format of the runner's manifest

LOG_PLAN = re.compile(r"^\[plan\] (?P<n>\d+) pending runs")
LOG_PROGRESS = re.compile(r"^\[(?P<k>\d+)/(?P<n>\d+)\] (?P<ds>\S+) rep (?P<rep>\d+):")
LOG_DONE = re.compile(r"^\[done\] (?P<done>\d+) completed, (?P<failed>\d+) failed, (?P<hours>[\d.]+) h wall clock")
LOG_FAIL = re.compile(r"^\[FAIL\]")


def run_key(ds: str, rep: int) -> str:
    return f"{ds}/rep_{rep:02d}"


def main() -> None:
    man = json.loads((REPO / IN_MANIFEST).read_text())
    log = (REPO / IN_LOG).read_text().splitlines()
    nb = pd.read_csv(REPO / IN_BUDGET)
    planned = sorted(run_key(r.dataset, int(r.rep)) for r in nb[nb.variant == "C"].itertuples())
    stored = sorted(f"{p.parent.parent.name}/{p.parent.name}" for p in (REPO / NSGA).glob("*/rep_*/nsga2_summary.json"))
    pool = sorted(man["runs"])
    outside = sorted(set(stored) - set(pool))
    assert set(pool) <= set(stored) <= set(planned) and len(set(planned)) == len(planned)

    plans = [int(m["n"]) for m in map(LOG_PLAN.match, log) if m]
    progress = [(m["ds"], int(m["rep"]), int(m["n"])) for m in map(LOG_PROGRESS.match, log) if m]
    done = [m for m in map(LOG_DONE.match, log) if m]
    fails = sum(bool(LOG_FAIL.match(x)) for x in log)
    assert len(plans) == 1 and len(done) == 1, "expected one pool invocation in nsga2.log"
    logged_runs = sorted(run_key(ds, rep) for ds, rep, _ in progress)
    assert logged_runs == pool and all(n == plans[0] for _, _, n in progress)
    assert plans[0] == len(pool) == len(planned) - len(outside)
    assert not any(man["runs"][k].get("skipped") for k in pool) and all("error" not in man["runs"][k] for k in pool)

    t0 = datetime.strptime(man["created_at"], STAMP); t1 = datetime.strptime(man["finished_at"], STAMP)
    pilots = {}
    for k in outside:
        s = json.loads((REPO / NSGA / k / "nsga2_summary.json").read_text())
        pilots[k] = {"dataset": s["dataset"], "rep": int(s["rep"]), "seconds": float(s["seconds"]),
                     "target_evals": int(s["target_evals"]), "actual_evals": int(s["actual_evals"]),
                     "n_nondominated": int(s["n_nondominated"]), "in_pool_manifest": False,
                     "summary_existed_when_pool_was_planned": True}
    pool_seconds = sum(float(man["runs"][k]["seconds"]) for k in pool)
    pilot_seconds = sum(p["seconds"] for p in pilots.values())

    summary = {
        "script": SCRIPT, "label": LABEL,
        "inputs": [str(IN_MANIFEST), str(IN_LOG), str(NSGA / "<dataset>" / "rep_XX" / "nsga2_summary.json"), str(IN_BUDGET)],
        "workers": int(man["workers"]),
        "runs_total": len(planned),
        "runs_in_pool": len(pool),
        "runs_outside_pool": len(outside),
        "pilot_runs": outside,
        "pilot": pilots,
        "pool_wall_clock_hours": float(man["wall_clock_hours"]),
        "pool": {
            "git_commit": man["git_commit"], "workers": int(man["workers"]),
            "created_at": man["created_at"], "finished_at": man["finished_at"],
            "wall_clock_hours_manifest": float(man["wall_clock_hours"]),
            "wall_clock_hours_from_stamps": (t1 - t0).total_seconds() / 3600.0,
            "wall_clock_hours_logged": float(done[0]["hours"]),
            "completed": int(man["completed"]), "failed": int(man["failed"]), "fail_lines_logged": fails,
            "pending_runs_planned_logged": plans[0], "progress_lines_logged": len(progress)},
        "run_seconds_sum_hours": {"pool": pool_seconds / 3600.0, "pilot": pilot_seconds / 3600.0,
                                  "all": (pool_seconds + pilot_seconds) / 3600.0},
        "derivation": {
            "runs_total": "dataset x replication rows of the runner's budget table (nbi_runs.csv, NBI-C)",
            "runs_in_pool": "runs recorded in nsga2_manifest.json, all logged by the pool and none skipped",
            "pilot_runs": "stored nsga2_summary.json files absent from the pool manifest; the runner skips runs whose "
                          "summary exists, and the pool's plan line counts only the other runs as pending, so these "
                          "ran before the pool started, outside it",
            "pool_wall_clock_hours": "manifest wall_clock_hours (perf_counter over the pool); excludes the pilot run"},
    }
    (REPO / OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    (REPO / OUT_JSON).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"workers {summary['workers']}; runs in pool {summary['runs_in_pool']} of {summary['runs_total']}; "
          f"outside the pool: {outside} ({pilot_seconds:.0f} s); pool wall clock {summary['pool_wall_clock_hours']:.4f} h "
          f"(stamps {summary['pool']['wall_clock_hours_from_stamps']:.4f} h, log {summary['pool']['wall_clock_hours_logged']} h)")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
