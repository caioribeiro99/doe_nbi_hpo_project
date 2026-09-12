#!/usr/bin/env python
"""Checkpointed, evaluation-matched NSGA-II baseline over the frozen R = 30 artifacts.

Reads the cached out-of-fold probabilities of the frozen benchmark and runs one NSGA-II
per dataset x replication at an objective-evaluation budget matched to what NBI-C
actually consumed in that same replication. Writes to a SEPARATE tree
(``experiments/pco213_postwork_nsga2``); the frozen benchmark directory is never written.

Replications are independent, so they are executed in a process pool. The objective
evaluator parallelizes across weight-chunks, which a 66-member population cannot exploit,
so per-process ``n_jobs=1`` plus several processes is far faster than the reverse.

Usage:
    python scripts/pco213_run_nsga2_baseline.py --workers 8
    python scripts/pco213_run_nsga2_baseline.py --datasets uci_credit --reps 1   # pilot
    python scripts/pco213_run_nsga2_baseline.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mixens.nsga2_baseline import POP_SIZE, n_gen_for_budget, run_nsga2, save_result  # noqa: E402

BENCH = REPO / "experiments" / "pco213_postwork_benchmark"
OUT = REPO / "experiments" / "pco213_postwork_nsga2"
CONFIG = REPO / "papers" / "surrogate_nbi_ensemble" / "nsga2_preregistered_config.json"
DATASETS = ["santander", "bnp", "porto", "uci_credit"]


def load_budgets() -> dict:
    """Per dataset x replication real-objective evaluation count consumed by NBI-C."""
    n = pd.read_csv(REPO / "reports" / "pco213_postwork_benchmark" / "tables" / "nbi_runs.csv")
    c = n[n.variant == "C"]
    return {ds: {int(r.rep): int(r.n_real_objective_evals) for r in g.itertuples()}
            for ds, g in c.groupby("dataset")}


def load_costs(ds: str, rep: int) -> np.ndarray:
    """Measured per-model inference cost (ms / 1k rows) for this replication."""
    meta = json.loads((BENCH / ds / f"rep_{rep:02d}" / "oof_meta.json").read_text())
    names = meta["model_names"] if "model_names" in meta else meta["models"]
    cost = meta["cost_ms_per_1k"]
    return np.array([float(cost[m]) for m in names], dtype=float)


def one_run(ds: str, rep: int, target: int) -> dict:
    """Execute (or skip) a single dataset x replication NSGA-II run."""
    out_dir = OUT / ds / f"rep_{rep:02d}"
    done = out_dir / "nsga2_summary.json"
    if done.exists():
        meta = json.loads(done.read_text())
        meta["skipped"] = True
        return meta
    z = np.load(BENCH / ds / f"rep_{rep:02d}" / "oof.npz")
    P, y = z["P"], z["y_train"]
    costs = load_costs(ds, rep)
    res = run_nsga2(P, y, costs, dataset=ds, rep=rep, target_evals=target,
                    pop_size=POP_SIZE, n_jobs=1)
    save_result(res, out_dir)
    meta = json.loads(done.read_text())
    meta["skipped"] = False
    return meta


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scale-budget", type=float, default=1.0,
                    help="DEBUG ONLY: multiply the matched budget (smoke runs). Never use for the real run.")
    args = ap.parse_args()

    datasets = [d for d in args.datasets.split(",") if d]
    budgets = load_budgets()
    OUT.mkdir(parents=True, exist_ok=True)

    work = []
    for ds in datasets:
        for rep in range(args.reps):
            target = int(round(budgets[ds][rep] * args.scale_budget))
            if (OUT / ds / f"rep_{rep:02d}" / "nsga2_summary.json").exists():
                continue
            work.append((ds, rep, target))

    total_target = sum(w[2] for w in work)
    print(f"[plan] {len(work)} pending runs across {len(datasets)} datasets; "
          f"{total_target:,} target evaluations "
          f"({sum(n_gen_for_budget(w[2]) for w in work):,} generations at pop {POP_SIZE})",
          flush=True)
    if args.dry_run:
        for ds, rep, t in work[:10]:
            print(f"  pending {ds} rep {rep:02d}: target {t:,} evals, "
                  f"{n_gen_for_budget(t):,} generations")
        if len(work) > 10:
            print(f"  ... and {len(work)-10} more")
        return 0
    if not work:
        print("[plan] nothing to do; all runs are complete")
        return 0

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "config_file": str(CONFIG.relative_to(REPO)),
        "config_sha_note": "the pre-specified configuration was committed before this run",
        "frozen_inputs": {"tag": "pco213-postwork-r30", "dir": str(BENCH.relative_to(REPO))},
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "platform": platform.platform(),
        },
        "pop_size": POP_SIZE, "workers": args.workers, "scale_budget": args.scale_budget,
        "runs": {},
    }
    try:
        import pymoo
        manifest["environment"]["pymoo"] = pymoo.__version__
    except Exception:
        pass

    t0 = time.perf_counter()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_run, ds, rep, t): (ds, rep) for ds, rep, t in work}
        for fut in as_completed(futs):
            ds, rep = futs[fut]
            try:
                meta = fut.result()
                done += 1
                manifest["runs"][f"{ds}/rep_{rep:02d}"] = meta
                el = time.perf_counter() - t0
                print(f"[{done}/{len(work)}] {ds} rep {rep:02d}: "
                      f"{meta['actual_evals']:,}/{meta['target_evals']:,} evals "
                      f"({meta['budget_ratio_actual_over_target']:.4f}), "
                      f"{meta['n_nondominated']} nondominated, {meta['seconds']:.0f}s "
                      f"| elapsed {el/3600:.2f}h", flush=True)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                manifest["runs"][f"{ds}/rep_{rep:02d}"] = {"error": repr(exc)}
                print(f"[FAIL] {ds} rep {rep:02d}: {exc!r}", flush=True)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["wall_clock_hours"] = (time.perf_counter() - t0) / 3600
    manifest["completed"] = done
    manifest["failed"] = failed
    (OUT / "nsga2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {done} completed, {failed} failed, "
          f"{manifest['wall_clock_hours']:.2f} h wall clock", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
