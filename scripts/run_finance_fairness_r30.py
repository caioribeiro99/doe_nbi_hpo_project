#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _suggest_max_workers() -> int:
    cpu = int(os.cpu_count() or 1)
    if cpu >= 12:
        return min(12, cpu - 2)
    if cpu >= 8:
        return max(1, cpu - 2)
    if cpu >= 4:
        return max(1, cpu - 1)
    return 1


def _run(cmd: List[str]) -> None:
    print("CMD:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="One-command launcher for the 3-base finance fairness R30 suite.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "fairness_suite_3bases_finance_r30.json"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "experiments" / "fairness_suite_3bases_finance_r30"))
    p.add_argument("--max-workers", type=int, default=_suggest_max_workers())
    p.add_argument("--inner-n-jobs", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--beta-step", type=float, default=0.005)
    p.add_argument("--nbi-n-starts", type=int, default=50)
    p.add_argument("--nbi-eval-k-stage1", type=int, default=180)
    p.add_argument("--refine-n-samples", type=int, default=350)
    p.add_argument("--nbi-eval-k-stage2", type=int, default=200)
    p.add_argument("--quality-floor", type=float, default=0.55)
    p.add_argument("--fairness-best-delta", type=float, default=0.02)
    p.add_argument("--replicas", type=int, default=None)
    p.add_argument("--dataset", action="append", default=None)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--skip-aggregate", action="store_true")
    args = p.parse_args()

    _run([sys.executable, str(REPO_ROOT / "scripts" / "prepare_local_fairness_layout.py")])

    if not args.skip_preflight:
        preflight_out = Path(args.out_root).resolve() / "preflight_fairness_suite.csv"
        _run([
            sys.executable,
            str(REPO_ROOT / "scripts" / "preflight_fairness_suite.py"),
            "--config", str(Path(args.config).resolve()),
            "--n-splits", str(int(args.n_splits)),
            "--out", str(preflight_out),
        ])

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_fairness_suite_parallel.py"),
        "--config", str(Path(args.config).resolve()),
        "--out-root", str(Path(args.out_root).resolve()),
        "--max-workers", str(int(args.max_workers)),
        "--inner-n-jobs", str(int(args.inner_n_jobs)),
        "--n-splits", str(int(args.n_splits)),
        "--beta-step", str(float(args.beta_step)),
        "--nbi-n-starts", str(int(args.nbi_n_starts)),
        "--nbi-eval-k-stage1", str(int(args.nbi_eval_k_stage1)),
        "--refine-n-samples", str(int(args.refine_n_samples)),
        "--nbi-eval-k-stage2", str(int(args.nbi_eval_k_stage2)),
        "--quality-floor", str(float(args.quality_floor)),
        "--fairness-best-delta", str(float(args.fairness_best_delta)),
    ]
    if args.replicas is not None:
        cmd += ["--replicas", str(int(args.replicas))]
    if args.dataset:
        for d in args.dataset:
            cmd += ["--dataset", str(d)]
    _run(cmd)

    if not args.skip_aggregate:
        _run([
            sys.executable,
            str(REPO_ROOT / "scripts" / "aggregate_fairness_suite_results.py"),
            "--config", str(Path(args.config).resolve()),
            "--out-root", str(Path(args.out_root).resolve()),
            "--quality-floor", str(float(args.quality_floor)),
        ])

    print("✅ Full 3-base finance fairness suite flow finished.")


if __name__ == "__main__":
    main()
