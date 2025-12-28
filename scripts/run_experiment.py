#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

# Allow running scripts directly without installing the package
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from doe_xgb.seeds import generate_seeds
from doe_xgb.tracking import build_replica_dir, write_manifest


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() != "" else default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def main() -> None:
    # Load .env (CLI must override env -> argparse defaults come from env)
    load_dotenv(REPO_ROOT / ".env", override=False)

    p = argparse.ArgumentParser(
        description="Run the full experiment: multiple replicas, each calling scripts/run_replica.py",
    )

    p.add_argument(
        "--dataset",
        default=os.getenv("DATASET_PATH"),
        help="Path to dataset (default from env DATASET_PATH)",
    )
    p.add_argument(
        "--design",
        default=os.getenv("DESIGN_PATH"),
        help="Path to hyperparameter design CSV (default from env DESIGN_PATH)",
    )
    p.add_argument(
        "--out-root",
        default=_env_str("OUT_ROOT", "experiments"),
        help="Root output directory (default from env OUT_ROOT)",
    )
    p.add_argument(
        "--target",
        default=_env_str("TARGET_COL", "y"),
        help="Target column in the dataset (default from env TARGET_COL)",
    )

    p.add_argument(
        "--n-replicas",
        type=int,
        default=_env_int("N_REPLICAS", 30),
        help="Number of replicas (default from env N_REPLICAS)",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=_env_int("SEED_BASE", 42),
        help="Base seed used to generate replica seeds (default from env SEED_BASE)",
    )

    # Budget for benchmarks: 0 (or <=0) means AUTO fairness budget per replica
    p.add_argument(
        "--budget",
        type=int,
        default=_env_int("BENCHMARK_BUDGET", 0),
        help="Benchmark budget (<=0 means auto-fairness, default from env BENCHMARK_BUDGET)",
    )

    # NBI knobs
    p.add_argument(
        "--beta-step",
        type=float,
        default=_env_float("NBI_BETA_STEP", 0.02),
        help="Beta grid step for NBI (default from env NBI_BETA_STEP)",
    )
    p.add_argument(
        "--nbi-eval-k",
        type=int,
        default=_env_int("NBI_EVAL_K", 0),
        help="Evaluate top-K NBI candidates (<=0 means evaluate all; default from env NBI_EVAL_K)",
    )
    p.add_argument(
        "--nbi-n-starts",
        type=int,
        default=_env_int("NBI_N_STARTS", 10),
        help="Number of random restarts for NBI optimization per beta (default from env NBI_N_STARTS)",
    )
    p.add_argument(
        "--nbi-constrain-pred-range",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("NBI_CONSTRAIN_PRED_RANGE", True),
        help="Constrain NBI to predicted score ranges (default from env NBI_CONSTRAIN_PRED_RANGE)",
    )

    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("CONTINUE_ON_ERROR", True),
        help="If true, keeps running even if a replica fails (default from env CONTINUE_ON_ERROR)",
    )

    args = p.parse_args()

    if args.dataset is None:
        p.error("--dataset is required (or set DATASET_PATH in .env)")
    if args.design is None:
        p.error("--design is required (or set DESIGN_PATH in .env)")

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    if not design_path.exists():
        raise FileNotFoundError(design_path)

    seeds: List[int] = generate_seeds(initial_seed=int(args.seed_base), n_replicas=int(args.n_replicas))

    # Experiment-level manifest
    exp_dir = (out_root / build_replica_dir(out_root, dataset_path, design_path, replica=1).parent).resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    exp_manifest_path = exp_dir / "experiment_manifest.json"
    write_manifest(
        exp_manifest_path,
        extra={
            "dataset": str(dataset_path),
            "design": str(design_path),
            "target": str(args.target),
            "n_replicas": int(args.n_replicas),
            "seed_base": int(args.seed_base),
            "seeds": seeds,
            "bench_budget": int(args.budget),
            "nbi_beta_step": float(args.beta_step),
            "nbi_eval_k": int(args.nbi_eval_k),
            "nbi_n_starts": int(args.nbi_n_starts),
            "nbi_constrain_pred_range": bool(args.nbi_constrain_pred_range),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    print(f"==> Experiment: {args.n_replicas} replicas")
    print(f"- dataset: {dataset_path}")
    print(f"- design:  {design_path}")
    print(f"- out:     {exp_dir}")
    print(f"- target:  {args.target}")
    print(f"- beta step: {args.beta_step}")
    print(f"- nbi eval k: {args.nbi_eval_k} (<=0 means all)")
    print(f"- benchmark budget: {args.budget} (<=0 means auto-fairness)")

    results: List[Dict[str, Any]] = []
    start_all = time.perf_counter()

    for i, seed in enumerate(seeds, start=1):
        print(f"\n==> Replica {i:02d}/{len(seeds)} (seed={seed})")

        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_replica.py"),
            "--dataset",
            str(dataset_path),
            "--design",
            str(design_path),
            "--replica",
            str(i),
            "--seed",
            str(seed),
            "--out-root",
            str(out_root),
            "--target",
            str(args.target),
            "--budget",
            str(int(args.budget)),
            "--beta-step",
            str(float(args.beta_step)),
            "--nbi-eval-k",
            str(int(args.nbi_eval_k)),
            "--nbi-n-starts",
            str(int(args.nbi_n_starts)),
        ]
        if args.nbi_constrain_pred_range:
            cmd.append("--nbi-constrain-pred-range")
        else:
            cmd.append("--no-nbi-constrain-pred-range")

        try:
            # Let run_replica stream output to the terminal; it will also write its own run_replica.log
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

            out_dir = build_replica_dir(out_root, dataset_path, design_path, replica=i)
            results.append(
                {
                    "replica": i,
                    "seed": seed,
                    "status": "ok",
                    "out_dir": str(out_dir),
                    "log": str(out_dir / "run_replica.log"),
                }
            )

        except subprocess.CalledProcessError as e:
            out_dir = build_replica_dir(out_root, dataset_path, design_path, replica=i)
            print(f"✗ Replica {i:02d} failed: run_replica.py exited with code {e.returncode}")
            print(f"  - Logs: {out_dir / 'run_replica.log'}")
            results.append(
                {
                    "replica": i,
                    "seed": seed,
                    "status": "failed",
                    "returncode": e.returncode,
                    "out_dir": str(out_dir),
                    "log": str(out_dir / "run_replica.log"),
                }
            )
            if not args.continue_on_error:
                break

    elapsed = time.perf_counter() - start_all

    # Save replica list
    (exp_dir / "replica_runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n✅ Experiment runner finished")
    print(f"- Manifest: {exp_manifest_path}")
    print(f"- Runs:     {exp_dir / 'replica_runs.json'}")
    print(f"- Elapsed:  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
