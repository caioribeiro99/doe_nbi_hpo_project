#!/usr/bin/env python
from __future__ import annotations

"""Run an experiment end-to-end across multiple replicas.

This is the *experiment runner* that orchestrates multiple calls to
`scripts/run_replica.py` using a deterministic seed list.

Hierarchy for configuration values:
  1) CLI flags
  2) .env / environment variables
  3) Script defaults

Reproducibility:
  - Writes an experiment-level manifest with the full seed list.
  - Each replica already writes its own `manifest.json` (seed + input hashes).
  - Aggregates all replica confirmation summaries into one CSV.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure local src/ is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

from doe_xgb.seeds import generate_seeds
from doe_xgb.tracking import safe_name, sha256_file


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return int(default)
    return int(v)

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return float(default)
    return float(v)


def _env_str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v)


def _experiment_base_dir(out_root: Path, dataset_path: Path, design_path: Path) -> Path:
    ds = safe_name(dataset_path.stem)
    design = safe_name(design_path.stem)
    base = out_root / ds / design
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    # Load .env if present (python-dotenv)
    load_dotenv()

    p = argparse.ArgumentParser(description="Run the full experiment across N replicas")

    p.add_argument(
        "--dataset",
        default=_env_str("DATASET_PATH", None),
        help="Dataset path (xlsx/csv/parquet). CLI > .env:DATASET_PATH",
    )
    p.add_argument(
        "--design",
        default=_env_str("DESIGN_PATH", None),
        help="DOE design CSV path. CLI > .env:DESIGN_PATH",
    )
    p.add_argument(
        "--out_root",
        default=_env_str("EXPERIMENTS_ROOT", "experiments"),
        help="Output root folder. CLI > .env:EXPERIMENTS_ROOT",
    )
    p.add_argument(
        "--target",
        default=_env_str("TARGET_COL", "y"),
        help="Target column name. CLI > .env:TARGET_COL",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=_env_int("BUDGET", 0),
        help="Budget for benchmark methods. CLI > .env:BUDGET",
    )
    p.add_argument(
        "--beta-step",
        type=float,
        default=_env_float("NBI_BETA_STEP", 0.05),
        help="NBI beta step (e.g., 0.05 or 0.02). CLI > .env:NBI_BETA_STEP",
    )
    p.add_argument(
        "--nbi-eval-k",
        type=int,
        default=_env_int("NBI_EVAL_K", 20),
        help="Max # NBI candidates to evaluate per replica. CLI > .env:NBI_EVAL_K",
    )
    p.add_argument(
        "--nbi-n-starts",
        type=int,
        default=_env_int("NBI_N_STARTS", 10),
        help="NBI multi-starts per beta (cheap). CLI > .env:NBI_N_STARTS",
    )
    p.add_argument(
        "--n",
        "--n-replicas",
        dest="n_replicas",
        type=int,
        default=_env_int("N_REPLICAS", 30),
        help="Number of replicas. CLI > .env:N_REPLICAS",
    )
    p.add_argument(
        "--seed-base",
        "--initial-seed",
        dest="seed_base",
        type=int,
        default=_env_int("SEED_BASE", 42),
        help="Initial seed. CLI > .env:SEED_BASE",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip replicas that already have confirmation_summary.csv",
    )
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining replicas even if one fails",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and exit without running",
    )
    args = p.parse_args()

    if args.dataset is None or args.design is None:
        raise SystemExit(
            "Missing required inputs. Provide --dataset and --design, or set DATASET_PATH and DESIGN_PATH in .env"
        )

    dataset_path = Path(args.dataset)
    design_path = Path(args.design)
    out_root = Path(args.out_root)

    # Deterministic seed list
    seeds = generate_seeds(initial_seed=int(args.seed_base), n_replicas=int(args.n_replicas))

    base_dir = _experiment_base_dir(out_root, dataset_path, design_path)

    # Experiment-level manifest (written once at start, updated at end)
    exp_manifest_path = base_dir / "experiment_manifest.json"
    exp_summary_csv = base_dir / "experiment_summary.csv"

    exp_manifest: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path) if dataset_path.exists() else None},
        "design": {"path": str(design_path), "sha256": sha256_file(design_path) if design_path.exists() else None},
        "n_replicas": int(args.n_replicas),
        "seed_base": int(args.seed_base),
        "seeds": seeds,
        "target": str(args.target),
        "budget": int(args.budget),
        "out_root": str(out_root),
        "runner": {
            "python": sys.executable,
            "argv": sys.argv,
        },
        "replicas": [],
    }

    _write_json(exp_manifest_path, exp_manifest)

    # Run each replica
    replica_results: List[Dict[str, Any]] = []
    t_exp0 = time.perf_counter()

    run_replica_script = Path(__file__).resolve().parent / "run_replica.py"

    for i, seed in enumerate(seeds, start=1):
        rep_out_dir = base_dir / f"replica_{i:02d}"  # matches tracking.build_replica_dir naming
        rep_out_dir.mkdir(parents=True, exist_ok=True)

        done_marker = rep_out_dir / "confirmation_summary.csv"
        if args.resume and done_marker.exists():
            replica_results.append(
                {
                    "replica": i,
                    "seed": int(seed),
                    "status": "skipped",
                    "reason": "resume: confirmation_summary.csv already exists",
                    "out_dir": str(rep_out_dir),
                }
            )
            continue

        cmd = [
            sys.executable,
            str(run_replica_script),
            "--dataset",
            str(dataset_path),
            "--design",
            str(design_path),
            "--replica",
            str(i),
            "--seed",
            str(int(seed)),
            "--out_root",
            str(out_root),
            "--budget",
            str(int(args.budget)),
            "--beta-step",
            str(float(args.beta_step)),
            "--nbi-eval-k",
            str(int(args.nbi_eval_k)),
            "--nbi-n-starts",
            str(int(args.nbi_n_starts)),
            "--target",
            str(args.target),
        ]

        if args.dry_run:
            print("DRY RUN:", " ".join(cmd))
            continue

        print(f"\n==> Replica {i:02d}/{len(seeds)} (seed={seed})")

        log_path = rep_out_dir / "run_replica.log"
        t0 = time.perf_counter()

        try:
            with log_path.open("w", encoding="utf-8") as f:
                f.write(f"# Command: {' '.join(cmd)}\n")
                f.write(f"# Started: {datetime.now().isoformat(timespec='seconds')}\n\n")
                f.flush()

                proc = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            dt = float(time.perf_counter() - t0)

            if proc.returncode != 0:
                raise RuntimeError(f"run_replica.py failed with exit code {proc.returncode}")

            replica_results.append(
                {
                    "replica": i,
                    "seed": int(seed),
                    "status": "ok",
                    "seconds": dt,
                    "out_dir": str(rep_out_dir),
                    "log": str(log_path),
                }
            )

        except Exception as e:
            dt = float(time.perf_counter() - t0)
            replica_results.append(
                {
                    "replica": i,
                    "seed": int(seed),
                    "status": "error",
                    "seconds": dt,
                    "out_dir": str(rep_out_dir),
                    "log": str(log_path),
                    "error": str(e),
                }
            )
            print(f"❌ Replica {i:02d} failed: {e}")
            if not args.keep_going:
                break

    exp_seconds = float(time.perf_counter() - t_exp0)

    # Aggregate per-replica confirmation_summary.csv into one CSV
    rows: List[pd.DataFrame] = []
    try:
        import pandas as pd

        for r in replica_results:
            if r.get("status") != "ok":
                continue
            rep_dir = Path(r["out_dir"])
            summary_path = rep_dir / "confirmation_summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path, sep=";", decimal=",")
            df.insert(0, "replica", int(r["replica"]))
            df.insert(1, "seed", int(r["seed"]))
            rows.append(df)

        if rows:
            all_df = pd.concat(rows, ignore_index=True)
            all_df.to_csv(exp_summary_csv, index=False, sep=";", decimal=",")
            print(f"\n✅ Experiment summary saved: {exp_summary_csv}")
        else:
            print("\n⚠️ No replica summaries found to aggregate.")

    except Exception as e:
        print(f"\n⚠️ Could not aggregate experiment summaries: {e}")

    # Update experiment manifest
    exp_manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    exp_manifest["total_seconds"] = exp_seconds
    exp_manifest["replicas"] = replica_results
    exp_manifest["experiment_summary_csv"] = str(exp_summary_csv) if exp_summary_csv.exists() else None
    _write_json(exp_manifest_path, exp_manifest)

    print("\n✅ Experiment runner finished")
    print(f"- Manifest: {exp_manifest_path}")
    if exp_summary_csv.exists():
        print(f"- Summary:  {exp_summary_csv}")


if __name__ == "__main__":
    main()