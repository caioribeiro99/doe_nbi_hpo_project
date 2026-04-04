#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _seed_for(replica: int, seed0: int) -> int:
    return int(seed0) + int(replica) - 1


def _dataset_out_root(base_out_root: Path, dataset_name: str) -> Path:
    return base_out_root / dataset_name


def _build_cmd(job: Dict[str, Any], python_exe: str) -> List[str]:
    cfg = job["dataset_cfg"]
    cmd: List[str] = [
        python_exe,
        str(REPO_ROOT / "scripts" / "run_fairness_replica.py"),
        "--dataset",
        str(cfg["path"]),
        "--design",
        str(job["design"]),
        "--replica",
        str(job["replica"]),
        "--seed",
        str(job["seed"]),
        "--out_root",
        str(job["out_root"]),
        "--dataset-kind",
        str(cfg.get("dataset_kind", "generic")),
        "--n-jobs",
        str(job["inner_n_jobs"]),
        "--run-baselines",
        "--stratify-by-group",
        "--refine",
        "--refine-anchor-strategy",
        "mixed",
        "--refine-anchor-mix",
        "0.4",
        "0.4",
        "0.2",
        "--beta-step",
        str(job["beta_step"]),
        "--nbi-eval-k-stage1",
        str(job["nbi_eval_k_stage1"]),
        "--refine-n-samples",
        str(job["refine_n_samples"]),
        "--nbi-eval-k-stage2",
        str(job["nbi_eval_k_stage2"]),
        "--quality-floor",
        str(job["quality_floor"]),
        "--nbi-range-quality-floor",
        str(job["quality_floor"]),
        "--fairness-best-floor",
        "rs",
        "--fairness-best-delta",
        str(job["fairness_best_delta"]),
    ]

    if "target_col" in cfg:
        cmd += ["--target-col", str(cfg["target_col"])]
    if cfg.get("target_positive") is not None:
        cmd += ["--target-positive", str(cfg["target_positive"])]
    if cfg.get("protected_col") is not None:
        cmd += ["--protected-col", str(cfg["protected_col"])]
    if cfg.get("protected_attr_mode") is not None:
        cmd += ["--protected-attr-mode", str(cfg["protected_attr_mode"])]
    if bool(cfg.get("drop_unknown_rows", False)):
        cmd += ["--drop-unknown-rows"]

    return cmd


def _job_env() -> Dict[str, str]:
    env = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = "1"
    return env


def _run_job(job: Dict[str, Any], python_exe: str) -> Dict[str, Any]:
    cmd = _build_cmd(job, python_exe)
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_job_env(), capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    return {
        "dataset": job["dataset_name"],
        "replica": job["replica"],
        "seed": job["seed"],
        "returncode": int(proc.returncode),
        "elapsed_seconds": float(elapsed),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
        "cmd": cmd,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the 4-base fairness R30 suite with maximum outer parallelism.")
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "fairness_suite_4bases_r30.json"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "experiments" / "fairness_suite_4bases_r30"))
    p.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--inner-n-jobs", type=int, default=1, help="Threads passed to each inner XGBoost evaluation. Keep at 1 when maxing outer parallelism.")
    p.add_argument("--beta-step", type=float, default=0.005)
    p.add_argument("--nbi-eval-k-stage1", type=int, default=180)
    p.add_argument("--refine-n-samples", type=int, default=350)
    p.add_argument("--nbi-eval-k-stage2", type=int, default=200)
    p.add_argument("--quality-floor", type=float, default=0.55)
    p.add_argument("--fairness-best-delta", type=float, default=0.02)
    p.add_argument("--dataset", action="append", default=None, help="Optional dataset name filter; may be used multiple times.")
    p.add_argument("--replicas", type=int, default=None, help="Override replica count from config.")
    args = p.parse_args()

    cfg = _load_cfg(Path(args.config))
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_cfgs = list(cfg["datasets"])
    if args.dataset:
        wanted = set(args.dataset)
        dataset_cfgs = [d for d in dataset_cfgs if d["name"] in wanted]
        if not dataset_cfgs:
            raise SystemExit(f"No datasets matched --dataset filter: {sorted(wanted)}")

    replicas = int(args.replicas or cfg.get("replicas", 30))
    seed0 = int(cfg.get("seed0", 42))
    design = str((REPO_ROOT / cfg["design"]).resolve())

    jobs: List[Dict[str, Any]] = []
    for dcfg in dataset_cfgs:
        dataset_path = Path(str(dcfg["path"]))
        if not dataset_path.is_absolute():
            dataset_path = (REPO_ROOT / dataset_path).resolve()
        dcfg = {**dcfg, "path": str(dataset_path)}
        for replica in range(1, replicas + 1):
            jobs.append(
                {
                    "dataset_name": dcfg["name"],
                    "dataset_cfg": dcfg,
                    "replica": replica,
                    "seed": _seed_for(replica, seed0),
                    "design": design,
                    "out_root": str(_dataset_out_root(out_root, dcfg["name"])),
                    "inner_n_jobs": int(args.inner_n_jobs),
                    "beta_step": float(args.beta_step),
                    "nbi_eval_k_stage1": int(args.nbi_eval_k_stage1),
                    "refine_n_samples": int(args.refine_n_samples),
                    "nbi_eval_k_stage2": int(args.nbi_eval_k_stage2),
                    "quality_floor": float(args.quality_floor),
                    "fairness_best_delta": float(args.fairness_best_delta),
                }
            )

    manifest = {
        "suite_name": cfg.get("suite_name", "fairness_suite"),
        "config": str(Path(args.config).resolve()),
        "out_root": str(out_root),
        "replicas": replicas,
        "seed0": seed0,
        "max_workers": int(args.max_workers),
        "inner_n_jobs": int(args.inner_n_jobs),
        "jobs": [{k: v for k, v in j.items() if k != "dataset_cfg"} | {"dataset_cfg": j["dataset_cfg"]} for j in jobs],
    }
    (out_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Running fairness suite with {len(jobs)} jobs | max_workers={args.max_workers} | inner_n_jobs={args.inner_n_jobs}")
    results: List[Dict[str, Any]] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
        fut_map = {ex.submit(_run_job, job, sys.executable): job for job in jobs}
        for fut in as_completed(fut_map):
            res = fut.result()
            results.append(res)
            status = "OK" if res["returncode"] == 0 else "FAIL"
            print(f"[{status}] {res['dataset']} replica={res['replica']:02d} seed={res['seed']} elapsed={res['elapsed_seconds']:.1f}s")
            if res["returncode"] != 0:
                failed += 1

    results = sorted(results, key=lambda d: (d["dataset"], d["replica"]))
    (out_root / "suite_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if failed:
        print(f"⚠️ Suite finished with {failed} failed jobs. See suite_results.json.")
        raise SystemExit(1)
    print("✅ Fairness suite completed successfully.")


if __name__ == "__main__":
    main()
