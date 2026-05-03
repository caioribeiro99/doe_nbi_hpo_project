#!/usr/bin/env python
"""Run batch_00_synthetic_canary on the dedicated Mac.

The dedicated-Mac gate that must pass before batch_01 (the first
real CC18 batch) is allowed. Steps:

1. Read the manifest at
   ``benchmarks/doctoral/openml_cc18/batches/batch_00_synthetic_canary.json``.
2. Copy the committed shard
   ``jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite``
   to a private temp path (the committed file is never mutated).
3. Prune the temp shard to a 12-cell canary slice
   (4 canary methods × 3 algorithms × 1 replica) where possible.
4. Invoke ``scripts/cc18_runner.py`` with
   ``--canary-only --train --synthetic-task --max-evaluations 5
   --n-folds 2 --max-jobs 12``.
5. Collect per-cell manifests written by the runner under
   ``--output-root experiments/_batch_runs/batch_00_synthetic_canary/``.
6. Emit the gate artifact at
   ``experiments/_batch_runs/batch_00_synthetic_canary_latest.{json,md}``.

The artifact records git SHA, hostname, Python version, package
versions, runner command, per-cell status, capability audit
summary, source-shard MD5 before/after, and confirmation that the
stage-3 sign-off file was NOT created.

Never trains on real CC18 data. Never downloads OpenML payloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb._versions import collect_package_versions  # noqa: E402

DEFAULT_MANIFEST = REPO / "benchmarks/doctoral/openml_cc18/batches/batch_00_synthetic_canary.json"
DEFAULT_SOURCE_SHARD = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/batch_00_synthetic_canary"
DEFAULT_GATE_DIR = REPO / "experiments/_batch_runs"
RUNNER = REPO / "scripts/cc18_runner.py"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
CANARY_ALGORITHMS = ("xgboost", "lightgbm", "catboost")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _pkg_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Map distribution names to versions (or ``None`` if unresolved).

    Delegates to :func:`doe_xgb._versions.collect_package_versions`,
    which handles distribution-vs-import name mismatches such as
    ``scikit-learn`` ↔ ``sklearn``.
    """
    return collect_package_versions(names)


def _platform() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "uname": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "machine": platform.machine(),
    }


def _capability_audit_summary() -> dict:
    """Run the capability audit in-process so the gate artifact ships
    a fresh snapshot, not whatever JSON happens to be on disk."""
    sys.path.insert(0, str(REPO / "scripts"))
    from audit_method_capabilities import audit  # type: ignore

    matrix = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"
    rep = audit(matrix)
    return {
        "n_benchmarked": rep["n_benchmarked"],
        "smoke_ready": rep["smoke_ready"],
        "dispatch_only": rep["dispatch_only"],
        "stub_only": rep["stub_only"],
        "missing_packages": rep["missing_packages_overall"],
    }


# ---------------------------------------------------------------------------
# Shard copy + prune
# ---------------------------------------------------------------------------


def _prune_shard(shard: Path, *, methods: tuple[str, ...],
                 algorithms: tuple[str, ...], stage: str,
                 max_replicas_per_cell: int = 1) -> int:
    """Delete every row that is not in the (method × algorithm × stage)
    canary slice. Within each cell, keep the lowest job_id replica only."""
    cx = sqlite3.connect(shard)
    placeholders_m = ",".join("?" * len(methods))
    placeholders_a = ",".join("?" * len(algorithms))
    cx.execute(
        f"DELETE FROM cc18_jobs WHERE method NOT IN ({placeholders_m}) "
        f"OR algorithm NOT IN ({placeholders_a}) OR stage != ?",
        (*methods, *algorithms, stage),
    )
    # Keep at most `max_replicas_per_cell` rows per (method, algorithm)
    # by job_id ordering (deterministic).
    cx.execute(
        "DELETE FROM cc18_jobs WHERE job_id NOT IN ("
        "SELECT job_id FROM cc18_jobs cj "
        "WHERE (SELECT COUNT(*) FROM cc18_jobs cj2 "
        "       WHERE cj2.method = cj.method AND cj2.algorithm = cj.algorithm "
        "       AND cj2.job_id <= cj.job_id) <= ?"
        ")",
        (max_replicas_per_cell,),
    )
    cx.commit()
    n = cx.execute("SELECT COUNT(*) FROM cc18_jobs").fetchone()[0]
    cx.close()
    return int(n)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_canary(*, manifest_path: Path, source_shard: Path,
               out_root: Path, gate_dir: Path,
               max_evaluations: int = 5, n_folds: int = 2,
               max_jobs: int = 12) -> dict:
    if not source_shard.exists():
        raise FileNotFoundError(source_shard)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("uses_openml") is True:
        raise RuntimeError(
            "batch_00 manifest claims to use OpenML data; refusing to "
            "run the synthetic canary against it."
        )
    if not manifest.get("synthetic_task"):
        raise RuntimeError("batch_00 manifest is missing synthetic_task")

    md5_before = _md5(source_shard)

    tmpdir = Path(tempfile.mkdtemp(prefix="cc18_batch00_"))
    temp_shard = tmpdir / "shard_00.sqlite"
    shutil.copy(source_shard, temp_shard)
    n_pruned = _prune_shard(
        temp_shard,
        methods=CANARY_METHODS,
        algorithms=CANARY_ALGORITHMS,
        stage="stage0_replica_001",
        max_replicas_per_cell=1,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(RUNNER),
        "--shard", str(temp_shard),
        "--canary-only", "--train", "--synthetic-task",
        "--max-evaluations", str(int(max_evaluations)),
        "--n-folds", str(int(n_folds)),
        "--max-jobs", str(int(max_jobs)),
        "--output-root", str(out_root),
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    runtime_s = time.perf_counter() - t0
    md5_after_source = _md5(source_shard)

    # Read final job statuses on the temp shard.
    cx = sqlite3.connect(temp_shard)
    rows = list(cx.execute(
        "SELECT method, algorithm, status, runtime_seconds, last_error "
        "FROM cc18_jobs ORDER BY method, algorithm"
    ))
    cx.close()

    def _safe_rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO))
        except ValueError:
            return str(p)

    cells: list[dict] = []
    for method, algorithm, status, runtime_seconds, last_error in rows:
        manifest_path_for_cell = None
        agg_metrics: dict | None = None
        # Find the per-cell manifest written by the runner.
        for mf in out_root.rglob("manifest.json"):
            try:
                payload = json.loads(mf.read_text())
            except Exception:
                continue
            if (payload.get("method_id") == method
                    and payload.get("algorithm") == algorithm):
                manifest_path_for_cell = _safe_rel(mf)
                agg_metrics = payload.get("aggregate_metrics")
                break
        cells.append({
            "method": method, "algorithm": algorithm,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "last_error": last_error,
            "manifest": manifest_path_for_cell,
            "aggregate_metrics": agg_metrics,
        })

    n_success = sum(1 for c in cells if c["status"] == "success")
    n_failed = sum(1 for c in cells if c["status"] == "failed")
    n_pending = sum(1 for c in cells if c["status"] == "pending")

    artifact = {
        "batch_id": manifest["batch_id"],
        "run_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": _git_sha(),
        "platform": _platform(),
        "package_versions": _pkg_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
        )),
        "runner_command": cmd,
        "manifest_path": _safe_rel(manifest_path),
        "source_shard": _safe_rel(source_shard),
        "source_shard_md5_before": md5_before,
        "source_shard_md5_after": md5_after_source,
        "source_shard_unchanged": md5_before == md5_after_source,
        "temp_shard": str(temp_shard),
        "n_cells_in_temp_shard": n_pruned,
        "n_cells_expected": len(CANARY_METHODS) * len(CANARY_ALGORITHMS),
        "n_cells_success": n_success,
        "n_cells_failed": n_failed,
        "n_cells_pending": n_pending,
        "runtime_seconds": runtime_s,
        "subprocess_stdout_tail": proc.stdout[-2000:],
        "subprocess_stderr_tail": proc.stderr[-1000:],
        "subprocess_returncode": proc.returncode,
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "stage3_signoff_path": _safe_rel(SIGNOFF_FILE),
        "capability_audit": _capability_audit_summary(),
        "cells": cells,
    }
    return artifact


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def write_artifact(artifact: dict, gate_dir: Path) -> tuple[Path, Path]:
    gate_dir.mkdir(parents=True, exist_ok=True)
    json_p = gate_dir / "batch_00_synthetic_canary_latest.json"
    md_p = gate_dir / "batch_00_synthetic_canary_latest.md"
    json_p.write_text(json.dumps(artifact, indent=2, sort_keys=True),
                      encoding="utf-8")

    lines: list[str] = []
    lines.append("# batch_00_synthetic_canary -- dedicated Mac gate\n")
    lines.append(f"- batch_id: `{artifact['batch_id']}`")
    lines.append(f"- run_timestamp: `{artifact['run_timestamp']}`")
    lines.append(f"- git_sha: `{artifact['git_sha'][:12]}`")
    lines.append(f"- hostname: `{artifact['platform']['hostname']}`")
    lines.append(f"- uname: `{artifact['platform']['uname']}`")
    lines.append(f"- python: `{artifact['platform']['python_version']}` "
                 f"({artifact['platform']['python_executable']})")
    lines.append(f"- runtime: {artifact['runtime_seconds']:.1f} s\n")
    lines.append(f"- temp_shard: `{artifact['temp_shard']}`")
    lines.append(f"- n_cells_in_temp_shard: {artifact['n_cells_in_temp_shard']}")
    lines.append(f"- n_cells_expected: {artifact['n_cells_expected']}")
    lines.append(f"- success: **{artifact['n_cells_success']}**, "
                 f"failed: **{artifact['n_cells_failed']}**, "
                 f"pending: {artifact['n_cells_pending']}\n")
    lines.append(f"- source_shard_unchanged: **{artifact['source_shard_unchanged']}**")
    lines.append(f"- source_shard_md5_before: `{artifact['source_shard_md5_before']}`")
    lines.append(f"- source_shard_md5_after:  `{artifact['source_shard_md5_after']}`")
    lines.append(f"- stage3_signoff_present:  {artifact['stage3_signoff_present']}\n")

    lines.append("## Package versions\n")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, ver in artifact["package_versions"].items():
        lines.append(f"| `{name}` | {ver if ver else 'MISSING'} |")
    lines.append("")

    lines.append("## Capability audit\n")
    cap = artifact["capability_audit"]
    lines.append(f"- smoke_ready: {cap['smoke_ready']}")
    lines.append(f"- dispatch_only: {cap['dispatch_only']}")
    lines.append(f"- stub_only: {cap['stub_only']}")
    lines.append(f"- missing_packages: {cap['missing_packages']}\n")

    lines.append("## 12-cell canary results\n")
    lines.append("| method | algorithm | status | runtime_s | last_error |")
    lines.append("|---|---|---|---:|---|")
    for c in artifact["cells"]:
        rt = f"{c['runtime_seconds']:.2f}" if c["runtime_seconds"] is not None else "—"
        err = (c["last_error"] or "—")[:60].replace("|", "\\|")
        lines.append(f"| `{c['method']}` | `{c['algorithm']}` | "
                     f"{c['status']} | {rt} | {err} |")
    lines.append("")

    if artifact["n_cells_failed"] == 0 and artifact["n_cells_pending"] == 0:
        lines.append("## Verdict: **GATE PASS**\n")
        lines.append("batch_01_cc18_tiny_3_tasks may proceed.\n")
    else:
        lines.append("## Verdict: **GATE FAIL**\n")
        lines.append("Resolve failures before attempting batch_01.\n")

    md_p.write_text("\n".join(lines), encoding="utf-8")
    return json_p, md_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-shard", type=Path,
                        default=DEFAULT_SOURCE_SHARD)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument("--max-evaluations", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve paths and exit without running cc18_runner")

    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({
            "manifest": str(args.manifest),
            "source_shard": str(args.source_shard),
            "output_root": str(args.output_root),
            "gate_dir": str(args.gate_dir),
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
            "max_jobs": args.max_jobs,
        }, indent=2))
        return 0

    artifact = run_canary(
        manifest_path=args.manifest,
        source_shard=args.source_shard,
        out_root=args.output_root,
        gate_dir=args.gate_dir,
        max_evaluations=args.max_evaluations,
        n_folds=args.n_folds,
        max_jobs=args.max_jobs,
    )
    json_p, md_p = write_artifact(artifact, args.gate_dir)
    print(f"success={artifact['n_cells_success']}/"
          f"{artifact['n_cells_expected']}  "
          f"failed={artifact['n_cells_failed']}  "
          f"pending={artifact['n_cells_pending']}")
    print(f"json: {json_p}")
    print(f"md:   {md_p}")
    rc = 0 if (artifact["n_cells_failed"] == 0
               and artifact["n_cells_pending"] == 0
               and artifact["source_shard_unchanged"]) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
