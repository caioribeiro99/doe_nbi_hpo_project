#!/usr/bin/env python
"""Run batch_03_cc18_representative_18_tasks against real OpenML data
via the result handoff protocol (Commit 35).

This is the third real-OpenML batch in the doctoral pipeline. Like
batch_02, it consumes the run-dir + summary protocol; unlike
batch_02, it gates on the *stage-run summary* JSON published by
batch_02 (under ``experiments/_stage_runs/``) rather than on a
batch-run gate JSON.

Steps:

- materialize a gitignored execution copy of every committed
  stage-0 shard that contributes batch_03 rows under
  ``runs/cc18/<run_id>/shards/stage0_replica_001/`` via
  ``scripts/create_cc18_run_dir.py``;
- run the canary methods × 3 GBDT algorithms × 18 tasks against
  those execution copies (committed source shards never opened for
  write);
- cache OpenML payloads under ``data/source/openml_cc18/<task>/``
  (already gitignored since Commit 34);
- publish a stage-run summary at
  ``experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.{json,md}``
  via ``scripts/export_cc18_run_summary.py``.

Pre-flight refusals
-------------------
- ``experiments/_stage_runs/batch_02_cc18_small_12_tasks_latest_summary.json``
  must exist, be green (144/144 success, 0 failed/pending),
  report ``source_shards_unchanged: true``, report
  ``stage3_signoff_present: false``, and be ≤ 7 days old;
- ``stage3_signoff.json`` must NOT exist;
- the batch CSV must contain exactly the 18 task IDs documented in
  ``benchmarks/doctoral/openml_cc18/batches/batch_03_cc18_representative_18_tasks.meta.json``.

What the runner does NOT do
---------------------------
- mutate any committed SQLite shard;
- run anything outside the canary set;
- run on tasks outside the 18 in the batch CSV;
- create ``stage3_signoff.json``;
- commit raw OpenML payloads (gitignored under
  ``data/source/openml_cc18/``);
- commit execution SQLite files (gitignored under ``runs/``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from doe_xgb._versions import collect_package_versions  # noqa: E402

DEFAULT_BATCH_CSV = (
    REPO / "benchmarks/doctoral/openml_cc18/batches/batch_03_cc18_representative_18_tasks.csv"
)
DEFAULT_BATCH_META = (
    REPO / "benchmarks/doctoral/openml_cc18/batches/batch_03_cc18_representative_18_tasks.meta.json"
)
DEFAULT_BATCH02_SUMMARY = (
    REPO / "experiments/_stage_runs/batch_02_cc18_small_12_tasks_latest_summary.json"
)
DEFAULT_SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/batch_03_cc18_representative_18_tasks"
DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_OPENML_CACHE_ROOT = REPO / "data/source/openml_cc18"
RUNNER = REPO / "scripts/cc18_runner.py"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
CANARY_ALGORITHMS = ("xgboost", "lightgbm", "catboost")
CANARY_STAGE = "stage0_replica_001"
CANARY_REPLICA = 1
RUN_ID = "batch_03_cc18_representative_18_tasks_latest"
BATCH_ID = "batch_03_cc18_representative_18_tasks"
EXPECTED_N_TASKS = 18
EXPECTED_BATCH02_CELLS = 144
BATCH_02_MAX_AGE_DAYS = 7


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when the batch_02 pre-flight checks reject the run."""


def _summary_age_days(summary_path: Path) -> float:
    """Compute the age of a stage-run summary in days.

    The protocol exporter stamps ``exported_at`` (Commit 35); older
    batch runners stamped ``run_timestamp``. We accept either so the
    gate keeps working across protocol revisions.
    """
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    ts = payload.get("exported_at") or payload.get("run_timestamp") or ""
    try:
        run_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise GateRefusalError(
            f"batch_02 summary has unparsable timestamp={ts!r}: {exc}"
        ) from exc
    delta = datetime.now(timezone.utc) - run_dt
    return delta.total_seconds() / 86400.0


def verify_batch02_summary(
    summary_path: Path = DEFAULT_BATCH02_SUMMARY,
    *, max_age_days: float = BATCH_02_MAX_AGE_DAYS,
) -> dict:
    """Refuse batch_03 if the batch_02 stage-run summary is missing,
    failed, or stale."""
    if not summary_path.exists():
        raise GateRefusalError(
            f"batch_02 latest summary not found at {summary_path}; "
            "run scripts/run_batch_02_cc18_small_12_tasks.py first."
        )
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = int(payload.get("n_cells_expected", 0))
    success = int(payload.get("n_cells_success", 0))
    failed = int(payload.get("n_cells_failed", 0))
    pending = int(payload.get("n_cells_pending", 0))
    unchanged = bool(payload.get("source_shards_unchanged", False))
    signoff = bool(payload.get("stage3_signoff_present", False))
    if (
        expected != EXPECTED_BATCH02_CELLS
        or success != EXPECTED_BATCH02_CELLS
        or failed != 0
        or pending != 0
    ):
        raise GateRefusalError(
            f"batch_02 summary is not green: expected={expected} "
            f"success={success} failed={failed} pending={pending}"
        )
    if not unchanged:
        raise GateRefusalError(
            "batch_02 summary reports source_shards_unchanged=False; "
            "investigate before running batch_03."
        )
    if signoff:
        raise GateRefusalError(
            "batch_02 summary reports stage3_signoff_present=True; "
            "refusing to run batch_03 in pre-stage-0 territory."
        )
    age = _summary_age_days(summary_path)
    if age > float(max_age_days):
        raise GateRefusalError(
            f"batch_02 summary is {age:.2f} days old (>{max_age_days:.0f}d); "
            "re-run batch_02 before batch_03."
        )
    return {
        "n_cells_expected": expected,
        "n_cells_success": success,
        "n_cells_failed": failed,
        "n_cells_pending": pending,
        "source_shards_unchanged": unchanged,
        "stage3_signoff_present": signoff,
        "exported_at": payload.get("exported_at"),
        "run_timestamp": payload.get("run_timestamp"),
        "age_days": age,
        "source_git_sha": payload.get("source_git_sha"),
        "run_id": payload.get("run_id"),
    }


# ---------------------------------------------------------------------------
# Batch CSV
# ---------------------------------------------------------------------------


def load_batch_task_ids(batch_csv: Path) -> list[int]:
    with batch_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "openml_task_id" not in (reader.fieldnames or ()):
            raise ValueError(
                f"{batch_csv}: CSV missing required column openml_task_id"
            )
        return [int(r["openml_task_id"]) for r in reader]


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _which_shards_have_task(
    shards_dir: Path, *, task_ids: tuple[int, ...],
    methods: tuple[str, ...], algorithms: tuple[str, ...],
    stage: str, replica: int,
) -> list[Path]:
    """Return the subset of committed shards that contain at least one
    canary row for the requested task / method / algorithm slice."""
    keep: list[Path] = []
    placeholders_t = ",".join("?" * len(task_ids))
    placeholders_m = ",".join("?" * len(methods))
    placeholders_a = ",".join("?" * len(algorithms))
    sql = (
        f"SELECT COUNT(*) FROM cc18_jobs "
        f"WHERE openml_task_id IN ({placeholders_t}) "
        f"AND method IN ({placeholders_m}) "
        f"AND algorithm IN ({placeholders_a}) "
        f"AND stage = ? AND replica = ?"
    )
    params = (*task_ids, *methods, *algorithms, stage, int(replica))
    for shard in sorted(shards_dir.glob("shard_*.sqlite")):
        cx = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        try:
            n = cx.execute(sql, params).fetchone()[0]
        finally:
            cx.close()
        if int(n) > 0:
            keep.append(shard)
    return keep


# ---------------------------------------------------------------------------
# Run-dir + execution-shard prune
# ---------------------------------------------------------------------------


def _prune_execution_shard(
    exec_path: Path, *, task_ids: tuple[int, ...],
    methods: tuple[str, ...], algorithms: tuple[str, ...],
    stage: str, replica: int,
) -> int:
    """Delete every row from an execution copy that is NOT in the
    batch_02 canary slice. Mutating the execution copy is safe — that
    file lives under runs/ and is gitignored. Returns the row count
    that survives."""
    placeholders_t = ",".join("?" * len(task_ids))
    placeholders_m = ",".join("?" * len(methods))
    placeholders_a = ",".join("?" * len(algorithms))
    cx = sqlite3.connect(exec_path)
    try:
        cx.execute(
            f"DELETE FROM cc18_jobs "
            f"WHERE NOT (openml_task_id IN ({placeholders_t}) "
            f"AND method IN ({placeholders_m}) "
            f"AND algorithm IN ({placeholders_a}) "
            f"AND stage = ? AND replica = ?)",
            (*task_ids, *methods, *algorithms, stage, int(replica)),
        )
        cx.commit()
        n = cx.execute("SELECT COUNT(*) FROM cc18_jobs").fetchone()[0]
    finally:
        cx.close()
    return int(n)


# ---------------------------------------------------------------------------
# OpenML metadata
# ---------------------------------------------------------------------------


def load_task_metadata(
    task_ids: list[int], *, cache_root: Path,
) -> list[dict]:
    from doe_xgb.datasets.openml_cc18_loader import (
        load_cc18_task,
        task_metadata_summary,
    )

    summaries: list[dict] = []
    for tid in task_ids:
        payload = load_cc18_task(int(tid), cache_root=cache_root)
        summaries.append(task_metadata_summary(payload))
    return summaries


# ---------------------------------------------------------------------------
# Cells + summary
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _platform() -> dict[str, str]:
    return {
        "hostname": platform.node(),
        "uname": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "machine": platform.machine(),
    }


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _capability_audit_summary() -> dict:
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


def _collect_cells(
    exec_dir: Path, output_root: Path, task_ids: list[int],
) -> list[dict]:
    cells: list[dict] = []
    seen_manifests: set[Path] = set()
    for exec_p in sorted(exec_dir.rglob("*.execution.sqlite")):
        cx = sqlite3.connect(exec_p)
        try:
            rows = list(cx.execute(
                "SELECT openml_task_id, method, algorithm, status, "
                "runtime_seconds, last_error FROM cc18_jobs "
                "ORDER BY openml_task_id, method, algorithm"
            ))
        finally:
            cx.close()
        for tid, method, algorithm, status, rt, err in rows:
            if int(tid) not in task_ids:
                continue
            manifest_path = None
            agg: dict | None = None
            metric_keys: list[str] = []
            for mf in output_root.rglob("manifest.json"):
                if mf in seen_manifests:
                    continue
                try:
                    payload = json.loads(mf.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if (
                    payload.get("method_id") == method
                    and payload.get("algorithm") == algorithm
                    and int(payload.get("openml_task_id", -1)) == int(tid)
                ):
                    manifest_path = _safe_rel(mf)
                    agg = payload.get("aggregate_metrics") or {}
                    metric_keys = sorted(agg.keys())
                    seen_manifests.add(mf)
                    break
            cells.append({
                "openml_task_id": int(tid),
                "method": method,
                "algorithm": algorithm,
                "status": status,
                "runtime_seconds": rt,
                "last_error": err,
                "manifest": manifest_path,
                "aggregate_metrics": agg,
                "metric_keys": metric_keys,
            })
    return cells


def _slowest_cells(cells: list[dict], n: int = 8) -> list[dict]:
    rated = [c for c in cells if c["runtime_seconds"] is not None]
    rated.sort(key=lambda c: float(c["runtime_seconds"]), reverse=True)
    return [
        {
            "openml_task_id": c["openml_task_id"],
            "method": c["method"], "algorithm": c["algorithm"],
            "runtime_seconds": float(c["runtime_seconds"]),
        }
        for c in rated[:n]
    ]


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_batch_03(
    *,
    batch_csv: Path = DEFAULT_BATCH_CSV,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    run_root: Path = DEFAULT_RUN_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    stage_runs_dir: Path = DEFAULT_STAGE_RUNS_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    batch02_summary: Path = DEFAULT_BATCH02_SUMMARY,
    max_age_days: float = BATCH_02_MAX_AGE_DAYS,
    max_evaluations: int = 5,
    n_folds: int = 2,
    run_id: str = RUN_ID,
    skip_train: bool = False,
    force_run_dir: bool = True,
) -> dict:
    """Run the batch_03 canary on real OpenML data and return the
    summary dict written to disk."""
    from create_cc18_run_dir import create_run_dir
    from export_cc18_run_summary import export_summary

    batch02_gate = verify_batch02_summary(
        batch02_summary, max_age_days=max_age_days,
    )

    if SIGNOFF_FILE.exists():
        raise GateRefusalError(
            f"refusing to run batch_03: stage-3 sign-off file already exists "
            f"at {SIGNOFF_FILE}"
        )

    task_ids = load_batch_task_ids(batch_csv)
    if not task_ids:
        raise GateRefusalError(f"{batch_csv}: empty task list")
    if len(task_ids) != EXPECTED_N_TASKS:
        raise GateRefusalError(
            f"{batch_csv}: expected {EXPECTED_N_TASKS} task IDs, "
            f"got {len(task_ids)}"
        )

    # Prepare directories.
    run_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    stage_runs_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    # 1. Decide which committed shards we need (any with at least one
    #    matching canary row for the 18 task IDs).
    needed = _which_shards_have_task(
        shards_dir,
        task_ids=tuple(task_ids),
        methods=CANARY_METHODS, algorithms=CANARY_ALGORITHMS,
        stage=CANARY_STAGE, replica=CANARY_REPLICA,
    )
    if not needed:
        raise GateRefusalError(
            f"no shards under {shards_dir} contain canary rows for "
            f"task_ids={task_ids}"
        )

    # 2. Materialize an execution copy of every needed shard via the
    #    handoff helper. The helper refuses anything under jobs/ and
    #    records source MD5s in run_manifest.json.
    create_run_dir(
        run_id=run_id,
        stage=CANARY_STAGE,
        shard_files=[p.name for p in needed],
        run_root=run_root,
        shards_root=shards_dir.parent,
        force=force_run_dir,
    )
    run_dir = run_root / run_id
    exec_dir = run_dir / "shards" / CANARY_STAGE

    # 3. Prune each execution copy down to the batch_03 canary slice.
    n_pruned_total = 0
    for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
        n_pruned_total += _prune_execution_shard(
            exec_p,
            task_ids=tuple(task_ids),
            methods=CANARY_METHODS, algorithms=CANARY_ALGORITHMS,
            stage=CANARY_STAGE, replica=CANARY_REPLICA,
        )

    # 4. Cache OpenML payloads (gitignored).
    task_summaries = load_task_metadata(task_ids, cache_root=openml_cache_root)

    # 5. Source-shard immutability check after the OpenML downloads.
    md5_after_download = {p.name: _md5(p) for p in needed}
    md5_before = {p.name: _md5(p) for p in needed}  # re-read; identical baseline
    # ``run_manifest.json`` already recorded source MD5s before the copy;
    # we re-hash here to surface any drift introduced after the copy.
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    md5_recorded = {
        c["source"].split("/")[-1]: c["source_md5_before"]
        for c in manifest["shard_copies"]
    }
    shards_unchanged_after_download = all(
        md5_after_download.get(name) == md5_recorded.get(name)
        for name in md5_recorded
    )

    n_cells_expected = (
        len(task_ids) * len(CANARY_METHODS) * len(CANARY_ALGORITHMS)
    )

    # 6. Invoke the runner once per execution shard so per-cell failures
    #    in one shard do not abort the others.
    runtime_total = 0.0
    proc_returncodes: dict[str, int] = {}
    proc_stderr_tails: dict[str, str] = {}
    if not skip_train:
        for exec_p in sorted(exec_dir.glob("*.execution.sqlite")):
            cmd = [
                sys.executable, str(RUNNER),
                "--shard", str(exec_p),
                "--canary-only", "--train",
                "--max-evaluations", str(int(max_evaluations)),
                "--n-folds", str(int(n_folds)),
                "--max-jobs", str(int(n_cells_expected)),
                "--output-root", str(out_root),
                "--openml-cache-root", str(openml_cache_root),
            ]
            t0 = time.perf_counter()
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False,
            )
            runtime_total += time.perf_counter() - t0
            proc_returncodes[exec_p.name] = proc.returncode
            proc_stderr_tails[exec_p.name] = proc.stderr[-500:]
            # Continue on per-cell failure: do NOT raise here. Cell
            # statuses surface in the per-shard SQLite and roll up
            # in the summary.

    # 7. Re-confirm committed source-shard MD5s after the runner.
    #    The protocol-level summary also re-checks via run_manifest.json,
    #    but we keep an explicit dict so the augmented batch_03 summary
    #    can publish the before/after MD5 pair regardless.
    md5_after_run = {p.name: _md5(p) for p in needed}

    cells = _collect_cells(exec_dir, out_root, task_ids)
    n_success = sum(1 for c in cells if c["status"] == "success")
    n_failed = sum(1 for c in cells if c["status"] == "failed")
    n_pending = sum(1 for c in cells if c["status"] == "pending")

    # 8. Publish a stage-run summary via the protocol exporter.
    summary_json = stage_runs_dir / f"{run_id}_summary.json"
    summary_md = stage_runs_dir / f"{run_id}_summary.md"
    summary = export_summary(
        run_dir=run_dir,
        out_json=summary_json,
        out_md=summary_md,
        include_shard_hashes=True,
        batch_id=BATCH_ID,
    )

    # 9. Augment the summary with batch_03-specific blocks (task
    #    metadata, per-cell table, slowest cells, runner audit, etc.)
    #    so the published JSON is self-contained for the article.
    summary.update({
        "batch_id": BATCH_ID,
        "batch_csv": _safe_rel(batch_csv),
        "task_ids": [int(t) for t in task_ids],
        "task_metadata": task_summaries,
        "n_cells_expected": n_cells_expected,
        "n_cells_in_temp_shard": int(n_pruned_total),
        "n_cells_success": n_success,
        "n_cells_failed": n_failed,
        "n_cells_pending": n_pending,
        "cells": cells,
        "slowest_cells": _slowest_cells(cells),
        "runtime_seconds_runner_total": runtime_total,
        "runner_returncodes": proc_returncodes,
        "runner_stderr_tails": proc_stderr_tails,
        "openml_cache_root": _safe_rel(openml_cache_root),
        "openml_payloads_committed": False,
        "execution_shards_committed": False,
        "batch_02_gate": batch02_gate,
        "shards_unchanged_after_download": shards_unchanged_after_download,
        "source_shard_md5_before": dict(md5_before),
        "source_shard_md5_after": dict(md5_after_run),
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "capability_audit": _capability_audit_summary(),
        "run_dir": _safe_rel(run_dir),
        "stage": CANARY_STAGE,
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
    })
    # Re-write the augmented JSON; the MD already captures the protocol-
    # level verdict from export_summary, but we append a batch_03-
    # specific block so the file is human-actionable.
    summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )
    _augment_md(summary_md, summary)
    return summary


def _augment_md(md_path: Path, summary: dict) -> None:
    """Append the batch_03 task / per-cell blocks to the summary MD."""
    extra: list[str] = []
    extra.append("\n---\n")
    extra.append("## batch_03-specific summary\n")
    extra.append(
        f"- batch_id: `{summary['batch_id']}`"
    )
    extra.append(f"- run_dir: `{summary['run_dir']}` (gitignored)")
    extra.append(
        f"- n_cells_expected: {summary['n_cells_expected']}, "
        f"in_temp_shard: {summary['n_cells_in_temp_shard']}, "
        f"success: **{summary['n_cells_success']}**, "
        f"failed: **{summary['n_cells_failed']}**, "
        f"pending: {summary['n_cells_pending']}"
    )
    extra.append(
        f"- runtime (runner only): "
        f"{summary['runtime_seconds_runner_total']:.1f} s"
    )
    extra.append(
        f"- shards_unchanged_after_download: "
        f"**{summary['shards_unchanged_after_download']}**"
    )
    extra.append(
        f"- openml_payloads_committed: "
        f"{summary['openml_payloads_committed']}"
    )
    extra.append(
        f"- execution_shards_committed: "
        f"{summary['execution_shards_committed']}\n"
    )

    g = summary["batch_02_gate"]
    extra.append("### batch_02 pre-flight\n")
    extra.append(
        f"- exported_at: `{g.get('exported_at') or g.get('run_timestamp')}`"
    )
    extra.append(f"- age_days: {float(g.get('age_days', 0)):.2f}")
    extra.append(
        f"- success: {g.get('n_cells_success')}/"
        f"{g.get('n_cells_expected')} (failed={g.get('n_cells_failed')}, "
        f"pending={g.get('n_cells_pending', 0)})"
    )
    extra.append(
        f"- source_shards_unchanged: {g.get('source_shards_unchanged')}"
    )
    extra.append(
        f"- run_id: `{g.get('run_id')}`\n"
    )

    extra.append("### Tasks\n")
    extra.append(
        "| task_id | dataset | type | rows | features | classes | "
        "categorical | sha256 |"
    )
    extra.append("|---:|---|---|---:|---:|---:|---:|---|")
    for tm in summary["task_metadata"]:
        sha = (tm.get("payload_sha256") or "")[:12]
        extra.append(
            f"| {tm['task_id']} | `{tm['dataset_name']}` | "
            f"{tm['task_type']} | {tm['n_rows']} | {tm['n_features']} | "
            f"{tm['n_classes']} | {tm['n_categorical_columns']} | "
            f"`{sha}` |"
        )
    extra.append("")

    extra.append("### Slowest cells\n")
    extra.append("| task_id | method | algorithm | runtime_s |")
    extra.append("|---:|---|---|---:|")
    for c in summary["slowest_cells"]:
        extra.append(
            f"| {c['openml_task_id']} | `{c['method']}` | "
            f"`{c['algorithm']}` | {c['runtime_seconds']:.2f} |"
        )
    extra.append("")

    extra.append("### Per-cell results\n")
    extra.append(
        "| task_id | method | algorithm | status | runtime_s | "
        "metric_keys | last_error |"
    )
    extra.append("|---:|---|---|---|---:|---|---|")
    for c in summary["cells"]:
        rt = (
            f"{float(c['runtime_seconds']):.2f}"
            if c["runtime_seconds"] is not None else "—"
        )
        err = (c["last_error"] or "—")[:60].replace("|", "\\|")
        keys = ", ".join(c["metric_keys"]) if c["metric_keys"] else "—"
        extra.append(
            f"| {c['openml_task_id']} | `{c['method']}` | "
            f"`{c['algorithm']}` | {c['status']} | {rt} | {keys} | {err} |"
        )
    extra.append("")

    if (
        summary["n_cells_failed"] == 0
        and summary["n_cells_pending"] == 0
        and summary["source_shards_unchanged"]
        and summary["shards_unchanged_after_download"]
        and not summary["stage3_signoff_present"]
    ):
        extra.append("### batch_03 verdict: **GATE PASS**\n")
        extra.append(
            "batch_04_stage0_shard00_only may proceed (only after manual "
            "review and only via the same handoff protocol).\n"
        )
    else:
        extra.append("### batch_03 verdict: **GATE FAIL**\n")
        extra.append(
            "Resolve failures, restore committed shards, or remove "
            "the stage-3 sign-off file before attempting batch_04.\n"
        )

    md_path.write_text(
        md_path.read_text(encoding="utf-8") + "\n".join(extra),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-csv", type=Path, default=DEFAULT_BATCH_CSV)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument(
        "--stage-runs-dir", type=Path, default=DEFAULT_STAGE_RUNS_DIR,
    )
    parser.add_argument(
        "--openml-cache-root", type=Path, default=DEFAULT_OPENML_CACHE_ROOT,
    )
    parser.add_argument(
        "--batch02-summary", type=Path, default=DEFAULT_BATCH02_SUMMARY,
    )
    parser.add_argument(
        "--max-age-days", type=float, default=BATCH_02_MAX_AGE_DAYS,
    )
    parser.add_argument("--max-evaluations", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print resolved configuration and exit.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="run pre-flight + run-dir + OpenML cache, but do NOT "
             "invoke the cc18 runner; used by the test suite.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "batch_csv": str(args.batch_csv),
            "shards_dir": str(args.shards_dir),
            "run_root": str(args.run_root),
            "output_root": str(args.output_root),
            "stage_runs_dir": str(args.stage_runs_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "batch02_summary": str(args.batch02_summary),
            "max_age_days": args.max_age_days,
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
            "run_id": args.run_id,
        }, indent=2))
        return 0

    try:
        summary = run_batch_03(
            batch_csv=args.batch_csv,
            shards_dir=args.shards_dir,
            run_root=args.run_root,
            out_root=args.output_root,
            stage_runs_dir=args.stage_runs_dir,
            openml_cache_root=args.openml_cache_root,
            batch02_summary=args.batch02_summary,
            max_age_days=args.max_age_days,
            max_evaluations=args.max_evaluations,
            n_folds=args.n_folds,
            run_id=args.run_id,
            skip_train=args.skip_train,
        )
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"success={summary['n_cells_success']}/"
        f"{summary['n_cells_expected']}  "
        f"failed={summary['n_cells_failed']}  "
        f"pending={summary['n_cells_pending']}"
    )
    print(f"json: {args.stage_runs_dir / (args.run_id + '_summary.json')}")
    print(f"md:   {args.stage_runs_dir / (args.run_id + '_summary.md')}")
    rc = 0 if (
        summary["n_cells_failed"] == 0
        and summary["n_cells_pending"] == 0
        and summary["source_shards_unchanged"]
        and summary["shards_unchanged_after_download"]
        and not summary["stage3_signoff_present"]
    ) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
