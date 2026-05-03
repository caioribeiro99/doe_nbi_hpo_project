#!/usr/bin/env python
"""Run batch_01_cc18_tiny_3_tasks against real OpenML CC18 data.

This is the first batch in the doctoral pipeline that touches real
OpenML payloads. It runs the four canary methods × 3 GBDT algorithms
across the 3 tiny CC18 tasks listed in
``benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv``
(9946 wdbc, 125920 dresses-sales, 11 balance-scale).

Pre-flight refusals
-------------------
1. The dedicated-Mac batch_00 gate must have run and passed
   (``experiments/_batch_runs/batch_00_synthetic_canary_latest.json``
   present, ``n_cells_failed == 0``,
   ``n_cells_success == n_cells_expected``,
   ``source_shard_unchanged == True``);
2. The batch_00 latest artifact must not be older than 7 days (the
   feedback loop on real OpenML data should be tight);
3. ``stage3_signoff.json`` must NOT exist — this batch deliberately
   stays in pre-stage-0 territory.

What the runner does NOT do
---------------------------
- Mutate any committed SQLite shard (each source is opened read-only;
  MD5s are checked before / after).
- Run anything outside the canary set (default_gbdt, random_search,
  tpe_optuna, doe_rsm_vrf_true_nbi).
- Run on more than the 3 tasks in the batch CSV.
- Create ``stage3_signoff.json``.
- Commit raw OpenML payloads — the cache directory under
  ``data/source/openml_cc18/`` is gitignored and intentionally stays
  on the dedicated Mac.
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
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb._versions import collect_package_versions  # noqa: E402

DEFAULT_BATCH_CSV = (
    REPO / "benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv"
)
DEFAULT_BATCH00_GATE = (
    REPO / "experiments/_batch_runs/batch_00_synthetic_canary_latest.json"
)
DEFAULT_SHARDS_DIR = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001"
DEFAULT_OUT_ROOT = REPO / "experiments/_batch_runs/batch_01_cc18_tiny_3_tasks"
DEFAULT_GATE_DIR = REPO / "experiments/_batch_runs"
DEFAULT_OPENML_CACHE_ROOT = REPO / "data/source/openml_cc18"
RUNNER = REPO / "scripts/cc18_runner.py"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
SCHEMA_SQL_PATH = REPO / "jobs/doctoral/openml_cc18/schema.sql"

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
CANARY_ALGORITHMS = ("xgboost", "lightgbm", "catboost")
CANARY_STAGE = "stage0_replica_001"
CANARY_REPLICA = 1

BATCH_00_MAX_AGE_DAYS = 7


# ---------------------------------------------------------------------------
# Pre-flight: batch_00 gate
# ---------------------------------------------------------------------------


class GateRefusalError(RuntimeError):
    """Raised when the batch_00 pre-flight checks reject the run."""


def _batch00_age_days(gate_path: Path) -> float:
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    ts = payload.get("run_timestamp", "")
    try:
        # ISO-8601 UTC with trailing Z.
        run_dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise GateRefusalError(
            f"batch_00 gate has unparsable run_timestamp={ts!r}: {exc}"
        ) from exc
    now = datetime.now(timezone.utc)
    delta = now - run_dt
    return delta.total_seconds() / 86400.0


def verify_batch00_gate(
    gate_path: Path = DEFAULT_BATCH00_GATE,
    *, max_age_days: float = BATCH_00_MAX_AGE_DAYS,
) -> dict:
    """Refuse batch_01 if the batch_00 gate is missing, failed, or stale."""
    if not gate_path.exists():
        raise GateRefusalError(
            f"batch_00 latest artifact not found at {gate_path}; "
            "run scripts/run_batch_00_synthetic_canary.py first."
        )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    expected = int(payload.get("n_cells_expected", 0))
    success = int(payload.get("n_cells_success", 0))
    failed = int(payload.get("n_cells_failed", 0))
    unchanged = bool(payload.get("source_shard_unchanged", False))
    if expected != success or failed != 0:
        raise GateRefusalError(
            f"batch_00 gate is not green: expected={expected} "
            f"success={success} failed={failed}"
        )
    if not unchanged:
        raise GateRefusalError(
            "batch_00 gate reports source_shard_unchanged=False; "
            "investigate before running batch_01."
        )
    age_days = _batch00_age_days(gate_path)
    if age_days > float(max_age_days):
        raise GateRefusalError(
            f"batch_00 gate is {age_days:.2f} days old (>"
            f"{max_age_days:.0f}d); re-run batch_00 before batch_01."
        )
    return {
        "n_cells_expected": expected,
        "n_cells_success": success,
        "n_cells_failed": failed,
        "source_shard_unchanged": unchanged,
        "run_timestamp": payload.get("run_timestamp"),
        "age_days": age_days,
        "git_sha": payload.get("git_sha"),
    }


# ---------------------------------------------------------------------------
# Batch CSV + shard discovery
# ---------------------------------------------------------------------------


def load_batch_task_ids(batch_csv: Path) -> list[int]:
    with batch_csv.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "openml_task_id" not in (reader.fieldnames or ()):
            raise ValueError(
                f"{batch_csv}: CSV missing required column openml_task_id"
            )
        return [int(r["openml_task_id"]) for r in reader]


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _select_canary_rows(
    shard: Path, *, task_ids: tuple[int, ...],
    methods: tuple[str, ...], algorithms: tuple[str, ...],
    stage: str, replica: int,
) -> tuple[list[str], list[tuple]]:
    """Return ``(columns, rows)`` from one shard for the canary slice."""
    cx = _open_ro(shard)
    try:
        cols = [r[1] for r in cx.execute("PRAGMA table_info(cc18_jobs)")]
        col_list = ", ".join(cols)
        placeholders_t = ",".join("?" * len(task_ids))
        placeholders_m = ",".join("?" * len(methods))
        placeholders_a = ",".join("?" * len(algorithms))
        sql = (
            f"SELECT {col_list} FROM cc18_jobs "
            f"WHERE openml_task_id IN ({placeholders_t}) "
            f"AND method IN ({placeholders_m}) "
            f"AND algorithm IN ({placeholders_a}) "
            f"AND stage = ? AND replica = ?"
        )
        params = (*task_ids, *methods, *algorithms, stage, int(replica))
        rows = list(cx.execute(sql, params).fetchall())
    finally:
        cx.close()
    return cols, rows


def assemble_canary_shard(
    *, shards_dir: Path, task_ids: tuple[int, ...],
    out_path: Path,
    methods: tuple[str, ...] = CANARY_METHODS,
    algorithms: tuple[str, ...] = CANARY_ALGORITHMS,
    stage: str = CANARY_STAGE, replica: int = CANARY_REPLICA,
    schema_sql_path: Path = SCHEMA_SQL_PATH,
) -> dict:
    """Merge the canary slice from every shard under ``shards_dir`` into one
    fresh SQLite at ``out_path``. Returns a discovery report including the
    per-source MD5 before / after and how many rows came from each shard.
    """
    shards = sorted(p for p in shards_dir.glob("shard_*.sqlite"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shards_dir}")

    md5_before: dict[str, str] = {p.name: _md5(p) for p in shards}
    rows_collected: list[tuple] = []
    contributions: dict[str, int] = {}
    cols_first: list[str] | None = None
    for shard in shards:
        cols, rows = _select_canary_rows(
            shard, task_ids=task_ids, methods=methods,
            algorithms=algorithms, stage=stage, replica=replica,
        )
        if cols_first is None:
            cols_first = cols
        else:
            assert cols == cols_first, f"column drift in {shard.name}"
        contributions[shard.name] = len(rows)
        rows_collected.extend(rows)

    # Re-hash to confirm the read-only opens did not perturb the files.
    md5_after: dict[str, str] = {p.name: _md5(p) for p in shards}
    unchanged = (md5_before == md5_after)

    if cols_first is None:
        raise RuntimeError("no shards produced any columns; impossible")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    schema_sql = schema_sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(out_path) as dst:
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.executescript(schema_sql)
        dst.execute("PRAGMA journal_mode=DELETE")
        if rows_collected:
            placeholders = ",".join(["?"] * len(cols_first))
            col_list = ",".join(cols_first)
            dst.executemany(
                f"INSERT INTO cc18_jobs ({col_list}) VALUES ({placeholders})",
                rows_collected,
            )
        # All rows come back as 'pending' from the source shards; no need
        # to reset their status, but normalize timestamp-y fields just in
        # case stale values were recorded earlier.
        dst.execute(
            "UPDATE cc18_jobs SET status='pending', "
            "assigned_worker=NULL, started_at=NULL, finished_at=NULL, "
            "runtime_seconds=NULL, last_error=NULL, retry_count=0",
        )
        dst.commit()
    for sc in (out_path.with_suffix(out_path.suffix + "-wal"),
               out_path.with_suffix(out_path.suffix + "-shm")):
        if sc.exists():
            sc.unlink()

    return {
        "n_rows_in_temp_shard": len(rows_collected),
        "shard_contributions": contributions,
        "shard_md5_before": md5_before,
        "shard_md5_after": md5_after,
        "shards_unchanged": unchanged,
    }


# ---------------------------------------------------------------------------
# OpenML payloads
# ---------------------------------------------------------------------------


def load_task_metadata(
    task_ids: list[int], *, cache_root: Path,
) -> list[dict]:
    """Load each task into the gitignored cache and return summaries."""
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
# Run + collect
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


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _collect_cells(
    temp_shard: Path, task_ids: list[int], output_root: Path,
) -> list[dict]:
    cx = sqlite3.connect(temp_shard)
    rows = list(cx.execute(
        "SELECT openml_task_id, method, algorithm, status, "
        "runtime_seconds, last_error FROM cc18_jobs "
        "ORDER BY openml_task_id, method, algorithm",
    ))
    cx.close()
    cells: list[dict] = []
    for tid, method, algorithm, status, runtime_seconds, last_error in rows:
        if int(tid) not in task_ids:
            continue
        manifest_path = None
        agg_metrics: dict | None = None
        metric_keys: list[str] = []
        for mf in output_root.rglob("manifest.json"):
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
                agg_metrics = payload.get("aggregate_metrics") or {}
                metric_keys = sorted(agg_metrics.keys())
                break
        cells.append({
            "openml_task_id": int(tid),
            "method": method,
            "algorithm": algorithm,
            "status": status,
            "runtime_seconds": runtime_seconds,
            "last_error": last_error,
            "manifest": manifest_path,
            "aggregate_metrics": agg_metrics,
            "metric_keys": metric_keys,
        })
    return cells


def run_batch_01(
    *,
    batch_csv: Path = DEFAULT_BATCH_CSV,
    shards_dir: Path = DEFAULT_SHARDS_DIR,
    out_root: Path = DEFAULT_OUT_ROOT,
    gate_dir: Path = DEFAULT_GATE_DIR,
    openml_cache_root: Path = DEFAULT_OPENML_CACHE_ROOT,
    batch00_gate: Path = DEFAULT_BATCH00_GATE,
    max_age_days: float = BATCH_00_MAX_AGE_DAYS,
    max_evaluations: int = 5,
    n_folds: int = 2,
    skip_train: bool = False,
) -> dict:
    """Run the batch_01 canary on real OpenML data and assemble the gate
    artifact dictionary. ``skip_train=True`` is used by tests."""
    batch00_summary = verify_batch00_gate(
        batch00_gate, max_age_days=max_age_days,
    )

    if SIGNOFF_FILE.exists():
        raise GateRefusalError(
            f"refusing to run batch_01: stage-3 sign-off file already exists "
            f"at {SIGNOFF_FILE}"
        )

    task_ids = load_batch_task_ids(batch_csv)
    if not task_ids:
        raise GateRefusalError(f"{batch_csv}: empty task list")

    out_root.mkdir(parents=True, exist_ok=True)
    gate_dir.mkdir(parents=True, exist_ok=True)
    openml_cache_root.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="cc18_batch01_"))
    temp_shard = tmpdir / "shard_batch_01.sqlite"

    discovery = assemble_canary_shard(
        shards_dir=shards_dir, task_ids=tuple(task_ids), out_path=temp_shard,
    )

    # Materialize the OpenML payloads so the runner only has to read
    # the cache. This also lets us record per-task metadata for the
    # gate artifact regardless of the runner's exit status.
    task_summaries = load_task_metadata(task_ids, cache_root=openml_cache_root)

    # Re-confirm shard immutability after the OpenML downloads.
    md5_after_download = {
        p.name: _md5(p)
        for p in sorted(shards_dir.glob("shard_*.sqlite"))
    }
    shards_still_unchanged = (
        md5_after_download == discovery["shard_md5_before"]
    )

    n_cells_expected = (
        len(task_ids) * len(CANARY_METHODS) * len(CANARY_ALGORITHMS)
    )

    cmd = [
        sys.executable, str(RUNNER),
        "--shard", str(temp_shard),
        "--canary-only", "--train",
        "--max-evaluations", str(int(max_evaluations)),
        "--n-folds", str(int(n_folds)),
        "--max-jobs", str(int(n_cells_expected)),
        "--output-root", str(out_root),
        "--openml-cache-root", str(openml_cache_root),
    ]

    if skip_train:
        proc_returncode = -1
        proc_stdout = ""
        proc_stderr = "skipped (skip_train=True)"
        runtime_s = 0.0
    else:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        runtime_s = time.perf_counter() - t0
        proc_returncode = proc.returncode
        proc_stdout = proc.stdout
        proc_stderr = proc.stderr

    md5_after_run = {
        p.name: _md5(p)
        for p in sorted(shards_dir.glob("shard_*.sqlite"))
    }
    shards_unchanged_overall = (
        md5_after_run == discovery["shard_md5_before"]
    )

    cells = _collect_cells(temp_shard, task_ids, out_root)
    n_success = sum(1 for c in cells if c["status"] == "success")
    n_failed = sum(1 for c in cells if c["status"] == "failed")
    n_pending = sum(1 for c in cells if c["status"] == "pending")

    artifact = {
        "batch_id": "batch_01_cc18_tiny_3_tasks",
        "run_timestamp": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "git_sha": _git_sha(),
        "platform": _platform(),
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "runner_command": cmd,
        "batch_csv": _safe_rel(batch_csv),
        "shards_dir": _safe_rel(shards_dir),
        "shard_contributions": discovery["shard_contributions"],
        "source_shard_md5_before": discovery["shard_md5_before"],
        "source_shard_md5_after": md5_after_run,
        "source_shards_unchanged": shards_unchanged_overall,
        "shards_unchanged_after_download": shards_still_unchanged,
        "temp_shard": str(temp_shard),
        "n_cells_in_temp_shard": discovery["n_rows_in_temp_shard"],
        "n_cells_expected": n_cells_expected,
        "n_cells_success": n_success,
        "n_cells_failed": n_failed,
        "n_cells_pending": n_pending,
        "task_metadata": task_summaries,
        "task_ids": [int(t) for t in task_ids],
        "openml_cache_root": _safe_rel(openml_cache_root),
        "openml_payloads_committed": False,  # gitignored by .gitignore
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "stage3_signoff_path": _safe_rel(SIGNOFF_FILE),
        "batch_00_gate": batch00_summary,
        "capability_audit": _capability_audit_summary(),
        "runtime_seconds": runtime_s,
        "subprocess_returncode": proc_returncode,
        "subprocess_stdout_tail": proc_stdout[-2000:],
        "subprocess_stderr_tail": proc_stderr[-2000:],
        "cells": cells,
    }
    return artifact


# ---------------------------------------------------------------------------
# Artifact writer
# ---------------------------------------------------------------------------


def write_artifact(artifact: dict, gate_dir: Path) -> tuple[Path, Path]:
    gate_dir.mkdir(parents=True, exist_ok=True)
    json_p = gate_dir / "batch_01_cc18_tiny_3_tasks_latest.json"
    md_p = gate_dir / "batch_01_cc18_tiny_3_tasks_latest.md"
    json_p.write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# batch_01_cc18_tiny_3_tasks -- dedicated Mac gate\n")
    lines.append(f"- batch_id: `{artifact['batch_id']}`")
    lines.append(f"- run_timestamp: `{artifact['run_timestamp']}`")
    lines.append(f"- git_sha: `{str(artifact['git_sha'])[:12]}`")
    lines.append(f"- hostname: `{artifact['platform']['hostname']}`")
    lines.append(f"- uname: `{artifact['platform']['uname']}`")
    lines.append(
        f"- python: `{artifact['platform']['python_version']}` "
        f"({artifact['platform']['python_executable']})"
    )
    lines.append(f"- runtime: {artifact['runtime_seconds']:.1f} s\n")
    lines.append(f"- temp_shard: `{artifact['temp_shard']}`")
    lines.append(
        f"- n_cells_in_temp_shard: {artifact['n_cells_in_temp_shard']}"
    )
    lines.append(f"- n_cells_expected: {artifact['n_cells_expected']}")
    lines.append(
        f"- success: **{artifact['n_cells_success']}**, "
        f"failed: **{artifact['n_cells_failed']}**, "
        f"pending: {artifact['n_cells_pending']}\n"
    )
    lines.append(
        f"- source_shards_unchanged: **{artifact['source_shards_unchanged']}**"
    )
    lines.append(
        f"- shards_unchanged_after_download: "
        f"**{artifact['shards_unchanged_after_download']}**"
    )
    lines.append(
        f"- stage3_signoff_present: {artifact['stage3_signoff_present']}"
    )
    lines.append(
        f"- openml_cache_root: `{artifact['openml_cache_root']}`"
    )
    lines.append(
        f"- openml_payloads_committed: {artifact['openml_payloads_committed']}\n"
    )

    g = artifact["batch_00_gate"]
    lines.append("## batch_00 pre-flight\n")
    lines.append(f"- run_timestamp: `{g.get('run_timestamp')}`")
    lines.append(f"- age_days: {g.get('age_days'):.2f}")
    lines.append(
        f"- success: {g.get('n_cells_success')}/"
        f"{g.get('n_cells_expected')} (failed={g.get('n_cells_failed')})"
    )
    lines.append(
        f"- source_shard_unchanged: {g.get('source_shard_unchanged')}\n"
    )

    lines.append("## Package versions\n")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, ver in artifact["package_versions"].items():
        lines.append(f"| `{name}` | {ver if ver else 'MISSING'} |")
    lines.append("")

    lines.append("## Tasks\n")
    lines.append(
        "| task_id | dataset | type | rows | features | classes | "
        "categorical | sha256 |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---|"
    )
    for tm in artifact["task_metadata"]:
        sha = (tm.get("payload_sha256") or "")[:12]
        lines.append(
            f"| {tm['task_id']} | `{tm['dataset_name']}` | "
            f"{tm['task_type']} | {tm['n_rows']} | {tm['n_features']} | "
            f"{tm['n_classes']} | {tm['n_categorical_columns']} | "
            f"`{sha}` |"
        )
    lines.append("")

    lines.append("## Capability audit\n")
    cap = artifact["capability_audit"]
    lines.append(f"- smoke_ready: {cap['smoke_ready']}")
    lines.append(f"- dispatch_only: {cap['dispatch_only']}")
    lines.append(f"- stub_only: {cap['stub_only']}")
    lines.append(f"- missing_packages: {cap['missing_packages']}\n")

    lines.append("## 36-cell canary results\n")
    lines.append(
        "| task_id | method | algorithm | status | runtime_s | "
        "metric_keys | last_error |"
    )
    lines.append("|---:|---|---|---|---:|---|---|")
    for c in artifact["cells"]:
        rt = (
            f"{c['runtime_seconds']:.2f}"
            if c["runtime_seconds"] is not None else "—"
        )
        err = (c["last_error"] or "—")[:60].replace("|", "\\|")
        keys = ", ".join(c["metric_keys"]) if c["metric_keys"] else "—"
        lines.append(
            f"| {c['openml_task_id']} | `{c['method']}` | `{c['algorithm']}` | "
            f"{c['status']} | {rt} | {keys} | {err} |"
        )
    lines.append("")

    lines.append("## Source shard MD5 (before / after)\n")
    lines.append("| shard | md5_before | md5_after |")
    lines.append("|---|---|---|")
    for shard in sorted(artifact["source_shard_md5_before"].keys()):
        b = artifact["source_shard_md5_before"][shard]
        a = artifact["source_shard_md5_after"].get(shard, "—")
        lines.append(f"| `{shard}` | `{b}` | `{a}` |")
    lines.append("")

    if (
        artifact["n_cells_failed"] == 0
        and artifact["n_cells_pending"] == 0
        and artifact["source_shards_unchanged"]
        and not artifact["stage3_signoff_present"]
    ):
        lines.append("## Verdict: **GATE PASS**\n")
        lines.append(
            "batch_02_cc18_small_12_tasks may proceed (only on the "
            "dedicated Mac, only after manual review of this artifact).\n"
        )
    else:
        lines.append("## Verdict: **GATE FAIL**\n")
        lines.append(
            "Resolve failures, restore committed shards, or remove "
            "the stage-3 sign-off file before attempting batch_02.\n"
        )

    md_p.write_text("\n".join(lines), encoding="utf-8")
    return json_p, md_p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-csv", type=Path, default=DEFAULT_BATCH_CSV)
    parser.add_argument("--shards-dir", type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gate-dir", type=Path, default=DEFAULT_GATE_DIR)
    parser.add_argument(
        "--openml-cache-root", type=Path, default=DEFAULT_OPENML_CACHE_ROOT,
    )
    parser.add_argument(
        "--batch00-gate", type=Path, default=DEFAULT_BATCH00_GATE,
    )
    parser.add_argument(
        "--max-age-days", type=float, default=BATCH_00_MAX_AGE_DAYS,
    )
    parser.add_argument("--max-evaluations", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resolved configuration and exit without running.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="run pre-flight + shard merge + OpenML cache, but do NOT "
             "invoke the cc18 runner; used by the test suite.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "batch_csv": str(args.batch_csv),
            "shards_dir": str(args.shards_dir),
            "output_root": str(args.output_root),
            "gate_dir": str(args.gate_dir),
            "openml_cache_root": str(args.openml_cache_root),
            "batch00_gate": str(args.batch00_gate),
            "max_age_days": args.max_age_days,
            "max_evaluations": args.max_evaluations,
            "n_folds": args.n_folds,
        }, indent=2))
        return 0

    try:
        artifact = run_batch_01(
            batch_csv=args.batch_csv,
            shards_dir=args.shards_dir,
            out_root=args.output_root,
            gate_dir=args.gate_dir,
            openml_cache_root=args.openml_cache_root,
            batch00_gate=args.batch00_gate,
            max_age_days=args.max_age_days,
            max_evaluations=args.max_evaluations,
            n_folds=args.n_folds,
            skip_train=args.skip_train,
        )
    except GateRefusalError as exc:
        print(f"GATE REFUSAL: {exc}", file=sys.stderr)
        return 3

    json_p, md_p = write_artifact(artifact, args.gate_dir)
    print(
        f"success={artifact['n_cells_success']}/"
        f"{artifact['n_cells_expected']}  "
        f"failed={artifact['n_cells_failed']}  "
        f"pending={artifact['n_cells_pending']}"
    )
    print(f"json: {json_p}")
    print(f"md:   {md_p}")
    rc = 0 if (
        artifact["n_cells_failed"] == 0
        and artifact["n_cells_pending"] == 0
        and artifact["source_shards_unchanged"]
        and not artifact["stage3_signoff_present"]
    ) else 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
