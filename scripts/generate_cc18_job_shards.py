#!/usr/bin/env python
"""Generate deterministic OpenML-CC18 SQLite job shards.

Reads the four frozen protocol CSVs and the cc18_jobs SQLite schema, then
materializes one SQLite file per (stage, shard) under
``jobs/doctoral/openml_cc18/shards/<stage>/shard_NN.sqlite``.

The generator is the single point of truth for the SQLite job matrix; no
method names, scope rules, or stage-gating logic are hardcoded here. Every
row of cc18_jobs is derived from method_matrix.csv + execution_policy.csv +
parego_subset.csv + tasks.csv.

Flags:
  --tasks <path>             tasks.csv (default: benchmarks/doctoral/openml_cc18/tasks.csv)
  --method-matrix <path>     method_matrix.csv
  --execution-policy <path>  execution_policy.csv
  --parego-subset <path>     parego_subset.csv
  --schema <path>            jobs/doctoral/openml_cc18/schema.sql
  --out-dir <path>           jobs/doctoral/openml_cc18/shards
  --shards N                 number of shard files per stage (default 10)
  --stage all|<stage_name>   restrict to one stage (default all)
  --force                    overwrite existing shard files
  --dry-run                  compute counts and write summary only; no SQLite

This module also exposes :func:`generate` so tests can call it directly with
in-memory paths and assert the produced row counts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
DEFAULT_MATRIX = REPO / "benchmarks/doctoral/openml_cc18/method_matrix.csv"
DEFAULT_POLICY = REPO / "benchmarks/doctoral/openml_cc18/execution_policy.csv"
DEFAULT_PAREGO = REPO / "benchmarks/doctoral/openml_cc18/parego_subset.csv"
DEFAULT_SCHEMA = REPO / "jobs/doctoral/openml_cc18/schema.sql"
DEFAULT_OUT = REPO / "jobs/doctoral/openml_cc18/shards"

ALGORITHMS = ("xgboost", "lightgbm", "catboost")

STAGE_REPLICA_RANGES: dict[str, range] = {
    "stage0_replica_001": range(1, 2),
    "stage1_topup_to_005": range(2, 6),
    "stage2_topup_to_010": range(6, 11),
    "stage3_topup_to_030": range(11, 31),
}
STAGES = tuple(STAGE_REPLICA_RANGES.keys())

PANEL_VERSION = "cc18_v1"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    openml_task_id: int
    openml_dataset_id: int
    dataset_name: str


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    primary_or_ablation: str
    full_cc18: bool
    subset_only: bool


@dataclass(frozen=True)
class PolicyRow:
    method_id: str
    execution_tier: str
    run_scope: str
    stage_flags: tuple[bool, bool, bool, bool]
    requires_manual_signoff_before_stage3: bool


@dataclass
class JobRow:
    job_id: str
    openml_task_id: int
    openml_dataset_id: int
    dataset_name: str
    algorithm: str
    method: str
    replica: int
    stage: str
    config_path: str
    output_dir: str
    estimated_seconds: float | None
    notes: str | None


@dataclass
class ShardCounts:
    rows_by_stage: dict[str, int] = field(default_factory=lambda: {s: 0 for s in STAGES})
    rows_by_shard: dict[tuple[str, int], int] = field(default_factory=dict)
    rows_by_method: Counter[str] = field(default_factory=Counter)
    rows_by_algorithm: Counter[str] = field(default_factory=Counter)
    parego_rows: int = 0
    ablation_rows: int = 0
    literature_rows: int = 0
    stage3_signoff_rows: int = 0
    total_rows: int = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _bool(v: str) -> bool:
    s = v.strip().lower()
    if s in {"true", "false"}:
        return s == "true"
    raise ValueError(f"non-boolean: {v!r}")


def load_tasks(path: Path) -> list[Task]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            Task(
                openml_task_id=int(r["openml_task_id"]),
                openml_dataset_id=int(r["openml_dataset_id"]),
                dataset_name=r["dataset_name"],
            )
            for r in reader
        ]


def load_method_matrix(path: Path) -> dict[str, MethodSpec]:
    with path.open() as f:
        out: dict[str, MethodSpec] = {}
        for r in csv.DictReader(f):
            mid = r["method_id"]
            out[mid] = MethodSpec(
                method_id=mid,
                primary_or_ablation=r["primary_or_ablation"],
                full_cc18=_bool(r["full_cc18"]),
                subset_only=_bool(r["subset_only"]),
            )
        return out


def load_execution_policy(path: Path) -> dict[str, PolicyRow]:
    with path.open() as f:
        out: dict[str, PolicyRow] = {}
        for r in csv.DictReader(f):
            mid = r["method_id"]
            out[mid] = PolicyRow(
                method_id=mid,
                execution_tier=r["execution_tier"],
                run_scope=r["run_scope"],
                stage_flags=(
                    _bool(r["stage0"]),
                    _bool(r["stage1_topup_to_005"]),
                    _bool(r["stage2_topup_to_010"]),
                    _bool(r["stage3_topup_to_030"]),
                ),
                requires_manual_signoff_before_stage3=_bool(
                    r["requires_manual_signoff_before_stage3"]
                ),
            )
        return out


def load_parego_subset(path: Path) -> set[int]:
    with path.open() as f:
        return {int(r["openml_task_id"]) for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Job-row construction
# ---------------------------------------------------------------------------


def deterministic_job_id(task_id: int, algorithm: str, method: str, replica: int) -> str:
    """Return a 16-hex-char job_id derived from the row's natural key.

    The hash function is BLAKE2s with a fixed digest size, no salt, no
    timestamp; calling this twice with the same arguments must return the
    same id forever.
    """
    key = f"{task_id}|{algorithm}|{method}|{replica}".encode()
    return hashlib.blake2s(key, digest_size=8).hexdigest()


def shard_index(task_id: int, algorithm: str, n_shards: int) -> int:
    """Deterministic round-robin shard assignment by (task_id, algorithm).

    The assignment is stable: ordering by (task_id, position-of-algorithm)
    and then mod n_shards. This keeps every replica/method for a single
    (task, algorithm) cell on the same shard, which reduces dataset
    download/cache churn at run time.
    """
    alg_idx = ALGORITHMS.index(algorithm)
    pos = task_id * len(ALGORITHMS) + alg_idx
    return pos % n_shards


def _config_path_for(algorithm: str, method: str, task_id: int) -> str:
    return f"configs/cc18/{algorithm}/{method}/task_{task_id:06d}.yaml"


def _output_dir_for(algorithm: str, method: str, task_id: int, replica: int) -> str:
    return (
        f"experiments/cc18/{algorithm}/{method}/task_{task_id:06d}/"
        f"replica_{replica:03d}"
    )


def build_jobs(
    tasks: list[Task],
    methods: dict[str, MethodSpec],
    policy: dict[str, PolicyRow],
    parego_subset: set[int],
    stages: Iterable[str] = STAGES,
) -> list[JobRow]:
    """Materialize the in-memory job-row list, no SQLite involved."""
    out: list[JobRow] = []
    stage_index = {s: i for i, s in enumerate(STAGES)}
    parego_subset_sorted = sorted(parego_subset)
    parego_subset_set = set(parego_subset_sorted)
    tasks_by_id = {t.openml_task_id: t for t in tasks}

    for stage in stages:
        replica_range = STAGE_REPLICA_RANGES[stage]
        s_idx = stage_index[stage]
        for mid in methods:
            if mid not in policy:
                raise ValueError(f"method {mid!r} missing from execution_policy")
            pol = policy[mid]
            if not pol.stage_flags[s_idx]:
                continue
            if pol.run_scope == "not_in_comparison":
                continue
            if pol.run_scope == "full_cc18":
                row_task_ids = [t.openml_task_id for t in tasks]
            elif pol.run_scope == "parego_subset":
                row_task_ids = [
                    t.openml_task_id for t in tasks
                    if t.openml_task_id in parego_subset_set
                ]
            else:
                raise ValueError(
                    f"unknown run_scope {pol.run_scope!r} for {mid!r}"
                )
            note = None
            if (
                stage == "stage3_topup_to_030"
                and pol.requires_manual_signoff_before_stage3
            ):
                note = "requires_manual_signoff_before_stage3"
            for task_id in row_task_ids:
                t = tasks_by_id[task_id]
                for algorithm in ALGORITHMS:
                    cfg = _config_path_for(algorithm, mid, task_id)
                    for replica in replica_range:
                        out_dir = _output_dir_for(algorithm, mid, task_id, replica)
                        jid = deterministic_job_id(task_id, algorithm, mid, replica)
                        out.append(JobRow(
                            job_id=jid,
                            openml_task_id=task_id,
                            openml_dataset_id=t.openml_dataset_id,
                            dataset_name=t.dataset_name,
                            algorithm=algorithm,
                            method=mid,
                            replica=replica,
                            stage=stage,
                            config_path=cfg,
                            output_dir=out_dir,
                            estimated_seconds=None,
                            notes=note,
                        ))
    # Stable order: stage, method, task_id, algorithm, replica.
    out.sort(key=lambda j: (
        stage_index[j.stage], j.method, j.openml_task_id,
        ALGORITHMS.index(j.algorithm), j.replica,
    ))
    return out


# ---------------------------------------------------------------------------
# SQLite materialization
# ---------------------------------------------------------------------------


def _apply_schema(cx: sqlite3.Connection, schema_sql: str) -> None:
    cx.executescript(schema_sql)


def _insert_jobs(cx: sqlite3.Connection, rows: list[JobRow]) -> None:
    cx.executemany(
        """
        INSERT INTO cc18_jobs (
            job_id, openml_task_id, openml_dataset_id, dataset_name,
            algorithm, method, replica, stage,
            config_path, output_dir, estimated_seconds,
            status, retry_count, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, ?)
        """,
        [
            (
                r.job_id, r.openml_task_id, r.openml_dataset_id, r.dataset_name,
                r.algorithm, r.method, r.replica, r.stage,
                r.config_path, r.output_dir, r.estimated_seconds, r.notes,
            )
            for r in rows
        ],
    )


def _insert_shard_meta(
    cx: sqlite3.Connection, *,
    shard_id: int, suite_id: int, generated_at: str,
    n_tasks: int, n_methods: int, n_replicas_max: int,
    notes: str,
) -> None:
    cx.execute(
        """
        INSERT INTO shard_meta (
            shard_id, suite_id, panel_version, generated_at,
            n_tasks, n_algorithms, n_methods, n_replicas_max, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            shard_id, suite_id, PANEL_VERSION, generated_at,
            n_tasks, len(ALGORITHMS), n_methods, n_replicas_max, notes,
        ),
    )


def _write_shard(
    path: Path, *,
    schema_sql: str,
    rows: list[JobRow],
    shard_id: int,
    suite_id: int,
    generated_at: str,
    n_tasks: int,
    n_methods: int,
    n_replicas_max: int,
    notes: str,
) -> None:
    if path.exists():
        path.unlink()
    # Remove any leftover WAL/SHM sidecars from a previous failed run.
    for sidecar in (path.with_suffix(path.suffix + "-wal"),
                    path.with_suffix(path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as cx:
        # The schema declares journal_mode=WAL for runtime claim/release;
        # for the committed snapshot we want a single self-contained file
        # and no WAL/SHM sidecars staged into git, so override to DELETE
        # for the writer. The runner can re-enable WAL on its workers.
        cx.execute("PRAGMA journal_mode=DELETE")
        _apply_schema(cx, schema_sql)
        # The schema script re-asserts journal_mode=WAL; force back to
        # DELETE before commit so the on-disk file has no sidecars.
        cx.execute("PRAGMA journal_mode=DELETE")
        if rows:
            _insert_jobs(cx, rows)
        _insert_shard_meta(
            cx,
            shard_id=shard_id, suite_id=suite_id, generated_at=generated_at,
            n_tasks=n_tasks, n_methods=n_methods, n_replicas_max=n_replicas_max,
            notes=notes,
        )
        cx.commit()
    # Final cleanup: delete any sidecars SQLite may still have left behind.
    for sidecar in (path.with_suffix(path.suffix + "-wal"),
                    path.with_suffix(path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()


# ---------------------------------------------------------------------------
# Public API for tests
# ---------------------------------------------------------------------------


@dataclass
class GenerateResult:
    counts: ShardCounts
    shard_paths: list[Path]
    summary_json: Path | None
    summary_md: Path | None


def generate(*,
             tasks_csv: Path,
             matrix_csv: Path,
             policy_csv: Path,
             parego_csv: Path,
             schema_sql_path: Path,
             out_dir: Path,
             n_shards: int = 10,
             stages: Iterable[str] = STAGES,
             dry_run: bool = False,
             force: bool = False,
             suite_id: int = 99) -> GenerateResult:
    tasks = load_tasks(tasks_csv)
    methods = load_method_matrix(matrix_csv)
    policy = load_execution_policy(policy_csv)
    parego_subset = load_parego_subset(parego_csv)
    schema_sql = schema_sql_path.read_text(encoding="utf-8")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    selected_stages = list(stages)
    rows = build_jobs(tasks, methods, policy, parego_subset, selected_stages)
    if not rows:
        raise RuntimeError("generator produced zero rows; refusing to write")

    counts = ShardCounts()
    counts.total_rows = len(rows)
    by_stage_shard: dict[tuple[str, int], list[JobRow]] = defaultdict(list)
    for r in rows:
        sh = shard_index(r.openml_task_id, r.algorithm, n_shards)
        by_stage_shard[(r.stage, sh)].append(r)
        counts.rows_by_stage[r.stage] = counts.rows_by_stage.get(r.stage, 0) + 1
        counts.rows_by_method[r.method] += 1
        counts.rows_by_algorithm[r.algorithm] += 1
        if methods[r.method].subset_only:
            counts.parego_rows += 1
        if methods[r.method].primary_or_ablation == "ablation":
            counts.ablation_rows += 1
        if methods[r.method].primary_or_ablation == "literature_only":
            counts.literature_rows += 1
        if r.notes == "requires_manual_signoff_before_stage3":
            counts.stage3_signoff_rows += 1
    counts.rows_by_shard = {
        (stage, sh): len(rs) for (stage, sh), rs in by_stage_shard.items()
    }

    shard_paths: list[Path] = []

    if not dry_run:
        for stage in selected_stages:
            stage_dir = out_dir / stage
            if force and stage_dir.exists():
                for p in stage_dir.glob("shard_*.sqlite"):
                    p.unlink()
            stage_dir.mkdir(parents=True, exist_ok=True)
            for sh in range(n_shards):
                shard_path = stage_dir / f"shard_{sh:02d}.sqlite"
                if shard_path.exists() and not force:
                    raise FileExistsError(
                        f"{shard_path} exists; pass --force to overwrite"
                    )
                shard_rows = by_stage_shard.get((stage, sh), [])
                _write_shard(
                    shard_path,
                    schema_sql=schema_sql,
                    rows=shard_rows,
                    shard_id=sh,
                    suite_id=suite_id,
                    generated_at=generated_at,
                    n_tasks=len({r.openml_task_id for r in shard_rows}),
                    n_methods=len({r.method for r in shard_rows}),
                    n_replicas_max=max(
                        (r.replica for r in shard_rows), default=0
                    ),
                    notes=f"stage={stage};shard={sh};rows={len(shard_rows)}",
                )
                shard_paths.append(shard_path)

    # Always write summary (dry-run too).
    summary_json = out_dir / "shard_summary.json"
    summary_md = out_dir / "shard_summary.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_json(summary_json, counts, n_shards, selected_stages, generated_at, dry_run)
    _write_summary_md(summary_md, counts, n_shards, selected_stages, generated_at, dry_run)

    return GenerateResult(
        counts=counts,
        shard_paths=shard_paths,
        summary_json=summary_json,
        summary_md=summary_md,
    )


def _write_summary_json(path: Path, counts: ShardCounts, n_shards: int,
                        stages: list[str], generated_at: str,
                        dry_run: bool) -> None:
    rows_by_shard_serializable = [
        {"stage": stage, "shard_id": sh, "rows": rows}
        for (stage, sh), rows in sorted(counts.rows_by_shard.items())
    ]
    payload = {
        "generated_at": generated_at,
        "panel_version": PANEL_VERSION,
        "n_shards_per_stage": n_shards,
        "stages": stages,
        "dry_run": dry_run,
        "total_rows": counts.total_rows,
        "rows_by_stage": counts.rows_by_stage,
        "rows_by_shard": rows_by_shard_serializable,
        "rows_by_method": dict(counts.rows_by_method),
        "rows_by_algorithm": dict(counts.rows_by_algorithm),
        "parego_rows": counts.parego_rows,
        "ablation_rows": counts.ablation_rows,
        "literature_rows": counts.literature_rows,
        "stage3_signoff_rows": counts.stage3_signoff_rows,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_summary_md(path: Path, counts: ShardCounts, n_shards: int,
                      stages: list[str], generated_at: str,
                      dry_run: bool) -> None:
    lines: list[str] = []
    lines.append("# OpenML-CC18 SQLite shard summary\n")
    lines.append(f"- generated_at: `{generated_at}`")
    lines.append(f"- panel_version: `{PANEL_VERSION}`")
    lines.append(f"- shards per stage: {n_shards}")
    lines.append(f"- dry_run: {dry_run}")
    lines.append(f"- total rows: **{counts.total_rows}**\n")
    lines.append("## Rows by stage\n")
    lines.append("| stage | rows |")
    lines.append("|---|---:|")
    cum = 0
    for s in stages:
        n = counts.rows_by_stage.get(s, 0)
        cum += n
        lines.append(f"| `{s}` | {n} (cum {cum}) |")
    lines.append("")
    lines.append("## Rows by method\n")
    lines.append("| method | rows |")
    lines.append("|---|---:|")
    for m in sorted(counts.rows_by_method.keys()):
        lines.append(f"| `{m}` | {counts.rows_by_method[m]} |")
    lines.append("")
    lines.append("## Rows by algorithm\n")
    lines.append("| algorithm | rows |")
    lines.append("|---|---:|")
    for a in ALGORITHMS:
        lines.append(f"| `{a}` | {counts.rows_by_algorithm.get(a, 0)} |")
    lines.append("")
    lines.append("## Tier counters\n")
    lines.append(f"- ParEGO subset rows: **{counts.parego_rows}**")
    lines.append(f"- Ablation rows: **{counts.ablation_rows}**")
    lines.append(f"- Literature-only rows: **{counts.literature_rows}** (must be 0)")
    lines.append(f"- Stage-3 manual-signoff rows: **{counts.stage3_signoff_rows}**\n")
    lines.append("## Rows by (stage, shard)\n")
    lines.append("| stage | shard | rows |")
    lines.append("|---|---:|---:|")
    for (stage, sh), rows in sorted(counts.rows_by_shard.items()):
        lines.append(f"| `{stage}` | {sh:02d} | {rows} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--method-matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--execution-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--parego-subset", type=Path, default=DEFAULT_PAREGO)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--stage", default="all",
                        choices=("all", *STAGES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    stages = STAGES if args.stage == "all" else (args.stage,)
    res = generate(
        tasks_csv=args.tasks,
        matrix_csv=args.method_matrix,
        policy_csv=args.execution_policy,
        parego_csv=args.parego_subset,
        schema_sql_path=args.schema,
        out_dir=args.out_dir,
        n_shards=args.shards,
        stages=stages,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"total rows: {res.counts.total_rows}")
    for s in stages:
        print(f"  {s}: {res.counts.rows_by_stage.get(s, 0)}")
    print(f"parego={res.counts.parego_rows}  "
          f"ablation={res.counts.ablation_rows}  "
          f"literature={res.counts.literature_rows}  "
          f"stage3_signoff={res.counts.stage3_signoff_rows}")
    print(f"summary: {res.summary_json}")
    if not args.dry_run:
        print(f"shards: {len(res.shard_paths)} files under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
