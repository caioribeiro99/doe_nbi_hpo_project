#!/usr/bin/env python
"""Filter a SQLite shard down to the rows needed for a batch.

Reads a source ``cc18_jobs`` shard and a batch task list (CSV produced
by ``scripts/create_cc18_batches.py`` or a JSON pointer file), then
writes a fresh SQLite file containing only the rows that match the
batch's task IDs (and optional method / algorithm / replica / stage
filters).

The source shard is opened **read-only** via the SQLite URI mode so
the committed shard can never be mutated, even by accident.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO / "jobs/doctoral/openml_cc18/batch_shards"
SCHEMA_SQL_PATH = REPO / "jobs/doctoral/openml_cc18/schema.sql"


# ---------------------------------------------------------------------------
# Batch loading
# ---------------------------------------------------------------------------


def load_batch_task_ids(path: Path) -> list[int]:
    """Accept either a CSV manifest (column ``openml_task_id``) or a JSON
    file with key ``task_ids``. JSON pointer files (``batch_04_*.json``
    / ``batch_00_*.json``) are also accepted but only meaningful when
    they reference a real task list."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open() as f:
            reader = csv.DictReader(f)
            if "openml_task_id" not in (reader.fieldnames or ()):
                raise ValueError(
                    f"{path}: CSV missing required column openml_task_id"
                )
            return [int(r["openml_task_id"]) for r in reader]
    if suffix == ".json":
        payload = json.loads(path.read_text())
        if "task_ids" in payload:
            return [int(t) for t in payload["task_ids"]]
        if payload.get("uses_openml") is False:
            return []
        if payload.get("source_shard"):
            # batch_04 pointer: no task subset; consumer should copy the
            # source shard verbatim instead.
            return []
        raise ValueError(f"{path}: JSON has no task_ids and is not a "
                         "synthetic/pointer manifest")
    raise ValueError(f"unsupported batch file: {path}")


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------


def open_source(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _get_columns(cx: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in cx.execute(f"PRAGMA table_info({table})")]


def _build_filter(task_ids: list[int], methods: list[str] | None,
                  algorithms: list[str] | None, replicas: list[int] | None,
                  stage: str | None) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if task_ids:
        clauses.append(f"openml_task_id IN ({','.join('?' * len(task_ids))})")
        params.extend(task_ids)
    if methods:
        clauses.append(f"method IN ({','.join('?' * len(methods))})")
        params.extend(methods)
    if algorithms:
        clauses.append(f"algorithm IN ({','.join('?' * len(algorithms))})")
        params.extend(algorithms)
    if replicas:
        clauses.append(f"replica IN ({','.join('?' * len(replicas))})")
        params.extend(replicas)
    if stage:
        clauses.append("stage = ?")
        params.append(stage)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


def filter_shard(*, source: Path, out: Path, task_ids: list[int],
                 methods: list[str] | None = None,
                 algorithms: list[str] | None = None,
                 replicas: list[int] | None = None,
                 stage: str | None = None,
                 schema_sql_path: Path = SCHEMA_SQL_PATH,
                 force: bool = False) -> dict:
    if out.exists():
        if not force:
            raise FileExistsError(f"{out} exists; pass --force")
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    src = open_source(source)
    try:
        cols = _get_columns(src, "cc18_jobs")
        col_list = ", ".join(cols)
        where, params = _build_filter(task_ids, methods, algorithms,
                                       replicas, stage)
        rows = src.execute(
            f"SELECT {col_list} FROM cc18_jobs{where}", params,
        ).fetchall()
        shard_meta_rows = src.execute(
            "SELECT * FROM shard_meta"
        ).fetchall()
        shard_meta_cols = _get_columns(src, "shard_meta")
    finally:
        src.close()

    schema_sql = schema_sql_path.read_text(encoding="utf-8")
    with sqlite3.connect(out) as dst:
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.executescript(schema_sql)
        dst.execute("PRAGMA journal_mode=DELETE")
        if rows:
            placeholders = ", ".join(["?"] * len(cols))
            dst.executemany(
                f"INSERT INTO cc18_jobs ({col_list}) VALUES ({placeholders})",
                rows,
            )
        # Add a shard_meta row describing this batch.
        dst.execute(
            "INSERT INTO shard_meta (shard_id, suite_id, panel_version, "
            "generated_at, n_tasks, n_algorithms, n_methods, "
            "n_replicas_max, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                999,  # batch shard ids start at 999
                int(shard_meta_rows[0][shard_meta_cols.index("suite_id")])
                    if shard_meta_rows else 99,
                "cc18_v1_batch",
                _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                len(set(r[cols.index("openml_task_id")] for r in rows)),
                len(set(r[cols.index("algorithm")] for r in rows)) or 0,
                len(set(r[cols.index("method")] for r in rows)) or 0,
                int(max((r[cols.index("replica")] for r in rows), default=0)),
                f"batch shard derived from {source.name}; "
                f"task_ids={task_ids[:8]}{'...' if len(task_ids) > 8 else ''}; "
                f"methods={methods}; algorithms={algorithms}; replicas={replicas}; "
                f"stage={stage}",
            ),
        )
        dst.commit()
    # Cleanup any sidecars.
    for sc in (out.with_suffix(out.suffix + "-wal"),
               out.with_suffix(out.suffix + "-shm")):
        if sc.exists():
            sc.unlink()

    return {
        "n_rows_written": len(rows),
        "out": str(out),
        "task_ids": task_ids,
        "methods": methods, "algorithms": algorithms,
        "replicas": replicas, "stage": stage,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_csv_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _split_int_list(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(tok.strip()) for tok in value.split(",") if tok.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="path to the source SQLite shard (opened "
                             "read-only).")
    parser.add_argument("--batch-file", type=Path, required=True,
                        help="batch CSV (with openml_task_id column) or "
                             "JSON pointer.")
    parser.add_argument("--out", type=Path, required=True,
                        help="output SQLite file (will be created).")
    parser.add_argument("--methods", default=None,
                        help="comma-separated method_ids to keep")
    parser.add_argument("--algorithms", default=None,
                        help="comma-separated algorithms to keep")
    parser.add_argument("--replicas", default=None,
                        help="comma-separated replica indices to keep")
    parser.add_argument("--stage", default=None,
                        help="restrict to one stage")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    task_ids = load_batch_task_ids(args.batch_file)

    if args.dry_run:
        print(json.dumps({
            "source": str(args.source),
            "out": str(args.out),
            "batch_file": str(args.batch_file),
            "task_ids": task_ids,
            "methods": _split_csv_list(args.methods),
            "algorithms": _split_csv_list(args.algorithms),
            "replicas": _split_int_list(args.replicas),
            "stage": args.stage,
        }, indent=2))
        return 0

    res = filter_shard(
        source=args.source, out=args.out,
        task_ids=task_ids,
        methods=_split_csv_list(args.methods),
        algorithms=_split_csv_list(args.algorithms),
        replicas=_split_int_list(args.replicas),
        stage=args.stage,
        force=args.force,
    )
    print(json.dumps(res, indent=2))
    return 0


def confirm_source_unmodified(source: Path) -> dict[str, str]:
    """Helper used by tests: hash the source shard so a caller can
    compare before/after."""
    return {
        "path": str(source),
        "md5": hashlib.md5(Path(source).read_bytes()).hexdigest(),
    }


__all__ = [
    "confirm_source_unmodified",
    "filter_shard",
    "load_batch_task_ids",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
