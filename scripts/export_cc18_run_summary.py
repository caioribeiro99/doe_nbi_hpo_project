#!/usr/bin/env python
"""Export a small JSON / Markdown summary from a CC18 run directory.

This is the publish-side of the CC18 result handoff protocol
(``docs/RESULT_HANDOFF_PROTOCOL.md``). It reads the gitignored
execution SQLite files under ``runs/cc18/<run_id>/`` and emits two
small files under ``experiments/_stage_runs/`` that ARE committed:

- ``<run_id>_summary.json``
- ``<run_id>_summary.md``

The summary captures the cross-machine status without shipping any
mutable execution state through Git.

Inputs
------
- ``--run-dir runs/cc18/<run_id>``
- ``--out-json experiments/_stage_runs/<run_id>_summary.json``
- ``--out-md   experiments/_stage_runs/<run_id>_summary.md``
- ``--include-shard-hashes`` — also record the SHA-256 of every
  ``.execution.sqlite`` file (slower but recommended).
- ``--archive-path`` / ``--archive-sha256`` / ``--archive-size`` —
  optional pointers to an external bundle (uploaded out-of-band).

Refusal rules
-------------
- ``--run-dir`` MUST resolve under a directory called ``runs``.
  Summarizing a path under ``jobs/`` (the committed shards) is
  refused outright.
- The summary re-checks each committed source shard against its
  ``run_manifest.json`` MD5; mismatches surface as
  ``source_shards_unchanged: false`` so downstream gates can refuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb._versions import collect_package_versions  # noqa: E402

EXECUTION_SUFFIX = ".execution.sqlite"
DEFAULT_OUT_DIR = REPO / "experiments/_stage_runs"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _is_under_runs(run_dir: Path) -> bool:
    parts = run_dir.resolve().parts
    return "runs" in parts


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _summarize_one_shard(path: Path) -> dict:
    """Inspect one execution SQLite, return per-shard status counts."""
    cx = _open_ro(path)
    try:
        statuses = Counter()
        runtime_total = 0.0
        runtime_n = 0
        runtime_max = 0.0
        started: list[str] = []
        finished: list[str] = []
        failures: list[dict] = []
        for row in cx.execute(
            "SELECT status, runtime_seconds, started_at, finished_at, "
            "method, algorithm, openml_task_id, last_error "
            "FROM cc18_jobs"
        ):
            status, runtime_seconds, started_at, finished_at, \
                method, algorithm, task_id, last_error = row
            statuses[status or "unknown"] += 1
            if runtime_seconds is not None:
                runtime_total += float(runtime_seconds)
                runtime_n += 1
                if float(runtime_seconds) > runtime_max:
                    runtime_max = float(runtime_seconds)
            if started_at:
                started.append(started_at)
            if finished_at:
                finished.append(finished_at)
            if status == "failed":
                failures.append({
                    "openml_task_id": int(task_id),
                    "method": method,
                    "algorithm": algorithm,
                    "last_error": (last_error or "")[:200],
                })
        n_total = sum(statuses.values())
    finally:
        cx.close()
    return {
        "shard_path": _safe_rel(path),
        "n_total": int(n_total),
        "status_counts": dict(statuses),
        "runtime_total_seconds": runtime_total,
        "runtime_max_seconds": runtime_max,
        "runtime_n_recorded": runtime_n,
        "started_at_min": min(started) if started else None,
        "finished_at_max": max(finished) if finished else None,
        "failures": failures,
    }


def _aggregate_failures(per_shard: list[dict]) -> list[dict]:
    """Group all failures by (method, algorithm, openml_task_id)."""
    rollup: dict[tuple[str, str, int], dict] = {}
    for sh in per_shard:
        for f in sh["failures"]:
            key = (f["method"], f["algorithm"], int(f["openml_task_id"]))
            entry = rollup.setdefault(key, {
                "method": f["method"],
                "algorithm": f["algorithm"],
                "openml_task_id": key[2],
                "count": 0,
                "last_error_sample": f["last_error"],
            })
            entry["count"] += 1
    return sorted(
        rollup.values(),
        key=lambda r: (r["method"], r["algorithm"], r["openml_task_id"]),
    )


def _verify_source_shards(manifest: dict) -> dict:
    """Re-hash committed source shards and compare against manifest."""
    drift: list[dict] = []
    rehashed_before: dict[str, str] = {}
    rehashed_now: dict[str, str] = {}
    all_ok = True
    for copy in manifest.get("shard_copies", []):
        src = REPO / copy["source"]
        recorded = copy["source_md5_before"]
        rehashed_before[str(src.name)] = recorded
        if not src.exists():
            drift.append({
                "source": copy["source"], "issue": "source_shard_missing",
            })
            all_ok = False
            continue
        actual = _md5_file(src)
        rehashed_now[str(src.name)] = actual
        if actual != recorded:
            drift.append({
                "source": copy["source"],
                "expected_md5": recorded, "actual_md5": actual,
                "issue": "source_shard_mutated",
            })
            all_ok = False
    return {
        "source_shards_unchanged": all_ok,
        "source_md5_recorded": rehashed_before,
        "source_md5_now": rehashed_now,
        "source_drift": drift,
    }


def export_summary(
    *,
    run_dir: Path,
    out_json: Path,
    out_md: Path,
    include_shard_hashes: bool = False,
    archive_path: str | None = None,
    archive_sha256: str | None = None,
    archive_size: int | None = None,
    batch_id: str | None = None,
) -> dict:
    if not _is_under_runs(run_dir):
        raise ValueError(
            f"--run-dir {run_dir} does not live under a 'runs/' directory; "
            "execution copies must be summarized from outside the "
            "committed shard tree."
        )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    manifest_p = run_dir / "run_manifest.json"
    if not manifest_p.exists():
        raise FileNotFoundError(
            f"run_manifest.json not found under {run_dir}; create the "
            "run dir with scripts/create_cc18_run_dir.py first."
        )
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    exec_files = sorted(
        run_dir.rglob(f"*{EXECUTION_SUFFIX}"),
    )
    per_shard = [_summarize_one_shard(p) for p in exec_files]
    if include_shard_hashes:
        for entry, p in zip(per_shard, exec_files, strict=True):
            entry["sha256"] = _sha256_file(p)

    aggregated_status = Counter()
    runtime_total = 0.0
    runtime_max = 0.0
    runtime_n = 0
    started_all: list[str] = []
    finished_all: list[str] = []
    for sh in per_shard:
        for k, n in sh["status_counts"].items():
            aggregated_status[k] += int(n)
        runtime_total += float(sh["runtime_total_seconds"])
        runtime_max = max(runtime_max, float(sh["runtime_max_seconds"]))
        runtime_n += int(sh["runtime_n_recorded"])
        if sh["started_at_min"]:
            started_all.append(sh["started_at_min"])
        if sh["finished_at_max"]:
            finished_all.append(sh["finished_at_max"])

    failures = _aggregate_failures(per_shard)
    source_check = _verify_source_shards(manifest)

    n_total = int(sum(aggregated_status.values()))
    summary = {
        "schema_version": 1,
        "run_id": manifest.get("run_id"),
        "stage": manifest.get("stage"),
        "batch_id": batch_id,
        "exported_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source_git_sha": manifest.get("source_git_sha"),
        "host": manifest.get("host") or platform.node(),
        "python": manifest.get("python") or sys.version.split()[0],
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "n_total": n_total,
        "n_pending": int(aggregated_status.get("pending", 0)),
        "n_claimed": int(aggregated_status.get("claimed", 0)),
        "n_running": int(aggregated_status.get("running", 0)),
        "n_success": int(aggregated_status.get("success", 0)),
        "n_failed": int(aggregated_status.get("failed", 0)),
        "n_skipped": int(aggregated_status.get("skipped", 0)),
        "status_counts": dict(aggregated_status),
        "runtime_seconds_total": runtime_total,
        "runtime_seconds_max": runtime_max,
        "runtime_n_recorded": runtime_n,
        "started_at_min": min(started_all) if started_all else None,
        "finished_at_max": max(finished_all) if finished_all else None,
        "failures_grouped": failures,
        "n_failures_grouped": len(failures),
        "shards": per_shard,
        "n_shards": len(per_shard),
        "run_dir": _safe_rel(run_dir),
        "run_manifest_path": _safe_rel(manifest_p),
        "execution_suffix": EXECUTION_SUFFIX,
        "source_shards_unchanged": source_check["source_shards_unchanged"],
        "source_md5_recorded": source_check["source_md5_recorded"],
        "source_md5_now": source_check["source_md5_now"],
        "source_drift": source_check["source_drift"],
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size,
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "stage3_signoff_path": _safe_rel(SIGNOFF_FILE),
        "protocol_doc": "docs/RESULT_HANDOFF_PROTOCOL.md",
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8",
    )

    lines: list[str] = []
    lines.append(f"# CC18 stage-run summary -- `{summary['run_id']}`\n")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- stage: `{summary['stage']}`")
    if summary["batch_id"]:
        lines.append(f"- batch_id: `{summary['batch_id']}`")
    lines.append(f"- exported_at: `{summary['exported_at']}`")
    lines.append(f"- source_git_sha: `{str(summary['source_git_sha'])[:12]}`")
    lines.append(f"- host: `{summary['host']}`")
    lines.append(f"- python: `{summary['python']}`")
    lines.append(f"- run_dir: `{summary['run_dir']}` (gitignored)\n")

    lines.append(f"- total jobs: **{summary['n_total']}**")
    lines.append(
        f"- success: **{summary['n_success']}**, "
        f"failed: **{summary['n_failed']}**, "
        f"pending: {summary['n_pending']}, "
        f"running: {summary['n_running']}, "
        f"claimed: {summary['n_claimed']}, "
        f"skipped: {summary['n_skipped']}"
    )
    if summary["runtime_n_recorded"]:
        lines.append(
            f"- runtime: total {summary['runtime_seconds_total']:.1f}s, "
            f"max {summary['runtime_seconds_max']:.2f}s "
            f"across {summary['runtime_n_recorded']} recorded jobs"
        )
    lines.append(
        f"- started_at_min: `{summary['started_at_min']}`, "
        f"finished_at_max: `{summary['finished_at_max']}`\n"
    )

    lines.append(
        f"- source_shards_unchanged: **{summary['source_shards_unchanged']}**"
    )
    lines.append(
        f"- stage3_signoff_present: {summary['stage3_signoff_present']}"
    )
    if summary["archive_path"]:
        lines.append(
            f"- archive: `{summary['archive_path']}` "
            f"(sha256 `{summary['archive_sha256']}`, "
            f"{summary['archive_size_bytes']} bytes)"
        )
    else:
        lines.append("- archive: _(none; large artifacts stay on the publishing machine)_")
    lines.append("")

    lines.append("## Package versions\n")
    lines.append("| package | version |")
    lines.append("|---|---|")
    for name, ver in summary["package_versions"].items():
        lines.append(f"| `{name}` | {ver if ver else 'MISSING'} |")
    lines.append("")

    lines.append("## Per-shard\n")
    if include_shard_hashes:
        lines.append("| shard | total | success | failed | runtime_s | sha256 |")
        lines.append("|---|---:|---:|---:|---:|---|")
    else:
        lines.append("| shard | total | success | failed | runtime_s |")
        lines.append("|---|---:|---:|---:|---:|")
    for sh in summary["shards"]:
        sc = sh["status_counts"]
        if include_shard_hashes:
            lines.append(
                f"| `{Path(sh['shard_path']).name}` | "
                f"{sh['n_total']} | {sc.get('success', 0)} | "
                f"{sc.get('failed', 0)} | "
                f"{sh['runtime_total_seconds']:.2f} | "
                f"`{sh.get('sha256', '')[:16]}` |"
            )
        else:
            lines.append(
                f"| `{Path(sh['shard_path']).name}` | "
                f"{sh['n_total']} | {sc.get('success', 0)} | "
                f"{sc.get('failed', 0)} | "
                f"{sh['runtime_total_seconds']:.2f} |"
            )
    lines.append("")

    if summary["failures_grouped"]:
        lines.append("## Failures (grouped)\n")
        lines.append("| method | algorithm | task_id | count | last_error |")
        lines.append("|---|---|---:|---:|---|")
        for f in summary["failures_grouped"]:
            err = (f["last_error_sample"] or "—")[:60].replace("|", "\\|")
            lines.append(
                f"| `{f['method']}` | `{f['algorithm']}` | "
                f"{f['openml_task_id']} | {f['count']} | {err} |"
            )
        lines.append("")
    else:
        lines.append("## Failures (grouped)\n\n_(none)_\n")

    if summary["source_drift"]:
        lines.append("## Source-shard drift\n")
        lines.append("| source | expected | actual | issue |")
        lines.append("|---|---|---|---|")
        for d in summary["source_drift"]:
            lines.append(
                f"| `{d.get('source','')}` | "
                f"`{d.get('expected_md5','')}` | "
                f"`{d.get('actual_md5','')}` | "
                f"{d.get('issue','')} |"
            )
        lines.append("")

    if (
        summary["n_failed"] == 0
        and summary["n_pending"] == 0
        and summary["n_claimed"] == 0
        and summary["n_running"] == 0
        and summary["source_shards_unchanged"]
        and not summary["stage3_signoff_present"]
    ):
        lines.append("## Verdict: **GREEN**\n")
        lines.append(
            "All jobs landed in a terminal status, the committed shards "
            "are byte-identical to the recorded MD5s, and no stage-3 "
            "sign-off file was created.\n"
        )
    else:
        lines.append("## Verdict: **NOT GREEN**\n")
        lines.append(
            "Re-run / archive the failures; investigate any "
            "source-shard drift; do not promote downstream until this "
            "summary clears.\n"
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="path to runs/cc18/<run_id>")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--include-shard-hashes", action="store_true")
    parser.add_argument("--archive-path", default=None)
    parser.add_argument("--archive-sha256", default=None)
    parser.add_argument("--archive-size", type=int, default=None)
    parser.add_argument("--batch-id", default=None,
                        help="optional batch identifier to embed in "
                             "the summary, when the run dir was created "
                             "for a specific batch.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "run_dir": str(args.run_dir),
            "out_json": str(args.out_json),
            "out_md": str(args.out_md),
            "include_shard_hashes": args.include_shard_hashes,
            "archive_path": args.archive_path,
            "archive_sha256": args.archive_sha256,
            "archive_size": args.archive_size,
            "batch_id": args.batch_id,
        }, indent=2))
        return 0

    try:
        summary = export_summary(
            run_dir=args.run_dir,
            out_json=args.out_json,
            out_md=args.out_md,
            include_shard_hashes=args.include_shard_hashes,
            archive_path=args.archive_path,
            archive_sha256=args.archive_sha256,
            archive_size=args.archive_size,
            batch_id=args.batch_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(
        f"success={summary['n_success']}/{summary['n_total']}  "
        f"failed={summary['n_failed']}  pending={summary['n_pending']}  "
        f"shards={summary['n_shards']}"
    )
    print(f"json: {args.out_json}")
    print(f"md:   {args.out_md}")
    return 0


# Public surface for tests.
def aggregate_failures(per_shard: list[dict]) -> list[dict]:
    return _aggregate_failures(per_shard)


__all__ = [
    "EXECUTION_SUFFIX",
    "aggregate_failures",
    "export_summary",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
