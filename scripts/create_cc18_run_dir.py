#!/usr/bin/env python
"""Create a local CC18 run directory by copying committed shards.

This is the first step of the CC18 result handoff protocol
(``docs/RESULT_HANDOFF_PROTOCOL.md``). It does not run anything —
its only job is to materialize a *gitignored* execution copy of one
or more committed shards so the runner can mutate them without
touching the canonical files.

Inputs
------
- ``--run-id <opaque-string>`` — caller-provided id, normally
  ``<batch_or_stage>__<host>__<utc>``;
- ``--stage <stage>`` — the source stage directory under
  ``jobs/doctoral/openml_cc18/shards/`` (e.g. ``stage0_replica_001``);
- ``--shard shard_NN.sqlite`` (repeatable) — the specific shards to
  copy; pass ``--all`` to copy every shard in the chosen stage.

Outputs (under ``runs/cc18/<run_id>/``)
---------------------------------------
- ``run_manifest.json`` — run id, host, git SHA, source MD5s,
  destination MD5s, list of copied shards;
- ``shards/<stage>/shard_NN.execution.sqlite`` — one execution copy
  per requested shard.

Refusal rules
-------------
- ``--run-root`` MUST resolve to a path inside ``runs/`` (or be a
  test-only override). The script refuses any destination that
  resolves under ``jobs/`` to prevent accidental shadowing of the
  committed shards.
- A missing source shard exits non-zero before any write.
- Re-running with the same ``run_id`` exits non-zero unless
  ``--force`` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_RUN_ROOT = REPO / "runs/cc18"
DEFAULT_SHARDS_ROOT = REPO / "jobs/doctoral/openml_cc18/shards"

EXECUTION_SUFFIX = ".execution.sqlite"


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def create_run_dir(
    *,
    run_id: str,
    stage: str,
    shard_files: list[str] | None,
    use_all: bool = False,
    run_root: Path = DEFAULT_RUN_ROOT,
    shards_root: Path = DEFAULT_SHARDS_ROOT,
    force: bool = False,
) -> dict:
    """Materialize a run dir for one stage. Returns the manifest dict."""
    if not run_id or any(c in run_id for c in (" ", "/", "\\")):
        raise ValueError(
            f"invalid run_id {run_id!r}: must be non-empty and contain "
            "no whitespace, '/', or '\\'"
        )

    # Resolve / refuse destination under jobs/.
    run_root_resolved = run_root.resolve()
    if _is_under(run_root_resolved, (REPO / "jobs").resolve()):
        raise ValueError(
            f"run_root {_safe_rel(run_root)} resolves under jobs/; "
            "execution copies must live outside the committed shard tree."
        )

    src_stage_dir = shards_root / stage
    if not src_stage_dir.is_dir():
        raise FileNotFoundError(f"source stage directory not found: {src_stage_dir}")

    if use_all:
        srcs = sorted(src_stage_dir.glob("shard_*.sqlite"))
    else:
        names = list(shard_files or [])
        if not names:
            raise ValueError(
                "must pass at least one --shard or --all"
            )
        srcs = [src_stage_dir / n for n in names]
        for s in srcs:
            if not s.exists():
                raise FileNotFoundError(f"source shard not found: {s}")

    run_dir = run_root_resolved / run_id
    if run_dir.exists() and not force:
        raise FileExistsError(
            f"run_dir already exists: {_safe_rel(run_dir)} "
            "(pass --force to overwrite)"
        )
    if run_dir.exists() and force:
        shutil.rmtree(run_dir)

    dst_stage_dir = run_dir / "shards" / stage
    dst_stage_dir.mkdir(parents=True, exist_ok=True)

    md5_before: dict[str, str] = {}
    md5_after_source: dict[str, str] = {}
    md5_execution: dict[str, str] = {}
    copies: list[dict] = []

    for src in srcs:
        md5_before[src.name] = _md5(src)
        dst = dst_stage_dir / src.name.replace(".sqlite", EXECUTION_SUFFIX)
        shutil.copy(src, dst)
        md5_after_source[src.name] = _md5(src)
        md5_execution[dst.name] = _md5(dst)
        copies.append({
            "source": _safe_rel(src),
            "execution": _safe_rel(dst),
            "source_md5_before": md5_before[src.name],
            "source_md5_after": md5_after_source[src.name],
            "execution_md5": md5_execution[dst.name],
        })

    sources_unchanged = (md5_before == md5_after_source)

    manifest = {
        "run_id": run_id,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host": platform.node(),
        "uname": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "source_git_sha": _git_sha(),
        "shards_root": _safe_rel(shards_root),
        "run_dir": _safe_rel(run_dir),
        "shards_dir": _safe_rel(dst_stage_dir),
        "shard_copies": copies,
        "n_shards": len(copies),
        "source_shards_unchanged": sources_unchanged,
        "execution_suffix": EXECUTION_SUFFIX,
        "protocol_doc": "docs/RESULT_HANDOFF_PROTOCOL.md",
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", required=True,
                        choices=("stage0_replica_001",
                                 "stage1_topup_to_005",
                                 "stage2_topup_to_010",
                                 "stage3_topup_to_030"))
    parser.add_argument("--shard", action="append", dest="shards",
                        default=[], help="repeat for each shard "
                        "(e.g. --shard shard_00.sqlite). Pass --all "
                        "to take every shard in the stage.")
    parser.add_argument("--all", action="store_true",
                        help="copy every shard under the chosen stage.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--shards-root", type=Path,
                        default=DEFAULT_SHARDS_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved configuration and exit.")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(json.dumps({
            "run_id": args.run_id,
            "stage": args.stage,
            "shards": args.shards,
            "all": args.all,
            "run_root": str(args.run_root),
            "shards_root": str(args.shards_root),
            "force": args.force,
        }, indent=2))
        return 0

    try:
        manifest = create_run_dir(
            run_id=args.run_id,
            stage=args.stage,
            shard_files=args.shards,
            use_all=args.all,
            run_root=args.run_root,
            shards_root=args.shards_root,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "run_id": manifest["run_id"],
        "run_dir": manifest["run_dir"],
        "stage": manifest["stage"],
        "n_shards": manifest["n_shards"],
        "source_shards_unchanged": manifest["source_shards_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
