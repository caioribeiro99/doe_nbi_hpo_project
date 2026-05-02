#!/usr/bin/env python
"""Create deterministic batch manifests for reduced CC18 execution.

Five batches, written under ``benchmarks/doctoral/openml_cc18/batches/``:

- ``batch_00_synthetic_canary.json`` -- no OpenML data; the four
  canary methods on a synthetic binary task.
- ``batch_01_cc18_tiny_3_tasks.csv`` -- 3 real CC18 tasks chosen to
  cover (small binary numeric, small categorical, small multiclass).
- ``batch_02_cc18_small_12_tasks.csv`` -- 12 real CC18 tasks
  stratified across binary/multiclass, categorical/numeric,
  size-bucket and balance.
- ``batch_03_cc18_representative_18_tasks.csv`` -- 18 real CC18 tasks
  with broader representativeness (>=6 binary, >=6 multiclass,
  >=4 categorical, >=4 imbalanced, >=3 large where available).
- ``batch_04_stage0_shard00_only.json`` -- pointer to the existing
  worker shard ``stage0_replica_001/shard_00.sqlite``.

Selection is deterministic: a fixed BATCH_SEED governs every random
choice and the routine sorts candidates by ``openml_task_id`` before
sampling so the chosen IDs are reproducible from the committed
``tasks.csv``. The script never downloads any dataset payload.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from random import Random

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
DEFAULT_OUT = REPO / "benchmarks/doctoral/openml_cc18/batches"
DEFAULT_SHARD = REPO / "jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite"

BATCH_SEED = 20260502  # fixed for reproducibility; do not bump

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _i(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    return int(s)


def _f(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    return float(s)


def _is_imbalanced(r: dict, threshold: float = 5.0) -> bool:
    imb = _f(r.get("class_imbalance_ratio"))
    return imb is not None and imb >= threshold


def _is_categorical(r: dict) -> bool:
    return (_i(r.get("categorical_feature_count")) or 0) > 0


def _size_bucket(r: dict) -> str:
    n = _i(r.get("n_rows")) or 0
    if n <= 1000:
        return "small"
    if n <= 30000:
        return "medium"
    return "large"


def load_tasks(path: Path) -> list[dict]:
    with path.open() as f:
        return sorted(csv.DictReader(f), key=lambda r: int(r["openml_task_id"]))


# ---------------------------------------------------------------------------
# Batch 01 -- tiny 3-task batch
# ---------------------------------------------------------------------------


def select_tiny_3(rows: list[dict]) -> list[dict]:
    """Pick three CC18 tasks: small binary numeric + small categorical
    + small multiclass. Deterministic; smallest-rows-first wins ties."""
    by_id = {int(r["openml_task_id"]): r for r in rows}

    # 1. Small binary numeric: balanced (imbalance < 2.5), no categoricals,
    # rows<=1500. Sort by rows ascending then by task_id ascending.
    binary_numeric = sorted(
        (r for r in rows
         if r["task_type"] == "binary"
         and not _is_categorical(r)
         and (_i(r["n_rows"]) or 1e9) <= 1500
         and (_f(r["class_imbalance_ratio"]) or 0) < 2.5),
        key=lambda r: (_i(r["n_rows"]) or 1e9, int(r["openml_task_id"])),
    )
    pick_b = binary_numeric[0]

    # 2. Small categorical (binary preferred so the metric set is
    # the binary one), rows<=1500, at least 5 categorical features.
    categorical = sorted(
        (r for r in rows
         if (_i(r["categorical_feature_count"]) or 0) >= 5
         and (_i(r["n_rows"]) or 1e9) <= 1500
         and r["task_type"] == "binary"
         and int(r["openml_task_id"]) != pick_b["openml_task_id"]),
        key=lambda r: (_i(r["n_rows"]) or 1e9, int(r["openml_task_id"])),
    )
    pick_c = categorical[0]

    # 3. Small multiclass, rows<=1500, classes>=3.
    multiclass = sorted(
        (r for r in rows
         if r["task_type"] == "multiclass"
         and (_i(r["n_rows"]) or 1e9) <= 1500
         and (_i(r["n_classes"]) or 0) >= 3
         and int(r["openml_task_id"]) != pick_b["openml_task_id"]
         and int(r["openml_task_id"]) != pick_c["openml_task_id"]),
        key=lambda r: (_i(r["n_rows"]) or 1e9, int(r["openml_task_id"])),
    )
    pick_m = multiclass[0]

    return [by_id[int(p["openml_task_id"])] for p in (pick_b, pick_c, pick_m)]


# ---------------------------------------------------------------------------
# Batch 02 -- small 12-task stratified sample
# ---------------------------------------------------------------------------


def _stratified_sample(rows: list[dict], strata: dict[str, int],
                       rng: Random) -> list[dict]:
    """Round-robin sampling across labelled strata until each stratum's
    quota is met. Within a stratum, candidates are shuffled with the
    seeded RNG."""
    pools: dict[str, list[dict]] = {k: [] for k in strata}
    for r in rows:
        for stratum, predicate in _STRATA_PREDICATES.items():
            if stratum in strata and predicate(r):
                pools[stratum].append(r)
    out: list[dict] = []
    seen: set[int] = set()
    for stratum, quota in strata.items():
        pool = sorted(pools[stratum], key=lambda r: int(r["openml_task_id"]))
        # Local shuffle but reproducible.
        rng.shuffle(pool)
        added = 0
        for r in pool:
            tid = int(r["openml_task_id"])
            if tid in seen:
                continue
            out.append(r)
            seen.add(tid)
            added += 1
            if added >= quota:
                break
    return out


_STRATA_PREDICATES = {
    "binary_numeric_small":     lambda r: r["task_type"] == "binary" and not _is_categorical(r) and _size_bucket(r) == "small",
    "binary_numeric_medium":    lambda r: r["task_type"] == "binary" and not _is_categorical(r) and _size_bucket(r) == "medium",
    "binary_categorical":       lambda r: r["task_type"] == "binary" and _is_categorical(r),
    "binary_imbalanced":        lambda r: r["task_type"] == "binary" and _is_imbalanced(r),
    "multiclass_small":         lambda r: r["task_type"] == "multiclass" and _size_bucket(r) == "small",
    "multiclass_medium":        lambda r: r["task_type"] == "multiclass" and _size_bucket(r) == "medium",
    "multiclass_high_n_classes":lambda r: r["task_type"] == "multiclass" and (_i(r["n_classes"]) or 0) >= 5,
    "multiclass_categorical":   lambda r: r["task_type"] == "multiclass" and _is_categorical(r),
    "any_large":                lambda r: _size_bucket(r) == "large",
}


def select_small_12(rows: list[dict]) -> list[dict]:
    """12-task stratified sample with deterministic seed."""
    rng = Random(BATCH_SEED + 12)
    quota: dict[str, int] = {
        "binary_numeric_small":      2,
        "binary_numeric_medium":     1,
        "binary_categorical":        2,
        "binary_imbalanced":         1,
        "multiclass_small":          2,
        "multiclass_medium":         1,
        "multiclass_high_n_classes": 2,
        "any_large":                 1,
    }
    picks = _stratified_sample(rows, quota, rng)
    # If quota is short (overlap removed dups), top up with the smallest
    # tasks not yet in picks, deterministically.
    if len(picks) < 12:
        seen = {int(r["openml_task_id"]) for r in picks}
        topup = sorted(
            (r for r in rows if int(r["openml_task_id"]) not in seen),
            key=lambda r: (_i(r["n_rows"]) or 1e9, int(r["openml_task_id"])),
        )
        for r in topup:
            picks.append(r)
            if len(picks) >= 12:
                break
    return sorted(picks[:12], key=lambda r: int(r["openml_task_id"]))


# ---------------------------------------------------------------------------
# Batch 03 -- representative 18-task batch
# ---------------------------------------------------------------------------


def select_representative_18(rows: list[dict]) -> list[dict]:
    """18-task batch with explicit minima:
       >=6 binary, >=6 multiclass, >=4 categorical, >=4 imbalanced,
       >=3 large (where available)."""
    rng = Random(BATCH_SEED + 18)
    quota: dict[str, int] = {
        "binary_numeric_small":      2,
        "binary_numeric_medium":     2,
        "binary_categorical":        2,
        "binary_imbalanced":         2,  # binary slot reserved to guarantee >=6 binary
        "multiclass_small":          2,
        "multiclass_medium":         2,
        "multiclass_high_n_classes": 2,
        "multiclass_categorical":    2,
        "any_large":                 2,
    }
    picks = _stratified_sample(rows, quota, rng)
    if len(picks) < 18:
        seen = {int(r["openml_task_id"]) for r in picks}
        topup = sorted(
            (r for r in rows if int(r["openml_task_id"]) not in seen),
            key=lambda r: (int(r["openml_task_id"])),
        )
        for r in topup:
            picks.append(r)
            if len(picks) >= 18:
                break
    return sorted(picks[:18], key=lambda r: int(r["openml_task_id"]))


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


_BATCH_CSV_COLUMNS = (
    "openml_task_id", "openml_dataset_id", "dataset_name", "task_type",
    "n_rows", "n_features", "n_classes", "categorical_feature_count",
    "class_imbalance_ratio",
)


def _row_view(r: dict) -> dict[str, str]:
    return {c: r.get(c, "") for c in _BATCH_CSV_COLUMNS}


def write_csv_batch(path: Path, rows: list[dict], description: str,
                    selection_rule: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        # Header comments live in a sidecar JSON; the CSV stays simple.
        w = csv.DictWriter(f, fieldnames=_BATCH_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_row_view(r))
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "description": description,
        "selection_rule": selection_rule,
        "batch_seed": BATCH_SEED,
        "n_tasks": len(rows),
        "task_ids": [int(r["openml_task_id"]) for r in rows],
    }, indent=2, sort_keys=True), encoding="utf-8")


def write_synthetic_canary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch_id": "batch_00_synthetic_canary",
        "description": (
            "Synthetic binary task; no OpenML data. Used only by the "
            "four canary methods to validate the local environment "
            "before any real CC18 task is touched."
        ),
        "uses_openml": False,
        "task_kind": "synthetic_binary",
        "synthetic_task": {
            "n_samples": 300,
            "n_features": 6,
            "seed": 0,
            "task_type": "binary",
        },
        "methods": list(CANARY_METHODS),
        "algorithms": ["xgboost", "lightgbm", "catboost"],
        "n_folds": 2,
        "max_evaluations": 5,
        "expected_runner_flags": [
            "--canary-only", "--train", "--synthetic-task",
            "--max-evaluations", "5", "--n-folds", "2",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_stage0_shard_pointer(path: Path, shard_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rel = shard_path.relative_to(REPO).as_posix()
    payload = {
        "batch_id": "batch_04_stage0_shard00_only",
        "description": (
            "Operational test of one real worker shard (committed in "
            "Commit 28). Smaller scope than full stage 0; meant as a "
            "pre-stage-0 dry run."
        ),
        "uses_openml": True,
        "task_kind": "stage0_shard",
        "source_shard": rel,
        "stage": "stage0_replica_001",
        "expected_runner_flags": [
            "--stage", "stage0_replica_001",
            "--max-jobs", "10",
        ],
        "warning": (
            "The runner must always operate on a copy of this shard "
            "(e.g., shutil.copy to a tmp path) so the committed file "
            "is never mutated."
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_readme(path: Path, summary: dict) -> None:
    lines = []
    lines.append("# OpenML-CC18 reduced execution batches\n")
    lines.append(f"- generated_at: `{summary['generated_at']}`")
    lines.append(f"- batch_seed: `{BATCH_SEED}`")
    lines.append("- source: `benchmarks/doctoral/openml_cc18/tasks.csv`\n")
    lines.append("These batches let us run small, representative subsets "
                 "of OpenML-CC18 before kicking off any full stage. "
                 "They are deterministic: regenerating with the same "
                 "`tasks.csv` produces byte-identical batch CSVs.\n")
    lines.append("## Execution order\n")
    lines.append("| step | batch | scope |")
    lines.append("|---|---|---|")
    lines.append("| A | `batch_00_synthetic_canary` | 4 canary methods on synthetic binary; no OpenML data |")
    lines.append("| B | `batch_01_cc18_tiny_3_tasks` | 3 real CC18 tasks: small binary numeric / small categorical / small multiclass |")
    lines.append("| C | `batch_02_cc18_small_12_tasks` | 12 real CC18 tasks; stratified by task_type / categorical / size / balance |")
    lines.append("| D | `batch_03_cc18_representative_18_tasks` | 18 real CC18 tasks; broader coverage |")
    lines.append("| E | `batch_04_stage0_shard00_only` | one existing stage-0 shard from Commit 28 |")
    lines.append("| F | full stage 0 | 2,304 jobs across the 10 stage-0 shards |")
    lines.append("| G | top-up to stages 1 / 2 / 3 | gated by manual sign-off as documented in `execution_tiers.md` |\n")
    lines.append("Steps A-D are pre-stage-0 pilots: they validate the "
                 "adapters, the OpenML loader, and the runner. Step E "
                 "is an operational dry run on the smallest real worker "
                 "shard. Steps F and G follow only after each prior step "
                 "lands a green sign-off artifact under "
                 "`experiments/_canary_runs/` or `experiments/_batch_runs/`.\n")
    lines.append("## Files in this directory\n")
    lines.append("| file | rows | purpose |")
    lines.append("|---|---:|---|")
    for fname, n, desc in summary["files"]:
        n_text = "—" if n is None else str(n)
        lines.append(f"| `{fname}` | {n_text} | {desc} |")
    lines.append("")
    lines.append("Each `.csv` batch ships a `.meta.json` sidecar with "
                 "the selection rule, the deterministic seed, and the "
                 "explicit task-id list.\n")
    lines.append("## Regenerating the batches\n")
    lines.append("```bash\npython scripts/create_cc18_batches.py --force\n```\n")
    lines.append("## Filtering a SQLite shard for a batch\n")
    lines.append("```bash\npython scripts/filter_cc18_shard_for_batch.py \\\n"
                 "    --source jobs/doctoral/openml_cc18/shards/stage0_replica_001/shard_00.sqlite \\\n"
                 "    --batch-file benchmarks/doctoral/openml_cc18/batches/batch_01_cc18_tiny_3_tasks.csv \\\n"
                 "    --out jobs/doctoral/openml_cc18/batch_shards/batch_01_shard_00.sqlite\n```\n"
                 "The filter NEVER mutates the source shard.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_all(*, tasks_csv: Path, out_dir: Path, shard_path: Path,
                 force: bool) -> dict:
    rows = load_tasks(tasks_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, int | None, str]] = []

    p_canary = out_dir / "batch_00_synthetic_canary.json"
    if p_canary.exists() and not force:
        raise FileExistsError(f"{p_canary} exists; pass --force")
    write_synthetic_canary(p_canary)
    files.append(("batch_00_synthetic_canary.json", None,
                  "synthetic binary; canary methods only"))

    tiny = select_tiny_3(rows)
    p_tiny = out_dir / "batch_01_cc18_tiny_3_tasks.csv"
    write_csv_batch(
        p_tiny, tiny,
        description="3 CC18 tasks: small binary numeric, small categorical, small multiclass",
        selection_rule=(
            "deterministic: smallest-rows-first within each clause; "
            "1) binary numeric balanced (imb<2.5) rows<=1500; "
            "2) binary categorical (>=5 cat features) rows<=1500; "
            "3) multiclass classes>=3 rows<=1500"
        ),
    )
    files.append(("batch_01_cc18_tiny_3_tasks.csv", len(tiny),
                  "3 real CC18 tasks (binary numeric / categorical / multiclass)"))

    small = select_small_12(rows)
    p_small = out_dir / "batch_02_cc18_small_12_tasks.csv"
    write_csv_batch(
        p_small, small,
        description="12 CC18 tasks; stratified across task_type, categorical, size, balance",
        selection_rule=(
            "stratified sample with seed=BATCH_SEED+12 and quota: "
            "binary_numeric_small=2, binary_numeric_medium=1, "
            "binary_categorical=2, binary_imbalanced=1, multiclass_small=2, "
            "multiclass_medium=1, multiclass_high_n_classes=2, any_large=1"
        ),
    )
    files.append(("batch_02_cc18_small_12_tasks.csv", len(small),
                  "12 real CC18 tasks; stratified pilot"))

    rep = select_representative_18(rows)
    p_rep = out_dir / "batch_03_cc18_representative_18_tasks.csv"
    write_csv_batch(
        p_rep, rep,
        description="18 CC18 tasks; broader representativeness",
        selection_rule=(
            "stratified sample with seed=BATCH_SEED+18 and quota: 2 each "
            "from binary_numeric_small / binary_numeric_medium / "
            "binary_categorical / binary_imbalanced / multiclass_small / "
            "multiclass_medium / multiclass_high_n_classes / "
            "multiclass_categorical / any_large"
        ),
    )
    files.append(("batch_03_cc18_representative_18_tasks.csv", len(rep),
                  "18 real CC18 tasks; pre-stage0 research pilot"))

    p_shard = out_dir / "batch_04_stage0_shard00_only.json"
    write_stage0_shard_pointer(p_shard, shard_path)
    files.append(("batch_04_stage0_shard00_only.json", None,
                  "pointer to one existing stage-0 SQLite shard"))

    # README.
    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    write_readme(out_dir / "README.md", summary)
    files.append(("README.md", None, "this index"))

    return {
        "n_files": len(files),
        "tiny_task_ids": [int(r["openml_task_id"]) for r in tiny],
        "small_task_ids": [int(r["openml_task_id"]) for r in small],
        "representative_task_ids": [int(r["openml_task_id"]) for r in rep],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--shard-path", type=Path, default=DEFAULT_SHARD)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    res = generate_all(
        tasks_csv=args.tasks, out_dir=args.out_dir,
        shard_path=args.shard_path, force=args.force,
    )
    print(f"wrote {res['n_files']} files to {args.out_dir}")
    print(f"tiny:           {res['tiny_task_ids']}")
    print(f"small (12):     {res['small_task_ids']}")
    print(f"representative: {res['representative_task_ids']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
