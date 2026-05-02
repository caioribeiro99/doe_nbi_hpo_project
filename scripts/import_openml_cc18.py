#!/usr/bin/env python
"""Import the OpenML-CC18 benchmark suite metadata (suite_id = 99).

Writes:
  benchmarks/doctoral/openml_cc18/tasks.csv
  benchmarks/doctoral/openml_cc18/datasets.csv
  benchmarks/doctoral/openml_cc18/openml_cc18_metadata.json
  benchmarks/doctoral/openml_cc18/coverage_report.md

Network-bound. Does NOT download raw / processed dataset payloads;
the importer fetches OpenML task + dataset metadata only.

Flags:
  --suite-id 99           override the suite id (default 99 = CC18)
  --out-dir <path>        override the output directory
  --dry-run               fetch but do not write anything to disk
  --validate-only         re-validate an already-written tasks.csv
  --no-download-data      forced; payloads are never fetched
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DEFAULT_SUITE_ID = 99
DEFAULT_OUT = REPO / "benchmarks" / "doctoral" / "openml_cc18"


TASK_COLUMNS = (
    "openml_task_id",
    "openml_dataset_id",
    "dataset_name",
    "target_name",
    "task_type",
    "n_rows",
    "n_features",
    "n_classes",
    "categorical_feature_count",
    "numeric_feature_count",
    "class_distribution",
    "class_imbalance_ratio",
    "license",
    "version",
    "status",
    "url",
    "notes",
)

DATASET_COLUMNS = (
    "openml_dataset_id",
    "dataset_name",
    "version",
    "n_rows",
    "n_features",
    "license",
    "url",
    "n_tasks_in_cc18",
)


def _emit_row(row: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in TASK_COLUMNS:
        v = row.get(c)
        if v is None:
            out[c] = ""
        elif isinstance(v, dict):
            out[c] = json.dumps(v, separators=(",", ":"), sort_keys=True)
        else:
            out[c] = str(v)
    return out


def _fetch_suite(suite_id: int) -> dict[str, Any]:
    """Pull suite + per-task metadata from OpenML."""
    import openml

    suite = openml.study.get_suite(suite_id)
    rows: list[dict[str, Any]] = []
    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id, download_data=False, download_qualities=False)
        except Exception as exc:
            rows.append({
                "openml_task_id": task_id,
                "status": "task_fetch_failed",
                "notes": str(exc)[:200],
            })
            continue

        try:
            ds = openml.datasets.get_dataset(
                task.dataset_id,
                download_data=False,
                download_qualities=True,
                download_features_meta_data=True,
            )
        except Exception as exc:  # pragma: no cover
            rows.append({
                "openml_task_id": task_id,
                "openml_dataset_id": task.dataset_id,
                "status": "dataset_fetch_failed",
                "notes": str(exc)[:200],
            })
            continue

        # Feature counts via .features (skip the target).
        cat = num = 0
        target_name = task.target_name
        try:
            for feat in ds.features.values():
                if feat.name == target_name:
                    continue
                if feat.data_type == "nominal" or feat.data_type == "string":
                    cat += 1
                elif feat.data_type == "numeric":
                    num += 1
        except Exception:
            cat = num = 0

        # Class distribution / imbalance from qualities (cheap).
        n_classes = None
        n_rows = None
        n_features = None
        class_imb = None
        cdist: dict[str, int] | None = None
        q = getattr(ds, "qualities", {}) or {}
        try:
            n_rows = int(q.get("NumberOfInstances")) if "NumberOfInstances" in q else None
            n_features_inc_target = int(q.get("NumberOfFeatures")) if "NumberOfFeatures" in q else None
            n_features = (n_features_inc_target - 1) if n_features_inc_target else (cat + num) or None
            n_classes = int(q.get("NumberOfClasses")) if "NumberOfClasses" in q else None
            maj = q.get("MajorityClassSize")
            mino = q.get("MinorityClassSize")
            if maj is not None and mino is not None and float(mino) > 0:
                class_imb = float(maj) / float(mino)
        except Exception:
            pass

        task_type = "binary" if (n_classes is not None and n_classes == 2) else (
            "multiclass" if (n_classes is not None and n_classes >= 3) else "binary"
        )
        url = f"https://www.openml.org/t/{task_id}"
        rows.append({
            "openml_task_id": int(task_id),
            "openml_dataset_id": int(task.dataset_id),
            "dataset_name": str(ds.name),
            "target_name": str(target_name) if target_name else "",
            "task_type": task_type,
            "n_rows": n_rows,
            "n_features": n_features,
            "n_classes": n_classes,
            "categorical_feature_count": cat,
            "numeric_feature_count": num,
            "class_distribution": cdist,
            "class_imbalance_ratio": round(class_imb, 4) if class_imb is not None else None,
            "license": getattr(ds, "licence", None) or getattr(ds, "license", None),
            "version": getattr(ds, "version", None),
            "status": "ok",
            "url": url,
            "notes": "",
        })

    return {
        "suite_id": suite_id,
        "suite_name": getattr(suite, "name", None),
        "n_tasks": len(suite.tasks),
        "task_ids": list(map(int, suite.tasks)),
        "rows": rows,
    }


def _write_tasks_csv(rows: list[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TASK_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(_emit_row(r))
    return len(rows)


def _write_datasets_csv(rows: list[dict[str, Any]], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[int] = Counter()
    by_id: dict[int, dict[str, Any]] = {}
    for r in rows:
        did = r.get("openml_dataset_id")
        if did is None:
            continue
        counts[int(did)] += 1
        if int(did) not in by_id:
            by_id[int(did)] = r
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DATASET_COLUMNS)
        w.writeheader()
        for did, r in sorted(by_id.items()):
            w.writerow({
                "openml_dataset_id": int(did),
                "dataset_name": r.get("dataset_name") or "",
                "version": r.get("version") or "",
                "n_rows": r.get("n_rows") or "",
                "n_features": r.get("n_features") or "",
                "license": r.get("license") or "",
                "url": f"https://www.openml.org/d/{did}",
                "n_tasks_in_cc18": counts[int(did)],
            })
    return len(by_id)


def _write_coverage_md(rows: list[dict[str, Any]], suite_id: int, out: Path) -> None:
    n = len(rows)
    binary = sum(1 for r in rows if r.get("task_type") == "binary")
    multiclass = sum(1 for r in rows if r.get("task_type") == "multiclass")
    with_cat = sum(1 for r in rows if (r.get("categorical_feature_count") or 0) > 0)
    imbalanced = sum(
        1 for r in rows
        if r.get("class_imbalance_ratio") is not None and float(r["class_imbalance_ratio"]) >= 5.0
    )

    def _bucket(n_rows: Any) -> str:
        if n_rows is None or n_rows == "":
            return "unknown"
        v = int(n_rows)
        if v <= 1000:
            return "small"
        if v <= 30000:
            return "medium"
        return "large"

    buckets = Counter(_bucket(r.get("n_rows")) for r in rows)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(f"# OpenML-CC18 coverage report (suite_id = {suite_id})\n\n")
        f.write(f"- Total tasks: **{n}** (expected = 72)\n")
        f.write(f"- Binary tasks: {binary}\n")
        f.write(f"- Multiclass tasks: {multiclass}\n")
        f.write(f"- Tasks with at least one categorical feature: {with_cat}\n")
        f.write(f"- Imbalanced tasks (ratio >= 5:1): {imbalanced}\n\n")
        f.write("## Size buckets (rows)\n\n")
        for k in ("small", "medium", "large", "unknown"):
            f.write(f"- {k}: {buckets.get(k, 0)}\n")
        f.write("\n## Note\n\n")
        f.write(
            "Generated by `scripts/import_openml_cc18.py`. The 12 internal smoke "
            "datasets (`benchmarks/doctoral/internal_smoke_panel/datasets.csv`) "
            "are NOT part of this coverage; they are kept for development / "
            "smoke / runtime profiling only.\n"
        )


def _validate_tasks_csv(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, [f"missing file: {path}"]
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return 0, [f"empty CSV: {path}"]
        for r in reader:
            rows.append(r)
    errors: list[str] = []
    seen_ids: set[str] = set()
    for r in rows:
        tid = r.get("openml_task_id", "")
        if not tid:
            errors.append("missing openml_task_id in row")
            continue
        if tid in seen_ids:
            errors.append(f"duplicate openml_task_id: {tid}")
        seen_ids.add(tid)
        tt = r.get("task_type", "")
        if tt not in ("binary", "multiclass"):
            errors.append(f"task_type must be binary/multiclass, got {tt!r} for task {tid}")
    return len(rows), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", type=int, default=DEFAULT_SUITE_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-download-data", action="store_true",
                        help="Forced; this importer never downloads dataset payloads.")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    tasks_csv = out_dir / "tasks.csv"

    if args.validate_only:
        n, errors = _validate_tasks_csv(tasks_csv)
        if errors:
            print(f"validate-only: {tasks_csv} -- {len(errors)} errors", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(f"validate-only: {tasks_csv} -- {n} rows OK.")
        return 0

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    suite = _fetch_suite(args.suite_id)
    rows = suite["rows"]
    n = len(rows)

    if args.dry_run:
        print(json.dumps({
            "suite_id": suite["suite_id"],
            "n_tasks": n,
            "first_5": [
                {k: r.get(k) for k in ("openml_task_id", "openml_dataset_id", "dataset_name",
                                       "task_type", "n_rows", "n_features", "n_classes")}
                for r in rows[:5]
            ],
            "would_write": [str(out_dir / f) for f in (
                "tasks.csv", "datasets.csv", "openml_cc18_metadata.json",
                "coverage_report.md")],
        }, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = _write_tasks_csv(rows, tasks_csv)
    n_datasets = _write_datasets_csv(rows, out_dir / "datasets.csv")

    metadata = {
        "suite_id": suite["suite_id"],
        "suite_name": suite["suite_name"],
        "imported_at": started_at,
        "n_tasks": n_tasks,
        "n_unique_datasets": n_datasets,
        "task_ids": suite["task_ids"],
        "openml_pkg_version": _openml_version(),
        "notes": (
            "Generated by scripts/import_openml_cc18.py. The 12 internal smoke "
            "datasets at benchmarks/doctoral/internal_smoke_panel/datasets.csv "
            "are NOT part of this benchmark."
        ),
    }
    (out_dir / "openml_cc18_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _write_coverage_md(rows, args.suite_id, out_dir / "coverage_report.md")
    print(f"Wrote {tasks_csv} ({n_tasks} tasks)")
    print(f"Wrote {(out_dir / 'datasets.csv')} ({n_datasets} unique datasets)")
    print(f"Wrote {(out_dir / 'openml_cc18_metadata.json')}")
    print(f"Wrote {(out_dir / 'coverage_report.md')}")
    return 0


def _openml_version() -> str:
    try:
        import openml

        return str(getattr(openml, "__version__", "?"))
    except Exception:
        return "?"


if __name__ == "__main__":
    sys.exit(main())
