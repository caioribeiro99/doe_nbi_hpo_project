#!/usr/bin/env python
"""Build the CC18 heavy-task policy CSV + report.

Reads:
- ``benchmarks/doctoral/openml_cc18/tasks.csv``
  (one row per CC18 task; required);
- ``benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml``
  (lane defaults; required);
- ``experiments/_stage_runs/batch_02_cc18_small_12_tasks_latest_summary.json``
  (optional; cell runtimes upgrade the classification);
- ``experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json``
  (optional; same).

Writes:
- ``benchmarks/doctoral/openml_cc18/heavy_task_policy.csv``
  (one row per task, with lane and per-task override columns;
  reproducibly regenerable);
- ``benchmarks/doctoral/openml_cc18/heavy_task_policy_report.md``
  (human-readable narrative of the classification).

Classification rules (precedence: extreme > heavy > standard):

extreme if ANY of
  - observed any cell runtime >= 3600 s
  - n_rows >= 75000 AND n_features >= 500
  - n_classes >= 25 AND n_rows >= 20000

heavy if ANY of (and not extreme)
  - observed any cell runtime >= 900 s
  - n_rows >= 40000
  - n_features >= 750
  - categorical_feature_count >= 500
  - n_classes >= 10 AND n_rows >= 10000

standard otherwise.

The builder does NOT touch committed SQLite shards, does NOT
download anything, and does NOT create stage3_signoff.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_TASKS_CSV = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
DEFAULT_GUARDRAILS_YAML = (
    REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
)
DEFAULT_OUT_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
DEFAULT_OUT_MD = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy_report.md"
DEFAULT_BATCH_SUMMARIES = (
    REPO / "experiments/_stage_runs/batch_02_cc18_small_12_tasks_latest_summary.json",
    REPO / "experiments/_stage_runs/batch_03_cc18_representative_18_tasks_latest_summary.json",
)

EXTREME_RUNTIME_S = 3600.0
HEAVY_RUNTIME_S = 900.0
EXTREME_ROWS_FEATURES = (75000, 500)
EXTREME_CLASSES_ROWS = (25, 20000)
HEAVY_ROWS = 40000
HEAVY_FEATURES = 750
HEAVY_CATEGORICAL = 500
HEAVY_CLASSES_ROWS = (10, 10000)

CSV_COLUMNS = (
    "openml_task_id",
    "dataset_name",
    "n_rows",
    "n_features",
    "n_classes",
    "categorical_feature_count",
    "lane",
    "reason",
    "default_max_evaluations",
    "gate_max_evaluations",
    "stage0_max_evaluations",
    "timeout_seconds_per_cell",
    "requires_manual_review_before_full_stage0",
    "notes",
)


@dataclass
class TaskRecord:
    openml_task_id: int
    dataset_name: str
    n_rows: int
    n_features: int
    n_classes: int
    categorical_feature_count: int
    observed_max_runtime_s: float = 0.0
    observed_in_batches: tuple[str, ...] = ()


def _read_tasks_csv(path: Path) -> list[TaskRecord]:
    rows: list[TaskRecord] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(TaskRecord(
                openml_task_id=int(r["openml_task_id"]),
                dataset_name=r.get("dataset_name", ""),
                n_rows=_safe_int(r.get("n_rows")),
                n_features=_safe_int(r.get("n_features")),
                n_classes=_safe_int(r.get("n_classes")),
                categorical_feature_count=_safe_int(
                    r.get("categorical_feature_count"),
                ),
            ))
    return rows


def _safe_int(value: str | None) -> int:
    if value is None:
        return 0
    v = value.strip()
    if not v:
        return 0
    return int(float(v))


def _merge_batch_runtimes(
    records: list[TaskRecord], summary_paths: tuple[Path, ...],
) -> dict[str, int]:
    seen_in: dict[str, int] = {}
    by_id = {r.openml_task_id: r for r in records}
    for sp in summary_paths:
        if not sp.exists():
            continue
        try:
            payload = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        batch_id = payload.get("batch_id") or sp.stem
        cells = payload.get("cells") or []
        for c in cells:
            tid = int(c.get("openml_task_id", -1))
            rt = c.get("runtime_seconds")
            if tid < 0 or rt is None:
                continue
            rec = by_id.get(tid)
            if rec is None:
                continue
            rt_f = float(rt)
            if rt_f > rec.observed_max_runtime_s:
                rec.observed_max_runtime_s = rt_f
            if batch_id not in rec.observed_in_batches:
                rec.observed_in_batches = (*rec.observed_in_batches, batch_id)
        seen_in[batch_id] = len(cells)
    return seen_in


def _classify(rec: TaskRecord) -> tuple[str, str]:
    """Return (lane, reason) for one task."""
    reasons: list[str] = []
    # Extreme tier.
    if rec.observed_max_runtime_s >= EXTREME_RUNTIME_S:
        reasons.append(
            f"observed_max_runtime_s={rec.observed_max_runtime_s:.0f}"
            f">={EXTREME_RUNTIME_S:.0f}"
        )
    if (
        rec.n_rows >= EXTREME_ROWS_FEATURES[0]
        and rec.n_features >= EXTREME_ROWS_FEATURES[1]
    ):
        reasons.append(
            f"n_rows>={EXTREME_ROWS_FEATURES[0]} AND "
            f"n_features>={EXTREME_ROWS_FEATURES[1]}"
        )
    if (
        rec.n_classes >= EXTREME_CLASSES_ROWS[0]
        and rec.n_rows >= EXTREME_CLASSES_ROWS[1]
    ):
        reasons.append(
            f"n_classes>={EXTREME_CLASSES_ROWS[0]} AND "
            f"n_rows>={EXTREME_CLASSES_ROWS[1]}"
        )
    if reasons:
        return "extreme", "; ".join(reasons)

    # Heavy tier.
    if rec.observed_max_runtime_s >= HEAVY_RUNTIME_S:
        reasons.append(
            f"observed_max_runtime_s={rec.observed_max_runtime_s:.0f}"
            f">={HEAVY_RUNTIME_S:.0f}"
        )
    if rec.n_rows >= HEAVY_ROWS:
        reasons.append(f"n_rows>={HEAVY_ROWS}")
    if rec.n_features >= HEAVY_FEATURES:
        reasons.append(f"n_features>={HEAVY_FEATURES}")
    if rec.categorical_feature_count >= HEAVY_CATEGORICAL:
        reasons.append(f"categorical_feature_count>={HEAVY_CATEGORICAL}")
    if (
        rec.n_classes >= HEAVY_CLASSES_ROWS[0]
        and rec.n_rows >= HEAVY_CLASSES_ROWS[1]
    ):
        reasons.append(
            f"n_classes>={HEAVY_CLASSES_ROWS[0]} AND "
            f"n_rows>={HEAVY_CLASSES_ROWS[1]}"
        )
    if reasons:
        return "heavy", "; ".join(reasons)
    return "standard", "metadata within standard envelope"


def build_policy(
    *,
    tasks_csv: Path = DEFAULT_TASKS_CSV,
    summary_paths: tuple[Path, ...] = DEFAULT_BATCH_SUMMARIES,
) -> list[dict]:
    """Return one row dict per task, ready for CSV writing."""
    records = _read_tasks_csv(tasks_csv)
    _merge_batch_runtimes(records, summary_paths)

    rows: list[dict] = []
    for rec in records:
        lane, reason = _classify(rec)
        rows.append({
            "openml_task_id": rec.openml_task_id,
            "dataset_name": rec.dataset_name,
            "n_rows": rec.n_rows,
            "n_features": rec.n_features,
            "n_classes": rec.n_classes,
            "categorical_feature_count": rec.categorical_feature_count,
            "lane": lane,
            "reason": reason,
            # Per-task overrides default to empty so lane defaults
            # apply via the YAML. A human can manually fill these in
            # for a one-off override.
            "default_max_evaluations": "",
            "gate_max_evaluations": "",
            "stage0_max_evaluations": "",
            "timeout_seconds_per_cell": "",
            "requires_manual_review_before_full_stage0":
                "true" if lane == "extreme" else "false",
            "notes": (
                f"observed_max_runtime_s={rec.observed_max_runtime_s:.1f}; "
                f"seen_in={','.join(rec.observed_in_batches) or 'none'}"
            ),
        })
    rows.sort(key=lambda r: r["openml_task_id"])
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})


def write_report(
    rows: list[dict], out_path: Path, *,
    tasks_csv: Path, summary_paths: tuple[Path, ...],
) -> None:
    counts = Counter(r["lane"] for r in rows)
    by_lane: dict[str, list[dict]] = {"extreme": [], "heavy": [], "standard": []}
    for r in rows:
        by_lane[r["lane"]].append(r)

    lines: list[str] = []
    lines.append("# OpenML-CC18 heavy-task policy report\n")
    lines.append(
        f"- generated_at: `"
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`"
    )
    lines.append(f"- tasks_csv: `{_safe_rel(tasks_csv)}`")
    lines.append("- input summaries:")
    for sp in summary_paths:
        if sp.exists():
            lines.append(f"  - `{_safe_rel(sp)}` (read)")
        else:
            lines.append(f"  - `{_safe_rel(sp)}` (missing — skipped)")
    lines.append("")
    lines.append(
        f"## Lane counts ({sum(counts.values())} CC18 tasks)\n"
    )
    lines.append("| lane | count |")
    lines.append("|---|---:|")
    for lane in ("standard", "heavy", "extreme"):
        lines.append(f"| `{lane}` | {counts.get(lane, 0)} |")
    lines.append("")

    for lane in ("extreme", "heavy", "standard"):
        bucket = by_lane[lane]
        if not bucket:
            continue
        lines.append(f"## Lane: {lane} ({len(bucket)})\n")
        lines.append(
            "| task_id | dataset | rows | features | classes | "
            "categorical | reason |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---|")
        for r in bucket:
            lines.append(
                f"| {r['openml_task_id']} | `{r['dataset_name']}` | "
                f"{r['n_rows']} | {r['n_features']} | "
                f"{r['n_classes']} | {r['categorical_feature_count']} | "
                f"{r['reason']} |"
            )
        lines.append("")

    lines.append("## Classification rules\n")
    lines.append("- `extreme`:")
    lines.append(f"  - observed any cell runtime >= {EXTREME_RUNTIME_S:.0f} s, OR")
    lines.append(
        f"  - n_rows >= {EXTREME_ROWS_FEATURES[0]} AND "
        f"n_features >= {EXTREME_ROWS_FEATURES[1]}, OR"
    )
    lines.append(
        f"  - n_classes >= {EXTREME_CLASSES_ROWS[0]} AND "
        f"n_rows >= {EXTREME_CLASSES_ROWS[1]}"
    )
    lines.append("- `heavy` (and not extreme):")
    lines.append(f"  - observed any cell runtime >= {HEAVY_RUNTIME_S:.0f} s, OR")
    lines.append(f"  - n_rows >= {HEAVY_ROWS}, OR")
    lines.append(f"  - n_features >= {HEAVY_FEATURES}, OR")
    lines.append(f"  - categorical_feature_count >= {HEAVY_CATEGORICAL}, OR")
    lines.append(
        f"  - n_classes >= {HEAVY_CLASSES_ROWS[0]} AND "
        f"n_rows >= {HEAVY_CLASSES_ROWS[1]}"
    )
    lines.append("- `standard` otherwise.")
    lines.append("")
    lines.append(
        "Lane defaults (timeouts, max_evaluations, include-by-default) "
        "live in `runtime_guardrails.yaml`. The "
        "`src/doe_xgb/runtime_guardrails.py` helper exposes "
        "`get_task_lane`, `get_timeout_seconds`, "
        "`get_effective_max_evaluations`, and `should_defer_task`."
    )
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--summary", type=Path, action="append",
                        default=None,
                        help="optional stage-run summary JSON; repeat for "
                             "each (defaults to batch_02 + batch_03 latest).")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary_paths: tuple[Path, ...] = (
        tuple(args.summary) if args.summary else DEFAULT_BATCH_SUMMARIES
    )

    rows = build_policy(
        tasks_csv=args.tasks_csv, summary_paths=summary_paths,
    )
    counts = Counter(r["lane"] for r in rows)
    print(
        f"standard={counts.get('standard', 0)} "
        f"heavy={counts.get('heavy', 0)} "
        f"extreme={counts.get('extreme', 0)} "
        f"total={sum(counts.values())}"
    )

    if args.dry_run:
        return 0

    write_csv(rows, args.out_csv)
    write_report(
        rows, args.out_md,
        tasks_csv=args.tasks_csv, summary_paths=summary_paths,
    )
    print(f"csv: {args.out_csv}")
    print(f"md:  {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
