#!/usr/bin/env python
"""Aggregate the three stage-0 lane summaries into a signoff plan.

stage 0 replica 1 has three published stage-run summaries
(``stage0_standard_lane_latest_summary.json``,
``stage0_heavy_lane_latest_summary.json``,
``stage0_extreme_lane_latest_summary.json``). This script reads all
three, validates that the aggregate replica is internally
consistent (same policy_version, all green, no committed-shard
mutation, no stage-3 sign-off), and emits a small **planning**
artifact under ``experiments/_stage_runs/`` with
``signoff_status = "planned_not_signed"``.

The artifact is **not** a sign-off file. Creating the actual
``jobs/doctoral/openml_cc18/stage3_signoff.json`` unlocks the
stage-3 top-up machinery (which the runner currently refuses on
the absence of that file) and is intentionally a separate,
operator-reviewed commit.

Refusal rules
-------------
- any of the three lane summaries missing;
- any lane summary reports failures, failed_timeout, pending,
  or running cells;
- any lane summary reports ``source_shards_unchanged == false``;
- the three summaries disagree on ``policy_version``;
- ``jobs/doctoral/openml_cc18/stage3_signoff.json`` exists.

What this script does NOT do
----------------------------
- create ``stage3_signoff.json`` (the sign-off file stays absent);
- unlock or start stage-3 / top-up execution;
- run any additional OpenML training;
- rerun standard / heavy / extreme lanes;
- mutate any committed SQLite shard;
- regenerate ``heavy_task_policy.csv`` /
  ``runtime_guardrails.yaml``;
- ship or aggregate per-cell fold metrics or fitted models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb._versions import collect_package_versions  # noqa: E402

DEFAULT_STAGE_RUNS_DIR = REPO / "experiments/_stage_runs"
DEFAULT_STANDARD = DEFAULT_STAGE_RUNS_DIR / "stage0_standard_lane_latest_summary.json"
DEFAULT_HEAVY = DEFAULT_STAGE_RUNS_DIR / "stage0_heavy_lane_latest_summary.json"
DEFAULT_EXTREME = DEFAULT_STAGE_RUNS_DIR / "stage0_extreme_lane_latest_summary.json"
DEFAULT_PLAN = (
    DEFAULT_STAGE_RUNS_DIR / "stage0_extreme_lane_plan_latest_summary.json"
)
DEFAULT_OUT_JSON = (
    DEFAULT_STAGE_RUNS_DIR
    / "stage0_replica_001_signoff_plan_latest_summary.json"
)
DEFAULT_OUT_MD = (
    DEFAULT_STAGE_RUNS_DIR
    / "stage0_replica_001_signoff_plan_latest_summary.md"
)
DEFAULT_TASKS_CSV = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"
SIGNOFF_DOC = "docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md"

RUN_ID = "stage0_replica_001_signoff_plan_latest"
STAGE = "stage0_replica_001"
EXPECTED_N_JOBS_TOTAL = 2304
EXPECTED_N_STANDARD = 684
EXPECTED_N_HEAVY = 156
EXPECTED_N_EXTREME = 24
EXPECTED_N_TASKS = 72
EXPECTED_N_STANDARD_TASKS = 57
EXPECTED_N_HEAVY_TASKS = 13
EXPECTED_N_EXTREME_TASKS = 2

CANARY_METHODS = (
    "default_gbdt", "random_search", "tpe_optuna", "doe_rsm_vrf_true_nbi",
)
ALGORITHMS = ("xgboost", "lightgbm", "catboost")

BINARY_METRIC_KEYS = (
    "accuracy", "precision", "recall", "specificity",
    "balanced_accuracy",
)
MULTICLASS_METRIC_KEYS = (
    "accuracy", "balanced_accuracy", "f1_macro", "mcc",
    "roc_auc_ovr_macro", "pr_auc_ovr_macro",
    "brier_multiclass", "ece_multiclass",
)


class SignoffRefusalError(RuntimeError):
    """Raised when the aggregate stage-0 replica is not ready for
    operator review."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
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


def _load_tasks_csv(path: Path = DEFAULT_TASKS_CSV) -> dict[int, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {int(r["openml_task_id"]): r for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Lane summary validation
# ---------------------------------------------------------------------------


def _load_lane(
    path: Path, *, lane_name: str, expected_executed: int,
) -> dict:
    if not path.exists():
        raise SignoffRefusalError(
            f"{lane_name}-lane summary not found at {path}; "
            f"run scripts/run_stage0_{lane_name}_lane.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    n_executed = int(payload.get(
        "n_jobs_executed", payload.get("n_success", 0),
    ))
    n_failed = int(payload.get(
        "n_jobs_failed", payload.get("n_failed", 0),
    ))
    failed_timeout = int(payload.get("n_jobs_failed_timeout", 0))
    failed_other = int(payload.get("n_jobs_failed_other", 0))
    pending = int(payload.get(
        "n_jobs_pending_after", payload.get("n_pending", 0),
    ))
    running = int(payload.get(
        "n_jobs_running_after", payload.get("n_running", 0),
    ))
    unchanged = bool(payload.get("source_shards_unchanged", False))
    signoff = bool(payload.get("stage3_signoff_present", False))

    if n_executed != expected_executed:
        raise SignoffRefusalError(
            f"{lane_name}-lane summary reports n_jobs_executed="
            f"{n_executed}; expected {expected_executed}."
        )
    if n_failed != 0 or failed_timeout != 0 or failed_other != 0:
        raise SignoffRefusalError(
            f"{lane_name}-lane summary is not green: n_failed="
            f"{n_failed}, failed_timeout={failed_timeout}, "
            f"failed_other={failed_other}"
        )
    if pending != 0 or running != 0:
        raise SignoffRefusalError(
            f"{lane_name}-lane has unfinished work: pending="
            f"{pending}, running={running}"
        )
    if not unchanged:
        raise SignoffRefusalError(
            f"{lane_name}-lane summary reports "
            "source_shards_unchanged=False."
        )
    if signoff:
        raise SignoffRefusalError(
            f"{lane_name}-lane summary reports "
            "stage3_signoff_present=True; cannot plan signoff over a "
            "lane that already records the signoff file."
        )
    payload["__path__"] = str(path)
    payload["__lane_name__"] = lane_name
    return payload


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------


def _mean_or_none(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return float(statistics.fmean(clean))


def _aggregate_metric(
    cells: list[dict], metric_key: str,
) -> float | None:
    vals: list[float] = []
    for c in cells:
        agg = c.get("aggregate_metrics") or {}
        v = agg.get(metric_key)
        if v is None:
            continue
        try:
            f = float(v)
        except Exception:
            continue
        # Skip NaN.
        if f != f:  # noqa: PLR0124
            continue
        vals.append(f)
    return _mean_or_none(vals)


def _by_task_type_breakdown(
    cells: list[dict], tasks_meta: dict[int, dict],
) -> dict[str, dict]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        tid = int(c.get("openml_task_id", -1))
        if tid < 0:
            continue
        meta = tasks_meta.get(tid)
        if not meta:
            # Infer from cell metric_keys if metadata is missing.
            mk = c.get("metric_keys") or []
            task_type = (
                "multiclass" if "f1_macro" in mk or "mcc" in mk
                else "binary"
            )
        else:
            task_type = (meta.get("task_type") or "binary").strip()
        by_type[task_type].append(c)
    out: dict[str, dict] = {}
    for task_type, sub in by_type.items():
        keys = (
            BINARY_METRIC_KEYS if task_type == "binary"
            else MULTICLASS_METRIC_KEYS
        )
        agg = {k: _aggregate_metric(sub, k) for k in keys}
        out[task_type] = {
            "n_cells": len(sub),
            "n_tasks": len(
                {int(c["openml_task_id"]) for c in sub}
            ),
            "metric_means": agg,
            "metric_keys_available": [k for k in keys if agg[k] is not None],
            "metric_keys_missing": [k for k in keys if agg[k] is None],
        }
    return out


def _by_group_breakdown(
    cells: list[dict], group_key: str, metric_keys: tuple[str, ...],
) -> list[dict]:
    by_group: dict[object, list[dict]] = defaultdict(list)
    for c in cells:
        by_group[c.get(group_key)].append(c)
    out: list[dict] = []
    for key in sorted(by_group.keys(), key=lambda k: str(k)):
        sub = by_group[key]
        means = {m: _aggregate_metric(sub, m) for m in metric_keys}
        out.append({
            group_key: key,
            "n_cells": len(sub),
            "metric_means": means,
        })
    return out


def _slowest_cells(cells: list[dict], n: int = 12) -> list[dict]:
    rated = [
        c for c in cells if c.get("runtime_seconds") is not None
    ]
    rated.sort(key=lambda c: float(c["runtime_seconds"]), reverse=True)
    return [
        {
            "openml_task_id": int(c["openml_task_id"]),
            "method": c["method"],
            "algorithm": c["algorithm"],
            "runtime_seconds": float(c["runtime_seconds"]),
            "lane": c.get("lane"),
        }
        for c in rated[:n]
    ]


def _slowest_tasks(cells: list[dict], n: int = 10) -> list[dict]:
    by_task: dict[int, list[float]] = defaultdict(list)
    for c in cells:
        rt = c.get("runtime_seconds")
        if rt is None:
            continue
        by_task[int(c["openml_task_id"])].append(float(rt))
    rows = [
        {
            "openml_task_id": tid,
            "n_cells": len(rts),
            "runtime_total_seconds": float(sum(rts)),
            "runtime_max_seconds": float(max(rts)),
        }
        for tid, rts in by_task.items()
    ]
    rows.sort(key=lambda r: r["runtime_total_seconds"], reverse=True)
    return rows[:n]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def build_signoff_plan(
    *,
    standard_path: Path = DEFAULT_STANDARD,
    heavy_path: Path = DEFAULT_HEAVY,
    extreme_path: Path = DEFAULT_EXTREME,
    plan_path: Path = DEFAULT_PLAN,
    tasks_csv: Path = DEFAULT_TASKS_CSV,
    out_json: Path = DEFAULT_OUT_JSON,
    out_md: Path = DEFAULT_OUT_MD,
    write_summary: bool = True,
) -> dict:
    if SIGNOFF_FILE.exists():
        raise SignoffRefusalError(
            f"{SIGNOFF_FILE} already exists; refusing to plan signoff "
            "over an already-signed replica."
        )

    standard = _load_lane(
        standard_path, lane_name="standard",
        expected_executed=EXPECTED_N_STANDARD,
    )
    heavy = _load_lane(
        heavy_path, lane_name="heavy",
        expected_executed=EXPECTED_N_HEAVY,
    )
    extreme = _load_lane(
        extreme_path, lane_name="extreme",
        expected_executed=EXPECTED_N_EXTREME,
    )

    pv_std = standard.get("policy_version")
    pv_hvy = heavy.get("policy_version")
    pv_ext = extreme.get("policy_version")
    policy_versions = {pv_std, pv_hvy, pv_ext}
    if len(policy_versions) != 1 or None in policy_versions:
        raise SignoffRefusalError(
            f"lane policy_versions differ: standard={pv_std} "
            f"heavy={pv_hvy} extreme={pv_ext}"
        )
    policy_version = next(iter(policy_versions))

    plan: dict | None = None
    if plan_path.exists():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # Tag every cell with its lane so per-lane aggregates work.
    def _tag(cells: list[dict], lane: str) -> list[dict]:
        out = []
        for c in cells:
            c2 = dict(c)
            c2["lane"] = lane
            out.append(c2)
        return out

    std_cells = _tag(standard.get("cells", []), "standard")
    hvy_cells = _tag(heavy.get("cells", []), "heavy")
    ext_cells = _tag(extreme.get("cells", []), "extreme")
    success_cells = [
        c for c in (std_cells + hvy_cells + ext_cells)
        if c.get("status") == "success"
    ]

    tasks_meta = _load_tasks_csv(tasks_csv)

    # Aggregate counts.
    n_std_success = int(standard["n_jobs_executed"])
    n_hvy_success = int(heavy["n_jobs_executed"])
    n_ext_success = int(extreme["n_jobs_executed"])
    n_canary_success_total = n_std_success + n_hvy_success + n_ext_success
    n_failed_total = sum(
        int(s.get("n_jobs_failed", 0)) for s in (standard, heavy, extreme)
    )
    n_failed_timeout_total = sum(
        int(s.get("n_jobs_failed_timeout", 0))
        for s in (standard, heavy, extreme)
    )
    n_pending_total = sum(
        int(s.get("n_jobs_pending_after", 0))
        for s in (standard, heavy, extreme)
    )
    n_running_total = sum(
        int(s.get("n_jobs_running_after", 0))
        for s in (standard, heavy, extreme)
    )
    non_canary_refused_total = sum(
        int(s.get("n_jobs_refused_non_canary", 0))
        for s in (standard, heavy, extreme)
    )

    # Deferred / skipped counts by lane.
    deferred_by_lane = {
        "standard": {
            "deferred_heavy": int(
                standard.get("n_jobs_deferred_heavy", 0),
            ),
            "deferred_extreme": int(
                standard.get("n_jobs_deferred_extreme", 0),
            ),
        },
        "heavy": {
            "deferred_standard": int(
                heavy.get("n_jobs_deferred_standard", 0),
            ),
            "deferred_extreme": int(
                heavy.get("n_jobs_deferred_extreme", 0),
            ),
        },
        "extreme": {
            "deferred_standard": int(
                extreme.get("n_jobs_deferred_standard", 0),
            ),
            "deferred_heavy": int(
                extreme.get("n_jobs_deferred_heavy", 0),
            ),
        },
    }

    # Task coverage.
    task_lane_counts = (
        standard.get("task_lane_counts_universe")
        or heavy.get("task_lane_counts_universe")
        or extreme.get("task_lane_counts_universe")
        or {}
    )

    # Runtime by lane.
    runtime_by_lane = {
        "standard": float(standard.get("runtime_seconds_runner_total", 0.0)),
        "heavy": float(heavy.get("runtime_seconds_runner_total", 0.0)),
        "extreme": float(extreme.get("runtime_seconds_runner_total", 0.0)),
    }
    runtime_total = sum(runtime_by_lane.values())

    # Slowest cells / tasks (overall, success only).
    all_success_with_lane = []
    for lane_cells, lane in (
        (std_cells, "standard"),
        (hvy_cells, "heavy"),
        (ext_cells, "extreme"),
    ):
        for c in lane_cells:
            if c.get("status") != "success":
                continue
            c2 = dict(c)
            c2["lane"] = lane
            all_success_with_lane.append(c2)

    slowest_cells_overall = _slowest_cells(all_success_with_lane, n=12)
    slowest_tasks_overall = _slowest_tasks(all_success_with_lane, n=10)

    # Per-task-type metric breakdown over all success cells.
    by_task_type = _by_task_type_breakdown(
        all_success_with_lane, tasks_meta,
    )

    # Per-lane / per-method / per-algorithm / per-task breakdowns
    # over the union of all metric keys (we mix binary + multiclass).
    union_metric_keys = tuple(
        sorted(set(BINARY_METRIC_KEYS) | set(MULTICLASS_METRIC_KEYS))
    )
    per_lane_breakdown = _by_group_breakdown(
        all_success_with_lane, "lane", union_metric_keys,
    )
    per_method_breakdown = _by_group_breakdown(
        all_success_with_lane, "method", union_metric_keys,
    )
    per_algorithm_breakdown = _by_group_breakdown(
        all_success_with_lane, "algorithm", union_metric_keys,
    )
    per_task_breakdown = _by_group_breakdown(
        all_success_with_lane, "openml_task_id", union_metric_keys,
    )

    # Methods + algorithms actually observed.
    methods_observed = sorted({c["method"] for c in success_cells})
    algorithms_observed = sorted({c["algorithm"] for c in success_cells})
    tasks_observed = sorted({
        int(c["openml_task_id"]) for c in success_cells
    })

    # Hashes of the three lane summary JSON files.
    hashes = {
        "standard_sha256": _sha256(standard_path),
        "heavy_sha256": _sha256(heavy_path),
        "extreme_sha256": _sha256(extreme_path),
    }

    all_lane_summaries_green = (
        n_failed_total == 0
        and n_failed_timeout_total == 0
        and n_pending_total == 0
        and n_running_total == 0
    )
    source_shards_unchanged_all_lanes = all(
        bool(s.get("source_shards_unchanged", False))
        for s in (standard, heavy, extreme)
    )
    no_pending_running_failed_all_lanes = (
        n_failed_total == 0
        and n_failed_timeout_total == 0
        and n_pending_total == 0
        and n_running_total == 0
    )

    # Final recommendation.
    if (
        all_lane_summaries_green
        and source_shards_unchanged_all_lanes
        and n_canary_success_total == (
            EXPECTED_N_STANDARD + EXPECTED_N_HEAVY + EXPECTED_N_EXTREME
        )
        and not SIGNOFF_FILE.exists()
    ):
        recommendation = "ready_for_operator_review"
    else:
        recommendation = "blocked_resolve_above"

    summary = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "stage": STAGE,
        "lane": "aggregate",
        "exported_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "signoff_status": "planned_not_signed",
        "stage3_signoff_present": SIGNOFF_FILE.exists(),
        "stage3_signoff_path": _safe_rel(SIGNOFF_FILE),
        "signoff_plan_doc": SIGNOFF_DOC,
        # Lane summary paths + content hashes.
        "standard_summary_path": _safe_rel(standard_path),
        "heavy_summary_path": _safe_rel(heavy_path),
        "extreme_summary_path": _safe_rel(extreme_path),
        "extreme_lane_plan_summary_path": (
            _safe_rel(plan_path) if plan_path.exists() else None
        ),
        "standard_sha256": hashes["standard_sha256"],
        "heavy_sha256": hashes["heavy_sha256"],
        "extreme_sha256": hashes["extreme_sha256"],
        "extreme_lane_plan_sha256": (
            _sha256(plan_path) if plan_path.exists() else None
        ),
        # Aggregate invariants.
        "policy_version": policy_version,
        "policy_version_consistent": len(policy_versions) == 1,
        "all_lane_summaries_green": all_lane_summaries_green,
        "source_shards_unchanged_all_lanes": source_shards_unchanged_all_lanes,
        "no_pending_running_failed_all_lanes":
            no_pending_running_failed_all_lanes,
        # Aggregate counts.
        "n_jobs_total_expected": EXPECTED_N_JOBS_TOTAL,
        "n_standard_success": n_std_success,
        "n_heavy_success": n_hvy_success,
        "n_extreme_success": n_ext_success,
        "n_canary_success_total": n_canary_success_total,
        "n_failed_total": n_failed_total,
        "n_failed_timeout_total": n_failed_timeout_total,
        "n_pending_total": n_pending_total,
        "n_running_total": n_running_total,
        "non_canary_refused_total": non_canary_refused_total,
        "deferred_skip_by_lane": deferred_by_lane,
        # Task / method / algorithm coverage.
        "task_coverage": {
            "n_tasks_total_expected": EXPECTED_N_TASKS,
            "n_standard_tasks_expected": EXPECTED_N_STANDARD_TASKS,
            "n_heavy_tasks_expected": EXPECTED_N_HEAVY_TASKS,
            "n_extreme_tasks_expected": EXPECTED_N_EXTREME_TASKS,
            "task_lane_counts_universe": dict(task_lane_counts),
            "n_tasks_observed_success": len(tasks_observed),
            "tasks_observed_success": tasks_observed,
        },
        "method_coverage": {
            "expected": list(CANARY_METHODS),
            "observed": methods_observed,
        },
        "algorithm_coverage": {
            "expected": list(ALGORITHMS),
            "observed": algorithms_observed,
        },
        # Runtime summary.
        "runtime_seconds_by_lane": runtime_by_lane,
        "runtime_seconds_total": runtime_total,
        "slowest_cells_overall": slowest_cells_overall,
        "slowest_tasks_overall": slowest_tasks_overall,
        # Metric aggregates.
        "metric_aggregates_by_task_type": by_task_type,
        "metric_aggregates_per_lane": per_lane_breakdown,
        "metric_aggregates_per_method": per_method_breakdown,
        "metric_aggregates_per_algorithm": per_algorithm_breakdown,
        "metric_aggregates_per_task": per_task_breakdown,
        # Caveats.
        "caveats": [
            "isolet (task 3481) ran in the standard lane but its "
            "doe_rsm_vrf_true_nbi catboost cell took ~1078.6 s. The "
            "observed-runtime>=900 rule would promote it to heavy "
            "in a future replica via "
            "scripts/build_cc18_heavy_task_policy.py; this replica "
            "intentionally stays on the Commit 38 policy_version "
            "to keep all four stage-0 artifacts consistent.",
            "Devnagari-Script (task 167121) ran in the extreme lane "
            "under the policy-defined extreme.stage0_max_evaluations=1 "
            "(YAML default). Standard- and heavy-lane cells ran at "
            "max_evaluations=5. random_search / tpe_optuna / "
            "default_gbdt cells on Devnagari therefore exercised "
            "fewer configurations than the rest of the panel, but "
            "doe_rsm_vrf_true_nbi (which uses n_doe=max(2*d, "
            "max_evaluations)) hit the d=4 floor and is unchanged "
            "from batch_03. Headline metrics for Devnagari are NOT "
            "directly budget-equivalent to the other 70 tasks.",
            "Heavy- and extreme-lane handling is a reproducibility "
            "feature (timeout protection, per-lane budgets), not a "
            "post-hoc result filter. The full replica is the 864 "
            "canary successes across 72 tasks; subsetting analyses "
            "should respect lane assignments only as a runtime "
            "budget marker.",
        ],
        # Final recommendation.
        "final_recommendation": recommendation,
        "what_a_later_signoff_commit_should_do": [
            "create jobs/doctoral/openml_cc18/stage3_signoff.json "
            "(operator-reviewed)",
            "record final policy_version (currently "
            f"{policy_version})",
            "record the three lane summary SHA-256 hashes recorded "
            "in this signoff plan",
            "record explicit approval metadata (operator, "
            "timestamp, justification)",
        ],
        # Provenance.
        "package_versions": collect_package_versions((
            "xgboost", "lightgbm", "catboost", "optuna",
            "scikit-learn", "openml", "smac", "pymoo", "dehb",
            "numpy", "pandas",
        )),
        "platform": _platform(),
        "git_sha": _git_sha(),
        "extreme_lane_plan": (
            {
                "execution_status": plan.get("execution_status"),
                "exported_at": plan.get("exported_at"),
                "n_runnable_extreme_canary": plan.get(
                    "n_runnable_extreme_canary",
                ),
                "extreme_tasks_to_execute": plan.get(
                    "extreme_tasks_to_execute",
                ),
            }
            if plan is not None else None
        ),
    }

    # Defensive cross-checks against the original prompt.
    assert n_std_success == EXPECTED_N_STANDARD
    assert n_hvy_success == EXPECTED_N_HEAVY
    assert n_ext_success == EXPECTED_N_EXTREME
    assert n_canary_success_total == (
        EXPECTED_N_STANDARD + EXPECTED_N_HEAVY + EXPECTED_N_EXTREME
    )

    if write_summary:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        out_md.write_text(_render_md(summary), encoding="utf-8")

    return summary


def _fmt_pct(v: float | None, total: float) -> str:
    if v is None or total <= 0:
        return "—"
    return f"{v / total * 100:.1f}%"


def _fmt_metric(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def _render_md(summary: dict) -> str:
    lines: list[str] = []
    lines.append(
        "# stage 0 replica 001 — aggregate signoff plan\n"
    )
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- stage: `{summary['stage']}`")
    lines.append(f"- exported_at: `{summary['exported_at']}`")
    lines.append(
        f"- **signoff_status: `{summary['signoff_status']}`**"
    )
    lines.append(
        f"- stage3_signoff_present: {summary['stage3_signoff_present']}"
    )
    lines.append(
        f"- final_recommendation: **{summary['final_recommendation']}**"
    )
    lines.append(
        f"- signoff_plan_doc: `{summary['signoff_plan_doc']}`\n"
    )

    lines.append("## Lane summary references\n")
    lines.append("| lane | path | sha256 |")
    lines.append("|---|---|---|")
    for lane in ("standard", "heavy", "extreme"):
        lines.append(
            f"| `{lane}` | `{summary[f'{lane}_summary_path']}` | "
            f"`{summary[f'{lane}_sha256']}` |"
        )
    if summary.get("extreme_lane_plan_summary_path"):
        lines.append(
            f"| `extreme_plan` | "
            f"`{summary['extreme_lane_plan_summary_path']}` | "
            f"`{summary['extreme_lane_plan_sha256']}` |"
        )
    lines.append("")

    lines.append("## Aggregate invariants\n")
    lines.append(
        f"- policy_version: `{str(summary['policy_version'])[:16]}` "
        f"(consistent across lanes: "
        f"{summary['policy_version_consistent']})"
    )
    lines.append(
        f"- all_lane_summaries_green: "
        f"**{summary['all_lane_summaries_green']}**"
    )
    lines.append(
        f"- source_shards_unchanged_all_lanes: "
        f"**{summary['source_shards_unchanged_all_lanes']}**"
    )
    lines.append(
        f"- no_pending_running_failed_all_lanes: "
        f"**{summary['no_pending_running_failed_all_lanes']}**\n"
    )

    lines.append("## Aggregate counts\n")
    lines.append(f"- n_jobs_total_expected: {summary['n_jobs_total_expected']}")
    lines.append(f"- n_standard_success: {summary['n_standard_success']}")
    lines.append(f"- n_heavy_success: {summary['n_heavy_success']}")
    lines.append(f"- n_extreme_success: {summary['n_extreme_success']}")
    lines.append(
        f"- **n_canary_success_total: {summary['n_canary_success_total']}**"
    )
    lines.append(f"- n_failed_total: {summary['n_failed_total']}")
    lines.append(
        f"- n_failed_timeout_total: {summary['n_failed_timeout_total']}"
    )
    lines.append(f"- n_pending_total: {summary['n_pending_total']}")
    lines.append(f"- n_running_total: {summary['n_running_total']}")
    lines.append(
        f"- non_canary_refused_total: "
        f"{summary['non_canary_refused_total']}\n"
    )

    tc = summary["task_coverage"]
    lines.append("## Task coverage\n")
    lines.append(
        f"- n_tasks_total_expected: {tc['n_tasks_total_expected']}"
    )
    lines.append(
        f"- n_standard_tasks_expected: "
        f"{tc['n_standard_tasks_expected']}"
    )
    lines.append(
        f"- n_heavy_tasks_expected: {tc['n_heavy_tasks_expected']}"
    )
    lines.append(
        f"- n_extreme_tasks_expected: {tc['n_extreme_tasks_expected']}"
    )
    lines.append(
        f"- task_lane_counts_universe: {tc['task_lane_counts_universe']}"
    )
    lines.append(
        f"- n_tasks_observed_success: {tc['n_tasks_observed_success']}\n"
    )

    mc = summary["method_coverage"]
    ac = summary["algorithm_coverage"]
    lines.append("## Method + algorithm coverage\n")
    lines.append(f"- methods expected: `{mc['expected']}`")
    lines.append(f"- methods observed: `{mc['observed']}`")
    lines.append(f"- algorithms expected: `{ac['expected']}`")
    lines.append(f"- algorithms observed: `{ac['observed']}`\n")

    lines.append("## Runtime summary\n")
    rt = summary["runtime_seconds_by_lane"]
    lines.append("| lane | runtime (s) | runtime (h) |")
    lines.append("|---|---:|---:|")
    for lane in ("standard", "heavy", "extreme"):
        lines.append(
            f"| `{lane}` | {rt[lane]:.1f} | {rt[lane] / 3600:.2f} |"
        )
    lines.append(
        f"| **total** | **{summary['runtime_seconds_total']:.1f}** | "
        f"**{summary['runtime_seconds_total'] / 3600:.2f}** |\n"
    )

    lines.append("### Slowest cells (success, all lanes)\n")
    lines.append("| task_id | method | algorithm | lane | runtime_s |")
    lines.append("|---:|---|---|---|---:|")
    for c in summary["slowest_cells_overall"]:
        lines.append(
            f"| {c['openml_task_id']} | `{c['method']}` | "
            f"`{c['algorithm']}` | `{c.get('lane', '—')}` | "
            f"{c['runtime_seconds']:.2f} |"
        )
    lines.append("")

    lines.append("### Slowest tasks (success, all lanes)\n")
    lines.append("| task_id | n_cells | runtime_total_s | runtime_max_s |")
    lines.append("|---:|---:|---:|---:|")
    for t in summary["slowest_tasks_overall"]:
        lines.append(
            f"| {t['openml_task_id']} | {t['n_cells']} | "
            f"{t['runtime_total_seconds']:.1f} | "
            f"{t['runtime_max_seconds']:.1f} |"
        )
    lines.append("")

    lines.append("## Metric availability (by task type)\n")
    for task_type, body in summary["metric_aggregates_by_task_type"].items():
        lines.append(
            f"### `{task_type}` ({body['n_tasks']} tasks, "
            f"{body['n_cells']} cells)\n"
        )
        lines.append("| metric | mean across cells |")
        lines.append("|---|---:|")
        for k, v in body["metric_means"].items():
            lines.append(f"| `{k}` | {_fmt_metric(v)} |")
        if body["metric_keys_missing"]:
            lines.append(
                f"\n_missing keys_: `{body['metric_keys_missing']}`\n"
            )
        else:
            lines.append("")

    lines.append("## Metric aggregates per lane\n")
    lines.append("| lane | n_cells | accuracy | balanced_accuracy | f1_macro | mcc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in summary["metric_aggregates_per_lane"]:
        means = r["metric_means"]
        lines.append(
            f"| `{r['lane']}` | {r['n_cells']} | "
            f"{_fmt_metric(means.get('accuracy'))} | "
            f"{_fmt_metric(means.get('balanced_accuracy'))} | "
            f"{_fmt_metric(means.get('f1_macro'))} | "
            f"{_fmt_metric(means.get('mcc'))} |"
        )
    lines.append("")

    lines.append("## Metric aggregates per method\n")
    lines.append("| method | n_cells | accuracy | balanced_accuracy | f1_macro |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in summary["metric_aggregates_per_method"]:
        means = r["metric_means"]
        lines.append(
            f"| `{r['method']}` | {r['n_cells']} | "
            f"{_fmt_metric(means.get('accuracy'))} | "
            f"{_fmt_metric(means.get('balanced_accuracy'))} | "
            f"{_fmt_metric(means.get('f1_macro'))} |"
        )
    lines.append("")

    lines.append("## Metric aggregates per algorithm\n")
    lines.append("| algorithm | n_cells | accuracy | balanced_accuracy | f1_macro |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in summary["metric_aggregates_per_algorithm"]:
        means = r["metric_means"]
        lines.append(
            f"| `{r['algorithm']}` | {r['n_cells']} | "
            f"{_fmt_metric(means.get('accuracy'))} | "
            f"{_fmt_metric(means.get('balanced_accuracy'))} | "
            f"{_fmt_metric(means.get('f1_macro'))} |"
        )
    lines.append("")

    lines.append("## Caveats\n")
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")

    lines.append("## What the later signoff commit should do\n")
    for item in summary["what_a_later_signoff_commit_should_do"]:
        lines.append(f"- {item}")
    lines.append("")

    if summary["final_recommendation"] == "ready_for_operator_review":
        lines.append("## Verdict: **READY FOR OPERATOR REVIEW**\n")
        lines.append(
            "Stage 0 replica 1 lane summaries are aggregate-consistent. "
            "A later commit (operator-reviewed) may create "
            "`jobs/doctoral/openml_cc18/stage3_signoff.json` to "
            "unlock the stage-3 top-up machinery. Commit 44 does NOT "
            "do that.\n"
        )
    else:
        lines.append("## Verdict: **BLOCKED — see counts above**\n")
        lines.append(
            "Resolve the items above before attempting signoff. The "
            "operator should not create stage3_signoff.json while "
            "this verdict stands.\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-summary", type=Path, default=DEFAULT_STANDARD,
    )
    parser.add_argument(
        "--heavy-summary", type=Path, default=DEFAULT_HEAVY,
    )
    parser.add_argument(
        "--extreme-summary", type=Path, default=DEFAULT_EXTREME,
    )
    parser.add_argument(
        "--extreme-plan-summary", type=Path, default=DEFAULT_PLAN,
    )
    parser.add_argument("--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute the plan but do not write to disk.",
    )
    args = parser.parse_args(argv)

    try:
        summary = build_signoff_plan(
            standard_path=args.standard_summary,
            heavy_path=args.heavy_summary,
            extreme_path=args.extreme_summary,
            plan_path=args.extreme_plan_summary,
            tasks_csv=args.tasks_csv,
            out_json=args.out_json,
            out_md=args.out_md,
            write_summary=not args.dry_run,
        )
    except SignoffRefusalError as exc:
        print(f"SIGNOFF REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"SIGNOFF PLAN: signoff_status={summary['signoff_status']}  "
        f"recommendation={summary['final_recommendation']}  "
        f"canary_success={summary['n_canary_success_total']}/"
        f"{EXPECTED_N_STANDARD + EXPECTED_N_HEAVY + EXPECTED_N_EXTREME}  "
        f"failed={summary['n_failed_total']}  "
        f"policy_consistent={summary['policy_version_consistent']}"
    )
    if not args.dry_run:
        print(f"json: {args.out_json}")
        print(f"md:   {args.out_md}")
    rc = 0 if summary["final_recommendation"] == "ready_for_operator_review" else 4
    return rc


__all__ = [
    "EXPECTED_N_EXTREME",
    "EXPECTED_N_HEAVY",
    "EXPECTED_N_JOBS_TOTAL",
    "EXPECTED_N_STANDARD",
    "EXPECTED_N_TASKS",
    "RUN_ID",
    "SIGNOFF_FILE",
    "SignoffRefusalError",
    "build_signoff_plan",
    "main",
]


# Silence unused-import warnings used by helpers.
_ = (Counter, statistics)


if __name__ == "__main__":
    sys.exit(main())
