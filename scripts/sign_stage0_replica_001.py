#!/usr/bin/env python
"""Materialize the operator sign-off for stage 0 replica 1.

Reads the aggregate signoff plan published by
``scripts/build_stage0_replica_signoff.py`` (Commit 44), verifies
every gate it advertises, and writes
``jobs/doctoral/openml_cc18/stage3_signoff.json`` with the operator's
explicit metadata. Then calls the aggregator again so the published
plan summary flips ``signoff_status`` from
``"planned_not_signed"`` to ``"signed"`` and
``final_recommendation`` to
``"signed_ready_for_next_stage_planning"``.

The sign-off file is the gate that the runner
(``scripts/cc18_runner.py``) checks before dispatching stage-3
top-up rows. **Creating this file is a capacity decision**, not an
accounting step — Commit 45 is the dedicated operator-reviewed
commit that does it; Commits 40 → 44 explicitly avoided it.

Default operator metadata matches the user-provided values in the
Commit 45 prompt; ``--operator-name`` / ``--operator-handle`` /
``--justification`` override per invocation. The default
justification is the long-form text from the prompt; it explicitly
acknowledges both required caveats:

1. ``isolet`` (task 3481) was observed slow in the standard lane —
   a future-policy-recalibration candidate. No mid-replica lane
   change is applied.
2. ``Devnagari-Script`` (task 167121) ran under policy-defined
   ``extreme.stage0_max_evaluations = 1``; its results are not
   budget-equivalent to standard / heavy lane cells that ran at 5.

Refusal rules
-------------
- Aggregate plan summary missing or unreadable.
- ``signoff_status`` already ``"signed"`` in the aggregate plan.
- ``final_recommendation`` is not ``"ready_for_operator_review"``.
- ``policy_version`` does not match the pinned Commit 40 SHA-256.
- ``policy_version_consistent`` is false, or
  ``source_shards_unchanged_all_lanes`` /
  ``no_pending_running_failed_all_lanes`` /
  ``all_lane_summaries_green`` is false.
- Lane success counts deviate from the canonical 684 / 156 / 24.
- ``stage3_signoff.json`` already exists on disk (unless
  ``--force`` is set — Commit 45 must NOT use ``--force``).

What this script does NOT do
----------------------------
- run any OpenML training;
- mutate committed shards;
- regenerate ``heavy_task_policy.csv`` /
  ``runtime_guardrails.yaml``;
- authorize downstream / stage-3 / top-up execution (the
  signoff JSON sets
  ``downstream_execution_authorized_in_this_commit = false``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Reuse aggregator constants + helpers. We import via the
# ``scripts.`` namespace package (implicit, PEP 420) so the
# ``SignoffRefusalError`` class object here is the same one the
# tests reach via ``from scripts.build_stage0_replica_signoff
# import SignoffRefusalError`` — otherwise we'd end up with two
# distinct classes and ``pytest.raises`` would miss it.
from scripts.build_stage0_replica_signoff import (  # noqa: E402
    DEFAULT_EXTREME,
    DEFAULT_HEAVY,
    DEFAULT_OUT_JSON,
    DEFAULT_OUT_MD,
    DEFAULT_PLAN,
    DEFAULT_STAGE_RUNS_DIR,
    DEFAULT_STANDARD,
    DEFAULT_TASKS_CSV,
    EXPECTED_N_EXTREME,
    EXPECTED_N_HEAVY,
    EXPECTED_N_JOBS_TOTAL,
    EXPECTED_N_STANDARD,
    SIGNOFF_FILE,
    SignoffRefusalError,
    build_signoff_plan,
)

DEFAULT_AGGREGATE_PLAN = DEFAULT_OUT_JSON
DEFAULT_OPERATOR_NAME = "Caio Tertuliano Ribeiro"
DEFAULT_OPERATOR_HANDLE = "caioribeiro99"
DEFAULT_BRANCH = "repo-publication-readiness"
PINNED_POLICY_VERSION = (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
SIGNOFF_TYPE = "stage0_replica_001"
SCHEMA_VERSION = 1

DEFAULT_JUSTIFICATION = (
    "I sign off stage0_replica_001 as a complete and reproducible "
    "stage-0 baseline under policy_version "
    f"{PINNED_POLICY_VERSION}. I acknowledge that isolet/task 3481 "
    "is a future policy-recalibration candidate due to observed "
    "runtime, but no lane change is applied mid-replica. I also "
    "acknowledge that Devnagari-Script/task 167121 was executed "
    "under the policy-defined extreme-lane budget with "
    "stage0_max_evaluations=1, so its results should be interpreted "
    "with that budget caveat. This signoff authorizes future "
    "planning/execution commits to use the stage0_replica_001 "
    "summaries as a gate, but does not itself run any "
    "downstream/top-up execution."
)

DEFAULT_DECLARED_SCOPE = [
    "sign off stage0_replica_001 as a complete, reproducible "
    "stage-0 baseline under the pinned policy_version",
    "authorize future commits to plan downstream/top-up execution "
    "using this signoff as a gate",
    "do not authorize any actual downstream/top-up execution "
    "inside this commit",
    "no cloud execution is authorized by this commit",
]

CAVEATS_ACKNOWLEDGED = [
    {
        "id": "isolet_future_recalibration_candidate",
        "task_id": 3481,
        "dataset": "isolet",
        "summary": (
            "isolet ran in the standard lane but its observed "
            "runtime crosses the 900 s heavy-promotion threshold "
            "(1078.6 s in Commit 40). It is a candidate for "
            "future policy recalibration via "
            "scripts/build_cc18_heavy_task_policy.py but lane "
            "assignments are NOT changed mid-replica; this signoff "
            "covers replica 1 under the Commit 38 policy."
        ),
    },
    {
        "id": "devnagari_extreme_budget_non_equivalence",
        "task_id": 167121,
        "dataset": "Devnagari-Script",
        "summary": (
            "Devnagari-Script ran under the policy-defined "
            "extreme.stage0_max_evaluations=1 (YAML default). "
            "Standard and heavy lanes ran at max_evaluations=5. "
            "random_search / tpe_optuna / default_gbdt cells on "
            "Devnagari therefore exercised fewer configurations than "
            "the rest of the panel; doe_rsm_vrf_true_nbi is "
            "unchanged (it floors at n_doe=max(2*d, "
            "max_evaluations)=8 for d=4). Headline panel-average "
            "metrics that aggregate Devnagari with the other 70 "
            "tasks should footnote this asymmetry."
        ),
    },
]


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


def _safe_rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def _verify_aggregate_plan(
    plan_path: Path,
    *, expected_policy_version: str = PINNED_POLICY_VERSION,
    expected_standard: int = EXPECTED_N_STANDARD,
    expected_heavy: int = EXPECTED_N_HEAVY,
    expected_extreme: int = EXPECTED_N_EXTREME,
    expected_total: int = EXPECTED_N_JOBS_TOTAL,
) -> dict:
    if not plan_path.exists():
        raise SignoffRefusalError(
            f"aggregate plan summary not found at {plan_path}; "
            "run scripts/build_stage0_replica_signoff.py first."
        )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    status = payload.get("signoff_status")
    if status != "planned_not_signed":
        raise SignoffRefusalError(
            f"aggregate plan has signoff_status={status!r}; expected "
            "'planned_not_signed'. The replica appears to already be "
            "signed; refusing to overwrite."
        )
    rec = payload.get("final_recommendation")
    if rec != "ready_for_operator_review":
        raise SignoffRefusalError(
            f"aggregate plan has final_recommendation={rec!r}; "
            "expected 'ready_for_operator_review'."
        )
    if not payload.get("policy_version_consistent", False):
        raise SignoffRefusalError(
            "aggregate plan reports policy_version_consistent=False."
        )
    pv = payload.get("policy_version")
    if pv != expected_policy_version:
        raise SignoffRefusalError(
            f"aggregate plan policy_version={pv} != pinned "
            f"{expected_policy_version}; refusing to sign off under "
            "a drifted policy."
        )
    if not payload.get("all_lane_summaries_green", False):
        raise SignoffRefusalError(
            "aggregate plan reports all_lane_summaries_green=False."
        )
    if not payload.get("source_shards_unchanged_all_lanes", False):
        raise SignoffRefusalError(
            "aggregate plan reports "
            "source_shards_unchanged_all_lanes=False."
        )
    if not payload.get("no_pending_running_failed_all_lanes", False):
        raise SignoffRefusalError(
            "aggregate plan reports "
            "no_pending_running_failed_all_lanes=False."
        )
    if int(payload.get("n_jobs_total_expected", 0)) != expected_total:
        raise SignoffRefusalError(
            f"aggregate plan n_jobs_total_expected="
            f"{payload.get('n_jobs_total_expected')} != "
            f"{expected_total}."
        )
    if int(payload.get("n_standard_success", 0)) != expected_standard:
        raise SignoffRefusalError(
            f"aggregate plan n_standard_success="
            f"{payload.get('n_standard_success')} != "
            f"{expected_standard}."
        )
    if int(payload.get("n_heavy_success", 0)) != expected_heavy:
        raise SignoffRefusalError(
            f"aggregate plan n_heavy_success="
            f"{payload.get('n_heavy_success')} != "
            f"{expected_heavy}."
        )
    if int(payload.get("n_extreme_success", 0)) != expected_extreme:
        raise SignoffRefusalError(
            f"aggregate plan n_extreme_success="
            f"{payload.get('n_extreme_success')} != "
            f"{expected_extreme}."
        )
    for k in ("n_failed_total", "n_failed_timeout_total",
              "n_pending_total", "n_running_total"):
        if int(payload.get(k, 0)) != 0:
            raise SignoffRefusalError(
                f"aggregate plan {k}={payload.get(k)} != 0."
            )
    return payload


def build_signoff_record(
    plan_payload: dict, *,
    aggregate_plan_path: Path,
    operator_name: str = DEFAULT_OPERATOR_NAME,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    branch: str = DEFAULT_BRANCH,
    justification: str = DEFAULT_JUSTIFICATION,
    declared_scope: list[str] | None = None,
    git_sha_at_signoff: str | None = None,
    signed_at_utc: str | None = None,
    standard_path: Path = DEFAULT_STANDARD,
    heavy_path: Path = DEFAULT_HEAVY,
    extreme_path: Path = DEFAULT_EXTREME,
    plan_path: Path = DEFAULT_PLAN,
    notes: str = "",
) -> dict:
    """Build the dict that will be written to ``stage3_signoff.json``.

    The hashes recorded here freeze the artifacts the operator
    reviewed; the aggregator cross-checks them on every subsequent
    invocation so a post-signoff modification to a lane summary is
    detected automatically.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "signoff_type": SIGNOFF_TYPE,
        "signoff_status": "signed",
        "signed_at_utc": signed_at_utc or datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ),
        "operator_name": operator_name,
        "operator_handle": operator_handle,
        "branch": branch,
        "git_sha_at_signoff": git_sha_at_signoff or _git_sha(),
        "policy_version": plan_payload["policy_version"],
        "declared_scope": list(declared_scope or DEFAULT_DECLARED_SCOPE),
        "justification": justification,
        "caveats_acknowledged": CAVEATS_ACKNOWLEDGED,
        "aggregate_plan_summary_path": _safe_rel(aggregate_plan_path),
        "aggregate_plan_summary_sha256": _sha256(aggregate_plan_path),
        "signoff_plan_summary_path": _safe_rel(aggregate_plan_path),
        "signoff_plan_summary_sha256": _sha256(aggregate_plan_path),
        "standard_lane_summary_path": _safe_rel(standard_path),
        "standard_lane_summary_sha256": _sha256(standard_path),
        "heavy_lane_summary_path": _safe_rel(heavy_path),
        "heavy_lane_summary_sha256": _sha256(heavy_path),
        "extreme_lane_summary_path": _safe_rel(extreme_path),
        "extreme_lane_summary_sha256": _sha256(extreme_path),
        "extreme_lane_plan_summary_path": (
            _safe_rel(plan_path) if plan_path.exists() else None
        ),
        "extreme_lane_plan_summary_sha256": (
            _sha256(plan_path) if plan_path.exists() else None
        ),
        "n_jobs_total_expected": int(plan_payload["n_jobs_total_expected"]),
        "n_canary_success_total": int(plan_payload["n_canary_success_total"]),
        "lane_success_counts": {
            "standard": int(plan_payload["n_standard_success"]),
            "heavy": int(plan_payload["n_heavy_success"]),
            "extreme": int(plan_payload["n_extreme_success"]),
        },
        "failure_counts": {
            "failed_total": int(plan_payload["n_failed_total"]),
            "failed_timeout_total": int(plan_payload["n_failed_timeout_total"]),
            "pending_total": int(plan_payload["n_pending_total"]),
            "running_total": int(plan_payload["n_running_total"]),
        },
        "source_shards_unchanged_all_lanes": bool(
            plan_payload["source_shards_unchanged_all_lanes"],
        ),
        "no_pending_running_failed_all_lanes": bool(
            plan_payload["no_pending_running_failed_all_lanes"],
        ),
        "downstream_execution_authorized_in_this_commit": False,
        "notes": (
            notes or (
                "Commit 45 explicit operator sign-off. No OpenML "
                "training was run by this commit; the standard, "
                "heavy, and extreme lane summaries referenced here "
                "were produced by Commits 40, 41, and 43 "
                "respectively. The aggregator (Commit 44) is the "
                "machine-readable review surface that preceded "
                "this signoff. The runner refuses stage-3 top-up "
                "rows until this file exists; creating it unlocks "
                "*planning* of the next tier — actual execution is "
                "a separate operator-reviewed commit."
            )
        ),
    }


def sign_stage0_replica_001(
    *,
    aggregate_plan_path: Path = DEFAULT_AGGREGATE_PLAN,
    standard_path: Path = DEFAULT_STANDARD,
    heavy_path: Path = DEFAULT_HEAVY,
    extreme_path: Path = DEFAULT_EXTREME,
    plan_path: Path = DEFAULT_PLAN,
    out_signoff: Path = SIGNOFF_FILE,
    operator_name: str = DEFAULT_OPERATOR_NAME,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    branch: str = DEFAULT_BRANCH,
    justification: str = DEFAULT_JUSTIFICATION,
    declared_scope: list[str] | None = None,
    notes: str = "",
    force: bool = False,
    refresh_aggregate: bool = True,
    tasks_csv: Path = DEFAULT_TASKS_CSV,
    aggregate_out_json: Path = DEFAULT_OUT_JSON,
    aggregate_out_md: Path = DEFAULT_OUT_MD,
) -> tuple[Path, dict, dict | None]:
    """Materialize ``stage3_signoff.json`` and (optionally) refresh
    the aggregate plan summary so it reflects the new ``signed``
    state.

    Returns ``(signoff_path, signoff_record, refreshed_summary | None)``.
    """
    if out_signoff.exists() and not force:
        raise SignoffRefusalError(
            f"{out_signoff} already exists; pass --force to overwrite. "
            "Commit 45 must NOT use --force."
        )
    plan_payload = _verify_aggregate_plan(aggregate_plan_path)

    record = build_signoff_record(
        plan_payload,
        aggregate_plan_path=aggregate_plan_path,
        operator_name=operator_name,
        operator_handle=operator_handle,
        branch=branch,
        justification=justification,
        declared_scope=declared_scope,
        standard_path=standard_path,
        heavy_path=heavy_path,
        extreme_path=extreme_path,
        plan_path=plan_path,
        notes=notes,
    )

    out_signoff.parent.mkdir(parents=True, exist_ok=True)
    out_signoff.write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8",
    )

    refreshed: dict | None = None
    if refresh_aggregate:
        refreshed = build_signoff_plan(
            standard_path=standard_path,
            heavy_path=heavy_path,
            extreme_path=extreme_path,
            plan_path=plan_path,
            tasks_csv=tasks_csv,
            out_json=aggregate_out_json,
            out_md=aggregate_out_md,
            write_summary=True,
            signoff_file=out_signoff,
        )

    return out_signoff, record, refreshed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate-plan", type=Path, default=DEFAULT_AGGREGATE_PLAN,
    )
    parser.add_argument(
        "--standard-summary", type=Path, default=DEFAULT_STANDARD,
    )
    parser.add_argument("--heavy-summary", type=Path, default=DEFAULT_HEAVY)
    parser.add_argument("--extreme-summary", type=Path, default=DEFAULT_EXTREME)
    parser.add_argument(
        "--extreme-plan-summary", type=Path, default=DEFAULT_PLAN,
    )
    parser.add_argument(
        "--out-signoff", type=Path, default=SIGNOFF_FILE,
        help="Path to the stage-3 sign-off JSON file. Defaults to "
             "the committed location the cc18_runner already checks.",
    )
    parser.add_argument(
        "--operator-name", default=DEFAULT_OPERATOR_NAME,
        help="Human-readable operator name. Default matches the "
             "Commit 45 prompt.",
    )
    parser.add_argument(
        "--operator-handle", default=DEFAULT_OPERATOR_HANDLE,
        help="Operator handle. Default matches the Commit 45 prompt.",
    )
    parser.add_argument(
        "--branch", default=DEFAULT_BRANCH,
        help="Branch the signoff was made on.",
    )
    parser.add_argument(
        "--justification", default=DEFAULT_JUSTIFICATION,
        help="Free-form justification. The default acknowledges both "
             "required caveats; if you override it, you must still "
             "explicitly mention both caveats.",
    )
    parser.add_argument(
        "--notes", default="",
        help="Optional free-form notes.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing stage3_signoff.json. Commit 45 "
             "must NOT pass this.",
    )
    parser.add_argument(
        "--no-refresh-aggregate", action="store_true",
        help="Do NOT re-run the aggregate signoff plan builder after "
             "writing the signoff file. Default behavior re-runs it.",
    )
    parser.add_argument(
        "--tasks-csv", type=Path, default=DEFAULT_TASKS_CSV,
    )
    parser.add_argument(
        "--aggregate-out-json", type=Path, default=DEFAULT_OUT_JSON,
    )
    parser.add_argument(
        "--aggregate-out-md", type=Path, default=DEFAULT_OUT_MD,
    )
    args = parser.parse_args(argv)

    try:
        signoff_path, record, refreshed = sign_stage0_replica_001(
            aggregate_plan_path=args.aggregate_plan,
            standard_path=args.standard_summary,
            heavy_path=args.heavy_summary,
            extreme_path=args.extreme_summary,
            plan_path=args.extreme_plan_summary,
            out_signoff=args.out_signoff,
            operator_name=args.operator_name,
            operator_handle=args.operator_handle,
            branch=args.branch,
            justification=args.justification,
            notes=args.notes,
            force=args.force,
            refresh_aggregate=not args.no_refresh_aggregate,
            tasks_csv=args.tasks_csv,
            aggregate_out_json=args.aggregate_out_json,
            aggregate_out_md=args.aggregate_out_md,
        )
    except SignoffRefusalError as exc:
        print(f"SIGNOFF REFUSAL: {exc}", file=sys.stderr)
        return 3

    print(
        f"SIGNED stage0_replica_001  "
        f"operator={record['operator_name']!r} "
        f"handle={record['operator_handle']!r}  "
        f"policy_version={record['policy_version'][:16]}  "
        f"canary_success={record['n_canary_success_total']}  "
        f"downstream_execution_authorized_in_this_commit="
        f"{record['downstream_execution_authorized_in_this_commit']}"
    )
    print(f"signoff: {_safe_rel(signoff_path)}")
    if refreshed is not None:
        print(
            f"aggregate refreshed: signoff_status="
            f"{refreshed['signoff_status']}  "
            f"recommendation={refreshed['final_recommendation']}"
        )
        print(f"aggregate json: {_safe_rel(args.aggregate_out_json)}")
        print(f"aggregate md:   {_safe_rel(args.aggregate_out_md)}")
    return 0


__all__ = [
    "CAVEATS_ACKNOWLEDGED",
    "DEFAULT_AGGREGATE_PLAN",
    "DEFAULT_BRANCH",
    "DEFAULT_DECLARED_SCOPE",
    "DEFAULT_JUSTIFICATION",
    "DEFAULT_OPERATOR_HANDLE",
    "DEFAULT_OPERATOR_NAME",
    "PINNED_POLICY_VERSION",
    "SCHEMA_VERSION",
    "SIGNOFF_TYPE",
    "build_signoff_record",
    "main",
    "sign_stage0_replica_001",
]


# Silence noqa for unused imports kept for future commits.
_ = DEFAULT_STAGE_RUNS_DIR


if __name__ == "__main__":
    sys.exit(main())
