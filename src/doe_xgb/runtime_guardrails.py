"""Runtime guardrails for the OpenML-CC18 doctoral benchmark.

Commit 38 ships a small policy layer that batch / stage runners
consult before dispatching a cell. The policy is split between two
files committed under ``benchmarks/doctoral/openml_cc18/``:

- ``runtime_guardrails.yaml`` — lane defaults (timeouts,
  max_evaluations by lane, include-by-default flag);
- ``heavy_task_policy.csv`` — one row per CC18 task, recording the
  assigned lane and per-task overrides where any column is set.

This helper is the in-process API:

>>> from doe_xgb.runtime_guardrails import RuntimeGuardrails
>>> g = RuntimeGuardrails.load()                # uses default paths
>>> g.get_task_lane(167121)
'extreme'
>>> g.get_timeout_seconds(167121)
14400.0
>>> g.get_effective_max_evaluations(167121, requested=5)
1
>>> g.should_defer_task(167121, include_extreme=False)
True

The module is dependency-light: it does NOT import the runner, the
adapter registry, or any of the heavy GBDT / OpenML packages. It is
safe to import on a worker that is still booting up its execution
environment, so the runner can short-circuit dispatch before pulling
adapter dependencies.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_YAML = REPO / "benchmarks" / "doctoral" / "openml_cc18" / "runtime_guardrails.yaml"
DEFAULT_CSV = REPO / "benchmarks" / "doctoral" / "openml_cc18" / "heavy_task_policy.csv"

LaneName = Literal["standard", "heavy", "extreme"]
LANE_NAMES: tuple[LaneName, ...] = ("standard", "heavy", "extreme")


class GuardrailError(RuntimeError):
    """Raised when the policy files are malformed or inconsistent."""


@dataclass(frozen=True)
class LaneSpec:
    name: str
    timeout_seconds_per_cell: float
    default_max_evaluations: int
    gate_max_evaluations: int
    stage0_max_evaluations: int
    include_by_default: bool
    requires_manual_review_before_full_stage0: bool
    notes: str = ""


@dataclass(frozen=True)
class TaskPolicy:
    openml_task_id: int
    dataset_name: str
    n_rows: int
    n_features: int
    n_classes: int
    categorical_feature_count: int
    lane: str
    reason: str
    default_max_evaluations: int | None
    gate_max_evaluations: int | None
    stage0_max_evaluations: int | None
    timeout_seconds_per_cell: float | None
    requires_manual_review_before_full_stage0: bool
    notes: str = ""


@dataclass(frozen=True)
class RuntimeGuardrails:
    """Top-level policy object. Use ``RuntimeGuardrails.load()`` to
    construct one from the default file locations."""

    lanes: dict[str, LaneSpec]
    tasks: dict[int, TaskPolicy]
    disposition_on_timeout: str
    include_extreme_tasks_default: bool
    yaml_path: Path
    csv_path: Path
    schema_version: int = 1
    raw_yaml: dict = field(default_factory=dict)

    # ----------------------------------------------------------- loaders

    @classmethod
    def load(
        cls,
        yaml_path: Path | None = None,
        csv_path: Path | None = None,
    ) -> RuntimeGuardrails:
        yaml_p = Path(yaml_path) if yaml_path else DEFAULT_YAML
        csv_p = Path(csv_path) if csv_path else DEFAULT_CSV
        if not yaml_p.exists():
            raise GuardrailError(
                f"runtime guardrails YAML not found at {yaml_p}; "
                "regenerate or supply --runtime-guardrails."
            )
        raw = yaml.safe_load(yaml_p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise GuardrailError(
                f"{yaml_p}: expected mapping at top level"
            )
        version = int(raw.get("schema_version", 1))
        lanes_raw = raw.get("lanes") or {}
        if not isinstance(lanes_raw, dict):
            raise GuardrailError(f"{yaml_p}: 'lanes' must be a mapping")
        lanes: dict[str, LaneSpec] = {}
        for name in LANE_NAMES:
            if name not in lanes_raw:
                raise GuardrailError(
                    f"{yaml_p}: lane '{name}' missing from 'lanes' mapping"
                )
            spec = lanes_raw[name]
            lanes[name] = LaneSpec(
                name=name,
                timeout_seconds_per_cell=float(spec["timeout_seconds_per_cell"]),
                default_max_evaluations=int(spec["default_max_evaluations"]),
                gate_max_evaluations=int(spec["gate_max_evaluations"]),
                stage0_max_evaluations=int(spec["stage0_max_evaluations"]),
                include_by_default=bool(spec.get("include_by_default", True)),
                requires_manual_review_before_full_stage0=bool(
                    spec.get("requires_manual_review_before_full_stage0", False),
                ),
                notes=str(spec.get("notes", "")).strip(),
            )

        tasks: dict[int, TaskPolicy] = {}
        if csv_p.exists():
            with csv_p.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                required = {
                    "openml_task_id", "dataset_name", "n_rows", "n_features",
                    "n_classes", "categorical_feature_count", "lane",
                }
                missing = required - set(reader.fieldnames or ())
                if missing:
                    raise GuardrailError(
                        f"{csv_p}: missing required columns: {sorted(missing)}"
                    )
                for row in reader:
                    tid = int(row["openml_task_id"])
                    lane = (row.get("lane") or "standard").strip().lower()
                    if lane not in lanes:
                        raise GuardrailError(
                            f"{csv_p}: task {tid} has unknown lane '{lane}'"
                        )
                    tasks[tid] = TaskPolicy(
                        openml_task_id=tid,
                        dataset_name=row.get("dataset_name", ""),
                        n_rows=int(row.get("n_rows") or 0),
                        n_features=int(row.get("n_features") or 0),
                        n_classes=int(row.get("n_classes") or 0),
                        categorical_feature_count=int(
                            row.get("categorical_feature_count") or 0,
                        ),
                        lane=lane,
                        reason=(row.get("reason") or "").strip(),
                        default_max_evaluations=_opt_int(
                            row.get("default_max_evaluations"),
                        ),
                        gate_max_evaluations=_opt_int(
                            row.get("gate_max_evaluations"),
                        ),
                        stage0_max_evaluations=_opt_int(
                            row.get("stage0_max_evaluations"),
                        ),
                        timeout_seconds_per_cell=_opt_float(
                            row.get("timeout_seconds_per_cell"),
                        ),
                        requires_manual_review_before_full_stage0=_opt_bool(
                            row.get("requires_manual_review_before_full_stage0"),
                        ),
                        notes=(row.get("notes") or "").strip(),
                    )

        return cls(
            lanes=lanes,
            tasks=tasks,
            disposition_on_timeout=str(
                raw.get("disposition_on_timeout", "failed_timeout"),
            ),
            include_extreme_tasks_default=bool(
                raw.get("include_extreme_tasks_default", False),
            ),
            yaml_path=yaml_p,
            csv_path=csv_p,
            schema_version=version,
            raw_yaml=raw,
        )

    # ----------------------------------------------------------- queries

    def get_task_lane(self, task_id: int) -> str:
        """Return the assigned lane, or 'standard' if the task is not
        listed in heavy_task_policy.csv (safe default)."""
        policy = self.tasks.get(int(task_id))
        if policy is None:
            return "standard"
        return policy.lane

    def get_lane_spec(self, lane: str) -> LaneSpec:
        if lane not in self.lanes:
            raise GuardrailError(f"unknown lane: {lane}")
        return self.lanes[lane]

    def get_timeout_seconds(
        self, task_id: int, method: str | None = None,
        algorithm: str | None = None,
    ) -> float:
        """Return the per-cell timeout. The (method, algorithm) tuple
        is accepted for symmetry with future overrides but is currently
        unused by the default policy."""
        del method, algorithm
        policy = self.tasks.get(int(task_id))
        if policy is not None and policy.timeout_seconds_per_cell is not None:
            return float(policy.timeout_seconds_per_cell)
        lane = self.get_task_lane(task_id)
        return float(self.lanes[lane].timeout_seconds_per_cell)

    def get_effective_max_evaluations(
        self, task_id: int, requested_max_evaluations: int,
        *, context: Literal["default", "gate", "stage0"] = "gate",
    ) -> int:
        """Return the effective max_evaluations cap for ``task_id``.

        ``requested_max_evaluations`` is the caller's wanted budget;
        the helper caps it at the lane's per-context limit (or the
        per-task override if set). Negative or zero requested values
        are clamped to 1.
        """
        wanted = max(1, int(requested_max_evaluations))
        policy = self.tasks.get(int(task_id))
        lane_name = (policy.lane if policy else "standard")
        lane = self.lanes[lane_name]

        # Per-task override beats lane default when set.
        override = None
        if policy is not None:
            if context == "default" and policy.default_max_evaluations is not None:
                override = policy.default_max_evaluations
            elif context == "gate" and policy.gate_max_evaluations is not None:
                override = policy.gate_max_evaluations
            elif context == "stage0" and policy.stage0_max_evaluations is not None:
                override = policy.stage0_max_evaluations

        if override is not None:
            cap = int(override)
        elif context == "default":
            cap = lane.default_max_evaluations
        elif context == "gate":
            cap = lane.gate_max_evaluations
        elif context == "stage0":
            cap = lane.stage0_max_evaluations
        else:
            raise GuardrailError(f"unknown context: {context}")
        return min(wanted, max(1, int(cap)))

    def should_defer_task(
        self, task_id: int, *, include_extreme: bool = False,
    ) -> bool:
        """Return True when ``task_id`` should be skipped (deferred) in
        the current batch / stage run.

        A task is deferred when its lane is extreme and the caller has
        NOT opted into the extreme lane via
        ``include_extreme=True``. Standard / heavy lanes never defer."""
        lane_name = self.get_task_lane(task_id)
        if lane_name != "extreme":
            return False
        return not bool(include_extreme)

    def lane_counts(self) -> dict[str, int]:
        """Counts of tasks per lane (handy for summaries)."""
        out = dict.fromkeys(LANE_NAMES, 0)
        for p in self.tasks.values():
            if p.lane in out:
                out[p.lane] += 1
        return out

    def deferred_task_ids(self, *, include_extreme: bool = False) -> list[int]:
        """All task_ids that would be skipped under the current policy."""
        if include_extreme:
            return []
        return sorted(
            tid for tid, p in self.tasks.items() if p.lane == "extreme"
        )


# ---------------------------------------------------------------------------
# Small CSV column helpers
# ---------------------------------------------------------------------------


def _opt_int(value: str | None) -> int | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    return int(v)


def _opt_float(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    return float(v)


def _opt_bool(value: str | None) -> bool:
    if value is None:
        return False
    v = value.strip().lower()
    return v in {"1", "true", "yes", "y", "t"}


__all__ = [
    "DEFAULT_CSV",
    "DEFAULT_YAML",
    "GuardrailError",
    "LaneSpec",
    "RuntimeGuardrails",
    "TaskPolicy",
]
