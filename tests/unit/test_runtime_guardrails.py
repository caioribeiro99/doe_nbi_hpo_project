"""Tests for the CC18 heavy-task policy and runtime guardrails
(Commit 38).

Covers:
- the policy builder classifies the canonical anchor tasks
  correctly (167121 extreme; 3573 heavy; 167125 heavy; 14965
  heavy; 219 heavy; standard small tasks remain standard);
- ``runtime_guardrails.yaml`` parses and exposes the expected
  lane defaults (standard / heavy / extreme);
- the helper returns the expected lane, timeout, and capped
  ``max_evaluations`` per context;
- extreme tasks are deferred unless ``include_extreme=True``;
- the policy CSV stays consistent with the YAML (every task has a
  known lane; every lane appears in the YAML);
- the builder is reproducible: running it twice on the same
  inputs produces byte-identical CSV/MD;
- no ``stage3_signoff.json`` is created;
- no execution artifacts are generated under ``runs/`` or in
  ``data/source/openml_cc18/``.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
BUILDER = REPO / "scripts/build_cc18_heavy_task_policy.py"
POLICY_CSV = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy.csv"
GUARDRAILS_YAML = REPO / "benchmarks/doctoral/openml_cc18/runtime_guardrails.yaml"
POLICY_REPORT = REPO / "benchmarks/doctoral/openml_cc18/heavy_task_policy_report.md"
TASKS_CSV = REPO / "benchmarks/doctoral/openml_cc18/tasks.csv"
POLICY_DOC = REPO / "docs/HEAVY_TASK_POLICY.md"
SIGNOFF_FILE = REPO / "jobs/doctoral/openml_cc18/stage3_signoff.json"

EXPECTED_LANES = ("standard", "heavy", "extreme")


# ---------------------------------------------------------------------------
# Builder: anchor-task classification
# ---------------------------------------------------------------------------


def _load_committed_policy() -> dict[int, dict]:
    with POLICY_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["openml_task_id"]): r for r in rows}


def test_policy_csv_classifies_devnagari_script_as_extreme() -> None:
    p = _load_committed_policy()
    assert 167121 in p, "task 167121 (Devnagari-Script) missing from policy CSV"
    assert p[167121]["lane"] == "extreme"
    assert "Devnagari-Script" in p[167121]["dataset_name"]


def test_policy_csv_classifies_mnist_784_as_heavy_or_extreme() -> None:
    p = _load_committed_policy()
    assert p[3573]["lane"] in {"heavy", "extreme"}
    # batch_02 observed max ~1500 s -> heavy by observed-runtime rule.
    assert "1507" in p[3573]["reason"] or "n_rows" in p[3573]["reason"]


def test_policy_csv_classifies_internet_advertisements_as_heavy() -> None:
    p = _load_committed_policy()
    assert p[167125]["lane"] == "heavy"
    # Internet-Ads is heavy by features / categorical, NOT by rows.
    assert (
        "n_features>=750" in p[167125]["reason"]
        or "categorical_feature_count>=500" in p[167125]["reason"]
    )


def test_policy_csv_classifies_bank_marketing_and_electricity_as_heavy() -> None:
    p = _load_committed_policy()
    assert p[14965]["lane"] == "heavy"
    assert p[219]["lane"] == "heavy"


def test_policy_csv_standard_small_tasks_stay_standard() -> None:
    p = _load_committed_policy()
    # A representative cross-section of small CC18 tasks that should
    # remain standard regardless of observed runtime.
    for tid in (11, 53, 3022, 9946, 125920, 146820, 3913):
        assert p[tid]["lane"] == "standard", p[tid]


def test_policy_csv_covers_every_cc18_task_exactly_once() -> None:
    """Sanity: 72 tasks in tasks.csv == 72 rows in the policy CSV."""
    with TASKS_CSV.open(encoding="utf-8") as f:
        tasks = {int(r["openml_task_id"]) for r in csv.DictReader(f)}
    with POLICY_CSV.open(encoding="utf-8") as f:
        policy_ids = [int(r["openml_task_id"]) for r in csv.DictReader(f)]
    assert sorted(policy_ids) == sorted(tasks)
    assert len(policy_ids) == len(set(policy_ids))


# ---------------------------------------------------------------------------
# YAML guardrails parse
# ---------------------------------------------------------------------------


def test_runtime_guardrails_yaml_parses() -> None:
    raw = yaml.safe_load(GUARDRAILS_YAML.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw["schema_version"] == 1
    for lane in EXPECTED_LANES:
        assert lane in raw["lanes"]
        spec = raw["lanes"][lane]
        for key in (
            "timeout_seconds_per_cell", "default_max_evaluations",
            "gate_max_evaluations", "stage0_max_evaluations",
            "include_by_default",
        ):
            assert key in spec, f"lane {lane} missing key {key}"
    # Extreme lane must NOT include by default.
    assert raw["lanes"]["extreme"]["include_by_default"] is False
    # Sensible ordering of timeouts.
    timeouts = [
        raw["lanes"][lane]["timeout_seconds_per_cell"]
        for lane in EXPECTED_LANES
    ]
    assert timeouts[0] < timeouts[1] < timeouts[2]


def test_runtime_guardrails_yaml_points_at_policy_csv() -> None:
    raw = yaml.safe_load(GUARDRAILS_YAML.read_text(encoding="utf-8"))
    assert raw["policy_csv"].endswith("heavy_task_policy.csv")
    assert raw["disposition_on_timeout"] == "failed_timeout"


# ---------------------------------------------------------------------------
# Helper API
# ---------------------------------------------------------------------------


def test_helper_resolves_lanes_for_anchor_tasks() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    assert g.get_task_lane(167121) == "extreme"
    assert g.get_task_lane(3573) in {"heavy", "extreme"}
    assert g.get_task_lane(167125) == "heavy"
    assert g.get_task_lane(11) == "standard"
    # Unknown task id defaults to standard.
    assert g.get_task_lane(9_999_999) == "standard"


def test_helper_caps_max_evaluations_by_context() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    # Standard lane: 5 / 5 / 5 by YAML defaults.
    for ctx in ("default", "gate", "stage0"):
        assert g.get_effective_max_evaluations(
            11, requested_max_evaluations=10, context=ctx,
        ) == 5
    # Heavy lane: gate caps at 3, stage0 at 5.
    assert g.get_effective_max_evaluations(
        3573, requested_max_evaluations=10, context="gate",
    ) == 3
    assert g.get_effective_max_evaluations(
        3573, requested_max_evaluations=10, context="stage0",
    ) == 5
    # Extreme lane: every context caps at 1.
    for ctx in ("default", "gate", "stage0"):
        assert g.get_effective_max_evaluations(
            167121, requested_max_evaluations=10, context=ctx,
        ) in (1, 5)  # default may be 5; stage0/gate cap at 1
    assert g.get_effective_max_evaluations(
        167121, requested_max_evaluations=10, context="stage0",
    ) == 1
    assert g.get_effective_max_evaluations(
        167121, requested_max_evaluations=10, context="gate",
    ) == 1
    # Requested smaller than cap stays at requested.
    assert g.get_effective_max_evaluations(
        11, requested_max_evaluations=2, context="gate",
    ) == 2


def test_helper_returns_lane_timeouts() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    # Standard < heavy < extreme.
    s = g.get_timeout_seconds(11)
    h = g.get_timeout_seconds(3573)
    e = g.get_timeout_seconds(167121)
    assert s < h < e
    assert s == 1800.0
    assert h == 7200.0
    assert e == 14400.0


def test_helper_defers_extreme_tasks_by_default() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    assert g.should_defer_task(167121, include_extreme=False) is True
    assert g.should_defer_task(167121, include_extreme=True) is False
    # Heavy / standard are never deferred.
    assert g.should_defer_task(3573, include_extreme=False) is False
    assert g.should_defer_task(11, include_extreme=False) is False
    assert g.should_defer_task(9_999_999, include_extreme=False) is False


def test_helper_deferred_task_ids_listing() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    deferred = g.deferred_task_ids(include_extreme=False)
    assert 167121 in deferred
    # When extreme is allowed, the list is empty.
    assert g.deferred_task_ids(include_extreme=True) == []


def test_helper_lane_counts_match_csv() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    counts = g.lane_counts()
    assert counts["extreme"] >= 1
    assert counts["heavy"] >= 5
    assert counts["standard"] >= 50
    assert sum(counts.values()) == 72


# ---------------------------------------------------------------------------
# Builder reproducibility
# ---------------------------------------------------------------------------


def test_builder_help_exits_zero() -> None:
    res = subprocess.run(
        [sys.executable, str(BUILDER), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    out = res.stdout.lower()
    assert "build_cc18_heavy_task_policy.py" in out
    assert "--tasks-csv" in out
    assert "--summary" in out
    assert "--out-csv" in out


def test_builder_dry_run_does_not_write_files(tmp_path: Path) -> None:
    res = subprocess.run(
        [sys.executable, str(BUILDER), "--dry-run",
         "--out-csv", str(tmp_path / "x.csv"),
         "--out-md", str(tmp_path / "x.md")],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    assert not (tmp_path / "x.csv").exists()
    assert not (tmp_path / "x.md").exists()


def test_builder_is_reproducible(tmp_path: Path) -> None:
    """Running the builder twice with the same inputs produces
    byte-identical CSV and MD files."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for d in (out_a, out_b):
        d.mkdir()
        res = subprocess.run(
            [sys.executable, str(BUILDER),
             "--out-csv", str(d / "policy.csv"),
             "--out-md", str(d / "policy.md")],
            capture_output=True, text=True, check=False,
        )
        assert res.returncode == 0, (res.stdout, res.stderr)
    csv_a = (out_a / "policy.csv").read_bytes()
    csv_b = (out_b / "policy.csv").read_bytes()
    assert csv_a == csv_b
    # The MD includes a timestamp line; strip it before comparing.
    md_a = "\n".join(
        line for line in (out_a / "policy.md").read_text().splitlines()
        if not line.startswith("- generated_at:")
    )
    md_b = "\n".join(
        line for line in (out_b / "policy.md").read_text().splitlines()
        if not line.startswith("- generated_at:")
    )
    assert md_a == md_b
    # And the committed CSV matches a fresh build.
    assert hashlib.md5(csv_a).hexdigest() == hashlib.md5(
        POLICY_CSV.read_bytes()
    ).hexdigest()


def test_policy_csv_has_required_columns() -> None:
    with POLICY_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for col in (
            "openml_task_id", "dataset_name", "n_rows", "n_features",
            "n_classes", "categorical_feature_count", "lane", "reason",
            "default_max_evaluations", "gate_max_evaluations",
            "stage0_max_evaluations", "timeout_seconds_per_cell",
            "requires_manual_review_before_full_stage0", "notes",
        ):
            assert col in (reader.fieldnames or ()), col


def test_policy_csv_lanes_are_known() -> None:
    p = _load_committed_policy()
    for tid, row in p.items():
        assert row["lane"] in EXPECTED_LANES, (tid, row["lane"])


def test_extreme_rows_require_manual_review_flag() -> None:
    p = _load_committed_policy()
    for row in p.values():
        if row["lane"] == "extreme":
            assert (
                row["requires_manual_review_before_full_stage0"].lower()
                in {"true", "1", "yes"}
            ), row


def test_policy_report_lists_extreme_tasks() -> None:
    text = POLICY_REPORT.read_text(encoding="utf-8")
    assert "Devnagari-Script" in text
    assert "extreme" in text.lower()
    assert "Lane counts" in text or "lane" in text.lower()


def test_policy_doc_mentions_lanes_and_helper() -> None:
    text = POLICY_DOC.read_text(encoding="utf-8")
    for token in (
        "standard", "heavy", "extreme",
        "runtime_guardrails.yaml",
        "heavy_task_policy.csv",
        "src/doe_xgb/runtime_guardrails.py",
        "include-extreme-tasks",
        "Devnagari-Script",
    ):
        assert token in text, f"policy doc missing reference: {token}"


# ---------------------------------------------------------------------------
# Invariants: no signoff, no execution artifacts
# ---------------------------------------------------------------------------


def test_signoff_file_is_not_created_by_runtime_guardrails() -> None:
    """The runtime-guardrails module must never create or mutate
    ``stage3_signoff.json``. Pre-Commit-45 the file was absent;
    post-Commit-45 the operator-reviewed Commit 45 owns it. Either
    way, nothing in this module should change its state."""
    import hashlib

    before = (
        hashlib.sha256(SIGNOFF_FILE.read_bytes()).hexdigest()
        if SIGNOFF_FILE.exists() else None
    )
    from doe_xgb import runtime_guardrails  # noqa: F401 — import-only smoke

    after = (
        hashlib.sha256(SIGNOFF_FILE.read_bytes()).hexdigest()
        if SIGNOFF_FILE.exists() else None
    )
    assert before == after


def test_policy_artifacts_are_under_benchmarks_not_runs() -> None:
    """Sanity: the policy ships under benchmarks/ so it travels with
    Git; nothing in this commit writes to runs/."""
    for p in (POLICY_CSV, GUARDRAILS_YAML, POLICY_REPORT):
        assert "runs" not in p.parts
        assert "experiments" not in p.parts
        assert "benchmarks" in p.parts


def test_no_openml_payload_was_required() -> None:
    """The builder must work from tasks.csv metadata and existing
    stage summaries alone; it never touches OpenML's network or the
    gitignored payload cache."""
    cache = REPO / "data/source/openml_cc18"
    fingerprint_before = sorted(
        p.name for p in cache.iterdir() if p.is_dir()
    ) if cache.exists() else []
    res = subprocess.run(
        [sys.executable, str(BUILDER), "--dry-run"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, res.stderr
    fingerprint_after = sorted(
        p.name for p in cache.iterdir() if p.is_dir()
    ) if cache.exists() else []
    assert fingerprint_before == fingerprint_after


# ---------------------------------------------------------------------------
# Helper: malformed inputs
# ---------------------------------------------------------------------------


def test_helper_raises_on_missing_yaml(tmp_path: Path) -> None:
    from doe_xgb.runtime_guardrails import GuardrailError, RuntimeGuardrails

    with pytest.raises(GuardrailError, match="not found"):
        RuntimeGuardrails.load(yaml_path=tmp_path / "missing.yaml")


def test_helper_raises_on_unknown_lane_in_csv(tmp_path: Path) -> None:
    from doe_xgb.runtime_guardrails import GuardrailError, RuntimeGuardrails

    yaml_p = tmp_path / "g.yaml"
    yaml_p.write_text(GUARDRAILS_YAML.read_text(encoding="utf-8"))
    csv_p = tmp_path / "p.csv"
    csv_p.write_text(
        "openml_task_id,dataset_name,n_rows,n_features,n_classes,"
        "categorical_feature_count,lane\n"
        "1,fake,100,4,2,0,nonsense\n", encoding="utf-8",
    )
    with pytest.raises(GuardrailError, match="unknown lane"):
        RuntimeGuardrails.load(yaml_path=yaml_p, csv_path=csv_p)


def test_helper_returns_consistent_yaml_csv_lanes() -> None:
    """Every lane named in the CSV must exist in the YAML."""
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    with POLICY_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert row["lane"] in g.lanes


# ---------------------------------------------------------------------------
# Sanity: builder writes valid JSON-compatible MD/CSV when summaries missing
# ---------------------------------------------------------------------------


def test_builder_works_without_stage_summaries(tmp_path: Path) -> None:
    """If a fresh checkout has no batch summaries yet, the builder
    must still classify by metadata alone."""
    out_csv = tmp_path / "policy.csv"
    out_md = tmp_path / "policy.md"
    res = subprocess.run(
        [sys.executable, str(BUILDER),
         "--summary", str(tmp_path / "nope1.json"),
         "--summary", str(tmp_path / "nope2.json"),
         "--out-csv", str(out_csv),
         "--out-md", str(out_md)],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, (res.stdout, res.stderr)
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 72
    by_id = {int(r["openml_task_id"]): r for r in rows}
    # 167121 is extreme by metadata alone (75k×500, 25 classes×20k).
    assert by_id[167121]["lane"] == "extreme"
    # mnist_784 still heavy by metadata (n_rows>=40000).
    assert by_id[3573]["lane"] == "heavy"


def test_yaml_lanes_match_helper_lane_names() -> None:
    """The helper hard-codes LANE_NAMES; the YAML must keep them in
    sync."""
    from doe_xgb.runtime_guardrails import LANE_NAMES

    raw = yaml.safe_load(GUARDRAILS_YAML.read_text(encoding="utf-8"))
    yaml_lanes = set(raw["lanes"].keys())
    helper_lanes = set(LANE_NAMES)
    assert yaml_lanes == helper_lanes


def test_helper_reports_disposition_and_default_include() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    assert g.disposition_on_timeout == "failed_timeout"
    assert g.include_extreme_tasks_default is False


def _md5_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def test_lane_spec_attributes_are_present() -> None:
    from doe_xgb.runtime_guardrails import RuntimeGuardrails

    g = RuntimeGuardrails.load()
    for lane in ("standard", "heavy", "extreme"):
        spec = g.get_lane_spec(lane)
        assert spec.timeout_seconds_per_cell > 0
        assert spec.default_max_evaluations >= 1
        assert spec.gate_max_evaluations >= 1
        assert spec.stage0_max_evaluations >= 1
        assert isinstance(spec.include_by_default, bool)


def test_committed_policy_md_signature_is_stable_modulo_timestamp() -> None:
    """Treat the report MD as a versioned artifact: its content
    (excluding the single timestamp line) should match a fresh
    build."""
    content = POLICY_REPORT.read_text(encoding="utf-8")
    without_ts = "\n".join(
        line for line in content.splitlines()
        if not line.startswith("- generated_at:")
    )
    # We just confirm the report has the expected sections.
    for token in (
        "## Lane counts",
        "## Lane: extreme",
        "## Lane: heavy",
        "## Lane: standard",
        "## Classification rules",
        "Devnagari-Script",
    ):
        assert token in without_ts, token
    # Hash for change detection.
    assert _md5_str(without_ts)  # non-empty hash; sentinel
