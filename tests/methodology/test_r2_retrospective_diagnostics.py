"""The coauthor-review (R2) diagnostics must stay what they claim to be.

Two retrospective computations were added in R2: the 88-versus-79 centre-point
sensitivity and the grid stopped at 186 evaluations. Their classification criteria were
committed before the first run, so the stored criteria must equal that commit's, and the
stored class must follow from the stored per-unit flags by the declared rule. The prose
that reports them, and the post hoc status of the signed-rank tests, are checked too.
None of these tests needs the uncommitted per-unit tree.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
A = PAPER / "analysis"
MAN = PAPER / "manuscript"
PRE_RUN_COMMIT = "933422a"     # the commit that fixed the criteria before the first run


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PAPER / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_centre_point_criteria_are_the_ones_committed_before_the_first_run() -> None:
    stored = json.loads((A / "center_point_sensitivity.json").read_text())["criteria"]
    assert stored == _load_script("center_point_sensitivity").CRITERIA
    old = subprocess.run(["git", "show", f"{PRE_RUN_COMMIT}:papers/xgboost_hpo_vrfnbi/scripts/"
                          "center_point_sensitivity.py"], cwd=PAPER, capture_output=True, text=True)
    assert old.returncode == 0, old.stderr
    block = re.search(r"^CRITERIA = \{.*?^\}\n", old.stdout, re.S | re.M).group(0)
    namespace: dict = {}
    exec(block, namespace)                      # a literal dict, nothing else
    assert namespace["CRITERIA"] == stored, "the criteria moved after the first run"


def test_centre_point_class_follows_from_the_stored_flags() -> None:
    mod = _load_script("center_point_sensitivity")
    res = json.loads((A / "center_point_sensitivity.json").read_text())
    rank = {"low": 0, "moderate": 1, "high": 2}
    worst = "low"
    for ds, d in res["datasets"].items():
        assert d["reproduced_88_row_stage"] is True
        assert len(d["units"]) == 30
        again = mod.classify(d["units"], d["factor_model"])
        assert again == d["counts"], ds
        worst = max(worst, again["dataset_class"], key=rank.get)
        p = d["premise"]
        assert p["centre_rows_per_unit"] == [10] and p["max_centre_response_spread"] == 0.0
        assert p["cache_distinct_keys"] == [79] and p["cache_served"] == [9]
    assert res["overall_class"] == worst


def test_supplement_reports_the_stored_centre_point_class() -> None:
    sup = (MAN / "SUPPLEMENT.md").read_text()
    res = json.loads((A / "center_point_sensitivity.json").read_text())
    assert f"classified **{res['overall_class'].upper()}**" in sup
    assert "## S18. Centre-point sensitivity" in sup


def test_grid_prefix_reproduces_table_5_at_386() -> None:
    gp = json.loads((A / "grid_budget_prefix.json").read_text())["datasets"]
    bva = json.loads((A / "baseline_vs_arms.json").read_text())
    for ds, d in gp.items():
        full = d["by_budget"]["386"]
        assert abs(full["median_diff_nbi_s_minus_grid"] - bva[ds]["median_diff_nbi_s_minus_grid"]) < 1e-12
        assert full["nbi_s_wins"] == bva[ds]["nbi_s_wins"]
        assert d["prefix_is_frozen_186_point_grid"] is True


def test_abstract_is_within_the_250_word_rule() -> None:
    text = (MAN / "sections" / "00_title_abstract.md").read_text()
    body = text.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0]
    assert len(body.split()) <= 250, len(body.split())


def test_signed_rank_significance_is_labelled_post_hoc_where_it_is_claimed() -> None:
    abstract = (MAN / "sections" / "00_title_abstract.md").read_text()
    assert "Holm-significant post hoc signed-rank tests" in abstract
    assert "pre-declared Nadeau–Bengio" in abstract
    conclusion = (MAN / "sections" / "10_conclusion.md").read_text()
    assert re.search(r"signed-rank test, a post hoc choice", conclusion)
    for f in ("01_introduction.md", "06b_results_contrasts.md", "07_discussion.md"):
        s = (MAN / "sections" / f).read_text()
        for m in re.finditer(r"Holm-significant", s):
            window = s[max(0, m.start() - 400): m.end() + 400]
            assert "post hoc" in window or "Table 5" in window or "AUGMENTED" in window, \
                f"{f}: Holm significance without its post hoc status nearby: {window[300:500]!r}"
