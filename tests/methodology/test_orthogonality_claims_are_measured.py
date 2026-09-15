"""Claims about the objective pair must match the artifact that measured them.

Three statements about `Pearson(quality, cost)` and the criterion-1 Spearman have
been written into this workspace, and TWO WERE WRONG:

  "zero by construction on every dataset"  -- exact only on the FITTING sample
  "|r| <= 0.04 on any subsample"           -- false; ~80% of n=30 draws exceed it
  "its sign is not stable"                 -- too broad; stable WITHIN a dataset,
                                              inconsistent BETWEEN datasets

Each was an over-generalization from a few numbers, written in prose where nothing
could check it. These tests bind the surviving claims to
`audits/latent_conflict_stability.json`, so a fourth version has to agree with a
measurement or fail.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.factor_model import load_reference_factor_model

PAPER = pathlib.Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
AUDIT = PAPER / "audits" / "latent_conflict_stability.json"
SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "doe_xgb" / "campaign"
DATASETS = ("magic", "spambase", "adult", "bank_marketing")


def _audit() -> dict:
    return json.loads(AUDIT.read_text())


@pytest.mark.parametrize("dataset", DATASETS)
def test_orthogonality_is_exact_on_the_fitting_sample(dataset):
    """The identity, recomputed here rather than read from the artifact."""
    pilot = PAPER / "audits" / "pilot_stage_a"
    design = pd.read_csv(pilot / f"{dataset}_design.csv")   # the fitting sample
    F = load_reference_factor_model(dataset).objectives(design)
    r = float(np.corrcoef(F[:, 0], F[:, 1])[0, 1])
    assert abs(r) < 1e-10, f"{dataset}: expected the construction identity, got {r}"
    assert _audit()["datasets"][dataset]["pearson_is_exactly_zero_where_fitted"]


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_identity_is_not_exact_away_from_the_fitting_sample(dataset):
    """The withdrawn claim, kept as a test so it cannot be reasserted."""
    d = _audit()["datasets"][dataset]["abs_pearson_at_n88"]
    assert d["p95"] > 0.04, (
        f"{dataset}: |r| p95 = {d['p95']}, which would make the withdrawn "
        f"'|r| <= 0.04 on any subsample' claim look true. Recheck the artifact.")
    assert d["p95"] < 0.5


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_rank_statistic_is_stable_within_a_dataset(dataset):
    """It is NOT noise; the third withdrawn claim said otherwise."""
    s = _audit()["datasets"][dataset]["latent_spearman_at_n88"]
    spread = s["p95"] - s["p05"]
    assert spread < 0.45, (
        f"{dataset}: criterion 1's statistic spans {spread:.3f} at n = 88, which "
        f"would make it noise rather than a stable per-dataset quantity")


def test_the_rank_statistic_has_no_consistent_direction_across_the_panel():
    """This is the claim that survived, and it is what makes criterion 1 unsound."""
    a = _audit()["datasets"]
    fracs = {ds: a[ds]["latent_spearman_at_n88"]["fraction_positive"] for ds in DATASETS}
    assert min(fracs.values()) < 0.10, f"no dataset is reliably negative: {fracs}"
    assert max(fracs.values()) > 0.90, f"no dataset is reliably positive: {fracs}"
    assert any(0.25 < v < 0.75 for v in fracs.values()), \
        f"no dataset has an indeterminate sign: {fracs}"


def test_criterion_one_passes_on_exactly_one_dataset():
    a = _audit()["datasets"]
    passing = [ds for ds in DATASETS if a[ds]["criterion_1_would_pass"]]
    assert passing == ["magic"], (
        f"criterion 1 now passes on {passing}; the open item in "
        f"PROTOCOL_AMENDMENTS.md is written against exactly one passing dataset")


@pytest.mark.parametrize("path", ["factor_model.py", "runner.py"])
def test_no_module_restates_the_withdrawn_bound(path):
    """The specific false sentence must not come back."""
    src = (SRC / path).read_text()
    assert not re.search(r"\|r\|\s*<=\s*0\.04\s+on\s+any", src), \
        f"{path} restates the withdrawn '|r| <= 0.04 on any subsample' claim"


def test_the_ledger_records_the_withdrawal():
    """A claim that was wrong twice is documented as such, not silently replaced."""
    text = (PAPER / "PROTOCOL_AMENDMENTS.md").read_text()
    assert "That is false" in text
    assert "latent_conflict_stability.json" in text
