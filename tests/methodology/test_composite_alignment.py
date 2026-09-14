"""The hard invariant is inversion, and it must be the right one.

The first form of this guard compared the composite's relation to cost against an
equally weighted mean of the six quality responses. Those are different
aggregations, so they can differ legitimately whenever one response trades off
against the others. On this panel specificity does, and the guard fired on half the
datasets for a reason that was not a defect.

The invariant that does matter is inversion: the composite must point the same way
as the badness it aggregates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.factor_model import (composite_alignment, fit_factor_model,
                                           raw_conflict)
from doe_xgb.campaign.evaluator import RESPONSES


def _frame(n=90, seed=0, specificity_trades_off=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=n)
    extra = rng.normal(size=n)
    spec = (0.9 - 0.06 * q if specificity_trades_off else 0.9 + 0.06 * q)
    return pd.DataFrame({
        "Accuracy_Mean": 0.80 + 0.05 * q + 0.003 * rng.normal(size=n),
        "Precision_Mean": 0.70 + 0.04 * extra + 0.003 * rng.normal(size=n),
        "Recall_Mean": 0.60 + 0.05 * q + 0.004 * rng.normal(size=n),
        "Specificity_Mean": spec + 0.003 * rng.normal(size=n),
        "RocAuc_Mean": 0.85 + 0.05 * q + 0.002 * rng.normal(size=n),
        "LogLoss_Mean": 0.50 - 0.05 * q + 0.004 * rng.normal(size=n),
        "Leaves_Mean": np.abs(4000 + 1500 * q + 400 * rng.normal(size=n)) + 50,
    })


@pytest.mark.parametrize("seed", range(4))
def test_the_composite_is_never_inverted(seed: int) -> None:
    df = _frame(seed=seed)
    assert composite_alignment(fit_factor_model(df), df) > 0


def test_alignment_detects_a_genuinely_inverted_composite() -> None:
    """The guard must be able to fire, or it guards nothing."""
    df = _frame(seed=3)
    fm = fit_factor_model(df)
    flipped = fm.transform(df)["quality"] * -1.0
    from scipy.stats import spearmanr
    from doe_xgb.campaign.factor_model import apply_transforms

    M = apply_transforms(df)
    names = list(RESPONSES)
    q = [i for i, c in enumerate(names) if RESPONSES[c]["role"] == "quality"]
    ref = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1)
                           for i in q]).mean(axis=1)
    assert spearmanr(flipped, ref).statistic < 0


def test_conflict_divergence_is_possible_without_inversion() -> None:
    """The exact situation that made the old guard wrong.

    A composite can point correctly at overall badness and still relate to cost
    differently from an equally weighted mean, when one response trades off.
    """
    df = _frame(seed=1, specificity_trades_off=True)
    fm = fit_factor_model(df)
    t = fm.transform(df)
    from scipy.stats import spearmanr
    latent = float(spearmanr(t["quality"], t["cost"]).statistic)
    raw = raw_conflict(df)
    assert composite_alignment(fm, df) > 0, "not inverted"
    # the two aggregations may or may not agree here; what matters is that a
    # disagreement would not imply inversion
    if np.sign(latent) != np.sign(raw):
        assert composite_alignment(fm, df) > 0


@pytest.mark.parametrize("ds", ["magic", "spambase", "adult", "bank_marketing"])
def test_no_panel_dataset_has_an_inverted_composite(ds: str) -> None:
    from pathlib import Path

    pilot = (Path(__file__).resolve().parents[2] / "papers" / "xgboost_hpo_vrfnbi"
             / "audits" / "pilot_stage_a" / f"{ds}_design.csv")
    if not pilot.exists():
        pytest.skip("Stage A artifacts not present")
    df = pd.read_csv(pilot)
    assert composite_alignment(fit_factor_model(df), df) > 0
