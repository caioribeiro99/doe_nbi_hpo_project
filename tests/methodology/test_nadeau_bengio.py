"""The corrected resampled t-test must be the published one, at the protocol's split.

Nadeau and Bengio (2003) give, for R repeated random splits,

    SE_corr^2 = (1/R + n_test/n_train) * s^2

Two things went wrong here in succession and both are pinned by these tests. The
first version inflated sd/sqrt(R) by sqrt(1 + rho/(1-rho)), which divides the
correction term by R. The second used rho = 0.25 as a free equicorrelation
parameter; that equals n_test/n_train only for a 75/25 split, while the protocol's
outer split is 80/20 and gives 0.25 for the RATIO and 0.20 for rho.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = (REPO / "papers" / "xgboost_hpo_vrfnbi" / "scripts"
          / "stage_b_statistical_sensitivity.py")
spec = importlib.util.spec_from_file_location("sens", SCRIPT)
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)


def test_the_protocol_ratio_is_the_eighty_twenty_split() -> None:
    assert sens.PROTOCOL_TEST_TRAIN_RATIO == pytest.approx(0.25)
    assert sens.R == 30


def test_the_inflation_factor_is_sqrt_eight_point_five() -> None:
    """Closed form: sqrt((1/30 + 0.25) / (1/30)) = sqrt(8.5) = 2.9155."""
    out = sens.resolution(1.0)
    key = f"se_inflation_test_train_{sens.PROTOCOL_TEST_TRAIN_RATIO}"
    # stored rounded to four decimals, so compare at that resolution
    assert out[key] == pytest.approx(float(np.sqrt(8.5)), abs=5e-5)
    assert out[key] == pytest.approx(2.9155, abs=1e-4)


def test_the_superseded_factor_is_not_reproduced() -> None:
    """3.3166 came from rho = 0.25 as an equicorrelation parameter. It must not recur."""
    out = sens.resolution(1.0)
    key = f"se_inflation_test_train_{sens.PROTOCOL_TEST_TRAIN_RATIO}"
    assert abs(out[key] - 3.3166) > 0.1, "the 75/25 parameterization is back"
    # and the even earlier form, sqrt(1 + rho/(1-rho)) = 1.1547, must not recur either
    assert abs(out[key] - 1.1547) > 0.1


def test_closed_form_standard_error_with_known_inputs() -> None:
    """s, R and the split are known, so the corrected SE is known exactly."""
    s, R, ratio = 0.05, sens.R, sens.PROTOCOL_TEST_TRAIN_RATIO
    expected_se = s * np.sqrt(1.0 / R + ratio)
    out = sens.resolution(s)
    from scipy import stats
    t_a = stats.t.ppf(0.975, R - 1)
    t_b = stats.t.ppf(0.80, R - 1)
    expected_mde = (t_a + t_b) * expected_se
    assert out[f"mde_corrected_test_train_{ratio}"] == pytest.approx(expected_mde, abs=1e-6)


def test_rho_and_the_size_ratio_coincide_only_at_the_matching_split() -> None:
    """They are different parameterizations and must not be used interchangeably."""
    for holdout in (0.10, 0.20, 0.25, 0.30):
        ratio = holdout / (1.0 - holdout)
        rho = holdout                      # rho = n_test / (n_test + n_train)
        assert ratio == pytest.approx(rho / (1.0 - rho), rel=1e-12)
    # the protocol's split: rho is 0.20, the ratio is 0.25, and they are not equal
    assert 0.20 != pytest.approx(sens.PROTOCOL_TEST_TRAIN_RATIO)


def test_the_uncorrected_standard_error_is_the_naive_one() -> None:
    out = sens.resolution(0.04)
    naive_half_width = None
    from scipy import stats
    t_a = stats.t.ppf(0.975, sens.R - 1)
    naive_half_width = t_a * 0.04 / np.sqrt(sens.R)
    assert out["expected_ci95_half_width"] == pytest.approx(naive_half_width, abs=1e-6)
