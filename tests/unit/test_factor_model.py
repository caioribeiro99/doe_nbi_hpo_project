"""Unit tests for the flexible factor model layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.factor_model import (
    FactorAutoCriteria,
    FactorConstruct,
    FactorModelSpec,
    fit_factor_model,
)


def _synth_dataset(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Three latent factors driving 6 observed metrics.
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    eps = rng.normal(scale=0.2, size=(n, 6))
    cols = {
        "Accuracy_Mean": 0.9 * f1 + eps[:, 0],
        "Precision_Mean": 0.8 * f1 + 0.1 * f3 + eps[:, 1],
        "Recall_Mean": 0.85 * f1 + eps[:, 2],
        "Time_MeanFold": 0.95 * f2 + eps[:, 3],
        "Specificity_Std": 0.9 * f3 + eps[:, 4],
        "Brier_Score": 0.5 * f2 + 0.3 * f3 + eps[:, 5],
    }
    return pd.DataFrame(cols)


def test_fixed_mode_returns_requested_n_factors() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="fixed", n_factors=3)
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert res.scores.shape[1] == 3
    assert res.loadings.shape == (6, 3)


def test_auto_mode_picks_at_least_one_factor() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="auto")
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert 1 <= res.scores.shape[1] <= 6
    assert res.cumulative_variance[-1] == pytest.approx(
        sum(res.explained_variance), rel=1e-6
    )


def test_auto_mode_diagnostics_contain_kmo_and_construct_map() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="auto")
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert "kmo" in res.diagnostics
    assert "construct_map" in res.diagnostics


def test_manual_mode_creates_named_constructs() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(
        mode="manual",
        constructs=(
            FactorConstruct(name="Quality", members=("Accuracy_Mean", "Precision_Mean", "Recall_Mean")),
            FactorConstruct(name="Cost", members=("Time_MeanFold",)),
            FactorConstruct(name="Robustness", members=("Specificity_Std", "Brier_Score")),
        ),
    )
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert res.scores.shape[1] == 3
    assert "Factor1" in res.construct_map
    assert res.construct_map["Factor1"] == ("Accuracy_Mean", "Precision_Mean", "Recall_Mean")


def test_manual_mode_rejects_unknown_member() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(
        mode="manual",
        constructs=(FactorConstruct(name="Q", members=("not_in_df",)),),
    )
    with pytest.raises(KeyError):
        fit_factor_model(df, df.columns.tolist(), spec)


def test_none_mode_passes_through_metrics() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="none")
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert res.scores.shape == df.shape
    np.testing.assert_allclose(res.loadings.values, np.eye(df.shape[1]))


def test_fixed_mode_requires_n_factors() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="fixed")
    with pytest.raises(ValueError):
        fit_factor_model(df, df.columns.tolist(), spec)


def test_manual_mode_requires_constructs() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="manual")
    with pytest.raises(ValueError):
        fit_factor_model(df, df.columns.tolist(), spec)


def test_sign_orientation_is_deterministic_across_calls() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="fixed", n_factors=3)
    a = fit_factor_model(df, df.columns.tolist(), spec)
    b = fit_factor_model(df, df.columns.tolist(), spec)
    np.testing.assert_allclose(a.loadings.values, b.loadings.values)
    np.testing.assert_allclose(a.scores.values, b.scores.values)


def test_explained_variance_decreasing_in_fixed_mode() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="fixed", n_factors=3)
    res = fit_factor_model(df, df.columns.tolist(), spec)
    ev = res.explained_variance
    assert all(ev[i] >= ev[i + 1] - 1e-12 for i in range(len(ev) - 1))


def test_no_rotation_path_runs() -> None:
    df = _synth_dataset()
    spec = FactorModelSpec(mode="fixed", n_factors=2, rotation="none")
    res = fit_factor_model(df, df.columns.tolist(), spec)
    np.testing.assert_allclose(res.rotation_matrix, np.eye(2))


def test_auto_threshold_overrides() -> None:
    df = _synth_dataset()
    # Force a tighter cumvar threshold; should still return >=1 factor.
    spec = FactorModelSpec(
        mode="auto",
        auto=FactorAutoCriteria(eigen_threshold=10.0, cumvar_threshold=0.99),
    )
    res = fit_factor_model(df, df.columns.tolist(), spec)
    assert res.scores.shape[1] >= 1
