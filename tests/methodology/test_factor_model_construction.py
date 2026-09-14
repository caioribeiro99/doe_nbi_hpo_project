"""The factor scores must be the scores the rotated loadings describe.

Two defects the final protocol review found, both present in the code AND in the
specification, and both in the construction of the study's primary objective:

* rotating raw component scores, whose variances are the eigenvalues, mixes axes
  of unequal scale and yields CORRELATED "orthogonal factors" (measured 0.379 to
  0.698 on the panel). The rotated loadings then do not describe the scores in use,
  so the role assignment and sign orientation are read off the wrong matrix;
* after an orthogonal rotation a component's share of explained variance is the sum
  of its squared rotated loadings. Indexing the UNROTATED eigenvalues by the rotated
  component index pairs two unrelated quantities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from doe_xgb.campaign.evaluator import RESPONSES
from doe_xgb.campaign.factor_model import (apply_transforms, fit_factor_model,
                                           varimax)


def _frame(n: int = 90, seed: int = 0) -> pd.DataFrame:
    """Responses with a genuine two-construct structure plus a cost axis."""
    rng = np.random.default_rng(seed)
    q1 = rng.normal(size=n)
    q2 = rng.normal(size=n)
    cost = 0.6 * q1 + 0.8 * rng.normal(size=n)
    return pd.DataFrame({
        "Accuracy_Mean": 0.80 + 0.05 * q1 + 0.004 * rng.normal(size=n),
        "Precision_Mean": 0.70 + 0.05 * q2 + 0.004 * rng.normal(size=n),
        "Recall_Mean": 0.60 + 0.04 * q1 + 0.008 * rng.normal(size=n),
        "Specificity_Mean": 0.90 - 0.03 * q1 + 0.004 * rng.normal(size=n),
        "RocAuc_Mean": 0.85 + 0.05 * q1 + 0.003 * rng.normal(size=n),
        "LogLoss_Mean": 0.50 - 0.05 * q2 + 0.008 * rng.normal(size=n),
        "Leaves_Mean": np.abs(5000 + 2000 * cost) + 50.0,
    })


@pytest.mark.parametrize("seed", range(4))
def test_the_factor_scores_are_orthogonal(seed: int) -> None:
    """Rotating already-standardized component scores keeps them uncorrelated."""
    fm = fit_factor_model(_frame(seed=seed))
    S = fm.transform(_frame(seed=seed))["factor_scores"]
    C = np.corrcoef(S, rowvar=False) - np.eye(S.shape[1])
    assert np.abs(C).max() < 1e-8, f"factors correlate at {np.abs(C).max():.4f}"


def test_rotating_unstandardized_scores_would_correlate_them() -> None:
    """The defect this guards against must be reachable, or the test proves nothing."""
    from sklearn.decomposition import PCA

    df = _frame(seed=11)
    M = apply_transforms(df)
    Z = (M - M.mean(0)) / np.where(M.std(0, ddof=1) == 0, 1.0, M.std(0, ddof=1))
    pca = PCA(n_components=3, random_state=0).fit(Z)
    lam = pca.explained_variance_
    _, R = varimax(pca.components_.T * np.sqrt(lam))
    wrong = (Z @ pca.components_.T) @ R                       # the old construction
    C = np.corrcoef(wrong, rowvar=False) - np.eye(3)
    assert np.abs(C).max() > 0.05, (
        "the defective construction did not produce correlated factors here, so "
        "this dataset cannot demonstrate the failure"
    )


@pytest.mark.parametrize("seed", range(4))
def test_the_rotated_loadings_describe_the_scores_in_use(seed: int) -> None:
    """The loading of a response on a factor is its correlation with that factor."""
    df = _frame(seed=seed)
    fm = fit_factor_model(df)
    S = fm.transform(df)["factor_scores"]
    M = apply_transforms(df)
    Zr = (M - fm.mu) / fm.sd
    empirical = np.corrcoef(np.column_stack([Zr, S]), rowvar=False)[:Zr.shape[1], Zr.shape[1]:]
    reported = fm.rotated_loadings
    assert np.abs(empirical - reported).max() < 1e-6, (
        f"reported loadings differ from the scores' own correlations by "
        f"{np.abs(empirical - reported).max():.4f}"
    )


@pytest.mark.parametrize("seed", range(4))
def test_quality_weights_are_the_rotated_variance_shares(seed: int) -> None:
    fm = fit_factor_model(_frame(seed=seed))
    ss = (fm.rotated_loadings ** 2).sum(axis=0)
    expected = np.array([ss[j] for j in fm.quality_indices], dtype=float)
    expected = expected / expected.sum()
    np.testing.assert_allclose(fm.quality_weights, expected, atol=1e-12)
    assert abs(fm.quality_weights.sum() - 1.0) < 1e-12


def test_unrotated_eigenvalue_weights_would_differ_materially() -> None:
    """The old weighting is reachable and different, so the fix is not cosmetic."""
    fm = fit_factor_model(_frame(seed=3))
    lam = fm.eigenvalues
    old = np.array([lam[j] for j in fm.quality_indices], dtype=float)
    old = old / old.sum()
    assert np.abs(old - fm.quality_weights).max() > 0.02, (
        "the two weightings agree here, so this frame cannot demonstrate the defect"
    )


def test_role_assignment_and_orientation_are_read_off_the_right_matrix() -> None:
    fm = fit_factor_model(_frame(seed=5))
    names = list(RESPONSES)
    cost_row = names.index("Leaves_Mean")
    # the cost factor is the one the cost response loads most heavily on ...
    assert int(np.argmax(np.abs(fm.rotated_loadings[cost_row]))) == fm.cost_index
    # ... and after orientation that loading is positive, so larger means costlier
    assert fm.rotated_loadings[cost_row, fm.cost_index] > 0
    q_rows = [i for i, c in enumerate(names) if RESPONSES[c]["role"] == "quality"]
    for j in fm.quality_indices:
        assert fm.rotated_loadings[q_rows, j].mean() > 0


def test_both_variance_shares_are_reported_and_named_apart() -> None:
    d = fit_factor_model(_frame(seed=7)).as_dict()
    assert "explained_variance_share_rotated" in d
    assert "explained_variance_share_unrotated" in d
    assert abs(sum(d["explained_variance_share_rotated"]) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# V1 algebraic invariants, asserted rather than inspected
# --------------------------------------------------------------------------


def _pieces(df: pd.DataFrame):
    from sklearn.decomposition import PCA

    fm = fit_factor_model(df)
    M = apply_transforms(df)
    Z = (M - fm.mu) / fm.sd
    k = fm.rotated_loadings.shape[1]
    pca = PCA(n_components=k, random_state=0).fit(Z)
    raw = Z @ pca.components_.T
    std = raw / np.sqrt(fm.eigenvalues)
    rot = std @ fm.rotation
    return fm, Z, pca, raw, std, rot, k


@pytest.mark.parametrize("seed", range(3))
def test_unrotated_score_covariance_is_diagonal(seed: int) -> None:
    _, _, _, raw, _, _, _ = _pieces(_frame(seed=seed))
    C = np.cov(raw, rowvar=False)
    assert np.abs(C - np.diag(np.diag(C))).max() < 1e-8


@pytest.mark.parametrize("seed", range(3))
def test_standardized_and_rotated_scores_have_identity_covariance(seed: int) -> None:
    _, _, _, _, std, rot, k = _pieces(_frame(seed=seed))
    for name, S in (("standardized", std), ("rotated", rot)):
        C = np.cov(S, rowvar=False)
        assert np.abs(C - np.eye(k)).max() < 1e-8, f"{name} scores are not white"


@pytest.mark.parametrize("seed", range(3))
def test_reconstruction_from_scores_and_loadings_matches_retained_pca(seed: int) -> None:
    """The rotation is orthogonal, so it must preserve the represented subspace."""
    fm, _, pca, raw, _, _, _ = _pieces(_frame(seed=seed))
    applied = fm.transform(_frame(seed=seed))["factor_scores"]
    np.testing.assert_allclose(applied @ fm.rotated_loadings.T,
                               raw @ pca.components_, atol=1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_sign_flips_change_orientation_only(seed: int) -> None:
    fm, _, _, raw, _, _, _ = _pieces(_frame(seed=seed))
    applied = fm.transform(_frame(seed=seed))["factor_scores"]
    flip = np.asarray(fm.diagnostics["sign_flips_applied"], dtype=float)
    unflipped = applied * flip
    np.testing.assert_allclose(np.cov(unflipped, rowvar=False),
                               np.cov(applied, rowvar=False), atol=1e-10)
    np.testing.assert_allclose(unflipped @ (fm.rotated_loadings * flip).T,
                               applied @ fm.rotated_loadings.T, atol=1e-10)


@pytest.mark.parametrize("seed", range(3))
def test_role_permutation_preserves_the_subspace(seed: int) -> None:
    fm = fit_factor_model(_frame(seed=seed))
    applied = fm.transform(_frame(seed=seed))["factor_scores"]
    perm = np.argsort(-(fm.rotated_loadings ** 2).sum(axis=0))
    np.testing.assert_allclose(applied[:, perm] @ fm.rotated_loadings[:, perm].T,
                               applied @ fm.rotated_loadings.T, atol=1e-10)


def test_the_factors_are_unit_variance_so_the_weights_are_not_variances() -> None:
    """Terminology: the weights are normalized rotated sums of squared loadings."""
    fm = fit_factor_model(_frame(seed=9))
    S = fm.transform(_frame(seed=9))["factor_scores"]
    np.testing.assert_allclose(S.var(axis=0, ddof=1), np.ones(S.shape[1]), atol=1e-8)
    # unit variances carry no information, so they cannot be the weights
    assert not np.allclose(fm.quality_weights,
                           np.ones(len(fm.quality_weights)) / len(fm.quality_weights))
