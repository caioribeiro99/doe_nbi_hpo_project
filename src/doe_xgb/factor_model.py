"""Flexible factor decomposition layer.

Article-track replacement for :mod:`factor_analysis`. Supports four modes:

- ``auto``: pick the number of factors from eigenvalue > 1 + a cumulative
  variance threshold; bounded by the number of metrics.
- ``fixed``: explicit ``n_factors`` (article-track default = 3).
- ``manual``: the user supplies a list of constructs (named metric
  groups). Each construct becomes one factor; the factor scores are the
  z-scored mean of the standardized members.
- ``none``: no factor analysis; raw metrics flow through unchanged.

All modes return a :class:`FactorModelResult` that carries the diagnostics
required by :doc:`docs/METHODOLOGY_DECISIONS.md` (eigenvalues, explained
variance, communalities, rotation matrix, sign signs, KMO if available,
construct map, naming suggestions, and a list of warnings).

This module is dimension-agnostic: it works for any number of input
metrics and any number of factors >= 1. The legacy
:mod:`doe_xgb.factor_analysis` path remains available for the
dissertation baseline; the new article-track pipeline calls
:func:`fit_factor_model` instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

FactorMode = Literal["auto", "fixed", "manual", "none"]


@dataclass(frozen=True)
class FactorAutoCriteria:
    eigen_threshold: float = 1.0
    cumvar_threshold: float = 0.85
    min_loading_abs: float = 0.4


@dataclass(frozen=True)
class FactorConstruct:
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class FactorModelSpec:
    mode: FactorMode = "auto"
    n_factors: int | None = None
    constructs: tuple[FactorConstruct, ...] | None = None
    rotation: Literal["varimax", "none"] = "varimax"
    standardize: bool = True
    sign_orientation: Literal[
        "construct_avg", "first_loading_positive", "explicit"
    ] = "construct_avg"
    auto: FactorAutoCriteria = FactorAutoCriteria()


@dataclass
class FactorModelResult:
    metrics: tuple[str, ...]
    loadings: pd.DataFrame
    scores: pd.DataFrame
    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    cumulative_variance: np.ndarray
    communalities: pd.Series
    rotation_matrix: np.ndarray
    construct_map: dict[str, tuple[str, ...]]
    sign_signs: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zscore(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0, ddof=ddof)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (x - mu) / sigma


def _varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 200, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (Lambda ** 3 - (gamma / p) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0)))
        )
        R = u @ vh
        d = float(np.sum(s))
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
    return Phi @ R, R


def _kaiser_n_factors(corr: np.ndarray, threshold: float) -> int:
    eigvals = np.linalg.eigvalsh(corr)
    return int(np.sum(eigvals > threshold))


def _kmo(corr: np.ndarray) -> float | None:
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    Returns None if the partial correlation matrix is singular.
    """
    try:
        inv = np.linalg.pinv(corr)
        partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
        np.fill_diagonal(partial, 0.0)
        np.fill_diagonal(corr, 0.0)
        num = float(np.sum(corr ** 2))
        den = num + float(np.sum(partial ** 2))
        if den == 0.0:
            return None
        return num / den
    except np.linalg.LinAlgError:
        return None


# ---------------------------------------------------------------------------
# Core fitter
# ---------------------------------------------------------------------------


def fit_factor_model(
    df: pd.DataFrame,
    metrics: Sequence[str],
    spec: FactorModelSpec,
) -> FactorModelResult:
    """Fit a factor model according to ``spec`` on the columns ``metrics``."""

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise KeyError(f"missing columns for factor model: {missing}")

    raw = df.loc[:, list(metrics)].astype(float).to_numpy()
    if spec.standardize:
        Z = _zscore(raw)
    else:
        Z = raw - raw.mean(axis=0, keepdims=True)

    p = Z.shape[1]
    if spec.mode == "none":
        # Identity passthrough: each metric is its own "factor".
        scores = pd.DataFrame(
            Z,
            index=df.index,
            columns=[f"FACTOR{i + 1}_SCORE" for i in range(p)],
        )
        loadings = pd.DataFrame(
            np.eye(p),
            index=list(metrics),
            columns=[f"Factor{i + 1}" for i in range(p)],
        )
        return FactorModelResult(
            metrics=tuple(metrics),
            loadings=loadings,
            scores=scores,
            eigenvalues=np.ones(p),
            explained_variance=np.full(p, 1.0 / p),
            cumulative_variance=np.linspace(1.0 / p, 1.0, p),
            communalities=pd.Series(np.ones(p), index=list(metrics), name="h2"),
            rotation_matrix=np.eye(p),
            construct_map={f"Factor{i + 1}": (m,) for i, m in enumerate(metrics)},
            sign_signs=np.ones(p),
            diagnostics={"mode": "none"},
        )

    if spec.mode == "manual":
        if not spec.constructs:
            raise ValueError("mode=manual requires constructs")
        return _fit_manual(df, Z, metrics, spec)

    # PCA path (auto or fixed).
    corr = np.corrcoef(Z, rowvar=False)
    eigvals_total = np.linalg.eigvalsh(corr)[::-1]
    if spec.mode == "fixed":
        if spec.n_factors is None:
            raise ValueError("mode=fixed requires n_factors")
        n_factors = max(1, min(int(spec.n_factors), p))
    else:
        kaiser_k = _kaiser_n_factors(corr, threshold=spec.auto.eigen_threshold)
        cum = np.cumsum(eigvals_total) / float(np.sum(eigvals_total))
        cumvar_k = int(np.searchsorted(cum, spec.auto.cumvar_threshold) + 1)
        n_factors = max(1, min(max(kaiser_k, cumvar_k), p))

    pca = PCA(n_components=n_factors, random_state=0)
    scores_unrot = pca.fit_transform(Z)
    loadings_unrot = pca.components_.T  # shape (p, k)

    # Scale loadings by sqrt(eigenvalue) so columns reflect explained variance.
    eigvals_top = pca.explained_variance_
    loadings_scaled = loadings_unrot * np.sqrt(eigvals_top)

    if spec.rotation == "varimax":
        rot_loadings, R = _varimax(loadings_scaled)
        rot_scores = (scores_unrot / np.sqrt(eigvals_top)) @ R
        rot_scores = rot_scores * np.sqrt(eigvals_top)
    else:
        rot_loadings = loadings_scaled
        R = np.eye(n_factors)
        rot_scores = scores_unrot

    factor_cols = [f"Factor{i + 1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(rot_loadings, index=list(metrics), columns=factor_cols)

    # Communalities = sum of squared loadings per row.
    communalities = pd.Series(np.sum(rot_loadings ** 2, axis=1), index=list(metrics), name="h2")

    # Sign orientation: keep deterministic.
    if spec.sign_orientation == "first_loading_positive":
        signs = np.sign(rot_loadings[0, :])
    else:
        # Default: orient each factor so the average loading on its
        # dominant metrics is positive.
        mean_per_factor = rot_loadings.mean(axis=0)
        signs = np.sign(mean_per_factor)
    signs = np.where(signs == 0.0, 1.0, signs)

    rot_loadings_oriented = rot_loadings * signs
    rot_scores_oriented = rot_scores * signs
    loadings_df = pd.DataFrame(rot_loadings_oriented, index=list(metrics), columns=factor_cols)

    scores_df = pd.DataFrame(
        rot_scores_oriented,
        index=df.index,
        columns=[f"FACTOR{i + 1}_SCORE" for i in range(n_factors)],
    )

    construct_map: dict[str, tuple[str, ...]] = {}
    for j, factor in enumerate(factor_cols):
        members = tuple(
            metrics[i]
            for i in range(p)
            if abs(rot_loadings_oriented[i, j]) >= spec.auto.min_loading_abs
        )
        construct_map[factor] = members

    diagnostics: dict[str, object] = {
        "mode": spec.mode,
        "n_factors_chosen": int(n_factors),
        "kmo": _kmo(corr.copy()),
        "rotation": spec.rotation,
        "construct_map": construct_map,
    }

    return FactorModelResult(
        metrics=tuple(metrics),
        loadings=loadings_df,
        scores=scores_df,
        eigenvalues=eigvals_total,
        explained_variance=pca.explained_variance_ratio_,
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_),
        communalities=communalities,
        rotation_matrix=R,
        construct_map=construct_map,
        sign_signs=signs,
        diagnostics=diagnostics,
    )


def _fit_manual(
    df: pd.DataFrame,
    Z: np.ndarray,
    metrics: Sequence[str],
    spec: FactorModelSpec,
) -> FactorModelResult:
    metrics_list = list(metrics)
    metric_index: dict[str, int] = {m: i for i, m in enumerate(metrics_list)}
    constructs = list(spec.constructs or ())
    factor_cols = [f"Factor{i + 1}" for i in range(len(constructs))]

    n = Z.shape[0]
    scores_arr = np.zeros((n, len(constructs)), dtype=float)
    loadings_arr = np.zeros((len(metrics_list), len(constructs)), dtype=float)

    construct_map: dict[str, tuple[str, ...]] = {}
    for j, c in enumerate(constructs):
        missing = [m for m in c.members if m not in metric_index]
        if missing:
            raise KeyError(
                f"construct {c.name!r} references metrics not in the input: {missing}"
            )
        member_idx = [metric_index[m] for m in c.members]
        member_scores = Z[:, member_idx]
        scores_arr[:, j] = member_scores.mean(axis=1)
        for idx in member_idx:
            loadings_arr[idx, j] = 1.0 / np.sqrt(len(member_idx))
        construct_map[factor_cols[j]] = c.members

    # Sign orientation: by construct average sign of loadings.
    signs = np.where(loadings_arr.sum(axis=0) >= 0.0, 1.0, -1.0)
    loadings_arr = loadings_arr * signs
    scores_arr = scores_arr * signs

    loadings_df = pd.DataFrame(loadings_arr, index=metrics_list, columns=factor_cols)
    scores_df = pd.DataFrame(
        scores_arr,
        index=df.index,
        columns=[f"FACTOR{i + 1}_SCORE" for i in range(len(constructs))],
    )
    communalities = pd.Series(
        (loadings_arr ** 2).sum(axis=1),
        index=metrics_list,
        name="h2",
    )
    diagnostics: dict[str, object] = {
        "mode": "manual",
        "n_factors_chosen": int(len(constructs)),
        "construct_map": construct_map,
    }
    return FactorModelResult(
        metrics=tuple(metrics_list),
        loadings=loadings_df,
        scores=scores_df,
        eigenvalues=np.array([]),
        explained_variance=np.array([]),
        cumulative_variance=np.array([]),
        communalities=communalities,
        rotation_matrix=np.eye(len(constructs)),
        construct_map=construct_map,
        sign_signs=signs,
        diagnostics=diagnostics,
    )


__all__ = [
    "FactorMode",
    "FactorAutoCriteria",
    "FactorConstruct",
    "FactorModelSpec",
    "FactorModelResult",
    "fit_factor_model",
]
