from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .io_utils import save_csv_ptbr

# ----------------------------------------------------------------------
# Defaults (match the columns produced by our DOE evaluator)
# ----------------------------------------------------------------------

DEFAULT_METRICS: Tuple[str, ...] = (
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "Specificity_Mean",
    "Time_MeanFold",
)

DEFAULT_TIME_COL = "Time_MeanFold"


@dataclass(frozen=True)
class FAResult:
    """Outputs of the (PCA-based) rotated factor score procedure."""
    loadings: pd.DataFrame
    scores: pd.DataFrame
    quality_factor: int
    cost_factor: int


def _zscore(x: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0, ddof=ddof)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (x - mu) / sigma


def _varimax(
    Phi: np.ndarray,
    *,
    gamma: float = 1.0,
    q: int = 100,
    tol: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0

    for _ in range(q):
        d_old = d
        Lambda = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T
            @ (
                Lambda**3
                - (gamma / p) * Lambda @ np.diag(np.sum(Lambda**2, axis=0))
            )
        )
        R = u @ vh
        d = float(np.sum(s))
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break

    return Phi @ R, R


def _kaiser_n_factors(corr: np.ndarray) -> int:
    eigvals = np.linalg.eigvalsh(corr)
    return int(np.sum(eigvals > 1.0))


def run_factor_analysis(
    df: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    time_col: str = DEFAULT_TIME_COL,
    n_factors: Optional[int] = 3,
    auto_n_factors: bool = False,
    force_time_factor: bool = True,
    rotation: str = "varimax",
    zscore_ddof: int = 1,
    time_transform: str = "log1p",
    combine_quality_factors: bool = True,
    quality_weights: Optional[Sequence[float]] = None,
    sanity_check: bool = True,
) -> FAResult:

    missing = [c for c in metrics if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for FA/PCA: {missing}")

    # Build X in the metric order (NO to_numpy)
    X_raw = np.asarray(df.loc[:, list(metrics)], dtype=float)

    # Optional transform for time
    if time_col in metrics and time_transform.lower() != "none":
        t_idx = list(metrics).index(time_col)
        t = X_raw[:, t_idx].copy()
        t = np.where(np.isfinite(t), t, np.nan)

        if time_transform.lower() == "log1p":
            t = np.log1p(np.clip(t, a_min=0.0, a_max=None))
        elif time_transform.lower() == "log":
            t = np.log(np.clip(t, a_min=1e-12, a_max=None))
        else:
            raise ValueError(f"Unknown time_transform: {time_transform}")

        X_raw[:, t_idx] = t

    Z = _zscore(X_raw, ddof=zscore_ddof)

    p = Z.shape[1]
    if auto_n_factors:
        corr = np.corrcoef(Z, rowvar=False)
        k = _kaiser_n_factors(corr)
        if force_time_factor and time_col in metrics:
            k = max(k, 3)
        n_factors_eff = max(1, min(k, p))
    else:
        if n_factors is None:
            n_factors_eff = 3 if (force_time_factor and time_col in metrics) else 2
        else:
            n_factors_eff = int(n_factors)
            if force_time_factor and time_col in metrics:
                n_factors_eff = max(n_factors_eff, 3)
            n_factors_eff = max(1, min(n_factors_eff, p))

    pca = PCA(n_components=n_factors_eff, random_state=0)
    scores = pca.fit_transform(Z)
    loadings = pca.components_.T

    if rotation.lower() == "none":
        rot_loadings = loadings
        R = np.eye(n_factors_eff)
        rot_scores = scores
    elif rotation.lower() == "varimax":
        rot_loadings, R = _varimax(loadings)
        rot_scores = scores @ R
    else:
        raise ValueError(f"Unknown rotation: {rotation}")

    factor_cols = [f"Factor{i+1}" for i in range(n_factors_eff)]
    loadings_df = pd.DataFrame(rot_loadings, index=list(metrics), columns=factor_cols)

    if time_col in metrics:
        abs_time = np.asarray(loadings_df.loc[time_col].abs(), dtype=float)  # NO to_numpy
        time_idx = int(np.argmax(abs_time))
        cost_factor = time_idx + 1
    else:
        time_idx = -1
        cost_factor = 0

    quality_metrics = [m for m in metrics if m != time_col]
    if quality_metrics:
        mean_abs = np.asarray(loadings_df.loc[quality_metrics].abs().mean(axis=0), dtype=float)  # NO to_numpy
        quality_idx = int(np.argmax(mean_abs))
        quality_factor = quality_idx + 1
    else:
        quality_idx = 0
        quality_factor = 1

    signs = np.ones(n_factors_eff, dtype=float)
    for j in range(n_factors_eff):
        if time_col in metrics and j == time_idx:
            s = float(np.sign(loadings_df.loc[time_col, factor_cols[j]]))
            signs[j] = 1.0 if s == 0.0 else s
        else:
            if not quality_metrics:
                continue
            s = float(np.sign(loadings_df.loc[quality_metrics, factor_cols[j]].mean()))
            signs[j] = 1.0 if s == 0.0 else s

    rot_loadings_oriented = rot_loadings * signs
    rot_scores_oriented = rot_scores * signs
    loadings_df = pd.DataFrame(rot_loadings_oriented, index=list(metrics), columns=factor_cols)

    scores_df = pd.DataFrame(
        rot_scores_oriented,
        index=df.index,
        columns=[f"FACTOR{i+1}_SCORE" for i in range(n_factors_eff)],
    )

    if time_col in metrics:
        time_factor_score = np.asarray(scores_df[f"FACTOR{time_idx+1}_SCORE"], dtype=float)
        score_cost = -_zscore(time_factor_score.reshape(-1, 1), ddof=zscore_ddof).ravel()
    else:
        score_cost = np.zeros(len(df), dtype=float)

    if combine_quality_factors and time_col in metrics and n_factors_eff >= 3:
        qual_idxs = [j for j in range(n_factors_eff) if j != time_idx]
    else:
        qual_idxs = [quality_idx]

    qual_cols = [f"FACTOR{j+1}_SCORE" for j in qual_idxs]
    Q = np.asarray(scores_df.loc[:, qual_cols], dtype=float)  # NO to_numpy
    Qz = _zscore(Q, ddof=zscore_ddof)

    if quality_weights is None:
        w = np.ones(Qz.shape[1], dtype=float)
    else:
        w = np.asarray(list(quality_weights), dtype=float)
        if w.shape[0] != Qz.shape[1]:
            raise ValueError(f"quality_weights length mismatch: expected {Qz.shape[1]}, got {w.shape[0]}")

    score_quality = (Qz * w).sum(axis=1) / float(np.sum(w))

    scores_df["Score_Quality"] = score_quality
    scores_df["Score_Cost"] = score_cost

    if sanity_check and time_col in metrics:
        raw_time = np.asarray(df[time_col], dtype=float)
        mask = np.isfinite(raw_time) & np.isfinite(score_cost)
        if mask.sum() >= 3:
            corr_cost_time = float(np.corrcoef(score_cost[mask], raw_time[mask])[0, 1])
            if corr_cost_time > -0.30:
                print(
                    "⚠️  [FA/PCA sanity] Score_Cost is weakly related to Time_MeanFold "
                    f"(corr={corr_cost_time:.3f}). Consider increasing n_factors, "
                    "ensuring time_col is included, or using time_transform='log1p'."
                )

    return FAResult(
        loadings=loadings_df,
        scores=scores_df,
        quality_factor=quality_factor,
        cost_factor=cost_factor,
    )


def save_fa_outputs(res: FAResult, loadings_path: str, scores_path: str) -> None:
    save_csv_ptbr(res.loadings.reset_index().rename(columns={"index": "Metric"}), loadings_path)
    save_csv_ptbr(res.scores.reset_index().rename(columns={"index": "Row"}), scores_path)
