from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer  # type: ignore

# Quality responses used in DOE
QUALITY_METRICS: List[str] = [
    "Accuracy_Mean",
    "Precision_Mean",
    "Recall_Mean",
    "Specificity_Mean",
]

# Cost proxy (smaller is better)
TIME_METRIC: str = "Time_MeanFold"


@dataclass(frozen=True)
class FAResult:
    loadings: pd.DataFrame
    scores: pd.DataFrame
    quality_factor: int  # 1-based index (Factor1 -> 1, Factor2 -> 2, ...)
    cost_factor: int  # 1-based if derived from FA; 0 if derived directly from time z-score


def _zscore_df(x: pd.DataFrame) -> np.ndarray:
    """Column-wise z-score using sample std (ddof=1), matching Minitab's Z-score."""
    x = x.astype(float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    zero_sd = sd[sd == 0]
    if len(zero_sd) > 0:
        cols = list(zero_sd.index.astype(str))
        raise ValueError(f"Cannot z-score columns with zero std: {cols}")
    z = (x - mu) / sd
    if not np.isfinite(z.to_numpy()).all():
        raise ValueError("Non-finite values after z-score normalization")
    return z.to_numpy()


def run_factor_analysis(
    df: pd.DataFrame,
    *,
    variables: Optional[Sequence[str]] = None,
    n_factors: int = 2,
    rotation: str = "varimax",
    method: str = "ml",
    min_abs_time_loading: float = 0.30,
    fallback_cost: str = "z_time",  # 'z_time' or 'z_log_time'
) -> FAResult:
    """Run Factor Analysis (ML + Varimax) on DOE responses.

    Important details (to match the original Minitab workflow):
    - Inputs are standardized via Z-score (mean=0, std=1 with ddof=1).
    - Quality score comes from the factor that best explains the QUALITY_METRICS.
    - Cost score should be aligned with TIME_METRIC (smaller time => better).
      If TIME_METRIC does not load meaningfully on any factor (common in practice
      when time is largely independent from the quality responses), we fall back
      to a direct cost score computed from the (negative) z-score of time.

    Returns
    -------
    FAResult:
      - loadings: variables x factors loadings table
      - scores: per-run factor scores + Score_Quality + Score_Cost
      - quality_factor: 1-based index for the selected quality factor
      - cost_factor: 1-based index if selected from FA, 0 if derived from time
    """
    if variables is None:
        variables = list(QUALITY_METRICS) + [TIME_METRIC]
    else:
        variables = list(variables)

    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for factor analysis: {missing}")

    # Z-score normalization (Minitab-style)
    X_df = df[variables].copy()
    Z = _zscore_df(X_df)

    # Fit FA (with a robust fallback)
    fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method)
    try:
        fa.fit(Z)
    except Exception:
        # ML can fail on some correlation matrices; MinRes is more robust.
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="minres")
        fa.fit(Z)

    loadings = pd.DataFrame(
        fa.loadings_,
        index=variables,
        columns=[f"Factor{i+1}" for i in range(n_factors)],
    )

    factor_scores = fa.transform(Z)  # (n_samples, n_factors)
    scores = pd.DataFrame(
        factor_scores,
        columns=[f"FACTOR{i+1}_SCORE" for i in range(n_factors)],
        index=df.index,
    )

    # ---------------------------
    # Pick QUALITY factor (from FA)
    # ---------------------------
    q_rows = loadings.loc[[c for c in QUALITY_METRICS if c in loadings.index]]
    mean_abs = q_rows.abs().mean(axis=0).to_numpy()
    quality_idx = int(np.argmax(mean_abs))  # 0-based

    # Orient quality so that higher => better (positive correlation with mean Z quality)
    q_pos = [variables.index(c) for c in QUALITY_METRICS if c in variables]
    quality_ref = Z[:, q_pos].mean(axis=1)
    corr_q = float(np.corrcoef(scores.iloc[:, quality_idx].to_numpy(), quality_ref)[0, 1])
    if np.isfinite(corr_q) and corr_q < 0:
        scores.iloc[:, quality_idx] *= -1
        loadings.iloc[:, quality_idx] *= -1

    scores["Score_Quality"] = scores.iloc[:, quality_idx]

    # ---------------------------
    # Pick COST factor (prefer FA, otherwise fallback to z-score time)
    # ---------------------------
    cost_factor = 0  # 0 => derived from time, not a FA factor

    if TIME_METRIC in variables:
        time_pos = variables.index(TIME_METRIC)

        # Prefer the factor where TIME_METRIC has the largest absolute loading
        time_abs_loads = loadings.loc[TIME_METRIC].abs().to_numpy()
        best_time_idx = int(np.argmax(time_abs_loads))
        best_time_loading = float(time_abs_loads[best_time_idx])

        use_fa_cost = (best_time_idx != quality_idx) and (best_time_loading >= min_abs_time_loading)

        if use_fa_cost:
            cost_idx = best_time_idx
            cost_factor = cost_idx + 1

            # Orient cost so that higher => faster (negative correlation with raw time)
            raw_time = df[TIME_METRIC].astype(float).to_numpy()
            corr_t = float(np.corrcoef(scores.iloc[:, cost_idx].to_numpy(), raw_time)[0, 1])
            if np.isfinite(corr_t) and corr_t > 0:
                scores.iloc[:, cost_idx] *= -1
                loadings.iloc[:, cost_idx] *= -1

            scores["Score_Cost"] = scores.iloc[:, cost_idx]
        else:
            # Fallback: direct cost score from time
            if fallback_cost == "z_log_time":
                t = df[TIME_METRIC].astype(float).to_numpy()
                t_log = np.log1p(t)
                z_t = (t_log - float(t_log.mean())) / float(t_log.std(ddof=1))
                scores["Score_Cost"] = -z_t
            else:
                # z-score of time is Z[:, time_pos]
                scores["Score_Cost"] = -Z[:, time_pos]
    else:
        # No time column available; define neutral cost
        scores["Score_Cost"] = 0.0

    return FAResult(
        loadings=loadings,
        scores=scores,
        quality_factor=quality_idx + 1,
        cost_factor=cost_factor,
    )


def save_fa_outputs(res: FAResult, loadings_path: str, scores_path: str) -> None:
    """Save FA loadings and scores in pt-BR friendly CSV."""
    res.loadings.to_csv(loadings_path, sep=";", decimal=",", index=False)
    res.scores.to_csv(scores_path, sep=";", decimal=",", index=False)
