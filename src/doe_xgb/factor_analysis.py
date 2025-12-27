from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import QUALITY_METRICS, TIME_METRIC
from .io_utils import save_csv_ptbr


@dataclass(frozen=True)
class FAResult:
    loadings: pd.DataFrame
    scores: pd.DataFrame
    quality_factor: int
    cost_factor: int


def run_factor_analysis(
    df: pd.DataFrame,
    *,
    variables: List[str] | None = None,
    n_factors: int = 2,
    method: str = "ml",
    rotation: str = "varimax",
    orient: bool = True,
) -> FAResult:
    """Run Factor Analysis (ML + Varimax) and return loadings and scores.

    Dependency
    ----------
    Requires `factor_analyzer`. Install via:
        pip install factor_analyzer

    Notes
    -----
    - We standardize variables (equivalent to working on the correlation matrix).
    - Factor signs are arbitrary; if `orient=True`, we enforce:
        * Quality score correlates positively with Accuracy_Mean
        * Cost score correlates negatively with Time_MeanFold (so higher is faster)
    """
    try:
        from factor_analyzer import FactorAnalyzer
    except ImportError as e:
        raise ImportError(
            "Missing dependency 'factor_analyzer'. Install it with: pip install factor_analyzer"
        ) from e

    if variables is None:
        variables = QUALITY_METRICS + [TIME_METRIC]

    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for FA: {missing}")

    X = df[variables].astype(float).to_numpy()
    Z = StandardScaler().fit_transform(X)

    fa = FactorAnalyzer(n_factors=n_factors, method=method, rotation=rotation)
    fa.fit(Z)

    loadings = pd.DataFrame(fa.loadings_, index=variables, columns=[f"Factor{i+1}" for i in range(n_factors)])
    scores_arr = fa.transform(Z)
    scores = pd.DataFrame(scores_arr, columns=[f"FACTOR{i+1}_SCORE" for i in range(n_factors)], index=df.index)

    # Identify which factor is "quality"
    quality_idx = int(
        np.argmax([
            np.mean(np.abs(loadings.loc[QUALITY_METRICS, f"Factor{i+1}"]))
            for i in range(n_factors)
        ])
    )
    cost_idx = 1 - quality_idx if n_factors == 2 else int(np.argmin([quality_idx]))

    q_col = f"FACTOR{quality_idx+1}_SCORE"
    c_col = f"FACTOR{cost_idx+1}_SCORE"
    scores["Score_Quality"] = scores[q_col]
    scores["Score_Cost"] = scores[c_col]

    if orient:
        # Quality: positive correlation with Accuracy
        if np.corrcoef(scores["Score_Quality"], df["Accuracy_Mean"].astype(float))[0, 1] < 0:
            scores["Score_Quality"] *= -1

        # Cost: we want "higher is faster" => negative corr with time
        if np.corrcoef(scores["Score_Cost"], df[TIME_METRIC].astype(float))[0, 1] > 0:
            scores["Score_Cost"] *= -1

    return FAResult(loadings=loadings, scores=scores, quality_factor=quality_idx + 1, cost_factor=cost_idx + 1)


def save_fa_outputs(result: FAResult, loadings_path: str, scores_path: str) -> None:
    save_csv_ptbr(result.loadings.reset_index().rename(columns={"index": "Variable"}), loadings_path)
    save_csv_ptbr(result.scores, scores_path)
