from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .config import PARAM_NAMES
from .evaluation import evaluate_xgb_cv
from .io_utils import save_csv_ptbr


def run_doe(
    design_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
    n_splits: int = 5,
    n_jobs: int = -1,
    tree_method: str = "hist",
) -> pd.DataFrame:
    """Run DOE evaluation for every row in the design matrix."""
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: List[dict] = []

    # Keep all original design columns + computed metrics
    for _, row in tqdm(design_df.iterrows(), total=len(design_df), desc="DOE runs"):
        params = {k: row[k] for k in PARAM_NAMES if k in design_df.columns}
        ev = evaluate_xgb_cv(
            params,
            X_np,
            y_np,
            kfold,
            seed=seed,
            n_jobs=n_jobs,
            tree_method=tree_method,
            measure="fit_predict",
        )
        out = {c: row[c] for c in design_df.columns}
        out.update(ev.metrics)
        out["Time_MeanFold"] = ev.time_mean_fold
        results.append(out)

    return pd.DataFrame(results)


def save_doe_results(df: pd.DataFrame, out_path: str | Path) -> None:
    """Save DOE results as CSV (pt-BR friendly)."""
    save_csv_ptbr(df, out_path)
