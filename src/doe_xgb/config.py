from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------
# Core hyperparameters (classic pipeline)
# -------------------------
#
# NOTE:
# - Keep this list stable for the classic (non-fairness) pipeline so old designs
#   and the original NBI implementation remain backwards compatible.
#
PARAM_NAMES: List[str] = [
    "subsample",
    "colsample_bytree",
    "colsample_bylevel",
    "learning_rate",
    "max_depth",
    "gamma",
    "n_estimators",
]

INT_PARAMS = {"max_depth", "n_estimators"}

# Default bounds for CCD face-centered (edit if you expand beyond the usual limits)
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "subsample": (0.05, 1.0),
    "colsample_bytree": (0.05, 1.0),
    "colsample_bylevel": (0.05, 1.0),
    "learning_rate": (0.01, 0.30),
    "max_depth": (3.0, 18.0),
    "gamma": (0.05, 5.0),
    "n_estimators": (50.0, 700.0),
}

# -------------------------
# Fairness pipeline hyperparameters
# -------------------------
#
# In fairness mode we additionally tune:
#   - scale_pos_weight: handles class imbalance (XGBoost built-in)
#   - threshold: probability threshold used to convert p(y=1) -> class label
#
# Keeping these separated avoids breaking the classic pipeline.
#
FAIRNESS_PARAM_NAMES: List[str] = [
    *PARAM_NAMES,
    "scale_pos_weight",
    "threshold",
]

FAIRNESS_DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    **DEFAULT_BOUNDS,
    # Typical useful ranges for highly imbalanced binary classification.
    # You can tighten these once you observe where good solutions concentrate.
    "scale_pos_weight": (1.0, 30.0),
    # Avoid extreme thresholds that tend to collapse predictions (all-0 or all-1).
    "threshold": (0.10, 0.90),
}

# Defaults used when a design does not include these columns.
FAIRNESS_DEFAULTS = {
    "scale_pos_weight": 1.0,
    "threshold": 0.5,
}

QUALITY_METRICS = ["Accuracy_Mean", "Precision_Mean", "Recall_Mean", "Specificity_Mean"]
TIME_METRIC = "Time_MeanFold"

FACTOR_SCORE_COLS = ["FACTOR1_SCORE", "FACTOR2_SCORE"]


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True


@dataclass(frozen=True)
class XGBConfig:
    n_jobs: int = -1
    tree_method: str = "hist"
    eval_metric: str = "logloss"
    verbosity: int = 0


@dataclass(frozen=True)
class BenchmarkConfig:
    budget: int = 40  # target number of objective evaluations
    grid_levels: int = 3  # used to build a small coarse grid (<= budget)


@dataclass(frozen=True)
class ExperimentPaths:
    experiments_root: Path = Path("experiments")
