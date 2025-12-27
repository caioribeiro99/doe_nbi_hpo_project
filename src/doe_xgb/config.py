from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# -------------------------
# Core hyperparameters
# -------------------------
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

