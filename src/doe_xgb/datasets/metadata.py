"""Typed metadata describing one entry of the dataset registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SourceType = Literal["uci", "openml", "sklearn", "local"]
TaskType = Literal["binary", "multiclass"]
Burden = Literal["light", "medium", "heavy"]


@dataclass(frozen=True)
class DatasetMetadata:
    """Static description of a dataset entry.

    Loaders attach two additional fields at load time:
    ``n_rows`` and ``class_distribution``.
    """

    dataset_id: str
    display_name: str
    source_type: SourceType
    source_url: str | None = None
    openml_id: int | None = None
    task_type: TaskType = "binary"
    target_column: str | None = None
    target_transform: str | None = None
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    missing_value_policy: str = "drop_unknown"  # or "keep" / "impute_median"
    recommended_metrics: tuple[str, ...] = (
        "roc_auc",
        "pr_auc",
        "f1",
        "balanced_accuracy",
        "mcc",
    )
    calibration_metrics_enabled: bool = True
    burden: Burden = "medium"
    license_note: str | None = None
    citation_key: str | None = None
    include_in_v1: bool = True
    fallback_dataset_id: str | None = None
    notes: str | None = None
    # Filled at load time.
    n_rows: int | None = field(default=None, compare=False)
    n_features: int | None = field(default=None, compare=False)
    class_distribution: dict[int | str, int] | None = field(default=None, compare=False)


@dataclass(frozen=True)
class LoadedDataset:
    """Tuple-like result of a loader call."""

    X: object  # pandas.DataFrame; typed as object to avoid hard pandas import here
    y: object  # pandas.Series
    metadata: DatasetMetadata


__all__ = [
    "DatasetMetadata",
    "LoadedDataset",
    "SourceType",
    "TaskType",
    "Burden",
]
