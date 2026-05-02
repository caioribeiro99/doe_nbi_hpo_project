"""Article-track guardrails connecting the dataset registry, the YAML
config schema, and :func:`doe_xgb.evaluation.assert_metric_set_compatible_with_task`.

This module exists to give the (future) orchestrator a single function
to call after loading a YAML config and resolving the dataset, so a
multiclass dataset cannot enter the FA / NBI stages with the binary
response defaults.

Until a stable orchestrator lands, the helper here is exercised by
unit tests and can be invoked manually inside ``scripts/run_replica.py``
or any equivalent runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..evaluation import (
    MultiClassNotConfiguredError,
    assert_metric_set_compatible_with_task,
)
from .registry import REGISTRY

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config_schema import ExperimentConfig


def _config_metric_names(config: ExperimentConfig) -> tuple[str, ...]:
    """Pull the FA / NBI response metric names out of an ExperimentConfig.

    Uses ``evaluation.raw_metrics`` (the canonical FA input list) and
    falls back to the legacy binary defaults when absent.
    """
    metrics = tuple(config.evaluation.raw_metrics)
    if metrics:
        return metrics
    return (
        "Accuracy_Mean",
        "Precision_Mean",
        "Recall_Mean",
        "Specificity_Mean",
        "Time_MeanFold",
    )


def _resolve_dataset_id(config: ExperimentConfig) -> str:
    """Best-effort dataset-id resolution.

    Strategy:
    1. If the YAML's ``experiment.name`` ends with a known registry id
       (e.g. ``article_3vrf_dry_bean_appendix`` -> ``dry_bean``), use it.
    2. Otherwise, match the trailing component of ``dataset.path``
       (e.g. ``data/source/dry_bean/processed/dry_bean.csv`` -> ``dry_bean``).
    3. Otherwise, raise ``KeyError``.
    """
    name = config.experiment.name.lower()
    for did in REGISTRY:
        if name.endswith(did) or f"_{did}_" in f"_{name}_":
            return did
    parts = [p.lower() for p in str(config.dataset.path).split("/")]
    for did in REGISTRY:
        if did in parts:
            return did
    raise KeyError(
        f"could not infer dataset_id from experiment.name={name!r} or "
        f"dataset.path={config.dataset.path!r}"
    )


def validate_task_metric_compatibility(
    config: ExperimentConfig,
    *,
    dataset_id: str | None = None,
    dataset_metadata: Any = None,
) -> str:
    """Validate that ``config``'s FA metric set is compatible with the
    task type of the resolved dataset. Returns the resolved dataset id.

    Raises :class:`MultiClassNotConfiguredError` if a multiclass dataset
    is paired with the binary response defaults.

    ``dataset_id`` and ``dataset_metadata`` are optional overrides; if
    neither is supplied, the dataset id is inferred from ``config``.
    """
    if dataset_metadata is not None:
        did = str(dataset_metadata.dataset_id)
        task = str(dataset_metadata.task_type)
    else:
        did = dataset_id or _resolve_dataset_id(config)
        if did not in REGISTRY:
            raise KeyError(f"unknown dataset_id {did!r}")
        task = REGISTRY[did].task_type

    fa_metrics = _config_metric_names(config)
    assert_metric_set_compatible_with_task(
        dataset_id=did,
        task=task,  # type: ignore[arg-type]
        fa_metrics=fa_metrics,
    )
    return did


__all__ = [
    "MultiClassNotConfiguredError",
    "validate_task_metric_compatibility",
]
