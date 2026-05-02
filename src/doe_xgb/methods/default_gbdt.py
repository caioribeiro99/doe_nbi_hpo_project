"""Adapter for the ``default_gbdt`` control: per-algorithm library defaults
with no HPO. One fit per CV split per replica."""

from __future__ import annotations

from .base import AdapterBase
from .canary import (
    CanaryResult,
    MethodRunContext,
    TaskData,
    aggregate_fold_metrics,
    evaluate_fitted_model,
)


class DefaultGbdtAdapter(AdapterBase):
    method_id = "default_gbdt"
    required_packages = ("xgboost", "lightgbm", "catboost")
    run_status = "smoke_ready"
    notes = (
        "No search; one fit per CV split with library defaults. "
        "Smoke-ready (Commit 30) — runs end-to-end on synthetic and CC18 "
        "tasks once the GBDT library for the target algorithm is installed."
    )

    def run(self, *, task: TaskData, ctx: MethodRunContext) -> CanaryResult:
        from ._gbdt_factory import make_fit_predict

        fit, predict = make_fit_predict(
            ctx.algorithm, params={}, seed=ctx.derived_seed(),
        )
        fold_metrics, runtime = evaluate_fitted_model(
            model_fit=fit, model_predict=predict,
            X=task.X, y=task.y, task_type=task.task_type,
            n_folds=ctx.n_folds, seed=ctx.derived_seed(salt=1),
        )
        return CanaryResult(
            method_id=self.method_id,
            algorithm=ctx.algorithm,
            replica=ctx.replica,
            task_type=task.task_type,
            n_folds=ctx.n_folds,
            fold_metrics=fold_metrics,
            aggregate_metrics=aggregate_fold_metrics(fold_metrics),
            best_config={},  # no HPO performed
            n_configurations_tried=1,
            runtime_seconds=runtime,
            notes="library defaults; no HPO",
        )
