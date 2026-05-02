"""Adapter for the ``random_search`` classical baseline."""

from __future__ import annotations

import numpy as np

from .base import AdapterBase
from .canary import (
    CanaryResult,
    MethodRunContext,
    TaskData,
    aggregate_fold_metrics,
    evaluate_fitted_model,
    sample_canary_config,
)


class RandomSearchAdapter(AdapterBase):
    method_id = "random_search"
    required_packages = ("scipy", "joblib")
    run_status = "smoke_ready"
    notes = (
        "Bergstra & Bengio 2012. Smoke-ready (Commit 30) over the "
        "canary search space; the full per-algorithm search space lives "
        "in configs/cc18/<algorithm>/random_search/<task>.yaml and is "
        "wired in by the runner via ctx.config_path."
    )

    def run(self, *, task: TaskData, ctx: MethodRunContext) -> CanaryResult:
        from ._gbdt_factory import make_fit_predict

        rng = np.random.default_rng(ctx.derived_seed(salt=2))
        n_evals = max(1, int(ctx.max_evaluations))
        best_score = -float("inf")
        best_cfg: dict | None = None
        best_fold_metrics: list[dict] = []
        all_configs: list[dict] = []
        total_runtime = 0.0
        for _ in range(n_evals):
            cfg = sample_canary_config(rng, ctx.algorithm)
            fit, predict = make_fit_predict(
                ctx.algorithm, params=cfg, seed=ctx.derived_seed(),
            )
            fold_metrics, runtime = evaluate_fitted_model(
                model_fit=fit, model_predict=predict,
                X=task.X, y=task.y, task_type=task.task_type,
                n_folds=ctx.n_folds, seed=ctx.derived_seed(salt=1),
            )
            agg = aggregate_fold_metrics(fold_metrics)
            score = agg.get("accuracy", agg.get("balanced_accuracy", 0.0))
            all_configs.append({"config": cfg, "score": score})
            total_runtime += runtime
            if score > best_score:
                best_score = score
                best_cfg = cfg
                best_fold_metrics = fold_metrics

        return CanaryResult(
            method_id=self.method_id,
            algorithm=ctx.algorithm,
            replica=ctx.replica,
            task_type=task.task_type,
            n_folds=ctx.n_folds,
            fold_metrics=best_fold_metrics,
            aggregate_metrics=aggregate_fold_metrics(best_fold_metrics),
            best_config=best_cfg,
            n_configurations_tried=n_evals,
            runtime_seconds=total_runtime,
            extra={"all_configs": all_configs, "best_score": best_score},
            notes=f"random_search over canary space; n_evals={n_evals}",
        )
