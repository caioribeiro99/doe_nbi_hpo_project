"""Adapter for ``tpe_optuna``: Optuna's TPESampler with default prior."""

from __future__ import annotations

from .base import AdapterBase
from .canary import (
    CANARY_SEARCH_SPACE,
    CanaryResult,
    MethodRunContext,
    TaskData,
    aggregate_fold_metrics,
    evaluate_fitted_model,
)


class TpeOptunaAdapter(AdapterBase):
    method_id = "tpe_optuna"
    required_packages = ("optuna",)
    run_status = "smoke_ready"
    notes = (
        "Akiba et al. 2019. Default-prior TPESampler. "
        "Smoke-ready (Commit 30) over the canary search space; raises "
        "ImportError(clear message) when Optuna is not installed."
    )

    def run(self, *, task: TaskData, ctx: MethodRunContext) -> CanaryResult:
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "optuna is required for tpe_optuna.run(); install with "
                "`pip install -e .[hpo_baselines]`. Original error: "
                f"{exc}"
            ) from exc

        from ._gbdt_factory import make_fit_predict

        space = CANARY_SEARCH_SPACE[ctx.algorithm]
        n_evals = max(1, int(ctx.max_evaluations))

        def suggest(trial) -> dict:
            cfg: dict = {}
            for name, (lo, hi) in space.items():
                if isinstance(lo, int) and isinstance(hi, int):
                    cfg[name] = trial.suggest_int(name, lo, hi)
                else:
                    cfg[name] = trial.suggest_float(name, float(lo), float(hi))
            return cfg

        all_configs: list[dict] = []
        total_runtime = [0.0]
        # Track the best fold metrics so we can report the headline
        # without re-running the best trial.
        state: dict = {"best_score": -float("inf"), "best_fold_metrics": []}

        def objective(trial) -> float:
            cfg = suggest(trial)
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
            total_runtime[0] += runtime
            if score > state["best_score"]:
                state["best_score"] = score
                state["best_fold_metrics"] = fold_metrics
            return score

        sampler = optuna.samplers.TPESampler(seed=ctx.derived_seed(salt=3))
        study = optuna.create_study(direction="maximize", sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_evals, show_progress_bar=False)

        best_trial = study.best_trial
        return CanaryResult(
            method_id=self.method_id,
            algorithm=ctx.algorithm,
            replica=ctx.replica,
            task_type=task.task_type,
            n_folds=ctx.n_folds,
            fold_metrics=state["best_fold_metrics"],
            aggregate_metrics=aggregate_fold_metrics(state["best_fold_metrics"]),
            best_config=dict(best_trial.params),
            n_configurations_tried=n_evals,
            runtime_seconds=total_runtime[0],
            extra={
                "all_configs": all_configs,
                "best_score": state["best_score"],
                "optuna_best_value": float(best_trial.value),
            },
            notes=f"optuna TPESampler over canary space; n_trials={n_evals}",
        )
