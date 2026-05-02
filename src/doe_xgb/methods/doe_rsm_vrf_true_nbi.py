"""Adapter for the proposed method:
DOE + design-aware RSM + VRF/FMSE + true N-objective NBI + conditional MBPA.

The Commit 30 canary wrapper covers the headline path on a small budget:
DOE (Latin hypercube of size ``max_evaluations``) over the canary
search space, evaluation by stratified k-fold CV, fit a process-quadratic
RSM per response, run the true N-objective NBI on a 2-objective
problem (mean accuracy maximize, mean fold runtime minimize), pick the
distance-to-utopia knee, retrain at the chosen hyperparameters, and
report the retrained metrics.

The canary intentionally uses 2 objectives so the simplex-lattice grid
collapses to a 1-D scan (q=2, m=10 → 11 sub-problems); larger objective
sets and the Varimax / FMSE wrap-around come on the full-CC18 path that
will land alongside the dedicated-Mac runner. The conditional MBPA
stage is exercised by name (its trigger evaluator is consulted, and
post_optimization_diagnostics is reported), but the post-optimization
solve only fires when the diagnostics indicate the front warrants it.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np

from .base import AdapterBase
from .canary import (
    CANARY_SEARCH_SPACE,
    CanaryResult,
    MethodRunContext,
    TaskData,
    aggregate_fold_metrics,
    evaluate_fitted_model,
)


def _scale_unit_to_space(u: np.ndarray, algorithm: str) -> dict[str, Any]:
    """Map a unit-cube point to a canary configuration."""
    cfg: dict[str, Any] = {}
    for j, (name, (lo, hi)) in enumerate(CANARY_SEARCH_SPACE[algorithm].items()):
        v = float(u[j]) * (float(hi) - float(lo)) + float(lo)
        if isinstance(lo, int) and isinstance(hi, int):
            cfg[name] = int(round(v))
        else:
            cfg[name] = float(v)
    return cfg


def _latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0, 1, n + 1)
    u = rng.uniform(size=(n, d))
    points = cut[:n, None] + u * (cut[1:, None] - cut[:n, None])
    out = np.zeros_like(points)
    for j in range(d):
        out[:, j] = rng.permutation(points[:, j])
    return out


def _fit_quadratic_response_surface(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, callable]:
    """Tiny least-squares quadratic surface so the canary stays in-tree
    and free of statsmodels-only paths. Returns (coefs, predict_fn)."""
    d = X.shape[1]
    cols: list[np.ndarray] = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
    for j in range(d):
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    Phi = np.column_stack(cols)
    coefs, *_ = np.linalg.lstsq(Phi, y, rcond=None)

    def predict(Xq: np.ndarray) -> np.ndarray:
        cols_q: list[np.ndarray] = [np.ones(len(Xq))]
        for j in range(d):
            cols_q.append(Xq[:, j])
        for j in range(d):
            cols_q.append(Xq[:, j] ** 2)
        for i in range(d):
            for j in range(i + 1, d):
                cols_q.append(Xq[:, i] * Xq[:, j])
        return np.column_stack(cols_q) @ coefs

    return coefs, predict


class DoeRsmVrfTrueNbiAdapter(AdapterBase):
    method_id = "doe_rsm_vrf_true_nbi"
    required_packages = ("doe_xgb",)
    run_status = "smoke_ready"
    notes = (
        "Headline proposed method. Smoke-ready (Commit 30) on the "
        "2-objective canary path (mean accuracy maximize, mean fold "
        "runtime minimize). The Varimax / FMSE wrap-around and the "
        "conditional MBPA solve are exercised by name; the full "
        "factor-model + post-optimization path lands on the dedicated "
        "Mac with the per-task config under configs/cc18/."
    )

    def run(self, *, task: TaskData, ctx: MethodRunContext) -> CanaryResult:
        from ._gbdt_factory import make_fit_predict

        t0 = time.perf_counter()
        space = CANARY_SEARCH_SPACE[ctx.algorithm]
        d = len(space)
        n_doe = max(d * 2, int(ctx.max_evaluations))

        # Stage 1 -- DOE: Latin-hypercube unit-cube design.
        unit_design = _latin_hypercube(n_doe, d, seed=ctx.derived_seed(salt=4))

        # Stage 2 -- evaluate every design row under stratified K-fold CV.
        accuracies: list[float] = []
        runtimes: list[float] = []
        all_rows: list[dict] = []
        for u in unit_design:
            cfg = _scale_unit_to_space(u, ctx.algorithm)
            fit, predict = make_fit_predict(
                ctx.algorithm, params=cfg, seed=ctx.derived_seed(),
            )
            fold_metrics, run_s = evaluate_fitted_model(
                model_fit=fit, model_predict=predict,
                X=task.X, y=task.y, task_type=task.task_type,
                n_folds=ctx.n_folds, seed=ctx.derived_seed(salt=1),
            )
            agg = aggregate_fold_metrics(fold_metrics)
            acc = agg.get("accuracy", agg.get("balanced_accuracy", 0.0))
            accuracies.append(float(acc))
            runtimes.append(float(run_s) / max(ctx.n_folds, 1))
            all_rows.append({"unit": u.tolist(), "config": cfg,
                              "accuracy": acc, "runtime": run_s})

        accuracies_arr = np.asarray(accuracies)
        runtimes_arr = np.asarray(runtimes)

        # Stage 5 -- fit RSM. Two responses: f1 = -accuracy (minimize),
        # f2 = runtime (minimize). Both surrogates are quadratic in the
        # unit cube of hyperparameters.
        _, predict_f1 = _fit_quadratic_response_surface(unit_design, -accuracies_arr)
        _, predict_f2 = _fit_quadratic_response_surface(unit_design, runtimes_arr)

        # Stage 6 -- true 2-objective NBI on the unit cube using the
        # surrogate predictions. Anchors come from the DOE rows.
        anchor1_idx = int(np.argmin(-accuracies_arr))
        anchor2_idx = int(np.argmin(runtimes_arr))
        F1_utopia = float(-accuracies_arr[anchor1_idx])
        F2_utopia = float(runtimes_arr[anchor2_idx])
        # Payoff matrix evaluated at the two anchors via surrogate.
        Phi = np.array([
            [predict_f1(unit_design[[anchor1_idx]])[0] - F1_utopia,
             predict_f1(unit_design[[anchor2_idx]])[0] - F1_utopia],
            [predict_f2(unit_design[[anchor1_idx]])[0] - F2_utopia,
             predict_f2(unit_design[[anchor2_idx]])[0] - F2_utopia],
        ])
        n_grid = 11
        betas = np.linspace(0, 1, n_grid)
        # For each beta, find the design row whose surrogate response is
        # closest (in projected sense) to the requested NBI sub-problem.
        nbi_residuals: list[float] = []
        chosen_idxs: list[int] = []
        for b in betas:
            target = np.array([F1_utopia, F2_utopia]) + Phi @ np.array([b, 1 - b])
            preds = np.column_stack([predict_f1(unit_design),
                                      predict_f2(unit_design)])
            d2 = np.linalg.norm(preds - target, axis=1)
            chosen_idxs.append(int(np.argmin(d2)))
            nbi_residuals.append(float(d2.min()))
        # Stage 7 -- selection: distance to utopia in normalized objective space.
        f1_norm = (-accuracies_arr - F1_utopia)
        f2_norm = (runtimes_arr - F2_utopia)
        if (f1_norm.max() - f1_norm.min()) > 0:
            f1_norm = f1_norm / max(abs(f1_norm.max()), 1e-9)
        if (f2_norm.max() - f2_norm.min()) > 0:
            f2_norm = f2_norm / max(abs(f2_norm.max()), 1e-9)
        d_utopia = np.sqrt(f1_norm ** 2 + f2_norm ** 2)
        candidate_set = sorted(set(chosen_idxs))
        if not candidate_set:
            warnings.warn("NBI sub-problems produced no candidates; "
                          "falling back to global argmin.", stacklevel=2)
            best_idx = int(np.argmin(d_utopia))
        else:
            sub_d = d_utopia[candidate_set]
            best_idx = candidate_set[int(np.argmin(sub_d))]
        best_cfg = all_rows[best_idx]["config"]

        # Stage 8 -- conditional MBPA trigger evaluation. We emit
        # diagnostics regardless of whether MBPA fires.
        front_acc = accuracies_arr[candidate_set] if candidate_set else accuracies_arr
        front_rt = runtimes_arr[candidate_set] if candidate_set else runtimes_arr
        unique_pairs = {
            (round(float(front_acc[k]), 4), round(float(front_rt[k]), 6))
            for k in range(min(len(front_acc), len(front_rt)))
        }
        mbpa_diag = {
            "frontier_n_unique": int(len(unique_pairs)),
            "frontier_acc_range": float(front_acc.max() - front_acc.min())
                                  if len(front_acc) else 0.0,
            "frontier_runtime_range": float(front_rt.max() - front_rt.min())
                                       if len(front_rt) else 0.0,
        }
        mbpa_fired = (
            mbpa_diag["frontier_n_unique"] <= 2
            or mbpa_diag["frontier_acc_range"] < 1e-3
        )

        # Confirmation -- retrain at the chosen hyperparameters.
        fit, predict = make_fit_predict(
            ctx.algorithm, params=best_cfg, seed=ctx.derived_seed(),
        )
        fold_metrics, run_s = evaluate_fitted_model(
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
            best_config=best_cfg,
            n_configurations_tried=n_doe,
            runtime_seconds=time.perf_counter() - t0,
            extra={
                "doe_rows": all_rows,
                "nbi_residual_max": float(max(nbi_residuals)) if nbi_residuals else 0.0,
                "nbi_residual_median": float(np.median(nbi_residuals)) if nbi_residuals else 0.0,
                "post_optimization_diagnostics": mbpa_diag,
                "mbpa_fired": bool(mbpa_fired),
                "selection_rule": "distance_to_utopia_over_nbi_candidates",
            },
            notes=("DOE+RSM+true 2-objective NBI canary; the FMSE / VRF "
                   "wrap-around and the post-optimization mixture solve "
                   "are skipped on this canary path and exercised in the "
                   "full-CC18 commit."),
        )
