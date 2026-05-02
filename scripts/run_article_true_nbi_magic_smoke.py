#!/usr/bin/env python
"""Article-track smoke: DOE -> FA -> RSM -> true N-objective NBI -> confirmation
-> conditional MBPA on UCI MAGIC, XGBoost, n_replicas=1, q=2.

Validates that every stage of the article-track pipeline composes cleanly
on real cached data before the 12-dataset campaign runs. Reduced
parameters (q=2, simplex_lattice {2, 10}, 1 replica) keep runtime
under 10 minutes. Uses ``nbi_core.run_nbi`` (true NBI), never the
legacy weighted-sum scalarization.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doe_xgb.config import DEFAULT_BOUNDS, INT_PARAMS, PARAM_NAMES  # noqa: E402
from doe_xgb.config_schema import load_config  # noqa: E402
from doe_xgb.datasets import load_magic, validate_task_metric_compatibility  # noqa: E402
from doe_xgb.evaluation import evaluate_xgb_cv  # noqa: E402
from doe_xgb.factor_model import FactorModelSpec, fit_factor_model  # noqa: E402
from doe_xgb.model_families import ProcessQuadraticRSM  # noqa: E402
from doe_xgb.nbi_core import NBIConfig, run_nbi  # noqa: E402
from doe_xgb.post_optimization import MBPASpec, run_mbpa  # noqa: E402
from doe_xgb.selection import SelectionRule, select  # noqa: E402
from doe_xgb.simplex import generate_simplex_lattice  # noqa: E402

CONFIG_PATH = REPO / "configs" / "smoke_article_true_nbi_magic.yaml"
OUT_JSON = REPO / "experiments" / "_v1_smoke" / "article_true_nbi_magic_smoke.json"
OUT_MD = REPO / "experiments" / "_v1_smoke" / "article_true_nbi_magic_smoke.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stage(label: str):
    t0 = time.perf_counter()

    def _end(extra: dict | None = None) -> dict:
        rec = {"stage": label, "wall_seconds": round(time.perf_counter() - t0, 3)}
        if extra:
            rec.update(extra)
        return rec

    return _end


def _cast_int_params(params: dict[str, float]) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


def _make_surrogate_callable(model: ProcessQuadraticRSM, factor_names: list[str]):
    """Return f(x_uncoded_array) -> float using the RSM predictor."""

    def f(x: np.ndarray) -> float:
        params = {name: float(val) for name, val in zip(factor_names, np.asarray(x).tolist(), strict=True)}
        return float(model.predict_row(params))

    return f


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def main() -> int:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    overall_t0 = time.perf_counter()
    stage_records: list[dict] = []
    payload: dict = {
        "smoke": "article_true_nbi_magic",
        "started_at": started_at,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config_path": str(CONFIG_PATH.relative_to(REPO)),
    }

    try:
        # ------------------------------------------------------------------
        # 0. Config + guardrail.
        # ------------------------------------------------------------------
        end = _stage("config")
        cfg = load_config(CONFIG_PATH)
        resolved_id = validate_task_metric_compatibility(cfg)
        payload["resolved_dataset_id"] = resolved_id
        payload["n_replicas"] = cfg.experiment.n_replicas
        payload["nbi_q"] = cfg.nbi.weights.q
        payload["nbi_m"] = cfg.nbi.weights.m
        stage_records.append(end({"resolved_dataset_id": resolved_id}))

        # ------------------------------------------------------------------
        # 1. Load MAGIC.
        # ------------------------------------------------------------------
        end = _stage("load_dataset")
        ds = load_magic()
        X_full, y_full = ds.X, ds.y
        # Encode any non-numeric column to integer codes for XGBoost.
        for col in X_full.columns:
            if not pd.api.types.is_numeric_dtype(X_full[col]):
                X_full[col] = pd.Categorical(X_full[col]).codes.astype("int64")
        stage_records.append(end({
            "n_rows": int(len(X_full)),
            "n_features": int(X_full.shape[1]),
            "task_type": ds.metadata.task_type,
        }))

        # ------------------------------------------------------------------
        # 2. DOE evaluation.
        # ------------------------------------------------------------------
        end = _stage("doe")
        design_path = REPO / cfg.design.external_path
        design = pd.read_csv(design_path, sep=";", decimal=",")
        design.columns = [c.strip() for c in design.columns]
        # Keep only the hyperparameter columns we care about.
        missing = [p for p in PARAM_NAMES if p not in design.columns]
        if missing:
            raise RuntimeError(f"design is missing hyperparameter columns: {missing}")

        cv_seed = cfg.experiment.seed_base
        kfold = StratifiedKFold(
            n_splits=cfg.evaluation.cv.n_splits,
            shuffle=cfg.evaluation.cv.shuffle,
            random_state=cv_seed,
        )
        X_np = X_full.to_numpy()
        y_np = y_full.to_numpy()

        doe_rows: list[dict] = []
        for i, row in design.iterrows():
            params = {p: float(row[p]) for p in PARAM_NAMES}
            casted = _cast_int_params(params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev = evaluate_xgb_cv(
                    casted, X_np, y_np, kfold,
                    seed=cv_seed, n_jobs=1, tree_method="hist",
                    eval_metric="logloss", task_type="binary",
                )
            row_out = {**casted, **ev.metrics, "Time_MeanFold": ev.time_mean_fold}
            doe_rows.append(row_out)
        doe_df = pd.DataFrame(doe_rows)
        stage_records.append(end({
            "n_design_rows": int(len(design)),
            "doe_columns": list(doe_df.columns),
            "binary_keys_present": all(k in doe_df.columns for k in (
                "Accuracy_Mean", "Precision_Mean", "Recall_Mean", "Specificity_Mean",
            )),
        }))

        # ------------------------------------------------------------------
        # 3. Factor analysis.
        # ------------------------------------------------------------------
        end = _stage("factor_model")
        fa_metrics = list(cfg.evaluation.raw_metrics)
        spec = FactorModelSpec(
            mode=cfg.factor_model.mode,
            n_factors=cfg.factor_model.n_factors,
            rotation=cfg.factor_model.rotation,
            standardize=cfg.factor_model.standardize,
        )
        fa_res = fit_factor_model(doe_df, fa_metrics, spec)
        scored_df = doe_df.join(fa_res.scores)
        loadings_shape = list(fa_res.loadings.shape)
        stage_records.append(end({
            "n_factors": int(fa_res.scores.shape[1]),
            "loadings_shape": loadings_shape,
            "explained_variance": [round(float(v), 4) for v in fa_res.explained_variance.tolist()],
            "cumulative_variance": [round(float(v), 4) for v in fa_res.cumulative_variance.tolist()],
            "construct_map": {
                k: list(v) for k, v in fa_res.diagnostics.get("construct_map", {}).items()
            },
        }))

        # ------------------------------------------------------------------
        # 4. RSM (one ProcessQuadraticRSM per factor score).
        # ------------------------------------------------------------------
        end = _stage("rsm")
        rsm_models: list[ProcessQuadraticRSM] = []
        rsm_reports = []
        backward = None  # smoke: no elimination so we keep deterministic shape.
        for k in range(int(fa_res.scores.shape[1])):
            target_col = f"FACTOR{k + 1}_SCORE"
            model = ProcessQuadraticRSM.fit(
                scored_df,
                scored_df[target_col],
                factor_names=PARAM_NAMES,
                order="quadratic",
                backward=backward,
            )
            rsm_models.append(model)
            rsm_reports.append({
                "factor": target_col,
                "n_terms": len(model.terms),
                "r2": round(model.fit_report.r2, 4),
                "r2_adj": round(model.fit_report.r2_adj, 4),
                "rank": int(model.fit_report.rank),
                "condition_number": round(model.fit_report.condition_number, 2),
            })
        stage_records.append(end({"models": rsm_reports}))

        # ------------------------------------------------------------------
        # 5. True N-objective NBI.
        # ------------------------------------------------------------------
        end = _stage("nbi_core")
        # Surrogate callables: NBI minimizes; the article-track wraps each
        # objective so "minimize" matches the underlying canonical-min
        # convention. For VRF1 (quality, maximize) we negate; for VRF2
        # (cost, minimize) we keep the sign.
        primaries = [s for s in cfg.objectives.specs if s.role.value == "primary_nbi"]

        def _wrap(model: ProcessQuadraticRSM, direction: str):
            base = _make_surrogate_callable(model, list(PARAM_NAMES))

            def f(x: np.ndarray) -> float:
                v = base(x)
                return -v if direction == "maximize" else v

            return f

        surrogates = []
        for spec_obj, model in zip(primaries, rsm_models, strict=True):
            surrogates.append(_wrap(model, spec_obj.direction.value))

        bounds_arr = np.array([DEFAULT_BOUNDS[p] for p in PARAM_NAMES], dtype=float)
        nbi_cfg = NBIConfig(
            objective_count=cfg.nbi.weights.q,
            bounds=bounds_arr,
            n_starts=cfg.nbi.n_starts,
            seed=cfg.experiment.seed_base,
            quasi_normal=cfg.nbi.quasi_normal,
            maxiter=400,
        )
        weights = generate_simplex_lattice(cfg.nbi.weights.q, cfg.nbi.weights.m)
        nbi_run = run_nbi(surrogates, weights, nbi_cfg)
        residuals = [c.residual_norm for c in nbi_run.candidates]
        ts = [c.t for c in nbi_run.candidates]
        stage_records.append(end({
            "n_subproblems": len(nbi_run.candidates),
            "n_weights": int(weights.shape[0]),
            "max_residual_norm": float(max(residuals)),
            "median_residual_norm": float(np.median(residuals)),
            "p95_residual_norm": float(np.percentile(residuals, 95)),
            "min_t": float(min(ts)),
            "max_t": float(max(ts)),
            "n_success": int(sum(1 for c in nbi_run.candidates if c.success)),
            "anchors_utopia": [round(float(v), 4) for v in nbi_run.anchors.utopia.tolist()],
            "anchors_pseudo_nadir": [round(float(v), 4) for v in nbi_run.anchors.pseudo_nadir.tolist()],
        }))

        # ------------------------------------------------------------------
        # 6. Confirmation: re-evaluate every NBI candidate on real CV.
        # ------------------------------------------------------------------
        end = _stage("confirmation")
        confirm_rows: list[dict] = []
        for c in nbi_run.candidates:
            params = {p: float(c.x[i]) for i, p in enumerate(PARAM_NAMES)}
            casted = _cast_int_params(params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ev = evaluate_xgb_cv(
                    casted, X_np, y_np, kfold,
                    seed=cv_seed, n_jobs=1, tree_method="hist",
                    eval_metric="logloss", task_type="binary",
                )
            row_out = {
                "beta": [round(float(b), 4) for b in c.beta.tolist()],
                "t": float(c.t),
                "residual_norm": float(c.residual_norm),
                **{k: float(v) for k, v in casted.items()},
                **{k: float(v) for k, v in ev.metrics.items()},
                "Time_MeanFold": float(ev.time_mean_fold),
            }
            confirm_rows.append(row_out)
        confirm_df = pd.DataFrame(confirm_rows)
        # Selection: distance to utopia in canonicalized space (maximize
        # accuracy proxy = -Accuracy_Mean; minimize Time_MeanFold).
        F_canon = np.column_stack([
            -confirm_df["Accuracy_Mean"].to_numpy(),
            confirm_df["Time_MeanFold"].to_numpy(),
        ])
        idx, info = select(F_canon, SelectionRule.DISTANCE_TO_UTOPIA)
        chosen = confirm_df.iloc[int(idx)].to_dict()
        stage_records.append(end({
            "n_candidates_confirmed": int(len(confirm_df)),
            "selection_rule": info["rule"],
            "selection_distance": info.get("distance"),
            "chosen_index": int(idx),
            "chosen_metrics": {
                k: float(chosen[k])
                for k in ("Accuracy_Mean", "Precision_Mean", "Recall_Mean",
                          "Specificity_Mean", "Time_MeanFold")
            },
            "chosen_hyperparameters": {p: chosen[p] for p in PARAM_NAMES},
        }))

        # ------------------------------------------------------------------
        # 7. Conditional MBPA.
        # ------------------------------------------------------------------
        end = _stage("mbpa")
        mbpa_spec = MBPASpec(
            inner_simplex_q=cfg.nbi.weights.q,
            inner_simplex_m=cfg.post_optimization.mbpa.inner_design.m,
            elliptical_radii=tuple(cfg.post_optimization.mbpa.elliptical_constraint.radii),
        )
        mbpa = run_mbpa(nbi_run, mbpa_spec, enabled=cfg.post_optimization.enabled)
        stage_records.append(end({
            "enabled": cfg.post_optimization.enabled,
            "triggered": bool(mbpa.triggered),
            "frontier_diagnostics": {
                "avg_pairwise_distance": float(mbpa.diagnostics.avg_pairwise_distance),
                "unique_nondominated": int(mbpa.diagnostics.unique_nondominated),
                "weight_concentration": float(mbpa.diagnostics.weight_concentration),
                "curvature_score": float(mbpa.diagnostics.curvature_score),
                "spread": float(mbpa.diagnostics.spread),
                "triggers": dict(mbpa.diagnostics.triggers),
            },
            "summary": dict(mbpa.summary),
        }))

        # ------------------------------------------------------------------
        # Done.
        # ------------------------------------------------------------------
        payload["stages"] = stage_records
        payload["ok"] = True
        payload["total_wall_seconds"] = round(time.perf_counter() - overall_t0, 2)
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc(limit=8)
        payload["stages"] = stage_records
        payload["total_wall_seconds"] = round(time.perf_counter() - overall_t0, 2)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Markdown summary.
    md: list[str] = []
    md.append("# Article-track true NBI smoke -- MAGIC + XGBoost\n\n")
    md.append(f"Wrote {OUT_JSON.relative_to(REPO)}.\n\n")
    md.append(f"- ok: **{payload.get('ok')}**\n")
    md.append(f"- total_wall_seconds: {payload.get('total_wall_seconds')}\n")
    md.append(f"- platform: {payload['platform']}\n\n")
    md.append("## Stages\n\n")
    md.append("| Stage | wall_seconds | summary |\n|---|---:|---|\n")
    for s in stage_records:
        summary = ", ".join(
            f"{k}={v}" for k, v in s.items() if k not in ("stage", "wall_seconds")
        )
        if len(summary) > 220:
            summary = summary[:200] + " ..."
        md.append(f"| {s['stage']} | {s['wall_seconds']} | {summary} |\n")
    md.append("\n")
    if not payload.get("ok"):
        md.append("\n## Error\n\n```\n" + payload.get("traceback", "") + "\n```\n")
    OUT_MD.write_text("".join(md), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
