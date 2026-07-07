#!/usr/bin/env python3
"""PCO213 Santander mixture-ensemble study — end-to-end runner.

Stages (run selectively with --stage, default 'all'):
  data     acquire train.csv (CLI -> public mirror -> manual instructions),
           validate official invariants, persist metadata;
  bench    microbenchmark (small sample, 1 fold per model) and pre-registered
           execution-mode decision with a 2h logical hard stop;
  oof      external 80/20 holdout + RepeatedStratifiedKFold(5, 2) OOF
           probability matrices, cached as .npz; holdout matrix Q;
  design   evaluate the 21-run mixture design + 25 Dirichlet validation
           points + baselines on the OOF matrices;
  scheffe  fit Scheffé quadratic metamodels (per repeat + pooled),
           external validation, coefficient stability table;
  optimize direct SLSQP / Dirichlet-scan optima, metamodel optimum,
           final holdout confirmation of every method;
  figures  produce the study figures from persisted artifacts.

Example:
  .venv-pco213/bin/python scripts/pco213_run_santander_study.py \
      --mode final_2h --max-runtime-minutes 120 \
      --output-dir experiments/pco213/santander
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mixens import base_models, data as data_mod

RAW_DIR = REPO / "data" / "pco213" / "raw" / "santander"
N_VALIDATION_POINTS = 25
METRICS_MODELED = ["log_loss", "roc_auc", "brier"]


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float))


def _read_json(path: Path):
    return json.loads(path.read_text())


# ---------------------------------------------------------------- stages


def stage_data(args) -> dict:
    train_path, source = data_mod.acquire(RAW_DIR)
    X, y, meta = data_mod.load_santander(
        RAW_DIR, mode="full_optional", sample_size=None, random_state=args.random_state
    )
    info = data_mod.meta_dict(meta) | {"train_path": str(train_path), "acquire_source": source}
    _write_json(info, args.out / "data_meta.json")
    print(f"[data] source={source} rows={meta.n_rows} dupes_dropped={meta.n_duplicates_dropped} "
          f"pos_rate={meta.pos_rate:.5f}")
    return info


def stage_bench(args) -> dict:
    X, y, _ = data_mod.load_santander(
        RAW_DIR, mode="fast", sample_size=25_000, random_state=args.random_state
    )
    bench = base_models.microbenchmark(X, y, n_bench=20_000, random_state=args.random_state)
    bench.to_csv(args.out / "microbenchmark.csv", index=False)
    n_available = _read_json(args.out / "data_meta.json")["n_rows"]
    decision = base_models.decide_execution(
        bench,
        requested_mode=args.mode,
        n_available=n_available,
        max_runtime_minutes=args.max_runtime_minutes,
    )
    _write_json(decision, args.out / "execution_decision.json")
    print(f"[bench] mode={decision['mode']} third_slot={decision['third_slot']} "
          f"n_rows={decision['n_rows']} est={decision['estimated_minutes']} min "
          f"(budget {args.max_runtime_minutes})")
    return decision


def stage_oof(args) -> None:
    from mixens import ensemble_eval as ee

    decision = _read_json(args.out / "execution_decision.json")
    X, y, meta = data_mod.load_santander(
        RAW_DIR,
        mode=decision["mode"],
        sample_size=args.sample_size or decision["n_rows"],
        random_state=args.random_state,
    )
    X_tr, X_te, y_tr, y_te = data_mod.external_split(X, y, random_state=args.random_state)
    t0 = time.perf_counter()
    oof = base_models.generate_oof(
        X_tr, y_tr,
        third_slot=decision["third_slot"],
        random_state=args.random_state,
    )
    Q, refit_seconds = base_models.fit_full_and_predict_holdout(
        X_tr, y_tr, X_te,
        third_slot=decision["third_slot"],
        random_state=args.random_state,
    )
    wall = time.perf_counter() - t0
    np.savez_compressed(
        args.out / "oof.npz",
        P0=oof.P[0], P1=oof.P[1],
        fold_ids0=oof.fold_ids[0], fold_ids1=oof.fold_ids[1],
        y_train=y_tr.to_numpy(), Q=Q, y_test=y_te.to_numpy(),
        model_names=np.array(oof.model_names),
    )
    _write_json(
        {
            "wall_seconds": wall,
            "fit_seconds": oof.fit_seconds,
            "predict_seconds": oof.predict_seconds,
            "refit_seconds": refit_seconds,
            "n_train": len(y_tr), "n_test": len(y_te),
            "sample_meta": data_mod.meta_dict(meta),
        },
        args.out / "oof_timings.json",
    )
    # per-fold, per-model OOF metrics (10 folds each) for the boxplot figure
    rows = []
    for r, (P, fids) in enumerate(zip(oof.P, oof.fold_ids)):
        for f in range(oof.n_splits):
            m = fids == f
            for j, name in enumerate(oof.model_names):
                rows.append({"model": name, "repeat": r, "fold": f,
                             **ee.score_probs(y_tr.to_numpy()[m], P[m, j])})
    pd.DataFrame(rows).to_csv(args.out / "fold_metrics_models.csv", index=False)
    print(f"[oof] {len(y_tr)} train / {len(y_te)} holdout — wall {wall/60:.1f} min")


def _load_oof(args):
    z = np.load(args.out / "oof.npz", allow_pickle=False)
    P_list = [z["P0"], z["P1"]]
    return z, P_list, z["y_train"], z["Q"], z["y_test"], [str(s) for s in z["model_names"]]


def stage_design(args) -> None:
    from mixens import ensemble_eval as ee
    from mixens.diagnostics import matrix_diagnostics
    from mixens.mixture_design import build_study_design, sample_dirichlet

    _, P_list, y_tr, _, _, names = _load_oof(args)
    M = P_list[0].shape[1]
    W_design = build_study_design(M)
    W_val = sample_dirichlet(M, N_VALIDATION_POINTS, random_state=args.random_state + 1)
    np.savez(args.out / "design_points.npz", W_design=W_design, W_val=W_val,
             model_names=np.array(names))
    ee.evaluate_design(W_design, P_list, y_tr).to_csv(args.out / "design_eval.csv", index=False)
    ee.evaluate_design(W_val, P_list, y_tr).to_csv(args.out / "validation_eval.csv", index=False)
    _write_json(
        {"design_runs": int(W_design.shape[0]), "validation_points": int(W_val.shape[0]),
         "linear_matrix_diagnostics": matrix_diagnostics(W_design)},
        args.out / "design_meta.json",
    )
    print(f"[design] {W_design.shape[0]} design runs + {W_val.shape[0]} validation points "
          f"evaluated on {len(P_list)} repeats")


def stage_scheffe(args) -> None:
    from mixens.scheffe import MixtureScheffeModel, external_validation, summarize_coefficients

    _, P_list, y_tr, _, _, names = _load_oof(args)
    z = np.load(args.out / "design_points.npz", allow_pickle=False)
    W_design, W_val = z["W_design"], z["W_val"]
    design_eval = pd.read_csv(args.out / "design_eval.csv")
    val_eval = pd.read_csv(args.out / "validation_eval.csv")
    comp = [f"w_{n}" for n in names]

    report: dict = {"component_names": comp}
    for metric in METRICS_MODELED:
        per_repeat = []
        for r in range(len(P_list)):
            sub = design_eval[design_eval["repeat"] == r].sort_values("point")
            df = pd.DataFrame(W_design, columns=comp)
            m = MixtureScheffeModel.fit(df, sub[metric].reset_index(drop=True),
                                        component_names=comp, order="quadratic")
            per_repeat.append(m)
        # pooled fit (both repeats stacked -> replicated design, 42 obs)
        pooled_y = design_eval.sort_values(["repeat", "point"])[metric].reset_index(drop=True)
        pooled_W = pd.DataFrame(np.vstack([W_design] * len(P_list)), columns=comp)
        pooled = MixtureScheffeModel.fit(pooled_W, pooled_y, component_names=comp,
                                         order="quadratic")
        val_y = val_eval.groupby("point")[metric].mean().to_numpy()
        ext = external_validation(pooled, W_val, val_y)
        coef_table = summarize_coefficients(per_repeat)
        coef_table.to_csv(args.out / f"scheffe_coefficients_{metric}.csv", index=False)
        report[metric] = {
            "pooled_r2": pooled.fit_report.r2,
            "pooled_r2_adj": pooled.fit_report.r2_adj,
            "condition_number": pooled.fit_report.condition_number,
            "external_validation": ext,
            "pooled_terms": list(pooled.terms),
            "pooled_coefficients": list(pooled.coefficients),
        }
    _write_json(report, args.out / "scheffe_report.json")
    print("[scheffe] fitted per-repeat + pooled quadratic metamodels for", METRICS_MODELED)


def _rebuild_pooled(args, metric: str):
    from mixens.scheffe import MixtureScheffeModel

    rep = _read_json(args.out / "scheffe_report.json")
    comp = rep["component_names"]
    m = MixtureScheffeModel(
        component_names=tuple(comp),
        terms=tuple(rep[metric]["pooled_terms"]),
        coefficients=tuple(rep[metric]["pooled_coefficients"]),
        fit_report=None,  # type: ignore[arg-type]
    )
    return m


def stage_optimize(args) -> None:
    from mixens import ensemble_eval as ee
    from mixens import optimize as opt

    _, P_list, y_tr, Q, y_te, names = _load_oof(args)
    M = len(names)
    P_all = np.vstack(P_list)                      # 10 OOF rounds stacked
    y_all = np.tile(y_tr, len(P_list))

    w_direct = opt.direct_logloss_optimum(P_all, y_all, random_state=args.random_state)
    scheffe_ll = _rebuild_pooled(args, "log_loss")
    w_meta = opt.metamodel_optimum(scheffe_ll, maximize=False, random_state=args.random_state)
    scheffe_auc = _rebuild_pooled(args, "roc_auc")
    w_meta_auc = opt.metamodel_optimum(scheffe_auc, maximize=True, random_state=args.random_state)
    w_scan_auc, scan_auc = opt.dirichlet_scan_auc(P_all, y_all, n=10_000,
                                                  random_state=args.random_state)
    j_best = ee.best_single_index(P_list, y_tr)
    stacker = ee.fit_stacking(P_all, y_all)

    methods: dict[str, dict] = {
        "best_single": {"oof": P_all[:, j_best], "hold": Q[:, j_best],
                        "detail": {"model": names[j_best]}},
        "uniform_voting": {"w": ee.uniform_weights(M)},
        "stacking_lr": {"oof": stacker.predict_proba(P_all)[:, 1],
                        "hold": stacker.predict_proba(Q)[:, 1],
                        "detail": {"coef": stacker.coef_[0].tolist(),
                                   "intercept": float(stacker.intercept_[0])}},
        "slsqp_direct_logloss": {"w": w_direct},
        "scheffe_optimum_logloss": {"w": w_meta},
        "scheffe_optimum_auc": {"w": w_meta_auc},
        "dirichlet_scan_auc": {"w": w_scan_auc, "detail": {"oof_auc": scan_auc}},
    }
    rows = []
    weights_out: dict[str, list] = {}
    for name, spec in methods.items():
        if "w" in spec:
            w = np.asarray(spec["w"], dtype=float)
            oof_p, hold_p = P_all @ w, Q @ w
            weights_out[name] = w.tolist()
        else:
            oof_p, hold_p = spec["oof"], spec["hold"]
        thr = ee.best_f1_threshold(y_all, oof_p)     # threshold chosen on OOF only
        rows.append({"method": name,
                     **{f"oof_{k}": v for k, v in ee.score_probs(y_all, oof_p).items()},
                     **ee.score_probs(y_te, hold_p, threshold=thr)})
    results = pd.DataFrame(rows)
    results.to_csv(args.out / "holdout_results.csv", index=False)
    _write_json({"weights": weights_out,
                 "best_single_model": names[j_best],
                 "gap_scheffe_vs_direct_logloss_oof":
                     float(abs(results.set_index('method').loc['scheffe_optimum_logloss', 'oof_log_loss']
                               - results.set_index('method').loc['slsqp_direct_logloss', 'oof_log_loss']))},
                args.out / "optimization_report.json")
    print("[optimize] holdout confirmation written:",
          results[["method", "roc_auc", "log_loss"]].to_string(index=False))


def stage_figures(args) -> None:
    from mixens import plots

    figdir = REPO / "figures" / "pco213"
    _, P_list, y_tr, _, _, names = _load_oof(args)
    z = np.load(args.out / "design_points.npz", allow_pickle=False)
    W_design, W_val = z["W_design"], z["W_val"]
    design_eval = pd.read_csv(args.out / "design_eval.csv")
    val_eval = pd.read_csv(args.out / "validation_eval.csv")
    fold_metrics = pd.read_csv(args.out / "fold_metrics_models.csv")
    results = pd.read_csv(args.out / "holdout_results.csv")

    made = [plots.fig_model_fold_boxplots(fold_metrics, figdir / "fig2_base_models_auc.png")]

    scheffe_auc = _rebuild_pooled(args, "roc_auc")
    w_report = _read_json(args.out / "optimization_report.json")["weights"]
    mean_w = np.mean([w_report[k] for k in ("slsqp_direct_logloss", "dirichlet_scan_auc")], axis=0)
    top3 = tuple(int(i) for i in np.argsort(mean_w)[::-1][:3])
    made.append(plots.fig_ternary_contour(
        scheffe_auc, top3, figdir / "fig3_ternary_auc.png",
        title="AUC prevista (Scheffé) — face dos 3 componentes dominantes",
        design_points=W_design))

    pooled_ll = _rebuild_pooled(args, "log_loss")
    obs_d = design_eval.groupby("point")["log_loss"].mean().to_numpy()
    obs_v = val_eval.groupby("point")["log_loss"].mean().to_numpy()
    made.append(plots.fig_pred_vs_obs(
        obs_d, pooled_ll.predict_weights(W_design),
        obs_v, pooled_ll.predict_weights(W_val),
        figdir / "fig4_pred_vs_obs_logloss.png"))

    coef = pd.read_csv(args.out / "scheffe_coefficients_roc_auc.csv")
    made.append(plots.fig_coefficients(coef, figdir / "fig5_scheffe_coefficients_auc.png",
                                       title="Coeficientes de Scheffé — resposta ROC-AUC"))
    made.append(plots.fig_holdout_comparison(results, figdir / "fig6_holdout_comparison.png"))
    print("[figures]", *[str(p) for p in made], sep="\n  ")


STAGES = {
    "data": stage_data,
    "bench": stage_bench,
    "oof": stage_oof,
    "design": stage_design,
    "scheffe": stage_scheffe,
    "optimize": stage_optimize,
    "figures": stage_figures,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--mode", default="final_2h", choices=list(data_mod.SAMPLE_CAPS))
    ap.add_argument("--max-runtime-minutes", type=float, default=120.0)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output-dir", default=str(REPO / "experiments" / "pco213" / "santander"))
    args = ap.parse_args()
    args.out = Path(args.output_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    for s in stages:
        STAGES[s](args)
    total = time.perf_counter() - t0
    print(f"[done] stages={stages} total wall {total/60:.1f} min")
    stamp = {"stages": stages, "wall_minutes": round(total / 60.0, 2)}
    _write_json(stamp, args.out / f"run_{'_'.join(stages[:1])}_timing.json")


if __name__ == "__main__":
    main()
