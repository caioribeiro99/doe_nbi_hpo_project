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

    # Corrected repeated-CV t-test (Nadeau-Bengio) on per-fold OOF differences:
    # proposed method vs each baseline, on the SAME folds (J = 5 folds x 2 repeats).
    z = np.load(args.out / "oof.npz", allow_pickle=False)
    fold_ids = [z["fold_ids0"], z["fold_ids1"]]
    proba_by_method = {
        "scheffe_optimum_logloss": [P @ w_meta for P in P_list],
        "slsqp_direct_logloss": [P @ w_direct for P in P_list],
        "uniform_voting": [P @ ee.uniform_weights(M) for P in P_list],
        "best_single": [P[:, j_best] for P in P_list],
        "stacking_lr": [stacker.predict_proba(P)[:, 1] for P in P_list],
    }
    stats_out: dict[str, dict] = {}
    for metric in ("roc_auc", "log_loss"):
        folds = {
            name: np.concatenate(
                [ee.per_fold_metric(ps[r], y_tr, fold_ids[r], metric)
                 for r in range(len(P_list))]
            )
            for name, ps in proba_by_method.items()
        }
        ref = folds["scheffe_optimum_logloss"]
        stats_out[metric] = {
            f"scheffe_vs_{other}": ee.corrected_repeated_cv_ttest(ref - folds[other])
            for other in ("uniform_voting", "best_single", "stacking_lr",
                          "slsqp_direct_logloss")
        }
    _write_json(stats_out, args.out / "statistics_corrected_ttest.json")
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


# ---------------------------------------------------------------- postwork
# Optional stages (DoE + RSM + NBI, docs/PCO213/POSTWORK_EXPERIMENT_PLAN.md).
# They only READ the delivered artifacts in --output-dir and WRITE to
# --postwork-output-dir; the frozen course delivery is never modified.

OBJECTIVE_CHOICES = ("auc", "logloss", "cost")


def _require(args, paths: list[Path], suggestion: str) -> bool:
    """Check artifact preconditions; print a clear message when missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        print("[postwork] artefatos necessários ausentes:")
        for p in missing:
            print(f"  - {p}")
        print(f"[postwork] rode antes: {suggestion}")
        return False
    return True


def _load_pooled_oof(args):
    z = np.load(args.out / "oof.npz", allow_pickle=False)
    P_all = np.vstack([z["P0"], z["P1"]])
    y_all = np.tile(z["y_train"], 2)
    names = [str(s) for s in z["model_names"]]
    return P_all, y_all, names


def _compute_costs(args) -> dict:
    """Per-model inference cost (ms per 1k predictions) from saved OOF timings.

    Documented fallback: models are not persisted by the pipeline, so the
    cost is derived from the predict timings recorded during OOF generation
    (each model predicted n_train rows once per repeat). For a fresh direct
    measurement, re-run `--stage oof` (heavy) — never triggered from here.
    """
    t = _read_json(args.out / "oof_timings.json")
    n_predictions = t["n_train"] * 2  # rows predicted once per repeat
    costs = {
        name: 1000.0 * 1000.0 * seconds / n_predictions  # ms per 1k predictions
        for name, seconds in t["predict_seconds"].items()
    }
    return {
        "method": "oof_timings_fallback",
        "note": ("Derived from OOF predict timings (models are not persisted). "
                 "For a direct measurement re-run --stage oof."),
        "n_predictions_basis": n_predictions,
        "cost_ms_per_1k": costs,
    }


def stage_cost(args) -> None:
    if not _require(args, [args.out / "oof_timings.json"],
                    "python scripts/pco213_run_santander_study.py --stage oof"):
        return
    if args.dry_run:
        print(f"[cost][dry-run] leria {args.out/'oof_timings.json'} e escreveria "
              f"{args.postwork_out/'inference_costs.json'} (custo ~0 s)")
        return
    report = _compute_costs(args)
    _write_json(report, args.postwork_out / "inference_costs.json")
    print("[cost] ms/1k predições:",
          {k: round(v, 2) for k, v in report["cost_ms_per_1k"].items()})


def _build_objectives(args, names: list[str]):
    """Objective callables on w (minimization convention) per --objectives."""
    from mixens.nbi import linear_cost

    wanted = [o.strip() for o in args.objectives.split(",") if o.strip()]
    bad = [o for o in wanted if o not in OBJECTIVE_CHOICES]
    if bad or len(wanted) < 2:
        raise SystemExit(f"--objectives deve ter >=2 entre {OBJECTIVE_CHOICES}; recebi {wanted}")
    cost_file = args.postwork_out / "inference_costs.json"
    objectives, labels = [], []
    for o in wanted:
        if o == "auc":
            m = _rebuild_pooled(args, "roc_auc")
            objectives.append(lambda w, m=m: -float(m.predict_weights(np.asarray(w)[None, :])[0]))
            labels.append("neg_auc(metamodelo)")
        elif o == "logloss":
            m = _rebuild_pooled(args, "log_loss")
            objectives.append(lambda w, m=m: float(m.predict_weights(np.asarray(w)[None, :])[0]))
            labels.append("logloss(metamodelo)")
        elif o == "cost":
            if not cost_file.exists():
                print("[nbi] inference_costs.json ausente — computando via stage cost")
                _write_json(_compute_costs(args), cost_file)
            c = _read_json(cost_file)["cost_ms_per_1k"]
            cvec = np.array([c[n] for n in names], dtype=float)
            objectives.append(lambda w, cvec=cvec: linear_cost(w, cvec))
            labels.append("custo_ms_1k(exato,linear)")
    return objectives, labels, wanted


def _real_objective_values(P_all, y_all, names, args, W) -> pd.DataFrame:
    """Evaluate REAL (OOF) metrics + cost for an (N, M) array of weights."""
    from sklearn.metrics import log_loss, roc_auc_score

    cost_file = args.postwork_out / "inference_costs.json"
    c = _read_json(cost_file)["cost_ms_per_1k"]
    cvec = np.array([c[n] for n in names], dtype=float)
    rows = []
    for w in W:
        p = P_all @ w
        rows.append({
            **{f"w_{n}": float(w[j]) for j, n in enumerate(names)},
            "real_roc_auc": float(roc_auc_score(y_all, p)),
            "real_log_loss": float(log_loss(y_all, p)),
            "real_cost_ms_1k": float(w @ cvec),
        })
    return pd.DataFrame(rows)


def stage_nbi(args) -> None:
    from mixens.nbi import run_nbi_on_simplex

    needed = [args.out / "scheffe_report.json", args.out / "oof.npz",
              args.out / "oof_timings.json"]
    if not _require(args, needed,
                    "python scripts/pco213_run_santander_study.py --stage all  (pipeline da entrega)"):
        return
    if args.dry_run:
        print(f"[nbi][dry-run] objetivos={args.objectives} | ~{args.nbi_points} subproblemas NBI "
              f"sobre metamodelos de Scheffé reconstruídos de {args.out/'scheffe_report.json'};\n"
              f"  candidatos revalidados nas métricas reais OOF; escreveria "
              f"{args.postwork_out/'nbi_candidates.csv'} e {args.postwork_out/'nbi_summary.json'} "
              f"(custo estimado: segundos)")
        return
    P_all, y_all, names = _load_pooled_oof(args)
    objectives, labels, wanted = _build_objectives(args, names)
    result = run_nbi_on_simplex(objectives, len(names), n_points=args.nbi_points,
                                seed=args.random_state)
    W = np.array([c["w"] for c in result["candidates"]], dtype=float)
    df = _real_objective_values(P_all, y_all, names, args, W)
    meta_cols = pd.DataFrame({
        "t": [c["t"] for c in result["candidates"]],
        "residual_norm": [c["residual_norm"] for c in result["candidates"]],
        "success": [c["success"] for c in result["candidates"]],
        **{f"beta_{i}": [c["beta"][i] for c in result["candidates"]]
           for i in range(len(wanted))},
    })
    out = pd.concat([meta_cols, df], axis=1)
    out.to_csv(args.postwork_out / "nbi_candidates.csv", index=False)
    anchors_df = _real_objective_values(P_all, y_all, names, args,
                                        np.array(result["anchors_w"], dtype=float))
    _write_json({
        "objectives": wanted,
        "objective_labels": labels,
        "n_subproblems": int(len(result["candidates"])),
        "n_success": int(sum(c["success"] for c in result["candidates"])),
        "anchors_w": result["anchors_w"],
        "anchors_real": anchors_df.to_dict(orient="records"),
        "payoff_raw": result["payoff_raw"],
        "utopia_raw": result["utopia_raw"],
        "pseudo_nadir_raw": result["pseudo_nadir_raw"],
        "normalized": result["normalized"],
        "model_names": names,
    }, args.postwork_out / "nbi_summary.json")
    print(f"[nbi] {len(result['candidates'])} subproblemas "
          f"({int(sum(c['success'] for c in result['candidates']))} ok) — "
          f"candidatos em {args.postwork_out/'nbi_candidates.csv'}")


def _objective_matrix(df: pd.DataFrame, wanted: list[str]) -> np.ndarray:
    """Real-metric columns -> minimization-convention objective matrix."""
    cols = {"auc": -df["real_roc_auc"].to_numpy(),
            "logloss": df["real_log_loss"].to_numpy(),
            "cost": df["real_cost_ms_1k"].to_numpy()}
    return np.column_stack([cols[o] for o in wanted])


def stage_pareto(args) -> None:
    from mixens.mixture_design import sample_dirichlet
    from mixens.selection import (
        generational_distance,
        inverted_generational_distance,
        normalize_objectives,
        pareto_filter,
        spacing_metric,
    )

    needed = [args.out / "oof.npz", args.postwork_out / "nbi_candidates.csv",
              args.postwork_out / "inference_costs.json"]
    if not _require(args, needed,
                    "python scripts/pco213_run_santander_study.py --stage nbi"):
        return
    if args.dry_run:
        print(f"[pareto][dry-run] avaliaria {args.pareto_dirichlet_points} pontos Dirichlet "
              f"+ vértices/centroide nas métricas reais OOF (~1 min por 1000 pontos), "
              f"filtraria não dominados e compararia com a fronteira NBI;\n  escreveria "
              f"{args.postwork_out/'pareto_reference.csv'} e "
              f"{args.postwork_out/'pareto_metrics.json'}")
        return
    P_all, y_all, names = _load_pooled_oof(args)
    M = len(names)
    wanted = [o.strip() for o in args.objectives.split(",") if o.strip()]
    W_ref = np.vstack([
        np.eye(M),
        np.full((1, M), 1.0 / M),
        sample_dirichlet(M, args.pareto_dirichlet_points, random_state=args.random_state),
    ])
    ref_df = _real_objective_values(P_all, y_all, names, args, W_ref)
    F_ref = _objective_matrix(ref_df, wanted)
    ref_mask = pareto_filter(F_ref)
    ref_df["non_dominated"] = ref_mask
    ref_df.to_csv(args.postwork_out / "pareto_reference.csv", index=False)

    nbi_df = pd.read_csv(args.postwork_out / "nbi_candidates.csv")
    F_nbi = _objective_matrix(nbi_df, wanted)
    nbi_mask = pareto_filter(F_nbi)

    # Normalize both fronts with the SAME reference scale before distances.
    utopia = F_ref[ref_mask].min(axis=0)
    nadir = F_ref[ref_mask].max(axis=0)
    ref_n = normalize_objectives(F_ref[ref_mask], utopia, nadir)
    nbi_n = normalize_objectives(F_nbi[nbi_mask], utopia, nadir)
    metrics = {
        "objectives": wanted,
        "n_reference_points": int(len(F_ref)),
        "n_reference_front": int(ref_mask.sum()),
        "n_nbi_candidates": int(len(F_nbi)),
        "n_nbi_front": int(nbi_mask.sum()),
        "gd_nbi_vs_reference": generational_distance(nbi_n, ref_n),
        "igd_nbi_vs_reference": inverted_generational_distance(nbi_n, ref_n),
        "spacing_nbi_front": spacing_metric(nbi_n),
        "spacing_reference_front": spacing_metric(ref_n),
        "note": ("GD baixo = fronteira NBI próxima da referência; IGD baixo = boa "
                 "cobertura; spacing menor = espaçamento mais uniforme (argumento "
                 "clássico do NBI). Referência = Dirichlet denso + filtro Pareto "
                 "nas métricas reais OOF."),
    }
    _write_json(metrics, args.postwork_out / "pareto_metrics.json")
    print(f"[pareto] referência: {int(ref_mask.sum())}/{len(F_ref)} não dominados | "
          f"NBI: {int(nbi_mask.sum())}/{len(F_nbi)} | GD={metrics['gd_nbi_vs_reference']:.4f} "
          f"IGD={metrics['igd_nbi_vs_reference']:.4f}")


CORE_STAGES = {
    "data": stage_data,
    "bench": stage_bench,
    "oof": stage_oof,
    "design": stage_design,
    "scheffe": stage_scheffe,
    "optimize": stage_optimize,
    "figures": stage_figures,
}
POSTWORK_STAGES = {
    "cost": stage_cost,
    "nbi": stage_nbi,
    "pareto": stage_pareto,
}
STAGES = {**CORE_STAGES, **POSTWORK_STAGES}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["all", "postwork_all", *STAGES])
    ap.add_argument("--mode", default="final_2h", choices=list(data_mod.SAMPLE_CAPS))
    ap.add_argument("--max-runtime-minutes", type=float, default=120.0)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output-dir", default=str(REPO / "experiments" / "pco213" / "santander"))
    # ---- postwork options (safe defaults; never re-run heavy stages) ----
    ap.add_argument("--postwork-output-dir",
                    default=str(REPO / "experiments" / "pco213_postwork" / "santander"))
    ap.add_argument("--nbi-points", type=int, default=15,
                    help="máximo de subproblemas NBI (lattice de betas)")
    ap.add_argument("--pareto-dirichlet-points", type=int, default=5000,
                    help="pontos Dirichlet da fronteira de referência")
    ap.add_argument("--objectives", default="auc,logloss,cost",
                    help=f">=2 dentre {','.join(OBJECTIVE_CHOICES)}")
    ap.add_argument("--skip-heavy", action="store_true", default=True,
                    help="nunca dispara OOF/refit a partir dos estágios postwork (default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="só valida artefatos e imprime o plano dos estágios postwork")
    args = ap.parse_args()
    args.out = Path(args.output_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    args.postwork_out = Path(args.postwork_output_dir)
    if not args.dry_run:
        args.postwork_out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.stage == "all":
        stages = list(CORE_STAGES)  # entrega original — postwork nunca entra aqui
    elif args.stage == "postwork_all":
        stages = list(POSTWORK_STAGES)
    else:
        stages = [args.stage]
    for s in stages:
        STAGES[s](args)
    total = time.perf_counter() - t0
    print(f"[done] stages={stages} total wall {total/60:.1f} min")
    if not args.dry_run:
        stamp = {"stages": stages, "wall_minutes": round(total / 60.0, 2)}
        target = args.postwork_out if stages[0] in POSTWORK_STAGES else args.out
        _write_json(stamp, target / f"run_{'_'.join(stages[:1])}_timing.json")


if __name__ == "__main__":
    main()
