#!/usr/bin/env python
"""Split-sample reading of the reliability gate: order selection and gate on disjoint halves.

Provenance label: post hoc (coauthor review R2)

The frozen procedure uses the same 100 external compositions of a replication twice:
the parsimony rule selects the Scheffe order on them (lowest order whose external RMSE
is within 10% of the best), and the reliability gate then judges the selected order on
them (external R^2 >= 0.5 and Spearman rho >= 0.9). The gate is therefore not an
independent test of the order it judges. This script asks how much of the recorded
pass rate depends on that reuse, from stored records only.

Definition (fixed before the first run):
  * Inputs per dataset x replication x gated response (roc_auc, log_loss, brier,
    pr_auc): the stored coefficients of every Scheffe order (scheffe.json), the 100
    stored compositions (design_points.npz, W_val) and their stored real values
    (validation_eval.csv). The surfaces are fitted on the 66 design points only, so
    no fit depends on the split; predictions at W_val come from the stored
    coefficients. The script first reproduces, from these predictions, every stored
    external RMSE, R^2 and rho, every stored selected order and every stored gate
    verdict, and stops if any differs.
  * Halves: 200 random stratified halves per case, 30 of the 60 Dirichlet(1)
    compositions and 20 of the 40 Dirichlet(0.5) ones, from
    numpy.random.default_rng(20261013), the same 200 halves for every case.
  * Cross-half (independent) reading: select the order on one half with the frozen
    rule and judge it with the frozen gate on the other half, both directions, 400
    evaluations per case. Same-half (reuse) control: select and judge on the same
    half, 400 evaluations per case. Both use 50 compositions, so their difference
    isolates reuse from the loss of sample size; R^2 uses the judging half's own mean,
    as the frozen external_validation does.
  * Reported per dataset x response: the recorded pass count of 30; the expected pass
    counts sum(p_same) and sum(p_cross), where p is a replication's pass fraction;
    their difference (the optimism attributable to reuse); the number of replications
    whose cross-half majority verdict (p_cross >= 0.5) differs from the recorded one;
    and how often the half-sample selection agrees with the recorded order.

Nothing here replaces the recorded gate, the selected orders or any NBI-A result.
NBI-A ran on the order selected with all 100 compositions; a different order would
need a new NBI run, which is not done.

Inputs (frozen, read-only):
  experiments/pco213_postwork_benchmark/<dataset>/rep_XX/
      scheffe.json, design_points.npz, validation_eval.csv
Outputs:
  reports/pco213_postwork_benchmark/statistics/gate_split_sample_r30.json
  reports/pco213_postwork_benchmark/statistics/gate_split_sample_r30.csv

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:
    python scripts/pco213_gate_split_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mixens.scheffe import model_from_coefficients  # noqa: E402

RAW = REPO / "experiments" / "pco213_postwork_benchmark"
OUT = REPO / "reports" / "pco213_postwork_benchmark" / "statistics"
DATASETS = ("santander", "bnp", "porto", "uci_credit")
RESPONSES = ("roc_auc", "log_loss", "brier", "pr_auc")
ORDERS = ("linear", "quadratic", "special_cubic")
REPS = 30
N_SPLITS, SEED = 200, 20261013
STRATA = (60, 40)                      # Dirichlet(1) block, then Dirichlet(0.5) block
TOL, R2_MIN, RHO_MIN = 0.10, 0.5, 0.9  # frozen parsimony rule and gate


def halves() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    a, b = STRATA
    out = np.zeros((N_SPLITS, a + b), dtype=bool)
    for s in range(N_SPLITS):
        out[s, rng.choice(a, a // 2, replace=False)] = True
        out[s, a + rng.choice(b, b // 2, replace=False)] = True
    return out


def stats_on(pred: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    resid = y - pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    rho = float(spearmanr(pred, y).correlation)
    return rmse, r2, rho


def rmse_on(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def select(rmse: dict[str, float]) -> str:
    best = min(rmse.values())
    return next(o for o in ORDERS if o in rmse and rmse[o] <= (1.0 + TOL) * best)


def passes(r2: float, rho: float) -> bool:
    return bool(r2 >= R2_MIN and rho >= RHO_MIN)


def one_case(sch: dict, names, W_val: np.ndarray, y: np.ndarray, H: np.ndarray) -> dict:
    fits = sch["orders"]
    est = [o for o in ORDERS if fits[o].get("estimable")]
    pred = {o: model_from_coefficients(names, fits[o]["terms"], fits[o]["coefficients"])
            .predict_weights(W_val) for o in est}
    # reproduce the stored stage
    for o in est:
        rmse, r2, rho = stats_on(pred[o], y)
        ext = fits[o]["external"]
        if (abs(rmse - ext["rmse"]) > 1e-12 or abs(r2 - ext["r2_external"]) > 1e-9
                or abs(rho - fits[o]["spearman_external"]) > 1e-12):
            raise SystemExit("stored external statistics not reproduced")
    sel = select({o: rmse_on(pred[o], y) for o in est})
    if sel != sch["selected_order"]:
        raise SystemExit("stored selected order not reproduced")
    rec = passes(*stats_on(pred[sel], y)[1:])
    if rec != bool(sch["reliable"]):
        raise SystemExit("stored gate verdict not reproduced")

    cross, same, agree = [], [], []
    for h in H:
        for s_mask, j_mask in ((h, ~h), (~h, h)):
            o = select({k: rmse_on(pred[k][s_mask], y[s_mask]) for k in est})
            agree.append(o == sel)
            cross.append(passes(*stats_on(pred[o][j_mask], y[j_mask])[1:]))
            same.append(passes(*stats_on(pred[o][s_mask], y[s_mask])[1:]))
    return {"selected_order": sel, "recorded_pass": rec,
            "p_cross": float(np.mean(cross)), "p_same": float(np.mean(same)),
            "order_agreement": float(np.mean(agree))}


def main() -> None:
    H = halves()
    rows = []
    for ds in DATASETS:
        for rep in range(REPS):
            d = RAW / ds / f"rep_{rep:02d}"
            sch = json.loads((d / "scheffe.json").read_text())
            W_val = np.load(d / "design_points.npz")["W_val"].astype(float)
            ve = pd.read_csv(d / "validation_eval.csv")
            names = sch["component_names"]
            if not np.allclose(ve[names].to_numpy(dtype=float), W_val, atol=1e-12):
                raise SystemExit(f"{ds} rep {rep}: validation rows do not match W_val")
            for resp in RESPONSES:
                r = one_case(sch[resp], names, W_val, ve[resp].to_numpy(dtype=float), H)
                rows.append({"dataset": ds, "replication": rep, "response": resp, **r})
    df = pd.DataFrame(rows)
    summary = []
    for (ds, resp), g in df.groupby(["dataset", "response"], sort=False):
        summary.append({
            "dataset": ds, "response": resp,
            "recorded_passes": int(g["recorded_pass"].sum()),
            "expected_passes_same_half": round(float(g["p_same"].sum()), 2),
            "expected_passes_cross_half": round(float(g["p_cross"].sum()), 2),
            "optimism_same_minus_cross": round(float((g["p_same"] - g["p_cross"]).sum()), 2),
            "replications_majority_verdict_differs": int(
                ((g["p_cross"] >= 0.5) != g["recorded_pass"]).sum()),
            "median_order_agreement": round(float(g["order_agreement"].median()), 3),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "gate_split_sample_r30.csv", index=False)
    (OUT / "gate_split_sample_r30.json").write_text(json.dumps({
        "provenance": "post hoc (coauthor review R2); stored records only",
        "definition": {"n_splits": N_SPLITS, "seed": SEED, "strata": STRATA,
                       "half": [STRATA[0] // 2, STRATA[1] // 2],
                       "parsimony_tolerance": TOL, "gate": {"r2_min": R2_MIN, "rho_min": RHO_MIN},
                       "reproduced_stored_stage": True},
        "summary": summary}, indent=1) + "\n")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
