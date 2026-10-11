#!/usr/bin/env python
"""Cross-check of the solver fix 6fa7877 against the benchmark stages kept from before it.

Provenance label: retrospective

Commit 6fa7877 (2026-09-05 00:15:12 -0300) changed mixens.optimize.minimize_on_simplex in
three ways: starts are compared at their end points renormalized onto the simplex; the
best feasible point is used when no SLSQP start reports convergence; and the simplex
vertices and the centroid are evaluated explicitly, replacing the converged solution when
one of them is lower by more than 1e-12. The run that had crashed was restarted under the
fix, and the stages that had already finished were kept. This script

  1. finds, from the stage timestamps in stage_status.json, the stages that finished before
     the fix commit, and confirms from benchmark.log that each was logged as done by the run
     launched before the fix and that no stage finished between the commit and the restart;
  2. for the minimize_on_simplex calls of those stages (the two Scheffe-surface optima of
     each refs stage and the random-scalarization solutions of each comparators stage, as
     called by scripts/pco213_run_postwork_benchmark.py), evaluates the stored objective at
     the stored solution and at the M vertices and the centroid, and records whether the
     explicit vertex/centroid rule would replace the stored solution, by how much in weight
     space (max and sum of absolute weight changes) and in the objective, and how the stored
     real objectives of the solution compare with those stored for the replacing vertex in
     design_eval.csv (the design contains every vertex and the centroid);
  3. validates the reconstruction of the objectives on every stage run after the fix: there
     the returned solution can exceed the best vertex/centroid value by at most the 1e-12
     tolerance (plus renormalization rounding), and the script stops if any does not.

The other two changes of 6fa7877 (renormalized comparison of starts, fallback to a
non-converged iterate) cannot be checked without rerunning the optimizer and are not.
The objectives are the stored Scheffe polynomials of scheffe.json (via the project loader
mixens.scheffe.model_from_coefficients) and, for the scalarizations, the runner's
normalization by the real anchors (mixens.nbi.anchors_from_points on free variables) with
the stored lambdas and measured costs.

Inputs (raw, gitignored, read-only):
  experiments/pco213_postwork_benchmark/benchmark.log
  experiments/pco213_postwork_benchmark/<dataset>/rep_XX/stage_status.json, scheffe.json,
      references.json, oof_meta.json, comparator_ws_random_scalarization.csv, design_eval.csv
Output:
  reports/pco213_postwork_benchmark/statistics/solver_fix_crosscheck.csv
      block "stage": one row per stage finished before the fix (attempts, finish time, the
      log's launch time of the run that executed it, number of checked/changed solutions;
      its move_* columns are the largest over the stage's solutions);
      block "solution": one row per checked minimize_on_simplex solution of those stages
      (stored_minus_vertex_* only where the rule would replace the solution).
      Timestamps are copied from the stored outputs, not generated.

Reads stored outputs only; runs no learner, optimizer or surrogate fit.

Usage:
    python scripts/pco213_solver_fix_crosscheck.py
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from mixens.nbi import anchors_from_points, linear_cost, objectives_on_free_vars  # noqa: E402
from mixens.scheffe import model_from_coefficients  # noqa: E402

SCRIPT = "scripts/pco213_solver_fix_crosscheck.py"
LABEL = "retrospective"
BENCH = Path("experiments") / "pco213_postwork_benchmark"
OUT_CSV = Path("reports") / "pco213_postwork_benchmark" / "statistics" / "solver_fix_crosscheck.csv"

FIX_COMMIT = "6fa7877"
FIX_COMMITTED_AT = datetime.fromisoformat("2026-09-05T00:15:12-03:00")  # git show -s --format=%cI 6fa7877
RULE_TOL = 1e-12        # optimize.py@6fa7877: replace when feas_v < best_v - 1e-12
RENORM_SLACK = 1e-15    # rounding of the final clip-and-renormalize step of minimize_on_simplex
MOVE_THRESHOLD = 1e-10  # the bound quoted by the manuscript (max absolute weight change)
# stages of scripts/pco213_run_postwork_benchmark.py that call minimize_on_simplex
OPTIMIZER_STAGES = {"refs": "stage_refs: scheffe_optimum_logloss, scheffe_optimum_auc",
                    "comparators": "stage_comparators: ws_random_scalarization"}
REFS_CALLS = (("scheffe_optimum_logloss", "log_loss", 1.0), ("scheffe_optimum_auc", "roc_auc", -1.0))
REAL_COLS = ("roc_auc", "log_loss", "cost_weighted", "cost_support")

COLUMNS = [
    "block", "dataset", "rep", "stage", "item", "attempts", "finished_at", "run_launched_at",
    "calls_minimize_on_simplex", "n_solutions_checked", "n_solutions_changed", "vertex_rule_changes_output",
    "move_max_abs_weight", "move_sum_abs_weight", "objective", "surface_orders",
    "f_stored", "f_best_vertex_or_centroid", "best_vertex_or_centroid", "f_stored_minus_best",
    "nearest_vertex", "max_abs_weight_dist_nearest_vertex",
    "stored_minus_vertex_roc_auc", "stored_minus_vertex_log_loss",
    "stored_minus_vertex_cost_weighted", "stored_minus_vertex_cost_support",
]
INT_COLS = ["rep", "attempts", "n_solutions_checked", "n_solutions_changed"]
BOOL_COLS = ["calls_minimize_on_simplex", "vertex_rule_changes_output"]

LOG_START = re.compile(r"^\[(?P<ts>[^\]]+)\] benchmark start:")
LOG_STAGE = re.compile(r"^\[(?P<ts>[^\]]+)\] \[(?P<ds>[^\]]+)\]\[rep (?P<rep>\d+)\]\[(?P<stage>[^\]]+)\] (?P<what>done|FAILED)")


def scheffe_surface(sch: dict, response: str):
    """Selected-order surface of a stored scheffe.json (as scheffe_surface in the runner)."""
    sel = sch[response]["selected_order"]
    fit = sch[response]["orders"][sel]
    return model_from_coefficients(sch["component_names"], fit["terms"], fit["coefficients"]), sel


def predict(model, w: np.ndarray) -> float:
    return float(model.predict_weights(np.asarray(w, dtype=float)[None, :])[0])


def candidates(M: int) -> np.ndarray:
    """The explicit candidates of 6fa7877: the M vertices, then the centroid."""
    return np.vstack([np.eye(M), np.full((1, M), 1.0 / M)])


def comparator_objectives(sch: dict, refs: dict, costs: np.ndarray):
    """Scalarization of stage_comparators: lambda . (F(w) - utopia) / span, F = (-AUC, LL, cost)
    on the selected surfaces, normalized by the real anchors (runner lines 615-628)."""
    m_auc, o_auc = scheffe_surface(sch, "roc_auc"); m_ll, o_ll = scheffe_surface(sch, "log_loss")
    M = len(sch["component_names"])
    objs = [lambda w: -predict(m_auc, w), lambda w: predict(m_ll, w), lambda w: linear_cost(w, costs)]
    raw = objectives_on_free_vars(objs)
    anchors = anchors_from_points(raw, np.array([np.asarray(refs["real_anchors"][k], dtype=float)[: M - 1]
                                                 for k in ("auc", "logloss", "cost")]))
    span = np.where(np.abs(anchors.pseudo_nadir - anchors.utopia) < 1e-12, 1.0, anchors.pseudo_nadir - anchors.utopia)

    def make(lam: np.ndarray):
        return lambda w: float(np.dot(lam, (np.array([o(w) for o in objs]) - anchors.utopia) / span))
    return make, f"roc_auc:{o_auc}|log_loss:{o_ll}"


def check(f, w: np.ndarray, names: list[str]) -> dict:
    """Would the explicit vertex/centroid candidates of 6fa7877 replace the stored solution w?"""
    C = candidates(len(names)); labels = names + ["centroid"]
    f_s = f(w); fc = np.array([f(c) for c in C]); j = int(np.argmin(fc))
    changes = bool(fc[j] < f_s - RULE_TOL)
    k = int(np.argmax(w))
    return {"f_stored": f_s, "f_best_vertex_or_centroid": float(fc[j]), "best_vertex_or_centroid": labels[j],
            "f_stored_minus_best": float(f_s - fc[j]), "vertex_rule_changes_output": changes,
            "move_max_abs_weight": float(np.abs(C[j] - w).max()) if changes else 0.0,
            "move_sum_abs_weight": float(np.abs(C[j] - w).sum()) if changes else 0.0,
            "nearest_vertex": names[k], "max_abs_weight_dist_nearest_vertex": float(np.abs(np.eye(len(names))[k] - w).max()),
            "_target": C[j]}


def main() -> None:
    root = REPO / BENCH
    # ------------------------------------------------------------------ 1. timeline
    log = (root / "benchmark.log").read_text().splitlines()
    launches = sorted(datetime.fromisoformat(m["ts"]) for m in map(LOG_START.match, log) if m)
    logged = {}
    for m in map(LOG_STAGE.match, log):
        if m:
            logged.setdefault((m["ds"], int(m["rep"]), m["stage"]), []).append((datetime.fromisoformat(m["ts"]), m["what"]))
    restart = min(t for t in launches if t > FIX_COMMITTED_AT)
    stages, finishes = [], []
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for rep_dir in sorted(ds_dir.glob("rep_*")):
            rep = int(rep_dir.name.split("_")[1])
            for stage, s in json.loads((rep_dir / "stage_status.json").read_text()).items():
                fin = datetime.fromisoformat(s["finished_at"]) if s.get("finished_at") else None
                finishes.append(fin)
                if s.get("status") == "done" and fin is not None and fin < FIX_COMMITTED_AT:
                    stages.append({"dataset": ds_dir.name, "rep": rep, "stage": stage, "attempts": int(s["attempts"]),
                                   "finished": fin, "rep_dir": rep_dir})
    assert not any(f is not None and FIX_COMMITTED_AT <= f < restart for f in finishes), "stage finished between fix and restart"
    for s in stages:
        # logged as done at the stored finish time, by a run launched before the fix
        assert (s["finished"], "done") in logged.get((s["dataset"], s["rep"], s["stage"]), []), s
        s["launched"] = max(t for t in launches if t <= s["finished"])
        assert s["launched"] < FIX_COMMITTED_AT

    # ------------------------------------------------------------------ 2. pre-fix solutions
    rows: list[dict] = []
    for s in sorted(stages, key=lambda s: (s["dataset"], s["rep"], s["stage"])):
        d = s["rep_dir"]
        base = {"dataset": s["dataset"], "rep": s["rep"], "stage": s["stage"], "attempts": s["attempts"],
                "finished_at": s["finished"].isoformat(), "run_launched_at": s["launched"].isoformat(),
                "calls_minimize_on_simplex": s["stage"] in OPTIMIZER_STAGES}
        sols = []
        if s["stage"] in OPTIMIZER_STAGES:
            sch = json.loads((d / "scheffe.json").read_text()); refs = json.loads((d / "references.json").read_text())
            meta = json.loads((d / "oof_meta.json").read_text()); names = list(meta["model_names"])
            assert sch["component_names"] == [f"w_{n}" for n in names]
            wc = [f"w_{n}" for n in names]
            de = pd.read_csv(d / "design_eval.csv")
            if s["stage"] == "refs":
                for key, resp, sign in REFS_CALLS:
                    m, order = scheffe_surface(sch, resp)
                    e = refs["references"][key]; w = np.asarray(e["w"], dtype=float)
                    assert predict(m, w) == e["detail"]["predicted"], "stored polynomial not reproduced"
                    r = check(lambda v, m=m, sign=sign: sign * predict(m, v), w, names)
                    stored = {"roc_auc": e["oof_roc_auc"], "log_loss": e["oof_log_loss"],
                              "cost_weighted": e["cost_weighted"], "cost_support": e["cost_support"]}
                    sols.append({**r, "item": key, "surface_orders": f"{resp}:{order}",
                                 "objective": ("" if sign > 0 else "-") + f"{resp} surface (minimized)", "_stored": stored})
            else:
                costs = np.array([meta["cost_ms_per_1k"][n] for n in names], dtype=float)
                make, orders = comparator_objectives(sch, refs, costs)
                df = pd.read_csv(d / "comparator_ws_random_scalarization.csv")
                W = df[wc].to_numpy(float); L = df[["lambda_0", "lambda_1", "lambda_2"]].to_numpy(float)
                for i in range(len(df)):
                    r = check(make(L[i]), W[i], names)
                    sols.append({**r, "item": f"ws_random_scalarization_{i:03d}", "surface_orders": orders,
                                 "objective": "lambda . (F - utopia) / span, F = (-roc_auc surface, log_loss surface, cost)",
                                 "_stored": {c: float(df.loc[i, c]) for c in REAL_COLS}})
            for r in sols:
                if r["vertex_rule_changes_output"]:
                    hit = de[np.all(de[wc].to_numpy(float) == r["_target"], axis=1)]
                    assert len(hit) == 1, "replacing candidate not in the stored design"
                    for c in REAL_COLS:
                        r[f"stored_minus_vertex_{c}"] = float(r["_stored"][c] - hit.iloc[0][c])
        n_changed = sum(r["vertex_rule_changes_output"] for r in sols)
        rows.append({**base, "block": "stage", "item": "", "n_solutions_checked": len(sols), "n_solutions_changed": n_changed,
                     "vertex_rule_changes_output": n_changed > 0,
                     "move_max_abs_weight": max([r["move_max_abs_weight"] for r in sols], default=0.0),
                     "move_sum_abs_weight": max([r["move_sum_abs_weight"] for r in sols], default=0.0)})
        for r in sols:
            rows.append({**base, **{k: v for k, v in r.items() if not k.startswith("_")}, "block": "solution",
                         "n_solutions_checked": 1, "n_solutions_changed": int(r["vertex_rule_changes_output"])})

    # ------------------------------------------------------------------ 3. validation on post-fix stages
    pre = {(s["dataset"], s["rep"], s["stage"]) for s in stages}
    gaps = {"refs": [], "comparators": []}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for d in sorted(ds_dir.glob("rep_*")):
            rep = int(d.name.split("_")[1])
            todo = [st for st in OPTIMIZER_STAGES if (ds_dir.name, rep, st) not in pre]
            if not todo:
                continue
            sch = json.loads((d / "scheffe.json").read_text()); refs = json.loads((d / "references.json").read_text())
            meta = json.loads((d / "oof_meta.json").read_text()); names = list(meta["model_names"])
            C = candidates(len(names))
            if "refs" in todo:
                for key, resp, sign in REFS_CALLS:
                    m, _ = scheffe_surface(sch, resp)
                    v = sign * m.predict_weights(np.vstack([np.asarray(refs["references"][key]["w"], dtype=float), C]))
                    gaps["refs"].append(float(v[0] - v[1:].min()))
            if "comparators" in todo:
                m_auc, _ = scheffe_surface(sch, "roc_auc"); m_ll, _ = scheffe_surface(sch, "log_loss")
                costs = np.array([meta["cost_ms_per_1k"][n] for n in names], dtype=float)
                M = len(names)
                raw = objectives_on_free_vars([lambda w: -predict(m_auc, w), lambda w: predict(m_ll, w),
                                               lambda w: linear_cost(w, costs)])
                anchors = anchors_from_points(raw, np.array([np.asarray(refs["real_anchors"][k], dtype=float)[: M - 1]
                                                             for k in ("auc", "logloss", "cost")]))
                span = np.where(np.abs(anchors.pseudo_nadir - anchors.utopia) < 1e-12, 1.0,
                                anchors.pseudo_nadir - anchors.utopia)
                df = pd.read_csv(d / "comparator_ws_random_scalarization.csv")
                W = df[[f"w_{n}" for n in names]].to_numpy(float); L = df[["lambda_0", "lambda_1", "lambda_2"]].to_numpy(float)
                X = np.vstack([W, C])
                F = np.column_stack([-m_auc.predict_weights(X), m_ll.predict_weights(X), X @ costs])
                G = ((F - anchors.utopia) / span) @ L.T          # (points, lambdas)
                n = len(W)
                gaps["comparators"].extend((G[np.arange(n), np.arange(n)] - G[n:, :].min(axis=0)).tolist())
    for st, g in gaps.items():
        bad = sum(x > RULE_TOL + RENORM_SLACK for x in g)
        print(f"validation on post-fix {st} solutions: n = {len(g)}, max(f_returned - best vertex/centroid) = {max(g):.3e}, "
              f"above tolerance: {bad}")
        assert bad == 0, "objective reconstruction inconsistent with the post-fix optimizer"

    # ------------------------------------------------------------------ CSV
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    for col in INT_COLS:
        df[col] = df[col].astype("Int64")
    for col in BOOL_COLS:
        df[col] = df[col].astype(bool)
    df = df.sort_values(["block", "dataset", "rep", "stage", "item"], kind="mergesort").reset_index(drop=True)
    (REPO / OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPO / OUT_CSV, index=False, lineterminator="\n")

    st = df[df.block == "stage"]; so = df[df.block == "solution"]
    print(f"fix {FIX_COMMIT} committed {FIX_COMMITTED_AT.isoformat()}; runs launched {[t.isoformat() for t in launches]}")
    print(f"stages finished before the fix: {len(st)} {st.groupby('dataset').size().to_dict()}; "
          f"of which call minimize_on_simplex: {int(st.calls_minimize_on_simplex.sum())}")
    for stage, g in so.groupby("stage"):
        ch = g[g.vertex_rule_changes_output]; un = g[~g.vertex_rule_changes_output]
        print(f"{stage}: {len(g)} solutions, changed {len(ch)} (max weight move {ch.move_max_abs_weight.max() if len(ch) else 0:.3e}, "
              f"< {MOVE_THRESHOLD:g}: {int((ch.move_max_abs_weight < MOVE_THRESHOLD).sum())}); unchanged {len(un)}: "
              f"best vertex/centroid worse {int((un.f_stored_minus_best < 0).sum())}, "
              f"better by <= {RULE_TOL:g} {int((un.f_stored_minus_best >= 0).sum())}")
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
