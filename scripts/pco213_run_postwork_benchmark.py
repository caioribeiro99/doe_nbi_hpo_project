#!/usr/bin/env python3
"""PCO213 post-work multi-dataset replicated benchmark (checkpointed, resumable).

Pipeline per dataset x outer replication (R independent stratified 80/20
partitions, seed = base_seed + rep):
  oof         5-fold stratified OOF probabilities of the 5-model zoo on the
              outer-train side; single refit -> holdout matrix Q; direct
              inference-cost measurement (median of 5 timed predictions)
  design      66-run controlled mixture design + 100 Dirichlet validation
              points evaluated on the REAL cached OOF (AUC, log-loss, Brier,
              PR-AUC, costs, N_eff, entropy)
  scheffe     Scheffé linear/quadratic/special-cubic per response, external
              validation, parsimony selection, reliability gate
  refs        single-objective references (best single, uniform, stacking,
              SLSQP log-loss, direct AUC search, Scheffé optima, real anchors)
              with OOF and untouched-holdout metrics
  reference   empirical Pareto reference: >=100k Dirichlet(1)/Dirichlet(0.3)
              points + vertices/centroid/edge & lattice points + references +
              exact epsilon-constraint sweep (log-loss x weighted cost), with
              convergence diagnostics; fronts under weighted AND support cost
  nbi_A       NBI on Scheffé surfaces, surrogate-derived anchors
  nbi_B       NBI on Scheffé surfaces, REAL anchors
  nbi_C       NBI directly on the real cached OOF objectives (metamodel-free)
  comparators random weighted scalarization on surfaces (real anchors),
              budget-matched random Dirichlet search, the DoE runs themselves
  quality     real-objective revalidation, Pareto filtering, GD/IGD/IGD+/
              spacing/HV/coverage/joint-ND vs the empirical reference (both
              cost definitions), face-by-face support analysis, size-matched
              spacing test, holdout confirmation of selected points

State: <root>/benchmark_manifest.json (master) + <root>/<dataset>/rep_XX/
stage_status.json. A rerun with --resume skips completed stages, retries
failed ones and never recomputes valid OOF matrices.

Example (detached):
  tmux new -d -s pco213bench \\
    ".venv-pco213/bin/python scripts/pco213_run_postwork_benchmark.py --run-all --resume"
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from mixens import bench_models, datasets as ds_mod  # noqa: E402
from mixens import fastmetrics as fm  # noqa: E402
from mixens import pareto_tools as pt  # noqa: E402
from mixens.mixture_design import build_benchmark_design, generate_simplex_lattice, sample_dirichlet  # noqa: E402
from mixens.nbi import linear_cost, run_nbi_on_simplex  # noqa: E402
from mixens.scheffe import compare_orders, model_from_coefficients  # noqa: E402
from mixens.selection import SelectionRule, select  # noqa: E402

STAGES = ["oof", "design", "scheffe", "refs", "reference", "nbi_A", "nbi_B", "nbi_C", "comparators", "quality"]
RESPONSES = ["roc_auc", "log_loss", "brier", "pr_auc"]
OBJ_LABELS = ["neg_auc", "log_loss", "cost"]
SUPPORT_EPS = 1e-3
RELIABILITY_R2 = 0.5
RELIABILITY_RHO = 0.9


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def env_info() -> dict:
    import sklearn, scipy, statsmodels, xgboost
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
        "xgboost": xgboost.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, args, log: Logger):
        self.args = args
        self.log = log
        self.root = Path(args.root)
        self.raw_root = REPO / "data" / "pco213" / "raw"
        self._data_cache: dict[str, tuple] = {}

    # ---- data
    def dataset(self, name: str):
        if name not in self._data_cache:
            self.log(f"[{name}] loading raw data")
            X, y, spec = ds_mod.load_dataset(name, self.raw_root)
            if self.args.sample_rows and self.args.sample_rows < len(X):
                from sklearn.model_selection import train_test_split
                X, _, y, _ = train_test_split(X, y, train_size=self.args.sample_rows, stratify=y,
                                              random_state=self.args.base_seed)
                X = X.reset_index(drop=True); y = y.reset_index(drop=True)
                spec.notes.append(f"DEBUG/SMOKE stratified subsample to {len(X)} rows")
                spec.n_rows_used = len(X)
            self._data_cache[name] = (X, y, spec)
            write_json_atomic({"spec": spec.to_dict(), "model_params": bench_models.model_params(spec),
                               "loaded_at": now()}, self.root / name / "dataset_manifest.json")
        return self._data_cache[name]

    def rep_dir(self, name: str, rep: int) -> Path:
        d = self.root / name / f"rep_{rep:02d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---- stage status
    def status(self, name: str, rep: int) -> dict:
        return read_json(self.rep_dir(name, rep) / "stage_status.json", {})

    def set_status(self, name: str, rep: int, stage: str, **fields) -> None:
        st = self.status(name, rep)
        entry = st.get(stage, {"attempts": 0})
        entry.update(fields)
        st[stage] = entry
        write_json_atomic(st, self.rep_dir(name, rep) / "stage_status.json")
        self._update_master(name, rep, stage, entry)

    def _update_master(self, name: str, rep: int, stage: str, entry: dict) -> None:
        mpath = self.root / "benchmark_manifest.json"
        m = read_json(mpath, {})
        m.setdefault("status", {}).setdefault(name, {}).setdefault(f"rep_{rep:02d}", {})[stage] = entry
        m["updated_at"] = now()
        write_json_atomic(m, mpath)

    # ---- cached loaders
    def load_oof(self, name: str, rep: int) -> dict:
        d = self.rep_dir(name, rep)
        z = np.load(d / "oof.npz", allow_pickle=False)
        meta = read_json(d / "oof_meta.json")
        names = [str(s) for s in z["model_names"]]
        costs = np.array([meta["cost_ms_per_1k"][n] for n in names])
        return {"P": z["P"], "Q": z["Q"], "y_train": z["y_train"], "y_test": z["y_test"],
                "fold_ids": z["fold_ids"], "names": names, "costs": costs, "meta": meta}


# ---------------------------------------------------------------------------
# helpers on cached probabilities
# ---------------------------------------------------------------------------

def eval_W(oof: dict, W: np.ndarray, *, n_jobs: int = 1, with_pr_auc: bool = True) -> pd.DataFrame:
    W = np.atleast_2d(np.asarray(W, dtype=float))
    out = fm.evaluate_weights(oof["P"], oof["y_train"], W, costs=oof["costs"], support_eps=SUPPORT_EPS,
                              n_jobs=n_jobs, with_pr_auc=with_pr_auc)
    df = pd.DataFrame(out)
    for j, n in enumerate(oof["names"]):
        df[f"w_{n}"] = W[:, j]
    return df


def objectives(df: pd.DataFrame, cost_col: str) -> np.ndarray:
    return np.column_stack([-df["roc_auc"].to_numpy(), df["log_loss"].to_numpy(), df[cost_col].to_numpy()])


def holdout_metrics(oof: dict, w: np.ndarray | None = None, p_oof: np.ndarray | None = None,
                    p_hold: np.ndarray | None = None) -> dict:
    """Holdout confirmation with the F1 threshold selected on the OOF predictions only."""
    if w is not None:
        p_oof = oof["P"].astype(np.float64) @ w
        p_hold = oof["Q"].astype(np.float64) @ w
    thr = fm.best_f1_threshold(oof["y_train"], p_oof)
    y = oof["y_test"]
    out = {"holdout_roc_auc": fm.rank_auc(y, p_hold), "holdout_log_loss": fm.log_loss_vec(y, p_hold),
           "holdout_brier": fm.brier_vec(y, p_hold), "holdout_pr_auc": fm.average_precision(y, p_hold)}
    out.update({f"holdout_{k}": v for k, v in fm.threshold_metrics(y, p_hold, thr).items()})
    return out


def minimize_real_logloss(P: np.ndarray, y: np.ndarray, *, cost_vec=None, cost_cap=None, n_starts=6, seed=0):
    """Exact convex minimization of log-loss over the simplex (SLSQP, analytic gradient),
    optionally under a linear cost cap."""
    from scipy.optimize import minimize
    Pd = P.astype(np.float64); yd = y.astype(np.float64); M = Pd.shape[1]

    def f(w):
        p = np.clip(Pd @ w, 1e-15, 1 - 1e-15)
        return float(-np.mean(yd * np.log(p) + (1 - yd) * np.log1p(-p)))

    def g(w):
        p = np.clip(Pd @ w, 1e-15, 1 - 1e-15)
        d = (p - yd) / (p * (1 - p))
        return (Pd.T @ d) / len(yd)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(M)}]
    if cost_cap is not None:
        cons.append({"type": "ineq", "fun": lambda w: cost_cap - w @ cost_vec, "jac": lambda w: -cost_vec})
    rng = np.random.default_rng(seed)
    starts = [np.full(M, 1 / M)] + [rng.dirichlet(np.ones(M)) for _ in range(n_starts - 1)]
    if cost_cap is not None:
        starts.append(np.eye(M)[int(np.argmin(cost_vec))])
    best = None; best_any = None
    for w0 in starts:
        r = minimize(f, w0, jac=g, method="SLSQP", bounds=[(0, 1)] * M, constraints=cons,
                     options={"maxiter": 500, "ftol": 1e-12})
        feas = abs(r.x.sum() - 1) < 1e-6 and (cost_cap is None or r.x @ cost_vec <= cost_cap + 1e-6)
        if feas and (best_any is None or r.fun < best_any.fun):
            best_any = r
        if r.success and feas and (best is None or r.fun < best.fun):
            best = r
    best = best if best is not None else best_any
    if best is None:
        return None
    w = np.clip(best.x, 0, 1); return w / w.sum()


def maximize_real_auc(P: np.ndarray, y: np.ndarray, W_candidates: np.ndarray, *, seed=0, maxfev=600):
    """Direct AUC search: best candidate from a dense sample, polished by Nelder-Mead on the
    free variables (AUC is piecewise constant; NM is derivative-free)."""
    from scipy.optimize import minimize
    yb = y.astype(bool); Pf = P.astype(np.float32); M = P.shape[1]
    aucs = np.array([fm.fast_auc_no_ties(yb, Pf @ w.astype(np.float32)) for w in W_candidates])
    w0 = W_candidates[int(np.argmax(aucs))]
    nfev = len(W_candidates)

    def obj(z):
        z = np.clip(z, 0, 1); s = z.sum()
        if s > 1:
            z = z / s
        w = np.concatenate([z, [1 - z.sum()]]).astype(np.float32)
        return -fm.fast_auc_no_ties(yb, Pf @ w)

    r = minimize(obj, w0[:-1], method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-5, "fatol": 1e-9})
    nfev += int(r.nfev)
    z = np.clip(r.x, 0, 1); s = z.sum()
    if s > 1:
        z = z / s
    w = np.concatenate([z, [max(0.0, 1 - z.sum())]]); w = w / w.sum()
    auc = fm.fast_auc_no_ties(yb, Pf @ w.astype(np.float32))
    if auc < aucs.max():
        w, auc = w0, float(aucs.max())
    return w, float(auc), nfev


def scheffe_surface(sch: dict, response: str):
    sel = sch[response]["selected_order"]
    fit = sch[response]["orders"][sel]
    return model_from_coefficients(sch["component_names"], fit["terms"], fit["coefficients"]), sel


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def stage_oof(ctx: Ctx, name: str, rep: int) -> dict:
    X, y, spec = ctx.dataset(name)
    d = ctx.rep_dir(name, rep)
    res = bench_models.run_replication(X, y, spec, rep=rep, base_seed=ctx.args.base_seed,
                                       cost_batch_rows=ctx.args.cost_batch_rows, log=ctx.log)
    np.savez_compressed(d / "oof.npz", P=res.P, Q=res.Q, y_train=res.y_train, y_test=res.y_test,
                        fold_ids=res.fold_ids, model_names=np.array(res.model_names))
    meta = {"rep": rep, "seed": res.seed, "n_train": int(len(res.y_train)), "n_test": int(len(res.y_test)),
            "fit_seconds": res.fit_seconds, "predict_seconds": res.predict_seconds,
            "refit_seconds": res.refit_seconds, "cost_ms_per_1k": res.cost_ms_per_1k,
            "cost_measurement": res.cost_measurement, "n_fits": res.n_fits,
            "model_names": res.model_names, "n_splits": 5, "holdout_size": 0.2}
    # model-level performance (OOF and holdout)
    perf = {}
    for j, n in enumerate(res.model_names):
        e = np.eye(len(res.model_names))[j]
        oof = {"P": res.P, "Q": res.Q, "y_train": res.y_train, "y_test": res.y_test}
        perf[n] = {"oof_roc_auc": fm.rank_auc(res.y_train, res.P[:, j]),
                   "oof_log_loss": fm.log_loss_vec(res.y_train, res.P[:, j]),
                   "oof_brier": fm.brier_vec(res.y_train, res.P[:, j]),
                   "oof_pr_auc": fm.average_precision(res.y_train, res.P[:, j]),
                   **holdout_metrics(oof, w=e)}
    meta["model_performance"] = perf
    # classifier disagreement / error diversity (for Q11)
    yt = res.y_train.astype(np.float64)
    err = np.abs(res.P.astype(np.float64) - yt[:, None])
    meta["error_correlation"] = np.corrcoef(err.T).tolist()
    hard = (res.P > 0.5).astype(int)
    M = len(res.model_names)
    meta["disagreement_rate"] = [[float(np.mean(hard[:, i] != hard[:, j])) for j in range(M)] for i in range(M)]
    write_json_atomic(meta, d / "oof_meta.json")
    return {"n_fits": res.n_fits, "n_train": int(len(res.y_train))}


def stage_design(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    M = len(oof["names"])
    W_design = build_benchmark_design(M)
    seed = ctx.args.base_seed + rep
    W_val = np.vstack([sample_dirichlet(M, 60, alpha=1.0, random_state=seed + 1000),
                       sample_dirichlet(M, 40, alpha=0.5, random_state=seed + 2000)])
    np.savez(d / "design_points.npz", W_design=W_design, W_val=W_val, model_names=np.array(oof["names"]))
    de = eval_W(oof, W_design); de.insert(0, "point", np.arange(len(W_design)))
    ve = eval_W(oof, W_val); ve.insert(0, "point", np.arange(len(W_val)))
    de.to_csv(d / "design_eval.csv", index=False); ve.to_csv(d / "validation_eval.csv", index=False)
    return {"n_design": int(len(W_design)), "n_validation": int(len(W_val))}


def stage_scheffe(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    z = np.load(d / "design_points.npz", allow_pickle=False)
    W_design, W_val = z["W_design"], z["W_val"]
    de = pd.read_csv(d / "design_eval.csv"); ve = pd.read_csv(d / "validation_eval.csv")
    comp = [f"w_{n}" for n in oof["names"]]
    report = {"component_names": comp, "n_design": int(len(W_design)), "n_validation": int(len(W_val))}
    for resp in RESPONSES:
        cmp = compare_orders(W_design, de[resp].to_numpy(), W_val, ve[resp].to_numpy(), component_names=comp)
        sel = cmp["orders"][cmp["selected_order"]]
        ext = sel["external"]
        cmp["reliable"] = bool(ext["r2_external"] >= RELIABILITY_R2 and sel["spearman_external"] >= RELIABILITY_RHO)
        cmp["reliability_rule"] = f"external R2 >= {RELIABILITY_R2} and Spearman >= {RELIABILITY_RHO} on unseen points"
        # always keep the quadratic interactions for stability analysis
        q = cmp["orders"].get("quadratic", {})
        if q.get("estimable"):
            cmp["quadratic_beta"] = dict(zip(q["terms"], q["coefficients"]))
        report[resp] = cmp
    write_json_atomic(report, d / "scheffe.json")
    return {resp: {"selected": report[resp]["selected_order"], "reliable": report[resp]["reliable"],
                   "r2_ext": report[resp]["orders"][report[resp]["selected_order"]]["external"]["r2_external"]}
            for resp in RESPONSES}


def stage_refs(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    P, y, Q, names, costs = oof["P"], oof["y_train"], oof["Q"], oof["names"], oof["costs"]
    M = len(names); seed = ctx.args.base_seed + rep
    sch = read_json(d / "scheffe.json")
    z = np.load(d / "design_points.npz", allow_pickle=False)
    counts = {"direct_auc_evals": 0}
    refs: dict[str, dict] = {}

    def add(key, w=None, p_oof=None, p_hold=None, detail=None):
        entry = {"detail": detail or {}}
        if w is not None:
            w = np.asarray(w, float); entry["w"] = w.tolist()
            m = eval_W(oof, w[None, :]).iloc[0].to_dict()
            entry.update({f"oof_{k}": float(m[k]) for k in ("roc_auc", "log_loss", "brier", "pr_auc")})
            entry.update({k: float(m[k]) for k in ("cost_weighted", "cost_support", "n_eff", "entropy", "n_support")})
            entry.update(holdout_metrics(oof, w=w))
        else:
            entry.update({"oof_roc_auc": fm.rank_auc(y, p_oof), "oof_log_loss": fm.log_loss_vec(y, p_oof),
                          "oof_brier": fm.brier_vec(y, p_oof), "oof_pr_auc": fm.average_precision(y, p_oof)})
            entry.update(holdout_metrics(oof, p_oof=p_oof, p_hold=p_hold))
        refs[key] = entry

    # best single (by OOF AUC), uniform
    aucs = [fm.rank_auc(y, P[:, j]) for j in range(M)]
    jb = int(np.argmax(aucs)); add("best_single", w=np.eye(M)[jb], detail={"model": names[jb]})
    add("uniform_voting", w=np.full(M, 1 / M))
    # logistic stacking on the OOF matrix
    from sklearn.linear_model import LogisticRegression
    stk = LogisticRegression(max_iter=2000).fit(P, y)
    add("stacking_lr", p_oof=stk.predict_proba(P)[:, 1], p_hold=stk.predict_proba(Q)[:, 1],
        detail={"coef": stk.coef_[0].tolist(), "intercept": float(stk.intercept_[0]),
                "cost_support": float(costs.sum())})
    # SLSQP log-loss oracle (exact, convex)
    w_ll = minimize_real_logloss(P, y, seed=seed)
    if w_ll is None:  # degenerate fallback: best design/validation point by log-loss
        de = pd.read_csv(d / "design_eval.csv"); w_ll = z["W_design"][int(de["log_loss"].idxmin())]
    add("slsqp_direct_logloss", w=w_ll)
    # direct AUC search: dense sample + polish
    W_cand = np.vstack([np.eye(M), np.full((1, M), 1 / M), z["W_design"], z["W_val"],
                        sample_dirichlet(M, ctx.args.auc_scan_points, alpha=1.0, random_state=seed + 3000),
                        sample_dirichlet(M, ctx.args.auc_scan_points, alpha=0.3, random_state=seed + 4000)])
    w_auc, auc_best, nfev = maximize_real_auc(P, y, W_cand, seed=seed)
    counts["direct_auc_evals"] = int(nfev)
    add("direct_auc_search", w=w_auc, detail={"n_evals": int(nfev)})
    # Scheffé-derived optima (selected order surfaces)
    from mixens.optimize import minimize_on_simplex
    m_ll, ord_ll = scheffe_surface(sch, "log_loss")
    m_auc, ord_auc = scheffe_surface(sch, "roc_auc")
    w_sll = minimize_on_simplex(lambda w: float(m_ll.predict_weights(w[None, :])[0]), M, random_state=seed)
    w_sauc = minimize_on_simplex(lambda w: -float(m_auc.predict_weights(w[None, :])[0]), M, random_state=seed)
    add("scheffe_optimum_logloss", w=w_sll, detail={"order": ord_ll, "predicted": float(m_ll.predict_weights(w_sll[None, :])[0])})
    add("scheffe_optimum_auc", w=w_sauc, detail={"order": ord_auc, "predicted": float(m_auc.predict_weights(w_sauc[None, :])[0])})
    # real anchors
    w_cost = np.eye(M)[int(np.argmin(costs))]
    add("cheapest_vertex", w=w_cost, detail={"model": names[int(np.argmin(costs))]})
    out = {"references": refs, "real_anchors": {"auc": w_auc.tolist(), "logloss": w_ll.tolist(), "cost": w_cost.tolist()},
           "surrogate_anchors": {"auc": w_sauc.tolist(), "logloss": w_sll.tolist(), "cost": w_cost.tolist()},
           "counts": counts, "model_oof_auc": dict(zip(names, map(float, aucs)))}
    write_json_atomic(out, d / "references.json")
    return {k: {"oof_roc_auc": round(v["oof_roc_auc"], 5), "oof_log_loss": round(v["oof_log_loss"], 5)}
            for k, v in refs.items()}


def stage_reference(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    P, y, names, costs = oof["P"], oof["y_train"], oof["names"], oof["costs"]
    M = len(names); seed = ctx.args.base_seed + rep
    refs = read_json(d / "references.json")
    z = np.load(d / "design_points.npz", allow_pickle=False)
    n_pts = ctx.args.reference_points
    t0 = time.perf_counter()
    blocks: list[tuple[str, np.ndarray]] = []
    blocks.append(("vertex", np.eye(M)))
    blocks.append(("centroid", np.full((1, M), 1 / M)))
    edge = generate_simplex_lattice(2, 20)[1:-1]
    E = []
    for i in range(M):
        for j in range(i + 1, M):
            for a, b in edge:
                w = np.zeros(M); w[i] = a; w[j] = b; E.append(w)
    blocks.append(("edge", np.array(E)))
    lat = generate_simplex_lattice(M, 6)
    blocks.append(("lattice6", lat[np.count_nonzero(lat, axis=1) >= 3]))
    blocks.append(("design", z["W_design"]))
    blocks.append(("validation", z["W_val"]))
    blocks.append(("refs", np.array([v["w"] for v in refs["references"].values() if "w" in v])))
    # exact epsilon-constraint sweep on log-loss x weighted cost (convex, exact)
    caps = np.linspace(costs.min(), max(float(np.asarray(refs["references"]["slsqp_direct_logloss"]["w"]) @ costs),
                                        costs.min() * 1.01), ctx.args.eps_caps)
    EC = []
    w_prev = np.eye(M)[int(np.argmin(costs))]
    for cap in caps:
        w = minimize_real_logloss(P, y, cost_vec=costs, cost_cap=cap, n_starts=3, seed=seed)
        if w is not None:
            EC.append(w); w_prev = w
    blocks.append(("eps_constraint", np.array(EC)))
    # random part (shuffled interleave of alpha=1 and alpha=0.3 so prefix diagnostics are unbiased)
    half = n_pts // 2
    R1 = sample_dirichlet(M, half, alpha=1.0, random_state=seed + 5000)
    R3 = sample_dirichlet(M, n_pts - half, alpha=0.3, random_state=seed + 6000)
    rng = np.random.default_rng(seed + 7000)
    Rall = np.vstack([R1, R3]); src_r = np.array(["dir1"] * len(R1) + ["dir03"] * len(R3))
    perm = rng.permutation(len(Rall)); Rall = Rall[perm]; src_r = src_r[perm]
    W = np.vstack([b for _, b in blocks] + [Rall])
    src = np.concatenate([np.array([lab] * len(b)) for lab, b in blocks] + [src_r])
    df = eval_W(oof, W, n_jobs=ctx.args.n_jobs, with_pr_auc=False)
    df.insert(0, "source", src)
    F_w = objectives(df, "cost_weighted"); F_s = objectives(df, "cost_support")
    m_w = pt.fast_pareto_mask(F_w); m_s = pt.fast_pareto_mask(F_s)
    # convergence diagnostics on the random part
    n_struct = sum(len(b) for _, b in blocks)
    conv = pt.front_convergence(F_w[n_struct:], fractions=(0.1, 0.25, 0.5, 0.75, 1.0))
    check = np.vstack([sample_dirichlet(M, ctx.args.reference_check_points // 2, alpha=1.0, random_state=seed + 8000),
                       sample_dirichlet(M, ctx.args.reference_check_points // 2, alpha=0.3, random_state=seed + 9000)])
    dfc = eval_W(oof, check, n_jobs=ctx.args.n_jobs, with_pr_auc=False)
    Fc = objectives(dfc, "cost_weighted")
    both = np.vstack([F_w[m_w], Fc]); mb = pt.fast_pareto_mask(both)
    displaced = float(1.0 - mb[: int(m_w.sum())].mean())
    rounds = 1
    # extend the sample once if the independent check displaces > 5% of the front (documented threshold)
    while displaced > ctx.args.reference_displace_tol and rounds < ctx.args.reference_max_rounds:
        rounds += 1
        extra = np.vstack([sample_dirichlet(M, half, alpha=1.0, random_state=seed + 10000 * rounds),
                           sample_dirichlet(M, n_pts - half, alpha=0.3, random_state=seed + 10000 * rounds + 1)])
        dfe = eval_W(oof, extra, n_jobs=ctx.args.n_jobs, with_pr_auc=False)
        dfe.insert(0, "source", np.array([f"extra{rounds}"] * len(extra)))
        df = pd.concat([df, dfe], ignore_index=True)
        F_w = objectives(df, "cost_weighted"); F_s = objectives(df, "cost_support")
        m_w = pt.fast_pareto_mask(F_w); m_s = pt.fast_pareto_mask(F_s)
        both = np.vstack([F_w[m_w], Fc]); mb = pt.fast_pareto_mask(both)
        displaced = float(1.0 - mb[: int(m_w.sum())].mean())
        ctx.log(f"    reference round {rounds}: {len(df)} points, front {int(m_w.sum())}, displaced {displaced:.3f}")
    df["nd_weighted"] = m_w; df["nd_support"] = m_s
    wcols = [f"w_{n}" for n in names]
    np.savez_compressed(d / "reference_sample.npz", W=df[wcols].to_numpy(np.float32),
                        roc_auc=df["roc_auc"].to_numpy(np.float32), log_loss=df["log_loss"].to_numpy(np.float32),
                        brier=df["brier"].to_numpy(np.float32), cost_weighted=df["cost_weighted"].to_numpy(np.float32),
                        cost_support=df["cost_support"].to_numpy(np.float32), n_eff=df["n_eff"].to_numpy(np.float32),
                        source=df["source"].to_numpy().astype("U16"), nd_weighted=m_w, nd_support=m_s)
    df[m_w].to_csv(d / "reference_front_weighted.csv", index=False)
    df[m_s].to_csv(d / "reference_front_support.csv", index=False)
    front_src = df.loc[m_w, "source"].value_counts().to_dict()
    diag = {"n_points": int(len(df)), "n_random_points": int(len(df) - n_struct), "rounds": rounds,
            "front_weighted_size": int(m_w.sum()), "front_support_size": int(m_s.sum()),
            "front_weighted_sources": {k: int(v) for k, v in front_src.items()},
            "front_support_sources": {k: int(v) for k, v in df.loc[m_s, "source"].value_counts().items()},
            "prefix_convergence_random_part": conv,
            "independent_check": {"n_check_points": int(len(check)), "front_displaced_fraction": displaced,
                                  "tolerance": ctx.args.reference_displace_tol},
            "eps_constraint_caps": caps.tolist(), "n_eps_constraint_solutions": int(len(EC)),
            "front_weighted_ranges": {"roc_auc": [float(df.loc[m_w, "roc_auc"].min()), float(df.loc[m_w, "roc_auc"].max())],
                                      "log_loss": [float(df.loc[m_w, "log_loss"].min()), float(df.loc[m_w, "log_loss"].max())],
                                      "cost_weighted": [float(df.loc[m_w, "cost_weighted"].min()), float(df.loc[m_w, "cost_weighted"].max())]},
            "seconds": time.perf_counter() - t0}
    write_json_atomic(diag, d / "reference_diagnostics.json")
    return {"n_points": diag["n_points"], "front_weighted": diag["front_weighted_size"],
            "front_support": diag["front_support_size"], "displaced": round(displaced, 4), "rounds": rounds}


def _nbi_common(ctx: Ctx, name: str, rep: int, variant: str) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    P, y, names, costs = oof["P"], oof["y_train"], oof["names"], oof["costs"]
    M = len(names); seed = ctx.args.base_seed + rep
    sch = read_json(d / "scheffe.json"); refs = read_json(d / "references.json")
    real_anchors = [np.asarray(refs["real_anchors"][k]) for k in ("auc", "logloss", "cost")]
    yb = y.astype(bool); Pf = P.astype(np.float32)
    n_real_evals = {"n": 0}

    if variant in ("A", "B"):
        m_auc, ord_auc = scheffe_surface(sch, "roc_auc")
        m_ll, ord_ll = scheffe_surface(sch, "log_loss")
        f_auc = lambda w: -float(m_auc.predict_weights(np.asarray(w)[None, :])[0])
        f_ll = lambda w: float(m_ll.predict_weights(np.asarray(w)[None, :])[0])
        labels = [f"neg_auc(scheffe:{ord_auc})", f"logloss(scheffe:{ord_ll})", "cost_weighted(exact)"]
        fd_eps = None; n_starts = ctx.args.nbi_starts
    else:
        def f_auc(w):
            n_real_evals["n"] += 1
            return -fm.fast_auc_no_ties(yb, Pf @ np.asarray(w, dtype=np.float32))
        def f_ll(w):
            n_real_evals["n"] += 1
            return fm.log_loss_vec(y, Pf @ np.asarray(w, dtype=np.float32))
        labels = ["neg_auc(real OOF)", "logloss(real OOF)", "cost_weighted(exact)"]
        fd_eps = ctx.args.nbi_c_fd_eps; n_starts = ctx.args.nbi_c_starts
    f_cost = lambda w: linear_cost(w, costs)
    t0 = time.perf_counter()
    res = run_nbi_on_simplex([f_auc, f_ll, f_cost], M, n_points=ctx.args.nbi_points, n_starts=n_starts,
                             seed=seed, anchors_w=None if variant == "A" else real_anchors, fd_eps=fd_eps,
                             maxiter=ctx.args.nbi_c_maxiter if variant == "C" else ctx.args.nbi_maxiter,
                             accept_feasible=(variant == "C"))
    secs = time.perf_counter() - t0
    W = np.array([c["w"] for c in res["candidates"]])
    df = eval_W(oof, W)
    meta = pd.DataFrame({"t": [c["t"] for c in res["candidates"]],
                         "residual_norm": [c["residual_norm"] for c in res["candidates"]],
                         "success": [c["success"] for c in res["candidates"]],
                         "nfev": [c["nfev"] for c in res["candidates"]],
                         **{f"beta_{i}": [c["beta"][i] for c in res["candidates"]] for i in range(3)},
                         **{f"F_norm_{i}": [c["F_normalized"][i] for c in res["candidates"]] for i in range(3)}})
    out = pd.concat([meta, df], axis=1)
    ok = out["success"].to_numpy(bool)
    Fw = objectives(out, "cost_weighted"); Fs = objectives(out, "cost_support")
    nd_w = np.zeros(len(out), bool); nd_s = np.zeros(len(out), bool)
    if ok.any():
        nd_w[ok] = pt.fast_pareto_mask(Fw[ok]); nd_s[ok] = pt.fast_pareto_mask(Fs[ok])
    out["nd_real_weighted"] = nd_w; out["nd_real_support"] = nd_s
    out.to_csv(d / f"nbi_{variant}_candidates.csv", index=False)
    anchors_eval = eval_W(oof, np.array(res["anchors_w"]))
    summary = {"variant": variant, "objective_labels": labels, "anchors_source": res["anchors_source"],
               "anchors_w": res["anchors_w"], "anchors_real_metrics": anchors_eval[["roc_auc", "log_loss", "cost_weighted", "cost_support"]].to_dict("records"),
               "payoff_raw": res["payoff_raw"], "utopia_raw": res["utopia_raw"], "pseudo_nadir_raw": res["pseudo_nadir_raw"],
               "payoff_normalized": res["payoff_normalized"], "n_hat": res["n_hat"],
               "n_subproblems": int(len(out)), "n_success": int(ok.sum()), "n_front_real_weighted": int(nd_w.sum()),
               "n_front_real_support": int(nd_s.sum()), "total_nfev": int(out["nfev"].sum()),
               "n_real_objective_evals": int(n_real_evals["n"]), "fd_eps": fd_eps, "n_starts": n_starts,
               "seconds": secs, "surrogate_reliable": {r: sch[r]["reliable"] for r in ("roc_auc", "log_loss")}}
    write_json_atomic(summary, d / f"nbi_{variant}_summary.json")
    return {"n_success": summary["n_success"], "n_front": summary["n_front_real_weighted"], "seconds": round(secs, 1)}


def stage_nbi_A(ctx, name, rep): return _nbi_common(ctx, name, rep, "A")
def stage_nbi_B(ctx, name, rep): return _nbi_common(ctx, name, rep, "B")
def stage_nbi_C(ctx, name, rep): return _nbi_common(ctx, name, rep, "C")


def stage_comparators(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    names, costs = oof["names"], oof["costs"]
    M = len(names); seed = ctx.args.base_seed + rep
    sch = read_json(d / "scheffe.json"); refs = read_json(d / "references.json")
    z = np.load(d / "design_points.npz", allow_pickle=False)
    n = ctx.args.nbi_points
    from mixens.nbi import anchors_from_points, objectives_on_free_vars, lift_simplex
    from mixens.optimize import minimize_on_simplex
    m_auc, _ = scheffe_surface(sch, "roc_auc"); m_ll, _ = scheffe_surface(sch, "log_loss")
    objs = [lambda w: -float(m_auc.predict_weights(np.asarray(w)[None, :])[0]),
            lambda w: float(m_ll.predict_weights(np.asarray(w)[None, :])[0]),
            lambda w: linear_cost(w, costs)]
    raw = objectives_on_free_vars(objs)
    anchors = anchors_from_points(raw, np.array([np.asarray(refs["real_anchors"][k])[: M - 1] for k in ("auc", "logloss", "cost")]))
    span = np.where(np.abs(anchors.pseudo_nadir - anchors.utopia) < 1e-12, 1.0, anchors.pseudo_nadir - anchors.utopia)
    rng = np.random.default_rng(seed + 11000)
    lambdas = rng.dirichlet(np.ones(3), size=n)
    WS = []
    for lam in lambdas:
        f = lambda w, lam=lam: float(np.dot(lam, (np.array([o(w) for o in objs]) - anchors.utopia) / span))
        WS.append(minimize_on_simplex(f, M, n_starts=3, random_state=seed))
    sets = {"ws_random_scalarization": np.array(WS),
            "random_dirichlet_budget": sample_dirichlet(M, n, alpha=1.0, random_state=seed + 12000),
            "design_runs": z["W_design"]}
    summary = {}
    for key, W in sets.items():
        df = eval_W(oof, W)
        Fw = objectives(df, "cost_weighted"); Fs = objectives(df, "cost_support")
        df["nd_real_weighted"] = pt.fast_pareto_mask(Fw); df["nd_real_support"] = pt.fast_pareto_mask(Fs)
        if key == "ws_random_scalarization":
            for i in range(3):
                df[f"lambda_{i}"] = lambdas[:, i]
        df.to_csv(d / f"comparator_{key}.csv", index=False)
        summary[key] = {"n": int(len(df)), "n_front_real_weighted": int(df["nd_real_weighted"].sum()),
                        "n_front_real_support": int(df["nd_real_support"].sum())}
    write_json_atomic(summary, d / "comparators_summary.json")
    return summary


def _quality_for_set(F_all: np.ndarray, ok: np.ndarray, F_ref_front: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                     F_ref_all: np.ndarray, rng: np.random.Generator) -> dict:
    """Quality indicators of a candidate set against the empirical reference front
    (common normalization lo/hi from the reference front). ``ok`` marks valid candidates."""
    F = F_all[ok] if ok.any() else np.empty((0, F_all.shape[1]))
    ref_n = pt.normalize(F_ref_front, lo, hi)
    hv_ref_point = np.full(3, 1.1)
    hv_ref = pt.hypervolume(ref_n, hv_ref_point)
    out = {"n_candidates": int(len(F_all)), "n_valid": int(ok.sum())}
    if len(F) == 0:
        return out | {"n_front": 0}
    m = pt.fast_pareto_mask(F)
    Fn_front = pt.normalize(F[m], lo, hi)
    Fn_all = pt.normalize(F, lo, hi)
    hv = pt.hypervolume(Fn_front, hv_ref_point)
    out.update({
        "n_front": int(m.sum()),
        "gd_front": pt.gd(Fn_front, ref_n), "gd_all_valid": pt.gd(Fn_all, ref_n),
        "igd": pt.igd(Fn_front, ref_n), "igd_plus": pt.igd_plus(Fn_front, ref_n),
        "spacing": pt.spacing(Fn_front), "spacing_cv": pt.spacing_cv(Fn_front),
        "hypervolume": hv, "hypervolume_reference": hv_ref, "hv_ratio": hv / hv_ref if hv_ref > 0 else float("nan"),
        "joint_nondominated_fraction_all_valid": pt.joint_nondominated_fraction(F, F_ref_all),
        "joint_nondominated_fraction_front": pt.joint_nondominated_fraction(F[m], F_ref_all),
        "coverage_ref_over_set": pt.coverage(ref_n, Fn_front), "coverage_set_over_ref": pt.coverage(Fn_front, ref_n),
        "extreme_gap": pt.extreme_point_recovery(Fn_front, ref_n),
    })
    k = int(m.sum())
    if 3 <= k < len(ref_n):
        sp = np.array([pt.spacing(ref_n[rng.choice(len(ref_n), k, replace=False)]) for _ in range(200)])
        out["spacing_size_matched_percentile"] = float(np.mean(sp < out["spacing"]))
        out["spacing_size_matched_mean"] = float(np.nanmean(sp))
    return out


def _support_key(w: np.ndarray, names: list[str]) -> str:
    return "+".join(n for n, v in zip(names, w) if v > SUPPORT_EPS)


def stage_quality(ctx: Ctx, name: str, rep: int) -> dict:
    oof = ctx.load_oof(name, rep)
    d = ctx.rep_dir(name, rep)
    names = oof["names"]; M = len(names); seed = ctx.args.base_seed + rep
    wcols = [f"w_{n}" for n in names]
    ref = np.load(d / "reference_sample.npz", allow_pickle=False)
    refs = read_json(d / "references.json")
    # candidate sets (real metrics already computed)
    sets: dict[str, pd.DataFrame] = {}
    for v in ("A", "B", "C"):
        sets[f"nbi_{v}"] = pd.read_csv(d / f"nbi_{v}_candidates.csv")
    for key in ("ws_random_scalarization", "random_dirichlet_budget", "design_runs"):
        sets[key] = pd.read_csv(d / f"comparator_{key}.csv")
    refdf = pd.DataFrame([{**{c: v["w"][j] for j, c in enumerate(wcols)}, "roc_auc": v["oof_roc_auc"],
                           "log_loss": v["oof_log_loss"], "cost_weighted": v["cost_weighted"],
                           "cost_support": v["cost_support"], "method": k}
                          for k, v in refs["references"].items() if "w" in v])
    sets["single_objective_refs"] = refdf
    result = {"support_eps": SUPPORT_EPS}
    rng = np.random.default_rng(seed + 13000)
    for cost_col, tag in (("cost_weighted", "weighted"), ("cost_support", "support")):
        F_ref_sample = np.column_stack([-ref["roc_auc"].astype(float), ref["log_loss"].astype(float), ref[cost_col].astype(float)])
        # final empirical reference = non-dominated(sample ∪ every candidate set)
        parts = [F_ref_sample] + [objectives(df, cost_col) for df in sets.values()]
        W_parts = [ref["W"].astype(float)] + [df[wcols].to_numpy(float) for df in sets.values()]
        src_parts = [ref["source"].astype(str)] + [np.array([k] * len(df)) for k, df in sets.items()]
        F_all = np.vstack(parts); W_all = np.vstack(W_parts); src_all = np.concatenate(src_parts)
        m_all = pt.fast_pareto_mask(F_all)
        F_front = F_all[m_all]; W_front = W_all[m_all]
        lo = F_front.min(axis=0); hi = F_front.max(axis=0)
        front_df = pd.DataFrame(W_front, columns=wcols)
        front_df["roc_auc"] = -F_front[:, 0]; front_df["log_loss"] = F_front[:, 1]; front_df[cost_col] = F_front[:, 2]
        front_df["source"] = src_all[m_all]; front_df["n_eff"] = fm.n_eff(W_front)
        front_df["support"] = [_support_key(w, names) for w in W_front]
        front_df.to_csv(d / f"empirical_reference_front_{tag}.csv", index=False)
        block = {"n_front": int(m_all.sum()), "n_points_total": int(len(F_all)),
                 "front_sources": {k: int(v) for k, v in pd.Series(src_all[m_all]).value_counts().items()},
                 "normalization_lo": lo.tolist(), "normalization_hi": hi.tolist(),
                 "front_support_distribution": {k: int(v) for k, v in front_df["support"].value_counts().items()},
                 "front_n_eff": {"min": float(front_df["n_eff"].min()), "median": float(front_df["n_eff"].median()),
                                 "max": float(front_df["n_eff"].max())},
                 "front_mean_weights": dict(zip(names, W_front.mean(axis=0).round(4).tolist())),
                 "front_largest_weight_share": dict(zip(names, (np.bincount(W_front.argmax(axis=1), minlength=M) / len(W_front)).round(4).tolist())),
                 "sets": {}}
        for key, df in sets.items():
            F = objectives(df, cost_col)
            ok = df["success"].to_numpy(bool) if "success" in df.columns else np.ones(len(df), bool)
            q = _quality_for_set(F, ok, F_front, lo, hi, F_all, rng)
            W = df[wcols].to_numpy(float)
            if ok.any():
                m = np.zeros(len(df), bool); m[ok] = pt.fast_pareto_mask(F[ok])
                q["front_support_distribution"] = {k: int(v) for k, v in pd.Series([_support_key(w, names) for w in W[m]]).value_counts().items()}
                q["front_mean_weights"] = dict(zip(names, W[m].mean(axis=0).round(4).tolist()))
                q["front_n_eff_median"] = float(np.median(fm.n_eff(W[m])))
                # MCDM picks + holdout confirmation
                picks = {}
                Fn = pt.normalize(F[m], lo, hi)
                idx_m = np.where(m)[0]
                for rule in (SelectionRule.KNEE, SelectionRule.TOPSIS, SelectionRule.DISTANCE_TO_UTOPIA):
                    try:
                        i, _ = select(Fn, rule)
                    except Exception:
                        continue
                    w = W[idx_m[i]]
                    picks[rule.value] = {"w": w.round(5).tolist(), "oof_roc_auc": float(-F[idx_m[i], 0]),
                                         "oof_log_loss": float(F[idx_m[i], 1]), cost_col: float(F[idx_m[i], 2]),
                                         **holdout_metrics(oof, w=w)}
                q["mcdm_picks"] = picks
            block["sets"][key] = q
        result[tag] = block
    # per-face fronts (weighted cost): best AUC/log-loss reachable per support set on the reference sample
    F_ref_sample = np.column_stack([-ref["roc_auc"].astype(float), ref["log_loss"].astype(float)])
    supports = np.array([_support_key(w, names) for w in ref["W"].astype(float)])
    faces = {}
    for s in pd.unique(supports):
        mm = supports == s
        Fs2 = F_ref_sample[mm]
        faces[s] = {"n_points": int(mm.sum()), "best_auc": float(-Fs2[:, 0].min()), "best_log_loss": float(Fs2[:, 1].min()),
                    "cost_support": float(ref["cost_support"][mm][0])}
    result["faces_reference_sample"] = faces
    write_json_atomic(result, d / "quality.json")
    return {tag: {k: {"n_front": v.get("n_front"), "igd_plus": round(v.get("igd_plus", float("nan")), 4),
                      "hv_ratio": round(v.get("hv_ratio", float("nan")), 4),
                      "joint_nd": round(v.get("joint_nondominated_fraction_all_valid", float("nan")), 3)}
                  for k, v in result[tag]["sets"].items()} for tag in ("weighted", "support")}


STAGE_FUNCS = {"oof": stage_oof, "design": stage_design, "scheffe": stage_scheffe, "refs": stage_refs,
               "reference": stage_reference, "nbi_A": stage_nbi_A, "nbi_B": stage_nbi_B, "nbi_C": stage_nbi_C,
               "comparators": stage_comparators, "quality": stage_quality}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_stage(ctx: Ctx, name: str, rep: int, stage: str) -> bool:
    st = ctx.status(name, rep).get(stage, {})
    if ctx.args.resume and st.get("status") == "done":
        return True
    attempts = int(st.get("attempts", 0))
    if attempts >= ctx.args.max_attempts:
        ctx.log(f"[{name}][rep {rep}][{stage}] skipped: {attempts} failed attempts")
        return False
    ctx.set_status(name, rep, stage, status="running", attempts=attempts + 1, started_at=now())
    t0 = time.perf_counter()
    try:
        summary = STAGE_FUNCS[stage](ctx, name, rep)
        secs = time.perf_counter() - t0
        ctx.set_status(name, rep, stage, status="done", seconds=round(secs, 2), finished_at=now(),
                       summary=summary, error=None)
        ctx.log(f"[{name}][rep {rep}][{stage}] done in {secs / 60:.1f} min :: {json.dumps(summary, default=_json_default)[:400]}")
        return True
    except Exception as exc:  # noqa: BLE001 — checkpoint the failure and continue
        secs = time.perf_counter() - t0
        tb = traceback.format_exc()
        ctx.set_status(name, rep, stage, status="failed", seconds=round(secs, 2), finished_at=now(),
                       error=f"{type(exc).__name__}: {exc}", traceback=tb[-4000:])
        ctx.log(f"[{name}][rep {rep}][{stage}] FAILED after {secs / 60:.1f} min: {type(exc).__name__}: {exc}")
        return False


def run_replication(ctx: Ctx, name: str, rep: int, stages: list[str]) -> bool:
    for stage in stages:
        if not run_stage(ctx, name, rep, stage):
            return False
    return True


def rep_seconds(ctx: Ctx, name: str, rep: int) -> float | None:
    st = ctx.status(name, rep)
    if all(st.get(s, {}).get("status") == "done" for s in STAGES):
        return float(sum(st[s]["seconds"] for s in STAGES))
    return None


def init_manifest(ctx: Ctx, datasets: list[str]) -> None:
    mpath = ctx.root / "benchmark_manifest.json"
    m = read_json(mpath, {})
    m.setdefault("created_at", now())
    m["git_commit"] = ds_mod.git_commit()
    m["environment"] = env_info()
    m["config"] = {k: v for k, v in vars(ctx.args).items()}
    m["stages"] = STAGES
    m["datasets"] = datasets
    m.setdefault("decisions", [])
    m.setdefault("status", {})
    write_json_atomic(m, mpath)


def record_decision(ctx: Ctx, text: str) -> None:
    mpath = ctx.root / "benchmark_manifest.json"
    m = read_json(mpath, {})
    m.setdefault("decisions", []).append({"at": now(), "decision": text})
    write_json_atomic(m, mpath)
    ctx.log(f"DECISION: {text}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(REPO / "experiments" / "pco213_postwork_benchmark"))
    ap.add_argument("--datasets", default=",".join(ds_mod.DATASET_NAMES))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--stages", default=",".join(STAGES))
    ap.add_argument("--run-all", action="store_true", help="run every dataset x replication x stage")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--base-seed", type=int, default=20260904)
    ap.add_argument("--sample-rows", type=int, default=None, help="DEBUG: stratified row subsample per dataset")
    ap.add_argument("--reference-points", type=int, default=100_000)
    ap.add_argument("--reference-check-points", type=int, default=20_000)
    ap.add_argument("--reference-displace-tol", type=float, default=0.05)
    ap.add_argument("--reference-max-rounds", type=int, default=3)
    ap.add_argument("--eps-caps", type=int, default=40)
    ap.add_argument("--auc-scan-points", type=int, default=20_000)
    ap.add_argument("--nbi-points", type=int, default=66)
    ap.add_argument("--nbi-starts", type=int, default=10)
    ap.add_argument("--nbi-c-starts", type=int, default=2)
    ap.add_argument("--nbi-c-fd-eps", type=float, default=1e-3)
    ap.add_argument("--nbi-c-maxiter", type=int, default=120)
    ap.add_argument("--nbi-maxiter", type=int, default=300)
    ap.add_argument("--cost-batch-rows", type=int, default=10_000)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--max-days", type=float, default=5.0)
    ap.add_argument("--reduced-reps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="tiny wiring run (sample rows, 1 rep, small reference)")
    args = ap.parse_args()
    if args.smoke:
        args.root = args.root.rstrip("/") + "_smoke" if "smoke" not in args.root else args.root
        args.reps = 1; args.sample_rows = args.sample_rows or 6000
        args.reference_points = 4000; args.reference_check_points = 2000; args.auc_scan_points = 1000
        args.nbi_points = 15; args.nbi_starts = 3; args.nbi_c_starts = 1; args.eps_caps = 8
        args.cost_batch_rows = 1000; args.nbi_maxiter = 100; args.nbi_c_maxiter = 60

    root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    log = Logger(root / "benchmark.log")
    ctx = Ctx(args, log)
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    init_manifest(ctx, datasets)
    log(f"benchmark start: datasets={datasets} reps={args.reps} stages={stages} root={root}")
    with open(root / "benchmark.pid", "w") as f:
        f.write(str(os.getpid()))

    R = args.reps
    # pilot pass: rep 0 of every dataset, then projection, then the rest
    order = [(d_, 0) for d_ in datasets]
    for d_, r in order:
        run_replication(ctx, d_, r, stages)
    measured = {d_: rep_seconds(ctx, d_, 0) for d_ in datasets}
    if all(v is not None for v in measured.values()) and R > 1:
        projected_days = sum(measured.values()) * R / 86400.0
        log(f"projection after pilot pass: per-rep seconds {measured} -> {projected_days:.2f} days for R={R}")
        m = read_json(root / "benchmark_manifest.json", {})
        m["runtime_projection"] = {"per_rep_seconds": measured, "R": R, "projected_days": projected_days}
        write_json_atomic(m, root / "benchmark_manifest.json")
        if projected_days > args.max_days and R > args.reduced_reps:
            record_decision(f"projected {projected_days:.2f} days for R={R} exceeds the {args.max_days}-day ceiling; "
                            f"reducing to R={args.reduced_reps} for ALL datasets")
            R = args.reduced_reps
            m = read_json(root / "benchmark_manifest.json", {}); m["effective_reps"] = R
            write_json_atomic(m, root / "benchmark_manifest.json")
    m = read_json(root / "benchmark_manifest.json", {}); m["effective_reps"] = R
    write_json_atomic(m, root / "benchmark_manifest.json")
    for d_ in datasets:
        for r in range(1, R):
            run_replication(ctx, d_, r, stages)
    # final summary of status
    done = {d_: sum(1 for r in range(R) if rep_seconds(ctx, d_, r) is not None) for d_ in datasets}
    log(f"benchmark finished: completed replications {done} of R={R}")
    m = read_json(root / "benchmark_manifest.json", {}); m["completed_replications"] = done; m["finished_at"] = now()
    write_json_atomic(m, root / "benchmark_manifest.json")


if __name__ == "__main__":
    main()
