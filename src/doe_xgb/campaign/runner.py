"""The confirmatory campaign runner: one checkpointed replication at a time.

Executes ``papers/xgboost_hpo_vrfnbi/protocol/EXPERIMENT_PROTOCOL.md`` at
``xgboost-hpo-protocol-v3``. Each (dataset, replication) unit runs seventeen stages
and checkpoints every one, so a resumed run never recomputes a completed expensive
stage.

Three invariants the code enforces rather than documents:

* the factor model and the response surfaces are fitted on the **design side only**
  and applied everywhere else;
* the external validation set is **audit-only**: its responses reach the gate
  diagnostics and nothing else;
* every method evaluates through its own isolated ``MethodView``, so physical
  memoization never becomes cross-method information.

Nothing here decides anything. Every threshold, budget and rule comes from the
frozen protocol.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..nbi_core import NBIConfig
from ..reporting import dominance_filter, dominated_fraction, hypervolume, igd_plus
from .arms import (Realizer, empirical_reference, run_historical_ws, run_nbi_arm,
                   run_ws_s, surrogate_reference, symmetric_weights)
from .design import (external_scores, external_validation_set, from_coded, gate_pass,
                     load_design, make_surrogate, fit_surface_backward, to_coded)
from .evaluation_cache import EvaluationCache
from .evaluator import BOUNDS, INT_PARAMS, PARAMS, RESPONSES, evaluate_config, init_worker
from .factor_model import fit_factor_model, raw_conflict

PROTOCOL_TAG = "xgboost-hpo-protocol-v3"
N_REPLICATIONS = 30
DATASETS = ("magic", "spambase", "adult", "bank_marketing")
SEED_BASE = 20260914

B_ANCHOR_PER_OBJECTIVE = 100      # real evaluations, NBI-R only
N_CANDIDATES = 20                 # weight-grid cardinality, shared by every arm
NSGA2_POP, NSGA2_GEN = 32, 12     # frozen, see protocol/NSGA2_DECISION.md

STAGES = ("split", "design", "factor_model", "surrogates", "external_validation",
          "historical_ws", "ws_s", "surrogate_anchors", "nbi_s", "empirical_anchors",
          "nbi_r", "direct_baselines", "candidate_revalidation", "reference_core",
          "augmented_reference", "holdout_confirmation", "metrics")


class MethodologicalFailure(RuntimeError):
    """A frozen assumption did not hold. The unit is recorded and excluded, not repaired."""


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------


def _np(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


@dataclass
class Checkpoint:
    root: Path

    def path(self, stage: str) -> Path:
        return self.root / f"{stage}.json"

    def done(self, stage: str) -> bool:
        p = self.path(stage)
        if not p.exists():
            return False
        try:
            return bool(json.loads(p.read_text()).get("_complete"))
        except Exception:
            return False

    def load(self, stage: str) -> dict:
        return json.loads(self.path(stage).read_text())

    def save(self, stage: str, payload: dict) -> dict:
        payload = {**payload, "_complete": True, "_stage": stage,
                   "_written_at": time.time()}
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.path(stage).with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=_np))
        tmp.replace(self.path(stage))     # atomic: a killed run leaves no half stage
        return payload


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------


def unit_seed(dataset: str, rep: int) -> int:
    return SEED_BASE + 1000 * (DATASETS.index(dataset) + 1) + rep


def realizer() -> Realizer:
    return Realizer(PARAMS, BOUNDS, list(INT_PARAMS))


def nbi_config(seed: int) -> NBIConfig:
    return NBIConfig(objective_count=2,
                     bounds=np.array([[-1.0, 1.0]] * len(PARAMS)),
                     integer_dims=tuple(PARAMS.index(p) for p in INT_PARAMS),
                     n_starts=10, seed=seed, maxiter=500)


def _evaluate_rows(view, rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        res = view.evaluate({k: r[k] for k in PARAMS})
        out.append({**{k: r[k] for k in PARAMS}, **res, "run": r.get("run", "")})
    return pd.DataFrame(out)


def run_unit(dataset: str, rep: int, root: Path, *, threads: int = 1,
             dry_run: bool = False) -> dict:
    """One (dataset, replication) unit, resumable at stage granularity."""
    seed = unit_seed(dataset, rep)
    unit_root = root / dataset / f"rep_{rep:02d}"
    ck = Checkpoint(unit_root)
    design, external = load_design(), external_validation_set()

    plan = {
        "dataset": dataset, "replication": rep, "seed": seed,
        "protocol_tag": PROTOCOL_TAG,
        "logical_budget": logical_budget_plan(),
        "stages": list(STAGES),
    }
    if dry_run:
        return plan

    init_worker(dataset, threads, seed)
    cache = EvaluationCache(unit_root / "evaluations.sqlite", dataset=dataset,
                            split_id=f"rep_{rep:02d}", seed=seed,
                            protocol_version=PROTOCOL_TAG)

    def compute(cfg: dict) -> dict:
        return evaluate_config(cfg)

    t0 = time.time()
    try:
        # ---- stage: design (shared, charged to every arm in the ledger) -------
        if ck.done("design"):
            design_df = pd.DataFrame(ck.load("design")["rows"])
        else:
            view = cache.view("design", compute)
            design_df = _evaluate_rows(view, design.to_dict("records"))
            ck.save("design", {"rows": design_df.to_dict("records"),
                               "n": len(design_df)})

        # ---- stage: factor model (DESIGN SIDE ONLY) --------------------------
        fm = fit_factor_model(design_df)
        if not ck.done("factor_model"):
            conflict_latent = float(pd.Series(fm.transform(design_df)["quality"]).corr(
                pd.Series(fm.transform(design_df)["cost"]), method="spearman"))
            conflict_raw = raw_conflict(design_df)
            if np.sign(conflict_latent) != np.sign(conflict_raw):
                raise MethodologicalFailure(
                    f"{dataset} rep {rep}: latent objective conflict {conflict_latent:+.3f} "
                    f"disagrees in sign with the raw-response conflict {conflict_raw:+.3f}. "
                    "The quality composite is inverted relative to the metrics it is built "
                    "from. Factors are NOT re-oriented after the fact; the unit is recorded "
                    "as a methodological failure.")
            ck.save("factor_model", {**fm.as_dict(),
                                     "objective_conflict_latent": conflict_latent,
                                     "objective_conflict_raw_responses": conflict_raw})

        Y = fm.objectives(design_df)                   # (88, 2), both minimized

        # ---- stage: surrogates (DESIGN SIDE ONLY) ----------------------------
        if ck.done("surrogates"):
            s = ck.load("surrogates")
            fits = [(list(map(tuple, t)), np.asarray(b)) for t, b in
                    zip(s["terms"], s["beta"])]
        else:
            fits = [fit_surface_backward(design_df, Y[:, j]) for j in range(2)]
            ck.save("surrogates", {"terms": [[list(t) for t in f[0]] for f in fits],
                                   "beta": [f[1].tolist() for f in fits],
                                   "n_terms": [len(f[0]) for f in fits],
                                   "coding": "coded units, backward elimination alpha=0.05"})
        surrogates = [make_surrogate(t, b) for t, b in fits]

        # ---- stage: external validation (AUDIT ONLY) -------------------------
        if not ck.done("external_validation"):
            view = cache.view("external_validation_audit", compute)
            ext_df = _evaluate_rows(view, external.to_dict("records"))
            Ye = fm.objectives(ext_df)                  # APPLIED, never refitted
            scores = [external_scores(design_df, Y[:, j], ext_df, Ye[:, j])
                      for j in range(2)]
            ck.save("external_validation", {
                "objectives": ["quality", "cost"],
                "scores": scores,
                "gate_pass": [gate_pass(s) for s in scores],
                "rows": ext_df.to_dict("records"),
                "note": ("audit-only: these responses reach the gate diagnostics and "
                         "nothing else. A gate failure changes no execution.")})

        # ---- the arms --------------------------------------------------------
        weights = symmetric_weights(N_CANDIDATES)
        cfg = nbi_config(seed)
        rz = realizer()

        if not ck.done("historical_ws"):
            hist = _run_historical(design_df, Y, fm, surrogates, rz, seed)
            ck.save("historical_ws", hist)

        if ck.done("surrogate_anchors"):
            pass
        else:
            ck.save("surrogate_anchors", {"source": "per-objective minimization of the "
                                                    "fitted surrogates over the coded box"})
        s_anchors, s_chim = surrogate_reference(surrogates, cfg)

        if not ck.done("ws_s"):
            ck.save("ws_s", run_ws_s(surrogates, cfg, s_anchors, rz, weights).as_dict())
        if not ck.done("nbi_s"):
            ck.save("nbi_s", run_nbi_arm("NBI-S", surrogates, cfg, s_anchors, s_chim,
                                         rz, weights).as_dict())

        # ---- empirical anchors (the only stage that spends B_anchor) ---------
        if ck.done("empirical_anchors"):
            ea = ck.load("empirical_anchors")
            x_star = np.asarray(ea["x_star"]); F_star = np.asarray(ea["F_star"])
        else:
            x_star, F_star, ea = _empirical_anchors(cache, compute, fm, cfg, seed)
            ck.save("empirical_anchors", ea)

        r_anchors, r_chim = empirical_reference(x_star, F_star, cfg)
        if not ck.done("nbi_r"):
            ck.save("nbi_r", run_nbi_arm("NBI-R", surrogates, cfg, r_anchors, r_chim,
                                         rz, weights).as_dict())

        # ---- stage: direct baselines ----------------------------------------
        budget = logical_budget_plan()["comparator_budget"]
        to_obj = lambda df: fm.objectives(df)          # noqa: E731
        if not ck.done("direct_baselines"):
            from . import baselines as B
            out = {}
            specs = [("grid", lambda v: B.coarse_grid(v, budget, seed)),
                     ("random", lambda v: B.random_search(v, budget, seed)),
                     ("bayes_quality", lambda v: B.bayesian_optimization(v, budget, seed, to_obj, 0)),
                     ("bayes_cost", lambda v: B.bayesian_optimization(v, budget, seed, to_obj, 1)),
                     ("tpe_quality", lambda v: B.tpe(v, budget, seed, to_obj, 0)),
                     ("tpe_cost", lambda v: B.tpe(v, budget, seed, to_obj, 1)),
                     ("nsga2", lambda v: B.nsga2(v, NSGA2_POP, NSGA2_GEN, seed, to_obj))]
            for name, fn in specs:
                df = fn(cache.view(name, compute))
                out[name] = {"n_rows": int(len(df)),
                             "rows": df.to_dict("records")}
            ck.save("direct_baselines", {"budget": budget, "methods": out,
                                         "single_objective_methods":
                                             ["bayes_quality", "bayes_cost",
                                              "tpe_quality", "tpe_cost"],
                                         "note": ("single-objective runs are reported on "
                                                  "endpoints only and never in the "
                                                  "front-indicator table")})

        # ---- stage: candidate revalidation on the REAL learner ---------------
        from .scoring import (augmented_reference, indicators, reference_core,
                              revalidate)
        if not ck.done("candidate_revalidation"):
            reval = {}
            for arm in ("historical_ws", "ws_s", "nbi_s", "nbi_r"):
                emitted = [c["config_realized"] for c in ck.load(arm)["candidates"]]
                r = revalidate(emitted, cache.view(f"{arm}_revalidation", compute), to_obj)
                reval[arm] = {k: v for k, v in r.items()
                              if k not in ("rows", "objectives", "nondominated_index")}
                reval[arm]["rows"] = r["rows"].to_dict("records")
            ck.save("candidate_revalidation", {"arms": reval})

        # ---- stage: the two references ---------------------------------------
        if not ck.done("reference_core"):
            ea = ck.load("empirical_anchors")
            anchor_rows = pd.DataFrame(
                [r for r in ck.load("design")["rows"]])       # design is method-independent
            core = reference_core([anchor_rows], to_obj)
            ck.save("reference_core", {k: v for k, v in core.items() if k != "front"}
                    | {"front": core["front"].tolist()})

        if not ck.done("augmented_reference"):
            core_front = np.asarray(ck.load("reference_core")["front"], dtype=float)
            per = {}
            rv = ck.load("candidate_revalidation")["arms"]
            for arm, payload in rv.items():
                df = pd.DataFrame(payload["rows"])
                per[arm] = to_obj(df) if len(df) else np.zeros((0, 2))
            db = ck.load("direct_baselines")["methods"]
            for name, payload in db.items():
                df = pd.DataFrame(payload["rows"])
                per[name] = to_obj(df) if len(df) else np.zeros((0, 2))
            aug = augmented_reference(core_front, per)
            ck.save("augmented_reference",
                    {k: v for k, v in aug.items() if k != "front"}
                    | {"front": aug["front"].tolist()})

        # ---- stage: metrics against both references --------------------------
        if not ck.done("metrics"):
            core_front = np.asarray(ck.load("reference_core")["front"], dtype=float)
            aug_front = np.asarray(ck.load("augmented_reference")["front"], dtype=float)
            rv = ck.load("candidate_revalidation")["arms"]
            db = ck.load("direct_baselines")["methods"]
            per_method = {}
            for name, payload in list(rv.items()) + list(db.items()):
                df = pd.DataFrame(payload["rows"])
                F = to_obj(df) if len(df) else np.zeros((0, 2))
                per_method[name] = {
                    "augmented": indicators(F, aug_front),
                    "core": indicators(F, core_front)}
            ck.save("metrics_by_method", {"methods": per_method,
                                          "primary_indicator": "hv_ratio",
                                          "references": ["augmented", "core"]})

        # ---- stage: holdout confirmation (labels untouched until here) -------
        if not ck.done("holdout_confirmation"):
            rv = ck.load("candidate_revalidation")["arms"]
            ho = {}
            for arm, payload in rv.items():
                df = pd.DataFrame(payload["rows"])
                if not len(df):
                    continue
                F = to_obj(df)
                sel = df.iloc[int(np.argmin(np.linalg.norm(
                    (F - F.min(axis=0)) / np.where(F.ptp(axis=0) == 0, 1, F.ptp(axis=0)),
                    axis=1)))]
                cfg_sel = {p: (int(sel[p]) if p in INT_PARAMS else float(sel[p]))
                           for p in PARAMS}
                ho[arm] = {"selected_config": cfg_sel,
                           "internal": {k: float(sel[k]) for k in RESPONSES},
                           "holdout": evaluate_config(cfg_sel, on_holdout=True)}
            ck.save("holdout_confirmation", {
                "selection_rule": "distance to the utopia of the arm's own revalidated set",
                "arms": ho,
                "note": "holdout labels are read here and nowhere earlier"})

        ck.save("split", {"seed": seed, "test_size": 0.2,
                          "stratified": True, "inner_folds": 5})
        result = {"dataset": dataset, "replication": rep, "seed": seed,
                  "wall_seconds": round(time.time() - t0, 1),
                  "accounting": cache.accounting(),
                  "stages_complete": [s for s in STAGES if ck.done(s)]}
        ck.save("metrics", result)
        return result
    except MethodologicalFailure as exc:
        ck.save("methodological_failure", {"message": str(exc),
                                           "traceback": traceback.format_exc()})
        raise
    finally:
        cache.close()


def _run_historical(design_df, Y, fm, surrogates, rz, seed) -> dict:
    """HISTORICAL-WS: the dissertation's own normalization and its own grid.

    The surfaces are fitted in the historical *uncoded* parameterization and the
    normalization box is the component-wise observed extrema of the design rows,
    as scripts/run_nbi.py built it. Neither is repaired.
    """
    import statsmodels.api as sm      # noqa: F401  (kept for parity of the fit path)
    from .design import fit_surface_backward as _fit
    # the historical box: component-wise observed extrema, both objectives minimized
    observed_utopia = tuple(float(Y[:, j].min()) for j in range(2))
    observed_nadir = tuple(float(Y[:, j].max()) for j in range(2))
    terms_beta = [_fit(design_df, Y[:, j]) for j in range(2)]
    models = []
    for terms, beta in terms_beta:
        names, coefs = [], []
        for tm, b in zip(terms, beta):
            if not tm:
                names.append("Intercept")
            elif len(tm) == 1:
                names.append(PARAMS[tm[0]])
            elif tm[0] == tm[1]:
                names.append(f"{PARAMS[tm[0]]}^2")
            else:
                names.append(f"{PARAMS[tm[0]]}*{PARAMS[tm[1]]}")
            coefs.append(float(b))
        models.append((names, coefs))
    run = run_historical_ws(models[0], models[1],
                            observed_utopia=observed_utopia,
                            observed_nadir=observed_nadir,
                            bounds={p: BOUNDS[p] for p in PARAMS},
                            realizer=rz, surrogates_coded=surrogates, seed=seed)
    return run.as_dict()


def _empirical_anchors(cache, compute, fm, cfg, seed) -> tuple[np.ndarray, np.ndarray, dict]:
    """Direct search on the REAL objectives, one budget per objective.

    These are the best configurations the search found within its declared budget.
    They are not certified optima and are never described as such.
    """
    rng = np.random.default_rng(seed + 77)
    view = cache.view("empirical_anchor_search", compute)
    k = len(PARAMS)
    x_star = np.zeros((2, k))
    rows_per_objective = []
    for j in range(2):
        best_x, best_v, rows = None, np.inf, []
        for _ in range(B_ANCHOR_PER_OBJECTIVE):
            xc = rng.uniform(-1.0, 1.0, size=k)
            cfg_nat, xc_real = Realizer(PARAMS, BOUNDS, list(INT_PARAMS)).realize(xc)
            res = view.evaluate(cfg_nat)
            f = fm.objectives(pd.DataFrame([res]))[0]
            rows.append({"config": cfg_nat, "objectives": f.tolist()})
            if f[j] < best_v:
                best_v, best_x = float(f[j]), xc_real
        x_star[j] = best_x
        rows_per_objective.append(rows)
    # payoff matrix: every objective measured at every anchor, on the REAL objectives
    F_star = np.zeros((2, 2))
    for j in range(2):
        cfg_nat, _ = Realizer(PARAMS, BOUNDS, list(INT_PARAMS)).realize(x_star[j])
        res = view.evaluate(cfg_nat)
        F_star[:, j] = fm.objectives(pd.DataFrame([res]))[0]
    return x_star, F_star, {
        "x_star": x_star.tolist(), "F_star": F_star.tolist(),
        "budget_per_objective": B_ANCHOR_PER_OBJECTIVE,
        "search": "uniform random over the coded box, realized before evaluation",
        "note": ("best found within the declared budget; NOT certified optima and "
                 "never described as such"),
        "evaluations_by_objective": [len(r) for r in rows_per_objective]}


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------


def logical_budget_plan(q: int = 2) -> dict:
    design_n, ext_n = 88, 78
    arms = {
        "HISTORICAL-WS": design_n + N_CANDIDATES,
        "WS-S": design_n + ext_n + N_CANDIDATES,
        "NBI-S": design_n + ext_n + N_CANDIDATES,
        "NBI-R": design_n + ext_n + B_ANCHOR_PER_OBJECTIVE * q + N_CANDIDATES,
    }
    comparator = max(arms.values())
    return {"B_design": design_n, "B_external_validation": ext_n,
            "B_external_validation_is_audit_only": True,
            "B_anchor_per_objective": B_ANCHOR_PER_OBJECTIVE,
            "B_candidate_validation": N_CANDIDATES,
            "B_total_solution_per_arm": arms,
            "comparator_budget": comparator,
            "nsga2": {"population": NSGA2_POP, "generations": NSGA2_GEN,
                      "evaluations": NSGA2_POP * NSGA2_GEN,
                      "shortfall_against_comparator_budget":
                          comparator - NSGA2_POP * NSGA2_GEN},
            "note": ("logical budget is what a method would consume running alone and is "
                     "the only figure entering a fairness comparison; unique physical "
                     "fits after memoization are reported separately and enter none")}


__all__ = ["run_unit", "logical_budget_plan", "unit_seed", "STAGES", "DATASETS",
           "N_REPLICATIONS", "PROTOCOL_TAG", "MethodologicalFailure", "Checkpoint"]
