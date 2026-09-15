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
from .factor_model import (composite_alignment, fit_factor_model,
                           load_reference_factor_model, raw_conflict,
                           tucker_congruence)
from .seeding import candidate_hash, derive_seed, seed_key

PROTOCOL_TAG = "xgboost-hpo-protocol-v3"
N_REPLICATIONS = 30
DATASETS = ("magic", "spambase", "adult", "bank_marketing")
SEED_BASE = 20260914

B_ANCHOR_PER_OBJECTIVE = 100      # real evaluations, NBI-R only
N_CANDIDATES = 20                 # weight-grid cardinality, shared by every arm
NSGA2_POP, NSGA2_GEN = 32, 12     # frozen, see protocol/NSGA2_DECISION.md
# The unmatched run defends the matched result against the objection that a
# 12-generation population is starved. It is a clearly labelled secondary,
# unmatched-context run, never an efficiency comparison, and it is scoped to ONE
# replication per dataset because on all thirty it would cost about 180 hours.
NSGA2_UNMATCHED_MULTIPLIER = 10
NSGA2_UNMATCHED_REPLICATION = 0

# Two single-objective runs are not a Pareto front. These are reported on their own
# endpoints only: they never enter the front-indicator table, and they never enter
# the common augmented reference, where a pure cost-minimizer would pin the cost
# extreme and move the normalization box every other method is scored against.
SINGLE_OBJECTIVE = ("bayes_quality", "bayes_cost", "tpe_quality", "tpe_cost")

# Every comparator the direct-baselines stage produces, by exact identifier. The
# augmented-reference and metrics stages iterate this namespace and score whatever
# they find in it, so it must contain methods and nothing else.
SCORED_BASELINES = ("grid", "random", "bayes_quality", "bayes_cost",
                    "tpe_quality", "tpe_cost", "nsga2")

STAGES = ("split", "design", "factor_model", "surrogates", "external_validation",
          "historical_ws_asrun", "ws_s", "historical_ws", "surrogate_anchors", "nbi_s",
          "empirical_anchors", "nbi_r", "direct_baselines", "nsga2_unmatched",
          "candidate_revalidation",
          "anchor_injection_control", "reference_core", "augmented_reference",
          "holdout_confirmation", "metrics")


class MethodologicalFailure(RuntimeError):
    """A frozen assumption did not hold. The unit is recorded and excluded, not repaired."""


def _baseline_methods(payload: dict) -> dict:
    """The comparators in a ``direct_baselines`` checkpoint, namespace validated.

    The seed ledger was once written into this namespace alongside the methods.
    Both the augmented-reference and the metrics stage iterate it and score every
    entry, so a metadata key carrying a ``rows`` field would have been scored into
    the shared reference without any error. It crashed instead, on a missing key,
    which was luck rather than a safeguard. This is the safeguard.
    """
    methods = payload["methods"]
    unknown = set(methods) - set(SCORED_BASELINES)
    if unknown:
        raise MethodologicalFailure(
            f"the direct-baselines methods namespace carries non-method entries "
            f"{sorted(unknown)}. Everything in it is iterated as a scored method by "
            f"the augmented-reference and metrics stages, so anything stored here "
            f"enters the shared reference and the indicator table.")
    missing = set(SCORED_BASELINES) - set(methods)
    if missing:
        raise MethodologicalFailure(
            f"direct baselines did not produce {sorted(missing)}; the comparator "
            f"set is incomplete and the reference would be built from a subset.")
    return methods


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

    def _compute_holdout(cfg: dict) -> dict:
        # A different measurement of the same configuration: the held-out partition
        # rather than the inner resampling. It is keyed under fold_id="holdout" so it
        # can never be served from, or serve, an internal evaluation of that config.
        return evaluate_config(cfg, on_holdout=True)

    t0 = time.time()
    try:
        # ---- stage: design (shared, charged to every arm in the ledger) -------
        if ck.done("design"):
            design_df = pd.DataFrame(ck.load("design")["rows"])
        else:
            view = cache.view("design", compute, stage="design")
            design_df = _evaluate_rows(view, design.to_dict("records"))
            ck.save("design", {"rows": design_df.to_dict("records"),
                               "n": len(design_df)})

        # ---- stage: factor model (APPLIED, never fitted here) ----------------
        # EXPERIMENT_PROTOCOL.md 7.2: ONE factor model per dataset, fitted on the
        # 166-point Stage A reference set and APPLIED to every replication. This
        # stage previously called fit_factor_model(design_df), refitting on each
        # replication's own design evaluations -- which is precisely the confound
        # 7.2 exists to remove: with a per-replication fit the objective is not the
        # same variable in every pair, so the 30 paired indicator values do not live
        # in one objective space and no normalized indicator is invariant to that.
        fm = load_reference_factor_model(dataset)
        if not ck.done("factor_model"):
            t_d = fm.transform(design_df)
            conflict_latent = float(pd.Series(t_d["quality"]).corr(
                pd.Series(t_d["cost"]), method="spearman"))
            conflict_raw = raw_conflict(design_df)
            alignment = composite_alignment(fm, design_df)

            # HARD invariant: the composite must point the same way as the badness
            # it aggregates. A negative alignment would mean the study is optimizing
            # toward worse models.
            #
            # The model is frozen per dataset, so the ORIENTATION this guards is a
            # property of the committed artifact and is verified once, before the
            # campaign, by the dry run and by the test suite. Here the same
            # quantity is recomputed against this replication's own design
            # measurements and REPORTED. It is deliberately not fatal at this point:
            # the frozen composite's alignment on the spambase reference set is only
            # +0.249, so a per-replication sign test on resampled data would abort
            # units at a rate driven by sampling noise and would do so non-randomly
            # with respect to the factor structure -- which would make the panel a
            # function of that noise.
            if alignment <= 0.0:
                ck.save("factor_alignment_warning", {
                    "dataset": dataset, "replication": rep, "alignment": alignment,
                    "note": ("the frozen composite's alignment measured NEGATIVE on "
                             "this replication's design sample. The frozen model is "
                             "unchanged and the unit continues; this is reported as "
                             "a covariate, exactly as the surrogate gate is.")})

            # The per-replication refit, reported as the sensitivity 7.2 requires:
            # Tucker congruence between the frozen reference loadings and the model
            # this replication would have produced had it fitted its own.
            refit = fit_factor_model(design_df)
            phi = tucker_congruence(fm.rotated_loadings, refit.rotated_loadings)

            ck.save("factor_model", {**fm.as_dict(),
                                     "source": "frozen reference model, APPLIED",
                                     "protocol_clause": "EXPERIMENT_PROTOCOL.md 7.2",
                                     "fitted_at_campaign_time": False,
                                     "objective_conflict_latent": conflict_latent,
                                     "objective_conflict_raw_responses": conflict_raw,
                                     "composite_alignment_with_raw_quality": alignment,
                                     "per_replication_refit_sensitivity": {
                                         "tucker_congruence": phi.tolist(),
                                         "min_abs_congruence": float(np.abs(phi).min()),
                                         "quality_weights_refit":
                                             refit.quality_weights.tolist(),
                                         "cost_index_refit": int(refit.cost_index),
                                         "cost_index_agrees":
                                             bool(refit.cost_index == fm.cost_index),
                                         "note": ("reported only; the refit is NEVER "
                                                  "applied. Conventional reading: "
                                                  "|phi| >= 0.95 equivalence, 0.85 to "
                                                  "0.95 fair similarity")},
                                     "conflict_divergence_note":
                                         ("the latent conflict is a Spearman between "
                                          "two axes that are Pearson-orthogonal BY "
                                          "CONSTRUCTION -- the quality composite is a "
                                          "weighted sum of rotated factors and the "
                                          "cost factor is another from the same "
                                          "orthogonal basis. Exactly zero on the "
                                          "166-point set the model was fitted to; at "
                                          "n = 88 the identity is approximate, |r| "
                                          "p95 = 0.14 to 0.17, and is NOT bounded by "
                                          "any small constant. The rank statistic is "
                                          "stable within a dataset but has no "
                                          "consistent direction across the panel. The "
                                          "conflict between the RAW responses is the "
                                          "meaningful quantity and is reported "
                                          "beside it. See "
                                          "audits/latent_conflict_stability.json.")})

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
            view = cache.view("external_validation_audit", compute, stage="external_audit")
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

        if not ck.done("historical_ws_asrun"):
            # Exactly as the dissertation ran it: the frozen solver, its own uncoded
            # surfaces, its own observed-extrema box, its own asymmetric 20-point
            # grid whose pure-quality vertex is absent.
            ck.save("historical_ws_asrun",
                    _run_historical(design_df, Y, fm, surrogates, rz, seed))

        if ck.done("surrogate_anchors"):
            pass
        else:
            ck.save("surrogate_anchors", {"source": "per-objective minimization of the "
                                                    "fitted surrogates over the coded box"})
        s_anchors, s_chim = surrogate_reference(surrogates, cfg)

        if not ck.done("ws_s"):
            ck.save("ws_s", run_ws_s(surrogates, cfg, s_anchors, rz, weights).as_dict())
        if not ck.done("historical_ws"):
            # The same weighted sum, the same solver, the same symmetric grid and the
            # same surrogates as WS-S, differing ONLY in the reference: the
            # dissertation's component-wise observed extrema of the design rows in
            # place of the payoff matrix. That is what makes WS-S to HISTORICAL-WS a
            # single-factor contrast on the normalization. HISTORICAL-WS-asrun is the
            # bit-faithful reproduction and the two are never mixed in one table.
            hist_lo = Y.min(axis=0)
            hist_hi = Y.max(axis=0)
            ck.save("historical_ws", run_ws_s(
                surrogates, cfg, s_anchors, rz, weights, arm="HISTORICAL-WS",
                reference_override=(hist_lo, hist_hi),
                reference_label=("component-wise observed extrema of the design rows, "
                                 "as the dissertation normalizes")).as_dict())
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
            def ms(method: str) -> int:
                return derive_seed(dataset, rep, method)

            specs = [("grid", lambda v: B.coarse_grid(v, budget, ms("grid"))),
                     ("random", lambda v: B.random_search(v, budget, ms("random"))),
                     ("bayes_quality", lambda v: B.bayesian_optimization(
                         v, budget, ms("bayes_quality"), to_obj, 0)),
                     ("bayes_cost", lambda v: B.bayesian_optimization(
                         v, budget, ms("bayes_cost"), to_obj, 1)),
                     ("tpe_quality", lambda v: B.tpe(v, budget, ms("tpe_quality"), to_obj, 0)),
                     ("tpe_cost", lambda v: B.tpe(v, budget, ms("tpe_cost"), to_obj, 1)),
                     ("nsga2", lambda v: B.nsga2(v, NSGA2_POP, NSGA2_GEN,
                                                 ms("nsga2"), to_obj))]
            seed_ledger = {}
            for name, fn in specs:
                df = fn(cache.view(name, compute, stage="direct_search"))
                out[name] = {"n_rows": int(len(df)),
                             "rows": df.to_dict("records")}
                seed_ledger[name] = {
                    "base_replication_seed": seed,
                    "derived_method_seed": ms(name),
                    "seed_key": seed_key(dataset, rep, name),
                    "candidate_hash": candidate_hash(
                        df[list(PARAMS)].to_dict("records")) if len(df) else None}
            # The seed ledger is metadata and is stored at the TOP level only. It
            # was also being written into `out`, the methods namespace, where the
            # augmented-reference and metrics stages iterate every entry as a
            # scored method. It crashed there on a missing "rows" key -- loudly,
            # which was luck: a metadata entry that happened to carry "rows" would
            # have been scored into the shared reference in silence.
            ck.save("direct_baselines", {"budget": budget, "methods": out,
                                         "seed_ledger": seed_ledger,
                                         "single_objective_methods":
                                             ["bayes_quality", "bayes_cost",
                                              "tpe_quality", "tpe_cost"],
                                         "note": ("single-objective runs are reported on "
                                                  "endpoints only and never in the "
                                                  "front-indicator table")})

        # ---- stage: the unmatched NSGA-II run, one replication per dataset ----
        if rep == NSGA2_UNMATCHED_REPLICATION and not ck.done("nsga2_unmatched"):
            from . import baselines as B
            gens = NSGA2_GEN * NSGA2_UNMATCHED_MULTIPLIER
            df = B.nsga2(cache.view("nsga2_unmatched", compute, stage="unmatched_context"), NSGA2_POP, gens,
                         derive_seed(dataset, rep, "nsga2_unmatched"), to_obj)
            ck.save("nsga2_unmatched", {
                "population": NSGA2_POP, "generations": gens,
                "evaluations": NSGA2_POP * gens,
                "multiplier_over_matched_budget": NSGA2_UNMATCHED_MULTIPLIER,
                "n_rows": int(len(df)), "rows": df.to_dict("records"),
                "note": ("secondary, unmatched-context run on one replication per "
                         "dataset. It answers whether budget starvation explains the "
                         "matched result. It is NOT an efficiency comparison, it "
                         "enters no front-indicator table beside the matched methods, "
                         "and no claim about any arm is made relative to it.")})

        # ---- stage: candidate revalidation on the REAL learner ---------------
        from .scoring import (augmented_reference, indicators, reference_core,
                              revalidate)
        if not ck.done("candidate_revalidation"):
            reval = {}
            for arm in ("historical_ws_asrun", "historical_ws", "ws_s", "nbi_s", "nbi_r"):
                emitted = [c["config_realized"] for c in ck.load(arm)["candidates"]]
                r = revalidate(emitted, cache.view(f"{arm}_revalidation", compute, stage="candidate_validation"), to_obj)
                reval[arm] = {k: v for k, v in r.items()
                              if k not in ("rows", "objectives", "nondominated_index")}
                reval[arm]["rows"] = r["rows"].to_dict("records")
            ck.save("candidate_revalidation", {"arms": reval})

        # ---- stage: the anchor-injection control -----------------------------
        # method_arms.md makes this mandatory, not optional. A vertex weight returns
        # its own anchor, so NBI-R's set CONTAINS the empirical anchors and part of
        # any NBI-S to NBI-R gap is the injection of those extreme points rather
        # than the relocated geometry. The control rescores NBI-S's own candidate
        # set augmented with the same anchors, changing nothing else. It costs no
        # new real evaluations: every point in it has already been measured.
        if not ck.done("anchor_injection_control"):
            ea = ck.load("empirical_anchors")
            rv_s = pd.DataFrame(ck.load("candidate_revalidation")["arms"]["nbi_s"]["rows"])
            anchor_cfgs = []
            for xr in np.asarray(ea["x_star"], dtype=float):
                cfg_nat, _ = rz.realize(xr)
                anchor_cfgs.append(cfg_nat)
            # Through the cache, not around it. Calling evaluate_config directly
            # performed two REAL evaluations per unit that no ledger recorded and
            # no budget table showed -- 240 across the campaign -- while the note
            # below claimed the control costs no new evaluations. Routed through
            # the view, the two anchors are charged as the budget registry declares
            # and served from cache, because the anchor search already measured
            # these exact configurations.
            aic_view = cache.view("anchor_injection_control", compute,
                                  stage="candidate_validation")
            anchor_rows = pd.DataFrame(
                [{**c, **aic_view.evaluate(c)} for c in anchor_cfgs])
            injected = pd.concat([rv_s, anchor_rows], ignore_index=True)
            ck.save("anchor_injection_control", {
                "rows": injected.to_dict("records"),
                "n_nbi_s_candidates": int(len(rv_s)),
                "n_anchors_injected": int(len(anchor_rows)),
                "note": ("NBI-S's revalidated set augmented with the empirical anchors; "
                         "scored alongside NBI-S and NBI-R so the share of the gap "
                         "attributable to injected extremes can be separated from the "
                         "share attributable to the relocated geometry."),
                "logical_charged": int(aic_view.logical_evaluations),
                "accounting_note": ("charged through the evaluation cache as "
                                    "candidate_validation; the anchor search already "
                                    "measured these exact configurations, so they are "
                                    "served from cache and cost no new physical fit. "
                                    "The per-method ledger in `accounting` is the "
                                    "authority on both figures.")})

        # The unmatched NSGA-II run is deliberately absent from both references and
        # from the matched indicator table: it received ten times the budget, so
        # admitting it would move the normalization box every matched method is
        # scored against.

        # ---- stage: the two references ---------------------------------------
        if not ck.done("reference_core"):
            # Method-independent real evaluations ONLY: the 88 design rows and the
            # 200 anchor-search rows. Nothing any compared method returned enters
            # this set, which is what makes it a reference a method cannot grade
            # itself against.
            #
            # The empirical anchors were previously loaded into an unused variable
            # while a variable NAMED anchor_rows held design rows, so the primary
            # indicator's reference was built from 88 points instead of 288 -- and
            # the 200 omitted ones are the best points direct search found on the
            # REAL objectives, which is precisely the part of the front that
            # matters.
            design_rows = pd.DataFrame(ck.load("design")["rows"])
            ea = ck.load("empirical_anchors")
            anchor_rows = pd.DataFrame(ea.get("rows", []))
            if len(anchor_rows) != ea["total_evaluations"]:
                raise MethodologicalFailure(
                    f"{dataset} rep {rep}: the anchor search measured "
                    f"{ea['total_evaluations']} points but persisted "
                    f"{len(anchor_rows)}. The core reference would be built from a "
                    f"subset of the method-independent evaluations.")
            core = reference_core([design_rows, anchor_rows], to_obj)
            ck.save("reference_core",
                    {k: v for k, v in core.items() if k != "front"}
                    | {"front": core["front"].tolist(),
                       "n_design_rows": int(len(design_rows)),
                       "n_anchor_rows": int(len(anchor_rows)),
                       "composition": "88 design rows + the anchor search's own "
                                      "measurements; method-independent by construction"})

        if not ck.done("augmented_reference"):
            core_front = np.asarray(ck.load("reference_core")["front"], dtype=float)
            per = {}
            rv = ck.load("candidate_revalidation")["arms"]
            for arm, payload in rv.items():
                df = pd.DataFrame(payload["rows"])
                per[arm] = to_obj(df) if len(df) else np.zeros((0, 2))
            db = _baseline_methods(ck.load("direct_baselines"))
            for name, payload in db.items():
                if name in SINGLE_OBJECTIVE:
                    continue                      # never enters the shared reference
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
            db = _baseline_methods(ck.load("direct_baselines"))
            per_method, endpoints = {}, {}
            control = ck.load("anchor_injection_control")
            scored = list(rv.items()) + list(db.items()) + [
                ("nbi_s_plus_anchors", {"rows": control["rows"]})]
            for name, payload in scored:
                df = pd.DataFrame(payload["rows"])
                F = to_obj(df) if len(df) else np.zeros((0, 2))
                if name in SINGLE_OBJECTIVE:
                    # its own objective only: quality runs on index 0, cost on 1
                    j = 0 if name.endswith("quality") else 1
                    endpoints[name] = {"objective_index": j,
                                       "best_value": (float(F[:, j].min())
                                                      if len(F) else float("nan")),
                                       "n_evaluations": int(len(F))}
                    continue
                per_method[name] = {
                    "augmented": indicators(F, aug_front),
                    "core": indicators(F, core_front)}
            ck.save("metrics_by_method", {
                "methods": per_method, "primary_indicator": "hv_ratio",
                "references": ["augmented", "core"],
                "single_objective_endpoints": endpoints,
                "note": ("single-objective runs are reported on their own endpoint only; "
                         "they are excluded from the front-indicator table and from the "
                         "augmented reference")})

        # ---- stage: holdout confirmation (labels untouched until here) -------
        if not ck.done("holdout_confirmation"):
            # The holdout measurement is a REAL learner evaluation on a partition no
            # earlier stage has touched. It was called directly, so five real
            # evaluations per unit -- 600 across the campaign -- were performed
            # outside every ledger and appeared in no budget table. It is audit-only,
            # exactly like the 78-point external set: it confirms and it steers
            # nothing. Audit-only is a reason to DECLARE it, not a reason to omit it.
            ho_view = cache.view("holdout_confirmation", _compute_holdout,
                                 stage="holdout_audit")
            rv = ck.load("candidate_revalidation")["arms"]
            ho = {}
            for arm, payload in rv.items():
                df = pd.DataFrame(payload["rows"])
                if not len(df):
                    continue
                F = to_obj(df)
                span = np.ptp(F, axis=0)          # numpy 2 removed ndarray.ptp
                norm = (F - F.min(axis=0)) / np.where(span == 0, 1.0, span)
                sel = df.iloc[int(np.argmin(np.linalg.norm(norm, axis=1)))]
                cfg_sel = {p: (int(sel[p]) if p in INT_PARAMS else float(sel[p]))
                           for p in PARAMS}
                ho[arm] = {"selected_config": cfg_sel,
                           "internal": {k: float(sel[k]) for k in RESPONSES},
                           "holdout": ho_view.evaluate(cfg_sel, fold_id="holdout")}
            ck.save("holdout_confirmation", {
                "selection_rule": "distance to the utopia of the arm's own revalidated set",
                "arms": ho,
                "note": "holdout labels are read here and nowhere earlier"})

        ck.save("split", {"seed": seed, "test_size": 0.2,
                          "stratified": True, "inner_folds": 5})
        result = {"dataset": dataset, "replication": rep, "seed": seed,
                  "wall_seconds": round(time.time() - t0, 1),
                  "accounting": cache.accounting(),
                  # "metrics" is this checkpoint itself and is complete by the time
                  # anyone reads it, so it is listed rather than omitted. Computing
                  # the list before the save made every unit report 19 of 20 stages.
                  "stages_complete": [s for s in STAGES
                                      if s == "metrics" or ck.done(s)]}
        ck.save("metrics", result)
        return result
    except MethodologicalFailure as exc:
        ck.save("methodological_failure", {"message": str(exc),
                                           "traceback": traceback.format_exc()})
        raise
    finally:
        cache.close()


def _run_historical(design_df, Y, fm, surrogates, rz, seed) -> dict:
    """HISTORICAL-WS-asrun: the dissertation's own normalization and its own grid.

    The surfaces are fitted in the historical *uncoded* parameterization and the
    normalization box is the component-wise observed extrema of the design rows,
    as scripts/run_nbi.py built it. Neither is repaired.
    """
    from .design import fit_surface_backward_uncoded

    # Two properties of the historical method, reproduced rather than repaired.
    #
    # Orientation: the dissertation MAXIMIZES its two scores (Score_Quality, and
    # Score_Cost which is a negated z-scored time), and its solver's feasibility
    # box requires nadir <= prediction <= utopia. This campaign's objectives are
    # canonicalized to minimization, so the historical arm sees their negation and
    # its utopia is the component-wise maximum.
    #
    # Parameterization: the dissertation fits its surfaces in NATURAL units
    # (METHODOLOGY_DECISIONS D6), and the frozen solver evaluates named terms at
    # natural values. Handing it coded coefficients would silently evaluate the
    # wrong surface.
    Y_hist = -np.asarray(Y, dtype=float)
    observed_utopia = tuple(float(Y_hist[:, j].max()) for j in range(2))
    observed_nadir = tuple(float(Y_hist[:, j].min()) for j in range(2))
    models = [fit_surface_backward_uncoded(design_df, Y_hist[:, j]) for j in range(2)]
    run = run_historical_ws(models[0], models[1],
                            observed_utopia=observed_utopia,
                            observed_nadir=observed_nadir,
                            bounds={p: BOUNDS[p] for p in PARAMS},
                            realizer=rz, surrogates_coded=surrogates, seed=seed)
    out = run.as_dict()
    out["diagnostics"].update({
        "orientation": ("the dissertation maximizes; this campaign's objectives are "
                        "minimized, so the historical arm sees their negation"),
        "surface_parameterization": "uncoded natural units, as the dissertation fits",
        "surface_terms": [len(m[0]) for m in models],
        "observed_utopia_maximization_orientation": list(observed_utopia),
        "observed_nadir_maximization_orientation": list(observed_nadir)})
    return out


def _empirical_anchors(cache, compute, fm, cfg, seed) -> tuple[np.ndarray, np.ndarray, dict]:
    """Direct search on the REAL objectives, one budget per objective.

    These are the best configurations the search found within its declared budget.
    They are not certified optima and are never described as such.
    """
    rng = np.random.default_rng(seed + 77)
    view = cache.view("empirical_anchor_search", compute, stage="anchor")
    rz = Realizer(PARAMS, BOUNDS, list(INT_PARAMS))
    k = len(PARAMS)
    x_star = np.zeros((2, k))
    F_star = np.zeros((2, 2))
    rows_per_objective = []
    for j in range(2):
        best_x, best_f, rows = None, None, []
        for _ in range(B_ANCHOR_PER_OBJECTIVE):
            xc = rng.uniform(-1.0, 1.0, size=k)
            cfg_nat, xc_real = rz.realize(xc)
            res = view.evaluate(cfg_nat)
            f = fm.objectives(pd.DataFrame([res]))[0]
            # The full measured row, in the same shape as a design row, because the
            # core reference is built from it. Storing only the config and the
            # objective vector made these 200 real evaluations unusable downstream.
            rows.append({**cfg_nat, **res, "run": f"anchor_obj{j}"})
            if best_f is None or f[j] < best_f[j]:
                best_x, best_f = xc_real, f
        x_star[j] = best_x
        # The payoff matrix column is the objective vector ALREADY MEASURED at this
        # anchor. Re-requesting it charged two further evaluations, which took
        # NBI-R to 202 against a declared 200 and pushed its total past the
        # comparator budget defined as the maximum over arms.
        F_star[:, j] = best_f
        rows_per_objective.append(rows)
    return x_star, F_star, {
        "x_star": x_star.tolist(), "F_star": F_star.tolist(),
        "budget_per_objective": B_ANCHOR_PER_OBJECTIVE,
        "search": "uniform random over the coded box, realized before evaluation",
        "note": ("best found within the declared budget; NOT certified optima and "
                 "never described as such"),
        "evaluations_by_objective": [len(r) for r in rows_per_objective],
        "total_evaluations": sum(len(r) for r in rows_per_objective),
        "payoff_matrix_reuses_search_measurements": True,
        # Method-independent real measurements. reference_core is documented as
        # "the design AND the anchor search"; without these rows persisted it was
        # built from the design alone.
        "rows": [r for rows in rows_per_objective for r in rows]}


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------


# The single source of truth for what each method spends, by stage. Every budget
# figure anywhere -- the planner, the dry run, the manifest, the protocol tables --
# derives from this, and the totals must reconcile exactly against the sum of the
# table with no unexplained remainder.
def method_stage_ledger(q: int = 2) -> dict[str, dict[str, int]]:
    """Logical evaluations per method per stage, for one unit.

    ``design`` and ``external_audit`` are shared stages: the campaign evaluates each
    once per unit, and the per-arm budget accounting in the protocol charges every
    arm for the design because every arm would have to pay it alone. The two views
    are kept apart here: this table is what the RUNNER requests, and
    ``B_total_solution`` below is what an arm would cost standalone.
    """
    design_n, ext_n, cand = 88, 78, N_CANDIDATES
    anchor = B_ANCHOR_PER_OBJECTIVE * q
    comparator = design_n + ext_n + anchor + cand          # the most expensive arm
    table: dict[str, dict[str, int]] = {
        "design": {"design": design_n},
        "external_validation_audit": {"external_audit": ext_n},
        "empirical_anchor_search": {"anchor": anchor},
    }
    for arm in ("historical_ws_asrun", "historical_ws", "ws_s", "nbi_s", "nbi_r"):
        table[f"{arm}_revalidation"] = {"candidate_validation": cand}
    table["anchor_injection_control"] = {"candidate_validation": 2}
    # Audit-only, and therefore DECLARED rather than omitted. One holdout
    # measurement per revalidated arm, on a partition no earlier stage touches.
    # These were previously performed by a direct call outside the cache, so 5 real
    # evaluations per unit -- 600 across the campaign -- were charged to no method
    # and appeared in no budget table.
    table["holdout_confirmation"] = {"holdout_audit": 5}
    for m in ("grid", "random", "bayes_quality", "bayes_cost",
              "tpe_quality", "tpe_cost"):
        table[m] = {"direct_search": comparator}
    table["nsga2"] = {"direct_search": NSGA2_POP * NSGA2_GEN}
    return table


def unit_budget(q: int = 2) -> dict:
    """Totals for one unit, reconciled against the method-stage table."""
    table = method_stage_ledger(q)
    by_stage: dict[str, int] = {}
    for stages in table.values():
        for s, n in stages.items():
            by_stage[s] = by_stage.get(s, 0) + n
    total = sum(by_stage.values())
    # Audit-only stages: they are measured and reported, and they steer nothing.
    # A gate failure and a holdout result both change zero execution decisions.
    audit_only = by_stage.get("external_audit", 0) + by_stage.get("holdout_audit", 0)
    solution = total - audit_only
    return {"method_stage_table": table, "by_stage": by_stage,
            "solution_producing_logical": solution,
            "audit_only_logical": audit_only,
            "total_logical": total}


def reconcile_unit_accounting(accounting: dict, q: int = 2,
                              replication: int | None = None) -> dict:
    """Compare what a unit ACTUALLY charged against what the registry declares.

    ``unit_budget`` previously returned a ``reconciles`` flag computed as
    ``sum(by_stage.values()) == sum(table values)``, where ``by_stage`` had itself
    been built by summing that same table. It was ``X == X``: it could not fail, and
    the dry run published it as a pre-launch proof. It would have reported a clean
    reconciliation while the runner charged a different budget entirely -- which it
    did, by 840 evaluations across the campaign, performed outside every ledger.

    This is the check that can fail. It takes the cache's own per-method-per-stage
    ledger from a completed unit and compares it entry by entry with the registry.
    """
    declared: dict[tuple[str, str], int] = {}
    for method, stages in method_stage_ledger(q).items():
        for stage, n in stages.items():
            declared[(method, stage)] = n

    # The unmatched NSGA-II run is scoped to ONE replication per dataset, so it is
    # budgeted campaign-wide rather than per unit. A unit that ran it charges it and
    # must have it declared; a unit that did not must not. Passing replication=None
    # accepts either, which is only appropriate when the caller does not know which
    # unit produced the ledger.
    unmatched = NSGA2_POP * NSGA2_GEN * NSGA2_UNMATCHED_MULTIPLIER
    unmatched_key = ("nsga2_unmatched", "unmatched_context")
    charged_unmatched = any(
        led["method"] == unmatched_key[0]
        for led in accounting.get("per_method", []))
    if replication is None:
        if charged_unmatched:
            declared[unmatched_key] = unmatched
    elif replication == NSGA2_UNMATCHED_REPLICATION:
        declared[unmatched_key] = unmatched

    charged: dict[tuple[str, str], int] = {}
    for led in accounting.get("per_method", []):
        for stage, st in (led.get("by_stage") or {}).items():
            charged[(led["method"], stage)] = st.get("logical", 0)

    mismatches, undeclared, unspent = [], [], []
    for key, n in sorted(declared.items()):
        got = charged.get(key)
        if got is None:
            unspent.append({"method": key[0], "stage": key[1], "declared": n})
        elif got != n:
            mismatches.append({"method": key[0], "stage": key[1],
                               "declared": n, "charged": got})
    for key, n in sorted(charged.items()):
        if key not in declared:
            undeclared.append({"method": key[0], "stage": key[1], "charged": n})

    total_declared = sum(declared.values())
    total_charged = sum(charged.values())
    return {"reconciles": not (mismatches or undeclared or unspent)
                          and total_declared == total_charged,
            "total_declared": total_declared,
            "total_charged": total_charged,
            "difference": total_charged - total_declared,
            "mismatched_stages": mismatches,
            "charged_but_never_declared": undeclared,
            "declared_but_never_charged": unspent}


def campaign_budget(q: int = 2) -> dict:
    """The authoritative campaign total, derived from the runner's own registry."""
    unit = unit_budget(q)
    n_units = len(DATASETS) * N_REPLICATIONS
    unmatched = NSGA2_POP * NSGA2_GEN * NSGA2_UNMATCHED_MULTIPLIER * len(DATASETS)
    total = unit["total_logical"] * n_units + unmatched
    parts = unit["total_logical"] * n_units + unmatched
    return {**unit, "units": n_units,
            "unmatched_nsga2_logical": unmatched,
            "unmatched_nsga2_scope": "one replication per dataset",
            "campaign_solution_producing_logical":
                unit["solution_producing_logical"] * n_units + unmatched,
            "campaign_audit_only_logical": unit["audit_only_logical"] * n_units,
            "campaign_total_logical": total,
            # An arithmetic identity, labelled as one. Whether the campaign SPENDS
            # this budget is answered by reconcile_unit_accounting against a
            # completed unit's ledger, not here.
            "arithmetic_consistent": total == parts}


def logical_budget_plan(q: int = 2) -> dict:
    design_n, ext_n = 88, 78
    # Standalone cost per executed arm, by exact identifier. The two historical
    # entities cost different amounts and must appear as different rows: the as-run
    # reproduction fits the dissertation's uncoded surfaces and has no gate, so it
    # never pays external validation, while the shared-specification arm uses WS-S's
    # own gated surrogates and pays exactly what WS-S pays. An earlier version of
    # this table carried one ambiguous "HISTORICAL-WS" row at 108 and omitted the
    # other arm entirely.
    arms = {
        "HISTORICAL-WS-asrun": design_n + N_CANDIDATES,
        "HISTORICAL-WS": design_n + ext_n + N_CANDIDATES,
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


__all__ = ["run_unit", "logical_budget_plan", "method_stage_ledger",
           "reconcile_unit_accounting", "SCORED_BASELINES",
           "unit_budget", "campaign_budget", "unit_seed", "STAGES", "DATASETS",
           "N_REPLICATIONS", "PROTOCOL_TAG", "MethodologicalFailure", "Checkpoint"]
