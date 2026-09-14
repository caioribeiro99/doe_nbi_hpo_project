"""The four frozen method arms.

Defined in ``papers/xgboost_hpo_vrfnbi/protocol/method_arms.md``. The arm set is a
two-by-two on the two things the historical method and a canonical NBI change at
once -- how the objective reference is built, and which scalarization geometry is
used -- plus anchor provenance:

    HISTORICAL-WS   weighted sum, observed-extrema normalization, asymmetric grid
    WS-S            weighted sum, payoff-matrix normalization, symmetric grid
    NBI-S           canonical NBI, surrogate anchors
    NBI-R           canonical NBI, empirically obtained anchors

Primary contrasts:

    NBI-S vs WS-S   geometry, at a fixed reference
    NBI-R vs NBI-S  anchor provenance, at a fixed geometry

Secondary:

    WS-S vs HISTORICAL-WS   historical reconstruction

Everything the contrasts do not name is held fixed: the same surrogates, the same
objective orientation, the same decision bounds, the same weight-grid cardinality,
the same tolerances.

Two things are deliberately *not* shared, and both are properties of the historical
method rather than concessions to it: HISTORICAL-WS keeps its observed-extrema
normalization and its asymmetric weight grid, which omits the pure-quality vertex.
Those are recorded as provenance, not repaired.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..nbi_core import (AnchorSet, CHIM, NBIConfig, build_chim, compute_anchors,
                        solve_nbi_subproblem)

SurrogateCallable = Callable[[np.ndarray], float]

DISSERTATION_TAG = "v0.1.0-dissertation"


FROZEN_PACKAGE = "doe_xgb_frozen"


def _ensure_frozen_tree(dest: Path | None = None) -> Path:
    """Extract the dissertation source at its tag, under the repository, and return it.

    The tree is extracted under a **different top-level package name**,
    ``doe_xgb_frozen``. The article track has already imported ``doe_xgb``, so
    putting the frozen tree on ``sys.path`` would not change what
    ``doe_xgb.nbi`` resolves to: the historical arm would silently call the
    rewrite it exists to be compared against. Renaming the package makes the two
    importable side by side, and the relative imports inside the frozen modules
    keep working because they do not name their own package.
    """
    import shutil
    import subprocess
    repo = Path(__file__).resolve().parents[3]
    dest = dest or (repo / ".frozen" / DISSERTATION_TAG)
    src = dest / "src"
    target = src / FROZEN_PACKAGE
    if not (target / "nbi.py").exists():
        dest.mkdir(parents=True, exist_ok=True)
        tar = subprocess.run(["git", "archive", DISSERTATION_TAG, "src/doe_xgb"],
                             cwd=repo, capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout, check=True)
        extracted = src / "doe_xgb"
        if target.exists():
            shutil.rmtree(target)
        extracted.rename(target)
    return src

# Weight-grid cardinality, shared by every arm. The historical grid has 20 points;
# matching it keeps B_candidate_validation equal across arms.
N_WEIGHTS = 20

# A subproblem counts as certified when the solver succeeded and the equality
# residual is below this. Declared here rather than inlined so the campaign and the
# tests cannot drift apart.
EQUALITY_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Candidate record
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One returned solution, with everything needed to audit how it was produced.

    The distinction between ``x_continuous`` and ``config_realized`` is load
    bearing: NBI certifies a subproblem on the continuous surrogate, and the
    learner is then run on a rounded configuration that does not satisfy that
    certificate. Both are kept, together with the displacement between them.
    """

    arm: str
    weight: tuple[float, ...]
    x_continuous: np.ndarray
    f_surrogate_continuous: np.ndarray
    config_realized: dict[str, Any]
    x_realized_coded: np.ndarray
    f_surrogate_realized: np.ndarray
    realization_displacement: float
    solver_success: bool
    solver_message: str
    # NBI-only; None for the weighted-sum arms
    t: float | None = None
    equality_residual: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "arm": self.arm,
            "weight": list(self.weight),
            "x_continuous": self.x_continuous.tolist(),
            "f_surrogate_continuous": self.f_surrogate_continuous.tolist(),
            "config_realized": self.config_realized,
            "x_realized_coded": self.x_realized_coded.tolist(),
            "f_surrogate_realized": self.f_surrogate_realized.tolist(),
            "realization_displacement": self.realization_displacement,
            "solver_success": self.solver_success,
            "solver_message": self.solver_message,
        }
        if self.t is not None:
            d["t"] = self.t
            d["equality_residual"] = self.equality_residual
        d.update(self.extra)
        return d


@dataclass
class ArmRun:
    arm: str
    candidates: list[Candidate]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"arm": self.arm,
                "candidates": [c.as_dict() for c in self.candidates],
                "diagnostics": self.diagnostics}


# ---------------------------------------------------------------------------
# Realization: continuous surrogate solution -> runnable configuration
# ---------------------------------------------------------------------------


class Realizer:
    """Maps a coded continuous point to a configuration the learner can run.

    Deterministic by construction: clip to the box, map to natural units, round the
    integer hyperparameters half-away-from-zero, clip again. No search, no
    neighbourhood polish. Protocol v2 defines no repair step, and one is not
    invented here.
    """

    def __init__(self, params: Sequence[str], bounds: dict[str, tuple[float, float]],
                 int_params: Sequence[str]) -> None:
        self.params = list(params)
        self.lo = np.array([bounds[p][0] for p in self.params], dtype=float)
        self.hi = np.array([bounds[p][1] for p in self.params], dtype=float)
        self.int_params = set(int_params)

    def to_natural(self, x_coded: np.ndarray) -> np.ndarray:
        xc = np.clip(np.asarray(x_coded, dtype=float), -1.0, 1.0)
        return self.lo + (xc + 1.0) * (self.hi - self.lo) / 2.0

    def to_coded(self, x_natural: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x_natural, float) - self.lo) / (self.hi - self.lo) - 1.0

    def realize(self, x_coded: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        nat = self.to_natural(x_coded)
        out: dict[str, Any] = {}
        realized = nat.copy()
        for i, p in enumerate(self.params):
            if p in self.int_params:
                # half away from zero, so the mapping does not depend on float parity
                v = float(np.floor(np.abs(nat[i]) + 0.5) * np.sign(nat[i]))
                v = float(np.clip(v, self.lo[i], self.hi[i]))
                realized[i] = v
                out[p] = int(v)
            else:
                out[p] = float(nat[i])
        return out, self.to_coded(realized)


def _record(arm: str, weight, x_cont: np.ndarray, surrogates, realizer: Realizer,
            success: bool, message: str, t=None, residual=None,
            extra: dict | None = None) -> Candidate:
    f_cont = np.array([float(f(x_cont)) for f in surrogates])
    cfg, x_real = realizer.realize(x_cont)
    f_real = np.array([float(f(x_real)) for f in surrogates])
    return Candidate(arm=arm, weight=tuple(float(w) for w in np.atleast_1d(weight)),
                     x_continuous=np.asarray(x_cont, float),
                     f_surrogate_continuous=f_cont,
                     config_realized=cfg, x_realized_coded=x_real,
                     f_surrogate_realized=f_real,
                     realization_displacement=float(np.linalg.norm(f_real - f_cont)),
                     solver_success=bool(success), solver_message=str(message),
                     t=t, equality_residual=residual, extra=extra or {})


# ---------------------------------------------------------------------------
# Weight grids
# ---------------------------------------------------------------------------


def symmetric_weights(n: int = N_WEIGHTS, q: int = 2) -> np.ndarray:
    """Simplex lattice including every vertex, shared by WS-S, NBI-S and NBI-R."""
    if q != 2:
        raise NotImplementedError("q > 2 weight grids are set by the protocol amendment")
    w = np.linspace(0.0, 1.0, n)
    return np.column_stack([1.0 - w, w])


def historical_weights(step: float = 0.05) -> np.ndarray:
    """The dissertation's grid, reproduced including its asymmetry.

    ``b`` runs from ``step`` to 1.0 and the pair is ``(1 - b, b)``, so the pair
    ``(1, 0)`` never occurs: the historical method solves at the pure-cost vertex
    and never at the pure-quality vertex. This is a property of the historical
    method and is not repaired.
    """
    b = np.arange(step, 1.0 + 1e-9, step)
    return np.column_stack([np.round(1.0 - b, 2), np.round(b, 2)])


# ---------------------------------------------------------------------------
# Arm 1 -- HISTORICAL-WS
# ---------------------------------------------------------------------------


def run_historical_ws(model_quality, model_cost, *, observed_utopia, observed_nadir,
                      bounds: dict[str, tuple[float, float]], realizer: Realizer,
                      surrogates_coded: Sequence[SurrogateCallable],
                      frozen_src: str | Path | None = None,
                      seed: int = 42, n_starts: int = 10) -> ArmRun:
    """Reproduce the dissertation optimizer by calling the frozen code itself.

    The frozen ``run_nbi_weighted_sum`` at tag ``v0.1.0-dissertation`` is imported
    from an extracted copy and invoked unmodified. Reimplementing it would be a
    claim about what it does; calling it is a reproduction of what it does.

    ``model_quality`` and ``model_cost`` are ``(terms, coefficients)`` in the
    historical *uncoded* parameterization, and the normalization box is the
    component-wise observed extrema of the design rows, as ``scripts/run_nbi.py``
    built it.
    """
    # The one arm whose job is bit-faithful reproduction must not depend on an
    # ephemeral path. Extract the tag into the repository's own scratch area if it is
    # not already there, and verify the module really came from the frozen tree.
    src = Path(frozen_src) if frozen_src is not None else _ensure_frozen_tree()
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import importlib
    _frozen_nbi = importlib.import_module(f"{FROZEN_PACKAGE}.nbi")
    _frozen_cfg = importlib.import_module(f"{FROZEN_PACKAGE}.config")
    if str(src) not in str(_frozen_nbi.__file__):
        raise RuntimeError(
            f"{FROZEN_PACKAGE}.nbi resolved to {_frozen_nbi.__file__}, not the frozen tree "
            f"at {src}. HISTORICAL-WS must call the dissertation code, not the "
            "article-track rewrite.")
    run_nbi_weighted_sum = _frozen_nbi.run_nbi_weighted_sum      # frozen
    FROZEN_PARAMS = _frozen_cfg.PARAM_NAMES                      # frozen

    raw = run_nbi_weighted_sum(
        model_quality, model_cost, bounds=bounds,
        observed_utopia=tuple(observed_utopia), observed_nadir=tuple(observed_nadir),
        beta_step=0.05, seed=seed, n_starts=n_starts, constrain_pred_range=True)

    cands: list[Candidate] = []
    for c in raw:
        x_nat = np.array([float(c.params[p]) for p in FROZEN_PARAMS], dtype=float)
        x_coded = realizer.to_coded(x_nat)
        cands.append(_record("HISTORICAL-WS", c.betas, x_coded, surrogates_coded,
                             realizer, c.success, c.message,
                             extra={"historical_predicted": list(map(float, c.predicted)),
                                    "historical_score": float(c.score)}))
    from ..reporting import dominated_fraction
    grid = historical_weights()
    F_real = np.array([c.f_surrogate_realized for c in cands])
    return ArmRun("HISTORICAL-WS", cands, {
        "dominated_share_of_returned_set": round(float(dominated_fraction(F_real)), 4),
        "source": "frozen v0.1.0-dissertation run_nbi_weighted_sum, called unmodified",
        "normalization": "component-wise observed extrema of the design rows",
        "observed_utopia": list(map(float, observed_utopia)),
        "observed_nadir": list(map(float, observed_nadir)),
        "weight_grid": grid.tolist(),
        "weight_grid_is_symmetric": bool(
            any(np.allclose(g, [1.0, 0.0]) for g in grid)
            and any(np.allclose(g, [0.0, 1.0]) for g in grid)),
        "missing_vertex": "(1.0, 0.0), the pure-quality endpoint; historical, not repaired",
        "surrogate_parameterization": "uncoded, as fitted by the dissertation",
    })


# ---------------------------------------------------------------------------
# Reference construction shared by WS-S, NBI-S and NBI-R
# ---------------------------------------------------------------------------


def surrogate_reference(surrogates: Sequence[SurrogateCallable],
                        cfg: NBIConfig) -> tuple[AnchorSet, CHIM]:
    """Payoff matrix from per-objective minimization of the surrogates."""
    anchors = compute_anchors(surrogates, cfg)
    return anchors, build_chim(anchors, cfg)


def empirical_reference(x_star: np.ndarray, F_star: np.ndarray,
                        cfg: NBIConfig) -> tuple[AnchorSet, CHIM]:
    """Payoff matrix from empirically obtained anchors.

    ``x_star[j]`` is the configuration a direct search on the *real* objectives
    returned for objective ``j``, and ``F_star[i, j]`` is objective ``i`` measured
    there. These are the best configurations the anchor search found within its
    declared budget; they are not called true optima.
    """
    F_star = np.asarray(F_star, dtype=float)
    utopia = np.diag(F_star).copy()
    anchors = AnchorSet(x_star=np.asarray(x_star, float), F_star=F_star,
                        utopia=utopia, nadir=F_star.max(axis=1),
                        pseudo_nadir=F_star.max(axis=1),
                        diagnostics={"source": "empirical anchor search on the real "
                                               "objectives; best found within budget, "
                                               "not certified optima"})
    return anchors, build_chim(anchors, cfg)


# ---------------------------------------------------------------------------
# Arm 2 -- WS-S
# ---------------------------------------------------------------------------


def run_ws_s(surrogates: Sequence[SurrogateCallable], cfg: NBIConfig,
             anchors: AnchorSet, realizer: Realizer,
             weights: np.ndarray | None = None,
             nadir_choice: Literal["pseudo_nadir", "nadir"] = "pseudo_nadir",
             *, arm: str = "WS-S",
             reference_override: tuple[np.ndarray, np.ndarray] | None = None,
             reference_label: str | None = None) -> ArmRun:
    """Weighted sum over the same surrogates, normalized by the payoff matrix.

    The geometry control. It differs from NBI-S in one respect only: it minimizes
    a weighted sum of normalized surrogates instead of solving the NBI subproblem.
    The normalization box is the payoff matrix's utopia and, by protocol, its
    pseudo-nadir -- the row-wise maximum -- so WS-S and NBI-S share exactly the
    same reference object.
    """
    from scipy.optimize import minimize

    weights = symmetric_weights() if weights is None else np.asarray(weights, float)
    if reference_override is not None:
        utopia, nadir = (np.asarray(v, dtype=float) for v in reference_override)
    else:
        utopia, nadir = anchors.utopia, getattr(anchors, nadir_choice)
    span = np.where(np.abs(nadir - utopia) < 1e-12, 1.0, nadir - utopia)
    rng = np.random.default_rng(cfg.seed)
    lo, hi = cfg.bounds[:, 0], cfg.bounds[:, 1]
    starts = [np.mean(cfg.bounds, axis=1)] + [rng.uniform(lo, hi)
                                              for _ in range(max(0, cfg.n_starts - 1))]

    cands: list[Candidate] = []
    for w in weights:
        def obj(x, w=w):
            f = np.array([float(s(x)) for s in surrogates])
            return float(np.dot(w, (f - utopia) / span))     # objectives are minimized

        # The same feasibility constraint the NBI path applies, the same acceptance
        # rule, and the same finiteness guard. Without them WS-S would be solving a
        # slightly different problem from NBI-S, which is the one thing the geometry
        # contrast may not permit.
        constraints = []
        if cfg.feasibility_constraint is not None:
            constraints.append({"type": "ineq",
                                "fun": lambda x, fc=cfg.feasibility_constraint: fc(x)})
        best, best_val, fallback = None, np.inf, None
        for x0 in starts:
            r = minimize(obj, x0, method="SLSQP", bounds=list(map(tuple, cfg.bounds)),
                         constraints=constraints,
                         options={"maxiter": cfg.maxiter, "ftol": 1e-9})
            if not np.all(np.isfinite(r.x)) or not np.isfinite(r.fun):
                continue
            if fallback is None or r.fun < float(fallback.fun):
                fallback = r
            if r.success and r.fun < best_val:
                best, best_val = r, float(r.fun)
        if best is None:                      # no start converged; keep the best seen
            best = fallback if fallback is not None else minimize(
                obj, starts[0], method="SLSQP", bounds=list(map(tuple, cfg.bounds)),
                options={"maxiter": cfg.maxiter})
            best_val = float(best.fun)
        cands.append(_record(arm, w, np.asarray(best.x, float), surrogates, realizer,
                             bool(best.success), str(best.message),
                             extra={"scalarized_value": best_val}))
    from ..reporting import dominated_fraction
    F_real = np.array([c.f_surrogate_realized for c in cands])
    n_unique = len({tuple(np.round(c.x_realized_coded, 9)) for c in cands})
    return ArmRun(arm, cands, {
        "scalarization": "weighted sum of normalized surrogates",
        "reference_construction": reference_label or
            f"payoff matrix: utopia and {nadir_choice}",
        "dominated_share_of_returned_set": round(float(dominated_fraction(F_real)), 4),
        "solver_success_fraction": round(
            float(np.mean([c.solver_success for c in cands])), 4),
        "distinct_realized_configurations": n_unique,
        "duplicate_share_after_realization": round(1.0 - n_unique / max(len(cands), 1), 4),
        "feasibility_constraint_applied": cfg.feasibility_constraint is not None,
        "utopia": utopia.tolist(), "nadir_used": np.asarray(nadir, float).tolist(),
        "true_nadir": anchors.nadir.tolist(),
        "pseudo_nadir": anchors.pseudo_nadir.tolist(),
        "weight_grid_symmetric": True, "n_weights": int(len(weights)),
    })


# ---------------------------------------------------------------------------
# Arms 3 and 4 -- NBI-S and NBI-R
# ---------------------------------------------------------------------------


def run_nbi_arm(arm: str, surrogates: Sequence[SurrogateCallable], cfg: NBIConfig,
                anchors: AnchorSet, chim: CHIM, realizer: Realizer,
                weights: np.ndarray | None = None) -> ArmRun:
    """Canonical NBI over the given reference. Used for both NBI-S and NBI-R."""
    weights = symmetric_weights() if weights is None else np.asarray(weights, float)
    cands: list[Candidate] = []
    for beta in weights:
        r = solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg)
        cands.append(_record(arm, beta, np.asarray(r.x, float), surrogates, realizer,
                             r.success, r.message, t=float(r.t),
                             residual=float(r.residual_norm),
                             extra={"optimizer_info": r.optimizer_info}))
    # `residual or 1.0` maps a residual of exactly 0.0 to 1.0, so a perfectly
    # feasible subproblem scored as uncertified. Test the value, not its truthiness.
    def _certified(c: Candidate) -> bool:
        r = c.equality_residual
        return bool(c.solver_success and r is not None and r < EQUALITY_TOLERANCE)

    certified = [c for c in cands if _certified(c)]
    # A subproblem solution is certified FEASIBLE, not Pareto optimal. Weighted-sum
    # minimizers are weakly Pareto optimal by construction; NBI solutions need not be.
    # Reporting the dominated share per arm keeps that asymmetry from being mistaken
    # for a difference in approximation quality.
    from ..reporting import dominated_fraction
    F_real = np.array([c.f_surrogate_realized for c in cands])
    dominated_share = float(dominated_fraction(F_real))
    dominated_among_certified = (
        float(dominated_fraction(np.array([c.f_surrogate_realized for c in certified])))
        if certified else float("nan"))
    Phi = chim.Phi
    return ArmRun(arm, cands, {
        "scalarization": "canonical NBI: max t subject to F(x) = utopia + Phi beta + t n_hat",
        "utopia": anchors.utopia.tolist(),
        "payoff_matrix": Phi.tolist(),
        "payoff_rank": int(np.linalg.matrix_rank(Phi)),
        "payoff_condition_number": float(np.linalg.cond(Phi))
            if np.linalg.matrix_rank(Phi) == Phi.shape[0] else None,
        "anchors_coincide": bool(
            any(np.allclose(anchors.x_star[i], anchors.x_star[j])
                for i in range(len(anchors.x_star))
                for j in range(i + 1, len(anchors.x_star)))),
        "quasi_normal": chim.n_hat.tolist(),
        "restrict_t_nonnegative": bool(cfg.restrict_t_nonnegative),
        "certified_fraction": round(len(certified) / max(len(cands), 1), 4),
        "equality_tolerance": EQUALITY_TOLERANCE,
        "solver_success_fraction": round(
            float(np.mean([c.solver_success for c in cands])), 4),
        "distinct_realized_configurations":
            len({tuple(np.round(c.x_realized_coded, 9)) for c in cands}),
        "duplicate_share_after_realization": round(
            1.0 - len({tuple(np.round(c.x_realized_coded, 9)) for c in cands})
            / max(len(cands), 1), 4),
        "feasibility_constraint_applied": cfg.feasibility_constraint is not None,
        "dominated_share_of_returned_set": round(dominated_share, 4),
        "dominated_share_among_certified": (round(dominated_among_certified, 4)
                                            if dominated_among_certified == dominated_among_certified
                                            else None),
        "t_range": [float(min(c.t for c in cands)), float(max(c.t for c in cands))],
        "max_equality_residual": float(max(c.equality_residual for c in cands)),
        "max_realization_displacement": float(max(c.realization_displacement
                                                  for c in cands)),
        "anchor_source": anchors.diagnostics.get("source", "surrogate optimization"),
        "n_weights": int(len(weights)),
    })


__all__ = ["ArmRun", "Candidate", "Realizer", "N_WEIGHTS", "EQUALITY_TOLERANCE",
           "symmetric_weights", "historical_weights",
           "surrogate_reference", "empirical_reference",
           "run_historical_ws", "run_ws_s", "run_nbi_arm"]
