"""True Normal Boundary Intersection (NBI) adapted to the ensemble-weight simplex.

Ported from doe_nbi_hpo_project @ origin/repo-publication-readiness:0465466,
file ``src/doe_xgb/nbi_core.py`` (Das & Dennis, 1998), adapted for PCO213
postwork. Original author: Caio Tertuliano Ribeiro (MIT License). The
mathematical core is unchanged and dimension-agnostic; this port removes
unused fields, does NOT depend on ``doe_xgb``, and adds a simplex layer:

- decision variables are the first ``M-1`` weights (free variables in
  ``[0, 1]``), with ``w_M = 1 - sum(w_1..w_{M-1})`` reconstructed;
- feasibility ``w_M >= 0`` enters as the inequality constraint;
- every returned candidate/anchor is lifted back to a valid point of the
  simplex (non-negative, summing to 1).

``post_optimization.py`` from the origin is deliberately NOT ported.

NBI subproblem:  max t  s.t.  F(x) = utopia + Phi @ beta + t * n_hat, x in Omega.
Objectives follow the MINIMIZATION convention (wrap "maximize AUC" as -AUC).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

from mixens.mixture_design import generate_simplex_lattice, simplex_lattice_count, validate_weights

SurrogateCallable = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class NBIConfig:
    objective_count: int
    bounds: np.ndarray  # shape (k, 2): per-decision-variable [lo, hi]
    n_starts: int = 10
    seed: int = 42
    solver: Literal["slsqp", "trust_constr"] = "slsqp"
    feasibility_constraint: Callable[[np.ndarray], np.ndarray] | None = None
    quasi_normal: Literal["minus_phi_ones", "gram_schmidt"] = "minus_phi_ones"
    maxiter: int = 500


@dataclass(frozen=True)
class AnchorSet:
    x_star: np.ndarray         # (q, k): minimizer of each objective
    F_star: np.ndarray         # (q, q): F_i(x_j*) -- the payoff matrix
    utopia: np.ndarray         # (q,)  : diag(F_star)
    nadir: np.ndarray          # (q,)  : per-objective worst (anti-utopia)
    pseudo_nadir: np.ndarray   # (q,)  : column-wise max of F_star
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class CHIM:
    Phi: np.ndarray     # (q, q): columns = F_star_j - utopia
    n_hat: np.ndarray   # (q,)  : quasi-normal direction


@dataclass(frozen=True)
class NBISubproblemResult:
    beta: np.ndarray           # (q,)
    x: np.ndarray              # (k,)
    t: float
    F_at_x: np.ndarray         # (q,)
    residual_norm: float
    success: bool
    message: str
    optimizer_info: dict[str, Any]


@dataclass(frozen=True)
class NBIRun:
    anchors: AnchorSet
    chim: CHIM
    betas: np.ndarray                    # (N, q)
    candidates: tuple[NBISubproblemResult, ...]
    config: NBIConfig


# ---------------------------------------------------------------------------
# Anchor / payoff matrix (ported core)
# ---------------------------------------------------------------------------


def _multistart_minimize(
    objective: SurrogateCallable,
    cfg: NBIConfig,
) -> tuple[np.ndarray, float, bool, str]:
    bounds = cfg.bounds
    rng = np.random.default_rng(cfg.seed)
    starts = [np.mean(bounds, axis=1)]
    for _ in range(max(0, cfg.n_starts - 1)):
        starts.append(rng.uniform(bounds[:, 0], bounds[:, 1]))

    constraints: list = []
    if cfg.feasibility_constraint is not None:
        constraints.append({"type": "ineq", "fun": cfg.feasibility_constraint})

    best_x: np.ndarray | None = None
    best_f = float("inf")
    msg = "no successful start"
    for x0 in starts:
        try:
            res = minimize(
                fun=lambda x: float(objective(np.asarray(x, dtype=float))),
                x0=np.asarray(x0, dtype=float),
                method=cfg.solver.upper() if cfg.solver == "slsqp" else cfg.solver,
                bounds=[(float(lo), float(hi)) for lo, hi in bounds],
                constraints=constraints,
                options={"maxiter": cfg.maxiter, "disp": False},
            )
        except Exception as exc:  # pragma: no cover - defensive
            msg = f"solver raised: {exc}"
            continue
        if not np.isfinite(res.fun):
            continue
        if res.fun < best_f:
            best_f = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)
            msg = str(res.message)
    if best_x is None:
        raise RuntimeError(f"anchor minimization failed: {msg}")
    return best_x, best_f, True, msg


def compute_anchors(
    surrogates: Sequence[SurrogateCallable],
    cfg: NBIConfig,
) -> AnchorSet:
    """Compute per-objective minimizers and the payoff matrix."""
    q = cfg.objective_count
    if q != len(surrogates):
        raise ValueError(f"objective_count={q} but {len(surrogates)} surrogates were given.")
    k = cfg.bounds.shape[0]
    x_star = np.zeros((q, k), dtype=float)
    F_star = np.zeros((q, q), dtype=float)
    nadir = np.full(q, -np.inf, dtype=float)
    diag_msgs: dict[str, Any] = {}
    for j, fj in enumerate(surrogates):
        x_min, _, _, msg_min = _multistart_minimize(fj, cfg)
        x_star[j, :] = x_min
        for i, fi in enumerate(surrogates):
            F_star[i, j] = float(fi(x_min))
        x_max, neg_f_max, _, msg_max = _multistart_minimize(lambda x, fj=fj: -float(fj(x)), cfg)
        nadir[j] = float(-neg_f_max)
        diag_msgs[f"obj_{j}_min"] = msg_min
        diag_msgs[f"obj_{j}_max"] = msg_max

    utopia = np.diag(F_star).copy()
    pseudo_nadir = F_star.max(axis=1)
    return AnchorSet(
        x_star=x_star,
        F_star=F_star,
        utopia=utopia,
        nadir=nadir,
        pseudo_nadir=pseudo_nadir,
        diagnostics={"messages": diag_msgs},
    )


def build_chim(anchors: AnchorSet, cfg: NBIConfig) -> CHIM:
    """Construct the convex-hull-of-individual-minima matrix and a quasi-normal."""
    q = cfg.objective_count
    Phi = anchors.F_star - anchors.utopia[:, None]
    if cfg.quasi_normal == "minus_phi_ones":
        n_unnorm = -Phi @ np.ones(q, dtype=float)
        norm = float(np.linalg.norm(n_unnorm))
        if norm == 0.0:
            n_hat = np.zeros(q, dtype=float)
            n_hat[0] = -1.0
        else:
            n_hat = n_unnorm / norm
    else:
        _, _, vh = np.linalg.svd(Phi)
        candidate = -vh[-1]
        norm = float(np.linalg.norm(candidate))
        n_hat = candidate / norm if norm > 0 else np.zeros(q)
    return CHIM(Phi=Phi, n_hat=n_hat)


def solve_nbi_subproblem(
    surrogates: Sequence[SurrogateCallable],
    chim: CHIM,
    beta: np.ndarray,
    *,
    anchors: AnchorSet,
    cfg: NBIConfig,
    x0: np.ndarray | None = None,
) -> NBISubproblemResult:
    """Solve `max t s.t. F(x) = utopia + Phi*beta + t*n_hat` over z = [x, t]."""
    q = cfg.objective_count
    k = cfg.bounds.shape[0]
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.shape != (q,):
        raise ValueError(f"beta must have shape ({q},); got {beta_arr.shape}")

    F_target_no_t = anchors.utopia + chim.Phi @ beta_arr

    rng = np.random.default_rng(cfg.seed)
    x0_x = np.mean(cfg.bounds, axis=1) if x0 is None else np.asarray(x0, dtype=float)
    z0 = np.concatenate([x0_x, [0.0]])
    bounds = [(float(lo), float(hi)) for lo, hi in cfg.bounds] + [(0.0, None)]

    def _F(x: np.ndarray) -> np.ndarray:
        return np.array([float(f(x)) for f in surrogates], dtype=float)

    def _eq_constraint(z: np.ndarray) -> np.ndarray:
        return _F(z[:k]) - F_target_no_t - z[k] * chim.n_hat

    constraints: list = [{"type": "eq", "fun": _eq_constraint}]
    if cfg.feasibility_constraint is not None:
        def _feas(z: np.ndarray, fc=cfg.feasibility_constraint) -> np.ndarray:
            return fc(z[:k])
        constraints.append({"type": "ineq", "fun": _feas})

    starts = [z0]
    for _ in range(max(0, cfg.n_starts - 1)):
        rx = rng.uniform(cfg.bounds[:, 0], cfg.bounds[:, 1])
        starts.append(np.concatenate([rx, [0.0]]))

    best: NBISubproblemResult | None = None
    best_t = -np.inf
    fallback: NBISubproblemResult | None = None  # least-residual candidate if no start converges
    for z_start in starts:
        res = minimize(
            fun=lambda z: -float(z[k]),
            x0=z_start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": cfg.maxiter, "ftol": 1e-9, "disp": False},
        )
        x_part = np.asarray(res.x[:k], dtype=float)
        t_val = float(res.x[k])
        F_val = _F(x_part)
        residual_norm = float(np.linalg.norm(F_val - F_target_no_t - t_val * chim.n_hat))
        candidate = NBISubproblemResult(
            beta=beta_arr.copy(),
            x=x_part,
            t=t_val,
            F_at_x=F_val,
            residual_norm=residual_norm,
            success=bool(res.success),
            message=str(res.message),
            optimizer_info={"nfev": int(res.nfev), "nit": int(getattr(res, "nit", 0)),
                            "fun": float(res.fun)},
        )
        if res.success and t_val > best_t and residual_norm < 1e-3:
            best = candidate
            best_t = t_val
        elif fallback is None or residual_norm < fallback.residual_norm:
            fallback = candidate
    if best is None:
        # No start satisfied the NBI equality within tolerance: report the
        # least-residual attempt, explicitly flagged as a failed subproblem
        # (never as a converged one, even if SLSQP claimed success).
        assert fallback is not None
        best = replace(fallback, success=False)
    return best


def run_nbi(
    surrogates: Sequence[SurrogateCallable],
    betas: np.ndarray,
    cfg: NBIConfig,
) -> NBIRun:
    """Run the NBI pipeline: anchors -> CHIM -> per-beta subproblems."""
    betas = np.asarray(betas, dtype=float)
    if betas.ndim != 2 or betas.shape[1] != cfg.objective_count:
        raise ValueError(f"betas must have shape (N, q={cfg.objective_count}); got {betas.shape}")
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    candidates = [
        solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg) for beta in betas
    ]
    return NBIRun(anchors=anchors, chim=chim, betas=betas,
                  candidates=tuple(candidates), config=cfg)


# ---------------------------------------------------------------------------
# Simplex adaptation layer (PCO213 postwork — not in the ported original)
# ---------------------------------------------------------------------------


def lift_simplex(z: np.ndarray, *, atol: float = 1e-8) -> np.ndarray:
    """Map (M-1,) free variables to a valid (M,) simplex point.

    ``w_M = 1 - sum(z)``; tiny numerical violations are clipped and the
    vector renormalized. Raises on violations beyond ``atol``.
    """
    z = np.asarray(z, dtype=float)
    w = np.concatenate([z, [1.0 - float(z.sum())]])
    if w.min() < -max(atol, 1e-6):
        raise ValueError(f"free variables leave the simplex: w={w}")
    w = np.clip(w, 0.0, None)
    w = w / w.sum()
    validate_weights(w[None, :])
    return w


def project_free_vars(z: np.ndarray, *, atol: float = 1e-6) -> tuple[np.ndarray, bool]:
    """Map (M-1,) free variables to a simplex point WITHOUT raising.

    Feasible points (``sum(z) <= 1``) go through :func:`lift_simplex`.
    Infeasible ones — which only arise from a failed NBI subproblem, e.g.
    at a degenerate vertex anchor where SLSQP cannot satisfy the equality
    constraint — are projected onto the face ``w_M = 0`` by renormalizing
    ``z``. Returns ``(w, feasible)`` so callers can flag the candidate.
    """
    z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
    s = float(z.sum())
    if s <= 1.0 + atol:
        return lift_simplex(z), True
    w = np.concatenate([z / s, [0.0]])
    validate_weights(w[None, :])
    return w, False


def objectives_on_free_vars(
    objectives_on_w: Sequence[Callable[[np.ndarray], float]],
) -> list[SurrogateCallable]:
    """Wrap objectives defined on the full weight vector w to act on z (M-1 vars)."""
    def _wrap(f: Callable[[np.ndarray], float]) -> SurrogateCallable:
        def g(z: np.ndarray) -> float:
            z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
            w = np.concatenate([z, [max(0.0, 1.0 - float(z.sum()))]])
            s = w.sum()
            if s > 0:
                w = w / s
            return float(f(w))
        return g
    return [_wrap(f) for f in objectives_on_w]


def simplex_nbi_config(
    n_components: int,
    n_objectives: int,
    *,
    n_starts: int = 10,
    seed: int = 42,
    maxiter: int = 500,
) -> NBIConfig:
    """NBIConfig for the ensemble-weight simplex: M-1 free vars in [0,1]
    with the inequality ``w_M = 1 - sum(z) >= 0``."""
    if n_components < 2:
        raise ValueError("need at least 2 components")
    bounds = np.tile(np.array([0.0, 1.0]), (n_components - 1, 1))
    return NBIConfig(
        objective_count=n_objectives,
        bounds=bounds,
        n_starts=n_starts,
        seed=seed,
        feasibility_constraint=lambda z: np.array([1.0 - float(np.sum(z))]),
        maxiter=maxiter,
    )


def beta_lattice(n_objectives: int, n_points: int) -> np.ndarray:
    """Simplex-lattice of NBI betas with at most ``n_points`` subproblems
    (largest lattice density m such that C(q+m-1, m) <= n_points; m >= 1)."""
    q = n_objectives
    m = 1
    while simplex_lattice_count(q, m + 1) <= max(n_points, q):
        m += 1
    return generate_simplex_lattice(q, m)


def linear_cost(w: np.ndarray, costs: np.ndarray) -> float:
    """Weighted inference cost of the blend: sum_i w_i * c_i (linear in w)."""
    w = np.asarray(w, dtype=float)
    costs = np.asarray(costs, dtype=float)
    if w.shape != costs.shape:
        raise ValueError(f"shape mismatch: w{w.shape} vs costs{costs.shape}")
    return float(w @ costs)


def run_nbi_on_simplex(
    objectives_on_w: Sequence[Callable[[np.ndarray], float]],
    n_components: int,
    *,
    n_points: int = 15,
    n_starts: int = 10,
    seed: int = 42,
    maxiter: int = 500,
    normalize: bool = True,
) -> dict[str, Any]:
    """Full NBI over ensemble weights, in two passes.

    Pass 1 computes anchors/payoff on the RAW objectives (minimization
    convention). If ``normalize``, objectives are rescaled to
    ``(f - utopia) / (pseudo_nadir - utopia)`` — NBI is scale-sensitive and
    AUC/log-loss/cost live on incommensurable scales. Pass 2 runs the NBI
    subproblems on the (normalized) objectives. All anchors and candidates
    are returned as valid simplex weight vectors.
    """
    cfg = simplex_nbi_config(n_components, len(objectives_on_w),
                             n_starts=n_starts, seed=seed, maxiter=maxiter)
    raw = objectives_on_free_vars(objectives_on_w)
    raw_anchors = compute_anchors(raw, cfg)

    if normalize:
        span = raw_anchors.pseudo_nadir - raw_anchors.utopia
        span = np.where(np.abs(span) < 1e-12, 1.0, span)
        surrogates = [
            (lambda z, f=f, u=u, s=s: (float(f(z)) - u) / s)
            for f, u, s in zip(raw, raw_anchors.utopia, span, strict=True)
        ]
    else:
        surrogates = list(raw)

    betas = beta_lattice(len(objectives_on_w), n_points)
    run = run_nbi(surrogates, betas, cfg)

    candidates = []
    for c in run.candidates:
        w, feasible = project_free_vars(c.x)
        candidates.append({
            "beta": c.beta.tolist(),
            "w": w.tolist(),
            "t": c.t,
            "F_normalized": c.F_at_x.tolist(),
            "residual_norm": c.residual_norm,
            "success": bool(c.success) and feasible,
        })
    anchors_w = [lift_simplex(np.clip(x, 0.0, 1.0)).tolist() for x in raw_anchors.x_star]
    return {
        "anchors_w": anchors_w,
        "payoff_raw": raw_anchors.F_star.tolist(),
        "utopia_raw": raw_anchors.utopia.tolist(),
        "pseudo_nadir_raw": raw_anchors.pseudo_nadir.tolist(),
        "n_hat": run.chim.n_hat.tolist(),
        "normalized": bool(normalize),
        "candidates": candidates,
    }


__all__ = [
    "AnchorSet",
    "CHIM",
    "NBIConfig",
    "NBIRun",
    "NBISubproblemResult",
    "beta_lattice",
    "build_chim",
    "compute_anchors",
    "lift_simplex",
    "linear_cost",
    "objectives_on_free_vars",
    "project_free_vars",
    "run_nbi",
    "run_nbi_on_simplex",
    "simplex_nbi_config",
    "solve_nbi_subproblem",
]
