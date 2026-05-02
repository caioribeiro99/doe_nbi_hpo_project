"""True Normal Boundary Intersection (NBI) for arbitrary q >= 2 objectives.

Implements the Das & Dennis (1998) NBI subproblem:

    max  t
   x, t
   s.t. F̂(x) = utopia + Phi @ beta + t * n_hat
        x in Omega
        optionally: g(x) <= 0       (e.g., x.T @ x <= rho^2)

where:
- ``F̂(x)`` is the vector of *minimization* surrogates (after objective
  canonicalization);
- ``Phi`` is the payoff matrix with columns ``F^*_j - utopia``;
- ``utopia`` is the diagonal of the payoff matrix (per-objective optimum);
- ``n_hat`` is the quasi-normal direction
  (default: ``-Phi @ ones / ||Phi @ ones||`` -- Das & Dennis convention).

The mathematical core in this module is *dimension-agnostic*: every
operation is expressed for an arbitrary ``q >= 2`` (no ``if q == 2`` or
``if q == 3`` branching). Visualization helpers in :mod:`reporting`
remain 2D/3D-specific.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize

SurrogateCallable = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class NBIConfig:
    objective_count: int
    bounds: np.ndarray  # shape (k, 2): per-decision-variable [lo, hi]
    integer_dims: tuple[int, ...] = ()
    n_starts: int = 10
    seed: int = 42
    solver: Literal["slsqp", "trust_constr"] = "slsqp"
    feasibility_constraint: Callable[[np.ndarray], np.ndarray] | None = None
    use_pseudo_nadir: bool = True
    quasi_normal: Literal["minus_phi_ones", "gram_schmidt"] = "minus_phi_ones"
    record_trajectory: bool = False
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
    weights: np.ndarray                  # (N, q)
    candidates: tuple[NBISubproblemResult, ...]
    config: NBIConfig


# ---------------------------------------------------------------------------
# Anchor / payoff matrix
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
    success_flag = False
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
            success_flag = bool(res.success)
            msg = str(res.message)
    if best_x is None:
        raise RuntimeError(f"anchor minimization failed: {msg}")
    return best_x, best_f, success_flag, msg


def compute_anchors(
    surrogates: Sequence[SurrogateCallable],
    cfg: NBIConfig,
) -> AnchorSet:
    """Compute per-objective minimizers and the payoff matrix.

    Each row ``i`` of ``F_star`` is the value of objective ``i`` at every
    minimizer ``x_j*``. The diagonal is the utopia point.
    """
    q = cfg.objective_count
    if q != len(surrogates):
        raise ValueError(
            f"objective_count={q} but {len(surrogates)} surrogates were given."
        )
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
        # Anti-optimization: the worst observed value of fj across all anchors
        # contributes to the nadir; we also explicitly maximize fj for nadir.
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


# ---------------------------------------------------------------------------
# CHIM and quasi-normal
# ---------------------------------------------------------------------------


def build_chim(anchors: AnchorSet, cfg: NBIConfig) -> CHIM:
    """Construct the convex-hull-of-individual-minima matrix and a quasi-normal."""
    q = cfg.objective_count
    Phi = anchors.F_star - anchors.utopia[:, None]  # columns = F_j* - utopia
    if cfg.quasi_normal == "minus_phi_ones":
        n_unnorm = -Phi @ np.ones(q, dtype=float)
        norm = float(np.linalg.norm(n_unnorm))
        if norm == 0.0:
            n_hat = np.zeros(q, dtype=float)
            n_hat[0] = -1.0
        else:
            n_hat = n_unnorm / norm
    else:
        # Gram-Schmidt: complement to the column space of Phi.
        # Project standard basis vectors onto null(Phi.T).
        u, s, vh = np.linalg.svd(Phi)
        # The smallest singular vector gives a direction of low Phi-projection.
        candidate = -vh[-1]
        norm = float(np.linalg.norm(candidate))
        n_hat = candidate / norm if norm > 0 else np.zeros(q)

    return CHIM(Phi=Phi, n_hat=n_hat)


# ---------------------------------------------------------------------------
# Subproblem
# ---------------------------------------------------------------------------


def solve_nbi_subproblem(
    surrogates: Sequence[SurrogateCallable],
    chim: CHIM,
    beta: np.ndarray,
    *,
    anchors: AnchorSet,
    cfg: NBIConfig,
    x0: np.ndarray | None = None,
) -> NBISubproblemResult:
    """Solve the NBI subproblem `max t s.t. F̂(x) = utopia + Phi*beta + t*n_hat`.

    Decision variables: ``[x_1, ..., x_k, t]``. Objective: ``-t`` (minimize).
    Equality constraints: ``q`` scalar equalities, fed to SLSQP as a vector.
    Optional inequality: ``cfg.feasibility_constraint(x)``.
    """
    q = cfg.objective_count
    k = cfg.bounds.shape[0]
    beta_arr = np.asarray(beta, dtype=float)
    if beta_arr.shape != (q,):
        raise ValueError(f"beta must have shape ({q},); got {beta_arr.shape}")

    F_target_no_t = anchors.utopia + chim.Phi @ beta_arr  # shape (q,)

    rng = np.random.default_rng(cfg.seed)
    if x0 is None:
        x0_x = np.mean(cfg.bounds, axis=1)
    else:
        x0_x = np.asarray(x0, dtype=float)
    z0 = np.concatenate([x0_x, [0.0]])

    bounds = [(float(lo), float(hi)) for lo, hi in cfg.bounds] + [(0.0, None)]

    def _F(x: np.ndarray) -> np.ndarray:
        return np.array([float(f(x)) for f in surrogates], dtype=float)

    def _eq_constraint(z: np.ndarray) -> np.ndarray:
        x_part = z[:k]
        t_val = z[k]
        return _F(x_part) - F_target_no_t - t_val * chim.n_hat

    constraints = [{"type": "eq", "fun": _eq_constraint}]
    if cfg.feasibility_constraint is not None:
        def _feas(z: np.ndarray, fc=cfg.feasibility_constraint) -> np.ndarray:
            return fc(z[:k])
        constraints.append({"type": "ineq", "fun": _feas})

    def _objective(z: np.ndarray) -> float:
        return -float(z[k])

    starts = [z0]
    for _ in range(max(0, cfg.n_starts - 1)):
        rx = rng.uniform(cfg.bounds[:, 0], cfg.bounds[:, 1])
        starts.append(np.concatenate([rx, [0.0]]))

    best: NBISubproblemResult | None = None
    best_t = -np.inf

    for z_start in starts:
        res = minimize(
            fun=_objective,
            x0=z_start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": cfg.maxiter, "ftol": 1e-9, "disp": False},
        )
        x_part = np.asarray(res.x[:k], dtype=float)
        t_val = float(res.x[k])
        F_val = _F(x_part)
        residual = F_val - F_target_no_t - t_val * chim.n_hat
        residual_norm = float(np.linalg.norm(residual))
        candidate = NBISubproblemResult(
            beta=beta_arr.copy(),
            x=x_part,
            t=t_val,
            F_at_x=F_val,
            residual_norm=residual_norm,
            success=bool(res.success),
            message=str(res.message),
            optimizer_info={
                "nfev": int(res.nfev),
                "nit": int(getattr(res, "nit", 0)),
                "fun": float(res.fun),
            },
        )
        if res.success and t_val > best_t and residual_norm < 1e-3:
            best = candidate
            best_t = t_val
        elif best is None:
            best = candidate
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_nbi(
    surrogates: Sequence[SurrogateCallable],
    weights: np.ndarray,
    cfg: NBIConfig,
) -> NBIRun:
    """Run the NBI pipeline: anchors -> CHIM -> per-weight subproblems."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2 or weights.shape[1] != cfg.objective_count:
        raise ValueError(
            f"weights must have shape (N, q={cfg.objective_count}); got {weights.shape}"
        )
    anchors = compute_anchors(surrogates, cfg)
    chim = build_chim(anchors, cfg)
    candidates: list = []
    for beta in weights:
        candidates.append(
            solve_nbi_subproblem(surrogates, chim, beta, anchors=anchors, cfg=cfg)
        )
    return NBIRun(
        anchors=anchors,
        chim=chim,
        weights=weights,
        candidates=tuple(candidates),
        config=cfg,
    )


__all__ = [
    "AnchorSet",
    "CHIM",
    "NBIConfig",
    "NBIRun",
    "NBISubproblemResult",
    "build_chim",
    "compute_anchors",
    "run_nbi",
    "solve_nbi_subproblem",
]
