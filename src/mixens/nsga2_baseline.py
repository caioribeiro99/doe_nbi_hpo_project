"""Evaluation-matched NSGA-II baseline on the ensemble-weight simplex.

External canonical multiobjective optimizer, run on EXACTLY the three real cached
out-of-fold objectives that NBI-C optimized:

    f1 = -ROC-AUC      f2 = log-loss      f3 = weighted cost  sum_i w_i c_i

Nothing here trains a model or regenerates out-of-fold predictions: the frozen
R = 30 artifacts under ``experiments/pco213_postwork_benchmark`` are read-only
inputs. The configuration is pinned in
``papers/surrogate_nbi_ensemble/nsga2_preregistered_config.json`` and was committed
before any run.

Design decisions, all fixed in advance:

* population size 66, matching the 66 NBI beta directions;
* termination by objective-evaluation budget matched per dataset x replication to
  the real-objective evaluation count NBI-C actually consumed;
* pymoo default variation operators (SBX eta=15 p=0.9, PM eta=20 p=1/n_var);
* Dirichlet(1) initialization and a deterministic simplex repair after every
  sampling, crossover and mutation step;
* seed = 20260904 + rep + 770000.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fastmetrics import evaluate_weights

SEED_BASE = 20260904
SEED_OFFSET = 770000
POP_SIZE = 66
SUPPORT_EPS = 1e-3


# --------------------------------------------------------------------------------------
# Simplex repair
# --------------------------------------------------------------------------------------
def repair_simplex(X: np.ndarray) -> np.ndarray:
    """Deterministically map rows of ``X`` onto the unit simplex.

    Negative entries are clipped to zero and each row is divided by its sum. A row
    whose entries all clip to zero falls back to the uniform composition. This is a
    documented repair, not the Euclidean projection onto the simplex; the two differ
    in general, and the manuscript describes it as a repair.

    Idempotent: ``repair_simplex(repair_simplex(X)) == repair_simplex(X)``.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X = np.clip(X, 0.0, None)
    s = X.sum(axis=1, keepdims=True)
    degenerate = (s <= 0).ravel()
    if degenerate.any():
        X[degenerate] = 1.0 / X.shape[1]
        s = X.sum(axis=1, keepdims=True)
    return X / s


def sample_simplex(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` compositions of ``m`` components drawn from Dirichlet(1) (uniform on the simplex)."""
    return rng.dirichlet(np.ones(m), size=n)


# --------------------------------------------------------------------------------------
# Problem definition
# --------------------------------------------------------------------------------------
@dataclass
class EvalCounter:
    n: int = 0


def make_problem(P: np.ndarray, y: np.ndarray, costs: np.ndarray, counter: EvalCounter,
                 n_jobs: int = 1):
    """Build the pymoo Problem wrapping the three real cached-OOF objectives."""
    from pymoo.core.problem import Problem

    M = P.shape[1]

    class EnsembleWeightProblem(Problem):
        def __init__(self):
            super().__init__(n_var=M, n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            W = repair_simplex(X)
            res = evaluate_weights(P, y, W, costs=costs, support_eps=SUPPORT_EPS,
                                   chunk=max(len(W), 1), with_pr_auc=False, n_jobs=n_jobs)
            counter.n += len(W)
            out["F"] = np.column_stack([-res["roc_auc"], res["log_loss"], res["cost_weighted"]])
            # hand the repaired composition back so the population stays feasible
            out["X"] = W

    return EnsembleWeightProblem()


class SimplexSampling:
    """Dirichlet(1) initialization, wired into pymoo's Sampling interface."""

    def __new__(cls, seed: int):
        from pymoo.core.sampling import Sampling

        class _S(Sampling):
            def __init__(self, seed):
                super().__init__()
                self.rng = np.random.default_rng(seed)

            def _do(self, problem, n_samples, **kwargs):
                return sample_simplex(n_samples, problem.n_var, self.rng)

        return _S(seed)


class SimplexRepair:
    """pymoo Repair that applies :func:`repair_simplex` after every variation step."""

    def __new__(cls):
        from pymoo.core.repair import Repair

        class _R(Repair):
            def _do(self, problem, X, **kwargs):
                return repair_simplex(X)

        return _R()


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
def n_gen_for_budget(target_evals: int, pop_size: int = POP_SIZE) -> int:
    """Generations such that ``pop_size * n_gen`` is as close as possible to ``target_evals``.

    pymoo's ``MaximumGenerationTermination(n)`` evaluates the initial population and then
    ``n - 1`` offspring generations, so the realized objective-evaluation count is
    ``pop_size * n_gen`` (verified by the evaluation counter in the unit tests).
    """
    return max(1, int(round(target_evals / pop_size)))


@dataclass
class NSGA2Result:
    dataset: str
    rep: int
    seed: int
    target_evals: int
    actual_evals: int
    n_gen: int
    pop_size: int
    seconds: float
    W_final: np.ndarray = field(repr=False)
    F_final: np.ndarray = field(repr=False)
    W_nd: np.ndarray = field(repr=False)
    F_nd: np.ndarray = field(repr=False)


def run_nsga2(P: np.ndarray, y: np.ndarray, costs: np.ndarray, *, dataset: str, rep: int,
              target_evals: int, pop_size: int = POP_SIZE, n_jobs: int = 1) -> NSGA2Result:
    """Run one evaluation-matched NSGA-II replication on cached OOF probabilities."""
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.termination.max_gen import MaximumGenerationTermination

    seed = SEED_BASE + rep + SEED_OFFSET
    n_gen = n_gen_for_budget(target_evals, pop_size)
    counter = EvalCounter()
    problem = make_problem(P, y, costs, counter, n_jobs=n_jobs)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SimplexSampling(seed),
        crossover=SBX(eta=15, prob=0.9),
        mutation=PM(eta=20),
        repair=SimplexRepair(),
        eliminate_duplicates=True,
    )

    t0 = time.perf_counter()
    res = minimize(problem, algorithm, MaximumGenerationTermination(n_gen),
                   seed=seed, save_history=False, verbose=False)
    seconds = time.perf_counter() - t0

    W_final = repair_simplex(np.atleast_2d(res.pop.get("X")))
    F_final = np.atleast_2d(res.pop.get("F"))
    W_nd = repair_simplex(np.atleast_2d(res.X))
    F_nd = np.atleast_2d(res.F)

    return NSGA2Result(dataset=dataset, rep=rep, seed=seed, target_evals=int(target_evals),
                       actual_evals=int(counter.n), n_gen=int(n_gen), pop_size=int(pop_size),
                       seconds=float(seconds), W_final=W_final, F_final=F_final,
                       W_nd=W_nd, F_nd=F_nd)


def save_result(res: NSGA2Result, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "nsga2_population.npz", W_final=res.W_final, F_final=res.F_final,
                        W_nd=res.W_nd, F_nd=res.F_nd)
    meta = {k: v for k, v in res.__dict__.items() if not isinstance(v, np.ndarray)}
    meta["n_final"] = int(len(res.W_final))
    meta["n_nondominated"] = int(len(res.W_nd))
    meta["budget_ratio_actual_over_target"] = res.actual_evals / res.target_evals
    (out_dir / "nsga2_summary.json").write_text(json.dumps(meta, indent=2))


__all__ = ["repair_simplex", "sample_simplex", "n_gen_for_budget", "run_nsga2",
           "save_result", "NSGA2Result", "SEED_BASE", "SEED_OFFSET", "POP_SIZE"]
