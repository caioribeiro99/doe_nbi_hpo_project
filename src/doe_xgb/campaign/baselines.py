"""Direct baselines and the evolutionary comparator, at a matched logical budget.

Every method here evaluates through its own isolated ``MethodView``. None can see
another's history, and physical memoization beneath that boundary never becomes
shared information.

Each returns realized configurations with their real responses. Nothing is scored
here; scoring happens once, against the common references, after every method has
returned.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from .design import from_coded, to_coded
from .evaluator import BOUNDS, INT_PARAMS, PARAMS


# Each comparator draws from its own stream. Sharing one seed made the panel
# substantially one baseline: at a budget of 386 over seven factors the grid is two
# levels per factor (128 corners) padded with 258 random points, and those padding
# points were bit-identical to random search's first 258, while the Bayesian and
# Parzen initial designs were bit-identical to random search's first 77. Pairing
# across replications is preserved because the offsets are fixed and declared.
SEED_OFFSET = {"grid": 101, "random": 202, "bayes_quality": 303, "bayes_cost": 404,
               "tpe_quality": 505, "tpe_cost": 606, "nsga2": 707}


def method_seed(base: int, method: str) -> int:
    if method not in SEED_OFFSET:
        raise KeyError(f"no declared seed offset for comparator {method!r}")
    return int(base) + SEED_OFFSET[method]


def _realize(xc: np.ndarray) -> dict[str, Any]:
    from .arms import Realizer
    cfg, _ = Realizer(PARAMS, BOUNDS, list(INT_PARAMS)).realize(xc)
    return cfg


def _collect(view, configs: list[dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        rows.append({**cfg, **view.evaluate(cfg)})
    return pd.DataFrame(rows)


def coarse_grid(view, budget: int, seed: int) -> pd.DataFrame:
    """A grid with as many levels per factor as the budget allows, then padded.

    At a budget of 386 over seven factors this is two levels per factor, so the
    mesh is 128 corners and 258 of the 386 points are uniform padding. The split is
    reported rather than hidden: calling a set that is two-thirds random a "coarse
    grid" without saying so is the kind of thing that costs a paper its comparator
    section.
    """
    k = len(PARAMS)
    levels = max(2, int(np.floor(budget ** (1.0 / k))))
    axes = [np.linspace(-1.0, 1.0, levels) for _ in range(k)]
    mesh = np.array(np.meshgrid(*axes)).reshape(k, -1).T
    rng = np.random.default_rng(seed)
    if len(mesh) > budget:
        mesh = mesh[rng.choice(len(mesh), size=budget, replace=False)]
    elif len(mesh) < budget:
        extra = rng.uniform(-1.0, 1.0, size=(budget - len(mesh), k))
        mesh = np.vstack([mesh, extra])
    out = _collect(view, [_realize(x) for x in mesh])
    out.attrs["grid_levels_per_factor"] = int(levels)
    out.attrs["mesh_points"] = int(min(levels ** k, budget))
    out.attrs["random_padding_points"] = int(max(0, budget - levels ** k))
    return out


def random_search(view, budget: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return _collect(view, [_realize(rng.uniform(-1.0, 1.0, size=len(PARAMS)))
                           for _ in range(budget)])


def _scalar_bo(view, budget: int, seed: int, objective_index: int,
               to_objectives: Callable[[pd.DataFrame], np.ndarray],
               kind: str) -> pd.DataFrame:
    """Single-objective model-based search on one of the two objectives.

    Reported on single-objective endpoints only and never in the front-indicator
    table: two single-objective optima are not a Pareto front, and scoring them as
    one would look terrible for a reason that says nothing about the method.
    """
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    rng = np.random.default_rng(seed)
    k = len(PARAMS)
    n_init = max(8, budget // 5)
    X = rng.uniform(-1.0, 1.0, size=(n_init, k))
    rows = _collect(view, [_realize(x) for x in X])
    y = to_objectives(rows)[:, objective_index]
    for _ in range(budget - n_init):
        if kind == "bayes":
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(1e-6),
                normalize_y=True, random_state=seed).fit(X, y)
            cand = rng.uniform(-1.0, 1.0, size=(512, k))
            mu, sd = gp.predict(cand, return_std=True)
            best = y.min()
            z = (best - mu) / np.maximum(sd, 1e-12)
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            x_next = cand[int(np.argmax(ei))]
        else:                                    # tree-structured Parzen estimator
            n_good = max(3, int(0.25 * len(y)))
            order = np.argsort(y)
            good, bad = X[order[:n_good]], X[order[n_good:]]
            cand = rng.uniform(-1.0, 1.0, size=(512, k))
            bw = max(0.1, 1.0 / np.sqrt(len(y)))

            def logdens(S):
                d2 = ((cand[:, None, :] - S[None, :, :]) ** 2).sum(-1)
                return np.log(np.exp(-0.5 * d2 / bw ** 2).mean(1) + 1e-300)
            x_next = cand[int(np.argmax(logdens(good) - logdens(bad)))]
        new = _collect(view, [_realize(x_next)])
        X = np.vstack([X, x_next])
        rows = pd.concat([rows, new], ignore_index=True)
        y = to_objectives(rows)[:, objective_index]
    return rows


def bayesian_optimization(view, budget: int, seed: int, to_objectives, obj: int) -> pd.DataFrame:
    return _scalar_bo(view, budget, seed, obj, to_objectives, "bayes")


def tpe(view, budget: int, seed: int, to_objectives, obj: int) -> pd.DataFrame:
    return _scalar_bo(view, budget, seed, obj, to_objectives, "tpe")


def nsga2(view, population: int, generations: int, seed: int,
          to_objectives: Callable[[pd.DataFrame], np.ndarray]) -> pd.DataFrame:
    """One canonical NSGA-II configuration, frozen in protocol/NSGA2_DECISION.md.

    Nothing about it is tuned. It optimizes the same two frozen objectives through
    the same evaluator, and its realized configurations are produced by the same
    deterministic realizer as the arms.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.termination.max_gen import MaximumGenerationTermination

    k = len(PARAMS)
    collected: list[dict] = []

    class _P(Problem):
        def __init__(self):
            super().__init__(n_var=k, n_obj=2, xl=-1.0, xu=1.0)

        def _evaluate(self, X, out, *args, **kwargs):
            rows = _collect(view, [_realize(x) for x in np.atleast_2d(X)])
            collected.extend(rows.to_dict("records"))
            out["F"] = to_objectives(rows)

    algo = NSGA2(pop_size=population,
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20, prob=1.0 / k),
                 eliminate_duplicates=True)
    minimize(_P(), algo, MaximumGenerationTermination(generations),
             seed=seed, verbose=False)
    return pd.DataFrame(collected)


__all__ = ["coarse_grid", "random_search", "bayesian_optimization", "tpe", "nsga2"]
