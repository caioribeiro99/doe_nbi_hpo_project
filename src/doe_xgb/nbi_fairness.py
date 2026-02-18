from __future__ import annotations

"""Fairness-friendly NBI wrapper.

Motivation
----------
The original NBI implementation in :mod:`doe_xgb.nbi` is tied to the classic
hyperparameter list (config.PARAM_NAMES). For the fairness pipeline we often
need extra decision variables such as:

- scale_pos_weight (class imbalance handling)
- threshold (probability -> class label)

To avoid changing the original NBI module (and keep merge with main safe),
this file provides a *parametric* variant of the same algorithm that works
with an explicit list of parameter names and bounds.

Notes on constraints
--------------------
In fairness mode we model both objectives as maximization:

- Score_Quality  = BalancedAccuracy_Mean
- Score_Fairness = FairnessScore_1_minus_Bias

When `clip_pred_range=True`, we add inequality constraints so the predicted
(quality, fairness) remain within the observed [nadir, utopia] box. This
prevents surrogate artifacts (e.g., predicting "fairness > 1" or negative bias).
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.optimize import minimize

from .config import FAIRNESS_DEFAULT_BOUNDS, FAIRNESS_PARAM_NAMES, INT_PARAMS
from .nbi import NBICandidate, predict_from_coeffs

Number = Union[int, float]


def _cast_int_params(params: Dict[str, Number], *, int_params: Sequence[str]) -> Dict[str, Union[int, float]]:
    out: Dict[str, Union[int, float]] = {}
    int_set = set(str(p) for p in int_params)
    for k, v in params.items():
        if k in int_set:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


def _beta_decimals(beta_step: float) -> int:
    """Choose rounding decimals for (1-beta, beta) pairs.

    The original implementation rounds to 2 decimals. That is fine for beta_step=0.05,
    but it collapses many distinct betas when beta_step is smaller (e.g., 0.005 or 0.0025).

    We pick decimals based on the step size to keep the grid meaningful while still
    avoiding floating noise.
    """
    bs = float(beta_step)
    if bs <= 0:
        return 2
    # Example: 0.05 -> 2, 0.01 -> 3, 0.005 -> 4, 0.0025 -> 4
    d = int(max(2, np.ceil(-np.log10(bs)) + 1))
    return int(min(d, 6))


def run_nbi_weighted_sum_parametric(
    model1: Tuple[Sequence[str], Sequence[float]],
    model2: Tuple[Sequence[str], Sequence[float]],
    *,
    param_names: Sequence[str],
    bounds: Dict[str, Tuple[float, float]],
    observed_utopia: Optional[Tuple[float, float]] = None,
    observed_nadir: Optional[Tuple[float, float]] = None,
    beta_step: float = 0.05,
    seed: int = 42,
    n_starts: int = 10,
    constrain_pred_range: bool = True,
    maxiter: int = 2000,
) -> List[NBICandidate]:
    """Parametric NBI weighted-sum optimizer (same logic as doe_xgb.nbi.run_nbi_weighted_sum)."""
    (t1, c1) = model1
    (t2, c2) = model2

    pnames = [str(p) for p in param_names]

    rng = np.random.default_rng(int(seed))
    bounds_list = [bounds[p] for p in pnames]
    centers = [(lo + hi) / 2.0 for (lo, hi) in bounds_list]

    # beta grid: (1-beta, beta)
    decimals = _beta_decimals(float(beta_step))
    b_values = np.arange(float(beta_step), 1.0 + 1e-9, float(beta_step))
    betas_grid = [(round(1.0 - float(b), decimals), round(float(b), decimals)) for b in b_values]
    # de-dup in case rounding collides
    betas_grid = list(dict.fromkeys(betas_grid))

    x0_list: List[np.ndarray] = [np.array(centers, dtype=float)]
    for _ in range(max(0, int(n_starts) - 1)):
        x0 = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds_list], dtype=float)
        x0_list.append(x0)

    def preds_and_norm(x_vec: np.ndarray, nadir: np.ndarray, utopia: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = dict(zip(pnames, x_vec.tolist()))
        p1 = predict_from_coeffs(params, t1, c1)
        p2 = predict_from_coeffs(params, t2, c2)
        preds = np.array([p1, p2], dtype=float)
        denom = np.where(np.abs(utopia - nadir) < 1e-12, 1.0, (utopia - nadir))
        norm = (preds - nadir) / denom
        return preds, norm

    candidates: List[NBICandidate] = []

    for betas in betas_grid:
        betas_arr = np.array(betas, dtype=float)

        best_res = None
        best_score = -np.inf

        for x0 in x0_list:

            def objective(x_vec: np.ndarray) -> float:
                if observed_utopia is None or observed_nadir is None:
                    params = dict(zip(pnames, x_vec.tolist()))
                    p1 = predict_from_coeffs(params, t1, c1)
                    p2 = predict_from_coeffs(params, t2, c2)
                    return -float(betas_arr[0] * p1 + betas_arr[1] * p2)

                nadir = np.array(observed_nadir, dtype=float)
                utopia = np.array(observed_utopia, dtype=float)
                _, norm = preds_and_norm(x_vec, nadir, utopia)
                return -float(np.dot(betas_arr, norm))

            constraints = []
            if constrain_pred_range and observed_utopia is not None and observed_nadir is not None:
                nadir = np.array(observed_nadir, dtype=float)
                utopia = np.array(observed_utopia, dtype=float)

                def ineq_pred(x_vec: np.ndarray) -> np.ndarray:
                    preds, _ = preds_and_norm(x_vec, nadir, utopia)
                    return np.array(
                        [
                            preds[0] - nadir[0],
                            utopia[0] - preds[0],
                            preds[1] - nadir[1],
                            utopia[1] - preds[1],
                        ],
                        dtype=float,
                    )

                constraints.append({"type": "ineq", "fun": ineq_pred})

            res = minimize(
                fun=objective,
                x0=x0,
                method="SLSQP",
                bounds=bounds_list,
                constraints=constraints,
                options={"maxiter": int(maxiter), "disp": False},
            )

            score = -float(res.fun) if res.success else -np.inf
            if score > best_score:
                best_score = score
                best_res = res

        assert best_res is not None

        params_vec = best_res.x
        params_dict = dict(zip(pnames, params_vec.tolist()))
        pred1 = predict_from_coeffs(params_dict, t1, c1)
        pred2 = predict_from_coeffs(params_dict, t2, c2)

        candidates.append(
            NBICandidate(
                betas=betas,
                score=float(best_score),
                predicted=(float(pred1), float(pred2)),
                params=_cast_int_params(cast(Dict[str, Number], params_dict), int_params=INT_PARAMS),
                success=bool(best_res.success),
                message=str(best_res.message),
            )
        )

    return candidates


def run_nbi_weighted_sum_fairness(
    model_quality: Tuple[Sequence[str], Sequence[float]],
    model_fairness: Tuple[Sequence[str], Sequence[float]],
    *,
    bounds: Dict[str, Tuple[float, float]] = FAIRNESS_DEFAULT_BOUNDS,
    observed_utopia: Optional[Tuple[float, float]] = None,
    observed_nadir: Optional[Tuple[float, float]] = None,
    beta_step: float = 0.05,
    seed: int = 42,
    n_starts: int = 10,
    clip_pred_range: bool = True,
    maxiter: int = 2000,
) -> List[NBICandidate]:
    """Fairness preset: param_names=FAIRNESS_PARAM_NAMES, bounds=FAIRNESS_DEFAULT_BOUNDS."""
    return run_nbi_weighted_sum_parametric(
        model_quality,
        model_fairness,
        param_names=FAIRNESS_PARAM_NAMES,
        bounds=bounds,
        observed_utopia=observed_utopia,
        observed_nadir=observed_nadir,
        beta_step=beta_step,
        seed=seed,
        n_starts=n_starts,
        constrain_pred_range=clip_pred_range,
        maxiter=maxiter,
    )
