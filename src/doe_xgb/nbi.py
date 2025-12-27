from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_BOUNDS, PARAM_NAMES, INT_PARAMS
from .io_utils import save_csv_ptbr


def evaluate_term(x: Dict[str, float], term: str, eps: float = 1e-12) -> float:
    """Evaluate a polynomial term for a given hyperparameter dict.

    Supports:
    - Intercept / constant terms: "Intercept", "1"
    - Products: "a*b*c"
    - Squares (as products): "a*a"
    - Power notation: "a^2"
    - Reciprocal: "1/a"

    Extra (legacy Minitab mixture patterns are ignored safely if not present).
    """
    t = term.strip().replace(" ", "").replace("(", "").replace(")", "")
    if t in {"Intercept", "const", "1", "CONST"}:
        return 1.0

    def safe_inv(v: float) -> float:
        return (1.0 / v) if abs(v) > eps else 0.0

    # Reciprocal
    if t.startswith("1/"):
        var = t[2:]
        return safe_inv(float(x.get(var, 0.0)))

    # Power notation
    if "^" in t and "*" not in t:
        base, pow_s = t.split("^")
        if base in x:
            return float(x[base]) ** float(pow_s)
        return 0.0

    # Generic product form
    parts = t.split("*")
    val = 1.0
    for p in parts:
        if not p:
            continue
        # power inside product (rare)
        if "^" in p:
            base, pow_s = p.split("^")
            if base in x:
                val *= float(x[base]) ** float(pow_s)
            else:
                val *= 0.0
        else:
            if p in x:
                val *= float(x[p])
            else:
                # unknown token -> neutral
                val *= 1.0
    return float(val)


def predict_from_coeffs(x: Dict[str, float], terms: Sequence[str], coefs: Sequence[float]) -> float:
    return float(sum(c * evaluate_term(x, t) for t, c in zip(terms, coefs)))


def load_coefficients_csv(path: str) -> Tuple[List[str], List[float]]:
    df = pd.read_csv(path, sep=';', decimal=',')
    term_col = [c for c in df.columns if 'term' in c.lower()][0]
    coef_col = [c for c in df.columns if 'coef' in c.lower()][0]
    terms = df[term_col].astype(str).tolist()
    coefs = df[coef_col].astype(float).tolist()
    return terms, coefs


def _cast_int_params(params: Dict[str, float]) -> Dict[str, float | int]:
    out: Dict[str, float | int] = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


@dataclass(frozen=True)
class NBICandidate:
    betas: Tuple[float, float]
    score: float
    predicted: Tuple[float, float]
    params: Dict[str, float | int]
    success: bool
    message: str


def run_nbi_weighted_sum(
    model1: Tuple[Sequence[str], Sequence[float]],
    model2: Tuple[Sequence[str], Sequence[float]],
    *,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    observed_utopia: Optional[Tuple[float, float]] = None,
    observed_nadir: Optional[Tuple[float, float]] = None,
    beta_step: float = 0.05,
    seed: int = 42,
    n_starts: int = 10,
    constrain_pred_range: bool = True,
    maxiter: int = 2000,
) -> List[NBICandidate]:
    """Generate a set of candidates along a beta grid.

    This follows the same logic used in the original notebook:
    maximize sum_i beta_i * normalized(pred_i), solved via SLSQP.

    Parameters
    ----------
    model1, model2:
        Tuples (terms, coefs) for objective 1 and 2.
    observed_utopia / observed_nadir:
        If None, prediction-range constraints are disabled (or you can pass them).
    constrain_pred_range:
        If True and utopia/nadir are provided, enforce nadir <= pred <= utopia for each objective.
    """
    (t1, c1) = model1
    (t2, c2) = model2

    rng = np.random.default_rng(seed)

    # bounds list in PARAM_NAMES order
    bounds_list = [bounds[p] for p in PARAM_NAMES]

    # build beta grid
    betas_grid = []
    b_values = np.arange(0, 1.0 + 1e-9, beta_step)
    for b in b_values:
        b1 = round(1 - float(b), 2)
        b2 = round(float(b), 2)
        betas_grid.append((b1, b2))

    # starting points: center + random
    centers = [(lo + hi) / 2.0 for (lo, hi) in bounds_list]

    x0_list = [centers]
    for _ in range(max(0, n_starts - 1)):
        x0 = [rng.uniform(lo, hi) for (lo, hi) in bounds_list]
        x0_list.append(x0)

    candidates: List[NBICandidate] = []

    # helper to compute preds and normalized objective values
    def preds_and_norm(x_vec: np.ndarray, nadir: np.ndarray, utopia: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = dict(zip(PARAM_NAMES, x_vec.tolist()))
        p1 = predict_from_coeffs(params, t1, c1)
        p2 = predict_from_coeffs(params, t2, c2)
        preds = np.array([p1, p2], dtype=float)
        denom = np.where(np.abs(utopia - nadir) < 1e-12, 1.0, (utopia - nadir))
        norm = (preds - nadir) / denom
        return preds, norm

    for betas in betas_grid:
        betas_arr = np.array(betas, dtype=float)

        best_res = None
        best_score = -np.inf

        for x0 in x0_list:
            x0 = np.array(x0, dtype=float)

            def objective(x_vec: np.ndarray) -> float:
                # Maximize -> minimize negative
                if observed_utopia is None or observed_nadir is None:
                    # if we don't have nadir/utopia, use raw preds
                    params = dict(zip(PARAM_NAMES, x_vec.tolist()))
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
                    # return array of constraints >= 0
                    return np.array([
                        preds[0] - nadir[0],
                        utopia[0] - preds[0],
                        preds[1] - nadir[1],
                        utopia[1] - preds[1],
                    ], dtype=float)

                constraints = [{"type": "ineq", "fun": ineq_pred}]

            res = minimize(
                objective,
                x0=x0,
                method="SLSQP",
                bounds=bounds_list,
                constraints=constraints,
                options={"disp": False, "maxiter": maxiter},
            )

            score = -float(res.fun)
            if score > best_score:
                best_score = score
                best_res = res

        assert best_res is not None
        x_best = best_res.x
        params_best = dict(zip(PARAM_NAMES, x_best.tolist()))
        pred1 = predict_from_coeffs(params_best, t1, c1)
        pred2 = predict_from_coeffs(params_best, t2, c2)

        candidates.append(
            NBICandidate(
                betas=betas,
                score=best_score,
                predicted=(float(pred1), float(pred2)),
                params=_cast_int_params(params_best),
                success=bool(best_res.success),
                message=str(best_res.message),
            )
        )

    return candidates


def save_nbi_candidates(candidates: List[NBICandidate], path: str) -> None:
    rows = []
    for c in candidates:
        rows.append({
            "beta_quality": c.betas[0],
            "beta_cost": c.betas[1],
            "score": c.score,
            "pred_quality": c.predicted[0],
            "pred_cost": c.predicted[1],
            "hyperparameters": c.params,
            "success": c.success,
            "message": c.message,
        })
    df = pd.DataFrame(rows)
    save_csv_ptbr(df, path)
