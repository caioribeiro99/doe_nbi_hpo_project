"""Legacy weighted-sum scalarization (formerly mis-labeled as ``nbi.py``).

This module preserves the dissertation-era implementation of the
``run_nbi_weighted_sum`` function and its supporting types. **It is not
NBI**: it solves
``min_x  -beta1 * norm(F1(x)) - beta2 * norm(F2(x))`` over a beta grid,
which is a normalized weighted-sum scalarization. See
``docs/METHODOLOGY_DECISIONS.md`` D1 for the full discussion.

True N-objective NBI lives in :mod:`doe_xgb.nbi_core`. This module is
kept as an ablation baseline and for backward compatibility with the
dissertation outputs.

The function names retained for compatibility (``run_nbi_weighted_sum``,
``NBICandidate``, etc.) are imported through the deprecated shim
:mod:`doe_xgb.nbi`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Union, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_BOUNDS, INT_PARAMS, PARAM_NAMES
from .io_utils import save_csv_ptbr

# Runtime alias kept as Union so the legacy module still imports on
# Python 3.9 (project minimum is 3.10). Pure annotations elsewhere
# already use ``X | Y``.
Number = Union[int, float]  # noqa: UP007


def evaluate_term(x: dict[str, Number], term: str, eps: float = 1e-12) -> float:
    t = term.strip().replace(" ", "").replace("(", "").replace(")", "")

    if t in {"Intercept", "const", "CONST", "1"}:
        return 1.0

    def safe_inv(v: float) -> float:
        return (1.0 / v) if abs(v) > eps else 0.0

    if t.startswith("1/"):
        var = t[2:]
        v = float(x.get(var, 0.0))
        return safe_inv(v)

    if "^" in t and "*" not in t:
        base, pow_s = t.split("^", 1)
        if base in x:
            return float(x[base]) ** float(pow_s)
        return 0.0

    parts = [p for p in t.split("*") if p]
    val = 1.0
    for p in parts:
        if "^" in p:
            base, pow_s = p.split("^", 1)
            if base in x:
                val *= float(x[base]) ** float(pow_s)
            else:
                val *= 0.0
        else:
            if p in x:
                val *= float(x[p])
            else:
                val *= 1.0
    return float(val)


def predict_from_coeffs(x: dict[str, Number], terms: Sequence[str], coefs: Sequence[float]) -> float:
    return float(sum(float(c) * evaluate_term(x, str(t)) for t, c in zip(terms, coefs, strict=False)))


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Find a column in df by trying candidates (case-insensitive exact, then contains match)."""
    cols: list[str] = [str(c) for c in df.columns]
    cols_lower = [c.strip().lower() for c in cols]

    # exact match
    for cand in candidates:
        cand_l = cand.lower()
        for i, c_l in enumerate(cols_lower):
            if cand_l == c_l:
                return cols[i]

    # contains match
    for cand in candidates:
        cand_l = cand.lower()
        for i, c_l in enumerate(cols_lower):
            if cand_l in c_l:
                return cols[i]

    raise KeyError(f"Could not find any of columns {list(candidates)} in dataframe columns: {cols}")


def load_coefficients_csv(path: str) -> tuple[list[str], list[float]]:
    """
    Load coefficients exported by our RSM module (CSV pt-BR friendly).

    Expected columns: something like ["Term", "Coef"] (case-insensitive).
    """
    df = pd.read_csv(path, sep=";", decimal=",")

    term_col = _find_column(df, ["term", "terms"])
    coef_col = _find_column(df, ["coef", "coefs", "coefficient", "estimate"])

    # Force types explicitly to satisfy type checkers (Pylance)
    term_series = cast(pd.Series, df[term_col]).astype("string")
    coef_series = cast(pd.Series, df[coef_col]).astype(float)

    terms: list[str] = [str(v) for v in term_series.tolist()]
    coefs: list[float] = [float(v) for v in coef_series.tolist()]

    return terms, coefs


def _cast_int_params(params: dict[str, Number]) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            out[k] = int(round(float(v)))
        else:
            out[k] = float(v)
    return out


@dataclass(frozen=True)
class NBICandidate:
    betas: tuple[float, float]
    score: float
    predicted: tuple[float, float]
    params: dict[str, int | float]
    success: bool
    message: str


def run_nbi_weighted_sum(
    model1: tuple[Sequence[str], Sequence[float]],
    model2: tuple[Sequence[str], Sequence[float]],
    *,
    bounds: dict[str, tuple[float, float]] = DEFAULT_BOUNDS,
    observed_utopia: tuple[float, float] | None = None,
    observed_nadir: tuple[float, float] | None = None,
    beta_step: float = 0.05,
    seed: int = 42,
    n_starts: int = 10,
    constrain_pred_range: bool = True,
    maxiter: int = 2000,
) -> list[NBICandidate]:
    (t1, c1) = model1
    (t2, c2) = model2

    rng = np.random.default_rng(seed)
    bounds_list = [bounds[p] for p in PARAM_NAMES]
    centers = [(lo + hi) / 2.0 for (lo, hi) in bounds_list]

    # 20 betas: 0.05..1.00
    b_values = np.arange(beta_step, 1.0 + 1e-9, beta_step)
    betas_grid = [(round(1.0 - float(b), 2), round(float(b), 2)) for b in b_values]

    x0_list: list[np.ndarray] = [np.array(centers, dtype=float)]
    for _ in range(max(0, n_starts - 1)):
        x0 = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds_list], dtype=float)
        x0_list.append(x0)

    def preds_and_norm(x_vec: np.ndarray, nadir: np.ndarray, utopia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        params = dict(zip(PARAM_NAMES, x_vec.tolist(), strict=False))
        p1 = predict_from_coeffs(params, t1, c1)
        p2 = predict_from_coeffs(params, t2, c2)
        preds = np.array([p1, p2], dtype=float)
        denom = np.where(np.abs(utopia - nadir) < 1e-12, 1.0, (utopia - nadir))
        norm = (preds - nadir) / denom
        return preds, norm

    candidates: list[NBICandidate] = []

    for betas in betas_grid:
        betas_arr = np.array(betas, dtype=float)

        best_res = None
        best_score = -np.inf

        for x0 in x0_list:

            def objective(x_vec: np.ndarray, betas_arr: np.ndarray = betas_arr) -> float:
                if observed_utopia is None or observed_nadir is None:
                    params = dict(zip(PARAM_NAMES, x_vec.tolist(), strict=False))
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

                def ineq_pred(
                    x_vec: np.ndarray,
                    nadir: np.ndarray = nadir,
                    utopia: np.ndarray = utopia,
                ) -> np.ndarray:
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
                options={"maxiter": maxiter, "disp": False},
            )

            score = -float(res.fun) if res.success else -np.inf
            if score > best_score:
                best_score = score
                best_res = res

        assert best_res is not None

        params_vec = best_res.x
        params_dict = dict(zip(PARAM_NAMES, params_vec.tolist(), strict=False))
        pred1 = predict_from_coeffs(params_dict, t1, c1)
        pred2 = predict_from_coeffs(params_dict, t2, c2)

        candidates.append(
            NBICandidate(
                betas=betas,
                score=float(best_score),
                predicted=(float(pred1), float(pred2)),
                params=_cast_int_params(cast(dict[str, Number], params_dict)),
                success=bool(best_res.success),
                message=str(best_res.message),
            )
        )

    return candidates


def nbi_candidates_to_df(candidates: list[NBICandidate]) -> pd.DataFrame:
    """Convert a list of :class:`NBICandidate` to a flat pandas DataFrame.

    The returned schema is used by the pipeline and by `load_nbi_candidates`:

    - beta_1, beta_2: the (beta1, beta2) weights
    - score: the scalarized objective used during optimization
    - Pred_Score_Quality, Pred_Score_Cost: predicted objective scores
    - hyperparameters: dict with param values (ints already coerced)
    - success, message: optimizer status
    """

    rows: list[dict[str, Any]] = []
    for c in candidates:
        rows.append(
            {
                "beta_1": float(c.betas[0]),
                "beta_2": float(c.betas[1]),
                "score": float(c.score),
                # Keep backwards compatible names expected by run_replica
                "Pred_Score_Quality": float(c.predicted[0]),
                "Pred_Score_Cost": float(c.predicted[1]),
                "hyperparameters": dict(c.params),
                "success": bool(c.success),
                "message": str(c.message),
            }
        )
    return pd.DataFrame(rows)


def save_nbi_candidates(candidates: list[NBICandidate], path: str) -> None:
    """Persist candidates to CSV (pt-BR friendly).

    We intentionally store both a flat view (predicted scores, betas) and the
    `hyperparameters` dict to keep compatibility with older consumers.
    """
    df = nbi_candidates_to_df(candidates)
    save_csv_ptbr(df, path)
