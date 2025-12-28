from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence, Optional, Any, Union, cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import DEFAULT_BOUNDS, PARAM_NAMES, INT_PARAMS
from .io_utils import save_csv_ptbr


Number = Union[int, float]


def evaluate_term(x: Dict[str, Number], term: str, eps: float = 1e-12) -> float:
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


def predict_from_coeffs(x: Dict[str, Number], terms: Sequence[str], coefs: Sequence[float]) -> float:
    return float(sum(float(c) * evaluate_term(x, str(t)) for t, c in zip(terms, coefs)))


def _find_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """Find a column in df by trying candidates (case-insensitive exact, then contains match)."""
    cols: List[str] = [str(c) for c in df.columns]
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


def load_coefficients_csv(path: str) -> Tuple[List[str], List[float]]:
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

    terms: List[str] = [str(v) for v in term_series.tolist()]
    coefs: List[float] = [float(v) for v in coef_series.tolist()]

    return terms, coefs


def _cast_int_params(params: Dict[str, Number]) -> Dict[str, Union[int, float]]:
    out: Dict[str, Union[int, float]] = {}
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
    params: Dict[str, Union[int, float]]
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
    (t1, c1) = model1
    (t2, c2) = model2

    rng = np.random.default_rng(seed)
    bounds_list = [bounds[p] for p in PARAM_NAMES]
    centers = [(lo + hi) / 2.0 for (lo, hi) in bounds_list]

    # 20 betas: 0.05..1.00
    b_values = np.arange(beta_step, 1.0 + 1e-9, beta_step)
    betas_grid = [(round(1.0 - float(b), 2), round(float(b), 2)) for b in b_values]

    x0_list: List[np.ndarray] = [np.array(centers, dtype=float)]
    for _ in range(max(0, n_starts - 1)):
        x0 = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds_list], dtype=float)
        x0_list.append(x0)

    def preds_and_norm(x_vec: np.ndarray, nadir: np.ndarray, utopia: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        params = dict(zip(PARAM_NAMES, x_vec.tolist()))
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
        params_dict = dict(zip(PARAM_NAMES, params_vec.tolist()))
        pred1 = predict_from_coeffs(params_dict, t1, c1)
        pred2 = predict_from_coeffs(params_dict, t2, c2)

        candidates.append(
            NBICandidate(
                betas=betas,
                score=float(best_score),
                predicted=(float(pred1), float(pred2)),
                params=_cast_int_params(cast(Dict[str, Number], params_dict)),
                success=bool(best_res.success),
                message=str(best_res.message),
            )
        )

    return candidates


def nbi_candidates_to_df(candidates: List[NBICandidate]) -> pd.DataFrame:
    """Convert a list of :class:`NBICandidate` to a flat pandas DataFrame.

    The returned schema is used by the pipeline and by `load_nbi_candidates`:

    - beta_1, beta_2: the (beta1, beta2) weights
    - score: the scalarized objective used during optimization
    - Pred_Score_Quality, Pred_Score_Cost: predicted objective scores
    - hyperparameters: dict with param values (ints already coerced)
    - success, message: optimizer status
    """

    rows: List[Dict[str, Any]] = []
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


def save_nbi_candidates(candidates: List[NBICandidate], path: str) -> None:
    """Persist candidates to CSV (pt-BR friendly).

    We intentionally store both a flat view (predicted scores, betas) and the
    `hyperparameters` dict to keep compatibility with older consumers.
    """
    df = nbi_candidates_to_df(candidates)
    save_csv_ptbr(df, path)
