from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import PARAM_NAMES
from .io_utils import save_csv_ptbr


@dataclass(frozen=True)
class RSMModel:
    response_name: str
    terms: List[str]
    coefs: List[float]
    alpha: float
    r2: float
    r2_adj: float


def build_quadratic_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full quadratic RSM design matrix in **uncoded (actual) units**.

    Includes:
      - Intercept (const)
      - linear: p
      - squares: p*p
      - 2-way interactions: a*b

    Returns
    -------
    X : pd.DataFrame
        Design matrix with an explicit 'const' column.
    terms : List[str]
        Terms aligned with coefficients export.
    """
    cols: Dict[str, Any] = {}

    # linear
    for p in PARAM_NAMES:
        cols[p] = df[p].astype(float).to_numpy()

    # squares
    for p in PARAM_NAMES:
        cols[f"{p}*{p}"] = (df[p].astype(float) ** 2).to_numpy()

    # interactions
    for i in range(len(PARAM_NAMES)):
        for j in range(i + 1, len(PARAM_NAMES)):
            a, b = PARAM_NAMES[i], PARAM_NAMES[j]
            cols[f"{a}*{b}"] = (df[a].astype(float) * df[b].astype(float)).to_numpy()

    X = pd.DataFrame(cols, index=df.index)

    # Add constant. statsmodels may return ndarray depending on input;
    # enforce DataFrame to keep .columns stable and satisfy type checkers.
    X_const = sm.add_constant(X, has_constant="add")
    if not isinstance(X_const, pd.DataFrame):
        X_const = pd.DataFrame(X_const, index=df.index, columns=["const"] + list(X.columns))

    terms = ["Intercept"] + [c for c in X.columns]
    return X_const, terms


def _as_dataframe(df_factors: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Pylance sometimes infers df[col_list] as Series[Any] in callers.
    To make the API resilient, accept Series and convert to DataFrame.

    - If Series name exists -> use it as single column.
    - If Series has no name -> use 'x' as column name.
    """
    if isinstance(df_factors, pd.DataFrame):
        return df_factors

    # Series -> DataFrame
    s = cast(pd.Series, df_factors)
    col_name = str(s.name) if s.name is not None else "x"
    return s.to_frame(name=col_name)


def _is_intercept(term: str) -> bool:
    return term == "Intercept"


def _is_main_effect(term: str) -> bool:
    return "*" not in term and term != "Intercept"


def _term_vars(term: str) -> List[str]:
    if term == "Intercept":
        return []
    if "*" in term:
        return term.split("*")
    return [term]


def _is_removable(term: str, active_terms: List[str]) -> bool:
    """
    Hierarchy rule:
      - Intercept never removable
      - Main effect p is NOT removable if any remaining term involves p
        in an interaction or square (e.g. p*q or p*p).
      - Interaction/square terms are removable.
    """
    if _is_intercept(term):
        return False

    if _is_main_effect(term):
        p = term
        for t in active_terms:
            if t in {"Intercept", p}:
                continue
            if "*" in t:
                if p in _term_vars(t):
                    return False
        return True

    return True


def fit_rsm_backward(
    df_factors: Union[pd.DataFrame, pd.Series],
    y: pd.Series,
    *,
    response_name: str,
    alpha: float = 0.05,
) -> RSMModel:
    """
    Fit quadratic RSM using backward elimination (α) with hierarchy.

    Notes:
    - Uses OLS on the full quadratic matrix, then removes the highest p-value term
      above alpha that is removable under hierarchy.
    - Stops when no removable term has p-value > alpha (or only intercept remains).

    Robustness:
    - Accepts DataFrame or Series for df_factors. Series will be converted to DataFrame.
      This prevents noisy Pylance warnings in callers.
    """
    df_factors_df = _as_dataframe(df_factors)

    # Validate columns: must contain all PARAM_NAMES (for full quadratic)
    missing = [p for p in PARAM_NAMES if p not in df_factors_df.columns]
    if missing:
        raise KeyError(f"Missing factor columns for RSM: {missing}")

    X_full, _terms_full = build_quadratic_design_matrix(df_factors_df)

    active_cols: List[str] = list(X_full.columns)

    def cols_to_terms(cols: List[str]) -> List[str]:
        return ["Intercept" if c == "const" else c for c in cols]

    while True:
        X = X_full.loc[:, active_cols]
        model = sm.OLS(y.astype(float).to_numpy(), X).fit()

        pvals = model.pvalues.to_dict()
        active_terms = cols_to_terms(active_cols)

        candidates: List[Tuple[float, str, str]] = []
        for col in active_cols:
            term = "Intercept" if col == "const" else col
            if term == "Intercept":
                continue

            pval = pvals.get(col, np.nan)
            pval_f = float(pval) if pval is not None and not np.isnan(pval) else 1.0

            if pval_f > alpha and _is_removable(term, active_terms):
                candidates.append((pval_f, col, term))

        if not candidates:
            break

        candidates.sort(reverse=True, key=lambda x: x[0])
        _, drop_col, _ = candidates[0]
        active_cols.remove(drop_col)

        if len(active_cols) <= 1:
            break

    X_final = X_full.loc[:, active_cols]
    final = sm.OLS(y.astype(float).to_numpy(), X_final).fit()

    terms: List[str] = []
    coefs: List[float] = []
    for col in X_final.columns:
        terms.append("Intercept" if col == "const" else col)
        coefs.append(float(final.params[col]))

    return RSMModel(
        response_name=response_name,
        terms=terms,
        coefs=coefs,
        alpha=float(alpha),
        r2=float(final.rsquared),
        r2_adj=float(final.rsquared_adj),
    )


def save_rsm_coefficients(model: RSMModel, path: str) -> None:
    df = pd.DataFrame({"Term": model.terms, "Coef": model.coefs})
    save_csv_ptbr(df, path)
