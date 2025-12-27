from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    """Full quadratic RSM design matrix in **uncoded (actual) units**.

    Columns:
      - const (Intercept)
      - linear: p
      - squares: p*p
      - 2-way interactions: a*b
    """
    cols: Dict[str, np.ndarray] = {}

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
    X = sm.add_constant(X, has_constant="add")  # adds 'const'
    terms = ["Intercept"] + list(X.columns[1:])
    return X, terms


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
    """Hierarchy rule:
    - Intercept never removable
    - Main effect p is NOT removable if any remaining term involves p in an interaction or square.
    - Interaction/square terms are removable.
    """
    if _is_intercept(term):
        return False

    if _is_main_effect(term):
        p = term
        for t in active_terms:
            if t == "Intercept" or t == p:
                continue
            if "*" in t:
                if p in _term_vars(t):
                    return False
        return True

    return True


def fit_rsm_backward(
    df_factors: pd.DataFrame,
    y: pd.Series,
    *,
    response_name: str,
    alpha: float = 0.05,
) -> RSMModel:
    """Fit quadratic RSM using backward elimination (α) with hierarchy."""
    # Validate columns
    missing = [p for p in PARAM_NAMES if p not in df_factors.columns]
    if missing:
        raise KeyError(f"Missing factor columns for RSM: {missing}")

    X_full, terms_full = build_quadratic_design_matrix(df_factors)

    active_cols = list(X_full.columns)  # includes 'const'
    active_terms = terms_full.copy()

    while True:
        X = X_full[active_cols]
        model = sm.OLS(y.astype(float).to_numpy(), X).fit()

        # pvalues for each column
        pvals = model.pvalues.to_dict()

        candidates = []
        for col in active_cols:
            term = "Intercept" if col == "const" else col
            if term == "Intercept":
                continue
            pval = float(pvals.get(col, np.nan))
            if np.isnan(pval):
                pval = 1.0
            if pval > alpha and _is_removable(term, ["Intercept"] + ["Intercept" if c == "const" else c for c in active_cols if c != "const"]):
                candidates.append((pval, col, term))

        if not candidates:
            break

        # Remove the highest p-value removable term
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, drop_col, _drop_term = candidates[0]
        active_cols.remove(drop_col)

        if len(active_cols) <= 1:
            break

    # Final fit
    X_final = X_full[active_cols]
    final = sm.OLS(y.astype(float).to_numpy(), X_final).fit()

    terms = []
    coefs = []
    for col in X_final.columns:
        if col == "const":
            terms.append("Intercept")
        else:
            terms.append(col)
        coefs.append(float(final.params[col]))

    return RSMModel(
        response_name=response_name,
        terms=terms,
        coefs=coefs,
        alpha=alpha,
        r2=float(final.rsquared),
        r2_adj=float(final.rsquared_adj),
    )


def save_rsm_coefficients(model: RSMModel, path: str) -> None:
    df = pd.DataFrame({"Term": model.terms, "Coef": model.coefs})
    save_csv_ptbr(df, path)
