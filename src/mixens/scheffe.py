"""Scheffé canonical mixture polynomial as an interpretable metamodel.

Ported from doe_nbi_hpo_project, branch ``repo-publication-readiness``
(commit 0465466), file ``src/doe_xgb/model_families.py`` (mixture parts
only: FitReport, Scheffé design matrix, MixtureScheffeModel). Original
author: Caio Tertuliano Ribeiro (MIT License). Adapted for the PCO213
final project: added external validation against held-out weight points
and coefficient aggregation across CV repeats. Backward elimination is
intentionally disabled for mixture models (the canonical interpretation
depends on the full term set), and the polynomial has NO intercept.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

ScheffeOrder = Literal["linear", "quadratic", "special_cubic", "cubic"]


@dataclass(frozen=True)
class FitReport:
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    r2: float
    r2_adj: float
    rank: int
    condition_number: float
    n_obs: int
    n_params: int
    notes: tuple[str, ...] = ()
    pvalues: dict[str, float] | None = None


def _build_scheffe_design_matrix(
    df: pd.DataFrame,
    *,
    component_names: Sequence[str],
    order: ScheffeOrder,
) -> tuple[pd.DataFrame, list[str]]:
    """Scheffé canonical polynomial in q components (sum to 1).

    Conventions (Cornell, 2002):
    - linear:        sum_i b_i x_i        (no intercept)
    - quadratic:     linear + sum_{i<j} b_{ij} x_i x_j
    - special_cubic: quadratic + sum_{i<j<k} b_{ijk} x_i x_j x_k
    - cubic:         special_cubic + sum_{i<j} d_{ij} x_i x_j (x_i - x_j)
    """
    cols: dict[str, np.ndarray] = {}
    q = len(component_names)
    for p in component_names:
        cols[p] = df[p].astype(float).to_numpy()
    if order in ("quadratic", "special_cubic", "cubic"):
        for i, j in combinations(range(q), 2):
            a, b = component_names[i], component_names[j]
            cols[f"{a}*{b}"] = df[a].astype(float).to_numpy() * df[b].astype(float).to_numpy()
    if order in ("special_cubic", "cubic"):
        for i, j, k in combinations(range(q), 3):
            a, b, c = component_names[i], component_names[j], component_names[k]
            cols[f"{a}*{b}*{c}"] = (
                df[a].astype(float).to_numpy()
                * df[b].astype(float).to_numpy()
                * df[c].astype(float).to_numpy()
            )
    if order == "cubic":
        for i, j in combinations(range(q), 2):
            a, b = component_names[i], component_names[j]
            cols[f"({a}-{b})*{a}*{b}"] = (
                df[a].astype(float).to_numpy()
                * df[b].astype(float).to_numpy()
                * (df[a].astype(float).to_numpy() - df[b].astype(float).to_numpy())
            )
    X = pd.DataFrame(cols, index=df.index)
    return X, list(X.columns)


def _evaluate_scheffe_term(term: str, x: dict[str, float]) -> float:
    """Evaluate a Scheffé canonical term, including the cubic ``(a-b)*a*b`` form."""
    t = term.strip()
    if t.startswith("(") and ")" in t:
        head, _, rest = t.partition(")")
        head = head[1:]
        a, _, b = head.partition("-")
        rest_parts = [p for p in rest.split("*") if p]
        val = float(x[a]) - float(x[b])
        for p in rest_parts:
            val *= float(x[p])
        return val
    parts = t.split("*")
    val = 1.0
    for p in parts:
        val *= float(x[p])
    return val


@dataclass
class MixtureScheffeModel:
    component_names: tuple[str, ...]
    terms: tuple[str, ...]
    coefficients: tuple[float, ...]
    fit_report: FitReport

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        component_names: Sequence[str],
        order: ScheffeOrder = "quadratic",
    ) -> MixtureScheffeModel:
        X, terms = _build_scheffe_design_matrix(df, component_names=component_names, order=order)
        y_arr = y.astype(float).to_numpy()
        # NOTE: backward elimination is intentionally disabled for Scheffé.
        fit = sm.OLS(y_arr, X).fit()
        coefs = [float(fit.params[c]) for c in X.columns]
        cond_number = float(np.linalg.cond(X.to_numpy()))
        rank = int(np.linalg.matrix_rank(X.to_numpy()))
        report = FitReport(
            terms=tuple(terms),
            coefficients=tuple(coefs),
            r2=float(fit.rsquared),
            r2_adj=float(fit.rsquared_adj),
            rank=rank,
            condition_number=cond_number,
            n_obs=int(X.shape[0]),
            n_params=int(X.shape[1]),
            notes=("Scheffé canonical polynomial; no intercept; backward elimination disabled.",),
            pvalues={str(t): float(fit.pvalues[c]) for c, t in zip(X.columns, terms, strict=True)},
        )
        return cls(
            component_names=tuple(component_names),
            terms=tuple(terms),
            coefficients=tuple(coefs),
            fit_report=report,
        )

    def predict_row(self, x: dict[str, float]) -> float:
        v = 0.0
        for term, coef in zip(self.terms, self.coefficients, strict=True):
            v += float(coef) * _evaluate_scheffe_term(term, x)
        return v

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X), dtype=float)
        rows = X.to_dict(orient="records")
        for i, row in enumerate(rows):
            out[i] = self.predict_row(row)  # type: ignore[arg-type]
        return out

    def predict_weights(self, weights: np.ndarray) -> np.ndarray:
        """Predict from an (N, q) weight array (PCO213 convenience)."""
        df = pd.DataFrame(np.asarray(weights, dtype=float), columns=list(self.component_names))
        return self.predict(df)


# ---------------------------------------------------------------------------
# Additions for the PCO213 study (not in the ported original)
# ---------------------------------------------------------------------------


def external_validation(
    model: MixtureScheffeModel,
    weights_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, float]:
    """Validate the metamodel on held-out weight points (e.g. Dirichlet).

    Reports RMSE both raw and relative to the observed response range on
    the validation set — on flat surfaces external R² alone is ill-posed,
    so the relative RMSE is the primary adequacy measure.
    """
    y_val = np.asarray(y_val, dtype=float)
    pred = model.predict_weights(weights_val)
    resid = y_val - pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    y_range = float(np.max(y_val) - np.min(y_val))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_val - y_val.mean()) ** 2))
    r2_ext = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "rmse": rmse,
        "rmse_relative_to_range": rmse / y_range if y_range > 0 else float("inf"),
        "r2_external": r2_ext,
        "n_val": int(len(y_val)),
        "response_range": y_range,
    }


def summarize_coefficients(models: Sequence[MixtureScheffeModel]) -> pd.DataFrame:
    """Aggregate coefficients across CV repeats.

    Returns one row per Scheffé term with mean, min and max across the
    fitted repeats. With few repeats these are *stability* ranges under
    CV re-partitioning — NOT sampling confidence intervals (the response
    is deterministic given the OOF matrix; see the report's methodology).
    """
    if not models:
        raise ValueError("need at least one fitted model")
    terms = models[0].terms
    for m in models:
        if m.terms != terms:
            raise ValueError("all models must share the same term set")
    coefs = np.array([m.coefficients for m in models], dtype=float)
    return pd.DataFrame(
        {
            "term": list(terms),
            "mean": coefs.mean(axis=0),
            "min": coefs.min(axis=0),
            "max": coefs.max(axis=0),
        }
    )


__all__ = [
    "ScheffeOrder",
    "FitReport",
    "MixtureScheffeModel",
    "external_validation",
    "summarize_coefficients",
]
