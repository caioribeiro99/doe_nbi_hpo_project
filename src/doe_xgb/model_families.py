"""Surrogate model families: design-aware response-surface models.

Provides three concrete families:

- :class:`ProcessQuadraticRSM`: full quadratic in process variables
  (intercept + linear + pure quadratics + 2FI). Optional backward
  elimination with strong hierarchy. Suitable for CCD/BB/factorial.
- :class:`MixtureScheffeModel`: Scheffé canonical mixture polynomial
  on the simplex sum(w) = 1 (no intercept; no pure quadratics; uses
  cross-products and special-cubic terms). Backward elimination is
  intentionally disabled for mixture models, since the canonical
  interpretation depends on the full term set.
- :class:`LinearOrTwoFI`: explicit linear or 2FI variant of the
  process-quadratic family, useful for low-resolution fractional
  factorials or screening designs.

A small registry function :func:`make_surrogate` returns a fitted
instance given a :class:`SurrogateSpec`; :func:`select_default_family`
recommends a default family + order from a :class:`DesignArtifact`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .design import DesignArtifact


# ---------------------------------------------------------------------------
# Specs and result types
# ---------------------------------------------------------------------------


ScheffeOrder = Literal["linear", "quadratic", "special_cubic", "cubic"]
ProcessOrder = Literal["linear", "two_factor_interaction", "quadratic"]
ModelFamilyName = Literal["process_quadratic", "mixture_scheffe", "linear_or_2fi"]


@dataclass(frozen=True)
class BackwardEliminationSpec:
    alpha: float = 0.05
    enforce_hierarchy: bool = True


@dataclass(frozen=True)
class SurrogateSpec:
    family: ModelFamilyName
    order: Union[ProcessOrder, ScheffeOrder] = "quadratic"
    coding: Literal["coded", "uncoded"] = "coded"
    backward_elimination: Optional[BackwardEliminationSpec] = BackwardEliminationSpec()


@dataclass(frozen=True)
class FitReport:
    terms: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    r2: float
    r2_adj: float
    rank: int
    condition_number: float
    n_obs: int
    n_params: int
    notes: Tuple[str, ...] = ()
    pvalues: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Internal helpers (term-name conventions)
# ---------------------------------------------------------------------------


def _term_vars(term: str) -> List[str]:
    if term in ("Intercept", "const", "1"):
        return []
    return term.split("*")


def _is_intercept(term: str) -> bool:
    return term in ("Intercept", "const", "1")


def _is_main_effect(term: str) -> bool:
    return ("*" not in term) and not _is_intercept(term)


def _is_removable_under_hierarchy(term: str, active_terms: Sequence[str]) -> bool:
    """Strong-hierarchy rule: a main effect cannot be removed while any
    higher-order term involves it.
    """
    if _is_intercept(term):
        return False
    if _is_main_effect(term):
        for t in active_terms:
            if t == term or _is_intercept(t):
                continue
            if "*" in t and term in _term_vars(t):
                return False
        return True
    return True


# ---------------------------------------------------------------------------
# Process-variable RSM
# ---------------------------------------------------------------------------


def _build_process_design_matrix(
    df: pd.DataFrame,
    *,
    factor_names: Sequence[str],
    order: ProcessOrder,
) -> Tuple[pd.DataFrame, List[str]]:
    cols: Dict[str, np.ndarray] = {}
    for p in factor_names:
        cols[p] = df[p].astype(float).to_numpy()
    if order in ("two_factor_interaction", "quadratic"):
        for i, j in combinations(range(len(factor_names)), 2):
            a, b = factor_names[i], factor_names[j]
            cols[f"{a}*{b}"] = df[a].astype(float).to_numpy() * df[b].astype(float).to_numpy()
    if order == "quadratic":
        for p in factor_names:
            cols[f"{p}*{p}"] = df[p].astype(float).to_numpy() ** 2
    X = pd.DataFrame(cols, index=df.index)
    X_const = sm.add_constant(X, has_constant="add")
    if not isinstance(X_const, pd.DataFrame):
        X_const = pd.DataFrame(X_const, index=df.index, columns=["const"] + list(X.columns))
    terms = ["Intercept"] + list(X.columns)
    return X_const, terms


def _backward_eliminate(
    X_full: pd.DataFrame,
    y: np.ndarray,
    alpha: float,
    enforce_hierarchy: bool,
) -> Tuple[List[str], "sm.regression.linear_model.RegressionResultsWrapper"]:
    active = list(X_full.columns)
    while True:
        X = X_full.loc[:, active]
        model = sm.OLS(y, X).fit()
        pvals = model.pvalues.to_dict()
        active_terms = ["Intercept" if c == "const" else c for c in active]
        candidates: List[Tuple[float, str]] = []
        for col in active:
            term = "Intercept" if col == "const" else col
            if _is_intercept(term):
                continue
            pval = float(pvals.get(col, 1.0))
            if pval > alpha:
                if enforce_hierarchy and not _is_removable_under_hierarchy(term, active_terms):
                    continue
                candidates.append((pval, col))
        if not candidates:
            break
        candidates.sort(reverse=True)
        active.remove(candidates[0][1])
        if len(active) <= 1:
            break
    return active, sm.OLS(y, X_full.loc[:, active]).fit()


@dataclass
class _LinearRSMBase:
    factor_names: Tuple[str, ...]
    terms: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    fit_report: FitReport

    def predict_row(self, x: Dict[str, float]) -> float:
        v = 0.0
        for term, coef in zip(self.terms, self.coefficients):
            v += float(coef) * _evaluate_term_value(term, x)
        return v

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X), dtype=float)
        rows = X.to_dict(orient="records")
        for i, row in enumerate(rows):
            out[i] = self.predict_row(row)  # type: ignore[arg-type]
        return out


def _evaluate_term_value(term: str, x: Dict[str, float]) -> float:
    if _is_intercept(term):
        return 1.0
    parts = term.split("*")
    val = 1.0
    for p in parts:
        if "^" in p:
            base, exp = p.split("^", 1)
            val *= float(x[base]) ** float(exp)
        else:
            val *= float(x[p])
    return val


class ProcessQuadraticRSM(_LinearRSMBase):
    """Full quadratic / 2FI / linear response surface in process variables."""

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        factor_names: Sequence[str],
        order: ProcessOrder = "quadratic",
        backward: Optional[BackwardEliminationSpec] = BackwardEliminationSpec(),
    ) -> "ProcessQuadraticRSM":
        X_full, _ = _build_process_design_matrix(df, factor_names=factor_names, order=order)
        y_arr = y.astype(float).to_numpy()
        notes: List[str] = []
        if backward is not None:
            active, fit = _backward_eliminate(
                X_full, y_arr, alpha=backward.alpha, enforce_hierarchy=backward.enforce_hierarchy
            )
            X_final = X_full.loc[:, active]
        else:
            X_final = X_full
            fit = sm.OLS(y_arr, X_final).fit()
        terms = ["Intercept" if c == "const" else c for c in X_final.columns]
        coefs = [float(fit.params[c]) for c in X_final.columns]
        cond_number = float(np.linalg.cond(X_final.to_numpy()))
        rank = int(np.linalg.matrix_rank(X_final.to_numpy()))
        report = FitReport(
            terms=tuple(terms),
            coefficients=tuple(coefs),
            r2=float(fit.rsquared),
            r2_adj=float(fit.rsquared_adj),
            rank=rank,
            condition_number=cond_number,
            n_obs=int(X_final.shape[0]),
            n_params=int(X_final.shape[1]),
            notes=tuple(notes),
            pvalues={str(t): float(fit.pvalues[c]) for c, t in zip(X_final.columns, terms)},
        )
        return cls(
            factor_names=tuple(factor_names),
            terms=tuple(terms),
            coefficients=tuple(coefs),
            fit_report=report,
        )


# ---------------------------------------------------------------------------
# Mixture Scheffé model
# ---------------------------------------------------------------------------


def _build_scheffe_design_matrix(
    df: pd.DataFrame,
    *,
    component_names: Sequence[str],
    order: ScheffeOrder,
) -> Tuple[pd.DataFrame, List[str]]:
    """Scheffé canonical polynomial in q components (sum to 1).

    Conventions (Cornell, 2002):
    - linear:        sum_i b_i x_i        (no intercept)
    - quadratic:     linear + sum_{i<j} b_{ij} x_i x_j
    - special_cubic: quadratic + sum_{i<j<k} b_{ijk} x_i x_j x_k
    - cubic:         special_cubic + sum_{i<j} d_{ij} x_i x_j (x_i - x_j)
    """
    cols: Dict[str, np.ndarray] = {}
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


def _evaluate_scheffe_term(term: str, x: Dict[str, float]) -> float:
    """Evaluate a Scheffé canonical term, including the cubic ``(a-b)*a*b`` form."""
    t = term.strip()
    if t.startswith("(") and ")" in t:
        # Pattern: "(a-b)*a*b"
        head, _, rest = t.partition(")")
        head = head[1:]  # strip leading "("
        a, _, b = head.partition("-")
        rest_parts = [p for p in rest.split("*") if p]
        val = (float(x[a]) - float(x[b]))
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
    component_names: Tuple[str, ...]
    terms: Tuple[str, ...]
    coefficients: Tuple[float, ...]
    fit_report: FitReport

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        y: pd.Series,
        *,
        component_names: Sequence[str],
        order: ScheffeOrder = "quadratic",
    ) -> "MixtureScheffeModel":
        X, terms = _build_scheffe_design_matrix(
            df, component_names=component_names, order=order
        )
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
            pvalues={str(t): float(fit.pvalues[c]) for c, t in zip(X.columns, terms)},
        )
        return cls(
            component_names=tuple(component_names),
            terms=tuple(terms),
            coefficients=tuple(coefs),
            fit_report=report,
        )

    def predict_row(self, x: Dict[str, float]) -> float:
        v = 0.0
        for term, coef in zip(self.terms, self.coefficients):
            v += float(coef) * _evaluate_scheffe_term(term, x)
        return v

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(len(X), dtype=float)
        rows = X.to_dict(orient="records")
        for i, row in enumerate(rows):
            out[i] = self.predict_row(row)  # type: ignore[arg-type]
        return out


# ---------------------------------------------------------------------------
# Registry / convenience
# ---------------------------------------------------------------------------


def make_surrogate(
    spec: SurrogateSpec,
    df: pd.DataFrame,
    y: pd.Series,
    *,
    factor_names: Sequence[str],
) -> Any:
    """Fit and return a surrogate of the requested family."""

    if spec.family == "process_quadratic":
        order = spec.order
        if order == "linear":
            order_p: ProcessOrder = "linear"
        elif order in ("two_factor_interaction", "2fi"):
            order_p = "two_factor_interaction"
        else:
            order_p = "quadratic"
        return ProcessQuadraticRSM.fit(
            df,
            y,
            factor_names=factor_names,
            order=order_p,
            backward=spec.backward_elimination,
        )
    if spec.family == "linear_or_2fi":
        order_p2: ProcessOrder = "two_factor_interaction" if spec.order in ("two_factor_interaction", "2fi") else "linear"
        return ProcessQuadraticRSM.fit(
            df,
            y,
            factor_names=factor_names,
            order=order_p2,
            backward=spec.backward_elimination,
        )
    if spec.family == "mixture_scheffe":
        order_s: ScheffeOrder = (
            spec.order
            if spec.order in ("linear", "quadratic", "special_cubic", "cubic")
            else "quadratic"
        )
        return MixtureScheffeModel.fit(
            df,
            y,
            component_names=factor_names,
            order=order_s,
        )
    raise ValueError(f"unsupported surrogate family: {spec.family}")


def select_default_family(artifact: DesignArtifact) -> SurrogateSpec:
    """Recommend a default surrogate family based on the design's metadata."""
    kind = str(artifact.metadata.get("kind", "")).lower()
    if kind in ("simplex_lattice", "simplex_centroid"):
        return SurrogateSpec(family="mixture_scheffe", order="quadratic", backward_elimination=None)
    if kind in ("lhs", "sobol"):
        return SurrogateSpec(family="process_quadratic", order="quadratic")
    if kind == "fractional_factorial":
        return SurrogateSpec(family="linear_or_2fi", order="two_factor_interaction")
    return SurrogateSpec(family="process_quadratic", order="quadratic")


__all__ = [
    "ProcessOrder",
    "ScheffeOrder",
    "BackwardEliminationSpec",
    "SurrogateSpec",
    "FitReport",
    "ProcessQuadraticRSM",
    "MixtureScheffeModel",
    "make_surrogate",
    "select_default_family",
]
