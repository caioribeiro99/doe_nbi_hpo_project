"""The frozen design and the frozen external validation set.

Both are fixed constructions, not samples. Neither is regenerated in response to
anything observed, and neither depends on a seed.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluator import BOUNDS, INT_PARAMS, PARAMS

REPO = Path(__file__).resolve().parents[3]
DESIGN_CSV = REPO / "data" / "design" / "hyperparameter_design.csv"
EXTERNAL_AXIAL_RADIUS = 0.5          # half the design's axial distance


def _lo_hi() -> tuple[np.ndarray, np.ndarray]:
    return (np.array([BOUNDS[p][0] for p in PARAMS], dtype=float),
            np.array([BOUNDS[p][1] for p in PARAMS], dtype=float))


def to_coded(df: pd.DataFrame) -> np.ndarray:
    lo, hi = _lo_hi()
    return 2 * (df[PARAMS].to_numpy(dtype=float) - lo) / (hi - lo) - 1


def from_coded(C: np.ndarray) -> pd.DataFrame:
    lo, hi = _lo_hi()
    nat = lo + (np.asarray(C, dtype=float) + 1.0) * (hi - lo) / 2.0
    df = pd.DataFrame(nat, columns=PARAMS)
    for p in INT_PARAMS:
        df[p] = np.round(df[p]).astype(int)
    return df


def load_design() -> pd.DataFrame:
    """The version-controlled 88-run face-centred central composite design."""
    d = pd.read_csv(DESIGN_CSV, sep=";", decimal=",", encoding="utf-8-sig")
    d.columns = [str(c).strip().strip('"') for c in d.columns]
    out = d[PARAMS].copy()
    for p in INT_PARAMS:
        out[p] = np.round(out[p]).astype(int)
    out.insert(0, "run", [f"design_{i:02d}" for i in range(len(out))])
    return out


def external_validation_set() -> pd.DataFrame:
    """The frozen 78-point audit-only external set.

    The design's 64 factorial corners form the half fraction with defining relation
    "product of all seven signs = +1". This is the complementary half fraction --
    the 64 corners whose sign product is -1 -- plus 14 axial runs at half the
    design's axial distance.

    Corners alone cannot test curvature: on a two-level set every squared coordinate
    equals one, so the quadratic terms collapse into the intercept and the surface
    basis loses rank (29 of 36 terms). The axial runs restore full rank.

    Deterministic, seedless, and never regenerated in response to an observed
    diagnostic.
    """
    k = len(PARAMS)
    corners = np.array([s for s in itertools.product([-1.0, 1.0], repeat=k)
                        if np.prod(s) < 0], dtype=float)
    axial = np.zeros((2 * k, k), dtype=float)
    for i in range(k):
        axial[2 * i, i] = -EXTERNAL_AXIAL_RADIUS
        axial[2 * i + 1, i] = +EXTERNAL_AXIAL_RADIUS
    coded = np.vstack([corners, axial])
    out = from_coded(coded)
    out.insert(0, "run", [f"external_{i:02d}" for i in range(len(out))])
    return out


def surface_terms(k: int = len(PARAMS)) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    out += [(i,) for i in range(k)]
    out += [(i, i) for i in range(k)]
    out += [(i, j) for i in range(k) for j in range(i + 1, k)]
    return out


def surface_basis(C: np.ndarray, terms) -> np.ndarray:
    cols = []
    for tm in terms:
        v = np.ones(len(C))
        for i in tm:
            v = v * C[:, i]
        cols.append(v)
    return np.column_stack(cols)


def fit_surface_backward(design: pd.DataFrame, y: np.ndarray, alpha: float = 0.05):
    """Quadratic surface in coded units, backward elimination, hierarchy enforced.

    Fitted on the design rows alone. The external set never enters term selection:
    if it did, the external R-squared would be scoring a surface the external data
    helped choose.
    """
    from scipy import stats
    C = to_coded(design)
    terms = surface_terms()
    y = np.asarray(y, dtype=float)
    while True:
        A = surface_basis(C, terms)
        n, p = A.shape
        if n - p <= 1 or len(terms) <= 1:
            break
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = n - p
        s2 = float(resid @ resid) / dof
        se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T @ A)) * s2, 1e-300))
        pvals = 2 * (1 - stats.t.cdf(np.abs(beta) / se, dof))
        protected = {0}
        for idx, tm in enumerate(terms):
            if len(tm) == 2:
                for i in set(tm):
                    if (i,) in terms:
                        protected.add(terms.index((i,)))
        cand = [i for i in range(len(terms)) if i not in protected]
        if not cand:
            break
        worst = max(cand, key=lambda i: pvals[i])
        if pvals[worst] <= alpha:
            break
        terms = [tm for i, tm in enumerate(terms) if i != worst]
    A = surface_basis(C, terms)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return terms, beta


def surface_predict(terms, beta: np.ndarray, C: np.ndarray) -> np.ndarray:
    return surface_basis(np.atleast_2d(C), terms) @ beta


def make_surrogate(terms, beta: np.ndarray):
    """A callable on a single coded point, as the arms expect.

    Returns a genuine Python float. ``surface_basis`` always produces a 2-D design
    matrix, so the product is a length-1 array; numpy 2 refuses to coerce that to a
    scalar, and every arm calls this through ``float(...)``.
    """
    beta = np.asarray(beta, dtype=float)

    def f(x: np.ndarray) -> float:
        A = surface_basis(np.atleast_2d(np.asarray(x, dtype=float)), terms)
        return float((A @ beta)[0])
    return f


def external_scores(design: pd.DataFrame, y_design: np.ndarray,
                    external: pd.DataFrame, y_external: np.ndarray,
                    alpha: float = 0.05) -> dict:
    """External R-squared, RMSE and Spearman, plus the spread that makes R-squared readable."""
    from scipy.stats import spearmanr
    terms, beta = fit_surface_backward(design, y_design, alpha)
    pred = surface_predict(terms, beta, to_coded(external))
    yv = np.asarray(y_external, dtype=float)
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    return {"terms": int(len(terms)),
            "external_r2": (1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
            "external_rmse": float(np.sqrt(ss_res / len(yv))),
            "external_spearman": float(spearmanr(pred, yv).statistic),
            "sd_design": float(np.std(y_design, ddof=1)),
            "sd_external": float(np.std(yv, ddof=1)),
            "spread_ratio_design_over_external":
                float(np.std(y_design, ddof=1) / max(np.std(yv, ddof=1), 1e-12))}


def fit_surface_backward_uncoded(design: pd.DataFrame, y: np.ndarray,
                                 alpha: float = 0.05):
    """The dissertation's own fit: full quadratic in NATURAL units, not coded.

    ``docs/METHODOLOGY_DECISIONS.md`` D6 records that the dissertation's tables
    report coded coefficients while its code fits uncoded. HISTORICAL-WS reproduces
    the code, so it fits uncoded and its coefficients are evaluated at natural
    values by the frozen solver. The article-track arms fit coded, which is the
    amendment D6 records; the difference is a property of the historical method and
    is not repaired.

    Returns ``(term_names, coefficients)`` in the frozen solver's own term grammar:
    ``Intercept``, ``name``, ``name^2``, ``name*other``.
    """
    from scipy import stats
    X = design[PARAMS].to_numpy(dtype=float)
    k = len(PARAMS)
    idx = [()] + [(i,) for i in range(k)] + [(i, i) for i in range(k)]
    idx += [(i, j) for i in range(k) for j in range(i + 1, k)]
    y = np.asarray(y, dtype=float)

    def basis(terms):
        cols = []
        for tm in terms:
            v = np.ones(len(X))
            for i in tm:
                v = v * X[:, i]
            cols.append(v)
        return np.column_stack(cols)

    terms = list(idx)
    while True:
        A = basis(terms)
        n, p = A.shape
        if n - p <= 1 or len(terms) <= 1:
            break
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = n - p
        s2 = float(resid @ resid) / dof
        se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T @ A)) * s2, 1e-300))
        pvals = 2 * (1 - stats.t.cdf(np.abs(beta) / se, dof))
        protected = {0}
        for i_, tm in enumerate(terms):
            if len(tm) == 2:
                for v in set(tm):
                    if (v,) in terms:
                        protected.add(terms.index((v,)))
        cand = [i for i in range(len(terms)) if i not in protected]
        if not cand:
            break
        worst = max(cand, key=lambda i: pvals[i])
        if pvals[worst] <= alpha:
            break
        terms = [tm for i, tm in enumerate(terms) if i != worst]
    beta, *_ = np.linalg.lstsq(basis(terms), y, rcond=None)

    names = []
    for tm in terms:
        if not tm:
            names.append("Intercept")
        elif len(tm) == 1:
            names.append(PARAMS[tm[0]])
        elif tm[0] == tm[1]:
            names.append(f"{PARAMS[tm[0]]}^2")
        else:
            names.append(f"{PARAMS[tm[0]]}*{PARAMS[tm[1]]}")
    return names, [float(b) for b in beta]


GATE_R2, GATE_SPEARMAN = 0.5, 0.9


def gate_pass(scores: dict) -> bool:
    """Diagnostic only. A failure changes nothing about execution (protocol 7.1)."""
    return bool(scores["external_r2"] >= GATE_R2
                and scores["external_spearman"] >= GATE_SPEARMAN)


__all__ = ["fit_surface_backward_uncoded", "load_design", "external_validation_set", "to_coded", "from_coded",
           "surface_terms", "surface_basis", "fit_surface_backward", "surface_predict",
           "make_surrogate", "external_scores", "gate_pass", "GATE_R2", "GATE_SPEARMAN"]
