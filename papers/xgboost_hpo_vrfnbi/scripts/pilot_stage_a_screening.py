#!/usr/bin/env python
"""Pilot Stage A: screen the candidate dataset panel.

`protocol/dataset_selection.md` sets four measurements that a dataset must pass to
stay in the panel, and `protocol/EXPERIMENT_PROTOCOL.md` §12 makes Stage A of the
pilot responsible for taking them on *every* candidate. This script is Stage A.

Per dataset, on one partition:

  * evaluate the 88 design rows on the real objectives;
  * evaluate 100 held-out Latin-hypercube points, which are both the surrogate
    gate's external set and the screening's adequacy measurement;
  * run the decided factor stage of protocol §6.3;
  * take the four screening measurements and the measured per-evaluation cost.

Nothing here runs an arm. Stage A exists to decide the panel and to replace the
budget projection with a measurement.

Usage:
    python pilot_stage_a_screening.py [--datasets magic spambase adult bank_marketing]
                                      [--seed 20260913] [--n-valid 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
OUT = REPO / "papers" / "xgboost_hpo_vrfnbi" / "audits" / "pilot_stage_a"
DESIGN = REPO / "data" / "design" / "hyperparameter_design.csv"

PARAMS = ["subsample", "colsample_bytree", "colsample_bylevel",
          "learning_rate", "max_depth", "gamma", "n_estimators"]
INTS = {"max_depth", "n_estimators"}
BOUNDS = {"subsample": (0.05, 1.0), "colsample_bytree": (0.05, 1.0),
          "colsample_bylevel": (0.05, 1.0), "learning_rate": (0.01, 0.30),
          "max_depth": (3, 18), "gamma": (0.05, 5.0), "n_estimators": (50, 700)}

# Protocol sections 6.2 and 6.3. Each response declares its direction and its
# transform; neither is inferred. "minimize" says which way the canonicalization
# flips the sign, "transform" is applied before standardization.
#
# The cost response is log-transformed, as the dissertation transformed its own cost
# response (time_transform="log1p" is the frozen default). It is not cosmetic: leaf
# count spans three orders of magnitude over the design box, and a quadratic surface
# fitted to it raw reaches an in-sample R-squared of only 0.49 to 0.65, against 0.95
# on every dataset once transformed.
RESPONSES = {
    "Accuracy_Mean":    {"minimize": False, "role": "quality", "transform": "none"},
    "Precision_Mean":   {"minimize": False, "role": "quality", "transform": "none"},
    "Recall_Mean":      {"minimize": False, "role": "quality", "transform": "none"},
    "Specificity_Mean": {"minimize": False, "role": "quality", "transform": "none"},
    "RocAuc_Mean":      {"minimize": False, "role": "quality", "transform": "none"},
    "LogLoss_Mean":     {"minimize": True,  "role": "quality", "transform": "none"},
    "Leaves_Mean":      {"minimize": True,  "role": "cost",    "transform": "log1p"},
}


def apply_transforms(df: pd.DataFrame) -> np.ndarray:
    """Declared per-response transform, then the declared direction."""
    cols = []
    for c, spec in RESPONSES.items():
        v = df[c].to_numpy(dtype=float)
        if spec["transform"] == "log1p":
            v = np.log1p(np.clip(v, 0.0, None))
        elif spec["transform"] != "none":
            raise ValueError(f"unknown transform {spec['transform']!r} for {c}")
        cols.append(v * (1.0 if spec["minimize"] else -1.0))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- data

def prepare(dataset_id: str):
    from doe_xgb.datasets.loaders import load
    d = load(dataset_id)
    X = d.X.copy()
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if cat:
        X = pd.get_dummies(X, columns=cat, dummy_na=False)
    return X.astype(float).to_numpy(), np.asarray(d.y, dtype=int), len(cat)


# ------------------------------------------------------------------- evaluation

def evaluate(p: dict, X, y, kf, seed: int) -> dict:
    """One configuration, five folds, every response of protocol section 6.2."""
    acc, pre, rec, spe, auc, ll, leaves, times = ([] for _ in range(8))
    for tr, va in kf.split(X, y):
        m = XGBClassifier(**p, eval_metric="logloss", verbosity=0,
                          tree_method="hist", n_jobs=8, random_state=seed)
        t0 = time.perf_counter()
        m.fit(X[tr], y[tr])
        prob = m.predict_proba(X[va])[:, 1]
        times.append(time.perf_counter() - t0)
        pred = (prob >= 0.5).astype(int)
        tn = int(((pred == 0) & (y[va] == 0)).sum())
        fp = int(((pred == 1) & (y[va] == 0)).sum())
        acc.append(accuracy_score(y[va], pred))
        pre.append(precision_score(y[va], pred, zero_division=0))
        rec.append(recall_score(y[va], pred, zero_division=0))
        spe.append(tn / (tn + fp) if (tn + fp) else 0.0)
        auc.append(roc_auc_score(y[va], prob))
        ll.append(log_loss(y[va], np.clip(prob, 1e-7, 1 - 1e-7)))
        leaves.append(sum(s.count("leaf=") for s in m.get_booster().get_dump()))
    return {"Accuracy_Mean": float(np.mean(acc)), "Precision_Mean": float(np.mean(pre)),
            "Recall_Mean": float(np.mean(rec)), "Specificity_Mean": float(np.mean(spe)),
            "RocAuc_Mean": float(np.mean(auc)), "LogLoss_Mean": float(np.mean(ll)),
            "Leaves_Mean": float(np.mean(leaves)), "Time_MeanFold": float(np.mean(times))}


def cast(row) -> dict:
    return {k: (int(round(float(row[k]))) if k in INTS else float(row[k])) for k in PARAMS}


def external_points(n: int, seed: int, kind: str = "spanning") -> pd.DataFrame:
    """The surrogate gate's external set.

    ``kind="uniform"`` is a plain Latin hypercube over the box. It is the obvious
    choice and it is the wrong one here: in seven dimensions, uniform sampling puts
    essentially no mass near the box corners, so every drawn configuration is a decent
    one and the external set carries 2 to 10 times less response spread than the design
    it is meant to validate. An R-squared computed against it is dominated by a
    near-zero denominator and says nothing about the surface.

    ``kind="complement"`` is the complementary half fraction of the design's own
    factorial, plus axial runs at half the design's axial distance. It is the only
    construction here that reproduces the design's response spread, because that
    spread comes from specific corner *combinations* which no independent sampling
    scheme reaches in seven dimensions.

    ``kind="spanning"`` mixes half a Latin hypercube with half an arcsine-marginal
    sample, whose Beta(0.5, 0.5) coordinates concentrate near the ends of each range.
    The result spans the response range the design spans, which is the condition an
    external set has to meet before R-squared against it means anything.
    """
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    if kind == "complement":
        # The 88-run design is a face-centred central composite: a 64-run half
        # fraction of the 2^7 factorial defined by the generator "product of all
        # seven signs = +1", plus 14 axial and 10 centre runs. Its complementary
        # half fraction -- the 64 corners whose sign product is -1 -- is disjoint
        # from it by construction, has identical corner structure and therefore
        # identical response spread, and is itself a resolution-VII design.
        #
        # Corners alone cannot test curvature: on a two-level set every squared
        # coordinate equals one, so the quadratic terms collapse into the
        # intercept. Axial runs at half the design's axial distance are added so
        # the surface is also validated away from its own axial points.
        import itertools
        corners = np.array([s for s in itertools.product([-1.0, 1.0], repeat=len(PARAMS))
                            if np.prod(s) < 0], dtype=float)
        axial = np.zeros((2 * len(PARAMS), len(PARAMS)))
        for i in range(len(PARAMS)):
            axial[2 * i, i] = -0.5
            axial[2 * i + 1, i] = +0.5
        coded = np.vstack([corners, axial])
        if n < len(coded):
            rng = np.random.default_rng(seed)
            coded = coded[rng.choice(len(coded), size=n, replace=False)]
        u = (coded + 1.0) / 2.0
    elif kind == "uniform":
        u = qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n)
    elif kind == "spanning":
        n_lhs = n // 2
        rng = np.random.default_rng(seed)
        u = np.vstack([
            qmc.LatinHypercube(d=len(PARAMS), seed=seed).random(n_lhs),
            rng.beta(0.5, 0.5, size=(n - n_lhs, len(PARAMS))),
        ])
    else:
        raise ValueError(f"unknown external-set kind {kind!r}")
    return pd.DataFrame(qmc.scale(u, lo, hi), columns=PARAMS)


def lhs_points(n: int, seed: int) -> pd.DataFrame:      # retained for the tests
    return external_points(n, seed, kind="uniform")


# ---------------------------------------------------------------- factor stage

def varimax(L: np.ndarray, tol=1e-7, it=200):
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(it):
        d_old = d
        Lam = L @ R
        u, s, vh = np.linalg.svd(L.T @ (Lam**3 - Lam @ np.diag((Lam**2).sum(0)) / p))
        R = u @ vh
        d = float(s.sum())
        if d_old and d / d_old < 1 + tol:
            break
    return L @ R, R


# ---------------------------------------------------------------------------
# SUPERSEDED. This class carries BOTH factor-algebra defects the protocol has
# since withdrawn, and it is retained ONLY so Stage A's committed output remains
# reproducible from the code that produced it.
#
#   * it rotates RAW component scores, whose variances are the eigenvalues,
#     instead of standardized ones, which yields correlated "orthogonal" factors
#     (measured 0.379 to 0.698) and loadings that do not describe the scores in use;
#   * it weights the quality axes by UNROTATED eigenvalue shares indexed by
#     ROTATED component, which on Adult gave 0.872/0.128 where the rotated shares
#     are 0.547/0.453.
#
# The corrected implementation is doe_xgb.campaign.factor_model. Nothing that
# produces current evidence may use this class; the design-geometry helpers in this
# module (PARAMS, BOUNDS, INTS, external_points) are unaffected and are still used.
# tests/methodology/test_no_superseded_factor_algebra.py enforces this.
# ---------------------------------------------------------------------------
class FactorModel:
    """The protocol section 6.3 factor stage, fitted once and then applied.

    Fitting separately on two sets of runs and comparing the results is meaningless:
    each fit has its own standardization, its own rotation and its own sign
    orientation, so the two composites live in different coordinate systems. The
    method only ever sees the design, so the model is fitted there and *applied* to
    held-out points, which is also what the surrogate gate requires.
    """

    def __init__(self, k: int = 3) -> None:
        self.k = k
        self.cols = list(RESPONSES)

    def fit(self, df: pd.DataFrame) -> "FactorModel":
        M = apply_transforms(df)                                 # transform, then canonicalize
        self.mu_, self.sd_ = M.mean(0), M.std(0, ddof=1)
        self.sd_ = np.where(self.sd_ == 0, 1.0, self.sd_)
        Z = (M - self.mu_) / self.sd_
        pca = PCA(n_components=self.k, random_state=0).fit(Z)
        self.pca_ = pca
        lam = pca.explained_variance_
        loadings = pca.components_.T * np.sqrt(lam)               # scaled, not eigenvectors
        self.rot_, self.R_ = varimax(loadings)

        # A principal component's sign is arbitrary and Varimax does not fix it, so an
        # orientation rule is required or the quality composite's sign -- and with it
        # the measured objective conflict -- comes out either way.
        #
        # Orienting by each factor's single largest loading is the obvious rule and it
        # is wrong here. Specificity anti-correlates with accuracy, recall, the area
        # under the curve and log loss, because they trade off across the decision
        # threshold, and on three of the four candidate datasets specificity is the
        # dominant loading on the leading quality factor. That rule therefore pointed
        # the quality composite at specificity-badness, and reported the objectives as
        # agreeing on datasets where the raw metrics plainly show them trading off.
        #
        # Orient instead by the mean loading over the responses in the factor's own
        # role block, which is the rule the dissertation used. Every response is
        # canonicalized to minimization, so after orientation a larger score means a
        # worse configuration on a quality factor and a more expensive one on the cost
        # factor.
        cost_col = self.cols.index("Leaves_Mean")
        q_rows = [i for i, c in enumerate(self.cols) if RESPONSES[c]["role"] == "quality"]
        self.cost_idx_ = int(np.argmax(np.abs(self.rot_[cost_col])))   # sign-independent
        self.q_idx_ = [j for j in range(self.k) if j != self.cost_idx_]
        flip = np.ones(self.k)
        flip[self.cost_idx_] = np.sign(self.rot_[cost_col, self.cost_idx_]) or 1.0
        for j in self.q_idx_:
            flip[j] = np.sign(self.rot_[q_rows, j].mean()) or 1.0
        self.flip_ = flip
        self.rot_ = self.rot_ * flip
        self.R_ = self.R_ * flip
        scores = pca.transform(Z) @ self.R_
        self.score_mu_, self.score_sd_ = scores.mean(0), scores.std(0, ddof=1)
        self.score_sd_ = np.where(self.score_sd_ == 0, 1.0, self.score_sd_)
        share = lam / lam.sum()
        self.share_ = share
        self.w_ = share[self.q_idx_] / share[self.q_idx_].sum()   # variance weighting
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        M = apply_transforms(df)
        Z = (M - self.mu_) / self.sd_
        scores = self.pca_.transform(Z) @ self.R_
        zs = (scores - self.score_mu_) / self.score_sd_
        return {"quality": zs[:, self.q_idx_] @ self.w_,
                "quality_equal": zs[:, self.q_idx_].mean(1),   # sensitivity declared in advance
                "cost": zs[:, self.cost_idx_]}

    def summary(self) -> dict:
        return {"loadings": pd.DataFrame(self.rot_, index=self.cols,
                                         columns=[f"F{j+1}" for j in range(self.k)]),
                "explained_variance_share": self.share_.tolist(),
                "cost_factor": self.cost_idx_ + 1,
                "quality_weights": self.w_.tolist()}


def factor_stage(df: pd.DataFrame, k: int = 3) -> dict:
    """Fit and transform on the same frame. Only for tests and for the design set."""
    m = FactorModel(k).fit(df)
    return {**m.transform(df), **m.summary()}


# -------------------------------------------------------------------- screening

def raw_conflict(df: pd.DataFrame) -> float:
    """Objective conflict measured without the factor stage.

    The factor stage involves an extraction, a rotation and an orientation, any of
    which can invert the composite. This computes the same quantity from the
    canonicalized responses directly -- an unweighted mean of the standardized
    quality responses against the standardized cost response -- so that the factor
    stage has something independent to be checked against.
    """
    M = apply_transforms(df)
    cols = list(RESPONSES)
    q = [i for i, c in enumerate(cols) if RESPONSES[c]["role"] == "quality"]
    z = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1) for i in q])
    return float(spearmanr(z.mean(1), M[:, cols.index("Leaves_Mean")]).statistic)


def nondominated(q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Indices minimizing both canonicalized objectives."""
    pts = np.column_stack([q, c])
    keep = []
    for i, a in enumerate(pts):
        if not np.any(np.all(pts <= a, axis=1) & np.any(pts < a, axis=1)):
            keep.append(i)
    return np.array(keep)


def front_curvature(q: np.ndarray, c: np.ndarray) -> float:
    """How far the front departs from the chord between its extremes, normalized.

    Zero means a straight front, where weighted sum and NBI agree by construction.
    """
    idx = nondominated(q, c)
    if len(idx) < 3:
        return 0.0
    P = np.column_stack([q[idx], c[idx]])
    rng_ = np.ptp(P, axis=0)     # numpy 2 removed ndarray.ptp
    P = (P - P.min(0)) / np.where(rng_ == 0, 1.0, rng_)
    P = P[np.argsort(P[:, 0])]
    a, b = P[0], P[-1]
    d = b - a
    n = np.linalg.norm(d)
    if n == 0:
        return 0.0
    # perpendicular distance from the chord; written out because numpy 2 deprecated
    # the 2-D form of np.cross
    rel = P - a
    dev = np.abs(d[0] * rel[:, 1] - d[1] * rel[:, 0]) / n
    return float(dev.max())


def _code(d: pd.DataFrame) -> np.ndarray:
    """Coded units, protocol section 7: each factor mapped to [-1, 1] over its box."""
    lo = np.array([BOUNDS[p][0] for p in PARAMS])
    hi = np.array([BOUNDS[p][1] for p in PARAMS])
    return 2 * (d[PARAMS].to_numpy(dtype=float) - lo) / (hi - lo) - 1


def _terms() -> list[tuple[int, ...]]:
    """Full quadratic basis as index tuples: (), (i,), (i,i), (i,j)."""
    k = len(PARAMS)
    out: list[tuple[int, ...]] = [()]
    out += [(i,) for i in range(k)]
    out += [(i, i) for i in range(k)]
    out += [(i, j) for i in range(k) for j in range(i + 1, k)]
    return out


def _basis(Xc: np.ndarray, terms) -> np.ndarray:
    cols = []
    for tm in terms:
        v = np.ones(len(Xc))
        for i in tm:
            v = v * Xc[:, i]
        cols.append(v)
    return np.column_stack(cols)


def fit_surface_backward(fit_df: pd.DataFrame, y: np.ndarray, alpha: float = 0.05):
    """Quadratic response surface in coded units, backward elimination, hierarchy kept.

    The protocol inherits the dissertation's surrogate, which is backward-eliminated
    at alpha = 0.05 with hierarchy enforced, not a full quadratic. The distinction
    matters for external behaviour: a 36-term quadratic fitted on 88 design points is
    far more prone to blowing up away from them than the 12-to-18-term model the
    elimination actually produces.
    """
    Xc = _code(fit_df)
    terms = _terms()
    y = np.asarray(y, dtype=float)
    while True:
        A = _basis(Xc, terms)
        n, p_ = A.shape
        if n - p_ <= 1 or len(terms) <= 1:
            break
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = n - p_
        s2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * s2, 1e-300))
        from scipy import stats as _st
        pvals = 2 * (1 - _st.t.cdf(np.abs(beta) / se, dof))
        # Hierarchy: a main effect stays while any term containing it stays.
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
    A = _basis(Xc, terms)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return terms, beta


def external_scores(fit_df, fit_y, val_df, val_y, alpha: float = 0.05):
    """External R-squared and rank correlation of the protocol's surrogate."""
    terms, beta = fit_surface_backward(fit_df, fit_y, alpha)
    pred = _basis(_code(val_df), terms) @ beta
    yv = np.asarray(val_y, dtype=float)
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return r2, float(spearmanr(pred, yv).statistic), len(terms)


# -------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["magic", "spambase", "adult", "bank_marketing"])
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--n-valid", type=int, default=100)
    ap.add_argument("--external-set", choices=["complement", "spanning", "uniform"],
                    default="complement",
                    help="how the surrogate gate's held-out set is drawn")
    ap.add_argument("--reuse", action="store_true",
                    help="recompute the screening from cached evaluations")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    design = pd.read_csv(DESIGN, sep=";", decimal=",", encoding="utf-8-sig")
    design.columns = [str(c).strip().strip('"') for c in design.columns]
    n_valid = args.n_valid if args.external_set != "complement" else 10**6
    valid = external_points(n_valid, args.seed, kind=args.external_set)

    # record the realized size, not the requested one: the complementary construction
    # has a fixed size and the argparse default previously leaked into the artifact
    report: dict = {"seed": args.seed, "n_design": int(len(design)),
                    "n_valid": int(len(valid)), "external_set_kind": args.external_set,
                    "datasets": {}}

    for ds in args.datasets:
        t0 = time.perf_counter()
        X, y, n_cat = prepare(ds)
        # one outer partition; the holdout is untouched here and reserved for the campaign
        Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=args.seed)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

        # Both external-set constructions are kept on disk. The comparison between
        # them is itself a pilot finding, so one must not overwrite the other.
        d_path = OUT / f"{ds}_design.csv"
        v_path = OUT / f"{ds}_validation_{args.external_set}.csv"
        # The design and the external set are cached independently, so changing the
        # external-set construction does not force the design to be re-evaluated.
        spent = 0
        if args.reuse and d_path.exists():
            d_df = pd.read_csv(d_path)
        else:
            d_df = pd.concat([design[PARAMS].reset_index(drop=True),
                              pd.DataFrame([evaluate(cast(design.iloc[i]), Xtr, ytr,
                                                     kf, args.seed)
                                            for i in range(len(design))])], axis=1)
            d_df.to_csv(d_path, index=False)
            spent += len(d_df)
        if args.reuse and v_path.exists():
            v_df = pd.read_csv(v_path)
        else:
            v_df = pd.concat([valid.reset_index(drop=True),
                              pd.DataFrame([evaluate(cast(valid.iloc[i]), Xtr, ytr,
                                                     kf, args.seed)
                                            for i in range(len(valid))])], axis=1)
            v_df.to_csv(v_path, index=False)
            spent += len(v_df)
        elapsed = (time.perf_counter() - t0) if spent else float("nan")
        n_eval = len(d_df) + len(v_df)
        per_eval = elapsed / spent if spent else float("nan")

        model = FactorModel().fit(d_df)          # the method only ever sees the design
        fs = model.transform(d_df)
        fs_v = model.transform(v_df)             # applied, not refitted
        summ = model.summary()
        summ["loadings"].to_csv(OUT / f"{ds}_loadings.csv")

        rho_conf = float(spearmanr(fs["quality"], fs["cost"]).statistic)
        rho_raw = raw_conflict(d_df)
        if np.sign(rho_conf) != np.sign(rho_raw):
            raise SystemExit(
                f"{ds}: the factor stage reports objective conflict {rho_conf:+.3f} while "
                f"the responses themselves give {rho_raw:+.3f}. A sign disagreement means "
                "the quality composite is inverted relative to the metrics it is built "
                "from, and no screening number below it can be trusted. Fix the factor "
                "orientation rather than the threshold.")
        curv = front_curvature(fs["quality"], fs["cost"])
        r2q, sq, nq = external_scores(d_df, fs["quality"], v_df, fs_v["quality"])
        r2c, sc, nc = external_scores(d_df, fs["cost"], v_df, fs_v["cost"])
        cost_ratio = float(d_df.Leaves_Mean.max() / max(d_df.Leaves_Mean.min(), 1.0))
        rho_weighting = float(spearmanr(fs["quality"], fs["quality_equal"]).statistic)

        report["datasets"][ds] = {
            "rows": int(X.shape[0]), "columns_after_encoding": int(X.shape[1]),
            "categorical_columns_one_hot_encoded": n_cat,
            "prevalence": float(y.mean()),
            "screening": {
                "objective_conflict_spearman": round(rho_conf, 4),
                "objective_conflict_from_raw_responses": round(rho_raw, 4),
                "front_curvature": round(curv, 4),
                "external_r2_quality": round(r2q, 4),
                "external_spearman_quality": round(sq, 4),
                "external_r2_cost": round(r2c, 4),
                "external_spearman_cost": round(sc, 4),
                "surface_terms_quality": nq,
                "surface_terms_cost": nc,
                "gate_pass_quality": bool(r2q >= 0.5 and sq >= 0.9),
                "gate_pass_cost": bool(r2c >= 0.5 and sc >= 0.9),
                "cost_range_ratio": round(cost_ratio, 1),
                "external_set_kind": args.external_set,
                "quality_sd_design": round(float(np.std(fs["quality"], ddof=1)), 4),
                "quality_sd_external": round(float(np.std(fs_v["quality"], ddof=1)), 4),
                "quality_spread_ratio_design_over_external": round(
                    float(np.std(fs["quality"], ddof=1) / max(np.std(fs_v["quality"], ddof=1), 1e-12)), 2),
            },
            "factor_stage": {
                "explained_variance_share": [round(v, 4) for v in summ["explained_variance_share"]],
                "cost_factor": summ["cost_factor"],
                "quality_weights_variance": [round(v, 4) for v in summ["quality_weights"]],
                "spearman_variance_vs_equal_weighting": round(rho_weighting, 4),
            },
            "cost": {
                "evaluations": n_eval,
                "seconds_total": None if np.isnan(elapsed) else round(elapsed, 1),
                "seconds_per_evaluation": None if np.isnan(per_eval) else round(per_eval, 3),
                "evaluations_performed_this_run": int(spent),
            },
        }
        timing = ("cached" if np.isnan(elapsed)
                  else f"{elapsed/60:5.1f} min ({per_eval:.2f} s/eval, {spent} new)")
        print(f"{ds:16s} {n_eval} evals, {timing} | "
              f"conflict {rho_conf:+.3f} curv {curv:.3f} "
              f"R2q {r2q:+.3f} SpRq {sq:+.3f} ({nq} terms) "
              f"R2c {r2c:+.3f} SpRc {sc:+.3f} ({nc} terms) costratio {cost_ratio:.0f}")

    (OUT / "stage_a_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT/'stage_a_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
