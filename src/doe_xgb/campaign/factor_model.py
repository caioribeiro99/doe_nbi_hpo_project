"""The frozen factor stage, fitted on the design side and applied everywhere else.

Implements ``papers/xgboost_hpo_vrfnbi/protocol/OBJECTIVE_DEFINITIONS.md`` exactly:
declared per-response transform and direction, standardization, principal
components with a fixed ``k``, Varimax on the *scaled* loadings, deterministic role
assignment and sign orientation by the role-block mean, and a quality composite
weighted by explained-variance share.

The fit/apply split is the point. PCA and Varimax are fitted once per replication
from the 88 design rows and then frozen; the external validation set, every arm's
revalidated candidates, the baselines, the reference sets and the holdout are all
*transformed* by that frozen mapping. Refitting anywhere else would mean the
objective is not the same variable across the points a replication compares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .evaluator import RESPONSES

N_COMPONENTS = 3          # fixed, protocol-level; see OBJECTIVE_DEFINITIONS.md section 8


def varimax(L: np.ndarray, tol: float = 1e-7, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation of a loading matrix. Returns rotated loadings and R."""
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lam = L @ R
        u, s, vh = np.linalg.svd(L.T @ (Lam ** 3 - Lam @ np.diag((Lam ** 2).sum(0)) / p))
        R = u @ vh
        d = float(s.sum())
        if d_old and d / d_old < 1 + tol:
            break
    return L @ R, R


def apply_transforms(df: pd.DataFrame) -> np.ndarray:
    """Declared per-response transform, then the declared direction."""
    cols = []
    for c, spec in RESPONSES.items():
        v = np.asarray(df[c], dtype=float)
        if spec["transform"] == "log1p":
            v = np.log1p(np.clip(v, 0.0, None))
        elif spec["transform"] != "none":
            raise ValueError(f"unknown transform {spec['transform']!r} for {c}")
        cols.append(v * (1.0 if spec["minimize"] else -1.0))
    return np.column_stack(cols)


@dataclass
class FrozenFactorModel:
    """A fitted factor stage. Everything needed to reproduce it is persisted."""

    mu: np.ndarray
    sd: np.ndarray
    components: np.ndarray          # PCA components_, shape (k, p)
    eigenvalues: np.ndarray
    rotation: np.ndarray            # R, sign-corrected
    rotated_loadings: np.ndarray    # scaled and oriented
    score_mu: np.ndarray
    score_sd: np.ndarray
    cost_index: int
    quality_indices: tuple[int, ...]
    quality_weights: np.ndarray
    response_names: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # -------------------------------------------------------------------- apply

    def transform(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        Z = (apply_transforms(df) - self.mu) / self.sd
        # Standardize the component scores BEFORE rotating. Rotating raw component
        # scores, whose variances are the eigenvalues, mixes axes of unequal scale
        # and produces CORRELATED factors: measured max off-diagonal correlation on
        # the panel was 0.379 to 0.698. It also means the rotated loadings do not
        # describe the scores actually used, so the role assignment and the sign
        # orientation would be read off the wrong matrix.
        scores = ((Z @ self.components.T) / np.sqrt(self.eigenvalues)) @ self.rotation
        zs = (scores - self.score_mu) / self.score_sd
        q = zs[:, list(self.quality_indices)]
        return {"quality": q @ self.quality_weights,
                "quality_equal": q.mean(axis=1),
                "cost": zs[:, self.cost_index],
                "factor_scores": zs}

    def objectives(self, df: pd.DataFrame) -> np.ndarray:
        """The two frozen objectives, both minimized, as an (n, 2) array."""
        t = self.transform(df)
        return np.column_stack([t["quality"], t["cost"]])

    def as_dict(self) -> dict[str, Any]:
        return {"response_names": list(self.response_names),
                "standardization_mean": self.mu.tolist(),
                "standardization_sd": self.sd.tolist(),
                "pca_components": self.components.tolist(),
                "eigenvalues": self.eigenvalues.tolist(),
                "explained_variance_share_unrotated":
                    (self.eigenvalues / self.eigenvalues.sum()).tolist(),
                "explained_variance_share_rotated":
                    ((self.rotated_loadings ** 2).sum(axis=0)
                     / (self.rotated_loadings ** 2).sum()).tolist(),
                "varimax_rotation": self.rotation.tolist(),
                "rotated_loadings": self.rotated_loadings.tolist(),
                "score_standardization_mean": self.score_mu.tolist(),
                "score_standardization_sd": self.score_sd.tolist(),
                "cost_factor_index": int(self.cost_index),
                "quality_factor_indices": list(self.quality_indices),
                "quality_weights": self.quality_weights.tolist(),
                **self.diagnostics}


def fit_factor_model(design: pd.DataFrame, k: int = N_COMPONENTS) -> FrozenFactorModel:
    """Fit on the design side. Never call this on anything else."""
    names = tuple(RESPONSES)
    M = apply_transforms(design)
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    sd = np.where(sd == 0.0, 1.0, sd)
    Z = (M - mu) / sd

    pca = PCA(n_components=k, random_state=0).fit(Z)
    lam = pca.explained_variance_
    loadings = pca.components_.T * np.sqrt(lam)      # scaled, not eigenvectors
    rot, R = varimax(loadings)

    cost_row = names.index("Leaves_Mean")
    cost_idx = int(np.argmax(np.abs(rot[cost_row])))          # orientation-independent
    q_idx = tuple(j for j in range(k) if j != cost_idx)
    q_rows = [i for i, c in enumerate(names) if RESPONSES[c]["role"] == "quality"]

    flip = np.ones(k)
    flip[cost_idx] = np.sign(rot[cost_row, cost_idx]) or 1.0
    for j in q_idx:
        flip[j] = np.sign(rot[q_rows, j].mean()) or 1.0
    rot = rot * flip
    R = R * flip

    scores = ((Z @ pca.components_.T) / np.sqrt(lam)) @ R
    score_mu = scores.mean(axis=0)
    score_sd = scores.std(axis=0, ddof=1)
    score_sd = np.where(score_sd == 0.0, 1.0, score_sd)

    # A component's share of explained variance AFTER an orthogonal rotation is the
    # sum of its squared rotated loadings, not its unrotated eigenvalue. Varimax
    # redistributes variance across components, so indexing the unrotated
    # eigenvalues by the rotated component index pairs two unrelated things: on one
    # panel dataset that gave 0.872/0.128 where the rotated shares are 0.547/0.453.
    rotated_ss = (rot ** 2).sum(axis=0)
    share = rotated_ss / rotated_ss.sum()
    w = np.asarray([share[j] for j in q_idx], dtype=float)
    w = w / w.sum()

    # Descriptive diagnostics, reported and never acted on.
    full = PCA(n_components=min(len(names), len(design)), random_state=0).fit(Z)
    lam_full = full.explained_variance_
    max_off_diagonal_score_correlation = float(
        np.abs(np.corrcoef(scores, rowvar=False) - np.eye(k)).max()) if k > 1 else 0.0
    diagnostics = {
        "n_components_fixed": int(k),
        "max_off_diagonal_score_correlation": max_off_diagonal_score_correlation,
        "rotated_ss_loading_share": (rotated_ss / rotated_ss.sum()).tolist(),
        "kaiser_retained": int((lam_full > 1.0).sum()),
        "eigenvalues_full": lam_full.tolist(),
        "lambda3_over_lambda4": (float(lam_full[2] / lam_full[3])
                                 if len(lam_full) > 3 and lam_full[3] > 0 else None),
        "dominant_response_per_factor": [names[int(np.argmax(np.abs(rot[:, j])))]
                                         for j in range(k)],
        "sign_flips_applied": flip.tolist(),
        "note": ("k is a fixed protocol-level latent representation inherited from the "
                 "NBI-VRF construction, not a per-dataset criterion-selected "
                 "dimensionality; the Kaiser count is descriptive only"),
    }
    return FrozenFactorModel(
        mu=mu, sd=sd, components=pca.components_, eigenvalues=lam, rotation=R,
        rotated_loadings=rot, score_mu=score_mu, score_sd=score_sd,
        cost_index=cost_idx, quality_indices=q_idx, quality_weights=w,
        response_names=names, diagnostics=diagnostics)


def raw_conflict(df: pd.DataFrame) -> float:
    """Objective conflict measured without the factor stage, as an independent check."""
    from scipy.stats import spearmanr
    M = apply_transforms(df)
    names = list(RESPONSES)
    q = [i for i, c in enumerate(names) if RESPONSES[c]["role"] == "quality"]
    z = np.column_stack([(M[:, i] - M[:, i].mean()) / M[:, i].std(ddof=1) for i in q])
    return float(spearmanr(z.mean(axis=1), M[:, names.index("Leaves_Mean")]).statistic)


__all__ = ["FrozenFactorModel", "fit_factor_model", "apply_transforms", "varimax",
           "raw_conflict", "N_COMPONENTS"]
