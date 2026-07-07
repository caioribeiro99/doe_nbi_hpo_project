"""Figures for the PCO213 Santander mixture-ensemble study.

Pure-matplotlib helpers (no extra dependencies): barycentric ternary
contours, predicted-vs-observed, coefficient stability plot, per-model
fold boxplots and the holdout comparison chart. All figure functions
save to a path and return it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mixens.scheffe import MixtureScheffeModel


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_model_fold_boxplots(
    fold_metrics: pd.DataFrame, path: str | Path, metric: str = "roc_auc"
) -> Path:
    """Boxplot of per-fold OOF metric for each base model (10 folds each)."""
    models = list(fold_metrics["model"].unique())
    data = [fold_metrics.loc[fold_metrics["model"] == m, metric].to_numpy() for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=models)
    ax.set_ylabel(metric)
    ax.set_xlabel("modelo-base")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


def _barycentric_xy(w3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,3) simplex points -> 2-D barycentric coordinates."""
    x = w3[:, 1] + 0.5 * w3[:, 2]
    y = (np.sqrt(3.0) / 2.0) * w3[:, 2]
    return x, y


def fig_ternary_contour(
    model: MixtureScheffeModel,
    face: tuple[int, int, int],
    path: str | Path,
    *,
    title: str = "",
    design_points: np.ndarray | None = None,
    resolution: int = 60,
) -> Path:
    """Contour of the metamodel restricted to a 3-component face of the simplex
    (remaining weights fixed at 0), drawn in barycentric coordinates."""
    M = len(model.component_names)
    grid = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            a = i / resolution
            b = j / resolution
            grid.append([a, b, 1.0 - a - b])
    w3 = np.asarray(grid)
    W = np.zeros((len(w3), M))
    for k, comp in enumerate(face):
        W[:, comp] = w3[:, k]
    z = model.predict_weights(W)
    x, y = _barycentric_xy(w3)

    fig, ax = plt.subplots(figsize=(6, 5.2))
    tcf = ax.tricontourf(x, y, z, levels=14, cmap="viridis")
    fig.colorbar(tcf, ax=ax, shrink=0.85)
    names = [model.component_names[c] for c in face]
    ax.text(-0.02, -0.04, names[0], ha="right", fontsize=11)
    ax.text(1.02, -0.04, names[1], ha="left", fontsize=11)
    ax.text(0.5, np.sqrt(3) / 2 + 0.03, names[2], ha="center", fontsize=11)
    if design_points is not None:
        sub = design_points[:, list(face)]
        mask = np.isclose(design_points.sum(axis=1) - sub.sum(axis=1), 0.0, atol=1e-9)
        if mask.any():
            px, py = _barycentric_xy(sub[mask])
            ax.plot(px, py, "o", color="white", mec="black", ms=6)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    return _save(fig, path)


def fig_pred_vs_obs(
    y_design: np.ndarray,
    pred_design: np.ndarray,
    y_val: np.ndarray,
    pred_val: np.ndarray,
    path: str | Path,
    *,
    metric_name: str = "log-loss",
) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y_design, pred_design, label="pontos do design (ajuste)", alpha=0.8)
    ax.scatter(y_val, pred_val, marker="^", label="pontos Dirichlet (validação)", alpha=0.8)
    lims = [
        min(y_design.min(), y_val.min(), pred_design.min(), pred_val.min()),
        max(y_design.max(), y_val.max(), pred_design.max(), pred_val.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel(f"{metric_name} observado (OOF)")
    ax.set_ylabel(f"{metric_name} predito (Scheffé)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def fig_coefficients(coef_table: pd.DataFrame, path: str | Path, *, title: str = "") -> Path:
    """Dot plot of Scheffé coefficients with min-max stability range across repeats."""
    t = coef_table.iloc[::-1]  # linear terms at the top
    ypos = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(t) + 1.2))
    ax.hlines(ypos, t["min"], t["max"], color="tab:blue", lw=3, alpha=0.5)
    ax.plot(t["mean"], ypos, "o", color="tab:blue")
    ax.axvline(0.0, color="k", lw=1, ls="--", alpha=0.6)
    ax.set_yticks(ypos, t["term"])
    ax.set_xlabel("coeficiente (média e faixa entre repetições)")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, path)


def fig_holdout_comparison(results: pd.DataFrame, path: str | Path) -> Path:
    """Grouped bars: holdout ROC-AUC (left axis) and log-loss (right axis)."""
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(results))
    width = 0.38
    ax1.bar(x - width / 2, results["roc_auc"], width, label="ROC-AUC", color="tab:blue")
    lo = max(0.5, results["roc_auc"].min() - 0.01)
    ax1.set_ylim(lo, results["roc_auc"].max() + 0.005)
    ax1.set_ylabel("ROC-AUC (holdout)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, results["log_loss"], width, label="log-loss", color="tab:orange")
    ax2.set_ylabel("log-loss (holdout)", color="tab:orange")
    ax1.set_xticks(x, results["method"], rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    return _save(fig, path)


__all__ = [
    "fig_coefficients",
    "fig_holdout_comparison",
    "fig_model_fold_boxplots",
    "fig_pred_vs_obs",
    "fig_ternary_contour",
]
