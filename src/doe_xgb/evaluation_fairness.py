from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


@dataclass(frozen=True)
class FairnessFoldMetrics:
    balanced_accuracy: float
    spd: float
    eod: float
    aod: float
    di: float
    bias_spd: float
    bias_eod: float
    bias_aod: float
    bias_di: float
    bias_mean: float
    fairness_score: float


def _safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a / (b if abs(b) > eps else eps))


def _group_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns:
      - positive prediction rate P(ŷ=1)
      - TPR = P(ŷ=1 | y=1)
      - FPR = P(ŷ=1 | y=0)
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    pos_rate = float(np.mean(y_pred == 1)) if y_pred.size else 0.0

    mask_pos = y_true == 1
    mask_neg = y_true == 0

    tpr = float(np.mean(y_pred[mask_pos] == 1)) if np.any(mask_pos) else 0.0
    fpr = float(np.mean(y_pred[mask_neg] == 1)) if np.any(mask_neg) else 0.0
    return pos_rate, tpr, fpr


def fairness_metrics_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    privileged_value: int = 1,
) -> Dict[str, float]:
    """
    protected: 1 => privileged, 0 => unprivileged (or any binary)
    """
    protected = protected.astype(int)
    priv_mask = protected == privileged_value
    unpriv_mask = ~priv_mask

    # Rates by group
    pr_priv, tpr_priv, fpr_priv = _group_rates(y_true[priv_mask], y_pred[priv_mask])
    pr_unp, tpr_unp, fpr_unp = _group_rates(y_true[unpriv_mask], y_pred[unpriv_mask])

    spd = pr_unp - pr_priv
    eod = tpr_unp - tpr_priv
    aod = 0.5 * ((fpr_unp - fpr_priv) + (tpr_unp - tpr_priv))

    di = _safe_div(pr_unp, pr_priv)  # may be >1 or <1

    return {
        "SPD": float(spd),
        "EOD": float(eod),
        "AOD": float(aod),
        "DI": float(di),
        "PPR_priv": float(pr_priv),
        "PPR_unpriv": float(pr_unp),
        "TPR_priv": float(tpr_priv),
        "TPR_unpriv": float(tpr_unp),
        "FPR_priv": float(fpr_priv),
        "FPR_unpriv": float(fpr_unp),
    }


def aggregate_bias(
    spd: float,
    eod: float,
    aod: float,
    di: float,
) -> Dict[str, float]:
    bias_spd = abs(float(spd))
    bias_eod = abs(float(eod))
    bias_aod = abs(float(aod))

    # symmetric around 1
    di_abs = abs(float(di))
    di_sym = min(di_abs, 1.0 / di_abs) if di_abs > 0 else 0.0
    bias_di = abs(1.0 - di_sym)

    bias_mean = float((bias_spd + bias_eod + bias_aod + bias_di) / 4.0)
    fairness_score = float(np.clip(1.0 - bias_mean, 0.0, 1.0))

    return {
        "Bias_SPD": bias_spd,
        "Bias_EOD": bias_eod,
        "Bias_AOD": bias_aod,
        "Bias_DI": bias_di,
        "BiasMean": bias_mean,
        "FairnessScore": fairness_score,
    }


def evaluate_fold_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protected: np.ndarray,
    privileged_value: int = 1,
) -> FairnessFoldMetrics:
    ba = float(balanced_accuracy_score(y_true, y_pred))

    fm = fairness_metrics_binary(
        y_true=y_true,
        y_pred=y_pred,
        protected=protected,
        privileged_value=privileged_value,
    )
    bm = aggregate_bias(fm["SPD"], fm["EOD"], fm["AOD"], fm["DI"])

    return FairnessFoldMetrics(
        balanced_accuracy=ba,
        spd=float(fm["SPD"]),
        eod=float(fm["EOD"]),
        aod=float(fm["AOD"]),
        di=float(fm["DI"]),
        bias_spd=float(bm["Bias_SPD"]),
        bias_eod=float(bm["Bias_EOD"]),
        bias_aod=float(bm["Bias_AOD"]),
        bias_di=float(bm["Bias_DI"]),
        bias_mean=float(bm["BiasMean"]),
        fairness_score=float(bm["FairnessScore"]),
    )


def summarize_folds(rows: list[FairnessFoldMetrics]) -> Dict[str, float]:
    def _mean(attr: str) -> float:
        return float(np.mean([getattr(r, attr) for r in rows])) if rows else 0.0

    return {
        "BalancedAccuracy_Mean": _mean("balanced_accuracy"),
        "Bias_SPD_Mean": _mean("bias_spd"),
        "Bias_EOD_Mean": _mean("bias_eod"),
        "Bias_AOD_Mean": _mean("bias_aod"),
        "Bias_DI_Mean": _mean("bias_di"),
        "BiasMean_Mean": _mean("bias_mean"),
        "FairnessScore_1_minus_Bias": _mean("fairness_score"),
    }
