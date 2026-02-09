from __future__ import annotations

"""
Fairness-friendly wrapper for NBI.

Why this exists:
- In fairness mode we model "cost" as Score_Cost = -BiasMean, where BiasMean >= 0.
- Without constraints, the RSM can predict BiasMean < 0 (=> Score_Cost > 0), which is
  mathematically possible for the surrogate but nonsensical for the real metric.
- The core NBI implementation already supports constraining predicted scores to the
  observed ranges; we expose a convenience wrapper with a clearer parameter name.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from .config import DEFAULT_BOUNDS
from .nbi import NBICandidate, run_nbi_weighted_sum


def run_nbi_weighted_sum_clipped(
    model_quality: Tuple[Sequence[str], Sequence[float]],
    model_cost: Tuple[Sequence[str], Sequence[float]],
    *,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    observed_utopia: Optional[Tuple[float, float]] = None,
    observed_nadir: Optional[Tuple[float, float]] = None,
    beta_step: float = 0.05,
    seed: int = 42,
    n_starts: int = 10,
    clip_pred_range: bool = True,
    maxiter: int = 2000,
) -> List[NBICandidate]:
    """
    Wrapper around `run_nbi_weighted_sum` using `constrain_pred_range`
    to keep predicted scores inside the observed ranges.

    Parameters
    ----------
    clip_pred_range:
        True -> enforce inequality constraints on predicted (Score_Quality, Score_Cost)
        so they remain within [nadir, utopia]. This prevents "negative bias" artifacts.
        False -> run unconstrained NBI (not recommended for fairness).
    """
    return run_nbi_weighted_sum(
        model_quality,
        model_cost,
        bounds=bounds,
        observed_utopia=observed_utopia,
        observed_nadir=observed_nadir,
        beta_step=beta_step,
        seed=seed,
        n_starts=n_starts,
        constrain_pred_range=clip_pred_range,
        maxiter=maxiter,
    )
