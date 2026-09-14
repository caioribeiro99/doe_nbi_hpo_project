"""Real-objective revalidation, the two references, and the indicators.

Everything a method emits is a *continuous surrogate solution*. Everything scored
is a *realized configuration evaluated on the real learner*. The two are kept apart
throughout, because an NBI subproblem is certified on the former and the learner
runs the latter.

Two references, both Pareto-filtered in real objective space, both used by every
method:

* the **core**: method-independent real evaluations only;
* the **common augmented reference**: the core together with the real-revalidated
  candidates of every compared method.

Neither is called a true Pareto front. The augmented union is self-graded by
construction and the degree of self-grading is quantified per method.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..reporting import dominance_filter, dominated_fraction, hypervolume, igd_plus
from .evaluator import INT_PARAMS, PARAMS

HV_REFERENCE_COORDINATE = 1.1      # per normalized axis, fixed before any front is seen


def canonical_key(cfg: dict[str, Any]) -> tuple:
    """Two emitted candidates that realize to the same learner are one configuration."""
    out = []
    for p in PARAMS:
        v = cfg[p]
        out.append(int(round(float(v))) if p in INT_PARAMS else round(float(v), 12))
    return tuple(out)


def revalidate(emitted: list[dict], view, objectives_of) -> dict[str, Any]:
    """Realize, deduplicate, evaluate on the real learner, and report the accounting.

    ``emitted`` carries each method's realized configuration. Every element is
    requested, so every element is charged; the cache serves the repeats, so the
    physical work is deduplicated and the logical budget is not.
    """
    # Every emitted candidate is REQUESTED, so every one is charged to the arm. The
    # cache serves the repeats, so the physical cost is unchanged. Deduplicating
    # before requesting, as an earlier version did, charged an arm the number of
    # UNIQUE realized configurations rather than the declared budget -- and the bias
    # was systematic, because a weighted sum's minimizers cluster at the anchors and
    # collapse under rounding far more often than NBI's spread subproblem solutions.
    # The arm whose budget was most understated was the geometry control in the
    # study's own primary contrast.
    results: dict[tuple, dict] = {}
    order: list[tuple] = []
    for c in emitted:
        key = canonical_key(c)
        cfg = {p: (int(v) if p in INT_PARAMS else float(v)) for p, v in zip(PARAMS, key)}
        res = view.evaluate(cfg)                 # charged every time, hit or miss
        if key not in results:
            results[key] = res
            order.append(key)
    rows, invalid = [], 0
    for key in order:
        cfg = {p: (int(v) if p in INT_PARAMS else float(v)) for p, v in zip(PARAMS, key)}
        res = results[key]
        if not all(np.isfinite(v) for v in res.values() if isinstance(v, (int, float))):
            invalid += 1
            continue
        rows.append({**cfg, **res})
    df = pd.DataFrame(rows)
    F = objectives_of(df) if len(df) else np.zeros((0, 2))
    finite = np.all(np.isfinite(F), axis=1) if len(F) else np.zeros(0, dtype=bool)
    invalid += int((~finite).sum())
    df, F = (df[finite].reset_index(drop=True), F[finite]) if len(F) else (df, F)
    nd = dominance_filter(F) if len(F) else np.zeros(0, dtype=int)
    return {"emitted": len(emitted),
            "unique_realizable": len(order),
            "real_valid": int(len(df)),
            "real_nondominated": int(len(nd)),
            "dominated_fraction_after_revalidation":
                float(dominated_fraction(F)) if len(F) else float("nan"),
            "rows": df, "objectives": F, "nondominated_index": nd}


def reference_core(frames: list[pd.DataFrame], objectives_of) -> dict[str, Any]:
    """Method-independent real evaluations only: the design and the anchor search.

    Nothing any compared method returned enters this set, so it is the reference a
    method cannot grade itself against.
    """
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if frames else pd.DataFrame()
    F = objectives_of(df) if len(df) else np.zeros((0, 2))
    nd = dominance_filter(F) if len(F) else np.zeros(0, dtype=int)
    return {"n_points": int(len(F)), "n_nondominated": int(len(nd)),
            "front": F[nd] if len(F) else F,
            "note": "method-independent; never called a true Pareto front"}


def augmented_reference(core_front: np.ndarray,
                        per_method: dict[str, np.ndarray]) -> dict[str, Any]:
    """The core together with every method's real-revalidated candidates.

    Every method is scored against this same set, so no method is advantaged by the
    union. Each contributes points to the front it is graded against, which is
    self-grading; the share each contributes is reported so the reader can see how
    much.
    """
    blocks = [core_front] + [v for v in per_method.values() if len(v)]
    allF = np.vstack([b for b in blocks if len(b)]) if blocks else np.zeros((0, 2))
    owner = (["core"] * len(core_front)
             + [m for m, v in per_method.items() for _ in range(len(v))])
    nd = dominance_filter(allF) if len(allF) else np.zeros(0, dtype=int)
    front_owner = [owner[i] for i in nd]
    share = {m: round(front_owner.count(m) / max(len(nd), 1), 4)
             for m in ["core", *per_method]}
    return {"n_points": int(len(allF)), "n_nondominated": int(len(nd)),
            "front": allF[nd] if len(allF) else allF,
            "self_grading_share_of_front": share,
            "note": ("self-graded by construction: a method contributing many front "
                     "points is partly graded against itself; never called a true "
                     "Pareto front")}


def normalize(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi - lo < 1e-12, 1.0, hi - lo)
    return (np.asarray(F, dtype=float) - lo) / span


def indicators(F: np.ndarray, reference_front: np.ndarray) -> dict[str, float]:
    """Primary: hypervolume ratio. Secondary and descriptive: everything else.

    Computed on the real-revalidated, dominance-filtered set of each method against
    a common reference. The normalization box is the reference front's own range, so
    every method is normalized identically.
    """
    if len(F) == 0 or len(reference_front) == 0:
        return {k: float("nan") for k in
                ("hv_ratio", "igd_plus", "gd", "spacing", "spacing_cv",
                 "joint_nondominated_fraction", "n_front")}
    lo = reference_front.min(axis=0)
    hi = reference_front.max(axis=0)
    S = normalize(F[dominance_filter(F)], lo, hi)
    R = normalize(reference_front, lo, hi)
    ref_pt = np.full(S.shape[1], HV_REFERENCE_COORDINATE)
    hv_ref = hypervolume(R, ref_pt)
    # generational distance and Schott spacing, descriptive
    gd = float(np.mean([np.min(np.linalg.norm(R - s, axis=1)) for s in S]))
    if len(S) > 1:
        dmin = [float(np.min(np.linalg.norm(np.delete(S, i, axis=0) - S[i], axis=1)))
                for i in range(len(S))]
        spacing = float(np.sqrt(np.mean((np.mean(dmin) - np.asarray(dmin)) ** 2)))
        spacing_cv = float(np.std(dmin, ddof=1) / max(np.mean(dmin), 1e-12))
    else:
        spacing = spacing_cv = float("nan")
    # dominance_filter returns INDICES, not a mask; build the mask explicitly so the
    # joint fraction counts the method's own points that survive against the reference
    joint = np.vstack([S, R])
    surviving = np.zeros(len(joint), dtype=bool)
    surviving[dominance_filter(joint)] = True
    return {"hv_ratio": float(hypervolume(S, ref_pt) / hv_ref) if hv_ref > 0 else float("nan"),
            "igd_plus": float(igd_plus(S, R)),
            "gd": gd, "spacing": spacing, "spacing_cv": spacing_cv,
            "joint_nondominated_fraction": float(np.mean(surviving[: len(S)])),
            "n_front": int(len(S))}


__all__ = ["revalidate", "reference_core", "augmented_reference", "indicators",
           "canonical_key", "normalize", "HV_REFERENCE_COORDINATE"]
