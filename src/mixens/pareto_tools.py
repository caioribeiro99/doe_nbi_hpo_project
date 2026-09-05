"""Pareto-front utilities for large point sets and front-quality indicators.

Minimization convention everywhere (objectives in an (N, q) array).
Complements ``mixens.selection`` (which keeps the small-N O(N^2) filter and
GD/IGD/spacing used by the delivered post-work) with:

- ``fast_pareto_mask``: lexicographic sweep, O(N * |front|), fine for 1e5+ points;
- ``igd_plus``: IGD+ (Ishibuchi et al., 2015);
- ``hypervolume``: exact 2-D / 3-D hypervolume (sweep algorithm);
- ``coverage``: C-metric (Zitzler & Thiele);
- ``extreme_point_recovery``: per-objective distance of a front's best value
  to the reference's best value;
- ``front_convergence``: displacement diagnostics of a sampled reference.
"""

from __future__ import annotations

import numpy as np


def fast_pareto_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimization). Duplicates kept.

    Rows are processed in lexicographic order; a row can only be dominated
    by rows that precede it, so each row is compared against the current
    front only.
    """
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("F must be 2-D")
    n, q = F.shape
    order = np.lexsort(F.T[::-1])  # sort by col 0, then col 1, ...
    Fs = F[order]
    keep_sorted = np.zeros(n, dtype=bool)
    cap = 256
    front = np.empty((cap, q))
    m = 0
    for i in range(n):
        f = Fs[i]
        if m:
            fr = front[:m]
            dom = np.all(fr <= f, axis=1) & np.any(fr < f, axis=1)
            if dom.any():
                continue
        keep_sorted[i] = True
        if m == cap:
            cap *= 2
            new_front = np.empty((cap, q)); new_front[:m] = front[:m]; front = new_front
        front[m] = f; m += 1
    mask = np.zeros(n, dtype=bool)
    mask[order[keep_sorted]] = True
    return mask


def normalize(F: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    span = np.asarray(hi, dtype=np.float64) - np.asarray(lo, dtype=np.float64)
    span = np.where(np.abs(span) < 1e-15, 1.0, span)
    return (F - np.asarray(lo, dtype=np.float64)) / span


def gd(front: np.ndarray, reference: np.ndarray) -> float:
    front = np.atleast_2d(front); reference = np.atleast_2d(reference)
    if front.shape[0] == 0:
        return float("nan")
    d = np.array([np.linalg.norm(reference - p, axis=1).min() for p in front])
    return float(d.mean())


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    return gd(reference, front)


def igd_plus(front: np.ndarray, reference: np.ndarray) -> float:
    """IGD+: for each reference point, distance to the nearest front point
    using only the components where the front point is WORSE (max(a-r, 0))."""
    front = np.atleast_2d(front); reference = np.atleast_2d(reference)
    if front.shape[0] == 0:
        return float("nan")
    d = np.empty(reference.shape[0])
    for i, r in enumerate(reference):
        diff = np.maximum(front - r, 0.0)
        d[i] = np.sqrt((diff ** 2).sum(axis=1)).min()
    return float(d.mean())


def spacing(front: np.ndarray) -> float:
    """Schott's spacing (std of L1 nearest-neighbour distances); nan if < 3 points."""
    front = np.atleast_2d(front)
    n = front.shape[0]
    if n < 3:
        return float("nan")
    d = np.empty(n)
    for i in range(n):
        diff = np.abs(front - front[i]).sum(axis=1); diff[i] = np.inf
        d[i] = diff.min()
    return float(np.std(d, ddof=1))


def spacing_cv(front: np.ndarray) -> float:
    """Coefficient of variation of nearest-neighbour distances (size-robust)."""
    front = np.atleast_2d(front)
    n = front.shape[0]
    if n < 3:
        return float("nan")
    d = np.empty(n)
    for i in range(n):
        diff = np.abs(front - front[i]).sum(axis=1); diff[i] = np.inf
        d[i] = diff.min()
    m = d.mean()
    return float(d.std(ddof=1) / m) if m > 0 else float("nan")


def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact hypervolume (minimization) for 2 or 3 objectives.

    Points not strictly better than ``ref_point`` in every coordinate are
    ignored. 3-D uses a sweep over the third objective with 2-D slices.
    """
    F = np.atleast_2d(np.asarray(front, dtype=np.float64))
    r = np.asarray(ref_point, dtype=np.float64)
    F = F[np.all(F < r, axis=1)]
    if F.shape[0] == 0:
        return 0.0
    F = F[fast_pareto_mask(F)]
    q = F.shape[1]
    if q == 2:
        return _hv2d(F, r)
    if q == 3:
        # sweep along objective 3 (ascending), maintaining the 2-D non-dominated
        # staircase incrementally; each slab contributes (2-D HV) x (slab height)
        order = np.argsort(F[:, 2], kind="stable")
        Fs = F[order]
        xs: list[float] = []; ys: list[float] = []   # staircase sorted by x asc (y desc)
        hv = 0.0
        for i in range(Fs.shape[0]):
            x, yv, z_lo = Fs[i]
            z_hi = Fs[i + 1, 2] if i + 1 < Fs.shape[0] else r[2]
            # insert (x, yv) unless dominated in 2-D
            dominated = any(xx <= x and yy <= yv for xx, yy in zip(xs, ys))
            if not dominated:
                keep = [(xx, yy) for xx, yy in zip(xs, ys) if not (x <= xx and yv <= yy)]
                keep.append((x, yv)); keep.sort()
                xs = [k[0] for k in keep]; ys = [k[1] for k in keep]
            if z_hi > z_lo:
                area = 0.0; prev_y = r[1]
                for xx, yy in zip(xs, ys):
                    if yy < prev_y:
                        area += (r[0] - xx) * (prev_y - yy); prev_y = yy
                hv += area * (z_hi - z_lo)
        return float(hv)
    raise ValueError("hypervolume implemented for 2 or 3 objectives only")


def _hv2d(F: np.ndarray, r: np.ndarray) -> float:
    if F.shape[0] == 0:
        return 0.0
    order = np.argsort(F[:, 0])
    Fs = F[order]
    hv = 0.0
    prev_y = r[1]
    for x, yv in Fs:
        if yv < prev_y:
            hv += (r[0] - x) * (prev_y - yv)
            prev_y = yv
    return float(hv)


def coverage(A: np.ndarray, B: np.ndarray) -> float:
    """C(A, B): fraction of points in B weakly dominated by some point of A."""
    A = np.atleast_2d(A); B = np.atleast_2d(B)
    if B.shape[0] == 0:
        return float("nan")
    cnt = 0
    for b in B:
        if np.any(np.all(A <= b, axis=1) & np.any(A < b, axis=1)):
            cnt += 1
    return float(cnt / B.shape[0])


def extreme_point_recovery(front: np.ndarray, reference: np.ndarray) -> list[float]:
    """Per objective: (front best - reference best), in the normalized units
    passed in (>= 0; 0 means the extreme is recovered)."""
    front = np.atleast_2d(front); reference = np.atleast_2d(reference)
    return [float(front[:, j].min() - reference[:, j].min()) for j in range(reference.shape[1])]


def joint_nondominated_fraction(candidates: np.ndarray, reference_all: np.ndarray) -> float:
    """Fraction of candidate rows that are non-dominated in candidates ∪ reference."""
    C = np.atleast_2d(candidates); R = np.atleast_2d(reference_all)
    if C.shape[0] == 0:
        return float("nan")
    both = np.vstack([R, C])
    m = fast_pareto_mask(both)
    return float(m[R.shape[0]:].mean())


def front_convergence(F_sorted_by_arrival: np.ndarray, fractions=(0.25, 0.5, 0.75, 1.0)) -> list[dict]:
    """Convergence diagnostics for a sampled reference: for growing prefixes
    of the sample, the front size and the fraction of the previous prefix's
    front that is displaced (dominated) by the larger prefix."""
    F = np.asarray(F_sorted_by_arrival, dtype=np.float64)
    n = F.shape[0]
    out = []
    prev_front = None
    for frac in fractions:
        k = max(int(round(frac * n)), 1)
        m = fast_pareto_mask(F[:k])
        front = F[:k][m]
        displaced = float("nan")
        if prev_front is not None and prev_front.shape[0]:
            both = np.vstack([prev_front, front])
            mm = fast_pareto_mask(both)
            displaced = float(1.0 - mm[: prev_front.shape[0]].mean())
        out.append({"fraction": frac, "n_points": int(k), "front_size": int(m.sum()),
                    "prev_front_displaced_fraction": displaced})
        prev_front = front
    return out


__all__ = [
    "coverage",
    "extreme_point_recovery",
    "fast_pareto_mask",
    "front_convergence",
    "gd",
    "hypervolume",
    "igd",
    "igd_plus",
    "joint_nondominated_fraction",
    "normalize",
    "spacing",
    "spacing_cv",
]
