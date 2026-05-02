"""Coded <-> uncoded factor coordinate transforms.

Coded units map a factor's [lo, hi] interval to [-1, +1] linearly, which
is the conventional RSM coordinate system. Working in coded units gives
orthogonal designs, comparable effect sizes, and better numerical
conditioning.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _broadcast_bounds(lo: Sequence[float], hi: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    if lo_arr.shape != hi_arr.shape:
        raise ValueError("lo and hi must have the same shape")
    if np.any(lo_arr >= hi_arr):
        raise ValueError("each lo must be strictly less than hi")
    return lo_arr, hi_arr


def encode(values: np.ndarray, lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """Map values in [lo, hi] to coded units in [-1, +1]."""
    lo_arr, hi_arr = _broadcast_bounds(lo, hi)
    arr = np.asarray(values, dtype=float)
    return 2.0 * (arr - lo_arr) / (hi_arr - lo_arr) - 1.0


def decode(coded: np.ndarray, lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    """Map coded units in [-1, +1] back to natural [lo, hi] units."""
    lo_arr, hi_arr = _broadcast_bounds(lo, hi)
    arr = np.asarray(coded, dtype=float)
    return lo_arr + (arr + 1.0) * 0.5 * (hi_arr - lo_arr)


__all__ = ["encode", "decode"]
