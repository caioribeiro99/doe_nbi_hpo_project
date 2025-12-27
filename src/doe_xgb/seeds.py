from __future__ import annotations

from typing import List
import numpy as np


def generate_seeds(initial_seed: int = 42, n_replicas: int = 30) -> List[int]:
    """Generate a deterministic list of replica seeds.

    The first seed is `initial_seed`, and the remaining `n_replicas-1` seeds
    are generated via NumPy's default RNG.

    Default number of replicas (30) follows the common statistical practice
    for stabilizing mean estimates across stochastic pipelines.
    """
    if n_replicas < 1:
        raise ValueError("n_replicas must be >= 1")

    rng = np.random.default_rng(initial_seed)
    extra = rng.integers(0, 2**32 - 1, size=n_replicas - 1, dtype=np.uint32)
    return [int(initial_seed)] + [int(x) for x in extra]
