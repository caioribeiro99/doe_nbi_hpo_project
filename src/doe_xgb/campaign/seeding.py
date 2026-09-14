"""Deterministic, method-specific seed derivation.

The first protocol review found substantial random-stream aliasing: the grid's
padding points were bit-identical to random search's first draws, and the Bayesian
and Parzen initial designs to its first draws too, because every comparator
received the same replication seed and constructed its own generator from it. Five
baselines were substantially one baseline.

A seed here is a pure function of ``(dataset, replication, method, stage)``, hashed
with BLAKE2b and fed to :class:`numpy.random.SeedSequence`. The properties that
matter, each of which a test asserts:

* reproducible across resumes, machines and Python processes -- BLAKE2b is
  specified, unlike Python's ``hash()`` which is salted per process;
* independent of execution order and of worker count, because nothing is drawn
  from a shared mutable generator;
* changing one method cannot alter another method's stream.

The replication seed remains a component, so a paired design still pairs: every
method sees the same data split for a given replication.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Bumping this re-derives every stream. It is part of the frozen protocol.
SEED_NAMESPACE = "xgboost-hpo-vrfnbi/v3"


def seed_key(dataset: str, replication: int, method: str, stage: str = "main") -> str:
    return f"{SEED_NAMESPACE}|{dataset}|{int(replication)}|{method}|{stage}"


def derive_seed(dataset: str, replication: int, method: str,
                stage: str = "main") -> int:
    """A 64-bit seed determined entirely by its four arguments.

    BLAKE2b rather than ``hash()``: Python's string hashing is randomized per
    process unless ``PYTHONHASHSEED`` is set, so a campaign resumed in a new
    process would silently draw different candidates.
    """
    digest = hashlib.blake2b(seed_key(dataset, replication, method, stage).encode(),
                             digest_size=8).digest()
    return int.from_bytes(digest, "big")


# numpy's SeedSequence accepts arbitrarily wide entropy, but several third-party
# seed APIs do not: scikit-learn validates ``random_state`` against
# ``[0, 2**32 - 1]`` and rejects anything wider. A 64-bit derived seed handed
# straight to ``GaussianProcessRegressor`` therefore raises
# ``InvalidParameterError`` -- which is how this was found, in the pre-freeze smoke,
# at the Bayesian comparator.
UINT32 = 2 ** 32


def as_uint32(seed: int) -> int:
    """Narrow a derived seed for a library whose seed API is 32-bit.

    The low 32 bits of a BLAKE2b digest are uniformly distributed and independent
    across keys, so narrowing preserves the property that actually matters here:
    two different ``(dataset, replication, method, stage)`` tuples get unrelated
    streams. It does not preserve full entropy, and it is therefore used ONLY where
    a library refuses a wider value -- never for numpy generators, which take the
    full 64 bits.

    Narrowing is a many-to-one map, so distinctness is not guaranteed by
    construction the way it is at 64 bits. It is asserted instead, over the
    campaign's entire set of streams, by
    ``tests/methodology/test_seed_derivation.py``.
    """
    return int(seed) % UINT32


def generator(dataset: str, replication: int, method: str,
              stage: str = "main") -> np.random.Generator:
    """A fresh generator for one (dataset, replication, method, stage)."""
    return np.random.default_rng(
        np.random.SeedSequence(derive_seed(dataset, replication, method, stage)))


def candidate_hash(configs) -> str:
    """A stable digest of a method's candidate stream, for the seed ledger.

    Two methods producing the same digest drew the same candidates, which is the
    aliasing the review found. Reported per method per replication so the claim of
    independence is checkable rather than asserted.
    """
    h = hashlib.blake2b(digest_size=16)
    for cfg in configs:
        if isinstance(cfg, dict):
            item = "|".join(f"{k}={float(cfg[k]):.12g}" for k in sorted(cfg))
        else:
            item = "|".join(f"{float(v):.12g}" for v in np.ravel(cfg))
        h.update(item.encode())
        h.update(b";")
    return h.hexdigest()


__all__ = ["SEED_NAMESPACE", "UINT32", "seed_key", "derive_seed", "as_uint32",
           "generator", "candidate_hash"]
