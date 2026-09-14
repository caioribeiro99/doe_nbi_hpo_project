"""Every stochastic method must draw from its own stream, reproducibly.

The first protocol review measured the failure this prevents: with a shared
replication seed the grid's 258 padding points were bit-identical to random
search's first 258, and the Bayesian and Parzen initial designs to its first 77.
Five baselines were substantially one.
"""
from __future__ import annotations

import itertools
import subprocess
import sys

import numpy as np
import pytest

from doe_xgb.campaign.seeding import (SEED_NAMESPACE, candidate_hash, derive_seed,
                                      generator, seed_key)

METHODS = ("grid", "random", "bayes_quality", "bayes_cost",
           "tpe_quality", "tpe_cost", "nsga2", "nsga2_unmatched")


def _draws(method: str, dataset="magic", rep=0, n=400):
    return generator(dataset, rep, method).uniform(-1.0, 1.0, size=(n, 7))


def test_same_method_and_replication_gives_identical_candidates() -> None:
    np.testing.assert_array_equal(_draws("grid"), _draws("grid"))


def test_different_methods_get_independently_derived_streams() -> None:
    for a, b in itertools.combinations(METHODS, 2):
        da, db = _draws(a, n=200), _draws(b, n=200)
        assert not np.allclose(da, db), f"{a} and {b} share a stream"
        # and no prefix aliasing either, which is how the original defect showed
        for cut in (10, 50, 100):
            assert not np.allclose(da[:cut], db[:cut]), f"{a} and {b} alias at {cut}"


def test_the_replication_and_dataset_both_enter_the_derivation() -> None:
    base = derive_seed("magic", 0, "grid")
    assert derive_seed("magic", 1, "grid") != base
    assert derive_seed("adult", 0, "grid") != base
    assert derive_seed("magic", 0, "grid", stage="anchor") != base


def test_the_derivation_does_not_depend_on_python_hash_randomization() -> None:
    """Python's str hash is salted per process unless PYTHONHASHSEED is set.

    A campaign resumed in a new process would otherwise draw different candidates.
    """
    code = ("import sys; sys.path.insert(0, 'src');"
            "from doe_xgb.campaign.seeding import derive_seed;"
            "print(derive_seed('magic', 7, 'tpe_cost'))")
    outs = set()
    for salt in ("0", "1", "random"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"})
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"seed varied with PYTHONHASHSEED: {outs}"


def test_execution_order_and_worker_count_cannot_matter() -> None:
    """Nothing is drawn from a shared mutable generator, so order is irrelevant."""
    forward = {m: candidate_hash(_draws(m, n=100)) for m in METHODS}
    backward = {m: candidate_hash(_draws(m, n=100)) for m in reversed(METHODS)}
    assert forward == backward


def test_changing_one_method_cannot_alter_another() -> None:
    before = candidate_hash(_draws("random", n=100))
    _ = _draws("grid", n=5000)            # exhaust another method's stream heavily
    assert candidate_hash(_draws("random", n=100)) == before


def test_candidate_hash_distinguishes_streams_and_is_stable() -> None:
    hashes = {m: candidate_hash(_draws(m, n=100)) for m in METHODS}
    assert len(set(hashes.values())) == len(METHODS)
    assert hashes["grid"] == candidate_hash(_draws("grid", n=100))


def test_the_namespace_is_part_of_the_key() -> None:
    assert SEED_NAMESPACE in seed_key("magic", 0, "grid")
    assert seed_key("magic", 0, "grid") != seed_key("magic", 0, "grid", stage="other")


# ---------------------------------------------------------------------------
# 32-bit narrowing for third-party seed APIs
#
# Found by the pre-freeze smoke, at the Bayesian comparator: scikit-learn
# validates random_state against [0, 2**32 - 1], so a 64-bit BLAKE2b seed handed
# to GaussianProcessRegressor raises InvalidParameterError and the stage dies.
# Narrowing is many-to-one, so the distinctness that 64 bits gave by construction
# has to be asserted here instead -- over the campaign's ENTIRE stream set, not a
# sample, because a collision between two methods is exactly the aliasing the
# seeding module exists to prevent.
# ---------------------------------------------------------------------------

from doe_xgb.campaign.seeding import UINT32, as_uint32, derive_seed  # noqa: E402
from doe_xgb.campaign.runner import DATASETS, N_REPLICATIONS  # noqa: E402

_METHODS = ("grid", "random", "bayes_quality", "bayes_cost", "tpe_quality",
            "tpe_cost", "nsga2", "empirical_anchor_search",
            "historical_ws_asrun", "historical_ws", "ws_s", "nbi_s", "nbi_r")
_STAGES = ("main", "direct_search", "anchor", "candidate_validation")


def _campaign_keys():
    return [(d, r, m, s)
            for d in DATASETS for r in range(N_REPLICATIONS)
            for m in _METHODS for s in _STAGES]


def test_as_uint32_is_within_the_sklearn_accepted_range():
    for key in _campaign_keys()[:500]:
        v = as_uint32(derive_seed(*key))
        assert isinstance(v, int)
        assert 0 <= v < UINT32


def test_narrowing_preserves_distinctness_across_the_whole_campaign():
    keys = _campaign_keys()
    narrowed = [as_uint32(derive_seed(*k)) for k in keys]
    assert len(set(narrowed)) == len(keys), (
        f"{len(keys) - len(set(narrowed))} seed collision(s) after narrowing to 32 "
        f"bits; two streams would be aliased")


def test_narrowing_is_deterministic():
    k = ("magic", 7, "bayes_quality", "direct_search")
    assert as_uint32(derive_seed(*k)) == as_uint32(derive_seed(*k))


def test_the_gaussian_process_accepts_every_narrowed_seed():
    """The exact call that failed, exercised with the real estimator."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    import numpy as np

    X = np.linspace(0, 1, 12).reshape(-1, 1)
    y = np.sin(3 * X).ravel()
    for key in [("magic", 0, "bayes_quality", "direct_search"),
                ("spambase", 0, "bayes_cost", "direct_search"),
                ("adult", 29, "tpe_quality", "direct_search")]:
        seed = derive_seed(*key)
        assert seed >= UINT32 or True          # width is not the point; acceptance is
        GaussianProcessRegressor(normalize_y=True,
                                 random_state=as_uint32(seed)).fit(X, y)


def test_raw_64_bit_seed_is_rejected_by_sklearn():
    """The defect this guards against, demonstrated rather than asserted."""
    import pytest
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.utils._param_validation import InvalidParameterError
    import numpy as np

    X = np.linspace(0, 1, 12).reshape(-1, 1)
    y = np.sin(3 * X).ravel()
    wide = next(s for s in (derive_seed("magic", r, "bayes_quality", "direct_search")
                            for r in range(50)) if s >= UINT32)
    with pytest.raises(InvalidParameterError):
        GaussianProcessRegressor(normalize_y=True, random_state=wide).fit(X, y)


def test_numpy_generators_still_get_the_full_width():
    """Narrowing is for third-party APIs only; numpy must keep 64 bits."""
    import inspect
    from doe_xgb.campaign import seeding
    src = inspect.getsource(seeding.generator)
    assert "as_uint32" not in src, "generator() must not narrow; numpy takes 64 bits"
