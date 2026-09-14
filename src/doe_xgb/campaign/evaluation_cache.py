"""Physical memoization with logical per-method accounting.

The four arms and the baselines request many identical XGBoost configurations.
Computing each one once saves wall clock. It must not make any optimizer appear to
have spent a smaller scientific budget, and it must not let one method see an
evaluation another method paid for.

Two ledgers, kept separately and both reported:

**Logical evaluations** -- what a method would have consumed running alone. Every
request a method makes counts, hit or miss. This is the scientific budget and the
only number that enters a fairness comparison.

**Unique physical fits** -- what the workstation actually computed. Counts misses
only. This is an engineering quantity and never enters a comparison between
methods.

Isolation is the other half. A method receives a result only for a configuration it
requested itself; it is never handed, and cannot enumerate, what any other method
requested. Physical memoization happens strictly beneath that boundary. Concretely,
a method's ``MethodView`` exposes ``evaluate`` and its own history, and nothing
else, so nothing can leak into a Bayesian surrogate, a tree-structured Parzen
history, an evolutionary population, an anchor search or a weighted-sum search.

The cache key pins everything that could change a result:

    dataset id, outer split identity, inner fold definition, the configuration
    after canonical type normalization, the training seed, and the evaluation
    protocol version.

A change to any of them is a different evaluation, which is what makes reuse safe.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "xgboost-hpo-protocol-v2"

# Hyperparameters that are integers in the learner and continuous in the surrogate.
# Normalizing them here is what makes "the same configuration" well defined.
INT_PARAMS = frozenset({"max_depth", "n_estimators"})
FLOAT_DECIMALS = 12


def canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a configuration so equal configurations produce equal keys.

    Integer hyperparameters are rounded to int; floats are rounded to a fixed
    number of decimals so that values differing only in the last representable
    bit are the same evaluation. Keys are sorted.
    """
    out: dict[str, Any] = {}
    for k in sorted(config):
        v = config[k]
        if k in INT_PARAMS:
            out[k] = int(round(float(v)))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = round(float(v), FLOAT_DECIMALS) + 0.0   # normalize -0.0
        else:
            out[k] = v
    return out


def evaluation_key(*, dataset: str, split_id: str, fold_id: str,
                   config: Mapping[str, Any], seed: int,
                   protocol_version: str = PROTOCOL_VERSION) -> str:
    payload = {
        "dataset": dataset,
        "split_id": split_id,
        "fold_id": fold_id,
        "config": canonical_config(config),
        "seed": int(seed),
        "protocol_version": protocol_version,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class MethodLedger:
    """Per-method accounting. ``logical`` is the scientific budget."""

    method: str
    logical: int = 0
    served_from_cache: int = 0
    computed: int = 0
    keys: list[str] = field(default_factory=list)
    by_stage: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def unique_logical(self) -> int:
        """Distinct evaluations this method asked for.

        Differs from ``logical`` when a method re-requests a configuration it
        already evaluated. That is the method's own repetition and is charged to
        it, so ``logical`` remains the budget.
        """
        return len(set(self.keys))

    def as_dict(self) -> dict[str, int]:
        return {"method": self.method, "logical_evaluations": self.logical,
                "unique_logical_evaluations": self.unique_logical,
                "served_from_cache": self.served_from_cache,
                "computed_by_this_method": self.computed,
                "by_stage": {k: dict(v) for k, v in sorted(self.by_stage.items())}}


class EvaluationCache:
    """Persistent per-replication memoization, with per-method ledgers.

    Use :meth:`view` to hand a method its own isolated accessor. Never pass the
    cache itself to an optimizer.
    """

    def __init__(self, path: str | Path, *, dataset: str, split_id: str,
                 seed: int, protocol_version: str = PROTOCOL_VERSION) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset, self.split_id = dataset, split_id
        self.seed, self.protocol_version = int(seed), protocol_version
        self._ledgers: dict[str, MethodLedger] = {}
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS evaluations ("
            "  key TEXT PRIMARY KEY, dataset TEXT, split_id TEXT, fold_id TEXT,"
            "  config TEXT, seed INTEGER, protocol_version TEXT,"
            "  result TEXT, first_requested_by TEXT)")
        # The logical ledger is the scientific budget, so it must survive a resumed
        # run. Holding it only in memory meant a unit resumed from checkpoints
        # reported zero logical evaluations beside thousands of physical fits.
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS requests ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT, method TEXT NOT NULL,"
            "  stage TEXT NOT NULL DEFAULT 'unspecified', key TEXT NOT NULL,"
            "  served_from_cache INTEGER NOT NULL, computed INTEGER NOT NULL,"
            "  config TEXT)")
        self._db.execute("CREATE INDEX IF NOT EXISTS requests_method "
                         "ON requests(method, stage)")
        self._db.commit()
        self._load_ledgers()

    def _load_ledgers(self) -> None:
        """Rebuild the per-method and per-stage ledgers from the request log.

        A resumed unit must reconstruct both accountings exactly; holding them in
        memory lost the scientific budget on every resume.
        """
        rows = self._db.execute(
            "SELECT method, stage, key, served_from_cache, computed FROM requests "
            "ORDER BY id").fetchall()
        for method, stage, key, served, computed in rows:
            led = self._ledgers.setdefault(method, MethodLedger(method))
            led.logical += 1
            led.keys.append(key)
            led.served_from_cache += int(served)
            led.computed += int(computed)
            s = led.by_stage.setdefault(stage, {"logical": 0, "served_from_cache": 0,
                                                "computed": 0})
            s["logical"] += 1
            s["served_from_cache"] += int(served)
            s["computed"] += int(computed)

    # ------------------------------------------------------------------ queries

    def physical_fits(self) -> int:
        """Distinct evaluations actually computed for this replication."""
        return int(self._db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0])

    def ledger(self, method: str) -> MethodLedger:
        return self._ledgers.setdefault(method, MethodLedger(method))

    def accounting(self) -> dict[str, Any]:
        logical_total = sum(l.logical for l in self._ledgers.values())
        physical = self.physical_fits()
        return {
            "dataset": self.dataset, "split_id": self.split_id,
            "protocol_version": self.protocol_version,
            "per_method": [l.as_dict() for l in
                           sorted(self._ledgers.values(), key=lambda x: x.method)],
            "by_stage": {s: {"logical": sum(l.by_stage.get(s, {}).get("logical", 0)
                                            for l in self._ledgers.values()),
                             "computed": sum(l.by_stage.get(s, {}).get("computed", 0)
                                             for l in self._ledgers.values())}
                         for s in sorted({s for l in self._ledgers.values()
                                          for s in l.by_stage})},
            "logical_evaluations_total": logical_total,
            "unique_physical_fits": physical,
            "cache_hit_rate": (round(1.0 - physical / logical_total, 6)
                               if logical_total else 0.0),
            "note": ("logical is the scientific budget and the only figure that enters a "
                     "fairness comparison; unique physical fits is an engineering "
                     "quantity and enters none"),
        }

    # ------------------------------------------------------------------ the core

    def _evaluate(self, method: str, config: Mapping[str, Any], fold_id: str,
                  compute: Callable[[dict[str, Any]], dict[str, Any]],
                  stage: str = "unspecified") -> dict[str, Any]:
        cfg = canonical_config(config)
        key = evaluation_key(dataset=self.dataset, split_id=self.split_id,
                             fold_id=fold_id, config=cfg, seed=self.seed,
                             protocol_version=self.protocol_version)
        with self._lock:
            led = self.ledger(method)
            led.logical += 1                 # charged on every request, hit or miss
            led.keys.append(key)
            st = led.by_stage.setdefault(stage, {"logical": 0, "served_from_cache": 0,
                                                 "computed": 0})
            st["logical"] += 1
            row = self._db.execute(
                "SELECT result FROM evaluations WHERE key = ?", (key,)).fetchone()
            if row is not None:
                led.served_from_cache += 1
                st["served_from_cache"] += 1
                self._db.execute(
                    "INSERT INTO requests (method, stage, key, served_from_cache, "
                    "computed, config) VALUES (?,?,?,1,0,?)",
                    (method, stage, key, json.dumps(cfg, sort_keys=True)))
                self._db.commit()
                return json.loads(row[0])
        # computed outside the lock: a fit takes seconds and must not block others
        result = compute(dict(cfg))
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?,?,?,?,?,?)",
                (key, self.dataset, self.split_id, fold_id,
                 json.dumps(cfg, sort_keys=True), self.seed, self.protocol_version,
                 json.dumps(result), method))
            self._db.execute(
                "INSERT INTO requests (method, stage, key, served_from_cache, "
                "computed, config) VALUES (?,?,?,0,1,?)",
                (method, stage, key, json.dumps(cfg, sort_keys=True)))
            self._db.commit()
            led = self.ledger(method)
            led.computed += 1
            led.by_stage.setdefault(stage, {"logical": 0, "served_from_cache": 0,
                                            "computed": 0})["computed"] += 1
        return result

    def view(self, method: str,
             compute: Callable[[dict[str, Any]], dict[str, Any]],
             stage: str = "unspecified") -> "MethodView":
        """An isolated accessor for one method.

        The returned object is the only thing an optimizer is given.
        """
        return MethodView(self, method, compute, stage)

    def close(self) -> None:
        self._db.close()


class MethodView:
    """One method's window onto the cache.

    Exposes evaluation and this method's own history. It deliberately exposes no
    way to reach another method's requests, the underlying table, or the cache
    object, because an optimizer that could reach those would be able to learn
    from evaluations it never paid for.
    """

    __slots__ = ("_cache", "_method", "_compute", "_history", "_stage")

    def __init__(self, cache: EvaluationCache, method: str,
                 compute: Callable[[dict[str, Any]], dict[str, Any]],
                 stage: str = "unspecified") -> None:
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_compute", compute)
        object.__setattr__(self, "_history", [])
        object.__setattr__(self, "_stage", stage)

    @property
    def method(self) -> str:
        return self._method

    @property
    def stage(self) -> str:
        return self._stage

    def evaluate(self, config: Mapping[str, Any], *, fold_id: str = "all") -> dict[str, Any]:
        result = self._cache._evaluate(self._method, config, fold_id, self._compute,
                                       self._stage)
        self._history.append({"config": canonical_config(config), "result": result})
        return result

    def history(self) -> list[dict[str, Any]]:
        """Only what this method requested, in request order."""
        return [dict(h) for h in self._history]

    @property
    def logical_evaluations(self) -> int:
        return self._cache.ledger(self._method).logical

    def __repr__(self) -> str:
        return f"MethodView(method={self._method!r}, logical={self.logical_evaluations})"


__all__ = ["EvaluationCache", "MethodView", "MethodLedger", "PROTOCOL_VERSION",
           "canonical_config", "evaluation_key"]
