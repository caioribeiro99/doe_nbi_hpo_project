"""Doctoral benchmark job-matrix planning utilities.

Pure-Python helpers used by the (planned) ``scripts/generate_doctoral_job_shards.py``
to enumerate the cartesian (dataset, algorithm, method, replica)
job rows for the staged campaign. The actual SQLite writing is
deferred to Commit 25; this module is the deterministic enumerator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from .registry import DatasetRow

STAGE_NAMES = (
    "stage0_replica_001",
    "stage1_topup_to_005",
    "stage2_topup_to_010",
    "stage3_topup_to_030",
)


@dataclass(frozen=True)
class JobRow:
    job_id: str
    dataset_id: str
    algorithm: str
    method: str
    replica: int
    stage: str
    config_path: str
    output_dir: str


def job_id(*, dataset_id: str, algorithm: str, method: str, replica: int) -> str:
    """Deterministic job id from (dataset, algorithm, method, replica)."""
    key = f"{dataset_id}|{algorithm}|{method}|{int(replica):04d}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def stage_topup_replicas(stage: str) -> tuple[int, int]:
    """Return ``(low, high)`` replica indices that belong to ``stage``.

    Replica indices are 1-based and inclusive on both ends.
    """
    if stage == "stage0_replica_001":
        return (1, 1)
    if stage == "stage1_topup_to_005":
        return (2, 5)
    if stage == "stage2_topup_to_010":
        return (6, 10)
    if stage == "stage3_topup_to_030":
        return (11, 30)
    raise ValueError(f"unknown stage: {stage}")


def generate_job_rows(
    *,
    datasets: Iterable[DatasetRow],
    algorithms: Iterable[str],
    methods: Iterable[str],
    stages: Iterable[str] = STAGE_NAMES,
    config_path_template: str = "configs/article_3vrf_{dataset_id}.yaml",
    output_dir_template: str = "experiments/doctoral_82/{dataset_id}/{algorithm}/{method}/replica_{replica:03d}",
) -> list[JobRow]:
    """Enumerate the staged cartesian job matrix.

    Only datasets with ``include=True`` are emitted.
    """
    out: list[JobRow] = []
    included = [d for d in datasets if d.include]
    algs = list(algorithms)
    meths = list(methods)
    for stage in stages:
        lo, hi = stage_topup_replicas(stage)
        for did_row in included:
            for alg in algs:
                for method in meths:
                    for rep in range(lo, hi + 1):
                        jid = job_id(
                            dataset_id=did_row.dataset_id,
                            algorithm=alg,
                            method=method,
                            replica=rep,
                        )
                        cfg = config_path_template.format(dataset_id=did_row.dataset_id)
                        outd = output_dir_template.format(
                            dataset_id=did_row.dataset_id,
                            algorithm=alg,
                            method=method,
                            replica=rep,
                        )
                        out.append(JobRow(
                            job_id=jid,
                            dataset_id=did_row.dataset_id,
                            algorithm=alg,
                            method=method,
                            replica=rep,
                            stage=stage,
                            config_path=cfg,
                            output_dir=outd,
                        ))
    return out


__all__ = [
    "JobRow",
    "STAGE_NAMES",
    "generate_job_rows",
    "job_id",
    "stage_topup_replicas",
]
