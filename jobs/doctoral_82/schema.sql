-- Doctoral benchmark SQLite job-matrix schema
-- =============================================
-- One row per (dataset, algorithm, method, replica). Sharded by stage:
-- the campaign tops up the replica count (1 -> 5 -> 10 -> 30) without
-- re-running earlier replicas.
--
-- The shard generator (Commit 25, planned) populates this table from
-- benchmarks/doctoral_82/datasets.csv and writes one .sqlite file per
-- shard under jobs/doctoral_82/shards/.

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS doctoral_jobs (
    job_id            TEXT    PRIMARY KEY,
    dataset_id        TEXT    NOT NULL,
    algorithm         TEXT    NOT NULL,
    method            TEXT    NOT NULL,
    replica           INTEGER NOT NULL CHECK (replica >= 1),
    stage             TEXT    NOT NULL CHECK (stage IN (
        'stage0_replica_001',
        'stage1_topup_to_005',
        'stage2_topup_to_010',
        'stage3_topup_to_030'
    )),
    config_path       TEXT    NOT NULL,
    output_dir        TEXT    NOT NULL,
    estimated_seconds REAL    NULL,
    status            TEXT    NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'claimed', 'running', 'success', 'failed', 'skipped'
    )),
    assigned_worker   TEXT    NULL,
    retry_count       INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT    NULL,
    started_at        TEXT    NULL,
    finished_at       TEXT    NULL,
    runtime_seconds   REAL    NULL,
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE (dataset_id, algorithm, method, replica)
);

-- Convenience indexes for the shard worker.
CREATE INDEX IF NOT EXISTS idx_doctoral_jobs_status_stage
    ON doctoral_jobs (status, stage);
CREATE INDEX IF NOT EXISTS idx_doctoral_jobs_dataset_algorithm
    ON doctoral_jobs (dataset_id, algorithm);
CREATE INDEX IF NOT EXISTS idx_doctoral_jobs_assigned_worker
    ON doctoral_jobs (assigned_worker, status);

-- Per-shard meta block; one row per shard file.
CREATE TABLE IF NOT EXISTS shard_meta (
    shard_id         INTEGER PRIMARY KEY,
    panel_version    TEXT    NOT NULL,
    generated_at     TEXT    NOT NULL,
    n_datasets       INTEGER NOT NULL,
    n_algorithms     INTEGER NOT NULL,
    n_methods        INTEGER NOT NULL,
    n_replicas_max   INTEGER NOT NULL,
    notes            TEXT    NULL
);

-- Touch trigger to keep updated_at honest.
CREATE TRIGGER IF NOT EXISTS trg_doctoral_jobs_touch
AFTER UPDATE ON doctoral_jobs
FOR EACH ROW
BEGIN
    UPDATE doctoral_jobs
    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE job_id = NEW.job_id;
END;
