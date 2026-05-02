-- OpenML-CC18 doctoral benchmark SQLite job-matrix schema.
-- ========================================================
-- Primary unit of work: one OpenML task × algorithm × method × replica.
-- 72 tasks × 3 algorithms × 1 method × 30 replicas = 6,480 jobs at full
-- scale. The shard generator (planned for Commit 26) writes one or more
-- shard SQLite files under jobs/doctoral/openml_cc18/shards/.

PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS cc18_jobs (
    job_id            TEXT    PRIMARY KEY,
    openml_task_id    INTEGER NOT NULL,
    openml_dataset_id INTEGER NOT NULL,
    dataset_name      TEXT    NOT NULL,
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
    UNIQUE (openml_task_id, algorithm, method, replica)
);

CREATE INDEX IF NOT EXISTS idx_cc18_jobs_status_stage
    ON cc18_jobs (status, stage);
CREATE INDEX IF NOT EXISTS idx_cc18_jobs_task_alg
    ON cc18_jobs (openml_task_id, algorithm);
CREATE INDEX IF NOT EXISTS idx_cc18_jobs_assigned_worker
    ON cc18_jobs (assigned_worker, status);

CREATE TABLE IF NOT EXISTS shard_meta (
    shard_id         INTEGER PRIMARY KEY,
    suite_id         INTEGER NOT NULL,
    panel_version    TEXT    NOT NULL,
    generated_at     TEXT    NOT NULL,
    n_tasks          INTEGER NOT NULL,
    n_algorithms     INTEGER NOT NULL,
    n_methods        INTEGER NOT NULL,
    n_replicas_max   INTEGER NOT NULL,
    notes            TEXT    NULL
);

CREATE TRIGGER IF NOT EXISTS trg_cc18_jobs_touch
AFTER UPDATE ON cc18_jobs
FOR EACH ROW
BEGIN
    UPDATE cc18_jobs
    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
    WHERE job_id = NEW.job_id;
END;
