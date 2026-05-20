# Stage-3 / top-up distributed runbook

This is the **operator runbook** for running OpenML-CC18 top-up
tiers on a second Mac (the dedicated Mac, the future university
Mac, or a contingency host). It assumes:

- Commit 45 signed `stage0_replica_001`
  (`jobs/doctoral/openml_cc18/stage3_signoff.json` exists with
  `signoff_status = "signed"`).
- Commit 46 published the top-up plan, manifests, and worker plan.
- Commit 47 ran the **tiny Stage-3 / top-up pilot**
  (`shard_00` / replica 2 / standard lane / canary methods only;
  see `experiments/_stage_runs/`
  `stage3_pilot_replica_002_shard00_standard_lane_latest_summary.{json,md}`).
- Commit 48 expanded the pilot to **replica_002 standard lane
  across all 10 shards** (still no heavy / extreme; still no
  full `topup_to_5`; see `experiments/_stage_runs/`
  `stage3_replica_002_standard_lane_latest_summary.{json,md}`).
  Operator review of that summary is required before any
  heavy-lane or broader top-up execution.
- The reader is the operator on a worker machine.

The runbook is **execution-oriented**. It does not change policy,
shards, or signoff content.

## Preconditions on the worker

1. macOS host with Python 3.10–3.12 installed via `pyenv` (the
   repository pins `>=3.10`, tested on 3.12).
2. Git installed and configured (the worker needs to push
   summaries back).
3. A spare ~50 GB of disk for `runs/` and `catboost_info/`. Note
   these directories are **gitignored**.
4. `caffeinate` (default on macOS) — required for any unattended
   run.

## Step 1 — clone / pull on the worker

First time:

```bash
mkdir -p ~/Projects && cd ~/Projects
git clone https://github.com/caioribeiro99/doe_nbi_hpo_project
cd doe_nbi_hpo_project
git switch repo-publication-readiness
```

Subsequent pulls:

```bash
cd ~/Projects/doe_nbi_hpo_project
git fetch origin
git switch repo-publication-readiness
git pull --ff-only
```

If `git pull --ff-only` refuses, the worker has accidentally
committed local artifacts (or the local branch has diverged).
Investigate via `git status`; **do not** force-merge.

## Step 2 — verify the commit SHA

```bash
git log -1 --oneline
```

Expected (post-Commit-46): the planner artifacts exist:

```bash
test -s experiments/_stage_runs/stage3_topup_plan_latest_summary.json
test -s benchmarks/doctoral/openml_cc18/stage3_topup_manifest.csv
test -s benchmarks/doctoral/openml_cc18/stage3_worker_plan.csv
```

If any of these files are missing, the worker is at a pre-Commit-46
revision; pull again.

## Step 3 — verify `stage3_signoff.json`

The signoff file is the gate that unlocks stage-3 top-up planning.

```bash
test -f jobs/doctoral/openml_cc18/stage3_signoff.json
python - <<'PY'
import json, pathlib
record = json.loads(
    pathlib.Path("jobs/doctoral/openml_cc18/stage3_signoff.json").read_text()
)
assert record["signoff_status"] == "signed"
assert record["signoff_type"] == "stage0_replica_001"
assert record["policy_version"] == (
    "47b6b50c6d1e1d09087c148bb69464bbed99eface9c411c621331a4ad7855f36"
)
assert record["downstream_execution_authorized_in_this_commit"] is False
print("signoff verified")
PY
```

If any assertion fails, **stop**. Re-fetch the branch or escalate.

## Step 4 — verify the Python environment

```bash
python -V                                # expect 3.10–3.12
pip install -e ".[gbdt,hpo_baselines,doctoral,dev]"
python scripts/audit_method_capabilities.py
```

The capability audit refreshes the `dispatch_only` / `smoke_ready`
/ `stub_only` lists. Compare its `--print-only` output against the
last committed `capability_audit` block in the lane summaries; the
canary methods (`default_gbdt`, `random_search`, `tpe_optuna`,
`doe_rsm_vrf_true_nbi`) must be in `smoke_ready`. If only
timestamps differ, do **not** commit the regenerated audit (the
commit guard already drops the file in this case).

## Step 5 — pick assigned shards

Open `benchmarks/doctoral/openml_cc18/stage3_worker_plan.csv` and
identify the row matching your worker:

- Personal Mac: standard lane, `topup_to_5` sample shard(s).
- Dedicated Mac: standard + heavy lanes, `topup_to_5` and
  `topup_to_10`.
- University Mac: heavy + extreme, especially `topup_to_30`.

Then open `benchmarks/doctoral/openml_cc18/stage3_topup_manifest.csv`
to get the exact (tier, lane, shard subdir) tuple.

## Step 6 — start `caffeinate`

```bash
# Keep the worker awake for the duration of the run.
caffeinate -i -s &
CAFFEINATE_PID=$!
```

Re-enable sleep afterwards with `kill "$CAFFEINATE_PID"`.

## Step 7 — run only the assigned shards / lanes

The runner takes a shard path and a lane. Example for the
Commit-47 pilot:

```bash
python scripts/cc18_runner.py \
    --shard jobs/doctoral/openml_cc18/shards/stage1_topup_to_005/shard_00.sqlite \
    --lane standard \
    --canary-only \
    --train \
    --resume
```

For heavy:

```bash
python scripts/cc18_runner.py \
    --shard jobs/doctoral/openml_cc18/shards/stage1_topup_to_005/shard_03.sqlite \
    --lane heavy \
    --canary-only \
    --train \
    --resume
```

For extreme, the runner refuses unless `--include-extreme-tasks`
is passed (per the YAML default). Confirm the operator-review
gate before doing so. The two extreme tasks are `letter` (task 6)
and `Devnagari-Script` (task 167121); the latter carries the
Commit-45 caveat `devnagari_extreme_budget_non_equivalence`. The
standard-lane caveat `isolet_future_recalibration_candidate`
(task 3481) is relevant when reviewing standard-lane summaries:
isolet's `doe_rsm_vrf_true_nbi` × catboost cell ran ~1078 s at
R=1, which is above the 900 s heavy-promotion threshold even
though the cell succeeded inside the 1800 s standard-lane timeout.

## Step 8 — export summaries

```bash
python scripts/export_cc18_run_summary.py \
    --run-id <run_id_assigned_by_runner> \
    --out-json experiments/_stage_runs/stage1_topup_to_005_standard_lane_latest_summary.json \
    --out-md   experiments/_stage_runs/stage1_topup_to_005_standard_lane_latest_summary.md
```

(Replace tier / lane in the output filename as appropriate.)

## Step 9 — commit only JSON / MD summaries

Stage **only** the summary artifacts and any updated docs:

```bash
git add experiments/_stage_runs/stage1_topup_to_005_standard_lane_latest_summary.json
git add experiments/_stage_runs/stage1_topup_to_005_standard_lane_latest_summary.md
git status --short          # double-check nothing else is staged
git diff --cached --stat
git commit -m "Commit N: stage1_topup_to_005 standard lane summary"
git push origin repo-publication-readiness
```

## Step 10 — what NOT to commit

Never `git add`:

- `runs/` — gitignored execution copies + per-cell fold outputs;
- `data/source/openml_cc18/` — raw OpenML payloads;
- `catboost_info/` — CatBoost training side-effects;
- fitted model files (`.cbm`, `.bin`, `.pkl`, `.joblib`, etc.);
- fold CSVs / nested run outputs;
- notebooks;
- fairness / per-fold artifacts;
- modified `jobs/doctoral/openml_cc18/shards/**.sqlite` —
  the committed shards must remain byte-identical to the Commit-28
  baseline;
- regenerated `heavy_task_policy.csv` or its report unless
  producing an explicit, named policy-change commit;
- regenerated `runtime_guardrails.yaml`.

The `.gitignore` already excludes most of these; the commit
hooks (where present) refuse the rest. Never use `git add -A`.

## Step 11 — resume after interruption

If the worker was killed mid-run (power, crash, accidental Ctrl-C):

```bash
# Verify branch + commit.
git status
git log -1 --oneline

# The runner's --resume flag picks up where the previous invocation
# left off. success / skipped rows are skipped; pending rows are
# retried; failed_other rows require triage (see step 13).
python scripts/cc18_runner.py \
    --shard <same_shard_as_before> \
    --lane <same_lane> \
    --canary-only \
    --train \
    --resume
```

The SQLite `claim_lease_seconds` expires automatically; if the
runner refuses on "row claimed by another worker", wait until the
lease times out (default 60 minutes) or, if you're certain the
other worker is dead, release the lease manually via:

```bash
sqlite3 <shard.sqlite> \
    "UPDATE cc18_jobs SET status = 'pending', claim_lease_expires_at = NULL
     WHERE status = 'running' AND claim_lease_expires_at < strftime('%s','now')"
```

## Step 12 — retry a failed cell

`failed_timeout` cells are a lane-policy outcome. Re-running them
under the same policy will produce the same result. Two options:

- **Accept the timeout** as part of the lane summary; the per-tier
  signoff will record `n_failed_timeout`.
- **Re-run on a more lenient lane**: not allowed inside a tier
  (would silently change policy). Promote the task to the next
  lane only via a deliberate policy-change commit (Option B in
  `docs/STAGE3_POLICY_DECISION.md`).

`failed_other` cells are real errors. To retry:

```bash
sqlite3 <shard.sqlite> \
    "UPDATE cc18_jobs SET status = 'pending', last_error = NULL
     WHERE status = 'failed' AND last_error NOT LIKE 'TimeoutError%'
       AND openml_task_id = <task_id> AND algorithm = '<algorithm>'
       AND method = '<method>'"
```

Then re-run the runner with `--resume`.

## Step 13 — mark failed_timeout vs failed_other

The runner already distinguishes these:

- `n_jobs_failed_timeout` — converted to `failed_timeout` when the
  worker subprocess exceeded `timeout_seconds_per_cell`.
- `n_jobs_failed_other` — everything else (segfault, OOM,
  framework exception, OpenML payload error).

The lane summary aggregates both into `n_jobs_failed`. The per-
tier aggregator (planned mirror of
`scripts/build_stage0_replica_signoff.py`) refuses to mix them
silently.

## Step 14 — sync results back to the personal Mac

The personal Mac is the source of truth for the planning surface.
After the worker pushes a summary commit:

```bash
# On the personal Mac:
cd ~/Projects/doe_nbi_hpo_project
git fetch origin
git switch repo-publication-readiness
git pull --ff-only

# Re-run the planner to incorporate the new tier summary into the
# top-up plan (the planner is read-only against the lane summaries).
python scripts/plan_stage3_topup.py
```

Commit 47 has executed the tiny Stage-3 / top-up pilot
(`shard_00` / replica 2 / standard lane / canary methods only).
Commit 48 has expanded the pilot to **replica_002 standard lane
across all 10 shards** (still no heavy, no extreme, no full
`topup_to_5`). Both summaries at
`experiments/_stage_runs/`
`stage3_pilot_replica_002_shard00_standard_lane_latest_summary.{json,md}`
and
`experiments/_stage_runs/`
`stage3_replica_002_standard_lane_latest_summary.{json,md}`
**must** be operator-reviewed before any further Stage-3 dispatch.
Do not scale directly to the full `topup_to_5` tier; the next
planned step is for the operator to decide between running
replica_002 heavy lane, a selected heavy-lane pilot first, or an
aggregate review for replica_002 standard before heavy execution.

When a per-tier aggregator script lands (planned after operator
review of Commit 47), the personal Mac will run it here and either
publish a planning-only artifact or, after operator review, write
the per-tier signoff.

## Step 15 — health checks before each run

Before a new run on any worker, verify:

1. `git status` is clean.
2. `git log -1 --oneline` matches the HEAD of
   `origin/repo-publication-readiness`.
3. `python scripts/plan_stage3_topup.py --dry-run` exits 0.
4. `python scripts/build_stage0_replica_signoff.py --dry-run`
   exits 0 with `signoff_status: signed`.
5. The shard MD5s under `jobs/doctoral/openml_cc18/shards/` match
   the values recorded in `shards/shard_summary.json` (the planner
   records the live MD5s in
   `committed_shard_md5_snapshot`).

If any of these fail, **stop** and escalate to the planning
machine.

## Step 16 — stale / drifted detection

The planner's JSON output exposes:

- `policy_drift_detected`: `true` if the live
  `heavy_task_policy.csv` SHA-256 differs from the signoff's
  recorded value.
- `lane_summary_sha256_live` vs `lane_summary_sha256_signed`:
  the per-lane comparison.
- `signoff_status` should always be `"signed"` when this commit
  is in play.

A worker is **drifted** if its planner output shows
`policy_drift_detected = true`. A worker is **stale** if its
`git_sha` does not match `origin/repo-publication-readiness` at
the moment the summary is exported.

Both conditions block a tier signoff; the per-tier aggregator
will refuse.

## Pointer to upstream docs

- `docs/STAGE3_TOPUP_EXECUTION_PLAN.md` — strategic context.
- `docs/STAGE3_POLICY_DECISION.md` — Option A vs B.
- `docs/STAGE0_REPLICA_001_SIGNOFF_PLAN.md` — Commit-44/45 review surface.
- `docs/RESULT_HANDOFF_PROTOCOL.md` — execution copy ↔ committed shard protocol.
- `docs/HEAVY_TASK_POLICY.md` — lane definitions + per-task overrides.
- `jobs/doctoral/openml_cc18/README.md` — shard layout.
