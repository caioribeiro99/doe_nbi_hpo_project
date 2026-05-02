# OpenML-CC18 — primary doctoral benchmark

The doctoral campaign evaluates the proposed DOE + RSM + VRF/FMSE +
true NBI framework on the **complete OpenML-CC18 benchmark suite**
(`suite_id = 99`). The suite ships **72 standardized classification
tasks**; this is a task-based benchmark, not an arbitrary list of
datasets.

## Files

- `tasks.csv` — one row per OpenML task. Schema in
  `scripts/import_openml_cc18.py` (TASK_COLUMNS).
- `datasets.csv` — one row per *unique* underlying dataset (a few
  CC18 tasks share the same dataset).
- `openml_cc18_metadata.json` — suite metadata + import provenance.
- `coverage_report.md` — auto-generated coverage statistics
  (binary vs multiclass, categorical share, imbalance share, size
  buckets).

All four files are written by:

```bash
python scripts/import_openml_cc18.py --suite-id 99 \
    --out-dir benchmarks/doctoral/openml_cc18
```

The importer is network-bound and pulls only OpenML *metadata*. It
**never** downloads dataset payloads; raw / processed CSV files
continue to be gitignored under `data/source/<id>/`.

## Job count

The doctoral job matrix is task-based:

```
72 tasks × 3 algorithms × 1 method × 30 replicas = 6,480 jobs
```

Stage breakdown (top-up cadence; each stage adds replicas to the
existing tasks rather than re-running them):

| Stage | Replicas added | Jobs added | Cumulative |
|---|---:|---:|---:|
| stage0_replica_001 | 1 | 216 | 216 |
| stage1_topup_to_005 | 4 | 864 | 1,080 |
| stage2_topup_to_010 | 5 | 1,080 | 2,160 |
| stage3_topup_to_030 | 20 | 4,320 | 6,480 |

## Versus the internal smoke panel

The 12-dataset internal panel at
`benchmarks/doctoral/internal_smoke_panel/` is a smoke / profiling /
development fixture. It is **not** part of this benchmark; do not
report results from it as doctoral results unless explicitly used as
supplementary examples.

## Versus the deprecated `doctoral_82/` directory

`benchmarks/doctoral_82/` is the pre-pivot scaffolding. Treat it as
deprecated; the canonical primary benchmark lives here, in
`benchmarks/doctoral/openml_cc18/`.
