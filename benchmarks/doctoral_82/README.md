# Doctoral benchmark registry (target = 82 datasets)

This directory holds the canonical list of datasets that drive the
doctoral-scale benchmark campaign for the DOE + RSM + VRF + true NBI
framework on GBDT models.

## Files

- `datasets.csv` — the registry. One row per candidate dataset.
  Columns are pinned by `dataset_schema.json`.
- `dataset_schema.json` — column contract, types, and validation
  rules.
- `selection_policy.md` — how datasets are chosen for inclusion
  (sources considered, balance criteria, exclusion rules).

## Status

- **12 / 82 entries seeded** (the article v1 panel; `include=true`).
- **70 / 82 entries pending**. They will be filled by following
  `selection_policy.md`. Use
  `python scripts/import_doctoral_benchmark_datasets.py --csv ...`
  to merge an external curated list into `datasets.csv`.

## CLI

```bash
# Validate the registry
python scripts/import_doctoral_benchmark_datasets.py \
    --csv benchmarks/doctoral_82/datasets.csv --validate-only

# Merge an external curated CSV (idempotent on dataset_id)
python scripts/import_doctoral_benchmark_datasets.py \
    --csv path/to/external.csv \
    --out benchmarks/doctoral_82/datasets.csv
```

## What this directory does NOT contain

- Real raw / processed dataset payloads (those stay under
  `data/source/<id>/raw|processed/`, gitignored).
- Job SQLite shards (those land under `jobs/doctoral_82/` once
  Commit 25 generates them).
