# Fairness diagnostics notebooks

## Included notebooks

- `notebooks/01_fairness_dataset_diagnostics.ipynb`
- `notebooks/02_doe_search_space_audit.ipynb`

## Suggested repo location

```text
repo/
  notebooks/
    01_fairness_dataset_diagnostics.ipynb
    02_doe_search_space_audit.ipynb
  artifacts/
```

## What each notebook does

### 01_fairness_dataset_diagnostics.ipynb
Runs one-shot diagnostics for:
- bank_marketing
- german_credit
- credit_card_default

Outputs:
- dataset summaries
- protected-target correlation
- feature-vs-protected correlation
- feature-vs-target correlation
- top-feature heatmaps and barplots

### 02_doe_search_space_audit.ipynb
Audits the experimental design:
- nominal ranges
- edge coverage
- point-type counts
- duplicate design rows
- fairness-critical pairwise projections

## Output folders

- `artifacts/dataset_diagnostics/`
- `artifacts/doe_audit/`

## Notes

- These notebooks are exploratory/diagnostic.
- They are not meant to run inside the R30 pipeline.
- They are designed to work both with repo-relative paths and with the files attached in `/mnt/data`.