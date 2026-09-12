# Data availability statement

**Statement for the submission form.** "The datasets analysed are publicly available third-party data and are not
redistributed by the authors. The repository records, for each dataset, the exact source, the acquisition command,
the expected row and feature counts, the expected class prevalence and the SHA-256 checksum of the file used, so an
independently obtained copy can be verified byte-for-byte. All derived artifacts, aggregated tables, statistics and
analysis code are openly available."

## Per-dataset provenance

| Dataset | Source | Redistribution |
|---|---|---|
| Santander Customer Transaction Prediction | Kaggle competition (2019) | Subject to competition rules; obtain from Kaggle |
| BNP Paribas Cardif Claims Management | Kaggle competition (2016) | Subject to competition rules; obtain from Kaggle |
| Porto Seguro Safe Driver Prediction | Kaggle competition (2017) | Subject to competition rules; obtain from Kaggle |
| Default of credit card clients | UCI Machine Learning Repository #350 (Yeh and Lien, 2009) | Freely downloadable |

Competition test labels are never used. All training, weighting, thresholding and model selection use only the
published training file, and every performance figure derives from it.

Porto Seguro was subsampled to 200,000 of 595,212 rows by a single stratified draw with a recorded seed.

Where a competition endpoint was unavailable at acquisition time, the file was obtained from a public mirror dataset
and validated against the recorded invariants (row count, feature count, class prevalence, checksum). The manifests
state which source was used for each dataset.

## What is openly available

- Aggregated tables, statistics outputs, manifests and figures: in the repository, versioned.
- Per-replication raw artifacts (out-of-fold probability matrices, candidate sets, reference samples, NSGA-II
  populations): approximately 829 MB plus the NSGA-II tree, deliberately not version-controlled, and fully
  reconstructible from the recorded seeds and the runner.
