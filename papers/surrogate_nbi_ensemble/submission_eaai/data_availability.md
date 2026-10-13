# Data availability statement

**Statement for the submission form.** "The datasets analysed are publicly available third-party data and are not
redistributed by the authors. The repository records, for each dataset, the exact source, the acquisition command,
the expected row and feature counts, the expected class prevalence and the SHA-256 checksum of the file used, so a
copy of the same file can be verified byte-for-byte. The aggregated tables, statistics, figures and
analysis code are openly available; the per-replication raw artifacts are not deposited at present."

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

All three Kaggle training files were obtained from public mirror datasets hosted on Kaggle and validated against the
recorded row count, feature count and class prevalence; the SHA-256 checksum identifies the exact file used. The
manifests name the mirror for each dataset.

## What is openly available, and what is not

- Openly available: aggregated tables, statistics outputs, manifests and figures, in the repository, versioned.
- Not deposited at present: per-replication raw artifacts (out-of-fold probability matrices, candidate sets, reference samples, NSGA-II
  populations): approximately 829 MB plus the NSGA-II tree, not version-controlled (the out-of-fold files carry the Kaggle
  training labels, so a deposit is bound by the competition rules), and
  reconstructible from the recorded seeds and the runners, except for the wall-clock inference costs, which are
  re-measured on each run, and the cost-dependent results derived from them; the measured costs are committed in
  `reports/pco213_postwork_benchmark/tables/inference_costs.csv`.
