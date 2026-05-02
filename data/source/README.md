# Datasets

Datasets are *not* committed to the repository. They are downloaded
on demand by `scripts/fetch_magic_dataset.py` and verified against
`CHECKSUMS.txt`.

## UCI MAGIC Gamma Telescope (canonical dataset)

The dissertation's headline experiments use this dataset.

```bash
make data
# or
python scripts/fetch_magic_dataset.py
```

The downloader:
1. Fetches `magic04.data` from the UCI ML repository.
2. Saves it to `data/source/magic.csv`.
3. Computes the SHA-256 and compares it against `CHECKSUMS.txt`.

If the upstream UCI mirror changes, update `CHECKSUMS.txt` and the
expected SHA-256 in `scripts/fetch_magic_dataset.py` after manually
validating the new file.

## Article-track v1 panel

Eleven non-sklearn datasets are fetched by per-dataset scripts under
`scripts/fetch_<id>_dataset.py`. Each script writes:

- `data/source/<id>/raw/<original-file>`
- `data/source/<id>/processed/<id>.csv`
- `data/source/<id>/manifest.json` (SHA-256 + byte size for every file)

The aggregated checksums live in `CHECKSUMS.txt`, segmented per dataset
with `# >>> <id>` ... `# <<< <id>` markers. The
`doe-xgb datasets fetch --all` and `doe-xgb datasets verify-checksums`
CLI helpers wrap these scripts. The `breast_cancer` entry is bundled
with scikit-learn and does not need a fetch step.

Real dataset payloads are **not** committed; only manifests, the
aggregated checksum file, and the README/availability documents are
versioned.
