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
