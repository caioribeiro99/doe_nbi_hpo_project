## 9. Reproducibility, data and code availability

### 9.1 Version tags

Five tags separate what was specified from what was measured (Supplement S1). `v0.1.0-dissertation`
(`67d9fe5`) is the historical implementation this study reconstructs; it is read and executed as
archived, never edited. `xgboost-hpo-protocol-v1` froze the specification before the
measurement-validation pilot, `-v2` incorporated the pilot, and `-v3` (`9b15ba7`) is the
confirmatory protocol under which every unit ran. These tags record what was specified before any
result existed; the 23 numbered amendments between them carry the cause of each change beside it.
`xgboost-hpo-confirmatory-results-v1` (`a4f20ed`) is the first verified analysis: the per-unit
indicator table (1,080 rows, one per method-unit), the primary and secondary analysis artifacts,
and the verified-claims file.

### 9.2 Seed derivation

A seed is a pure function of `(dataset, replication, method, stage)`. The four fields are joined
with the frozen namespace `xgboost-hpo-vrfnbi/v3`, hashed with BLAKE2b to eight bytes, and the
resulting 64-bit integer is passed to `numpy.random.SeedSequence`. BLAKE2b rather than Python's
`hash()`, which is salted per process: a campaign resumed in a new interpreter would otherwise draw
different candidates. All 3,840 campaign streams are distinct, stable across processes and
`PYTHONHASHSEED`, and independent of execution order and worker count. Because the replication index
remains a seed component, the paired design still pairs — every method sees the same outer partition
at a given replication. Seeds are narrowed to 32 bits only where a third-party API validates that
range. Unit seeds derive from `SEED_BASE = 20260914`.

### 9.3 Runner, ledger and cache

Each of the 120 units executes 20 stages, each checkpointed as a JSON payload written to a
temporary file and atomically replaced, so an interrupted run leaves no half-written stage and a
resumed run never recomputes a completed one. Request rows carry an attempt epoch and only the
latest attempt per `(method, stage)` is counted, which makes a resumed campaign equivalent to an
uninterrupted one in the accounting as well as in the science.

Two ledgers are kept separately and both reported. The **logical** ledger charges every evaluation a
method requests, hit or miss; it is the scientific budget and the only quantity entering a fairness
comparison. The campaign consumed **396,120 logical evaluations**, exactly the declared registry total, every
one of the 120 units reconciling individually. The **physical** ledger counts misses only:
**377,316 unique physical fits**, a **4.75% cache hit rate** — an engineering quantity that enters
no comparison between methods.

Memoization sits beneath method isolation. The cache key pins dataset identity, outer split
identity, inner fold definition, the configuration after canonical type normalization (integers
rounded, floats fixed to twelve decimals, keys sorted), the training seed and the evaluation
protocol version; a change to any of them is a different evaluation. Each method reaches the cache
only through its own view, which exposes its own history and nothing else, so reuse never becomes
cross-method information.

### 9.4 Environment and how to re-run

The campaign manifest stamps the environment: Python 3.11.15 on macOS 26.6.2 (arm64), numpy 2.4.6,
pandas 3.0.5, and in the same environment xgboost 3.2.0, scikit-learn 1.9.0 and scipy 1.17.1.
Execution used 14 process workers, one XGBoost thread per fit, `spawn` start method, completing
120/120 units with 0 failures in 9h46m. Two provenance scripts that
execute the archived dissertation code require an interpreter contemporary with it (numpy < 2,
pandas < 3); they refuse to run on pandas 3 and print that recipe rather than patch a historical
artifact.

The campaign is driven by `scripts/xgb_hpo_campaign.py` (`plan`, `dry-run`, `run`, `status`);
`run` is resumable and idempotent over completed stages. Analysis is reproduced in order by
`aggregate_campaign.py`, `primary_analysis.py` and `secondary_analysis.py`, and the claims map by
`build_claims_map.py`, which reads every headline figure from a committed artifact at generation
time. Audit artifacts rebuild under `--check`, comparing against the committed versions without
writing.

Raw datasets and heavy evaluation caches are deliberately unversioned. The four panel datasets are
UCI or OpenML mirrors, obtainable independently and checksum-verified on acquisition; the 88-run
face-centred central composite design is versioned with its SHA-256. The per-unit experiment tree,
including one SQLite evaluation cache per unit, is not committed. Committed instead are the frozen
code, the protocol, the amendment ledger, the per-unit indicator table and the analysis artifacts —
enough to recompute every reported quantity without re-running the campaign.
