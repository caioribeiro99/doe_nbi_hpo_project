# Pilot Stage A artifacts

Every file here is a real evaluation on the real objectives: 376 KB in total, 1,464 evaluations,
about 34 minutes of compute. Findings are in `../PILOT_STAGE_A_FINDINGS.md`; the screening verdict is
in `stage_a_report.json`.

| File | Rows | What it is |
|---|---|---|
| `<ds>_design.csv` | 88 | the version-controlled face-centred central composite design, evaluated on one partition |
| `<ds>_validation_complement.csv` | 78 | **the protocol's external set**: the design's complementary half fraction (64 corners, sign product −1) plus 14 axial runs at half the axial distance |
| `<ds>_validation_spanning.csv` | 100 | a rejected construction: half Latin hypercube, half arcsine-marginal |
| `<ds>_validation_uniform.csv` | 100 | the originally specified construction: a plain Latin hypercube, inherited from Paper 1 |
| `<ds>_loadings.csv` | 7 | Varimax-rotated scaled loadings from the design |
| `stage_a_report.json` | — | screening measurements, factor-stage summary and measured per-evaluation cost |

The two rejected external sets are kept because the comparison between the three is the evidence for
Finding 4, which changed a protocol parameter inherited from Paper 1. The same surfaces score:

| Dataset | uniform R²_quality | spanning R²_quality | complement R²_quality |
|---|---|---|---|
| MAGIC | +0.461 | +0.468 | **+0.775** |
| Spambase | −23.99 | −12.74 | **+0.953** |
| Adult | −2.68 | −0.92 | **+0.911** |
| Bank Marketing | −0.89 | −0.71 | **+0.917** |

Reproduce:

```
python ../../scripts/pilot_stage_a_screening.py                      # complement, the default
python ../../scripts/pilot_stage_a_screening.py --external-set uniform
python ../../scripts/pilot_stage_a_screening.py --reuse              # recompute from cache
```

`--reuse` caches the design and the external set independently, so changing the construction does not
re-evaluate the design.
