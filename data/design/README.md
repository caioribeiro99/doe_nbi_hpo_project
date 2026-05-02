# DOE design matrices

Versioned designs live here. They are *experimental designs*, not data,
and are the canonical artifacts that dictate every subsequent stage of
the pipeline.

## Files

- `hyperparameter_design.csv` — Minitab-generated face-centered CCD
  (CCDFC / CCFCD) over the seven XGBoost hyperparameters. 88 rows
  (corners + axial + 4 center points). Used by every dissertation and
  article-track replica.
- `hyperparameter_design.csv.metadata.json` — sidecar with generator,
  design type, factor bounds, SHA-256, and CSV format conventions
  (separator `;`, decimal `,`).

## Loading

```python
from doe_xgb.design import DesignProvider, DesignSpec, DesignKind, FactorMeta

spec = DesignSpec(
    kind=DesignKind.EXTERNAL_CSV,
    factors=(...,),  # see metadata JSON
    external_path=Path("data/design/hyperparameter_design.csv"),
)
artifact = DesignProvider.build(spec)
```

The provider validates the SHA-256, builds both coded and uncoded
matrices, and returns matrix diagnostics (rank, condition number,
coverage of the [-1, +1] box).
