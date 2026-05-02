# Dissertation example: XGBoost on UCI MAGIC

Minimal working example for the dissertation-baseline pipeline.

## Steps

```bash
# 1. Install the package and dev tools.
make install-dev

# 2. Download the MAGIC dataset.
make data

# 3. Validate the YAML config.
python -m doe_xgb.cli validate --config configs/dissertation_baseline_xgb_magic.yaml

# 4. Run the legacy orchestration script (still the preferred entry
#    point for full 30-replica reproductions on the publication branch
#    until the new orchestration module lands).
python scripts/run_experiment.py \
  --dataset data/source/magic.csv \
  --design  data/design/hyperparameter_design.csv \
  --n-replicas 30 \
  --seed-base 42
```

## Article-track config

For the article-track 3-VRF run, swap the YAML:

```bash
python -m doe_xgb.cli validate --config configs/article_3vrf_xgb_magic.yaml
```

Until the v0.3 orchestration module lands, the heavy execution still
flows through `scripts/run_replica.py`. The new modules
(`doe_xgb.objectives`, `doe_xgb.factor_model`, `doe_xgb.nbi_core`,
`doe_xgb.post_optimization`, ...) are exercised individually via the
`tests/` suite and the `doe-xgb smoke` CLI.

## See also

- `docs/METHOD.md`
- `docs/REPRODUCIBILITY.md`
- `docs/ARTIFACT_GUIDE.md`
- `docs/METHODOLOGY_DECISIONS.md`
