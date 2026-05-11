# Treeffuser Benchmarks

This directory contains lightweight, implementation-focused benchmarks for comparing
Treeffuser variants. It is separate from `testbed/`: `testbed/` is for broad model
comparisons, while this harness is for paired diagnostics during Treeffuser development.

The benchmark grain is one result row per:

```text
dataset x seed x variant x sampler
```

This keeps comparisons paired and makes it possible to ask whether a variant improves
coverage or CRPS without hiding the cost in wider intervals, slower sampling, or more
training rows.

## Layout

```text
benchmarks/
  run.py
  harness.py
  variants.py
  datasets.py
  metrics.py
  configs/
    smoke.yaml
    synthetic_core.yaml
    real_smoke.yaml
  results/raw/
```

## Running

The runner uses PyYAML if available and otherwise falls back to a small parser for the
simple YAML subset used by these configs:

```bash
python -m benchmarks.run --config benchmarks/configs/smoke.yaml
```

By default, results are written to `benchmarks/results/raw/<config>_<timestamp>.csv`.
The `real_smoke.yaml` config uses local datasets bundled with scikit-learn, so it does
not download external benchmark data.

## Seeding Policy

For each dataset/seed pair, the harness derives and records three seeds:

- `data_seed`: controls data generation and paired train/test splits.
- `model_seed`: controls model training randomness.
- `sampler_seed`: controls Monte Carlo sampling randomness.

The same resolved `model_seed` and `sampler_seed` are used across variants for the same
dataset/seed pair unless the config explicitly changes the offsets. This gives paired
comparisons a stable stochastic contract.

## Provenance

Every result row records:

- `git_sha`
- `git_dirty`
- `treeffuser_source_hash`

Because `baseline_current` means "the current baseline behavior in this checkout", these
columns are required to interpret old-vs-new comparisons later.
