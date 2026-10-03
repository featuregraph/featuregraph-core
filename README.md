# FeatureGraph Core

Code and data behind "A Compiler-Level Account of Smoothing-Parameter
Choice." Contains `OscillationConfig` (the construction used throughout
the paper), a BIDMC data loader, and the validation logic behind the
paper's human-annotation results.

This package reproduces the construction behind the smoothing paper's
results (BIDMC respiration data) and additionally supports the Hongn et
al. wearable stress-protocol dataset (heart rate, electrodermal activity,
and skin temperature), as an example of the same construction applied to
a second signal.

## Install

```bash
pip install -e .
```

## Reproduce the paper's results

```bash
pytest tests/ -v
```

Two tests are skipped by default, since they need a live download from
PhysioNet. Their skip reasons record the exact confirmed results;
remove the `@pytest.mark.skip` decorator above either one and rerun to
reproduce it directly:

- `test_reproduces_paper_correlation_and_ratio_range` — the paper's
  population correlation (0.39) and ratio range (1.14x–40.25x) across
  all 53 BIDMC subjects.
- `test_bidmc_peak_matched_recall_excluding_flagged_subjects` — the
  paper's human-annotation recall results (Table 1), N=32.

## Using OscillationConfig directly

```python
from featuregraph_smoothing_core.behaviors.oscillation import OscillationConfig

config = OscillationConfig(signal="respiration", smooth_window=100)
added = config.add_primitives(df, "subject")
summary = config.summarize(added, ["subject", config.trough_event_id_col])
```

## What's here

- `src/featuregraph_smoothing_core/` — the package
- `notebooks/bidmc_visual_demo.ipynb` — the code that produced Figure 1
- `artifacts/paper/compiler/smoothing.md` — the paper manuscript
- `tests/` — the full test suite

### Wearable dataset

Heart rate data loaded by this package comes from:

> Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). Wearable Device
> Dataset from Induced Stress and Structured Exercise Sessions (version
> 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17

See also the associated paper:

> Hongn, A., Bosch, F., & Prado, L. (2025). Wearable Physiological Signals
> under Acute Stress and Exercise Conditions. Scientific Data.
> https://doi.org/10.1038/s41597-025-04845-9

Licensed under CC BY 4.0. This package does not redistribute the raw
dataset; the loader fetches it directly from PhysioNet.

### BIDMC dataset

Respiration data loaded by this package comes from:

> Pimentel, M. A. F., Johnson, A. E. W., Charlton, P. H., Birrenkott, D.,
> Watkinson, P. J., Tarassenko, L., & Clifton, D. A. (2016). Toward a
> Robust Estimation of Respiratory Rate From Pulse Oximeters. IEEE
> Transactions on Biomedical Engineering, 64(8), 1914-1923.

> BIDMC dataset (version 1.0.0). PhysioNet.

This package does not redistribute the raw dataset; the loader fetches it
directly from PhysioNet.

## Citation

Software: https://doi.org/10.5281/zenodo.22947447

## License

MIT. See `LICENSE`.
