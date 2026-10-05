# featuregraph-core

A Python library for constructing explicit behavioral objects (events and
intervals such as breaths) from time-series signals, using deterministic
rise/fall state detection. The same code runs on any signal; the choices
that matter, such as the smoothing window, are parameters you declare.

**Status: early (0.x).** The interface may change between releases.

`featuregraph-core` is the successor to `featuregraph-smoothing-core`. The
import name changed from `featuregraph_smoothing_core` to
`featuregraph_core`, and 0.1.0 contains the same code as
`featuregraph-smoothing-core` 1.1.0.

## Install

```bash
pip install featuregraph-core
```

For development, clone this repository and run `pip install -e .`.
Python 3.9 or later.

## Quick start

```python
from featuregraph_core.behaviors.oscillation import OscillationConfig
from featuregraph_core.datasets import bidmc

df = bidmc(subject=1)  # downloads from PhysioNet on first use

config = OscillationConfig(signal="respiration", smooth_window=100)
added = config.add_primitives(df, "subject")
summary = config.summarize(added, ["subject", config.trough_event_id_col]).reset_index()

complete = summary[summary["is_complete"]]  # drop partial objects at the recording edges
```

`smooth_window` is a number of samples, not seconds, and you choose it. It
changes what gets detected: in the paper below, the same construction at two
different windows gave object counts that correlated at only 0.39 across
subjects. Convert to seconds using your signal's sampling rate (125 Hz for
BIDMC).

## Datasets

Loaders fetch data directly from PhysioNet; this package does not
redistribute any raw data.

- `bidmc(subject=1)`: respiration (BIDMC dataset)
- `wearable_hr(subject="S01")`: heart rate (Hongn et al. wearable dataset)

### BIDMC dataset

> Pimentel, M. A. F., Johnson, A. E. W., Charlton, P. H., Birrenkott, D.,
> Watkinson, P. J., Tarassenko, L., & Clifton, D. A. (2016). Toward a
> Robust Estimation of Respiratory Rate From Pulse Oximeters. IEEE
> Transactions on Biomedical Engineering, 64(8), 1914-1923.

BIDMC dataset (version 1.0.0). PhysioNet.

### Wearable dataset

> Hongn, A., Bosch, F., Prado, L., & Bonomini, P. (2025). Wearable Device
> Dataset from Induced Stress and Structured Exercise Sessions (version
> 1.0.1). PhysioNet. https://doi.org/10.13026/he0v-tf17

See also the associated paper:

> Hongn, A., Bosch, F., & Prado, L. (2025). Wearable Physiological Signals
> under Acute Stress and Exercise Conditions. Scientific Data.
> https://doi.org/10.1038/s41597-025-04845-9

The wearable dataset is licensed CC BY 4.0.

## Validation module

`featuregraph_core.validation` holds the period-estimation and matching
code behind the paper's human-annotation results. It is the paper-era
version and was tuned on BIDMC (125 Hz respiration). On other signals,
check the sampling rate and the `at_boundary` flag before trusting a
reported period: on slowly sampled or smooth signals the search can stop at
its minimum lag and report that floor as the period.

## Reproducing the paper

The paper is "A Compiler-Level Account of Smoothing-Parameter Choice"
([manuscript](https://github.com/featuregraph/featuregraph-core/blob/main/artifacts/paper/compiler/smoothing.md)).
The exact code behind its published results is the archived `v1.0.0`
release of `featuregraph-smoothing-core`:

- Zenodo archive: https://doi.org/10.5281/zenodo.22947447
- PyPI: `pip install featuregraph-smoothing-core==1.0.0`
- Git tag: `v1.0.0`

Use one of those for a guaranteed reproduction. The tests in this
repository run the same checks against the current code:

```bash
pytest tests/ -v
```

Two tests are skipped by default because they need a live download from
PhysioNet. Their skip reasons record the confirmed results; remove the
`@pytest.mark.skip` decorator and rerun to reproduce them directly:

- `test_reproduces_paper_correlation_and_ratio_range`: the paper's
  population correlation (0.39) and ratio range (1.14x to 40.25x) across all
  53 BIDMC subjects.
- `test_bidmc_peak_matched_recall_excluding_flagged_subjects`: the paper's
  human-annotation results (Table 1), N=32.

## Repository layout

- `src/featuregraph_core/`: the package
- `notebooks/`: demo notebooks (`bidmc_visual_demo.ipynb` produced Figure 1)
- `artifacts/paper/compiler/smoothing.md`: the paper manuscript
- `tests/`: the test suite

## Citing

To cite the paper's results, cite the archived release:
https://doi.org/10.5281/zenodo.22947447

## License

MIT. See [LICENSE](https://github.com/featuregraph/featuregraph-core/blob/main/LICENSE).
Datasets are covered by their own licenses and citation requirements.
