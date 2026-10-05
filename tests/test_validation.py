"""
Tests for the ground-truth validation module (characterize_signal,
match_breaths, compute_canonical_indices).

Three tiers, same pattern as test_oscillation.py:

1. Synthetic tests that pass right now, with no network access, proving
   the transcribed logic actually works -- not just that it imports.
2. A synthetic phase-mismatch test, specifically chosen to reproduce
   the shape of the real finding (BIDMC annotators mark peaks, not
   troughs -- matching against the wrong one gives near-zero recall).
3. The real end-to-end computation against actual BIDMC data. Confirmed
   passing on 2026-09-24 against N=32 subjects (53 minus 16 with no
   reliable ACF period, minus 5 flagged, with some overlap between those
   two groups) -- see that test's skip reason for the full figures.
   Left skipped by default so routine runs don't require a real
   PhysioNet download every time.
"""

import pytest
import numpy as np
import pandas as pd

import featuregraph_core as fg
from featuregraph_core.validation.exclusions import FLAGGED_ANOMALY_SUBJECTS


def _make_synthetic_cohort(subjects=(1, 2, 3), n_samples=20000, period_samples=366, noise=0.02, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    for subject in subjects:
        t = np.arange(n_samples) / period_samples * 2 * np.pi
        frames.append(pd.DataFrame({
            'respiration': np.sin(t) + noise * rng.randn(n_samples),
            'subject': subject,
        }))
    return pd.concat(frames, ignore_index=True)


def test_characterize_signal_recovers_known_period():
    df = _make_synthetic_cohort()
    characterization = fg.validation.characterize_signal(df, 'respiration', 'subject')
    for subject in (1, 2, 3):
        assert characterization.loc[subject, 'is_confident']
        assert abs(characterization.loc[subject, 'estimated_period'] - 366) < 5


def test_compute_canonical_indices_is_database_free_and_correct():
    df = _make_synthetic_cohort()
    peak_indices, _ = fg.validation.compute_canonical_indices(
        df, 'respiration', 'subject', index_kind='peak'
    )
    for subject in (1, 2, 3):
        n_peaks = len(peak_indices[subject])
        assert abs(n_peaks - (20000 / 366)) < 3


def test_match_breaths_detects_phase_mismatch():
    """
    Reproduces the shape of the real finding: BIDMC annotators mark the
    peak phase, not the trough. Comparing detected peaks against a set
    shifted by half a period should show near-zero recall at a tight
    tolerance, exactly like the real trough-vs-peak mismatch did.
    """
    df = _make_synthetic_cohort(subjects=(1,))
    peak_indices, _ = fg.validation.compute_canonical_indices(
        df, 'respiration', 'subject', index_kind='peak'
    )
    detected = peak_indices[1]

    result_self = fg.validation.match_breaths(detected, detected, tolerance_samples=5)
    assert result_self['recall'] == 1.0
    assert result_self['precision'] == 1.0

    shifted = [i + 183 for i in detected]  # half the period (366/2)
    result_shifted = fg.validation.match_breaths(detected, shifted, tolerance_samples=5)
    assert result_shifted['recall'] < 0.1


@pytest.mark.skip(
    reason=(
        "Confirmed passing against real BIDMC data on 2026-09-24: N=32 "
        "subjects (53 minus 16 with no reliable ACF period, minus 5 "
        "flagged, with some overlap between those two groups), peak-"
        "matched recall of 84.2%/80.6% (0.25s), 94.4%/96.2% (0.5s), and "
        "95.8%/99.1% (1.0s) against annotators 1/2, against an inter-"
        "annotator ceiling of 89.6%/95.9%/97.9%. Left skipped by default "
        "so routine test runs don't require a ~15-minute PhysioNet "
        "download every time; un-skip and rerun with network access to "
        "reconfirm if the underlying data or logic ever changes."
    )
)
def test_bidmc_peak_matched_recall_excluding_flagged_subjects():
    BIDMC_SAMPLING_RATE = 125
    TOLERANCE_SECONDS = [0.25, 0.5, 1.0]
    ANNOTATOR_1_COLUMN = 'breaths ann1 [signal sample no]'
    ANNOTATOR_2_COLUMN = 'breaths ann2 [signal sample no]'

    # Compute peak indices per subject, on that subject's own individually
    # loaded dataframe -- NOT on a pre-concatenated cohort. bidmc_breaths()
    # gives sample indices local to each recording (starting at 0 every
    # time), so concatenating first would compare each subject's peaks
    # against a global row position that has no relationship to the
    # per-recording sample numbers the annotations use. An earlier version
    # of this test had exactly this bug: every subject after the first
    # was being compared against the wrong index scale, producing a
    # frozen, meaningless ~3% recall regardless of tolerance.
    peak_indices = {}
    for subject_id in range(1, 54):
        subj_df = fg.datasets.bidmc(subject=subject_id)
        indices, _ = fg.validation.compute_canonical_indices(
            subj_df, 'respiration', 'subject', index_kind='peak'
        )
        if subject_id in indices:
            peak_indices[subject_id] = indices[subject_id]

    algorithm_results = []
    inter_annotator_results = []
    for subject in range(1, 54):
        if subject in FLAGGED_ANOMALY_SUBJECTS:
            continue
        if subject not in peak_indices:
            continue  # not confident -- same exclusion already applied elsewhere

        breaths = fg.datasets.bidmc_breaths(subject)
        ann1 = breaths[ANNOTATOR_1_COLUMN].dropna().astype(int).tolist()
        ann2 = breaths[ANNOTATOR_2_COLUMN].dropna().astype(int).tolist()
        detected = peak_indices[subject]

        for tol_sec in TOLERANCE_SECONDS:
            tol_samples = tol_sec * BIDMC_SAMPLING_RATE

            for annotator_name, expert in [('annotator_1', ann1), ('annotator_2', ann2)]:
                result = fg.validation.match_breaths(detected, expert, tol_samples)
                result['subject'] = subject
                result['tolerance_seconds'] = tol_sec
                result['annotator'] = annotator_name
                algorithm_results.append(result)

            inter = fg.validation.match_breaths(ann2, ann1, tol_samples)
            inter['subject'] = subject
            inter['tolerance_seconds'] = tol_sec
            inter_annotator_results.append(inter)

    algorithm_df = pd.DataFrame(algorithm_results)
    inter_annotator_df = pd.DataFrame(inter_annotator_results)

    n_subjects = algorithm_df['subject'].nunique()
    print(f"\nN subjects in reduced population (confident, not flagged): {n_subjects}")

    algorithm_summary = algorithm_df.groupby(['tolerance_seconds', 'annotator'])[['recall', 'precision']].mean()
    inter_annotator_summary = inter_annotator_df.groupby('tolerance_seconds')[['recall', 'precision']].mean()

    print("\nAlgorithm vs. each annotator, excluding flagged subjects:")
    print(algorithm_summary)
    print("\nInter-annotator ceiling, excluding flagged subjects:")
    print(inter_annotator_summary)

    # Confirmed against real BIDMC data on 2026-09-24 -- see the skip
    # reason above for the full figures. abs=0.005 tolerance accounts
    # for ordinary floating-point/pandas-version variation, not for
    # genuine disagreement in the underlying result.
    assert n_subjects == 32

    assert algorithm_summary.loc[(0.25, 'annotator_1'), 'recall'] == pytest.approx(0.842287, abs=0.005)
    assert algorithm_summary.loc[(0.25, 'annotator_2'), 'recall'] == pytest.approx(0.805585, abs=0.005)
    assert algorithm_summary.loc[(0.50, 'annotator_1'), 'recall'] == pytest.approx(0.943977, abs=0.005)
    assert algorithm_summary.loc[(0.50, 'annotator_2'), 'recall'] == pytest.approx(0.961998, abs=0.005)
    assert algorithm_summary.loc[(1.00, 'annotator_1'), 'recall'] == pytest.approx(0.958220, abs=0.005)
    assert algorithm_summary.loc[(1.00, 'annotator_2'), 'recall'] == pytest.approx(0.990872, abs=0.005)

    assert inter_annotator_summary.loc[0.25, 'recall'] == pytest.approx(0.896119, abs=0.005)
    assert inter_annotator_summary.loc[0.50, 'recall'] == pytest.approx(0.959052, abs=0.005)
    assert inter_annotator_summary.loc[1.00, 'recall'] == pytest.approx(0.979238, abs=0.005)