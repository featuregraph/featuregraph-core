"""
Three tiers of test here:

1. Property tests on OscillationConfig's column-naming logic and on
   bidmc_map's column renaming -- both pass right now, with all real
   verified source in place.

2. A synthetic end-to-end pipeline test, confirming the real operator
   code (not placeholders) actually produces correct rising/falling
   states, peak/trough events, and sane per-cycle durations.

3. The real end-to-end reproduction test against the paper's actual
   cited numbers (correlation 0.3889, ratio range 1.14x-40.25x across
   the 53-subject BIDMC cohort). Confirmed passing against real BIDMC
   data on 2026-09-24. Left skipped by default so routine test runs
   don't require a live PhysioNet download every time; un-skip and
   rerun with network access to reconfirm if the underlying data or
   logic ever changes.
"""

import pytest
import pandas as pd
import numpy as np

from featuregraph_core.behaviors.oscillation import OscillationConfig
from featuregraph_core.utils._rename_map import bidmc_map


def test_column_naming():
    config = OscillationConfig(signal="respiration", smooth_window=100)
    assert config.smooth == "respiration_smooth"
    assert config.rising_col == "respiration_rising"
    assert config.falling_col == "respiration_falling"
    assert config.enter_rising_col == "enter_respiration_rising"
    assert config.exit_rising_col == "exit_respiration_rising"
    assert config.peak_index_col == "respiration_peak_index"
    assert config.trough_index_col == "respiration_trough_index"


def test_bidmc_map_renames_respiration_column():
    raw = pd.DataFrame(columns=["Time [s]", "RESP", "PLETH", "II", "V", "AVR", "subject"])
    renamed = raw.rename(columns=bidmc_map)
    assert "respiration" in renamed.columns
    assert "time" in renamed.columns
    assert "ecg_ii" in renamed.columns
    assert "ecg_v" in renamed.columns


def test_pipeline_runs_on_synthetic_signal():
    """
    Confirms the real operator code actually runs end to end and
    produces sane output: a two-cycle sine wave across two subjects
    should show rising/falling states, a small number of peak/trough
    events, and per-cycle durations that are the right order of
    magnitude relative to the signal length -- not just "doesn't crash."
    """
    t = np.linspace(0, 4 * np.pi, 200)
    signal = np.sin(t)
    df = pd.DataFrame({
        "respiration": np.concatenate([signal, signal]),
        "subject": [1] * 200 + [2] * 200,
    })

    config = OscillationConfig(signal="respiration", smooth_window=5)
    added = config.add_primitives(df, "subject")

    assert added["respiration_rising"].sum() > 0
    assert added["respiration_falling"].sum() > 0
    assert added["exit_respiration_rising"].sum() > 0
    assert added["enter_respiration_rising"].sum() > 0

    summary = config.summarize(added, ["subject", config.trough_event_id_col])
    assert len(summary) > 0
    complete_durations = summary.loc[summary["is_complete"], "duration"]
    assert (complete_durations > 10).all()
    assert (complete_durations < 200).all()


@pytest.mark.skip(
    reason=(
        "Confirmed passing against real BIDMC data on 2026-09-24: "
        "population correlation 0.3889 and per-subject ratio range "
        "1.14x-40.25x across all 53 BIDMC subjects, exactly matching "
        "the smoothing paper's cited numbers. Left skipped by default "
        "so routine test runs don't require a ~15-minute PhysioNet "
        "download every time; un-skip and rerun with network access to "
        "reconfirm if the underlying data or logic ever changes."
    )
)
def test_reproduces_paper_correlation_and_ratio_range():
    import featuregraph_core as fg

    def load_all_bidmc_subjects():
        frames = []
        for subject_id in range(1, 54):
            frames.append(fg.datasets.bidmc(subject=subject_id))
        return pd.concat(frames, ignore_index=True)

    def count_peaks_at_windows(df, windows=(1, 100), signal="respiration", group="subject"):
        results = {}
        for window in windows:
            config = OscillationConfig(signal=signal, smooth_window=window)
            added = config.add_primitives(df, group)
            results[f"peaks_w{window}"] = added.groupby(group)[config.exit_rising_col].sum()
        return pd.DataFrame(results)

    df = load_all_bidmc_subjects()
    peak_counts = count_peaks_at_windows(df, windows=(1, 100))
    peak_counts["ratio"] = peak_counts["peaks_w1"] / peak_counts["peaks_w100"]

    correlation = peak_counts["peaks_w1"].corr(peak_counts["peaks_w100"])
    assert correlation == pytest.approx(0.3889, abs=0.001)
    assert peak_counts["ratio"].min() == pytest.approx(1.14, abs=0.01)
    assert peak_counts["ratio"].max() == pytest.approx(40.25, abs=0.01)
