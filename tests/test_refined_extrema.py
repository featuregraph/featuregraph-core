import numpy as np
import pandas as pd
import pytest
from featuregraph_core.behaviors.oscillation import OscillationConfig


def triangular():
    t = np.arange(2000) % 200 / 200
    return np.where(t <= .35, -1+2*t/.35, 1-2*(t-.35)/.65)


def test_refinement_preserves_original_and_recovers_symmetry():
    df = pd.DataFrame({"rec": 0, "sig": triangular()})
    cfg = OscillationConfig(signal="sig", smooth_window=50)
    s = cfg.summarize(cfg.add_primitives(df, "rec"), ["rec", cfg.trough_event_id_col])
    s = s[s.is_complete & (s.peak_index >= 250) & (s.peak_index < 1750)]
    assert len(s) >= 6
    r = cfg.refine_temporal(df, s, "rec")
    pd.testing.assert_frame_equal(r[s.columns], s)
    np.testing.assert_allclose(r.refined_temporal_symmetry, .7, atol=.01)
    assert (s.temporal_symmetry - .7).abs().mean() > .08
    np.testing.assert_array_equal(r.refined_end_index.iloc[:-1], r.refined_start_index.iloc[1:])
    assert (r.refined_rising_duration > 0).all()
    assert (r.refined_falling_duration > 0).all()


def test_refinement_handles_group_offsets_and_invalid_cycles():
    x = triangular()
    df = pd.DataFrame({"rec": np.repeat([0, 1], len(x)), "sig": np.tile(x, 2)})
    cfg = OscillationConfig(signal="sig", smooth_window=50)
    s = cfg.summarize(cfg.add_primitives(df, "rec"), ["rec", cfg.trough_event_id_col])
    s = s[s.is_complete & ((s.peak_index % len(x)) >= 250) &
          ((s.peak_index % len(x)) < 1750)].copy()
    assert len(s) >= 12
    s.iloc[0, s.columns.get_loc("peak_index")] = np.nan
    r = cfg.refine_temporal(df, s, "rec")
    pd.testing.assert_frame_equal(r[s.columns], s)
    assert pd.isna(r.refined_temporal_symmetry.iloc[0])
    np.testing.assert_allclose(r.refined_temporal_symmetry.iloc[1:], .7, atol=.01)
    assert (r.loc[1].refined_start_index >= len(x)).all()
    # A nonfinite raw search window must not yield a fabricated extremum.
    bad = df.copy()
    bad.loc[int(s.iloc[1].peak_index)-1, "sig"] = np.nan
    assert pd.isna(cfg.refine_temporal(bad, s, "rec").refined_temporal_symmetry.iloc[1])


def test_refinement_requires_unique_ordered_labels():
    df = pd.DataFrame({"rec": 0, "sig": triangular()})
    cfg = OscillationConfig(signal="sig", smooth_window=50)
    s = cfg.summarize(cfg.add_primitives(df, "rec"), ["rec", cfg.trough_event_id_col])
    s = s[s.is_complete]
    with pytest.raises(ValueError, match="increasing"):
        cfg.refine_temporal(df.iloc[::-1], s, "rec")
