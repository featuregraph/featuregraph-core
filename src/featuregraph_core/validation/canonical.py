"""
The private repo's ground-truth validation notebook fetched "canonical"
detected objects (the ones built with each subject's own ACF-recommended
window) from a Postgres table that cached an earlier computation. That
table is just a cache -- the same objects can be recomputed directly,
identically, using characterize_signal() and OscillationConfig(), both
of which are public in this package. This module does that recomputation
so the ground-truth validation doesn't depend on the private storage
layer at all.
"""

from featuregraph_core.behaviors.oscillation import OscillationConfig
from featuregraph_core.validation.characterize import characterize_signal


def compute_canonical_indices(df, signal, group, index_kind='peak', min_correlation=0.6):
    """
    For each group (e.g. subject or case_id), estimate a smoothing
    window via ACF-based characterization and, if confident, construct
    Oscillation objects using that group's own recommended window.

    Mirrors the private repo's canonical_bidmc_troughs / canonical_bidmc_peaks
    / canonical_capno_troughs functions, but recomputes directly instead
    of reading a cached Postgres table -- the result is the same, since
    those functions cached exactly this computation.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations for all groups (e.g. all 53 BIDMC subjects).
    signal : str
        Name of the observed signal column (e.g. "respiration").
    group : str
        Name of the grouping column (e.g. "subject" or "case_id").
    index_kind : "peak" or "trough"
        Which boundary index to return.
    min_correlation : float
        Passed through to characterize_signal.

    Returns
    -------
    (indices_by_group, characterization)
        indices_by_group : dict mapping each confident group's value to
            a sorted list of int sample indices.
        characterization : the full characterize_signal() output, so
            callers can inspect is_confident / estimated_period / etc.
            for every group, not only the confident ones.
    """
    if index_kind not in ('peak', 'trough'):
        raise ValueError("index_kind must be 'peak' or 'trough'")

    characterization = characterize_signal(df, signal, group, min_correlation=min_correlation)

    indices_by_group = {}
    for group_value, row in characterization.iterrows():
        if not row['is_confident']:
            continue

        window = int(row['window_medium'])
        subj_df = df[df[group] == group_value].copy()

        config = OscillationConfig(signal=signal, smooth_window=window)
        added = config.add_primitives(subj_df, group)

        # summarize() gives one row per actual oscillation object -- the
        # add_primitives() output has one row per SAMPLE, with peak/trough
        # index forward-filled across every row between events, so reading
        # straight from added would count each sample between two peaks
        # as if it were its own detected peak.
        summarized = config.summarize(added, [group, config.trough_event_id_col])
        index_col = 'peak_index' if index_kind == 'peak' else 'trough_index'
        indices_by_group[group_value] = sorted(
            summarized[index_col].dropna().astype(int).tolist()
        )

    return indices_by_group, characterization
