"""Offline, sample-position-based refinement of selected cycle anchors."""
import numpy as np
import pandas as pd


def refine_cycle_anchors(df, summary, *, signal, group, radius):
    """Return a copy with refined indices, durations and symmetry appended.

    The original event labels denote the first sample after a turning point.
    Search around that sample minus one, clipping at adjacent anchor midpoints.
    A missing or invalid anchor leaves only the refined measurements undefined.
    """
    if not isinstance(radius, (int, np.integer)) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if not isinstance(summary.index, pd.MultiIndex) or summary.index.names[0] != group:
        raise ValueError("summary must be indexed by recording then cycle")
    result = summary.copy()
    names = ["refined_start_index", "refined_peak_index", "refined_end_index",
             "refined_rising_duration", "refined_falling_duration",
             "refined_temporal_symmetry"]
    # Index labels may be nonconsecutive; use row positions for durations.
    output = np.full((len(summary), len(names)), np.nan)
    for key, positions in summary.groupby(level=0, sort=False).indices.items():
        samples = df.loc[df[group].eq(key), signal]
        if not samples.index.is_unique or not pd.api.types.is_numeric_dtype(samples.index.dtype):
            raise ValueError("sample indices must be unique numeric labels per recording")
        if not samples.index.is_monotonic_increasing:
            raise ValueError("sample indices must be increasing within recording")
        x = samples.to_numpy(dtype=float)
        cycles = summary.iloc[positions]
        anchors = set()
        valid = []
        for pos, (_, row) in zip(positions, cycles.iterrows()):
            labels = [row.start_index, row.peak_index, row.end_index]
            if not row.is_complete or any(pd.isna(v) for v in labels):
                continue
            a, p, b = samples.index.get_indexer(labels)
            if not 0 < a < p < b:
                continue
            keys = ((a, False), (p, True), (b, False))
            anchors.update(keys)
            valid.append((pos, keys))
        anchors = sorted(anchors)
        refined = {}
        for j, (event, peak) in enumerate(anchors):
            a = event - 1
            prev = anchors[j-1][0]-1 if j else None
            nxt = anchors[j+1][0]-1 if j+1 < len(anchors) else None
            lo = max(0, a-radius, (prev+a)//2+1 if prev is not None else 0)
            hi = min(len(x)-1, a+radius, (a+nxt)//2 if nxt is not None else len(x)-1)
            if hi < lo or not np.isfinite(x[lo:hi+1]).all():
                continue
            segment = x[lo:hi+1]
            refined[event, peak] = lo + int(np.argmax(segment) if peak else np.argmin(segment))
        for pos, keys in valid:
            a, p, b = (refined.get(k) for k in keys)
            if None in (a, p, b) or not a < p < b:
                continue
            rise, fall = p-a, b-p
            output[pos] = [samples.index[a], samples.index[p], samples.index[b],
                           rise, fall, 1-abs(rise-fall)/(rise+fall)]
    for j, name in enumerate(names):
        result[name] = output[:, j]
    return result
