import numpy as np
import pandas as pd


def estimate_period_acf(signal, max_lag=None, min_lag=100):
    """
    Estimate the dominant period of a signal using autocorrelation,
    computed directly on the raw signal with no smoothing window involved.
    Returns (peak_lag, peak_correlation, at_boundary).
    at_boundary=True means the search never found a real local peak --
    the max was just the start of the region, i.e. the curve was still
    decaying from lag 0 rather than showing genuine periodic recurrence.
    """
    x = signal.dropna().to_numpy()
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0]

    search_region = acf[min_lag:max_lag]
    if len(search_region) == 0:
        return None, None, True

    peak_offset = np.argmax(search_region)
    peak_lag = peak_offset + min_lag
    peak_corr = acf[peak_lag]
    at_boundary = (peak_offset == 0)

    return peak_lag, peak_corr, at_boundary


def characterize_signal(df, signal, group, min_correlation=0.6):
    """
    Estimate each group's dominant period directly from the raw signal
    via autocorrelation. is_confident requires both a sufficiently strong
    peak correlation AND a genuine local peak (not just the search
    boundary), since a boundary hit means no real periodicity was found.
    """
    results = []
    for subj, subj_df in df.groupby(group):
        period, corr, at_boundary = estimate_period_acf(subj_df[signal])
        is_confident = (
            period is not None
            and corr is not None
            and corr >= min_correlation
            and not at_boundary
        )
        results.append({
            group: subj,
            'estimated_period': period,
            'acf_peak_correlation': corr,
            'at_boundary': at_boundary,
            'is_confident': is_confident,
            'window_small': max(int(round(period * 0.25)), 2) if is_confident else None,
            'window_medium': max(int(round(period * 0.5)), 2) if is_confident else None,
        })
    return pd.DataFrame(results).set_index(group)
