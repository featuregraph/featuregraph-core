"""Shared simulation and information-access contract."""
import numpy as np
import pandas as pd
from featuregraph_core.behaviors.oscillation import OscillationConfig

FS = 50
T = 600
WIN = 500
HALO = 250
SMOOTH = 50
DEV_SEEDS = tuple(range(10))
TEST_SEEDS = tuple(range(100, 110))
FEATURES = ["period_s", "amplitude_raw", "rising_duration", "temporal_symmetry"]

def simulate(seed):
    rng = np.random.default_rng(seed)
    t = np.arange(T * FS) / FS
    offsets = np.zeros(5) if seed == 0 else rng.uniform(0, 2*np.pi, 5)
    period = 4 + np.sin(2*np.pi*t/170 + offsets[0]) + .3*np.sin(2*np.pi*t/41 + offsets[1])
    phase = 2*np.pi*np.cumsum(1/period)/FS
    amp = 1 + .45*np.sin(2*np.pi*t/97 + 1 + offsets[2]) + .15*np.sin(2*np.pi*t/23 + offsets[3])
    drift = .35*np.sin(2*np.pi*t/130 + .5 + offsets[4])
    signal = amp*np.sin(phase) + drift + rng.normal(0, .22, len(t))
    peaks = np.flatnonzero(np.diff(np.floor((phase-np.pi/2)/(2*np.pi))) > 0)+1
    return signal, peaks, amp[peaks]

def objects(x):
    cfg = OscillationConfig(signal="sig", smooth_window=SMOOTH)
    df = pd.DataFrame({"rec": 0, "sig": x})
    added = cfg.add_primitives(df, "rec")
    s = cfg.summarize(added, ["rec", cfg.trough_event_id_col]).reset_index()
    s = s[s["is_complete"] & s["period"].notna()].copy()
    s["period_s"] = s["period"]/FS
    return s[(s.peak_index >= HALO) & (s.peak_index < HALO+WIN)]

def examples(seed, stride=WIN):
    signal, peaks, amps = simulate(seed)
    for start in range(HALO, len(signal)-WIN-HALO+1, stride):
        x = signal[start-HALO:start+WIN+HALO].copy()
        assert len(x) == WIN+2*HALO
        mask = (peaks[1:] >= start) & (peaks[1:] < start+WIN)
        y = np.array([np.diff(peaks)[mask].mean()/FS, amps[1:][mask].mean()]) if mask.any() else np.full(2,np.nan)
        o = objects(x)
        f = o[FEATURES].mean().to_numpy(dtype=float)
        yield dict(seed=seed, start=start, x=x, y=y, f=f,
                   detected_peaks=o.peak_index.to_numpy()+start-HALO,
                   true_peaks=peaks[(peaks>=start)&(peaks<start+WIN)])
