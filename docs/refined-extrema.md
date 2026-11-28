# Refining oscillation measurements

The optional refinement policy separates behavior identification from temporal measurement. Existing smoothing, state construction, cycle membership and original measurement columns are unchanged.

```python
from featuregraph_core.behaviors.oscillation import OscillationConfig

cfg = OscillationConfig(signal="sig", smooth_window=50)
samples = cfg.add_primitives(df, "rec")
cycles = cfg.summarize(samples, ["rec", cfg.trough_event_id_col])
cycles = cycles[cycles.is_complete]
refined = cfg.refine_temporal(df, cycles, "rec")
```

Filter to the desired cycle population before refinement (for example, central observation intervals). The anchor set is determined by this selection. Retain complete cycles even when an unrelated measurement such as peak-to-peak period is missing.

For each distinct selected event, search the original signal around the event position minus one, with radius max(1, smooth_window // 2) samples. Clip the interval at midpoints of adjacent selected anchors. Use the first maximum or minimum in the interval; share trough anchors between adjacent cycles. Require strictly ordered trough–peak–trough positions. Invalid or nonfinite search windows leave affected refined measurements undefined, without dropping the cycle.

Outputs: refined_start_index, refined_peak_index, refined_end_index (original sample labels), refined_rising_duration, refined_falling_duration (sample counts), and refined_temporal_symmetry. Existing period, amplitude, duration, rates, and all other original measurements are preserved. Use the refined rise and symmetry columns explicitly when reproducing the research benchmark; do not mix durations from different policies without labeling them.

Inputs must use ordered, unique numeric index labels per recording, with row order representing regular sampling. This is an offline operation; it is not a causal streaming method. The refinement radius is tied to the smoothing window, not a universal accuracy guarantee. Synthetic confirmation supports the policy under tested conditions only.
