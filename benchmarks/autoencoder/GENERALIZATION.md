# Waveform and frequency generalization

Exploratory extension of Experiment B. The goal is to separate the effect of changing waveform morphology from that of changing the frequency range. All models are fitted only on the original 600 sinusoidal training recordings (seeds 10000–10599). The new evaluation seeds are 40000–40199, shared across conditions to preserve paired nuisance draws. The construction settings, training imputation, standardization, regression hyperparameters and learned-encoder training procedure are fixed; no fitting or selection on the new evaluation conditions.

## Conditions

| Condition | Carrier | Nominal frequency (Hz) |
| --- | --- | --- |
| sinusoid_in_range | cosine | 0.20–0.35 |
| triangle_in_range | triangle | 0.20–0.35 |
| asymmetric_in_range | asymmetric triangle | 0.20–0.35 |
| flattened_in_range | tanh-transformed cosine | 0.20–0.35 |
| sinusoid_slow | cosine | 0.12–0.19 |
| sinusoid_fast | cosine | 0.36–0.55 |
| triangle_fast | triangle | 0.36–0.55 |

Each synthetic carrier has unit extrema and is multiplied by the same exponential-envelope model used in Experiment B. Initial amplitude, phase, baseline, noise, and damping-ratio draws are paired by seed across conditions. Only the carrier function and/or the frequency distribution change. The non-sinusoidal signals are synthetic morphology perturbations, NOT exact solutions of the linear damped harmonic oscillator equation. For those conditions the original target names damping_ratio and natural_frequency_hz designate the nominal envelope and carrier-generating parameters; they are not independently established physical system parameters. The frequency-shift conditions also change the envelope decay rate through the original parameterization.

## Methods

FeatureGraph: retain all structurally complete cycles with peaks in the central interval; undefined measurements remain missing until feature-wise aggregation and training-only median imputation. This is the revised retention policy motivated by the earlier failure analysis, not a change to cycle detection. Smoothing remains 50 samples. Bycycle 1.2.0: fixed band 0.15–0.45 Hz, 8-second extrema-localization filter, separate 2-cycle internal band-amplitude filter. No per-record frequencies are provided. Both methods use the original four measurements and the original two-bin aggregation. Compare two-amplitude subsets as well. Both methods receive 1000 samples, evaluated as a central ten-second interval with five seconds of context on each side.

Include raw and smoothed waveform and spectrum baselines. The optional full benchmark fits the eight existing MLP autoencoder configurations (five seeds each, max_iter=400) on the same original training inputs and freezes the encoders before evaluation. It is a replication of the earlier training budget, not a convergence-optimized final baseline. Ridge alpha=1 and Random Forest 200 trees, seed 0. No target-specific changes.

Report R², MAE, RMSE, prediction values, coefficient values, measurement availability, detected-cycle counts, and construction errors for each condition. A completely unavailable representation is imputed from training medians rather than silently excluded; such performance must be interpreted alongside coverage. A fixed 20-second window may not contain enough cycles at low frequencies, so failure there cannot be attributed to the bycycle band alone. The high-frequency condition straddles the upper band edge. Report results separately; do not interpret these deliberately shifted synthetic conditions as real-world cross-domain validation.

This study is exploratory. Final confirmation of the original B findings and any newly formulated hypotheses requires a separate frozen protocol and fresh evaluation sample. The original scripts and results are preserved.
