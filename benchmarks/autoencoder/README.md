# Autoencoder comparison protocol (proposed; not yet validated)

Research question: When the type of physical signal behavior is known, can a representation designed for human and agent consumption also retain information useful for downstream prediction, compared with learned representations?

Preserve demos/demo_features.py and demos/demo_deep.py as exploratory baselines. Do not tune the FeatureGraph construction against test outcomes.

## Primary protocol
- Preserve the original 50 Hz, 600 s oscillation generator and 50-sample smoothing specification. Use independent noise and modulation phase realizations, with disjoint development and held-out recording seeds.
- Each example receives exactly the same 20-second, full-resolution input interval: a central 10-second scoring interval plus five seconds on either side. No method may access samples outside that interval. Construct FeatureGraph objects independently per input interval; retain complete objects whose peaks fall in the central interval and whose period is defined locally. All learned encoders and raw baselines receive all 1000 samples, without pooling. Ground truth uses simulated peak times and clean amplitudes in the central interval, with period assigned to the later peak.
- Primary central intervals are non-overlapping; repeat with one-second stride as sensitivity analysis. Context may overlap within a recording; never treat windows as independent replicates.
- Use the same labeled development examples for Ridge and Random Forest probes. Fit scaling and missing-value imputation on development only. Train autoencoders on development inputs only, with five fixed initialization seeds. Report four- and eight-dimensional learned bottlenecks; the original four FeatureGraph aggregate features remain fixed.
- Report per-record and pooled held-out R², MAE, and RMSE; report paired differences and variability across independent recordings and initialization seeds separately. Report missingness and complete-case results alongside imputed results.
- Report direct FeatureGraph period/amplitude measurement errors before regression. Match detected and true peaks one-to-one within a fixed one-second tolerance; report precision, recall and timing errors. The tolerance is inherited from the exploratory demo, not optimized.
- Report raw and spectral reconstruction separately, since they reconstruct different inputs. Include uncompressed spectral and raw baselines. Fit latent alignment on development and evaluate on held-out data; correlation is not proof of semantic equivalence.
- Supervised MLP/CNN are separate labeled reference models, not unsupervised representation peers.
- Save configuration, dependency versions, per-example predictions, per-record metrics and seed-level results. Verify boundary isolation, disjoint recordings, matching uniqueness and deterministic FeatureGraph output in tests.

## Scope
The study tests known oscillatory behavior, not discovery of unknown physical structure. The agent demonstration is separate. Competitive performance is a hypothesis, not an established result of this revised protocol. Five-second context and independent modulation phases are proposed design choices; confirm coverage and freeze them before inspecting held-out outcomes. The smoothing sensitivity experiment belongs primarily to the separate smoothing study.

## Status
Protocol drafted; implementation, execution and independent numerical verification pending.

## Smoothing control amendment
Before held-out evaluation, add uncompressed smoothed waveform and smoothed spectrum baselines, plus four- and eight-dimensional MLP autoencoders on each. Use the existing centered rolling median followed by centered rolling mean, both with 50 samples. Fill undefined edge samples for learned inputs with the nearest valid smoothed value inside the common input interval; no external samples are accessed. This edge policy is specific to fixed-length learned inputs, not a change to FeatureGraph. Report reconstruction separately for each input type. FeatureGraph retains its original raw-amplitude measurement. The convolutional controls remain to be implemented. This amendment has not been execution-verified.
