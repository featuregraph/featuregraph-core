"""
Learned features vs declared features on the same oscillating signal.

Same downstream models (Ridge, RandomForest), same targets, same time split.
Only the feature step differs:
  A) a deep autoencoder learns features from raw windows (scikit-learn MLP;
     PyTorch was not available in the sandbox)
  B) featuregraph-core builds explicit cycle objects with declared rules

Run:  PYTHONPATH=<featuregraph-core>/src python demo_features.py
"""
import hashlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from featuregraph_core.behaviors.oscillation import OscillationConfig

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
FS = 50                      # samples per second
T = 600                      # seconds
WIN = 500                    # 10 s windows
POOL = 5                     # autoencoder sees windows downsampled 5x
SMOOTH = 50                  # FeatureGraph smoothing window, in samples (1 s)

# ----------------------------------------------------------------------------
# 1. Simulated signal with known ground truth
# ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
t = np.arange(T * FS) / FS
period = 4.0 + 1.0 * np.sin(2 * np.pi * t / 170) + 0.3 * np.sin(2 * np.pi * t / 41)
phase = 2 * np.pi * np.cumsum(1.0 / period) / FS
amp = 1.0 + 0.45 * np.sin(2 * np.pi * t / 97 + 1.0) + 0.15 * np.sin(2 * np.pi * t / 23)
clean = amp * np.sin(phase)
drift = 0.35 * np.sin(2 * np.pi * t / 130 + 0.5)
signal = clean + drift + rng.normal(0, 0.22, t.size)

# True cycles: peaks of the noiseless oscillation (phase = pi/2 mod 2pi)
k = np.floor((phase - np.pi / 2) / (2 * np.pi))
peak_idx = np.where(np.diff(k) > 0)[0] + 1
true_t = peak_idx / FS
true_period = np.diff(true_t)
true_amp = amp[peak_idx]
true_t_mid, true_amp_mid = true_t[1:], true_amp[1:]  # period belongs to the later peak

df = pd.DataFrame({"rec": 0, "sig": signal})

# ----------------------------------------------------------------------------
# 2. Windows, targets, and a time split with a gap
# ----------------------------------------------------------------------------
split = int(0.6 * signal.size)
def windows(lo, hi, stride):
    return np.arange(lo, hi - WIN + 1, stride)

train_w = windows(0, split, 50)
test_w = windows(split + WIN, signal.size, 50)
ae_w = windows(0, split, 10)                       # unlabeled windows for the autoencoder

def targets(starts):
    y = []
    for s in starts:
        m = (true_t_mid * FS >= s) & (true_t_mid * FS < s + WIN)
        y.append([true_period[m].mean(), true_amp_mid[m].mean()])
    return np.array(y)

ytr, yte = targets(train_w), targets(test_w)
names = ["period (s)", "amplitude"]

def raw(starts):
    return np.array([signal[s:s + WIN].reshape(-1, POOL).mean(1) for s in starts])

def spec(starts):
    # magnitude spectrum of each window: phase removed by a human-chosen transform
    X = raw(starts)
    return np.abs(np.fft.rfft(X - X.mean(1, keepdims=True), axis=1))[:, :30]

# ----------------------------------------------------------------------------
# 3A. Learned features: deep autoencoder bottleneck (8 dims)
# ----------------------------------------------------------------------------
def train_autoencoder(seed, featurize=raw):
    Xa = featurize(ae_w)
    mu, sd = Xa.mean(), Xa.std()
    ae = MLPRegressor(hidden_layer_sizes=(64, 32, 8, 32, 64), activation="tanh",
                      max_iter=400, random_state=seed, learning_rate_init=1e-3)
    ae.fit((Xa - mu) / sd, (Xa - mu) / sd)
    def encode(X):
        h = (X - mu) / sd
        for W, b in list(zip(ae.coefs_, ae.intercepts_))[:3]:
            h = np.tanh(h @ W + b)
        return h
    encode.recon_r2 = lambda X: r2_score((X - mu) / sd, ae.predict((X - mu) / sd), multioutput='variance_weighted')
    return encode

# ----------------------------------------------------------------------------
# 3B. Declared features: featuregraph-core objects
# ----------------------------------------------------------------------------
def build_objects(smooth_window):
    cfg = OscillationConfig(signal="sig", smooth_window=smooth_window)
    added = cfg.add_primitives(df, "rec")
    s = cfg.summarize(added, ["rec", cfg.trough_event_id_col]).reset_index()
    s = s[s["is_complete"]].copy()
    s = s[s["period"].notna()].copy()   # first object sits at the recording edge (no previous peak)
    s["period_s"] = s["period"] / FS
    s["peak_t"] = s["peak_index"] / FS
    return s[["start_index", "end_index", "peak_index", "peak_t", "period_s", "amplitude_raw",
              "rising_duration", "falling_duration", "temporal_symmetry"]].reset_index(drop=True)

def fg_features(objs, starts):
    cols = ["period_s", "amplitude_raw", "rising_duration", "temporal_symmetry"]
    rows = []
    for s in starts:
        m = (objs["peak_index"] >= s) & (objs["peak_index"] < s + WIN)
        rows.append(objs.loc[m, cols].mean().values if m.any() else [np.nan] * len(cols))
    return np.array(rows)

objs = build_objects(SMOOTH)
Ftr, Fte = fg_features(objs, train_w), fg_features(objs, test_w)
fill = np.nanmedian(Ftr, axis=0)
n_missing = int(np.isnan(Ftr).any(1).sum() + np.isnan(Fte).any(1).sum())
Ftr, Fte = np.where(np.isnan(Ftr), fill, Ftr), np.where(np.isnan(Fte), fill, Fte)

# ----------------------------------------------------------------------------
# 4. Same downstream models, same targets
# ----------------------------------------------------------------------------
def evaluate(Xtr, Xte, seed=0):
    out = {}
    sc = StandardScaler().fit(Xtr)
    for j, name in enumerate(names):
        r = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr[:, j])
        f = RandomForestRegressor(200, random_state=seed, n_jobs=-1).fit(Xtr, ytr[:, j])
        out[name] = (r2_score(yte[:, j], r.predict(sc.transform(Xte))),
                     r2_score(yte[:, j], f.predict(Xte)))
    return out

results = {"raw windows (no feature step)": evaluate(raw(train_w), raw(test_w))}
seeds = list(range(5))
ae_runs, encoders = [], {}
for sd in seeds:
    enc = train_autoencoder(sd)
    encoders[sd] = enc
    ae_runs.append(evaluate(enc(raw(train_w)), enc(raw(test_w))))
print(f"autoencoder (raw windows) held-out reconstruction R2, seed 0: {encoders[0].recon_r2(raw(test_w)):.3f}")

spec_runs, spec_enc = [], {}
for sd in seeds:
    enc_s = train_autoencoder(sd, spec)
    spec_enc[sd] = enc_s
    spec_runs.append(evaluate(enc_s(spec(train_w)), enc_s(spec(test_w))))

# end-to-end supervised deep model: raw window -> target, labels at stride 10 (5x more labels)
sup_w = windows(0, split, 10)
ysup = targets(sup_w)
Xs, Xt = raw(sup_w), raw(test_w)
scs = StandardScaler().fit(Xs)
sup = {}
for j, name in enumerate(names):
    sc_y = StandardScaler().fit(ysup[:, [j]])
    r2s = []
    for sd in seeds:
        net = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600, random_state=sd).fit(
            scs.transform(Xs), sc_y.transform(ysup[:, [j]]).ravel())
        pred = sc_y.inverse_transform(net.predict(scs.transform(Xt)).reshape(-1, 1)).ravel()
        r2s.append(r2_score(yte[:, j], pred))
    sup[name] = (np.mean(r2s), np.std(r2s), min(r2s), max(r2s))

results["FeatureGraph objects (deterministic)"] = evaluate(Ftr, Fte)

def fmt(v):
    return f"{v[0]:6.2f} / {v[1]:5.2f}"

print("=== Held-out R2 (Ridge / RandomForest), same downstream models ===")
print(f"{'feature step':44s} {names[0]:>16s}   {names[1]:>16s}")
for k_, v in results.items():
    print(f"{k_:44s} {fmt(v[names[0]]):>16s}   {fmt(v[names[1]]):>16s}")
for j, name in enumerate(names):
    arr = np.array([[r[name][0], r[name][1]] for r in ae_runs])
    results.setdefault("_ae", {})[name] = (arr.mean(0), arr.std(0), arr.min(0), arr.max(0))
m = results["_ae"]
print(f"{'deep autoencoder, mean over 5 seeds':44s} "
      f"{m[names[0]][0][0]:6.2f} / {m[names[0]][0][1]:5.2f}    {m[names[1]][0][0]:6.2f} / {m[names[1]][0][1]:5.2f}")
print(f"{'  ...std over seeds':44s} "
      f"{m[names[0]][1][0]:6.2f} / {m[names[0]][1][1]:5.2f}    {m[names[1]][1][0]:6.2f} / {m[names[1]][1][1]:5.2f}")
print(f"{'  ...min..max (Ridge)':44s} "
      f"{m[names[0]][2][0]:6.2f}..{m[names[0]][3][0]:5.2f}   {m[names[1]][2][0]:6.2f}..{m[names[1]][3][0]:5.2f}")
for label, runs in (("deep autoencoder on magnitude spectrum, mean of 5", spec_runs),):
    vals = [np.mean([r[n][i] for r in runs]) for n in names for i in (0, 1)]
    print(f"{label:44s} {vals[0]:6.2f} / {vals[1]:5.2f}    {vals[2]:6.2f} / {vals[3]:5.2f}")
print(f"{'end-to-end supervised deep net (5x labels)':44s} "
      f"{sup[names[0]][0]:6.2f} (sd {sup[names[0]][1]:.2f})          {sup[names[1]][0]:6.2f} (sd {sup[names[1]][1]:.2f})")
print(f"spectrum autoencoder held-out reconstruction R2, seed 0: {spec_enc[0].recon_r2(spec(test_w)):.3f}")
print(f"(windows with no complete FeatureGraph object, imputed: {n_missing})")

# ----------------------------------------------------------------------------
# 5. What do the features mean?  Correlation of each feature with true properties
# ----------------------------------------------------------------------------
all_w = np.concatenate([train_w, test_w])
yall = targets(all_w)
def corr_table(F):
    return np.array([[abs(np.corrcoef(F[:, i], yall[:, j])[0, 1]) for j in range(2)]
                     for i in range(F.shape[1])])

Fall = fg_features(objs, all_w)
Fall = np.where(np.isnan(Fall), fill, Fall)
CF = corr_table(Fall)
for label, encs, feat in (("autoencoder on raw windows", encoders, raw),
                          ("autoencoder on magnitude spectrum", spec_enc, spec)):
    print(f"\n{label}: best single latent dimension vs TRUE period / amplitude (|r|), by seed")
    for sd in seeds:
        C = corr_table(encs[sd](feat(all_w)))
        print(f"  seed {sd}: period {C[:, 0].max():.2f} (dim {C[:, 0].argmax()}), "
              f"amplitude {C[:, 1].max():.2f} (dim {C[:, 1].argmax()})")
print("FeatureGraph named features, |r| with TRUE period / amplitude:")
for nm, row in zip(["period_s", "amplitude_raw", "rising_duration", "temporal_symmetry"], CF):
    print(f"  {nm:18s} period {row[0]:.2f}  amplitude {row[1]:.2f}")

C0 = corr_table(spec_enc[0](spec(all_w)))
C1 = corr_table(spec_enc[1](spec(all_w)))
Z0, Z1 = spec_enc[0](spec(all_w)), spec_enc[1](spec(all_w))
Zc0 = (Z0 - Z0.mean(0)) / Z0.std(0)
Zc1 = (Z1 - Z1.mean(0)) / Z1.std(0)
cross = np.abs(Zc0.T @ Zc1 / len(Zc0))
print(f"\nspectrum autoencoder, seed 0 vs seed 1: best single-dimension match, mean |r| = {cross.max(1).mean():.2f}")
lin = LinearRegression().fit(Z1, Z0)
print(f"seed 0 latents linearly predictable from seed 1 latents, R2 = {r2_score(Z0, lin.predict(Z1)):.2f}"
      "  (high = same information, different axes)")

# ----------------------------------------------------------------------------
# 6. Reproducibility and a changed assumption
# ----------------------------------------------------------------------------
def fingerprint(o):
    return hashlib.sha256(o.round(9).to_csv(index=False).encode()).hexdigest()[:12]

o_again = build_objects(SMOOTH)
print("\n=== FeatureGraph: rerun, then change one declared assumption ===")
print(f"rerun fingerprint (smooth_window={SMOOTH}) : {fingerprint(objs)} vs {fingerprint(o_again)}"
      f"  identical={fingerprint(objs) == fingerprint(o_again)}")

def vs_truth(o):
    # match each detected peak to the nearest true peak within 1 s
    tp = true_t
    err, matched = [], 0
    for pt in o["peak_t"]:
        d = np.abs(tp - pt)
        if d.min() <= 1.0:
            matched += 1
            err.append(d.min())
    return matched, len(o)

rows = []
for w in (5, 25, 50, 100):
    o = build_objects(w)
    mt, n = vs_truth(o)
    rows.append((w, fingerprint(o), n, mt, n - mt))
print("smooth_window | fingerprint  | objects | match a true peak | spurious")
for r in rows:
    print(f"{r[0]:13d} | {r[1]} | {r[2]:7d} | {r[3]:17d} | {r[4]:8d}")
print(f"true cycles in recording: {len(true_period)}")

o25, o50 = build_objects(25), objs
only25 = [p for p in o25["peak_t"] if np.abs(o50["peak_t"] - p).min() > 1.0]
print(f"peaks present at window=25 but not at window=50: {len(only25)} "
      f"(first few at t = {[round(x, 1) for x in only25[:5]]} s)")

# ----------------------------------------------------------------------------
# 7. Figure
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
z0, z1 = 200, 260
m_ = (t >= z0) & (t <= z1)
a = ax[0, 0]
a.plot(t[m_], signal[m_], color="#b0b0b0", lw=0.8, label="signal")
a.plot(t[m_], clean[m_] + drift[m_], color="#2b2b2b", lw=1.0, label="noiseless truth")
tp = true_t[(true_t >= z0) & (true_t <= z1)]
a.plot(tp, np.interp(tp, t, clean + drift), "v", color="#2980b9", ms=7, label="true peaks")
op = objs[(objs["peak_t"] >= z0) & (objs["peak_t"] <= z1)]
a.plot(op["peak_t"], np.interp(op["peak_t"], t, signal), "o", mfc="none", color="#c0392b", ms=9,
       label=f"FeatureGraph peaks (smooth_window={SMOOTH})")
a.set_title("Declared rules: each cycle becomes an object"); a.legend(fontsize=7, loc="upper right")
a.set_xlabel("time (s)")

def heat(a_, C, title):
    im = a_.imshow(C, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    a_.set_xticks([0, 1]); a_.set_xticklabels(["true period", "true amplitude"], fontsize=8)
    a_.set_yticks(range(C.shape[0])); a_.set_yticklabels([f"z{i}" for i in range(C.shape[0])], fontsize=8)
    for i in range(C.shape[0]):
        for j in range(2):
            a_.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if C[i, j] > 0.6 else "black")
    a_.set_title(title, fontsize=10)
heat(ax[0, 1], C0, "Learned features (spectrum autoencoder), seed 0")
heat(ax[1, 0], C1, "Same code, same data, seed 1: different axes")

a = ax[1, 1]
a.plot(t[m_], signal[m_], color="#b0b0b0", lw=0.8)
for w, col, mk, off in ((25, "#c0392b", "o", 0.0), (100, "#27ae60", "s", 0.25)):
    o = build_objects(w)
    oz = o[(o["peak_t"] >= z0) & (o["peak_t"] <= z1)]
    a.plot(oz["peak_t"], np.interp(oz["peak_t"], t, signal) + off, mk, mfc="none", color=col, ms=8,
           label=f"smooth_window={w}: {len(oz)} peaks in this 60 s")
a.set_title("Change one declared assumption: the difference is visible\n(window=100 markers drawn slightly higher so both sets can be seen)", fontsize=10)
a.legend(fontsize=7, loc="upper right"); a.set_xlabel("time (s)")
plt.tight_layout()
plt.savefig(f"{OUT}/learned_vs_declared.png", dpi=140)
print(f"\nfigure written to {OUT}/learned_vs_declared.png")

objs.head(8).round(3).to_csv(f"{OUT}/object_table_head.csv", index=False)
print("\nfirst FeatureGraph objects:\n", objs.head(8).round(3).to_string(index=False))
