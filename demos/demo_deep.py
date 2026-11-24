"""
Deep learning comparison (PyTorch) -- companion to demo_features.py.

Reuses demo_features.py for the simulation, windows, targets, time split and
FeatureGraph features, so everything except the feature step is identical.

Adds two real deep models:
  1. a 1D convolutional autoencoder (learned features, 8 latent dims),
     whose latents are fed to the same Ridge and RandomForest as before
  2. an end-to-end supervised 1D CNN (raw window -> period, amplitude)

Needs: featuregraph-core, scikit-learn, pandas, matplotlib, torch.
Run in the same folder as demo_features.py:
    python demo_deep.py out
"""
import sys

import numpy as np

# --- reuse the first demo's simulation, windows, targets and FeatureGraph features ---
_src = open("demo_features.py").read()
exec(_src.split("# 4. Same downstream models")[0])


def evaluate(Xtr, Xte, seed=0):
    """Same downstream models as demo_features.py: Ridge(alpha=1) and RandomForest(200)."""
    out = {}
    sc = StandardScaler().fit(Xtr)
    for j, name in enumerate(names):
        r = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr[:, j])
        f = RandomForestRegressor(200, random_state=seed, n_jobs=-1).fit(Xtr, ytr[:, j])
        out[name] = (r2_score(yte[:, j], r.predict(sc.transform(Xte))),
                     r2_score(yte[:, j], f.predict(Xte)))
    return out


def fmt(v):
    return f"{v[0]:6.2f} / {v[1]:5.2f}"


print("=== Reference rows (same as demo_features.py) ===")
ref_raw = evaluate(raw(train_w), raw(test_w))
ref_fg = evaluate(Ftr, Fte)
print(f"{'feature step':46s} {names[0]:>16s}   {names[1]:>16s}")
print(f"{'raw windows (no feature step)':46s} {fmt(ref_raw[names[0]]):>16s}   {fmt(ref_raw[names[1]]):>16s}")
print(f"{'FeatureGraph objects (deterministic)':46s} {fmt(ref_fg[names[0]]):>16s}   {fmt(ref_fg[names[1]]):>16s}")

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("\nPyTorch is not installed. Install it (pip install torch) and run again.")
    sys.exit(0)

torch.set_num_threads(2)
N_IN = WIN // POOL                      # 100 samples per window after pooling
assert N_IN == 100, "the layer sizes below assume 100 input samples"


def to_tensor(X, mu, sd):
    return torch.tensor(((X - mu) / sd)[:, None, :], dtype=torch.float32)


# ----------------------------------------------------------------------------
# 1. Convolutional autoencoder: learned features, no labels
# ----------------------------------------------------------------------------
class ConvAE(nn.Module):
    def __init__(self, k=8):
        super().__init__()
        # lengths: 100 -> 50 -> 25 -> 13
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.Tanh(),
            nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.Tanh(),
            nn.Flatten(), nn.Linear(32 * 13, k), nn.Tanh())
        self.fc = nn.Linear(k, 32 * 13)
        # lengths: 13 -> 25 -> 50 -> 100
        self.dec = nn.Sequential(
            nn.Tanh(),
            nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=0), nn.Tanh(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.Tanh(),
            nn.ConvTranspose1d(16, 1, 5, stride=2, padding=2, output_padding=1))

    def forward(self, x):
        out = self.dec(self.fc(self.enc(x)).view(-1, 32, 13))
        assert out.shape == x.shape, (out.shape, x.shape)
        return out


def train_conv_ae(seed, epochs=150):
    torch.manual_seed(seed)
    Xa = raw(ae_w)
    mu, sd = Xa.mean(), Xa.std()
    X = to_tensor(Xa, mu, sd)
    model = ConvAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            b = X[perm[i:i + 64]]
            loss = ((model(b) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()

    def encode(Xr):
        with torch.no_grad():
            return model.enc(to_tensor(Xr, mu, sd)).numpy()

    def recon_r2(Xr):
        with torch.no_grad():
            t_ = to_tensor(Xr, mu, sd)
            return r2_score(t_[:, 0, :].numpy(), model(t_)[:, 0, :].numpy(),
                            multioutput="variance_weighted")
    encode.recon_r2 = recon_r2
    return encode


seeds = list(range(5))
ae_runs, encs = [], {}
for sd_ in seeds:
    encs[sd_] = train_conv_ae(sd_)
    ae_runs.append(evaluate(encs[sd_](raw(train_w)), encs[sd_](raw(test_w))))
    print(f"conv autoencoder seed {sd_}: held-out reconstruction R2 = {encs[sd_].recon_r2(raw(test_w)):.3f}")

# ----------------------------------------------------------------------------
# 2. End-to-end supervised 1D CNN (uses labels, 5x more labelled windows)
# ----------------------------------------------------------------------------
class ConvReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.h = nn.Linear(32, 2)

    def forward(self, x):
        return self.h(self.f(x))


sup_w = windows(0, split, 10)
ysup = targets(sup_w)
ymu, ysd = ysup.mean(0), ysup.std(0)
Xs_raw, Xt_raw = raw(sup_w), raw(test_w)
mu_s, sd_s = Xs_raw.mean(), Xs_raw.std()
Xs, Xt = to_tensor(Xs_raw, mu_s, sd_s), to_tensor(Xt_raw, mu_s, sd_s)
Ys = torch.tensor((ysup - ymu) / ysd, dtype=torch.float32)

sup_r2 = []
for sd_ in seeds:
    torch.manual_seed(sd_)
    net = ConvReg()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(100):
        perm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), 64):
            idx = perm[i:i + 64]
            loss = ((net(Xs[idx]) - Ys[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xt).numpy() * ysd + ymu
    sup_r2.append([r2_score(yte[:, j], pred[:, j]) for j in range(2)])
sup_r2 = np.array(sup_r2)

# ----------------------------------------------------------------------------
# 3. Results
# ----------------------------------------------------------------------------
print("\n=== Held-out R2 (Ridge / RandomForest on the features), same split and targets ===")
print(f"{'feature step':46s} {names[0]:>16s}   {names[1]:>16s}")
print(f"{'raw windows (no feature step)':46s} {fmt(ref_raw[names[0]]):>16s}   {fmt(ref_raw[names[1]]):>16s}")
print(f"{'FeatureGraph objects (deterministic)':46s} {fmt(ref_fg[names[0]]):>16s}   {fmt(ref_fg[names[1]]):>16s}")
m = {n: np.array([[r[n][0], r[n][1]] for r in ae_runs]) for n in names}
print(f"{'conv autoencoder, mean of 5 seeds':46s} "
      f"{m[names[0]].mean(0)[0]:6.2f} / {m[names[0]].mean(0)[1]:5.2f}    "
      f"{m[names[1]].mean(0)[0]:6.2f} / {m[names[1]].mean(0)[1]:5.2f}")
print(f"{'  ...std over seeds':46s} "
      f"{m[names[0]].std(0)[0]:6.2f} / {m[names[0]].std(0)[1]:5.2f}    "
      f"{m[names[1]].std(0)[0]:6.2f} / {m[names[1]].std(0)[1]:5.2f}")
print(f"{'supervised 1D CNN, mean of 5 seeds (labels)':46s} "
      f"{sup_r2[:, 0].mean():6.2f} (sd {sup_r2[:, 0].std():.2f})          "
      f"{sup_r2[:, 1].mean():6.2f} (sd {sup_r2[:, 1].std():.2f})")

all_w = np.concatenate([train_w, test_w])
yall = targets(all_w)
print("\nconv autoencoder: best single latent dimension vs TRUE period / amplitude (|r|), by seed")
for sd_ in seeds:
    Z = encs[sd_](raw(all_w))
    c = np.array([[abs(np.corrcoef(Z[:, i], yall[:, j])[0, 1]) for j in range(2)] for i in range(Z.shape[1])])
    print(f"  seed {sd_}: period {c[:, 0].max():.2f} (dim {c[:, 0].argmax()}), "
          f"amplitude {c[:, 1].max():.2f} (dim {c[:, 1].argmax()})")
print("FeatureGraph named features: period_s |r| with true period = 0.95, amplitude_raw |r| with true amplitude = 0.99"
      " (from demo_features.py)")
