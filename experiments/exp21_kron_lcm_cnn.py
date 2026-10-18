"""Exp 21 — is a *learned* feature map chain-structured? Kron-LCM on a 1-D CNN.

Exps 18 and 20 both put the chain on raw, physically ordered inputs (simulated AR(1)
channels, Tecator wavelengths). That is the easy case and it leaves the question that
actually decides whether any of this concerns deep learning untouched:

    when a network *learns* a 1-D feature map, is the covariance of those learned
    features still a smooth 1-D chain?

This is the construction the KFAC x LCM note recommends (and what the original REMIX paper
applied the kernel to -- conv feature maps -- whereas this project had been applying it to
weight curvature). Here a small 1-D CNN is trained on Tecator spectra, and last-layer
Laplace is taken over `Linear(8*L, 3)`, i.e. the Kronecker input factor is the covariance
of the flattened penultimate feature map. Three outputs keep the Kronecker structure
non-vacuous; with an isotropic Gaussian likelihood KFAC is exact, so `kron` vs `full` is
again a free internal check.

Two things are measured, and the first matters more than the predictive numbers:

1. **Is the learned factor chain-like at all?** Neighbour correlation along the flattened
   index, how well spectral seriation agrees with it, and the relative Frobenius error of
   the fitted LCM against the true factor.
2. **Does it predict?** In-distribution and real covariate-shifted (high-fat) likelihood,
   *with* the oracle scalar-scale control that Exps 17/18/20 showed to be indispensable.

Writes results/exp21_cnn.csv.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from laplace import Laplace  # noqa: E402
from laplace.lcm import seriation_order  # noqa: E402

import exp20_kron_lcm_real as R  # reuse the Tecator loader/split  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

SEEDS, EPOCHS, CH = 5, 400, 8
STRUCTURES = [
    ("full", "full", {}),
    ("kron", "kron", {}),
    ("kron_lcm", "kron_lcm", {"lcm_order": "natural"}),
    ("kron_lcm_ser", "kron_lcm", {"lcm_order": "seriate"}),
    ("diag", "diag", {}),
]


class CNN(nn.Module):
    """Conv stack over the spectrum, then a linear head on the flattened feature map."""

    def __init__(self, length, c_out, ch=CH):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, ch, 5), nn.ReLU(),
            nn.Conv1d(ch, ch, 5), nn.ReLU(),
            nn.MaxPool1d(2), nn.Flatten(),
        )
        with torch.no_grad():
            self.n_feat = self.body(torch.zeros(1, 1, length)).shape[1]
        self.last_layer = nn.Linear(self.n_feat, c_out)

    def features(self, x):
        return self.body(x.unsqueeze(1))

    def forward(self, x):
        return self.last_layer(self.features(x))


def train(model, X, Y, epochs=EPOCHS, lr=3e-3, wd=1e-4, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        idx = torch.randperm(len(X), generator=g)[:64]
        opt.zero_grad()
        ((model(X[idx]) - Y[idx]) ** 2).mean().backward()
        opt.step()
    return model


@torch.no_grad()
def structure_diagnostics(model, X):
    """Is the learned feature map a 1-D chain? All O(n_feat) or one eigendecomposition."""
    Phi = model.features(X)
    A = Phi.T @ Phi / len(Phi)
    g = torch.sqrt(torch.diagonal(A).clamp(min=1e-30))
    Rm = (A / g[:, None] / g[None, :])
    nb = float(torch.diagonal(Rm, 1).abs().mean())
    far = float(Rm.abs().mean())
    o = seriation_order(A).numpy()
    nat = np.arange(len(o))
    return {
        "n_feat": A.shape[0],
        "nb_corr": nb,
        "mean_abs_corr": far,
        "seriation_spearman": abs(spearmanr(o, nat).statistic),
        "eff_rank": float(torch.diagonal(A).sum() ** 2 / (A**2).sum()),
    }, A


@torch.no_grad()
def lcm_fit_error(la, A):
    """Relative Frobenius error of the fitted chain against the true input factor."""
    for b in la.posterior_precision.blocks:
        if b["kind"] == "lcm":
            Ah = b["lcm"].dense()
            s = float((Ah * A).sum() / (Ah * Ah).sum().clamp(min=1e-300))  # best scalar
            return float((s * Ah - A).norm() / A.norm())
    return float("nan")


@torch.no_grad()
def nll(la, X, Y, sigma, scales=None):
    f_mu, f_var = la(X, pred_type="glm")
    C = Y.shape[1]
    eye = torch.eye(C, dtype=Y.dtype)
    e = Y - f_mu
    def _v(t):
        S = t * f_var + sigma**2 * eye
        sol = torch.linalg.solve(S, e.unsqueeze(-1)).squeeze(-1)
        return float((0.5 * (C * math.log(2 * math.pi) + torch.logdet(S)
                             + (sol * e).sum(-1))).mean()), float((sol * e).sum(-1).mean() / C)
    base, z2 = _v(1.0)
    if scales is None:
        return base, z2, base, 1.0
    best = min(((_v(t)[0], t) for t in scales), key=lambda r: r[0])
    return base, z2, best[0], best[1]


def run(seed):
    X, Y = R.load_tecator()
    D = R.split(X, Y, seed)
    torch.manual_seed(seed)
    model = train(CNN(X.shape[1], Y.shape[1]), *D["tr"], seed=seed)
    resid = (model(D["tr"][0]) - D["tr"][1]).detach()
    sigma = float(resid.std().clamp(min=1e-3))
    diag, A = structure_diagnostics(model, D["tr"][0])
    out = {"seed": seed, "sigma": sigma, **diag}
    out["train_mse"] = float((resid**2).mean())
    dl = DataLoader(TensorDataset(*D["tr"]), batch_size=64)
    grid = np.logspace(-1, 3, 41)

    for label, structure, kw in STRUCTURES:
        la = Laplace(model, "regression", subset_of_weights="last_layer",
                     hessian_structure=structure, sigma_noise=sigma, **kw)
        la.fit(dl)
        la.optimize_prior_precision(method="marglik")
        out[f"marglik_{label}"] = float(la.log_marginal_likelihood())
        if label.startswith("kron_lcm"):
            out[f"fitA_{label}"] = lcm_fit_error(la, A)
        for tag in ("in", "sh"):
            b, z2, t_nll, t = nll(la, *D[tag], sigma, grid if tag == "sh" else None)
            out[f"nll_{tag}_{label}"], out[f"z2_{tag}_{label}"] = b, z2
            if tag == "sh":
                out[f"nll_sh_temp_{label}"], out[f"temp_sh_{label}"] = t_nll, t
    return out


def main():
    df = pd.DataFrame([run(s) for s in range(SEEDS)])
    df.to_csv(f"{RESULTS}/exp21_cnn.csv", index=False)
    m = df.mean(numeric_only=True)
    labels = [s[0] for s in STRUCTURES]
    print(f"  1-D CNN on Tecator: {CH} channels -> {int(m['n_feat'])} flattened features,"
          f" C=3 targets, {SEEDS} seeds, train MSE {m['train_mse']:.4f}")
    print("\n  IS THE LEARNED FEATURE MAP A CHAIN?")
    print(f"    neighbour |corr| along the flattened index : {m['nb_corr']:.4f}")
    print(f"    mean |corr| over all pairs                 : {m['mean_abs_corr']:.4f}"
          f"   (a chain needs neighbour >> mean)")
    print(f"    seriation agreement with that index        : {m['seriation_spearman']:.4f}")
    print(f"    effective rank of the factor               : {m['eff_rank']:.2f}"
          f"  of {int(m['n_feat'])}")
    print(f"    LCM fit error to the true factor           : natural {m['fitA_kron_lcm']:.4f}"
          f"   seriated {m['fitA_kron_lcm_ser']:.4f}")
    print(f"\n  {'structure':15}{'marglik':>10}{'NLL in':>10}{'NLL shift':>11}"
          f"{'z2 shift':>10}{'tempered':>10}{'scale':>8}")
    for k in labels:
        print(f"  {k:15}{m[f'marglik_{k}']:10.1f}{m[f'nll_in_{k}']:10.4f}"
              f"{m[f'nll_sh_{k}']:11.4f}{m[f'z2_sh_{k}']:10.2f}"
              f"{m[f'nll_sh_temp_{k}']:10.4f}{m[f'temp_sh_{k}']:8.1f}")
    print(f"\n  internal check (KFAC exact for isotropic Gaussian): "
          f"dNLL {m['nll_in_kron']-m['nll_in_full']:+.2e}")
    print(f"\nsaved {RESULTS}/exp21_cnn.csv")


if __name__ == "__main__":
    main()
