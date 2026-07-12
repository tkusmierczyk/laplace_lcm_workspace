"""Experiment 1: posterior fidelity and the 'when does structure help' ablation.

Where the exact full Laplace posterior is tractable, we measure how closely each
structured approximation (Diag, KFAC, LCM) reproduces it, using a *common fixed
prior* so the comparison isolates the curvature-approximation quality. We report
KL( approx || full ) and the relative Frobenius error of the posterior covariance.

The ablation varies the feature-correlation level rho: as the curvature becomes
more off-diagonal, LCM's advantage over Diag grows, and at rho=0 the two coincide.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as C

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
COMMON_PRIOR = 1.0
SEEDS = [0, 1, 2]


def ar1_features(n, d, rho, rng):
    """n x d features with AR(1) correlation rho^|i-j| across the d dims."""
    idx = np.arange(d)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(d))
    return (rng.standard_normal((n, d)) @ L.T).astype("float32")


def random_corr_features(n, d, strength, rng):
    """Dense random-correlation features (a less LCM-favourable structure)."""
    B = rng.standard_normal((d, d))
    Sigma = strength * (B @ B.T) / d + (1 - strength) * np.eye(d)
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(d))
    return (rng.standard_normal((n, d)) @ L.T).astype("float32")


def make_net(d, k, hidden=32):
    return nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, k))


def fit_and_fidelity(net, ldr, likelihood, subset):
    """Return dict of KL and Frobenius(cov) to the full posterior for each method."""
    covs = {}
    for name, spec in C.METHODS.items():
        if spec is None:
            continue
        la, _ = C.fit_laplace(net, ldr, spec[0], likelihood, subset=subset,
                              common_prior=COMMON_PRIOR)
        covs[name] = C.dense_posterior_cov(la)
    full = covs["Full"]
    out = {}
    for name in ["Diag", "KFAC", "LCM"]:
        out[f"KL_{name}"] = C.kl_gaussian(covs[name], full)
        out[f"Frob_{name}"] = C.frob_rel(covs[name], full)
    return out


# --------------------------------------------------- Part A: fidelity table
def make_problem(kind, n, d, k, rho, seed):
    rng = np.random.default_rng(seed)
    X = ar1_features(n, d, rho, rng)
    (Xs,), _ = C.standardize(X)
    if kind == "regression":
        w = rng.standard_normal((d, 1))
        y = (np.tanh(Xs @ w) + 0.1 * rng.standard_normal((n, 1))).astype("float32")
        y = ((y - y.mean()) / y.std()).astype("float32")
        return Xs, y
    else:
        W = rng.standard_normal((d, k))
        y = (Xs @ W + 0.5 * rng.standard_normal((n, k))).argmax(1).astype("int64")
        return Xs, y


SETTINGS = [
    ("reg / last-layer", "regression", "last_layer", 1, dict(n=500, d=8, rho=0.7)),
    ("reg / all-weights", "regression", "all",       1, dict(n=500, d=6, rho=0.7)),
    ("cls / last-layer", "classification", "last_layer", 3, dict(n=600, d=8, rho=0.7)),
    ("cls / all-weights", "classification", "all",   3, dict(n=600, d=6, rho=0.7)),
]

print("=== Part A: posterior fidelity (common prior, mean over seeds) ===")
rowsA = []
for label, lik, subset, k, cfg in SETTINGS:
    accum = {key: [] for key in
             ["KL_Diag", "KL_KFAC", "KL_LCM", "Frob_Diag", "Frob_KFAC", "Frob_LCM"]}
    for seed in SEEDS:
        C.set_seed(seed)
        X, y = make_problem(lik, k=k, seed=seed, **cfg)
        hidden = 24 if subset == "all" else 32
        net = make_net(cfg["d"], k, hidden)
        net = C.train_map(net, C.loader(X, y, 128, True),
                          epochs=120, lr=5e-3, weight_decay=1e-3, likelihood=lik)
        res = fit_and_fidelity(net, C.loader(X, y, 128), lik, subset)
        for key in accum:
            accum[key].append(res[key])
    row = {"setting": label, "P": net[-1].in_features * k + k if subset == "last_layer"
           else sum(p.numel() for p in net.parameters())}
    for key, vals in accum.items():
        row[key] = np.mean(vals)
    rowsA.append(row)
    print(f"  {label:20} KL: Diag={row['KL_Diag']:8.2f}  KFAC={row['KL_KFAC']:8.2f}  "
          f"LCM={row['KL_LCM']:8.2f}")

dfA = pd.DataFrame(rowsA)
dfA.to_csv("experiments/results/exp1_fidelity.csv", index=False)

# --------------------------------------------------- Part B: when-it-helps ablation
print("\n=== Part B: fidelity vs feature correlation (last-layer classification) ===")
RHOS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
rowsB = []
for corr_type in ["ar1", "random"]:
    for rho in RHOS:
        kl_diag, kl_lcm = [], []
        for seed in SEEDS:
            C.set_seed(seed)
            rng = np.random.default_rng(seed)
            n, d, k = 600, 8, 3
            if corr_type == "ar1":
                X = ar1_features(n, d, rho, rng)
            else:
                X = random_corr_features(n, d, rho, rng)
            (Xs,), _ = C.standardize(X)
            W = rng.standard_normal((d, k))
            y = (Xs @ W + 0.5 * rng.standard_normal((n, k))).argmax(1).astype("int64")
            net = make_net(d, k, 32)
            net = C.train_map(net, C.loader(Xs, y, 128, True),
                              epochs=120, lr=5e-3, weight_decay=1e-3,
                              likelihood="classification")
            res = fit_and_fidelity(net, C.loader(Xs, y, 128), "classification", "last_layer")
            kl_diag.append(res["KL_Diag"]); kl_lcm.append(res["KL_LCM"])
        rowsB.append(dict(corr_type=corr_type, rho=rho,
                          KL_Diag=np.mean(kl_diag), KL_Diag_sd=np.std(kl_diag),
                          KL_LCM=np.mean(kl_lcm), KL_LCM_sd=np.std(kl_lcm)))
        print(f"  {corr_type:7} rho={rho:.2f}  KL_Diag={np.mean(kl_diag):7.2f}  "
              f"KL_LCM={np.mean(kl_lcm):7.2f}")

dfB = pd.DataFrame(rowsB)
dfB.to_csv("experiments/results/exp1_whenithelps.csv", index=False)

# --------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, ct, title in zip(axes, ["ar1", "random"],
                         ["AR(1) feature correlation", "random dense correlation"]):
    sub = dfB[dfB.corr_type == ct]
    ax.plot(sub.rho, sub.KL_Diag, "o-", label="Diag", color="#c44")
    ax.fill_between(sub.rho, sub.KL_Diag - sub.KL_Diag_sd, sub.KL_Diag + sub.KL_Diag_sd,
                    color="#c44", alpha=0.15)
    ax.plot(sub.rho, sub.KL_LCM, "s-", label="LCM", color="#268")
    ax.fill_between(sub.rho, sub.KL_LCM - sub.KL_LCM_sd, sub.KL_LCM + sub.KL_LCM_sd,
                    color="#268", alpha=0.15)
    ax.set_title(title); ax.set_xlabel("correlation strength"); ax.grid(alpha=0.3)
axes[0].set_ylabel("KL( approx || full )")
axes[0].legend()
fig.suptitle("Posterior fidelity vs curvature correlation (lower = closer to exact posterior)")
fig.tight_layout()
fig.savefig("experiments/results/exp1_whenithelps.png", dpi=140)
print("\nsaved: experiments/results/exp1_fidelity.csv, exp1_whenithelps.csv, exp1_whenithelps.png")
