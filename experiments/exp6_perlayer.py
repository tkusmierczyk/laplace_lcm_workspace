"""Experiment 6 — per-layer (block-diagonal) LCM (fixing assumption (ii)).

A single global 1-D chain across *all* weights is the model's core limitation:
it must order unrelated parameters (across layers) on one line. KFAC instead is
block-diagonal over layers. This prototype tests the natural fix: fit a *separate*
LCM chain to each parameter tensor's GGN block, assembling a block-diagonal
structured precision (LCM-block). We compare its fidelity to the exact full
posterior against diagonal, KFAC, and the single-chain LCM (LCM-global), under a
common fixed prior so the comparison isolates the curvature structure.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector

import common as C
from laplace import Laplace
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
COMMON_PRIOR = 1.0
SEEDS = [0, 1, 2]


def block_lcm_cov(H_full, param_sizes, h_factor, prior, steps=800, lr=3e-2):
    """Fit a separate LCM to each parameter-tensor GGN block; return the dense
    posterior covariance of the assembled block-diagonal precision."""
    P = H_full.shape[0]
    A = torch.zeros(P, P, dtype=torch.float64)
    off = 0
    for n in param_sizes:
        sl = slice(off, off + n)
        Hb = H_full[sl, sl].double()
        if n == 1:
            A[sl, sl] = Hb
        else:
            lcm = LCMCovariance(n, dtype=torch.float64)
            lcm.fit_dense(0.5 * (Hb + Hb.T), n_steps=steps, lr=lr)
            A[sl, sl] = lcm.dense()
        off += n
    prec = h_factor * A + prior * torch.eye(P, dtype=torch.float64)
    return torch.linalg.inv(prec)


def make_net(d, k, subset):
    hidden = 24 if subset == "all" else 32
    return nn.Sequential(nn.Linear(d, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, k))


def make_problem(kind, n, d, k, rho, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(d)
    L = np.linalg.cholesky(rho ** np.abs(idx[:, None] - idx[None, :]) + 1e-6 * np.eye(d))
    X = (rng.standard_normal((n, d)) @ L.T).astype("float32")
    (Xs,), _ = C.standardize(X)
    if kind == "regression":
        y = (np.tanh(Xs @ rng.standard_normal((d, 1))) +
             0.1 * rng.standard_normal((n, 1))).astype("float32")
        return Xs, ((y - y.mean()) / y.std()).astype("float32")
    W = rng.standard_normal((d, k))
    return Xs, (Xs @ W + 0.5 * rng.standard_normal((n, k))).argmax(1).astype("int64")


SETTINGS = [
    ("reg / all-weights", "regression", "all", 1, dict(n=500, d=6, rho=0.7)),
    ("cls / all-weights", "classification", "all", 3, dict(n=600, d=6, rho=0.7)),
    ("cls / last-layer", "classification", "last_layer", 3, dict(n=600, d=8, rho=0.7)),
]


def main():
    rows = []
    for label, lik, subset, k, cfg in SETTINGS:
        acc = {m: [] for m in ["Diag", "KFAC", "LCM-global", "LCM-block"]}
        for seed in SEEDS:
            C.set_seed(seed)
            X, y = make_problem(lik, k=k, seed=seed, **cfg)
            net = make_net(cfg["d"], k, subset)
            net = C.train_map(net, C.loader(X, y, 128, True), 120, 5e-3, 1e-3, lik)
            ldr = C.loader(X, y, 128)

            covs = {}
            for name, hs in [("Full", "full"), ("Diag", "diag"), ("KFAC", "kron"),
                             ("LCM-global", "lcm")]:
                la, _ = C.fit_laplace(net, ldr, hs, lik, subset=subset,
                                      common_prior=COMMON_PRIOR)
                covs[name] = C.dense_posterior_cov(la)
                if name == "Full":
                    H_full = (la.H if hasattr(la, "H") else None)
                    h_factor = float(la._H_factor)
            # block-diagonal LCM (per parameter tensor of the fitted subset)
            if subset == "last_layer":
                psizes = [p.numel() for p in la.model.last_layer.parameters()]
            else:
                psizes = [p.numel() for p in net.parameters()]
            covs["LCM-block"] = block_lcm_cov(
                torch.as_tensor(H_full).cpu(), psizes, h_factor, COMMON_PRIOR)

            for name in acc:
                acc[name].append(C.kl_gaussian(covs[name], covs["Full"]))
        row = {"setting": label}
        for name in acc:
            row[f"KL_{name}"] = float(np.mean(acc[name]))
        rows.append(row)
        print(f"  {label:20} KL  Diag={row['KL_Diag']:8.2f}  KFAC={row['KL_KFAC']:8.2f}  "
              f"LCM-global={row['KL_LCM-global']:8.2f}  LCM-block={row['KL_LCM-block']:8.2f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp6_perlayer.csv", index=False)
    print("\nsaved exp6_perlayer.csv")


if __name__ == "__main__":
    main()
