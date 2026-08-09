"""Experiment 10b — ordering fragility + recovery failure (the decisive negative).

Two questions that determine whether LCM's *learned 1-D coordinate* (its novelty
vs. classical banded / state-space-GP models) is worth anything:

  (a) FRAGILITY: how accurate must the 1-D ordering be for the Laplace-kernel chain
      to beat diagonal on joint predictive UQ? We take the true order, corrupt it by
      adding Gaussian noise to positions, and measure joint-LL vs. order accuracy.

  (b) RECOVERY: can the order be recovered when it is latent? We compare
      gradient-learned coordinates (multiple restarts) and classical spectral
      (Fiedler) seriation against the accuracy threshold from (a).

Mechanism: a mis-ordered chain places truly-distant features at adjacent
coordinates and so asserts a strong (~e^{-gap}) correlation that is not there -- a
confidently wrong model, worse than diagonal's safe "no correlation". Diagonal is a
robust under-model; the chain is fragile to fine-ordering error.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import exp10_ordering as E
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)


def setup(d, n, rho, seed, ntest, noise, prior):
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    GGN = X.T @ X / noise**2
    Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
    mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))
    Inoise = noise**2 * torch.eye(ntest)

    def jll(C):
        Cj = Xt @ C @ Xt.T + Inoise
        L = torch.linalg.cholesky(Cj)
        r = torch.linalg.solve_triangular(L, (yt - mu_t).unsqueeze(1), upper=False).squeeze(1)
        return float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                             + ntest * np.log(2 * np.pi)) / ntest)

    return GGN, Cov_full, jll, X, Xt


def spear(order, d):
    ranks = torch.empty(d)
    ranks[order] = torch.arange(d, dtype=torch.float64)
    return abs(spearmanr(ranks.numpy(), np.arange(d)).correlation)


def main():
    d, n, rho, ntest, noise, prior = 400, 2000, 0.9, 300, 0.5, 1.0
    seeds = [0, 1, 2]
    pcov = lambda A: torch.linalg.inv(A + prior * torch.eye(d))

    # ---------- (a) fragility curve ----------
    frag_rows = []
    for seed in seeds:
        GGN, Cov_full, jll, X, Xt = setup(d, n, rho, seed, ntest, noise, prior)
        ll_full = jll(Cov_full)
        ll_diag = jll(torch.diag(1.0 / (GGN.diagonal() + prior)))
        true = torch.arange(d, dtype=torch.float64)
        for lvl in [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            g = torch.Generator().manual_seed(seed)
            order = torch.argsort(true + lvl * torch.randn(d, generator=g))
            sp = spear(order, d)
            A = E.fit_fixed_order(GGN, order, steps=500)
            frag_rows.append(dict(seed=seed, pos_noise=lvl, spearman=sp,
                                  joint_ll=jll(pcov(A)), ll_full=ll_full, ll_diag=ll_diag))
    frag = pd.DataFrame(frag_rows)
    frag.to_csv("experiments/results/exp10b_fragility.csv", index=False)
    g = frag.groupby("pos_noise")
    print("=== (a) ordering fragility: joint-LL vs order accuracy (mean over seeds) ===")
    print(f"    full={frag.ll_full.mean():+.3f}   diag={frag.ll_diag.mean():+.3f}   "
          f"(chain beats diag only when it is ABOVE diag)\n")
    print(f"    {'|Spearman|':>11} {'joint-LL':>9} {'beats diag?':>12}")
    for _, r in g.mean(numeric_only=True).iterrows():
        beats = "yes" if r.joint_ll > r.ll_diag else "NO"
        print(f"    {r.spearman:>11.3f} {r.joint_ll:>+9.3f} {beats:>12}")

    # ---------- (b) recovery: can the latent order be found? ----------
    print("\n=== (b) order recovery on PERMUTED data (need |Spearman| ~0.999 to help) ===")
    rec_rows = []
    for seed in seeds:
        GGN0, _, _, X, Xt = setup(d, n, rho, seed, ntest, noise, prior)
        g = torch.Generator().manual_seed(2000 + seed)
        perm = torch.randperm(d, generator=g)
        Xp = X[:, perm]
        GGN = Xp.T @ Xp / noise**2
        true_order = torch.argsort(perm).double()
        # classical spectral seriation
        seri = E.fiedler_order(GGN)
        sp_seri = abs(spearmanr(
            torch.empty(d).scatter_(0, seri, torch.arange(d).double()).numpy(),
            true_order.numpy()).correlation)
        # gradient-learned coordinate, best of restarts
        best_learn = 0.0
        for init in range(4):
            lcm = LCMCovariance(d, dtype=torch.float64)
            gg = torch.Generator().manual_seed(init)
            with torch.no_grad():
                lcm.a.copy_(torch.randn(d, generator=gg) * (0.1 * d) / 3)
            lcm.fit_dense(GGN, n_steps=1500, lr=5e-2)
            best_learn = max(best_learn, abs(spearmanr(
                lcm.a.detach().numpy(), true_order.numpy()).correlation))
        rec_rows.append(dict(seed=seed, seriation=sp_seri, learned=best_learn))
        print(f"    seed {seed}: seriation |Spearman|={sp_seri:.3f}   "
              f"learned(best/4) |Spearman|={best_learn:.3f}")
    pd.DataFrame(rec_rows).to_csv("experiments/results/exp10b_recovery.csv", index=False)
    rr = pd.DataFrame(rec_rows).mean(numeric_only=True)
    print(f"\n    mean: seriation={rr.seriation:.3f}  learned={rr.learned:.3f}  "
          f"(both far below the ~0.999 the chain needs)")
    print("\nsaved exp10b_fragility.csv, exp10b_recovery.csv")


if __name__ == "__main__":
    main()
