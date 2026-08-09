"""Experiment 8c — headline joint-vs-marginal result with 10-seed 95% CIs.

Statistical-hygiene version of exp8/8b: the central positive finding (LCM ties
diagonal on MARGINAL predictive LL but closes most of the diagonal->full gap on
JOINT predictive LL) reported over 10 seeds with 95% confidence intervals.
Synthetic 1-D-correlated GLM, order KNOWN (a initialised at the natural order;
exp10 shows the learned coordinate cannot recover an unknown order anyway).
Uses the library's fixed auto-init fit (no hand-scaled initialisation).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

import exp10_ordering as E  # reuse simulate()
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)


def joint_ll(mu, Cov, y):
    T = len(y)
    L = torch.linalg.cholesky(Cov)
    r = torch.linalg.solve_triangular(L, (y - mu).unsqueeze(1), upper=False).squeeze(1)
    return float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                         + T * np.log(2 * np.pi)) / T)


def run(d, seed, n=3000, rho=0.9, ntest=400, noise=0.5, prior=1.0):
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    GGN = X.T @ X / noise**2
    Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
    mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))

    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():                       # order known -> natural init
        lcm.a.copy_(torch.arange(d) * -np.log(rho))
    lcm.fit_dense(GGN, n_steps=700, lr=5e-2)    # auto_init handles w/d scaling
    Cov_lcm = torch.linalg.inv(lcm.dense() + prior * torch.eye(d))
    Cov_diag = torch.diag(1.0 / (GGN.diagonal() + prior))

    Inoise = noise**2 * torch.eye(ntest)
    out = {"d": d, "seed": seed}
    for nm, C in [("diag", Cov_diag), ("lcm", Cov_lcm), ("full", Cov_full)]:
        var = torch.einsum("ti,ij,tj->t", Xt, C, Xt) + noise**2
        out[f"marg_{nm}"] = float((-0.5 * (torch.log(2 * np.pi * var)
                                           + (yt - mu_t) ** 2 / var)).mean())
        out[f"joint_{nm}"] = joint_ll(mu_t, Xt @ C @ Xt.T + Inoise, yt)
    out["gap_closed"] = ((out["joint_lcm"] - out["joint_diag"])
                         / (out["joint_full"] - out["joint_diag"] + 1e-12))
    return out


def ci(x):
    x = np.asarray(x)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    seeds = list(range(10))
    rows = [run(d, s) for d in (500, 1000, 1500) for s in seeds]
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp8c_joint_ci.csv", index=False)
    print("=== Joint vs marginal predictive test-LL, mean +/- 95% CI (10 seeds) ===\n")
    for d, sub in df.groupby("d"):
        print(f"d = {d}")
        for kind in ("marg", "joint"):
            parts = []
            for nm in ("diag", "lcm", "full"):
                m, h = ci(sub[f"{kind}_{nm}"])
                parts.append(f"{nm}={m:+.3f}+/-{h:.3f}")
            print(f"  {kind:6}: " + "   ".join(parts))
        gm, gh = ci(sub["gap_closed"])
        print(f"  joint gap-to-full closed by LCM: {100*gm:.0f}% +/- {100*gh:.0f}%\n")
    print("saved experiments/results/exp8c_joint_ci.csv")


if __name__ == "__main__":
    main()
