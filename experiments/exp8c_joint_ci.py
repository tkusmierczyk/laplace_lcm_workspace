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
import common as G  # GLM helpers live in the shared harness

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
    """Posteriors and predictives come from `laplace-torch` via `glm_laplace`, so
    these numbers describe the shipped `LCMLaplace` / `DiagLaplace` / `FullLaplace`
    classes rather than a parallel implementation.

    The LCM `a` is initialised at the natural order through the library's own
    `lcm_init_a_scale`: the default init is `linspace(-1, 1, d) * init_a_scale`, so
    `init_a_scale = -log(rho) * d / 2` reproduces the true AR(1) spacing
    `-log(rho)` between adjacent coordinates (the kernel depends only on
    differences, so the offset is irrelevant).
    """
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    w_map = G.glm_map_weights(X, y, prior, noise)

    las = {
        "diag": G.fit_glm(X, y, "diag", prior, noise, w_map=w_map),
        "lcm": G.fit_glm(X, y, "lcm", prior, noise, w_map=w_map,
                         lcm_steps=700, lcm_lr=5e-2,
                         lcm_init_a_scale=-np.log(rho) * d / 2),
        "full": G.fit_glm(X, y, "full", prior, noise, w_map=w_map),
    }

    out = {"d": d, "seed": seed}
    for nm, la in las.items():
        mu_m, var = G.glm_marginal(la, Xt, noise)
        out[f"marg_{nm}"] = float((-0.5 * (torch.log(2 * np.pi * var)
                                           + (yt - mu_m) ** 2 / var)).mean())
        mu_j, Cj = G.glm_joint(la, Xt, noise)
        out[f"joint_{nm}"] = joint_ll(mu_j, 0.5 * (Cj + Cj.T), yt)
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
