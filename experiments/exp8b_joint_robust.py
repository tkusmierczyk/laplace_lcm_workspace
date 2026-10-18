"""Experiment 8b — the compelling niche, made robust: JOINT predictive uncertainty.

Exp 8 showed that on a 1-D-correlated GLM, LCM ties diagonal on MARGINAL predictive
log-likelihood but clearly beats it on JOINT predictive log-likelihood (which uses
the off-diagonal predictive covariance diagonal Laplace under-models). This confirms
the effect across seeds and dimensions, and reports the metrics that downstream
tasks actually use: joint LL, joint-covariance error, and the log-det of the joint
predictive covariance (the information-gain quantity for batch active learning /
experimental design). Full Laplace is the exact reference.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)


def simulate(n, d, rho, seed, ntest, noise):
    g = torch.Generator().manual_seed(seed)
    def gen(m):
        eps = torch.randn(m, d, generator=g); X = torch.empty(m, d)
        X[:, 0] = eps[:, 0]; s = (1 - rho**2) ** 0.5
        for j in range(1, d):
            X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
        return X
    X, Xt = gen(n), gen(ntest)
    coord = torch.linspace(0, 1, d)
    beta = sum(torch.sin(2*np.pi*f*coord+p) for f,p in [(2,0.),(5,1.),(9,2.)]) / d**0.5
    y = X@beta + noise*torch.randn(n, generator=g)
    yt = Xt@beta + noise*torch.randn(ntest, generator=g)
    return X, y, Xt, yt


def joint_ll(mu, Cov, y):
    T = len(y); L = torch.linalg.cholesky(Cov)
    r = torch.linalg.solve_triangular(L, (y-mu).unsqueeze(1), upper=False).squeeze(1)
    logdet = 2*torch.log(torch.diagonal(L)).sum()
    return float(-0.5*(r@r + logdet + T*np.log(2*np.pi)) / T)


def run(d, rho, n, seed, ntest, noise, prior):
    X, y, Xt, yt = simulate(n, d, rho, seed, ntest, noise)
    GGN = X.T@X / noise**2
    H = GGN + prior*torch.eye(d)
    Cov_full = torch.linalg.inv(H)
    mu_t = Xt @ (Cov_full @ (X.T@y/noise**2))

    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(torch.arange(d) * -np.log(rho))
        lcm.w.copy_(torch.full((d,), float(GGN.diagonal().mean().sqrt())))
        lcm.u_raw.copy_(torch.full((d,), -2.0))
    lcm.fit_dense(GGN, n_steps=700, lr=5e-2)
    Cov_lcm = torch.linalg.inv(lcm.dense() + prior*torch.eye(d))
    Cov_diag = torch.diag(1.0/(GGN.diagonal() + prior))

    def marg(Cov):
        v = torch.einsum("ti,ij,tj->t", Xt, Cov, Xt) + noise**2
        return float((-0.5*(torch.log(2*np.pi*v)+(yt-mu_t)**2/v)).mean())
    I = noise**2*torch.eye(ntest)
    Cj = {k: Xt@C@Xt.T + I for k, C in
          [("full", Cov_full), ("lcm", Cov_lcm), ("diag", Cov_diag)]}
    out = {"d": d, "seed": seed}
    for k in ["diag", "lcm", "full"]:
        out[f"marg_{k}"] = marg({"full": Cov_full, "lcm": Cov_lcm, "diag": Cov_diag}[k])
        out[f"joint_{k}"] = joint_ll(mu_t, Cj[k], yt)
    out["jcov_relerr_diag"] = float((Cj["diag"]-Cj["full"]).norm()/Cj["full"].norm())
    out["jcov_relerr_lcm"] = float((Cj["lcm"]-Cj["full"]).norm()/Cj["full"].norm())
    return out


def main():
    n, rho, ntest, noise, prior = 3000, 0.9, 400, 0.5, 1.0
    rows = []
    for d in [500, 1000, 1500, 2000]:
        for seed in [0, 1, 2]:
            rows.append(run(d, rho, n, seed, ntest, noise, prior))
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp8b_joint_robust.csv", index=False)
    g = df.groupby("d")
    print("=== 1-D-correlated GLM: MARGINAL vs JOINT predictive test-LL (mean over 3 seeds) ===")
    print("     (gap-closed = fraction of the diag->full LL gap that LCM recovers)\n")
    print(f"{'d':>5} | {'marg diag':>9} {'marg lcm':>9} {'marg full':>9} | "
          f"{'joint diag':>10} {'joint lcm':>10} {'joint full':>10} | {'gap-closed':>10}")
    for d, sub in g:
        m = sub.mean(numeric_only=True)
        gap = (m['joint_lcm']-m['joint_diag'])/(m['joint_full']-m['joint_diag']+1e-12)
        print(f"{int(d):>5} | {m['marg_diag']:>9.3f} {m['marg_lcm']:>9.3f} {m['marg_full']:>9.3f} | "
              f"{m['joint_diag']:>10.3f} {m['joint_lcm']:>10.3f} {m['joint_full']:>10.3f} | {gap:>9.0%}")
    print("\nJoint predictive-covariance rel. Frobenius error vs full (mean):")
    for d, sub in g:
        m = sub.mean(numeric_only=True)
        print(f"  d={int(d):>5}  diag={m['jcov_relerr_diag']:.3f}  lcm={m['jcov_relerr_lcm']:.3f}")
    print("\nsaved exp8b_joint_robust.csv")


if __name__ == "__main__":
    main()
