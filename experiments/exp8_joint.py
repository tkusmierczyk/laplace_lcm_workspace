"""Experiment 8 — diagnostic: does a well-fit LCM realize a joint-UQ advantage?

Settles two questions on the ideal 1-D-correlated GLM (AR(1) features, whose GGN
is exactly Laplace-kernel structured):
  1. Fit quality: can LCM actually fit the chain-structured GGN (dense vs
     matrix-free fit residual)?
  2. Where the advantage lives: MARGINAL predictive log-likelihood (per-point
     variance) vs JOINT predictive log-likelihood over a batch of test points
     (which uses the *off-diagonal* predictive covariance that diagonal Laplace
     gets wrong). Full Laplace is the exact reference.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
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
    """Log-likelihood of y under N(mu, Cov) (Cov = predictive cov incl. noise)."""
    T = len(y); L = torch.linalg.cholesky(Cov)
    r = torch.linalg.solve_triangular(L, (y-mu).unsqueeze(1), upper=False).squeeze(1)
    logdet = 2*torch.log(torch.diagonal(L)).sum()
    return float(-0.5*(r@r + logdet + T*np.log(2*np.pi)) / T)


def main():
    n, d, rho, ntest, noise, prior = 3000, 1500, 0.9, 400, 0.5, 1.0
    X, y, Xt, yt = simulate(n, d, rho, seed=0, ntest=ntest, noise=noise)

    # exact posterior precision and covariance
    GGN = X.T@X / noise**2
    H = GGN + prior*torch.eye(d)
    Cov_full = torch.linalg.inv(H)
    beta = Cov_full @ (X.T@y/noise**2)          # MAP
    mu_t = Xt@beta

    # --- fit LCM to the GGN, two ways ---
    def init_lcm():
        lcm = LCMCovariance(d, dtype=torch.float64)
        with torch.no_grad():
            lcm.a.copy_(torch.arange(d)* -np.log(rho))
            lcm.w.copy_(torch.full((d,), float(GGN.diagonal().mean().sqrt())))
            lcm.u_raw.copy_(torch.full((d,), -2.0))
        return lcm
    lcm_d = init_lcm(); rel_dense = lcm_d.fit_dense(GGN, n_steps=700, lr=5e-2)
    V = X/noise
    lcm_f = init_lcm(); lcm_f.fit_frobenius(V, n_steps=700, lr=5e-2)
    rel_free = float((lcm_f.dense()-GGN).norm()/GGN.norm())
    print(f"LCM fit residual to GGN:  dense={rel_dense:.4f}   matrix-free={rel_free:.4f}")
    diag_resid = float((torch.diag(GGN.diagonal())-GGN).norm()/GGN.norm())
    print(f"diagonal residual to GGN: {diag_resid:.4f}\n")

    # posterior covariances (dense here, for an exact diagnostic)
    A_lcm = lcm_d.dense() + prior*torch.eye(d)
    Cov_lcm = torch.linalg.inv(A_lcm)
    prec_diag = GGN.diagonal() + prior
    Cov_diag = torch.diag(1.0/prec_diag)

    # --- MARGINAL predictive log-likelihood ---
    def marg_ll(Cov):
        var = torch.einsum("ti,ij,tj->t", Xt, Cov, Xt) + noise**2
        return float((-0.5*(torch.log(2*np.pi*var)+(yt-mu_t)**2/var)).mean())
    print("MARGINAL predictive test-LL (higher=better):")
    print(f"  diag={marg_ll(Cov_diag):+.4f}   lcm={marg_ll(Cov_lcm):+.4f}   full={marg_ll(Cov_full):+.4f}")

    # --- JOINT predictive log-likelihood over the test batch ---
    I = noise**2*torch.eye(ntest)
    Cj_full = Xt@Cov_full@Xt.T + I
    Cj_lcm  = Xt@Cov_lcm @Xt.T + I
    Cj_diag = Xt@Cov_diag@Xt.T + I
    print("\nJOINT predictive test-LL over the batch (uses off-diagonal cov):")
    print(f"  diag={joint_ll(mu_t,Cj_diag,yt):+.4f}   lcm={joint_ll(mu_t,Cj_lcm,yt):+.4f}   full={joint_ll(mu_t,Cj_full,yt):+.4f}")

    # accuracy of the joint predictive covariance vs full
    print("\nRel. Frobenius error of the JOINT predictive covariance vs full:")
    print(f"  diag={float((Cj_diag-Cj_full).norm()/Cj_full.norm()):.4f}   "
          f"lcm={float((Cj_lcm-Cj_full).norm()/Cj_full.norm()):.4f}")
    print("\nlog-det of the joint predictive covariance (information content):")
    print(f"  diag={float(torch.logdet(Cj_diag)):.2f}   lcm={float(torch.logdet(Cj_lcm)):.2f}   "
          f"full={float(torch.logdet(Cj_full)):.2f}")


if __name__ == "__main__":
    main()
