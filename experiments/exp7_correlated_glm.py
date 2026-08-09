"""Experiment 7 — LCM where its assumption is CORRECT: 1-D-correlated GLM Laplace.

Weight-space Laplace failed for LCM because NN-layer curvature is Kronecker-
structured. Here we target the setting LCM is actually designed for: a Bayesian
linear model whose *features have a natural 1-D order with exponentially-decaying
correlation* (genomics / linkage-disequilibrium, spectroscopy, time-series
regression). The GGN then has genuine chain structure, KFAC does not apply (no
layers), and dense Laplace is O(d^2) memory / O(d^3) compute.

Posterior over the d coefficients, fit with:
  * Diag — diagonal Laplace,
  * Full — exact dense Laplace (computed via the n x n dual/Woodbury form, so it
           is an exact reference at every d), and
  * LCM  — fit MATRIX-FREE to the GGN via `fit_frobenius` on the curvature-weighted
           feature vectors, predicting with the O(d) Kalman quadratic form; never
           forms a d x d matrix (3d parameters total).

Reported: test log-likelihood (predictive calibration), relative error of the
predictive variance vs the exact posterior, and cost/scaling to d >> n.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

from laplace.lcm import LCMCovariance, kalman_stats

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def simulate(n, d, rho, seed, ntest=2000, noise=0.5):
    g = torch.Generator().manual_seed(seed)

    def gen(m):  # AR(1) features along the 1-D coordinate; O(md), no d x d matrix
        eps = torch.randn(m, d, generator=g)
        X = torch.empty(m, d)
        X[:, 0] = eps[:, 0]
        s = (1 - rho**2) ** 0.5
        for j in range(1, d):
            X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
        return X

    X, Xt = gen(n), gen(ntest)
    coord = torch.linspace(0, 1, d)
    beta = sum(torch.sin(2 * np.pi * f * coord + p)
               for f, p in [(2, 0.0), (5, 1.0), (9, 2.0)]) / d**0.5
    y = X @ beta + noise * torch.randn(n, generator=g)
    yt = Xt @ beta + noise * torch.randn(ntest, generator=g)
    return X.to(DEV), y.to(DEV), Xt.to(DEV), yt.to(DEV), noise, rho


def dual_map(X, y, prior_prec, noise):
    """Ridge MAP via the n x n dual form: beta = X^T (X X^T + prior*noise^2 I)^-1 y."""
    n = X.shape[0]
    Kmat = X @ X.T + prior_prec * noise**2 * torch.eye(n, device=X.device)
    return X.T @ torch.linalg.solve(Kmat, y)


def full_predvar(X, Xt, prior_prec, noise):
    """Exact x*^T H^-1 x* for H = X^T X/noise^2 + prior I, via Woodbury (n x n)."""
    n = X.shape[0]
    lam = prior_prec
    M = noise**2 * lam * torch.eye(n, device=X.device) + X @ X.T  # n x n
    XXt = Xt @ X.T                                                # (t, n)
    quad = (Xt**2).sum(1) / lam - (XXt @ torch.linalg.solve(M, XXt.T)).diagonal() / lam
    return quad


def gaussian_ll(mu, var, y):
    return float((-0.5 * (torch.log(2 * np.pi * var) + (y - mu) ** 2 / var)).mean())


def fit_lcm_to_ggn(V, ggn_diag, rho_guess, steps=800, lr=5e-2, batch_size=None):
    """Fit LCM to the GGN whose empirical form is sum_n v_n v_n^T (V = curvature-
    weighted features, one row per sample). fit_frobenius targets (1/N) sum v v^T,
    so the fitted A approximates GGN / N.

    We only seed the *ordering* prior: the natural index order with spacing
    -log(rho_guess) (domain knowledge fit_frobenius's auto_init does not have
    access to). The w/d *scale* is left to the library's default `auto_init=True`,
    which derives it from the empirical diagonal of the vectors -- doing this by
    hand here previously used a stale, badly-scaled nugget init and was the root
    cause of a large matrix-free-vs-dense fit gap (see PUBLICATION_ASSESSMENT.md)."""
    N, d = V.shape
    lcm = LCMCovariance(d, dtype=torch.float64).to(V.device)
    with torch.no_grad():
        spacing = max(-np.log(max(rho_guess, 1e-3)), 1e-3)
        lcm.a.copy_(torch.arange(d, device=V.device) * spacing)
    lcm.fit_frobenius(V, n_steps=steps, lr=lr, batch_size=batch_size)
    return lcm


def run(d, rho=0.9, n=3000, seed=0, prior_prec=1.0, lcm_steps=800):
    X, y, Xt, yt, noise, rho = simulate(n, d, rho, seed)
    beta = dual_map(X, y, prior_prec, noise)
    mu_t = Xt @ beta
    out = {"d": d, "rho": rho, "n": n}

    V = X / noise
    ggn_diag = (V**2).sum(0)

    # exact full reference (Woodbury, n x n)
    var_full = full_predvar(X, Xt, prior_prec, noise) + noise**2
    out["ll_full"] = gaussian_ll(mu_t, var_full, yt)

    # diagonal
    var_diag = (Xt**2 / (ggn_diag + prior_prec)).sum(1) + noise**2
    out["ll_diag"] = gaussian_ll(mu_t, var_diag, yt)
    out["pvar_relerr_diag"] = float(((var_diag - var_full).abs() / var_full).mean())

    # LCM (matrix-free)
    t0 = time.perf_counter()
    lcm = fit_lcm_to_ggn(V, ggn_diag, rho, steps=lcm_steps)
    out["t_lcm_fit"] = time.perf_counter() - t0
    a_p, w_p, d_p = lcm.scaled_params(float(n), prior_prec * torch.ones(d, device=DEV))
    S, v = kalman_stats(a_p, w_p, d_p, Xt)
    var_lcm = (v**2 / S).sum(-1) + noise**2
    out["ll_lcm"] = gaussian_ll(mu_t, var_lcm, yt)
    out["pvar_relerr_lcm"] = float(((var_lcm - var_full).abs() / var_full).mean())
    return out


def main():
    print("=== LCM vs Diag vs exact Full for a 1-D-correlated GLM (rho=0.9, n=3000) ===")
    print("    (test-LL higher=better; pred-var rel.err vs exact posterior lower=better)\n")
    rows = []
    for d in [500, 1000, 2000, 5000, 10000, 20000]:
        seeds = [0, 1, 2] if d <= 5000 else [0]
        res = [run(d, seed=s) for s in seeds]
        agg = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
        rows.append(agg)
        print(f"  d={d:6d} | test-LL  diag={agg['ll_diag']:+.3f}  lcm={agg['ll_lcm']:+.3f}  "
              f"full={agg['ll_full']:+.3f}  | pred-var rel.err  diag={agg['pvar_relerr_diag']:.3f} "
              f" lcm={agg['pvar_relerr_lcm']:.3f}  | LCM fit {agg['t_lcm_fit']:.1f}s")
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp7_correlated_glm.csv", index=False)
    print("\nsaved exp7_correlated_glm.csv")


if __name__ == "__main__":
    main()
