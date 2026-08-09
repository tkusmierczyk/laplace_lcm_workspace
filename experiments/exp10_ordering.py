"""Experiment 10 — does the LEARNED ordering actually matter? (the decisive baseline)

The central reviewer objection to LCM's "learned 1-D coordinate" is: on data whose
features are already ordered, a plain banded/AR precision in the *given* order would
match LCM, so the learning does nothing. We test this directly on the 1-D-correlated
GLM (LCM's home regime, n >= d), measuring the JOINT predictive test-LL (the metric
where structured covariance helps) against the exact full posterior.

Methods (all approximate the same GGN, then form the posterior + joint predictive):
  * Full            -- exact reference.
  * Diag            -- diagonal Laplace (lower bound).
  * LCM-fixed(nat)  -- Laplace-kernel precision with the ORDER FROZEN to the given
                       feature order; only the global spacing (decay) and w/d are fit.
                       This is "you know the order, just fit a banded/AR model".
  * LCM-fixed(seri) -- same, but the order is recovered by classical spectral
                       (Fiedler) SERIATION of the empirical correlation.
  * LCM-learned     -- ours: the full latent coordinate `a` is learned from an
                       uninformed (identity) initialization.

Two data conditions:
  (A) NATURAL order  -- features in their generating order (order is "known").
  (B) PERMUTED order -- features randomly shuffled (order is latent/unknown).

Decisive outcome: if in (B) LCM-learned recovers full-level joint LL while
LCM-fixed(nat) collapses toward diagonal, the learned ordering is doing real,
necessary work; the seriation baseline shows whether a classical two-step method
would have sufficed.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from laplace.lcm import LCMCovariance, _inverse_softplus

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)
EPS = 1e-6


# --------------------------------------------------------------------- data
def simulate(n, d, rho, seed, ntest, noise):
    g = torch.Generator().manual_seed(seed)

    def gen(m):
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
    return X, y, Xt, yt


# ------------------------------------------------------------------- metrics
def joint_ll(mu, Cov, y):
    T = len(y)
    L = torch.linalg.cholesky(Cov)
    r = torch.linalg.solve_triangular(L, (y - mu).unsqueeze(1), upper=False).squeeze(1)
    logdet = 2 * torch.log(torch.diagonal(L)).sum()
    return float(-0.5 * (r @ r + logdet + T * np.log(2 * np.pi)) / T)


def marg_ll(mu, Cov, Xt, y, noise):
    var = torch.einsum("ti,ij,tj->t", Xt, Cov, Xt) + noise**2
    return float((-0.5 * (torch.log(2 * np.pi * var) + (y - mu) ** 2 / var)).mean())


# ------------------------------------------------------ ordering / fitting
def fiedler_order(GGN):
    """Classical spectral seriation: order variables by the Fiedler vector of the
    Laplacian of the |correlation| similarity graph."""
    d = GGN.shape[0]
    dinv = GGN.diagonal().clamp(min=1e-12).rsqrt()
    R = (GGN * dinv[:, None]) * dinv[None, :]
    S = R.abs()
    L = torch.diag(S.sum(1)) - S
    evals, evecs = torch.linalg.eigh(L)
    return torch.argsort(evecs[:, 1])  # order of positions -> original indices


def fit_fixed_order(GGN, order, steps=800, lr=5e-2):
    """Laplace-kernel precision with the permutation FROZEN to `order` (a_i = s * rank_i);
    only the global spacing `s`, scales `w`, and nugget `d` are learned."""
    d = GGN.shape[0]
    ranks = torch.empty(d, dtype=torch.float64)
    ranks[order] = torch.arange(d, dtype=torch.float64)  # position of each variable
    diag_est = GGN.diagonal().clamp(min=1e-12)
    log_s = torch.zeros((), requires_grad=True)
    w = torch.sqrt(0.85 * diag_est).clone().requires_grad_(True)
    u = _inverse_softplus(0.15 * diag_est).clone().requires_grad_(True)
    opt = torch.optim.Adam([log_s, w, u], lr=lr)
    tnorm2 = (GGN**2).sum().clamp(min=1e-12)
    for _ in range(steps):
        opt.zero_grad()
        a = log_s.exp() * ranks
        K = torch.exp(-torch.abs(a[:, None] - a[None, :]))
        A = torch.diag(torch.nn.functional.softplus(u) + EPS) + w[:, None] * K * w[None, :]
        loss = ((A - GGN) ** 2).sum() / tnorm2
        loss.backward()
        opt.step()
    with torch.no_grad():
        a = log_s.exp() * ranks
        K = torch.exp(-torch.abs(a[:, None] - a[None, :]))
        A = torch.diag(torch.nn.functional.softplus(u) + EPS) + w[:, None] * K * w[None, :]
    return A


def fit_learned(GGN, steps=800, lr=5e-2):
    """LCM with the full latent coordinate learned from an uninformed init."""
    d = GGN.shape[0]
    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(torch.linspace(0.0, float(d) * 0.1, d))  # uninformed identity init
    lcm.fit_dense(GGN, n_steps=steps, lr=lr)  # learns a, w, d (auto_init scales w/d)
    return lcm.dense().detach(), lcm.a.detach()


# ----------------------------------------------------------------- one run
def run(d, seed, condition, n=3000, rho=0.9, ntest=400, noise=0.5, prior=1.0):
    X, y, Xt, yt = simulate(n, d, rho, seed, ntest, noise)
    if condition == "permuted":
        g = torch.Generator().manual_seed(1000 + seed)
        perm = torch.randperm(d, generator=g)
        X, Xt = X[:, perm], Xt[:, perm]
        # X[:, perm] puts original column perm[j] at position j, and original column
        # i carries latent coordinate i -- so permuted column j has coordinate perm[j].
        # (Verified empirically: freezing a=perm fits the GGN to 0.14 rel-Frobenius,
        # whereas a=argsort(perm) gives 0.98, i.e. no structure captured.)
        true_order = perm
    else:
        true_order = torch.arange(d)

    GGN = X.T @ X / noise**2
    H = GGN + prior * torch.eye(d)
    Cov_full = torch.linalg.inv(H)
    mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))
    Inoise = noise**2 * torch.eye(ntest)

    def posterior_cov(A):
        return torch.linalg.inv(A + prior * torch.eye(d))

    covs = {"full": Cov_full,
            "diag": torch.diag(1.0 / (GGN.diagonal() + prior)),
            "lcm_fixed_nat": posterior_cov(fit_fixed_order(GGN, torch.arange(d))),
            "lcm_fixed_seri": posterior_cov(fit_fixed_order(GGN, fiedler_order(GGN)))}
    A_learned, a_learned = fit_learned(GGN)
    covs["lcm_learned"] = posterior_cov(A_learned)

    out = {"d": d, "seed": seed, "condition": condition}
    for name, C in covs.items():
        Cj = Xt @ C @ Xt.T + Inoise
        out[f"joint_{name}"] = joint_ll(mu_t, Cj, yt)
        out[f"marg_{name}"] = marg_ll(mu_t, C, Xt, yt, noise)
    # how well did learning recover the latent order?
    out["order_spearman"] = float(abs(spearmanr(
        a_learned.cpu().numpy(),
        true_order.double().cpu().numpy()).correlation))
    return out


def main():
    d, seeds = 800, [0, 1, 2]
    rows = []
    for cond in ["natural", "permuted"]:
        for s in seeds:
            rows.append(run(d, s, cond))
            r = rows[-1]
            print(f"[{cond:8} seed {s}] joint-LL  full={r['joint_full']:+.3f}  "
                  f"diag={r['joint_diag']:+.3f}  fix-nat={r['joint_lcm_fixed_nat']:+.3f}  "
                  f"fix-seri={r['joint_lcm_fixed_seri']:+.3f}  learned={r['joint_lcm_learned']:+.3f}  "
                  f"| order-recov(rho)={r['order_spearman']:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp10_ordering.csv", index=False)

    print("\n=== JOINT predictive test-LL, mean over seeds (higher=better) ===")
    methods = ["diag", "lcm_fixed_nat", "lcm_fixed_seri", "lcm_learned", "full"]
    for cond, sub in df.groupby("condition"):
        m = sub.mean(numeric_only=True)
        def gc(x):  # % of diag->full gap closed
            return 100 * (m[f"joint_{x}"] - m["joint_diag"]) / (m["joint_full"] - m["joint_diag"] + 1e-12)
        print(f"\n[{cond}]  full={m['joint_full']:+.3f}  diag={m['joint_diag']:+.3f}")
        for x in ["lcm_fixed_nat", "lcm_fixed_seri", "lcm_learned"]:
            print(f"    {x:16s} joint-LL={m['joint_'+x]:+.3f}  gap-closed={gc(x):5.0f}%")
        print(f"    order recovery (|Spearman| of learned coord vs latent order): "
              f"{m['order_spearman']:.2f}")
    print("\nsaved experiments/results/exp10_ordering.csv")


if __name__ == "__main__":
    main()
