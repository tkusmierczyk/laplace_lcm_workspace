"""Controls answering: 'LCM sorts internally -- so is the ordering an optimization
problem or an architecture problem?' On PERMUTED-feature GLM data (order latent):

  C0  uninformed init + learn a      -- can gradient descent DISCOVER the order?
  C1  true-order init + learn a      -- permutation-equivariance: given the right
                                        coordinates, does the model match full?
  C1' true-order init, a FROZEN      -- pure model, no optimizer at all.
  C2  seriation-warm-start + refine  -- can the method bootstrap its own order?

If C1/C1' ~ full but C0/C2 ~ diag, the model is order-agnostic (internal sort works)
and the ONLY difficulty is that fitting `a` cannot discover a latent order.
"""
from __future__ import annotations

import warnings
import numpy as np
import torch
from scipy.stats import spearmanr

import exp10_ordering as E
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

d, n, rho, ntest, noise, prior = 300, 2000, 0.9, 300, 0.5, 1.0
SP = -np.log(rho)


def fit_and_eval(a_init, GGN, Xt, yt, mu_t, true_order, learn=True, steps=1000):
    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(a_init)
    if not learn:
        lcm.a.requires_grad_(False)
    lcm.fit_dense(GGN, n_steps=steps, lr=5e-2)
    C = torch.linalg.inv(lcm.dense() + prior * torch.eye(d))
    Cj = Xt @ C @ Xt.T + noise**2 * torch.eye(ntest)
    L = torch.linalg.cholesky(Cj)
    r = torch.linalg.solve_triangular(L, (yt - mu_t).unsqueeze(1), upper=False).squeeze(1)
    jll = float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                        + ntest * np.log(2 * np.pi)) / ntest)
    sp = float(abs(spearmanr(lcm.a.detach().numpy(), true_order.numpy()).correlation))
    return jll, sp


for seed in [0, 1]:
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    g = torch.Generator().manual_seed(2000 + seed)
    perm = torch.randperm(d, generator=g)
    Xp, Xtp = X[:, perm], Xt[:, perm]
    GGN = Xp.T @ Xp / noise**2
    Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
    mu_t = Xtp @ (Cov_full @ (Xp.T @ y / noise**2))
    true_order = perm.double()

    def jll_of(C):
        Cj = Xtp @ C @ Xtp.T + noise**2 * torch.eye(ntest)
        L = torch.linalg.cholesky(Cj)
        r = torch.linalg.solve_triangular(L, (yt - mu_t).unsqueeze(1), upper=False).squeeze(1)
        return float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                             + ntest * np.log(2 * np.pi)) / ntest)

    ll_full = jll_of(Cov_full)
    ll_diag = jll_of(torch.diag(1.0 / (GGN.diagonal() + prior)))
    ranks = torch.empty(d)
    ranks[E.fiedler_order(GGN)] = torch.arange(d).double()

    print(f"[seed {seed}] full={ll_full:+.3f}  diag={ll_diag:+.3f}", flush=True)
    for tag, a_init, learn in [
        ("C0  uninformed init + learn ", torch.linspace(0, d * 0.1, d), True),
        ("C1  true-order  init + learn", true_order * SP, True),
        ("C1' true-order  init  FROZEN", true_order * SP, False),
        ("C2  seriation   init + learn", ranks * SP, True),
    ]:
        jll, sp = fit_and_eval(a_init, GGN, Xtp, yt, mu_t, true_order, learn=learn)
        verdict = "~full" if jll > ll_diag + 0.6 * (ll_full - ll_diag) else \
                  ("~diag" if jll < ll_diag + 0.2 * (ll_full - ll_diag) else "mid")
        print(f"   {tag}: jLL={jll:+.3f}  order|rho|={sp:.3f}  [{verdict}]", flush=True)
