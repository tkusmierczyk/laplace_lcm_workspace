"""Exp 23 — a probe a temperature cannot fake: conditional (transductive) prediction.

THE GAP. Exps 17-21 established that every probe this project tried is *scale*-dominated:
given one oracle scalar variance scale, a diagonal posterior matches, and sometimes beats,
the exact one on covariate-shifted likelihood (Exp 18: diag 1.4107 vs full 1.4154). That
left the central open problem -- **no probe here separates off-diagonal structure from a
correctly scaled diagonal** -- and with it the suspicion that off-diagonals may simply not
matter for anything measurable.

THE IDEA. Marginal likelihood averages over points, so only per-point variances survive.
*Conditional* prediction does not. Reveal the labels of some test points and predict the
rest: the conditional mean

    mu_{Q|O} = mu_Q + t Sigma_QO (t Sigma_OO + sigma^2 I)^-1 (y_O - mu_O)

is driven entirely by the off-diagonal block Sigma_QO, and as t -> infinity it becomes

    mu_Q + Sigma_QO Sigma_OO^-1 (y_O - mu_O) ,

which is **exactly invariant** to rescaling the posterior covariance by any t. A scalar
temperature cannot touch it. So the error of that estimator is a pure structure metric:
whatever separation appears there is attributable to the off-diagonals alone.

HYPOTHESIS. Structured posteriors (full, LCM) transfer information between correlated test
points; a diagonal posterior cannot, at any temperature.

WHAT WOULD MAKE US DROP THE IDEA. If a diagonal posterior conditions as well as the exact
one in the scale-invariant limit, off-diagonal covariance carries no usable information for
prediction in this model class, and the LCM programme has no predictive case left.

Controls. rho=0 is a negative control: with uncorrelated features every family should tie,
and if they do not, the probe is measuring something else. The exact posterior ("full") is
the gold standard, not a competitor: the generative prior precision and noise are handed to
every arm, so `full` *is* the true Bayesian posterior and the chain/diagonal are
approximations of a known target.

Writes results/exp23_conditional.csv.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common  # noqa: E402

STRUCTURES = [("full", "full", {}), ("lcm", "lcm", {}), ("diag", "diag", {})]


def ar1_cov(d: int, rho: float) -> torch.Tensor:
    i = torch.arange(d, dtype=torch.float64)
    return rho ** (i[:, None] - i[None, :]).abs()


def make_data(d, rho, n_train, n_test, sigma, prior, seed):
    g = torch.Generator().manual_seed(seed)
    L = torch.linalg.cholesky(ar1_cov(d, rho) + 1e-10 * torch.eye(d))
    draw = lambda n: torch.randn(n, d, generator=g) @ L.T          # noqa: E731
    w = torch.randn(d, generator=g) / math.sqrt(prior)
    Xtr, Xte = draw(n_train), draw(n_test)
    noise = lambda X: X @ w + sigma * torch.randn(len(X), generator=g)  # noqa: E731
    return Xtr, noise(Xtr), Xte, noise(Xte), w


def conditional_mean(Sig, mu, y_obs, io, iq, sigma, t):
    """``mu_{Q|O}`` at posterior scale ``t``; ``t=None`` gives the scale-free limit."""
    Soo, Sqo = Sig[io][:, io], Sig[iq][:, io]
    if t is None:                                   # drop the noise: exactly scale-free
        jit = 1e-8 * float(torch.diagonal(Soo).mean())
        A = Soo + jit * torch.eye(len(io), dtype=Sig.dtype)
    else:
        Soo, Sqo = t * Soo, t * Sqo
        A = Soo + sigma**2 * torch.eye(len(io), dtype=Sig.dtype)
    gain = torch.linalg.solve(A, (y_obs - mu[io]).unsqueeze(-1)).squeeze(-1)
    return mu[iq] + Sqo @ gain


def conditional_nll(Sig, mu, y_obs, y_q, io, iq, sigma, t):
    Soo, Sqo, Sqq = t * Sig[io][:, io], t * Sig[iq][:, io], t * Sig[iq][:, iq]
    Io = torch.eye(len(io), dtype=Sig.dtype)
    A = Soo + sigma**2 * Io
    gain = torch.linalg.solve(A, Sqo.T).T
    m = mu[iq] + gain @ (y_obs - mu[io])
    S = Sqq + sigma**2 * torch.eye(len(iq), dtype=Sig.dtype) - gain @ Sqo.T
    S = 0.5 * (S + S.T)
    e = (y_q - m).unsqueeze(-1)
    L = torch.linalg.cholesky(S + 1e-12 * torch.eye(len(iq), dtype=Sig.dtype))
    sol = torch.cholesky_solve(e, L)
    quad = float((e * sol).sum())
    logdet = float(2 * torch.log(torch.diagonal(L)).sum())
    return 0.5 * (len(iq) * math.log(2 * math.pi) + logdet + quad) / len(iq)


def run_seed(args, rho, seed):
    d, sigma, prior = args.d, args.sigma, args.prior
    Xtr, ytr, Xte, yte, _ = make_data(d, rho, args.n_train, args.n_test, sigma, prior, seed)
    g = torch.Generator().manual_seed(1000 + seed)
    perm = torch.randperm(args.n_test, generator=g)
    io, iq = perm[: args.n_test // 2], perm[args.n_test // 2:]
    grid = np.logspace(-2, 4, 49)
    rows = []

    for label, structure, kw in STRUCTURES:
        t0 = time.perf_counter()
        la = common.fit_glm(Xtr, ytr, structure, prior, sigma, **kw)
        mu, S_tot = common.glm_joint(la, Xte, sigma)
        Sig = S_tot - sigma**2 * torch.eye(len(S_tot), dtype=S_tot.dtype)  # epistemic only
        err = lambda m: float(((m - yte[iq]) ** 2).mean().sqrt())          # noqa: E731

        rmse_marg = float(((mu[iq] - yte[iq]) ** 2).mean().sqrt())
        rmse_t1 = err(conditional_mean(Sig, mu, ytr.new_tensor(yte[io]), io, iq, sigma, 1.0))
        rmse_lim = err(conditional_mean(Sig, mu, yte[io], io, iq, sigma, None))
        best_r = min((err(conditional_mean(Sig, mu, yte[io], io, iq, sigma, t)), t) for t in grid)
        nll_t1 = conditional_nll(Sig, mu, yte[io], yte[iq], io, iq, sigma, 1.0)
        best_n = min((conditional_nll(Sig, mu, yte[io], yte[iq], io, iq, sigma, t), t)
                     for t in grid)
        rows.append({
            "rho": rho, "seed": seed, "structure": label,
            "rmse_marginal": rmse_marg, "rmse_cond_t1": rmse_t1,
            "rmse_cond_limit": rmse_lim, "rmse_cond_best": best_r[0], "t_rmse": best_r[1],
            "nll_cond_t1": nll_t1, "nll_cond_best": best_n[0], "t_nll": best_n[1],
            "fit_s": time.perf_counter() - t0,
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--n-train", type=int, default=60, help="< d, so the posterior is broad")
    p.add_argument("--n-test", type=int, default=40, help="half revealed, half queried")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--prior", type=float, default=1.0, help="true prior precision; given to all")
    p.add_argument("--rhos", default="0.0,0.9,0.99")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--tag", default="exp23_conditional")
    args = p.parse_args()
    args.rhos = [float(r) for r in args.rhos.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    rows = [r for rho in args.rhos for s in range(args.seeds) for r in run_seed(args, rho, s)]
    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag,
                               config={**vars(args), "elapsed_s": time.perf_counter() - t0})

    labels = [s[0] for s in STRUCTURES]
    print(f"\n  Bayesian linear model, d={args.d}, n_train={args.n_train} (< d), "
          f"sigma={args.sigma}, {args.seeds} seeds")
    print(f"  {args.n_test // 2} test labels revealed, {args.n_test - args.n_test // 2} queried; "
          f"prior precision {args.prior} handed to every arm, so `full` is the exact posterior")
    for rho, grp in df.groupby("rho"):
        print(f"\n  rho = {rho}" + ("   (negative control: everything should tie)"
                                    if rho == 0 else ""))
        print(f"    {'structure':10}{'RMSE marg':>11}{'RMSE t=1':>10}"
              f"{'RMSE limit':>12}{'RMSE best t':>12}{'NLL t=1':>10}{'NLL best t':>12}")
        for k in labels:
            g = grp[grp.structure == k]
            print(f"    {k:10}{g.rmse_marginal.mean():11.4f}{g.rmse_cond_t1.mean():10.4f}"
                  f"{g.rmse_cond_limit.mean():12.4f}{g.rmse_cond_best.mean():12.4f}"
                  f"{g.nll_cond_t1.mean():10.4f}{g.nll_cond_best.mean():12.4f}")
        # The limit column is exactly scale-invariant, so a paired test on it is a
        # statement about off-diagonal structure and nothing else.
        print("    paired vs full on the SCALE-INVARIANT limit (positive = worse than exact):")
        ref = grp[grp.structure == "full"].set_index("seed").rmse_cond_limit
        for k in labels:
            if k == "full":
                continue
            dd = grp[grp.structure == k].set_index("seed").rmse_cond_limit - ref
            sem = dd.std() / math.sqrt(len(dd))
            sig = abs(dd.mean()) / sem if sem > 0 else float("inf")
            print(f"      {k:10} {dd.mean():+8.4f} +- {sem:.4f}  ({sig:4.1f} sigma)"
                  f"{'  <- separated' if sig > 3 else '  <- within noise'}")
        # The one comparison the project has never been able to make: chain vs diagonal
        # with the scale advantage removed by construction.
        dd = (grp[grp.structure == "diag"].set_index("seed").rmse_cond_limit
              - grp[grp.structure == "lcm"].set_index("seed").rmse_cond_limit)
        sem = dd.std() / math.sqrt(len(dd))
        sig = dd.mean() / sem if sem > 0 else float("inf")
        full_gap = (grp[grp.structure == "diag"].rmse_cond_limit.mean()
                    - grp[grp.structure == "full"].rmse_cond_limit.mean())
        closed = 100 * dd.mean() / full_gap if full_gap > 0 else float("nan")
        print(f"    LCM vs diag on that metric: {dd.mean():+8.4f} +- {sem:.4f} "
              f"({sig:4.1f} sigma) -> " + (f"LCM closes {closed:.0f}% of the diag->exact gap"
                                           if sig > 3 else "no structural advantage"))
        for k in labels:
            g = grp[grp.structure == k]
            print(f"      conditioning changes {k:5}'s RMSE {g.rmse_marginal.mean():.4f}"
                  f" -> {g.rmse_cond_limit.mean():.4f} "
                  f"({100 * (1 - g.rmse_cond_limit.mean() / g.rmse_marginal.mean()):+.1f}%)")
    print(f"\n  total {time.perf_counter() - t0:.1f}s\nsaved {path}")


if __name__ == "__main__":
    main()
