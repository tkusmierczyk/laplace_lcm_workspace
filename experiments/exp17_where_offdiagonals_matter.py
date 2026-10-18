"""Exp 17 — where off-diagonal posterior covariance actually shows up.

The project's framing was "marginal metrics are blind to off-diagonal covariance, so
use joint ones". That is imprecise: the marginal predictive variance is x^T Sigma x,
which *does* depend on off-diagonals. The real reason our marginal metrics tied is
narrower, and it is a property of the TEST DISTRIBUTION, not of the metric:

  for in-distribution x ~ N(0, Sigma_f) and Sigma ~ (n Sigma_f / sigma^2)^-1,
      E[x^T Sigma x] = tr(Sigma Sigma_f) = sigma^2 d / n,
  so the mean predictive variance is sigma^2 (1 + d/n) whatever the covariance
  STRUCTURE is. Any approximation that gets the trace right matches on average.

So the question is not marginal-versus-joint, it is: which queries break that
identity? Four probes on the same three posteriors:

  (a) in-distribution inputs                  -- what every earlier experiment used
  (b) shifted inputs: identical marginal variances, correlation removed
  (c) the SUM of 20 independent predictions    -- a scalar needing the joint covariance
  (d) the SUM of 20 *correlated* predictions   -- same, but with real cross terms

Writes results/exp17_offdiagonals.csv.
"""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common as G  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

D, N, RHO, SIGMA, PRIOR, NTE, T, NB, SEEDS = 300, 1500, 0.9, 0.5, 1.0, 2000, 20, 100, 5


def _ar1(n_rows, d, rho, g):
    e = torch.randn(n_rows, d, generator=g)
    X = torch.empty(n_rows, d)
    X[:, 0] = e[:, 0]
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + math.sqrt(1 - rho**2) * e[:, j]
    return X


def _aggregate(la, Xb, yb, sigma):
    """Calibration and log-likelihood of 1^T f over one batch."""
    mu, Cj = G.glm_joint(la, Xb, sigma)
    ones = torch.ones(len(yb), dtype=Xb.dtype)
    v = float(ones @ Cj @ ones)
    e = float((yb - mu).sum())
    return e * e / v, -0.5 * (math.log(2 * math.pi * v) + e * e / v)


def tempered_ll(la, Xt, yt, sigma, grid=np.logspace(-1, 2.5, 36)):
    """Best marginal LL under one oracle scalar scale on the *epistemic* variance.

    Added after Exp 18/20 showed the shift probe can be won by variance inflation alone.
    If every structure reaches the same LL once rescaled, the probe is measuring the
    overall variance scale rather than off-diagonal structure.
    """
    mu, var = G.glm_marginal(la, Xt, sigma)
    epi = (var - sigma**2).clamp(min=0.0)
    best = (-float("inf"), None)
    for t in grid:
        v = t * epi + sigma**2
        ll = float((-0.5 * (torch.log(2 * math.pi * v) + (yt - mu) ** 2 / v)).mean())
        if ll > best[0]:
            best = (ll, float(t))
    return best


def run(seed, d=D, n=N, rho=RHO, sigma=SIGMA, prior=PRIOR, nte=NTE, T=T, nB=NB):
    g = torch.Generator().manual_seed(seed)
    coord = torch.linspace(0, 1, d)
    beta = sum(torch.sin(2 * math.pi * f * coord + p)
               for f, p in [(2, 0.0), (5, 1.0), (9, 2.0)]) / math.sqrt(d)

    X = _ar1(n, d, rho, g)
    y = X @ beta + sigma * torch.randn(n, generator=g)
    Xin = _ar1(nte, d, rho, g)
    yin = Xin @ beta + sigma * torch.randn(nte, generator=g)
    # shifted: same unit marginal variances, correlation removed
    Xsh = torch.randn(nte, d, generator=g)
    ysh = Xsh @ beta + sigma * torch.randn(nte, generator=g)
    # correlated query batches: T points clustered around a common base point
    base = _ar1(nB, d, rho, g)
    Xco = (base.repeat_interleave(T, 0) + 0.3 * _ar1(nB * T, d, rho, g))
    yco = Xco @ beta + sigma * torch.randn(nB * T, generator=g)

    w_map = G.glm_map_weights(X, y, prior, sigma)
    las = {
        "diag": G.fit_glm(X, y, "diag", prior, sigma, w_map=w_map),
        "lcm": G.fit_glm(X, y, "lcm", prior, sigma, w_map=w_map, lcm_steps=700, lcm_lr=5e-2,
                         lcm_init_a=torch.arange(d) * -math.log(rho)),
        "full": G.fit_glm(X, y, "full", prior, sigma, w_map=w_map),
    }

    out = {"seed": seed}
    for m, la in las.items():
        for tag, Xt, yt in (("in", Xin, yin), ("shift", Xsh, ysh)):
            mu, var = G.glm_marginal(la, Xt, sigma)
            out[f"mll_{tag}_{m}"] = float((-0.5 * (torch.log(2 * math.pi * var)
                                                   + (yt - mu) ** 2 / var)).mean())
            out[f"z2_{tag}_{m}"] = float(((yt - mu) ** 2 / var).mean())
            if tag == "in":
                out[f"predvar_{m}"] = float(var.mean())
            ll_t, t_best = tempered_ll(la, Xt, yt, sigma)
            out[f"mll_{tag}_temp_{m}"], out[f"temp_{tag}_{m}"] = ll_t, t_best
        for tag, Xa, ya in (("iid", Xin, yin), ("corr", Xco, yco)):
            z2, ll = zip(*(_aggregate(la, Xa[b * T:(b + 1) * T], ya[b * T:(b + 1) * T], sigma)
                           for b in range(nB)))
            out[f"z2_agg{tag}_{m}"] = float(np.mean(z2))
            out[f"mll_agg{tag}_{m}"] = float(np.mean(ll))
    return out


def main():
    df = pd.DataFrame([run(s) for s in range(SEEDS)])
    df.to_csv(f"{RESULTS}/exp17_offdiagonals.csv", index=False)
    m = df.mean(numeric_only=True)
    print(f"  AR(1) GLM, d={D}, n={N}, rho={RHO}, {SEEDS} seeds, prior fixed at {PRIOR}")
    print(f"  mean predictive variance, in-dist: " + "  ".join(
        f"{k}={m[f'predvar_{k}']:.4f}" for k in ("diag", "lcm", "full"))
        + f"   [theory sigma^2(1+d/n) = {SIGMA**2*(1+D/N):.4f}]")
    print(f"\n  {'probe':34}{'diag':>11}{'LCM':>11}{'full':>11}{'gap closed':>13}")
    for key, lbl in (("mll_in", "(a) marginal LL, in-dist"),
                     ("mll_shift", "(b) marginal LL, shifted inputs"),
                     ("mll_aggiid", "(c) LL of sum, i.i.d. queries"),
                     ("mll_aggcorr", "(d) LL of sum, correlated queries")):
        dg, lc, fl = (m[f"{key}_{k}"] for k in ("diag", "lcm", "full"))
        print(f"  {lbl:34}{dg:11.4f}{lc:11.4f}{fl:11.4f}{100*(lc-dg)/(fl-dg+1e-12):12.1f}%")
    print(f"\n  {'calibration E[z^2] (ideal 1)':34}{'diag':>11}{'LCM':>11}{'full':>11}")
    for key, lbl in (("z2_in", "in-dist"), ("z2_shift", "shifted inputs"),
                     ("z2_aggiid", "sum, i.i.d. queries"), ("z2_aggcorr", "sum, correlated queries")):
        print(f"  {lbl:34}" + "".join(f"{m[f'{key}_{k}']:11.4f}" for k in ("diag", "lcm", "full")))
    print("\n  control -- best LL under an ORACLE scalar scale on the epistemic variance:")
    print(f"  {'probe':34}{'diag':>11}{'LCM':>11}{'full':>11}   best scales (d/L/f)")
    for key, lbl in (("mll_in_temp", "(a) in-dist, rescaled"),
                     ("mll_shift_temp", "(b) shifted, rescaled")):
        tag = "in" if "in" in key else "shift"
        print(f"  {lbl:34}" + "".join(f"{m[f'{key}_{k}']:11.4f}" for k in ("diag", "lcm", "full"))
              + "   " + " / ".join(f"{m[f'temp_{tag}_{k}']:.1f}" for k in ("diag", "lcm", "full")))
    print(f"\nsaved {RESULTS}/exp17_offdiagonals.csv")


if __name__ == "__main__":
    main()
