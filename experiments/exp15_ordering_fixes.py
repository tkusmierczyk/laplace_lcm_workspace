"""Exp 15 — D2: a hybrid LCM+low-rank.  D4: nugget annealing for the ordering.

Both target the project's central negative (Exp 10): with an UNKNOWN feature
order, gradient descent recovers |Spearman| ~ 0.03 and LCM lands at -84% of the
diagonal->full gap -- i.e. *worse than diagonal*, because it overfits into a
confidently wrong chain. A fixed wrong order merely degrades to diagonal (0%).

  (D2) HYBRID  A = D + W K(a) W + V V^T.
       Rationale: the low-rank term is order-INDEPENDENT, so it can absorb the
       structure a mis-ordered chain currently tries and fails to represent.
       Pre-registered prediction: this will NOT fix order recovery -- nothing in
       the parameterisation makes the discrete search easier -- but it should
       convert "confidently wrong" (-84%) into "falls back to low-rank" (~0%),
       which is the difference between unusable and safe-by-default.

  (D4) ANNEALING the nugget, as proposed: start with a large diagonal and anneal
       down. Rationale (graduated non-convexity): with a large nugget the loss is
       dominated by matching the diagonal, so the surface over `a` is flatter and
       better conditioned; shrinking it gradually follows the solution path into
       the rugged regime instead of starting there.
       Implementation: at stage t the nugget fraction phi_t is FIXED (d and w set
       so the model diagonal matches the target exactly), and only `a` is trained;
       phi anneals high -> low with `a` warm-started across stages; a final stage
       unfreezes everything.
       Pre-registered prediction: recovery should improve on 0.028, but Exp 10b's
       threshold (|Spearman| >= 0.999, with 0.998 already failing) means it must be
       near-exact to pay off. Most likely outcome: better but still under threshold.

  CONTROLS (without these, any improvement is uninterpretable):
    * `plain`      -- one fit, the Exp 10 baseline.
    * `restarts`   -- the SAME total compute as annealing, spent on random restarts
                      keeping the best. If annealing only matches this, it is buying
                      compute, not curvature.
    * `seriation`  -- spectral (Fiedler) init, the existing best-known initialiser.
    * `oracle`     -- initialised at the TRUE order. Upper bound.

Writes results/exp15_hybrid.csv and results/exp15_anneal.csv.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import exp10_ordering as E
from exp13_sameclass import DiagLowRank, fit_any, fit_model, joint_ll
from laplace.lcm import LCMCovariance

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
# D2 — hybrid LCM + low-rank
# --------------------------------------------------------------------------- #
class LCMLowRank(nn.Module):
    """A = D + W K(a) W + V V^T  -- LCM plus an order-independent low-rank term."""

    def __init__(self, d, k, target=None):
        super().__init__()
        self.lcm = LCMCovariance(d, dtype=torch.float64)
        base = target.diagonal().clamp(min=1e-8) if target is not None else torch.ones(d)
        with torch.no_grad():
            self.lcm.w.copy_(torch.sqrt(0.6 * base))
            self.lcm.u_raw.copy_(torch.log(torch.expm1((0.3 * base).clamp(min=1e-8))))
        scale = (0.1 * base.mean() / k).sqrt()
        self.V = nn.Parameter(scale * torch.randn(d, k))

    def matrix(self):
        return self.lcm.dense() + self.V @ self.V.T

    @property
    def a(self):
        return self.lcm.a


def spearman(x, y):
    rx = torch.argsort(torch.argsort(x)).double()
    ry = torch.argsort(torch.argsort(y)).double()
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx @ ry) / (rx.norm() * ry.norm() + 1e-12))


def posterior_scores(M, GGN, X, y, Xt, yt, d, prior=1.0, noise=0.5, ref=None):
    Cov_full, mu_t, Inoise = ref
    A = M + prior * torch.eye(d)
    try:
        Cv = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (A + A.T)))
    except Exception:
        return float("nan"), float("nan")
    Cp = Xt @ Cv @ Xt.T + Inoise
    j = joint_ll(mu_t, 0.5 * (Cp + Cp.T), yt)
    var = torch.einsum("ti,ij,tj->t", Xt, Cv, Xt) + noise**2
    m = float((-0.5 * (torch.log(2 * np.pi * var) + (yt - mu_t) ** 2 / var)).mean())
    return m, j


def setup(d, seed, n=2500, rho=0.9, ntest=300, noise=0.5, prior=1.0, permute=True):
    X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
    perm = torch.randperm(d, generator=torch.Generator().manual_seed(1000 + seed)) if permute \
        else torch.arange(d)
    X, Xt = X[:, perm], Xt[:, perm]
    # X_new[:, j] == X_old[:, perm[j]], and X_old column i sits at true position i,
    # so the true coordinate of new column j is perm[j] -- NOT argsort(perm).
    # (Using the inverse here is the project's documented Bug 3; the assertion in
    # _check_true_coord below pins it down empirically.)
    true_coord = perm.double()
    GGN = X.T @ X / noise**2
    Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
    mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))
    ref = (Cov_full, mu_t, noise**2 * torch.eye(ntest))
    return X, y, Xt, yt, GGN, true_coord, ref


def part_d2(d=400, seeds=4, steps=600, k=2):
    print(f"\n=== D2: hybrid LCM + low-rank, UNKNOWN order, d={d}, {seeds} seeds ===")
    rows = []
    for seed in range(seeds):
        X, y, Xt, yt, GGN, true_c, ref = setup(d, seed)
        Cov_full, mu_t, Inoise = ref
        gd = torch.diag(1.0 / (GGN.diagonal() + 1.0))
        _, j_diag = posterior_scores(torch.diag(GGN.diagonal()), GGN, X, y, Xt, yt, d, ref=ref)
        Cpf = Xt @ Cov_full @ Xt.T + Inoise
        j_full = joint_ll(mu_t, 0.5 * (Cpf + Cpf.T), yt)
        den = j_full - j_diag + 1e-12
        s = GGN.diagonal().mean()
        for name in ["lcm", f"hybrid_k{k}", f"rank{k+2}"]:   # rank(k+2) == (3+k)d, matched
            torch.manual_seed(seed)
            if name == "lcm":
                m = LCMCovariance(d, dtype=torch.float64)
                err = m.fit_dense(GGN / s, n_steps=steps, lr=5e-2)
                M, a_hat = m.dense().detach() * s, m.a.detach()
            elif name.startswith("hybrid"):
                m = LCMLowRank(d, k, GGN / s)
                err = fit_model(m, GGN / s, n_steps=steps)
                M, a_hat = m.matrix().detach() * s, m.a.detach()
            else:
                M, err = fit_any(name, lambda T: DiagLowRank(d, k + 2, T), d, GGN, n_steps=steps)
                a_hat = None
            _, j = posterior_scores(M, GGN, X, y, Xt, yt, d, ref=ref)
            rows.append(dict(seed=seed, model=name, rel_frob=err,
                             gap=(j - j_diag) / den,
                             recovery=abs(spearman(a_hat, true_c)) if a_hat is not None else np.nan))
    df = pd.DataFrame(rows)
    print("  model        rel-Frob    joint gap closed        order recovery")
    for m in df.model.unique():
        g = df[df.model == m]
        rec = g.recovery.mean()
        print(f"  {m:11}  {g.rel_frob.mean():8.4f}   {100*g.gap.mean():7.1f}% "
              f"+/- {100*1.96*g.gap.std()/np.sqrt(len(g)):5.1f}    "
              f"{'n/a' if np.isnan(rec) else f'{rec:.3f}'}")
    return df


# --------------------------------------------------------------------------- #
# D4 — nugget annealing
# --------------------------------------------------------------------------- #
def fit_annealed(d, T, phis=(0.9, 0.7, 0.5, 0.3, 0.15, 0.05), steps_per=120, final=300, lr=5e-2,
                 a_init=None):
    """Homotopy in the nugget fraction.

    At each stage the diagonal split is FIXED (d = phi*diag(T), w = sqrt((1-phi)*diag(T)),
    so the model diagonal matches the target exactly) and only `a` is optimised,
    warm-started from the previous stage. A final stage unfreezes everything.
    """
    diag = T.diagonal().clamp(min=1e-12)
    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(a_init if a_init is not None else torch.linspace(0, d * 0.1, d))
    tn = T.norm()
    for phi in phis:
        with torch.no_grad():
            lcm.w.copy_(torch.sqrt((1.0 - phi) * diag))
            lcm.u_raw.copy_(torch.log(torch.expm1((phi * diag).clamp(min=1e-8))))
        lcm.w.requires_grad_(False)
        lcm.u_raw.requires_grad_(False)
        opt = torch.optim.Adam([lcm.a], lr=lr)
        for _ in range(steps_per):
            opt.zero_grad()
            (((lcm.dense() - T) ** 2).sum() / tn**2).backward()
            opt.step()
    lcm.w.requires_grad_(True)
    lcm.u_raw.requires_grad_(True)
    opt = torch.optim.Adam(lcm.parameters(), lr=lr)
    for _ in range(final):
        opt.zero_grad()
        (((lcm.dense() - T) ** 2).sum() / tn**2).backward()
        opt.step()
    with torch.no_grad():
        return lcm, float((lcm.dense() - T).norm() / tn)


def fit_plain(d, T, steps, lr=5e-2, a_init=None):
    lcm = LCMCovariance(d, dtype=torch.float64)
    if a_init is not None:
        with torch.no_grad():
            lcm.a.copy_(a_init)
    err = lcm.fit_dense(T, n_steps=steps, lr=lr)
    return lcm, err


def seriation_coord(G):
    """Spectral (Fiedler) seriation of |correlation|, the existing best initialiser."""
    dg = torch.sqrt(G.diagonal().clamp(min=1e-12))
    Cr = (G / dg[:, None] / dg[None, :]).abs()
    Lap = torch.diag(Cr.sum(1)) - Cr
    ev = torch.linalg.eigh(Lap).eigenvectors[:, 1]
    return torch.argsort(torch.argsort(ev)).double() * 0.1


def part_d4(d=400, seeds=4, steps=1020):
    """`steps` is the TOTAL budget; annealing and restarts get the same."""
    print(f"\n=== D4: nugget annealing for order recovery, d={d}, {seeds} seeds ===")
    print(f"    (equal compute: plain={steps} steps, anneal=6x120+300, restarts=3x340)")
    rows = []
    for seed in range(seeds):
        X, y, Xt, yt, GGN, true_c, ref = setup(d, seed)
        Cov_full, mu_t, Inoise = ref
        _, j_diag = posterior_scores(torch.diag(GGN.diagonal()), GGN, X, y, Xt, yt, d, ref=ref)
        Cpf = Xt @ Cov_full @ Xt.T + Inoise
        j_full = joint_ll(mu_t, 0.5 * (Cpf + Cpf.T), yt)
        den = j_full - j_diag + 1e-12
        s = GGN.diagonal().mean()
        Tn = GGN / s

        variants = {}
        torch.manual_seed(seed)
        variants["plain"] = fit_plain(d, Tn, steps)
        torch.manual_seed(seed)
        variants["anneal"] = fit_annealed(d, Tn)
        torch.manual_seed(seed)                       # equal-compute control
        best = None
        for r in range(3):
            m_, e_ = fit_plain(d, Tn, steps // 3,
                               a_init=torch.linspace(0, d * 0.1, d)[torch.randperm(d)])
            if best is None or e_ < best[1]:
                best = (m_, e_)
        variants["restarts"] = best
        torch.manual_seed(seed)
        variants["seriation"] = fit_plain(d, Tn, steps, a_init=seriation_coord(GGN))
        torch.manual_seed(seed)
        variants["anneal+seri"] = fit_annealed(d, Tn, a_init=seriation_coord(GGN))
        torch.manual_seed(seed)
        variants["oracle"] = fit_plain(d, Tn, steps, a_init=true_c * 0.1)

        for name, (m, err) in variants.items():
            _, j = posterior_scores(m.dense().detach() * s, GGN, X, y, Xt, yt, d, ref=ref)
            rows.append(dict(seed=seed, variant=name, rel_frob=err,
                             recovery=abs(spearman(m.a.detach(), true_c)),
                             gap=(j - j_diag) / den))
    df = pd.DataFrame(rows)
    print("  variant        rel-Frob   order recovery |rho_s|    joint gap closed")
    for v in ["plain", "restarts", "seriation", "anneal", "anneal+seri", "oracle"]:
        g = df[df.variant == v]
        print(f"  {v:13}  {g.rel_frob.mean():8.4f}   {g.recovery.mean():.3f} "
              f"(max {g.recovery.max():.3f})       {100*g.gap.mean():7.1f}% "
              f"+/- {100*1.96*g.gap.std()/np.sqrt(len(g)):5.1f}")
    return df


if __name__ == "__main__":
    h = part_d2(); h.to_csv(f"{RESULTS}/exp15_hybrid.csv", index=False)
    a = part_d4(); a.to_csv(f"{RESULTS}/exp15_anneal.csv", index=False)
    print(f"\nsaved {RESULTS}/exp15_hybrid.csv, exp15_anneal.csv")
