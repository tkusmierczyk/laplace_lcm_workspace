"""Exp 13 — Is LCM the right model in its own complexity class?

Every earlier experiment compared LCM against something *weaker* (diagonal, `d`
parameters) or something *richer* (KFAC, Full). It was never compared against its
peers: the other structured PSD families that also cost O(d) parameters. That
leaves the obvious reviewer question unanswered -- why *this* 3d parameterisation
and not a banded Cholesky or a low-rank correction?

The competitors, all fit to the same target with the same objective and budget:

  diag          D                       d        the floor
  chol_band1    L L^T, L bidiagonal     2d-1     -> TRIDIAGONAL precision
  chol_band2    L L^T, L 2 subdiags     3d-3     -> pentadiagonal precision
  diag_rank1    D + v v^T               2d       classic low-rank correction
  diag_rank2    D + V V^T, V in R^{dx2} 3d       low-rank at LCM's exact budget
  lcm           D + W K(a) W            3d       ours

`chol_band1` is the direct "AA^T that gives a tridiagonal matrix" comparison.

The distinction that makes this non-obvious: a banded Cholesky gives a *sparse*
(banded) precision, whereas LCM gives a *dense* one -- ``K(a)`` is the exponential
kernel, which is the INVERSE of a tridiagonal matrix (semiseparable), not a
tridiagonal matrix. So the two families are different despite both being O(d) and
both being "chain-like". Which one fits a given curvature is an empirical question.

Part (a): fit quality on four targets chosen so each family has a home.
Part (b): the metric that actually mattered -- joint predictive LL on the GLM,
          using the exp8c protocol with each model swapped in.

Writes results/exp13_fit.csv and results/exp13_joint.csv.
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

import exp10_ordering as E  # reuse simulate()
from laplace.lcm import LCMCovariance

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


# --------------------------------------------------------------------------- #
# structured PSD families  (all model the PRECISION, i.e. Formulation A)
# --------------------------------------------------------------------------- #
class Diag(nn.Module):
    name, budget = "diag", lambda d: d

    def __init__(self, d, target=None):
        super().__init__()
        init = target.diagonal().clamp(min=1e-8) if target is not None else torch.ones(d)
        self.u = nn.Parameter(torch.log(torch.expm1(init.clamp(min=1e-8))))

    def matrix(self):
        return torch.diag(F.softplus(self.u) + 1e-8)


class BandChol(nn.Module):
    """Precision = L L^T with L lower-banded (``band`` subdiagonals).

    band=1 -> tridiagonal precision; band=2 -> pentadiagonal.
    """

    def __init__(self, d, band, target=None):
        super().__init__()
        self.d, self.band = d, band
        diag0 = torch.sqrt(target.diagonal().clamp(min=1e-8)) if target is not None else torch.ones(d)
        self.u = nn.Parameter(torch.log(torch.expm1(diag0.clamp(min=1e-8))))
        self.off = nn.Parameter(torch.zeros(band, d))

    def matrix(self):
        d = self.d
        L = torch.zeros(d, d, dtype=self.u.dtype, device=self.u.device)
        idx = torch.arange(d, device=self.u.device)
        L = L.index_put((idx, idx), F.softplus(self.u) + 1e-8)
        for k in range(1, self.band + 1):
            r = torch.arange(k, d, device=self.u.device)
            L = L.index_put((r, r - k), self.off[k - 1, : d - k])
        return L @ L.T


class DiagLowRank(nn.Module):
    """Precision = D + V V^T with V in R^{d x k}."""

    def __init__(self, d, k, target=None):
        super().__init__()
        self.k = k
        base = target.diagonal().clamp(min=1e-8) if target is not None else torch.ones(d)
        self.u = nn.Parameter(torch.log(torch.expm1((0.9 * base).clamp(min=1e-8))))
        scale = (0.1 * base.mean() / k).sqrt() if target is not None else torch.tensor(0.1)
        self.V = nn.Parameter(scale * torch.randn(d, k))

    def matrix(self):
        return torch.diag(F.softplus(self.u) + 1e-8) + self.V @ self.V.T


def make_models(d, extended=False):
    """Every competitor, with its exact parameter count.

    Constructors take the (already normalised) target, so each family can scale
    its own initialisation to it. ``extended`` adds higher-rank members that are
    OUTSIDE LCM's 3d budget -- they answer "how much low-rank does it take to
    match LCM?", which prices LCM's structural advantage in parameters.
    """
    models = [
        ("diag", lambda T: Diag(d, T), d),
        ("chol_band1", lambda T: BandChol(d, 1, T), 2 * d - 1),
        ("chol_band2", lambda T: BandChol(d, 2, T), 3 * d - 3),
        ("diag_rank1", lambda T: DiagLowRank(d, 1, T), 2 * d),
        ("diag_rank2", lambda T: DiagLowRank(d, 2, T), 3 * d),
        ("lcm", None, 3 * d),
    ]
    if extended:
        models += [
            ("diag_rank4", lambda T: DiagLowRank(d, 4, T), 5 * d),
            ("diag_rank8", lambda T: DiagLowRank(d, 8, T), 9 * d),
            ("diag_rank16", lambda T: DiagLowRank(d, 16, T), 17 * d),
            ("chol_band4", lambda T: BandChol(d, 4, T), 5 * d - 10),
        ]
    return models


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def fit_model(model, target, n_steps=800, lr=5e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tn = target.norm()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = ((model.matrix() - target) ** 2).sum() / tn**2
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((model.matrix() - target).norm() / tn)


def fit_lcm(d, target, a_init=None, n_steps=800, lr=5e-2):
    lcm = LCMCovariance(d, dtype=torch.float64)
    if a_init is not None:
        with torch.no_grad():
            lcm.a.copy_(a_init)
    err = lcm.fit_dense(target, n_steps=n_steps, lr=lr)
    return lcm, err


def fit_any(name, ctor, d, target, a_init=None, n_steps=800, lr=5e-2):
    """Fit one family to ``target``, SCALE-NORMALISED.

    Every family here is positively homogeneous (``D -> sD``, ``VV^T -> sVV^T``,
    ``LL^T -> sLL^T``, ``D + WKW -> sD + (sqrt(s) W) K (sqrt(s) W)``), so fitting to
    ``target/s`` and multiplying the result by ``s`` is exact. Without it a target
    at GGN scale (~1e5) makes Adam at lr=5e-2 diverge for the low-rank families --
    the same class of bug that once produced a false "matrix-free fit is weak"
    conclusion. Normalising makes the comparison scale-free and therefore fair.

    Returns (dense matrix at the ORIGINAL scale, relative Frobenius residual).
    """
    s = target.diagonal().mean().clamp(min=1e-30)
    Tn = target / s
    if name == "lcm":
        lcm, err = fit_lcm(d, Tn, a_init=a_init, n_steps=n_steps, lr=lr)
        M = lcm.dense().detach()
    else:
        m = ctor(Tn)
        err = fit_model(m, Tn, n_steps=n_steps, lr=lr)
        M = m.matrix().detach()
    M = 0.5 * (M + M.T) * s          # symmetrise and restore scale
    return M, err                    # residual is scale-invariant


def kl_precisions(M, T):
    """KL( N(0, M^-1) || N(0, T^-1) ) -- approx precision M vs true precision T."""
    d = M.shape[0]
    Minv = torch.linalg.inv(M)
    tr = torch.trace(T @ Minv)
    _, ldM = torch.linalg.slogdet(M)
    _, ldT = torch.linalg.slogdet(T)
    return float(0.5 * (tr - d + ldM - ldT))


# --------------------------------------------------------------------------- #
# targets
# --------------------------------------------------------------------------- #
def make_targets(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = {}

    # (1) GGN of an AR(1)-correlated GLM -- LCM's niche. DENSE semiseparable.
    rho, n = 0.9, 4 * d
    Z = torch.randn(n, d, generator=g)
    X = torch.zeros(n, d)
    X[:, 0] = Z[:, 0]
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * Z[:, j]
    out["glm_ar1"] = X.T @ X / n

    # (2) the same target under a random permutation -- order unknown
    perm = torch.randperm(d, generator=g)
    out["glm_ar1_perm"] = out["glm_ar1"][perm][:, perm]

    # (3) a genuinely SPARSE tridiagonal precision -- band-Cholesky's home
    Lb = torch.zeros(d, d)
    i = torch.arange(d)
    Lb[i, i] = 1.0 + 0.5 * torch.rand(d, generator=g)
    r = torch.arange(1, d)
    Lb[r, r - 1] = -0.6 + 0.2 * torch.randn(d - 1, generator=g)
    out["tridiag"] = Lb @ Lb.T

    # (4) a real trained last-layer GGN -- Kronecker, nobody's home.
    # Pick C, H with C*H == d so the target matches the other three in size.
    C = next(c for c in (5, 4, 6, 8, 10, 2) if d % c == 0)
    H, n_in, N = d // C, 20, 600
    torch.manual_seed(seed)
    Xn = torch.randn(N, n_in)
    yn = (Xn @ torch.randn(n_in, C) + 0.5 * torch.randn(N, C)).argmax(1)
    net = nn.Sequential(nn.Linear(n_in, H), nn.Tanh(), nn.Linear(H, C))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(net(Xn), yn).backward()
        opt.step()
    with torch.no_grad():
        Phi = torch.tanh(net[0](Xn))                       # (N, H)
        p = torch.softmax(net(Xn), 1)                      # (N, C)
        Lam = torch.diag_embed(p) - torch.einsum("ic,id->icd", p, p)   # (N, C, C)
        # G[c1*H+h1, c2*H+h2] = mean_i Lam[i,c1,c2] Phi[i,h1] Phi[i,h2]
        G = torch.einsum("icd,ih,ik->chdk", Lam, Phi, Phi).reshape(C * H, C * H) / N
    out["nn_ggn"] = G
    return out


# --------------------------------------------------------------------------- #
# part (a) — fit quality across families and targets
# --------------------------------------------------------------------------- #
def part_a(d=200, seed=0, extended=True):
    print(f"\n=== (a) fit quality, same parameter budget, d={d} ===")
    targets = make_targets(d, seed)
    rows = []
    for tname, T in targets.items():
        ridge = 1e-3 * T.diagonal().mean()
        Tp = T + ridge * torch.eye(d)
        print(f"\n  target: {tname}")
        for name, ctor, budget in make_models(d, extended=extended):
            torch.manual_seed(seed)
            a0 = torch.arange(d) * -np.log(0.9) if (name == "lcm" and tname == "glm_ar1") else None
            M, err = fit_any(name, ctor, d, T, a_init=a0)
            try:
                kl = kl_precisions(M + ridge * torch.eye(d), Tp)
            except Exception:
                kl = float("nan")
            rows.append(dict(target=tname, model=name, n_params=budget,
                             params_per_d=budget / d, rel_frob=err, kl=kl))
            flag = "   <-- over LCM's budget" if budget > 3 * d else ""
            print(f"    {name:12} {budget/d:5.1f}d params   rel-Frob {err:7.4f}   KL {kl:11.3f}{flag}")
        inbudget = [r for r in rows if r["target"] == tname and r["n_params"] <= 3 * d]
        best = min(inbudget, key=lambda r: r["rel_frob"])
        print(f"    -> best within 3d budget: {best['model']} ({best['rel_frob']:.4f})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# part (b) — joint predictive LL (the metric that mattered), exp8c protocol
# --------------------------------------------------------------------------- #
def joint_ll(mu, Cov, y):
    T = len(y)
    L = torch.linalg.cholesky(Cov)
    r = torch.linalg.solve_triangular(L, (y - mu).unsqueeze(1), upper=False).squeeze(1)
    return float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                         + T * np.log(2 * np.pi)) / T)


def part_b(d=1000, seeds=5, n=3000, rho=0.9, ntest=400, noise=0.5, prior=1.0):
    print(f"\n=== (b) joint predictive LL, d={d}, {seeds} seeds (exp8c protocol) ===")
    rows = []
    for seed in range(seeds):
        X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
        GGN = X.T @ X / noise**2
        Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
        mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))
        Inoise = noise**2 * torch.eye(ntest)

        covs = {"full": Cov_full,
                "diag": torch.diag(1.0 / (GGN.diagonal() + prior))}
        for name, ctor, _ in make_models(d):
            if name == "diag":
                continue
            torch.manual_seed(seed)
            a0 = torch.arange(d) * -np.log(rho) if name == "lcm" else None
            M, _ = fit_any(name, ctor, d, GGN, a_init=a0, n_steps=700)
            A = M + prior * torch.eye(d)
            # invert via Cholesky so the result is symmetric and PD by construction
            try:
                covs[name] = torch.cholesky_inverse(torch.linalg.cholesky(A))
            except Exception:
                covs[name] = None  # fit produced a non-PD precision; recorded as NaN

        out = {"d": d, "seed": seed}
        for nm, Cv in covs.items():
            if Cv is None:
                out[f"marg_{nm}"] = out[f"joint_{nm}"] = float("nan")
                continue
            var = torch.einsum("ti,ij,tj->t", Xt, Cv, Xt) + noise**2
            out[f"marg_{nm}"] = float((-0.5 * (torch.log(2 * np.pi * var)
                                               + (yt - mu_t) ** 2 / var)).mean())
            Cp = Xt @ Cv @ Xt.T + Inoise
            out[f"joint_{nm}"] = joint_ll(mu_t, 0.5 * (Cp + Cp.T), yt)
        denom = out["joint_full"] - out["joint_diag"] + 1e-12
        for nm in covs:
            if nm in ("diag", "full"):
                continue
            out[f"gap_{nm}"] = (out[f"joint_{nm}"] - out["joint_diag"]) / denom
        rows.append(out)
        print(f"  seed {seed}: " + "  ".join(
            f"{k.replace('gap_','')}={100*v:.0f}%" for k, v in out.items() if k.startswith("gap_")))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = part_a()
    a.to_csv(os.path.join(RESULTS, "exp13_fit.csv"), index=False)

    b = part_b()
    b.to_csv(os.path.join(RESULTS, "exp13_joint.csv"), index=False)
    print("\n=== joint gap-to-full closed, mean +/- 95% CI ===")
    for c in [c for c in b.columns if c.startswith("gap_")]:
        m, h = b[c].mean(), 1.96 * b[c].std() / np.sqrt(len(b))
        print(f"  {c.replace('gap_',''):12} {100*m:6.1f}% +/- {100*h:.1f}")
    print(f"\nsaved {RESULTS}/exp13_fit.csv, exp13_joint.csv")
