"""Exp 14 — D1: are we actually beating low-rank?  D3: does very large d change it?

Exp 13 found LCM beats low-rank *at matched parameter budget* in its niche, but
that low-rank wins on NN curvature once given more parameters. Two things were
left open, and both decide whether the method has a future.

  (A) RANK SWEEP vs CORRELATION. Where does low-rank overtake LCM as a function of
      how correlated the features are? If low-rank catches up at realistic rho,
      LCM has no regime.

  (B) THE DECISIVE TEST -- does matched FIT imply matched PREDICTION?
      Exp 13 showed rank-16 (17d) matches LCM's Frobenius error (0.147 vs 0.140).
      If it also closes the joint-LL gap, then LCM's only advantage is parameter
      efficiency. If it does NOT, LCM captures the *right* structure, not merely
      more of it -- a much stronger claim.

      There is a concrete reason to expect the latter. We fit the PRECISION. The
      posterior covariance is (A + prior*I)^-1. For the AR(1) GLM the GGN is ~ n*K,
      and K^-1 is TRIDIAGONAL -- so the true posterior covariance is near-banded.
      A low-rank precision inverts (Woodbury) to diagonal-plus-low-rank, which
      cannot be banded. So low-rank should fail on the predictive even when its
      Frobenius error on the precision matches.

  (C) NN CURVATURE, predictively. Exp 13 only compared *fit* there. A last-layer
      Laplace is exactly a GLM in the last-layer features, so the same joint-LL
      machinery applies to a trained network.

  (D) D3 -- FEASIBILITY AT SCALE. Which families can even be built at d = 10^4+,
      measured rather than asserted, and is there an uncontested regime for LCM?

Writes results/exp14_{ranksweep,decisive,nn,scale}.csv.
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import exp10_ordering as E
from exp13_sameclass import BandChol, DiagLowRank, fit_any, joint_ll, make_targets
from laplace.lcm import LCMCovariance

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)


def ar1_ggn(d, rho, n, seed):
    """GGN = X^T X / n for AR(1)-correlated ordered features."""
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, d, generator=g)
    X = torch.zeros(n, d)
    X[:, 0] = Z[:, 0]
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * Z[:, j]
    return X.T @ X / n


def competitors(d, ranks=(1, 2, 4, 8, 16, 32), bands=(1, 2)):
    out = [("lcm", None, 3 * d)]
    out += [(f"rank{k}", (lambda k: lambda T: DiagLowRank(d, k, T))(k), (k + 1) * d) for k in ranks]
    out += [(f"band{b}", (lambda b: lambda T: BandChol(d, b, T))(b),
             d + b * d - b * (b + 1) // 2) for b in bands]
    return out


# --------------------------------------------------------------------------- #
# (A) rank sweep vs correlation strength
# --------------------------------------------------------------------------- #
def part_a(d=300, n=1200, rhos=(0.5, 0.9, 0.99), seeds=3, steps=600):
    print(f"\n=== (A) rank sweep vs correlation, d={d}, {seeds} seeds ===")
    rows = []
    for rho in rhos:
        for seed in range(seeds):
            T = ar1_ggn(d, rho, n, seed)
            a0 = torch.arange(d) * -np.log(rho)
            for name, ctor, npar in competitors(d):
                torch.manual_seed(seed)
                _, err = fit_any(name, ctor, d, T,
                                 a_init=a0 if name == "lcm" else None, n_steps=steps)
                rows.append(dict(rho=rho, seed=seed, model=name,
                                 n_params=npar, params_per_d=npar / d, rel_frob=err))
        df = pd.DataFrame([r for r in rows if r["rho"] == rho]).groupby("model").rel_frob.mean()
        lcm = df["lcm"]
        beats = [m for m in df.index if m.startswith("rank") and df[m] < lcm]
        print(f"  rho={rho}: LCM {lcm:.4f} | " +
              " ".join(f"{m}={df[m]:.3f}" for m in ["rank2", "rank8", "rank16", "rank32", "band1"]))
        print(f"    low-rank variants beating LCM: {beats if beats else 'none'}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (B) the decisive test: does matched fit imply matched prediction?
# --------------------------------------------------------------------------- #
def joint_eval(GGN, X, y, Xt, yt, d, models, prior=1.0, noise=0.5, steps=600, seed=0, a0=None):
    """Fit each model to GGN, then score marginal/joint predictive LL."""
    Cov_full = torch.linalg.inv(GGN + prior * torch.eye(d))
    mu_t = Xt @ (Cov_full @ (X.T @ y / noise**2))
    Inoise = noise**2 * torch.eye(len(yt))
    covs = {"full": Cov_full, "diag": torch.diag(1.0 / (GGN.diagonal() + prior))}
    errs = {"full": 0.0, "diag": float((torch.diag(GGN.diagonal()) - GGN).norm() / GGN.norm())}
    for name, ctor, _ in models:
        torch.manual_seed(seed)
        M, err = fit_any(name, ctor, d, GGN,
                         a_init=a0 if name == "lcm" else None, n_steps=steps)
        errs[name] = err
        try:
            covs[name] = torch.cholesky_inverse(torch.linalg.cholesky(M + prior * torch.eye(d)))
        except Exception:
            covs[name] = None
    out = {}
    for nm, Cv in covs.items():
        if Cv is None:
            out[f"joint_{nm}"] = out[f"marg_{nm}"] = float("nan")
            continue
        var = torch.einsum("ti,ij,tj->t", Xt, Cv, Xt) + noise**2
        out[f"marg_{nm}"] = float((-0.5 * (torch.log(2 * np.pi * var) + (yt - mu_t) ** 2 / var)).mean())
        Cp = Xt @ Cv @ Xt.T + Inoise
        out[f"joint_{nm}"] = joint_ll(mu_t, 0.5 * (Cp + Cp.T), yt)
    den = out["joint_full"] - out["joint_diag"] + 1e-12
    for nm in covs:
        if nm not in ("diag", "full"):
            out[f"gap_{nm}"] = (out[f"joint_{nm}"] - out["joint_diag"]) / den
        out[f"err_{nm}"] = errs[nm]
    return out


def part_b(d=500, n=2500, rho=0.9, seeds=5, ntest=300, noise=0.5, steps=600):
    print(f"\n=== (B) decisive: does matched FIT imply matched PREDICTION? d={d} ===")
    models = competitors(d, ranks=(2, 8, 16, 32), bands=(1,))
    rows = []
    for seed in range(seeds):
        X, y, Xt, yt = E.simulate(n, d, rho, seed, ntest, noise)
        GGN = X.T @ X / noise**2
        r = joint_eval(GGN, X, y, Xt, yt, d, models, noise=noise, steps=steps, seed=seed,
                       a0=torch.arange(d) * -np.log(rho))
        r.update(d=d, seed=seed)
        rows.append(r)
        print(f"  seed {seed}: " + "  ".join(
            f"{k[4:]}={100*v:.0f}%" for k, v in r.items() if k.startswith("gap_")))
    df = pd.DataFrame(rows)
    print("\n  model      rel-Frob(precision)   joint gap closed")
    for m in ["lcm", "rank2", "rank8", "rank16", "rank32", "band1"]:
        e, g = df[f"err_{m}"].mean(), df[f"gap_{m}"]
        print(f"  {m:9}  {e:18.4f}   {100*g.mean():7.1f}% +/- {100*1.96*g.std()/np.sqrt(len(g)):.1f}")
    return df


# --------------------------------------------------------------------------- #
# (C) NN last-layer curvature, predictively
# --------------------------------------------------------------------------- #
def part_c(seeds=3, H=60, C=4, n_in=20, N=2500, ntest=300, steps=600, prior=1.0):
    """A last-layer Laplace IS a GLM in the last-layer features -- so the same
    joint-LL protocol applies to a trained network. Regression head, so the GGN is
    exactly X^T X / sigma^2 with X = the last-layer features."""
    d = H
    print(f"\n=== (C) NN last-layer curvature, predictive (d={d} features) ===")
    rows = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        Xr = torch.randn(N + ntest, n_in)
        yr = (torch.sin(Xr @ torch.randn(n_in)) * 2 + 0.3 * torch.randn(N + ntest))
        net = nn.Sequential(nn.Linear(n_in, H), nn.Tanh(), nn.Linear(H, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(500):
            opt.zero_grad()
            F.mse_loss(net(Xr[:N]).squeeze(), yr[:N]).backward()
            opt.step()
        with torch.no_grad():
            Phi = torch.tanh(net[0](Xr))          # last-layer features
        noise = float((net(Xr[:N]).squeeze() - yr[:N]).std())
        X, y, Xt, yt = Phi[:N], yr[:N], Phi[N:], yr[N:]
        GGN = X.T @ X / noise**2
        models = competitors(d, ranks=(1, 2, 4, 8), bands=(1, 2))
        r = joint_eval(GGN, X, y, Xt, yt, d, models, prior=prior, noise=noise,
                       steps=steps, seed=seed, a0=None)
        r.update(seed=seed, noise=noise)
        rows.append(r)
    df = pd.DataFrame(rows)
    print("  model      rel-Frob(precision)   joint gap closed")
    for m in ["lcm", "rank1", "rank2", "rank4", "rank8", "band1", "band2"]:
        e, g = df[f"err_{m}"].mean(), df[f"gap_{m}"]
        print(f"  {m:9}  {e:18.4f}   {100*g.mean():7.1f}% +/- {100*1.96*g.std()/np.sqrt(len(g)):.1f}")
    return df


# --------------------------------------------------------------------------- #
# (D) D3 — feasibility at scale
# --------------------------------------------------------------------------- #
def part_d(dims=(1000, 5000, 20000, 100000), k=8):
    """Which families can be BUILT at large d? Measured, not asserted.

    Only the matrix-free paths are exercised: LCM's O(d) kernel matvec and a
    low-rank matvec. Dense (Full) and KFAC-for-a-wide-layer are reported as the
    allocation they would require.
    """
    print(f"\n=== (D) feasibility at scale (float64; matrix-free paths only) ===")
    rows = []
    GB = 8 / 1024**3
    for d in dims:
        rec = dict(d=d,
                   mem_lcm_gb=3 * d * GB, mem_rank_gb=(k + 1) * d * GB,
                   mem_full_gb=d * d * GB, mem_kfac_sq_gb=2 * d * GB,
                   mem_kfac_wide_gb=(d**2) * GB)
        # LCM matvec (O(d) state-space path)
        lcm = LCMCovariance(d, dtype=torch.float64)
        with torch.no_grad():
            lcm.a.copy_(torch.linspace(0, d * 0.01, d))
        v = torch.randn(d)
        t0 = time.perf_counter(); lcm.matvec(v); rec["t_lcm_matvec_ms"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter(); lcm.logdet(); rec["t_lcm_logdet_ms"] = (time.perf_counter() - t0) * 1e3
        # low-rank matvec
        Dg, V = torch.rand(d) + 0.5, torch.randn(d, k) * 0.1
        t0 = time.perf_counter(); Dg * v + V @ (V.T @ v); rec["t_rank_matvec_ms"] = (time.perf_counter() - t0) * 1e3
        rows.append(rec)
        print(f"  d={d:>7,}  LCM {3*d*GB:8.4f} GB (matvec {rec['t_lcm_matvec_ms']:6.2f} ms, "
              f"logdet {rec['t_lcm_logdet_ms']:6.2f} ms) | rank{k} {(k+1)*d*GB:8.4f} GB "
              f"({rec['t_rank_matvec_ms']:5.2f} ms) | full {d*d*GB:10.2f} GB")
    print("  KFAC on a WIDE layer (in=d, out small) would need ~d^2 floats -- the `full` column.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a = part_a(); a.to_csv(f"{RESULTS}/exp14_ranksweep.csv", index=False)
    b = part_b(); b.to_csv(f"{RESULTS}/exp14_decisive.csv", index=False)
    c = part_c(); c.to_csv(f"{RESULTS}/exp14_nn.csv", index=False)
    dd = part_d(); dd.to_csv(f"{RESULTS}/exp14_scale.csv", index=False)
    print(f"\nsaved {RESULTS}/exp14_{{ranksweep,decisive,nn,scale}}.csv")
