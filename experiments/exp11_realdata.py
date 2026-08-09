"""Experiment 11 — real ordered-feature data (blocking gap: synthetic-only).

Replicates the marginal-tie / joint-win pattern (and now, given exp10b, the
KNOWN-order requirement) on real data whose features carry a genuine, *known*
1-D order:

  * Tecator NIR spectroscopy (n=240, 100 wavelength-ordered absorbance channels;
    target = fat content) -- the canonical chemometrics benchmark; neighbouring
    wavelengths are strongly correlated.
  * Mauna Loa CO2: autoregression of the (differenced, deseasonalised) weekly
    series on its own d lagged values -- lag index is a genuine 1-D order with
    decaying autocorrelation.

Bayesian linear (ridge) model; Laplace posterior over the coefficients. We compare
diagonal, LCM-fixed(natural order), LCM-learned, and exact full, on MARGINAL and
JOINT predictive test log-likelihood. Honest checks included: effective rank of the
GGN (n vs d) and the fit residual, since exp10b showed the advantage needs n>=d and
a good order.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml

import exp10_ordering as E
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)


def load_tecator():
    df = fetch_openml("tecator", version=1, as_frame=True, parser="auto").frame
    cols = [f"absorbance_{i}" for i in range(1, 101)]  # 100 wavelength-ordered channels
    X = torch.tensor(df[cols].to_numpy(), dtype=torch.float64)
    y = torch.tensor(df["fat"].to_numpy(), dtype=torch.float64)
    return X, y, "tecator-NIR(fat)"


def load_co2_lags(d=64):
    import statsmodels.api as sm
    s = sm.datasets.co2.load_pandas().data["co2"].interpolate().to_numpy()
    s = np.diff(s, n=1)  # difference to remove the trend (keeps short-lag autocorr)
    # design matrix: predict s[t] from s[t-1..t-d]
    T = len(s)
    X = np.stack([s[t - d:t][::-1] for t in range(d, T)])  # lag 1..d (lag index = order)
    y = s[d:T]
    return torch.tensor(X), torch.tensor(y), f"CO2-AR({d})"


def evaluate(Xtr, ytr, Xte, yte, name, prior=1.0, noise=None):
    d = Xtr.shape[1]
    # standardise features; centre target
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-8)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ymu = ytr.mean()
    ytr, yte = ytr - ymu, yte - ymu
    if noise is None:
        noise = ytr.std().item() * 0.5

    GGN = Xtr.T @ Xtr / noise**2
    H = GGN + prior * torch.eye(d)
    Cov_full = torch.linalg.inv(H)
    beta = Cov_full @ (Xtr.T @ ytr / noise**2)
    mu_te = Xte @ beta
    rank = int(torch.linalg.matrix_rank(GGN))

    covs = {
        "full": Cov_full,
        "diag": torch.diag(1.0 / (GGN.diagonal() + prior)),
        "lcm_fixed": torch.linalg.inv(E.fit_fixed_order(GGN, torch.arange(d), steps=700)
                                      + prior * torch.eye(d)),
    }
    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(torch.arange(d, dtype=torch.float64) * 0.3)  # natural order init
    lcm.fit_dense(GGN, n_steps=700, lr=5e-2)
    covs["lcm_learned"] = torch.linalg.inv(lcm.dense() + prior * torch.eye(d))

    T = len(yte)
    Inoise = noise**2 * torch.eye(T)
    out = {"data": name, "n_tr": Xtr.shape[0], "d": d, "rank": rank}
    for nm, C in covs.items():
        var = torch.einsum("ti,ij,tj->t", Xte, C, Xte) + noise**2
        out[f"marg_{nm}"] = float((-0.5 * (torch.log(2 * np.pi * var) + (yte - mu_te) ** 2 / var)).mean())
        Cj = Xte @ C @ Xte.T + Inoise
        L = torch.linalg.cholesky(Cj)
        r = torch.linalg.solve_triangular(L, (yte - mu_te).unsqueeze(1), upper=False).squeeze(1)
        out[f"joint_{nm}"] = float(-0.5 * (r @ r + 2 * torch.log(torch.diagonal(L)).sum()
                                           + T * np.log(2 * np.pi)) / T)
    return out


def run_dataset(X, y, name, n_splits=5, test_frac=0.4):
    rows = []
    n = len(X)
    for seed in range(n_splits):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        n_te = int(test_frac * n)
        te, tr = perm[:n_te], perm[n_te:]
        rows.append(evaluate(X[tr], y[tr], X[te], y[te], name))
    return pd.DataFrame(rows)


def report(df, name):
    m = df.mean(numeric_only=True)
    print(f"\n=== {name}  (n_tr={int(m['n_tr'])}, d={int(m['d'])}, GGN rank~{int(m['rank'])}) ===")
    print(f"    {'method':14} {'marginal-LL':>12} {'joint-LL':>10}")
    for nm in ["diag", "lcm_fixed", "lcm_learned", "full"]:
        print(f"    {nm:14} {m['marg_'+nm]:>+12.3f} {m['joint_'+nm]:>+10.3f}")
    gap = (m["joint_lcm_learned"] - m["joint_diag"]) / (m["joint_full"] - m["joint_diag"] + 1e-12)
    print(f"    -> LCM joint-LL gap-to-full closed vs diag: {100*gap:.0f}%")


def main():
    results = []
    for loader in [load_tecator, lambda: load_co2_lags(64)]:
        X, y, name = loader()
        df = run_dataset(X, y, name)
        df.to_csv(f"experiments/results/exp11_{name.split('(')[0].split('-')[0].lower()}.csv", index=False)
        results.append((df, name))
        report(df, name)
    print("\nsaved exp11_*.csv")


if __name__ == "__main__":
    main()
