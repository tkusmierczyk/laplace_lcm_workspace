"""Exp 20 — Kron-LCM on real ordered data, with more than one output.

Exp 18 tested the construction on synthetic AR(1) features. Two things were still
missing, and both are fixed here.

1. **Real ordered inputs.** Tecator near-infrared spectroscopy: 240 meat samples, 100
   wavelength-ordered absorbance channels. The chain axis is a physical wavelength axis,
   not a simulated one.
2. **More than one output.** Earlier plans used a single regression target, which makes
   the Kronecker structure vacuous (`C=1` means KFAC *is* the full layer Hessian, so
   there is nothing to test). Tecator ships three targets -- moisture, fat, protein -- so
   `C=3` and the Kronecker factorisation is real.

Note what that buys and costs: with an isotropic Gaussian likelihood the output factor is
constant across examples, so **KFAC is exact here** and coincides with the full layer
posterior. That removes KFAC's own approximation error and isolates the one question that
matters for Kron-LCM: what does replacing the 100x100 input factor with a 300-number
chain cost, when everything else is exact? It also gives a free internal check -- `kron`
and `full` must agree.

Probes: in-distribution test likelihood, and a **real covariate shift** -- train on the
low- and mid-fat samples, test on the highest-fat ones, which is extrapolation along the
physically meaningful direction rather than a synthetic perturbation.

Writes results/exp20_real.csv.
"""

from __future__ import annotations

import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from laplace import Laplace  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

N_SHIFT, N_TEST, SEEDS = 50, 50, 5
STRUCTURES = [
    ("full", "full", {}),
    ("kron", "kron", {}),
    ("kron_lcm", "kron_lcm", {"lcm_order": "natural"}),
    ("kron_lcm_ser", "kron_lcm", {"lcm_order": "seriate"}),
    ("diag", "diag", {}),
]


def load_tecator():
    df = fetch_openml("tecator", version=1, as_frame=True, parser="auto").frame
    X = torch.tensor(df[[f"absorbance_{i}" for i in range(1, 101)]].values, dtype=torch.float64)
    Y = torch.tensor(df[["moisture", "fat", "protein"]].values, dtype=torch.float64)
    return X, Y


def split(X, Y, seed):
    """Highest-fat samples are the shifted test set; the rest split train/in-dist test."""
    fat_order = torch.argsort(Y[:, 1])
    shift_idx = fat_order[-N_SHIFT:]
    rest = fat_order[:-N_SHIFT]
    g = torch.Generator().manual_seed(seed)
    rest = rest[torch.randperm(len(rest), generator=g)]
    te_idx, tr_idx = rest[:N_TEST], rest[N_TEST:]
    mx, sx = X[tr_idx].mean(0), X[tr_idx].std(0).clamp(min=1e-8)
    my, sy = Y[tr_idx].mean(0), Y[tr_idx].std(0).clamp(min=1e-8)
    out = {}
    for tag, idx in (("tr", tr_idx), ("in", te_idx), ("sh", shift_idx)):
        out[tag] = ((X[idx] - mx) / sx, (Y[idx] - my) / sy)
    return out


def train_map(X, Y, wd=1e-2, steps=500):
    model = nn.Linear(X.shape[1], Y.shape[1])
    opt = torch.optim.LBFGS(model.parameters(), max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = ((model(X) - Y) ** 2).sum() + 0.5 * wd * sum((p**2).sum() for p in model.parameters())
        loss.backward()
        return loss
    opt.step(closure)
    return model


@torch.no_grad()
def gaussian_nll(la, X, Y, sigma):
    """Mean multivariate-Gaussian NLL per sample, using the library's GLM predictive."""
    f_mu, f_var = la(X, pred_type="glm")
    C = Y.shape[1]
    S = f_var + sigma**2 * torch.eye(C, dtype=Y.dtype)
    e = (Y - f_mu).unsqueeze(-1)
    sol = torch.linalg.solve(S, e).squeeze(-1)
    quad = (sol * (Y - f_mu)).sum(-1)
    logdet = torch.logdet(S)
    nll = 0.5 * (C * math.log(2 * math.pi) + logdet + quad)
    z2 = quad / C
    return float(nll.mean()), float(z2.mean())


@torch.no_grad()
def gaussian_nll_tempered(la, X, Y, sigma, grid=np.logspace(-1, 3, 41)):
    """Best NLL achievable by scaling the posterior covariance by a constant.

    The decisive control for the shifted probe: if simply inflating another method's
    variance matches Kron-LCM, its advantage is calibration slack rather than structure.
    This is an *oracle* -- the scale is chosen on the evaluation set itself -- so it
    upper-bounds what variance inflation alone can do.
    """
    f_mu, f_var = la(X, pred_type="glm")
    C = Y.shape[1]
    eye = torch.eye(C, dtype=Y.dtype)
    e = Y - f_mu
    best = (float("inf"), None)
    for t in grid:
        S = t * f_var + sigma**2 * eye
        sol = torch.linalg.solve(S, e.unsqueeze(-1)).squeeze(-1)
        nll = 0.5 * (C * math.log(2 * math.pi) + torch.logdet(S) + (sol * e).sum(-1))
        v = float(nll.mean())
        if v < best[0]:
            best = (v, float(t))
    return best


def run(seed):
    X, Y = load_tecator()
    D = split(X, Y, seed)
    model = train_map(*D["tr"])
    resid = (model(D["tr"][0]) - D["tr"][1]).detach()
    sigma = float(resid.std().clamp(min=1e-3))
    dl = DataLoader(TensorDataset(*D["tr"]), batch_size=64)
    out = {"seed": seed, "sigma": sigma, "n_train": len(D["tr"][0])}

    for label, structure, kw in STRUCTURES:
        t0 = time.perf_counter()
        la = Laplace(model, "regression", subset_of_weights="all",
                     hessian_structure=structure, sigma_noise=sigma, **kw)
        la.fit(dl)
        la.optimize_prior_precision(method="marglik")
        out[f"build_s_{label}"] = time.perf_counter() - t0
        out[f"prior_{label}"] = float(la.prior_precision.mean())
        out[f"marglik_{label}"] = float(la.log_marginal_likelihood())
        P = la.posterior_precision
        if label.startswith("kron_lcm"):
            out[f"nparam_{label}"] = sum(
                3 * b["n_large"] + b["n_small"] ** 2 if b["kind"] == "lcm"
                else sum(len(l) ** 2 for l in b["ls"]) for b in P.blocks)
        elif label == "kron":
            out[f"nparam_{label}"] = sum(len(l) ** 2 for ls in P.eigenvalues for l in ls)
        else:
            out[f"nparam_{label}"] = la.n_params if label == "diag" else la.n_params**2
        for tag in ("in", "sh"):
            n, z = gaussian_nll(la, *D[tag], sigma)
            out[f"nll_{tag}_{label}"], out[f"z2_{tag}_{label}"] = n, z
        nl, t = gaussian_nll_tempered(la, *D["sh"], sigma)
        out[f"nll_sh_temp_{label}"], out[f"temp_sh_{label}"] = nl, t
    return out


def main():
    df = pd.DataFrame([run(s) for s in range(SEEDS)])
    df.to_csv(f"{RESULTS}/exp20_real.csv", index=False)
    m = df.mean(numeric_only=True)
    labels = [s[0] for s in STRUCTURES]
    print(f"  Tecator NIR: 100 wavelength channels, C=3 targets, n_train={int(m['n_train'])}, "
          f"{SEEDS} seeds; shifted set = {N_SHIFT} highest-fat samples")
    print(f"\n  {'structure':15}{'stored':>9}{'prior':>9}{'marglik':>10}"
          f"{'NLL in':>10}{'NLL shift':>11}{'z2 in':>9}{'z2 shift':>10}")
    for k in labels:
        print(f"  {k:15}{int(m[f'nparam_{k}']):9d}{m[f'prior_{k}']:9.3f}{m[f'marglik_{k}']:10.1f}"
              f"{m[f'nll_in_{k}']:10.4f}{m[f'nll_sh_{k}']:11.4f}"
              f"{m[f'z2_in_{k}']:9.3f}{m[f'z2_sh_{k}']:10.3f}")
    print("\n  diag -> full gap and gap closed (suppressed when within noise):")
    for probe, lbl in (("nll_in", "in-dist"), ("nll_sh", "shifted (high fat)")):
        dg, fl = m[f"{probe}_diag"], m[f"{probe}_full"]
        sem = (df[f"{probe}_diag"] - df[f"{probe}_full"]).std() / np.sqrt(len(df))
        head = f"    {lbl:20} gap {dg-fl:+8.4f} +- {sem:.4f}  "
        print(head + ("".join(f"{k}={100*(dg-m[f'{probe}_{k}'])/(dg-fl):6.1f}%  "
                              for k in labels if k not in ("diag", "full"))
                      if (dg - fl) > 3 * sem else "DEGENERATE: ratios suppressed"))
    print(f"\n  control -- best NLL on the shifted set under an ORACLE scalar variance scale:")
    print(f"    {'structure':15}{'NLL shift':>11}{'oracle-scaled':>15}{'best scale':>12}")
    for k in labels:
        print(f"    {k:15}{m[f'nll_sh_{k}']:11.4f}{m[f'nll_sh_temp_{k}']:15.4f}"
              f"{m[f'temp_sh_{k}']:12.1f}")
    d = (m["nll_in_kron"] - m["nll_in_full"], m["marglik_kron"] - m["marglik_full"])
    print(f"\n  internal check, KFAC should be exact for isotropic Gaussian output:"
          f" dNLL {d[0]:+.2e}  dmarglik {d[1]:+.2e}")
    print(f"\nsaved {RESULTS}/exp20_real.csv")


if __name__ == "__main__":
    main()
