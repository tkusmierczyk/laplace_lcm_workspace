"""Exp 26 — does Exp 23's separation survive on real features, and on a network?

Exp 23 found the project's first scale-invariant separation, but on a synthetic AR(1)
GLM, and only above `rho ~ 0.9`. Two things decide whether that matters:

  1. do real ordered features reach that regime, and
  2. does a *learned* feature map reach it?

This project has already characterised both settings, with opposite diagnostics, so the
predictions are sharp and falsifiable before the run:

  Tecator NIR spectra (Exp 20)  neighbour |corr| 1.0000, seriation recovers the wavelength
                                order exactly  ->  Kron-LCM SHOULD capture much of the
                                conditional gap.
  CNN last-layer map (Exp 21)   effective rank 1.75 of 368, neighbour |corr| 0.7152 against
                                a mean of 0.2529, seriation agreement 0.0375  ->  Kron-LCM
                                should NOT, and a low-rank structure should do better.

If the spectra arm fails, Exp 23's result is an artifact of synthetic data. If the CNN arm
succeeds, Exp 21's structural diagnostics do not predict predictive behaviour and the
chain is more robust than this project has concluded.

The metric is Exp 23's: reveal every output at some test points, predict the outputs at the
others through `mu_Q + Sigma_QO Sigma_OO^-1 (y_O - mu_O)`, which is exactly invariant to
rescaling the posterior covariance. With C=3 outputs the joint covariance is taken over
(points x outputs) via the library's own `functional_covariance`.

Writes results/exp26_conditional_real.csv.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common  # noqa: E402
import exp20_kron_lcm_real as R  # Tecator loader/split   # noqa: E402
from exp21_kron_lcm_cnn import CNN, structure_diagnostics  # noqa: E402
from laplace import Laplace  # noqa: E402

ARMS = [
    ("full", "full", {}),
    ("kron", "kron", {}),
    ("kron_lcm", "kron_lcm", {"lcm_order": "natural"}),
    ("kron_lcm_ser", "kron_lcm", {"lcm_order": "seriate"}),
    ("diag", "diag", {}),
]


@torch.no_grad()
def joint_cov(la, X, subset):
    """Joint predictive mean and epistemic covariance over (points x outputs)."""
    f_mu, _ = la(X, pred_type="glm")
    Js = (la.backend.last_layer_jacobians(X) if subset == "last_layer"
          else la.backend.jacobians(X))[0]
    return f_mu, la.functional_covariance(Js)


def cond_rmse(mu, Sig, Y, io, iq, C, scale=None, sigma=0.0, rtol=1e-6):
    """RMSE of the conditional mean at the query block; ``scale=None`` is the
    exactly scale-invariant limit (noise dropped, pseudo-inverse conditioning)."""
    flat = lambda idx: torch.cat([idx * C + c for c in range(C)])  # noqa: E731
    fo, fq = flat(io).sort().values, flat(iq).sort().values
    Soo, Sqo = Sig[fo][:, fo], Sig[fq][:, fo]
    e = (Y - mu).reshape(-1)[fo]
    if scale is None:
        # Sigma_OO is rank-deficient on real features (effective rank ~1), so a jittered
        # solve would report the jitter rather than the conditional. The pseudo-inverse is
        # the well-defined limit, and stays exactly scale-free: pinv(t S) = pinv(S) / t.
        pred = mu.reshape(-1)[fq] + Sqo @ torch.linalg.pinv(Soo, rtol=rtol) @ e
        truth = Y.reshape(-1)[fq]
        return float(((pred - truth) ** 2).mean().sqrt()), float(
            ((mu.reshape(-1)[fq] - truth) ** 2).mean().sqrt())
    if scale is not None:
        Soo, Sqo = scale * Soo, scale * Sqo
        A = Soo + sigma**2 * torch.eye(len(fo), dtype=Sig.dtype)
    pred = mu.reshape(-1)[fq] + Sqo @ torch.linalg.solve(A, e.unsqueeze(-1)).squeeze(-1)
    truth = Y.reshape(-1)[fq]
    return float(((pred - truth) ** 2).mean().sqrt()), float(
        ((mu.reshape(-1)[fq] - truth) ** 2).mean().sqrt())


def run_setting(name, args, seed):
    X, Y = R.load_tecator()
    D = R.split(X, Y, seed)
    C = Y.shape[1]
    torch.manual_seed(seed)
    if args.n_train:                        # subsample into the broad-posterior regime
        g0 = torch.Generator().manual_seed(90 + seed)
        sel = torch.randperm(len(D["tr"][0]), generator=g0)[: args.n_train]
        D["tr"] = (D["tr"][0][sel], D["tr"][1][sel])

    if name == "spectra":                       # linear model on raw wavelengths
        model, subset = R.train_map(*D["tr"]), "all"
        Xtr, Xte = D["tr"][0], D["in"][0]
        feat = Xtr
    else:                                       # learned 1-D CNN feature map
        from exp21_kron_lcm_cnn import train as train_cnn
        model = train_cnn(CNN(X.shape[1], C), *D["tr"], seed=seed)
        subset, Xtr, Xte = "last_layer", D["tr"][0], D["in"][0]
        feat = model.features(Xtr).detach()
    diag, _ = structure_diagnostics(model, Xtr) if name != "spectra" else (
        {"n_feat": feat.shape[1], "nb_corr": float("nan"), "mean_abs_corr": float("nan"),
         "seriation_spearman": float("nan"), "eff_rank": float("nan")}, None)
    # chain diagnostics of whatever the Kronecker input factor actually is
    A = feat.T @ feat / len(feat)
    g = torch.sqrt(torch.diagonal(A).clamp(min=1e-30))
    Rm = A / g[:, None] / g[None, :]
    diag = {**diag, "nb_corr": float(torch.diagonal(Rm, 1).abs().mean()),
            "mean_abs_corr": float(Rm.abs().mean()),
            "eff_rank": float(torch.diagonal(A).sum() ** 2 / (A**2).sum())}

    resid = (model(Xtr) - D["tr"][1]).detach()
    sigma = float(resid.std().clamp(min=1e-3))
    dl = DataLoader(TensorDataset(*D["tr"]), batch_size=64)
    n_te = len(Xte)
    g2 = torch.Generator().manual_seed(500 + seed)
    perm = torch.randperm(n_te, generator=g2)
    io, iq = perm[: n_te // 2], perm[n_te // 2:]
    grid = np.logspace(-2, 4, 25)
    rows = []

    for label, structure, kw in ARMS:
        t0 = time.perf_counter()
        la = Laplace(model, "regression", subset_of_weights=subset,
                     hessian_structure=structure, sigma_noise=sigma, **kw)
        la.fit(dl)
        la.optimize_prior_precision(method="marglik")
        mu, Sig = joint_cov(la, Xte, subset)
        lim, marg = cond_rmse(mu, Sig, D["in"][1], io, iq, C, None)
        t1, _ = cond_rmse(mu, Sig, D["in"][1], io, iq, C, 1.0, sigma)
        best = min(cond_rmse(mu, Sig, D["in"][1], io, iq, C, float(t), sigma)[0] for t in grid)
        rows.append({"setting": name, "seed": seed, "arm": label, "sigma": sigma,
                     "rmse_marginal": marg, "rmse_cond_t1": t1,
                     "rmse_cond_limit": lim, "rmse_cond_best": best,
                     "fit_s": time.perf_counter() - t0, **diag})
        print(f"    {name:8} {label:14} marg {marg:8.4f}  cond(t=1) {t1:8.4f}  "
              f"limit {lim:8.4f}  best-t {best:8.4f}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--settings", default="spectra,cnn")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--n-train", type=int, default=0,
                   help="subsample the training set (0 = all); < input dim makes the "
                        "posterior broad, which is the regime Exp 23 ran in")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--tag", default="exp26_conditional_real")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    rows = [r for s in args.settings.split(",") for sd in range(args.seeds)
            for r in run_setting(s, args, sd)]
    for r in rows:
        r["n_train_used"] = args.n_train or -1
    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag,
                               config={**vars(args), "elapsed_s": time.perf_counter() - t0})

    names = [a[0] for a in ARMS]
    for setting, grp in df.groupby("setting"):
        d0 = grp.iloc[0]
        print(f"\n  {setting}: {int(d0['n_feat'])} input dims, neighbour |corr| "
              f"{d0['nb_corr']:.4f} vs mean {d0['mean_abs_corr']:.4f}, "
              f"effective rank {d0['eff_rank']:.2f}, {args.seeds} seeds")
        print(f"    {'arm':14}{'RMSE marg':>11}{'cond t=1':>11}{'cond limit':>12}"
              f"{'cond best-t':>13}")
        for k in names:
            g = grp[grp.arm == k]
            print(f"    {k:14}{g.rmse_marginal.mean():11.4f}{g.rmse_cond_t1.mean():11.4f}"
                  f"{g.rmse_cond_limit.mean():12.4f}{g.rmse_cond_best.mean():13.4f}")
        piv = grp.pivot_table(index="seed", columns="arm", values="rmse_cond_limit")
        print("    paired vs full on the SCALE-INVARIANT limit (positive = worse):")
        for k in names:
            if k == "full":
                continue
            dd = piv[k] - piv["full"]
            sem = dd.std() / math.sqrt(len(dd))
            sig = abs(dd.mean()) / sem if sem > 0 else float("inf")
            print(f"      {k:14}{dd.mean():+9.4f} +- {sem:.4f} ({sig:4.1f} sigma)"
                  f"{'  <- separated' if sig > 3 else '  <- within noise'}")
        dd = piv["diag"] - piv["kron_lcm"]
        sem = dd.std() / math.sqrt(len(dd))
        gap = piv["diag"].mean() - piv["full"].mean()
        sig = abs(dd.mean()) / sem if sem > 0 else float("inf")
        print(f"    Kron-LCM vs diag: {dd.mean():+8.4f} +- {sem:.4f} ({sig:4.1f} sigma)"
              + (f" -> closes {100 * dd.mean() / gap:.0f}% of the diag->exact gap"
                 if sig > 3 and gap > 0 else " -> no structural advantage"))
        for k in names:
            g = grp[grp.arm == k]
            print(f"      conditioning changes {k:14} {g.rmse_marginal.mean():8.4f} -> "
                  f"{g.rmse_cond_limit.mean():8.4f} "
                  f"({100 * (1 - g.rmse_cond_limit.mean() / g.rmse_marginal.mean()):+6.1f}%)")
    print(f"\n  total {time.perf_counter() - t0:.1f}s\nsaved {path}")


if __name__ == "__main__":
    main()
