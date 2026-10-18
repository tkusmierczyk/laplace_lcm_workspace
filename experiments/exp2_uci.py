"""Experiment 2: UCI regression uncertainty-quantification benchmark.

Standard single-hidden-layer BNN protocol (Hernandez-Lobato & Adams style):
inputs and targets standardised, MAP training, then an all-weights Laplace
approximation whose prior precision and observation noise are tuned by the
marginal likelihood. We report the test log-likelihood (original target scale)
and RMSE, averaged over random splits, for MAP / Diag / KFAC / Full / LCM.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import (fetch_california_housing, fetch_openml,
                              load_diabetes)

import common as C

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)

N_SPLITS = 8
HIDDEN = 50
MAX_N = 3000  # subsample the largest datasets for tractable full/kfac/lcm fits
EPOCHS = 120


def _load():
    ds = {}
    d = load_diabetes()
    ds["diabetes"] = (d.data.astype("float32"), d.target.astype("float32"))
    ch = fetch_california_housing()
    ds["california"] = (ch.data.astype("float32"), ch.target.astype("float32"))
    for key, name in [("concrete", "concrete_compressive_strength"),
                      ("energy", "energy-efficiency"),
                      ("yacht", "yacht_hydrodynamics"),
                      ("wine", "wine-quality-red")]:
        o = fetch_openml(name, as_frame=False, parser="auto")
        X = np.asarray(o.data, dtype="float32")
        y = np.asarray(o.target, dtype="float32")
        ds[key] = (X, y)
    return ds


def run_dataset(name, X, y):
    rows = []
    for split in range(N_SPLITS):
        C.set_seed(split)
        rng = np.random.default_rng(split)
        perm = rng.permutation(len(X))
        if len(perm) > MAX_N:
            perm = perm[:MAX_N]
        n_test = max(20, int(0.1 * len(perm)))
        te, tr = perm[:n_test], perm[n_test:]
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr].reshape(-1, 1), y[te].reshape(-1, 1)
        (Xtr_s, Xte_s), _ = C.standardize(Xtr, Xte)
        y_mu, y_sd = ytr.mean(), ytr.std() + 1e-8
        ytr_s = ((ytr - y_mu) / y_sd).astype("float32")

        d = Xtr_s.shape[1]
        net = nn.Sequential(nn.Linear(d, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
        net = C.train_map(net, C.loader(Xtr_s, ytr_s, 128, True),
                          epochs=EPOCHS, lr=1e-2, weight_decay=1e-4,
                          likelihood="regression")
        ldr = C.loader(Xtr_s, ytr_s, 256)

        for mname, spec in C.METHODS.items():
            if spec is None:
                # MAP with a homoscedastic noise set to the training residual std
                with torch.no_grad():
                    pred_tr = net(torch.as_tensor(Xtr_s, device=C.DEVICE)).squeeze(-1)
                res = pred_tr.cpu().numpy() - ytr_s.reshape(-1)
                sigma = max(res.std(), 1e-2)
                mu, var = C.map_regression_predictive(net, Xte_s, sigma)
            else:
                la, _ = C.fit_laplace(net, ldr, spec[0], "regression", subset="all",
                                      tune_steps=150, tune_lr=1e-1)
                mu, var = C.regression_predictive(la, Xte_s)
            ll = C.gaussian_test_ll(mu * y_sd + y_mu, var * y_sd**2,
                                    yte, 1.0)  # predict in original scale directly
            rmse = C.rmse(mu * y_sd + y_mu, yte, 1.0)
            rows.append(dict(dataset=name, N=len(perm), d=d, split=split,
                             method=mname, test_ll=ll, rmse=rmse))
    return rows


def main():
    ds = _load()
    all_rows = []
    for name, (X, y) in ds.items():
        print(f"[{name}] N={len(X)} d={X.shape[1]}")
        rows = run_dataset(name, X, y)
        all_rows.extend(rows)
        df = pd.DataFrame(rows)
        piv = df.groupby("method")[["test_ll", "rmse"]].mean()
        for m in C.METHODS:
            print(f"    {m:6} LL={piv.loc[m,'test_ll']:+.3f}  RMSE={piv.loc[m,'rmse']:.3f}")
    full = pd.DataFrame(all_rows)
    full.to_csv("experiments/results/exp2_uci_raw.csv", index=False)

    # summary table: mean +/- std over splits, per dataset x method
    summ = (full.groupby(["dataset", "method"])
            .agg(ll_mean=("test_ll", "mean"), ll_std=("test_ll", "std"),
                 rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"))
            .reset_index())
    summ.to_csv("experiments/results/exp2_uci_summary.csv", index=False)

    # average rank of each method by test-LL (higher LL = rank 1)
    ranks = (full.groupby(["dataset", "split", "method"])["test_ll"].mean()
             .reset_index())
    ranks["rank"] = (ranks.groupby(["dataset", "split"])["test_ll"]
                     .rank(ascending=False))
    avg_rank = ranks.groupby("method")["rank"].mean().sort_values()
    print("\n=== Average test-LL rank across datasets/splits (1 = best) ===")
    print(avg_rank.to_string())
    avg_rank.to_csv("experiments/results/exp2_uci_avgrank.csv")
    print("\nsaved exp2_uci_{raw,summary,avgrank}.csv")


if __name__ == "__main__":
    main()
