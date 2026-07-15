"""Experiment 5 — epistemic uncertainty for OOD detection (fixing metric (iv)).

Exp 3/3b scored OOD by *max predictive confidence*, which a MAP point estimate
already does well because it captures aleatoric (in-distribution) uncertainty.
The uncertainty a Laplace posterior adds is *epistemic*, best measured by the
Monte-Carlo predictive **mutual information** (BALD) and **predictive entropy**.
A MAP model has zero epistemic uncertainty by construction, so these scores are
the fair test of whether weight uncertainty — and its covariance structure —
helps. CIFAR-10 (in) vs SVHN (out), last-layer Laplace.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import common as C
from exp3b_cifar import CifarCNN, cifar_xy, svhn_x, train_cnn

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
SEEDS = [0, 1]
N_MC = 100
N_EVAL = 2000  # in/out points for the OOD eval


def _entropy(p, eps=1e-12):
    return -(p * np.log(p + eps)).sum(-1)


@torch.no_grad()
def mc_scores(la_or_net, X, is_map, n_mc=N_MC, batch=1000):
    """Return per-point (pred_mean, predictive_entropy, mutual_information)."""
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    means, ent, mi = [], [], []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(C.DEVICE)
        if is_map:
            p = torch.softmax(la_or_net(xb), -1).cpu().numpy()  # (b,C)
            means.append(p)
            ent.append(_entropy(p))
            mi.append(np.zeros(len(p)))  # MAP: no epistemic uncertainty
        else:
            s = la_or_net.predictive_samples(xb, pred_type="glm",
                                             n_samples=n_mc).cpu().numpy()  # (K,b,C)
            pm = s.mean(0)                       # (b,C)
            H_total = _entropy(pm)               # total predictive entropy
            H_exp = _entropy(s).mean(0)          # expected entropy (aleatoric)
            means.append(pm)
            ent.append(H_total)
            mi.append(H_total - H_exp)           # mutual information (epistemic)
    return (np.concatenate(means), np.concatenate(ent), np.concatenate(mi))


def auroc(score_in, score_out):
    y = np.concatenate([np.zeros(len(score_in)), np.ones(len(score_out))])
    s = np.concatenate([score_in, score_out])
    return float(roc_auc_score(y, s))


def main():
    print("loading CIFAR-10 / SVHN ...")
    Xtr, ytr = cifar_xy(True)
    Xte, yte = cifar_xy(False)
    Xte, yte = Xte[:N_EVAL], yte[:N_EVAL]
    Xood = svhn_x(N_EVAL)
    from torch.utils.data import DataLoader, TensorDataset

    rows = []
    for seed in SEEDS:
        C.set_seed(seed)
        net = train_cnn(CifarCNN(), Xtr, ytr, epochs=15)
        gg_ldr = DataLoader(TensorDataset(Xtr[:10000], ytr[:10000]), batch_size=256)

        for mname, spec in C.METHODS.items():
            if spec is None:
                obj, is_map = net, True
            else:
                la, _ = C.fit_laplace(net, gg_ldr, spec[0], "classification",
                                      subset="last_layer", tune_steps=100,
                                      lcm_steps=500, lcm_lr=3e-2)
                obj, is_map = la, False
            pm_in, ent_in, mi_in = mc_scores(obj, Xte.numpy(), is_map)
            pm_out, ent_out, mi_out = mc_scores(obj, Xood.numpy(), is_map)
            conf_in, conf_out = pm_in.max(1), pm_out.max(1)  # confidence
            rows.append(dict(
                seed=seed, method=mname,
                auroc_conf=auroc(-conf_in, -conf_out),   # low confidence => OOD
                auroc_entropy=auroc(ent_in, ent_out),
                auroc_mi=auroc(mi_in, mi_out),
                nll=C.nll_classification(pm_in, yte.numpy()),
                ece=C.ece(pm_in, yte.numpy()),
            ))
            r = rows[-1]
            print(f"  [seed {seed}] {mname:5} "
                  f"AUROC conf={r['auroc_conf']:.3f} entropy={r['auroc_entropy']:.3f} "
                  f"MI={r['auroc_mi']:.3f} | nll={r['nll']:.3f} ece={r['ece']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp5_epistemic_raw.csv", index=False)
    summ = (df.groupby("method")
            .agg(auroc_conf=("auroc_conf", "mean"),
                 auroc_entropy=("auroc_entropy", "mean"),
                 auroc_mi=("auroc_mi", "mean"),
                 nll=("nll", "mean"), ece=("ece", "mean"))
            .reindex(list(C.METHODS)))
    summ.to_csv("experiments/results/exp5_epistemic_summary.csv")
    print("\n=== CIFAR-10 vs SVHN, OOD-AUROC by score type (higher=better) ===")
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
