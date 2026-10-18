"""Exp 22 — reproduce the paper's OWN claim, to test our implementation.

Every negative result in this project concerns a *transplant*: we took the Laplace
Covariance Model out of REMIX (model inversion for data-free continual learning) and used
it as the structured **precision** of a Laplace posterior. The paper does something
different. In arXiv:2605.11804, LCM models the **covariance of features**, is fitted by a
**Frobenius** objective, and is judged by the *feature log-likelihood* (their Figure 4) and
by continual-learning accuracy (their Tables 1-2).

So before trusting any of our negatives, the implementation has to clear the paper's own
bar. This reproduces their Figure 4 experiment in miniature:

    extract flattened feature maps (D = C x H x W) from several layers of a CNN,
    fit LCM and a diagonal Gaussian to the *covariance* of those features,
    and compare held-out log-likelihood.

The paper's claim is that LCM beats the diagonal baseline at every layer, including at very
high D (they report up to ~2e5). A CNN trained on local CIFAR-10 stands in for their
ResNet-34 on ImageNet classes; no pretrained weights are downloaded.

If our code reproduces this, the implementation is sound and the project's negatives are
about the transplant, not about the model. If it does not, the implementation is suspect
and every negative has to be revisited.

Writes results/exp22_paper_claim.csv.
"""

from __future__ import annotations

import math
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from laplace.lcm import LCMCovariance  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
CIFAR = "/home/tom/data/cifar-10-batches-py"

CLASSES, N_PER_CLASS, EPOCHS, STEPS_LCM = (0, 1, 8, 9), 400, 4, 150
# Sizes chosen for this laptop: the Frobenius fit costs O(N D) per step, so the
# paper's D ~ 2e5 at full dataset size is out of reach here (one layer alone ran
# past 880 s). D = 4096-16384 with N ~ 1300 keeps all three layers under a few
# minutes while still spanning the high-D regime the claim is about.


def load_cifar(classes=CLASSES, n_per=N_PER_CLASS):
    Xs, ys = [], []
    for b in range(1, 6):
        with open(f"{CIFAR}/data_batch_{b}", "rb") as f:
            d = pickle.load(f, encoding="bytes")
        Xs.append(d[b"data"])
        ys.append(np.array(d[b"labels"]))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    keep = np.concatenate([np.where(y == c)[0][:n_per] for c in classes])
    X = torch.tensor(X[keep], dtype=torch.float64).reshape(-1, 3, 32, 32) / 255.0
    X = (X - X.mean((0, 2, 3), keepdim=True)) / X.std((0, 2, 3), keepdim=True).clamp(min=1e-6)
    lab = torch.tensor([classes.index(c) for c in y[keep]])
    return X, lab


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.l1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Linear(64, n_cls)

    def forward(self, x):
        h = self.l3(self.l2(self.l1(x)))
        return self.head(h.mean((2, 3)))

    def maps(self, x):
        a = self.l1(x)
        b = self.l2(a)
        c = self.l3(b)
        return {"layer1": a, "layer2": b, "layer3": c}


def train(model, X, y, epochs=EPOCHS, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
    return model


def diag_nll(tr, te):
    """Held-out Gaussian NLL under a diagonal fit -- the paper's baseline."""
    mu, var = tr.mean(0), tr.var(0).clamp(min=1e-8)
    return float((0.5 * (math.log(2 * math.pi) + var.log() + (te - mu) ** 2 / var)).sum(1).mean())


def run(seed=0):
    X, y = load_cifar()
    torch.manual_seed(seed)
    model = train(Net(len(CLASSES)), X, y)
    rows = []
    with torch.no_grad():
        maps = {k: v.detach() for k, v in model.maps(X).items()}
    for name, M in maps.items():
        F = M.reshape(len(M), -1)                        # D = C x H x W, as in the paper
        n_tr = int(0.8 * len(F))
        tr, te = F[:n_tr], F[n_tr:]
        mu = tr.mean(0)
        ctr, cte = tr - mu, te - mu
        D = F.shape[1]
        lcm = LCMCovariance(D)
        err = lcm.fit_frobenius(ctr, n_steps=STEPS_LCM, lr=5e-2)   # the paper's objective
        with torch.no_grad():
            rows.append({
                "layer": name, "D": D, "C": M.shape[1], "HW": M.shape[2] * M.shape[3],
                "n_train": n_tr, "fit_frob_err": float(err),
                "nll_lcm": float(lcm.nll(cte)), "nll_diag": diag_nll(ctr, cte),
            })
        print(f"  {name:7} D={D:6d}  LCM {rows[-1]['nll_lcm']:12.1f}   "
              f"diag {rows[-1]['nll_diag']:12.1f}", flush=True)
    return rows


def main():
    df = pd.DataFrame(run())
    df["lcm_better"] = df.nll_lcm < df.nll_diag
    df["gain_per_dim"] = (df.nll_diag - df.nll_lcm) / df.D
    df.to_csv(f"{RESULTS}/exp22_paper_claim.csv", index=False)
    print(f"\n  Reproducing the paper's Figure 4 claim: LCM vs diagonal feature log-likelihood")
    print(f"  (CNN on local CIFAR-10, {len(CLASSES)} classes, {N_PER_CLASS} images each;"
          f" held-out 20%)\n")
    print(f"  {'layer':8}{'D = CxHxW':>12}{'n_train':>9}{'NLL LCM':>14}{'NLL diag':>14}"
          f"{'per-dim gain':>14}{'LCM wins?':>11}")
    for _, r in df.iterrows():
        print(f"  {r['layer']:8}{int(r['D']):12d}{int(r['n_train']):9d}{r['nll_lcm']:14.1f}"
              f"{r['nll_diag']:14.1f}{r['gain_per_dim']:14.4f}"
              f"{('YES' if r['lcm_better'] else 'NO'):>11}")
    print(f"\n  paper's claim (LCM beats diagonal at every layer): "
          f"{'REPRODUCED' if df.lcm_better.all() else 'NOT REPRODUCED'}")
    print(f"\nsaved {RESULTS}/exp22_paper_claim.csv")


if __name__ == "__main__":
    main()
