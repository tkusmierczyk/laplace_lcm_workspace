"""Experiment 3: image classification uncertainty + OOD detection (last-layer Laplace).

A small CNN is trained on an in-distribution dataset (MNIST); last-layer Laplace
approximations (Diag / KFAC / Full / LCM) are then fit and compared on
in-distribution calibration (test accuracy, NLL, ECE) and out-of-distribution
detection (AUROC separating the in-distribution test set from an OOD set,
FashionMNIST, by predictive confidence). This is the standard 'Laplace Redux'
last-layer evaluation.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets, transforms

import common as C

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
DATA = "/home/tom/data"
SEEDS = [0, 1, 2]


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 84), nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def _tensors(train, ood=False):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    D = datasets.FashionMNIST if ood else datasets.MNIST
    ds = D(DATA, train=train, download=True, transform=tf)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    return X, y


def loader_xy(X, y, bs, shuffle=False):
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)


def train_cnn(net, Xtr, ytr, epochs=5):
    net = net.to(C.DEVICE).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    ldr = loader_xy(Xtr, ytr, 256, shuffle=True)
    for _ in range(epochs):
        for xb, yb in ldr:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    return net.eval()


def main():
    print("loading data ...")
    Xtr, ytr = _tensors(train=True)
    Xte, yte = _tensors(train=False)
    Xood, _ = _tensors(train=False, ood=True)
    # subsample for speed of the OOD/eval and last-layer GGN
    Xte, yte = Xte[:4000], yte[:4000]
    Xood = Xood[:4000]

    rows = []
    for seed in SEEDS:
        C.set_seed(seed)
        net = train_cnn(SmallCNN(), Xtr, ytr, epochs=5)
        with torch.no_grad():
            acc = (net(Xte.to(C.DEVICE)).argmax(1).cpu() == yte).float().mean().item()
        print(f"[seed {seed}] MAP test acc = {acc:.4f}")
        train_ldr = loader_xy(Xtr[:8000], ytr[:8000], 256)  # GGN over a subset

        for mname, spec in C.METHODS.items():
            if spec is None:
                p_in = C.map_probs(net, Xte.numpy())
                p_out = C.map_probs(net, Xood.numpy())
            else:
                la, _ = C.fit_laplace(net, train_ldr, spec[0], "classification",
                                      subset="last_layer", tune_steps=100)
                p_in = C.classification_probs(la, Xte.numpy(), link="probit")
                p_out = C.classification_probs(la, Xood.numpy(), link="probit")
            rows.append(dict(
                seed=seed, method=mname,
                acc=C.accuracy(p_in, yte.numpy()),
                nll=C.nll_classification(p_in, yte.numpy()),
                ece=C.ece(p_in, yte.numpy()),
                ood_auroc=C.ood_auroc(p_in, p_out),
            ))
            print(f"    {mname:6} acc={rows[-1]['acc']:.4f} nll={rows[-1]['nll']:.4f} "
                  f"ece={rows[-1]['ece']:.4f} ood_auroc={rows[-1]['ood_auroc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp3_image_raw.csv", index=False)
    summ = (df.groupby("method")
            .agg(acc=("acc", "mean"), nll_mean=("nll", "mean"), nll_std=("nll", "std"),
                 ece_mean=("ece", "mean"), ece_std=("ece", "std"),
                 auroc_mean=("ood_auroc", "mean"), auroc_std=("ood_auroc", "std"))
            .reindex(list(C.METHODS)))
    summ.to_csv("experiments/results/exp3_image_summary.csv")
    print("\n=== MNIST last-layer, mean over seeds (NLL/ECE lower=better, AUROC higher=better) ===")
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nsaved exp3_image_{raw,summary}.csv")


if __name__ == "__main__":
    main()
