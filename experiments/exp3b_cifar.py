"""Experiment 3b: CIFAR-10 classification calibration & OOD (last-layer Laplace).

MNIST (Exp 3) is easy enough that the MAP network is already well calibrated, so
Laplace has little to improve. CIFAR-10 is the standard harder setting where the
MAP is over-confident; last-layer Laplace is expected to help calibration and OOD
detection. In-distribution metrics on the CIFAR-10 test set (accuracy, NLL, ECE)
and OOD-AUROC separating CIFAR-10 (in) from SVHN (out) by predictive confidence.
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
SEEDS = [0, 1]


class CifarCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
        self.features = nn.Sequential(
            *block(3, 32), *block(32, 32), nn.MaxPool2d(2),
            *block(32, 64), *block(64, 64), nn.MaxPool2d(2),
            *block(64, 128), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


NORM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))


def cifar_xy(train):
    tf = transforms.Compose([transforms.ToTensor(), NORM])
    ds = datasets.CIFAR10(DATA, train=train, download=True, transform=tf)
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    return X, y


def svhn_x(n=4000):
    tf = transforms.Compose([transforms.ToTensor(), NORM])
    ds = datasets.SVHN(DATA, split="test", download=True, transform=tf)
    return torch.stack([ds[i][0] for i in range(n)])


def train_cnn(net, Xtr, ytr, epochs=15):
    net = net.to(C.DEVICE).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.CrossEntropyLoss()
    from torch.utils.data import DataLoader, TensorDataset
    ldr = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
    for _ in range(epochs):
        for xb, yb in ldr:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
        sched.step()
    return net.eval()


def main():
    from torch.utils.data import DataLoader, TensorDataset
    print("loading CIFAR-10 / SVHN ...")
    Xtr, ytr = cifar_xy(True)
    Xte, yte = cifar_xy(False)
    Xte, yte = Xte[:5000], yte[:5000]
    Xood = svhn_x(5000)

    rows = []
    for seed in SEEDS:
        C.set_seed(seed)
        net = train_cnn(CifarCNN(), Xtr, ytr, epochs=15)
        with torch.no_grad():
            acc = (net(Xte[:2000].to(C.DEVICE)).argmax(1).cpu() == yte[:2000]).float().mean()
        print(f"[seed {seed}] MAP test acc ~ {acc:.4f}")
        gg_ldr = DataLoader(TensorDataset(Xtr[:10000], ytr[:10000]), batch_size=256)

        for mname, spec in C.METHODS.items():
            if spec is None:
                p_in = C.map_probs(net, Xte.numpy())
                p_out = C.map_probs(net, Xood.numpy())
            else:
                la, _ = C.fit_laplace(net, gg_ldr, spec[0], "classification",
                                      subset="last_layer", tune_steps=100,
                                      lcm_steps=500, lcm_lr=3e-2)
                p_in = C.classification_probs(la, Xte.numpy(), link="probit")
                p_out = C.classification_probs(la, Xood.numpy(), link="probit")
            rows.append(dict(seed=seed, method=mname,
                             acc=C.accuracy(p_in, yte.numpy()),
                             nll=C.nll_classification(p_in, yte.numpy()),
                             ece=C.ece(p_in, yte.numpy()),
                             ood_auroc=C.ood_auroc(p_in, p_out)))
            print(f"    {mname:6} acc={rows[-1]['acc']:.4f} nll={rows[-1]['nll']:.4f} "
                  f"ece={rows[-1]['ece']:.4f} ood_auroc={rows[-1]['ood_auroc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp3b_cifar_raw.csv", index=False)
    summ = (df.groupby("method")
            .agg(acc=("acc", "mean"), nll_mean=("nll", "mean"), nll_std=("nll", "std"),
                 ece_mean=("ece", "mean"), ece_std=("ece", "std"),
                 auroc_mean=("ood_auroc", "mean"), auroc_std=("ood_auroc", "std"))
            .reindex(list(C.METHODS)))
    summ.to_csv("experiments/results/exp3b_cifar_summary.csv")
    print("\n=== CIFAR-10 last-layer, mean over seeds ===")
    print(summ.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
