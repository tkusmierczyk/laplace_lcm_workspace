"""Experiment 4b: controlled ablation of the LCM design choices.

The Exp-4 ablation on a *real* GGN is limited by that matrix's intrinsic 1-D
embeddability. Here we use a target that IS exactly LCM-representable but is
presented in a *scrambled* coordinate order (and carries genuine diagonal
excess), so the effect of (a) learning the 1-D ordering and (b) the independent
diagonal 'nugget' is isolated. A method that cannot learn the ordering, or lacks
the nugget, should fail to recover the (recoverable) target.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

import common as C  # noqa: F401  (ensures repo on path / consistent env)
from exp4_scaling import fit_lcm  # reuse the ablation fitter

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
torch.set_default_dtype(torch.float64)


def make_scrambled_lcm_target(P=80, seed=0, nugget_scale=0.3):
    g = torch.Generator().manual_seed(seed)
    a = torch.sort(torch.randn(P, generator=g))[0]
    w = 0.5 + torch.rand(P, generator=g)
    d = nugget_scale * (0.2 + torch.rand(P, generator=g))  # genuine diagonal excess
    K = torch.exp(-torch.abs(a[:, None] - a[None, :]))
    A = torch.diag(d) + w[:, None] * K * w[None, :]
    perm = torch.randperm(P, generator=g)
    return A[perm][:, perm]  # same matrix, but the natural index order is scrambled


def main():
    A = make_scrambled_lcm_target()
    diag_err = float((torch.diag(torch.diag(A)) - A).norm() / A.norm())
    rows = []
    for learn_a in [True, False]:
        for nugget in [True, False]:
            err = fit_lcm(A, learn_a=learn_a, nugget=nugget, steps=1500, lr=3e-2)
            rows.append(dict(learn_a=learn_a, nugget=nugget, rel_frob=err))
            print(f"  learn_a={learn_a!s:5} nugget={nugget!s:5} -> rel Frob err={err:.4f}")
    rows.append(dict(learn_a="diag-only", nugget="-", rel_frob=diag_err))
    print(f"  diagonal-only reference          -> rel Frob err={diag_err:.4f}")
    pd.DataFrame(rows).to_csv(
        "experiments/results/exp4b_ablation_controlled.csv", index=False)
    print("\nsaved exp4b_ablation_controlled.csv")


if __name__ == "__main__":
    main()
