"""Experiment 9 — the application: batch active learning needs JOINT uncertainty.

Batch selection is where joint predictive uncertainty pays off. To pick B jointly
informative points one must account for *redundancy* between candidates — their
predictive correlation. Diagonal Laplace under-models that correlation, so it
selects redundant high-variance points; a structured covariance selects a diverse,
jointly-informative batch. We use each posterior's predictive covariance to greedily
maximise the batch information gain  0.5 logdet(I + Cov_S / noise^2)  (D-optimal /
BatchBALD-style), then score the selected batch by its *true* information gain under
the exact full posterior. Higher = better decisions. Random and marginal-variance
(top-k) selection are included as references.

Setting: 1-D-correlated-feature regression (LCM's structural home; KFAC N/A, full
is O(d^3)). LCM uses only 3d parameters and an O(d) predictive quadratic form.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch

from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
os.makedirs("experiments/results", exist_ok=True)


def simulate(n, d, rho, seed, noise):
    g = torch.Generator().manual_seed(seed)
    def gen(m):
        eps = torch.randn(m, d, generator=g); X = torch.empty(m, d)
        X[:, 0] = eps[:, 0]; s = (1 - rho**2) ** 0.5
        for j in range(1, d):
            X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
        return X
    coord = torch.linspace(0, 1, d)
    beta = sum(torch.sin(2*np.pi*f*coord+p) for f,p in [(2,0.),(5,1.),(9,2.)]) / d**0.5
    return gen, beta, g


def info_gain(Cov_pred):
    """0.5 logdet(I + Cov_pred/noise^2) is folded into noise already; here Cov_pred
    is the batch predictive covariance (incl. the process part only)."""
    T = Cov_pred.shape[0]
    return float(torch.logdet(torch.eye(T) + Cov_pred))


def run(d, seed, n_lab=40, n_pool=800, n_test=1000, B=25, rho=0.9, noise=0.5, prior=1.0):
    gen, beta, _ = simulate(n_lab, d, rho, seed, noise)
    Xl = gen(n_lab); Xp = gen(n_pool); Xt = gen(n_test)
    g = torch.Generator().manual_seed(seed + 100)
    yl = Xl@beta + noise*torch.randn(n_lab, generator=g)

    GGN = Xl.T@Xl / noise**2
    H = GGN + prior*torch.eye(d)
    Cov_full = torch.linalg.inv(H)

    lcm = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        lcm.a.copy_(torch.arange(d) * -np.log(rho))
        lcm.w.copy_(torch.full((d,), float(GGN.diagonal().mean().sqrt())))
        lcm.u_raw.copy_(torch.full((d,), -2.0))
    lcm.fit_dense(GGN, n_steps=700, lr=5e-2)
    Cov_lcm = torch.linalg.inv(lcm.dense() + prior*torch.eye(d))
    Cov_diag = torch.diag(1.0/(GGN.diagonal() + prior))

    # predictive covariance over the pool for each method: K = Xp Cov Xp^T
    def predcov(Cov):
        return (Xp @ Cov) @ Xp.T
    K = {"full": predcov(Cov_full), "lcm": predcov(Cov_lcm), "diag": predcov(Cov_diag)}
    Ktrue = K["full"]

    def greedy(Kmat):
        sel, avail = [], list(range(n_pool))
        C = None
        for _ in range(B):
            diagK = Kmat.diagonal()
            if not sel:
                i = int(torch.tensor([diagK[j] for j in avail]).argmax()); i = avail[i]
            else:
                Ci = torch.linalg.inv(C)
                best, bg = avail[0], -1e18
                for j in avail:
                    c = Kmat[sel, j]
                    cond = float(diagK[j] - c @ Ci @ c)
                    if cond > bg:
                        bg, best = cond, j
                i = best
            sel.append(i); avail.remove(i)
            C = Kmat[np.ix_(sel, sel)] + 1e-9*torch.eye(len(sel))
        return sel

    def topk(Kmat):  # marginal-variance selection (ignores redundancy)
        return list(torch.topk(Kmat.diagonal(), B).indices.numpy())

    rng = np.random.default_rng(seed)
    batches = {
        "random": list(rng.choice(n_pool, B, replace=False)),
        "top-k(diag)": topk(K["diag"]),
        "greedy(diag)": greedy(K["diag"]),
        "greedy(lcm)": greedy(K["lcm"]),
        "greedy(full)": greedy(K["full"]),
    }
    # score every selected batch by its TRUE information gain under the full posterior
    out = {"d": d, "seed": seed}
    for name, S in batches.items():
        Ctrue = Ktrue[np.ix_(S, S)]
        out[name] = info_gain(Ctrue)
    return out


def main():
    rows = []
    for d in [500, 1000, 1500]:
        for seed in [0, 1, 2]:
            rows.append(run(d, seed))
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp9_active_learning.csv", index=False)
    cols = ["random", "top-k(diag)", "greedy(diag)", "greedy(lcm)", "greedy(full)"]
    print("=== Batch active learning: TRUE information gain of the selected batch ===")
    print("    (higher = more informative batch; greedy(full) is the oracle)\n")
    print(f"{'d':>5} | " + " ".join(f"{c:>13}" for c in cols))
    for d, sub in df.groupby("d"):
        m = sub[cols].mean()
        print(f"{int(d):>5} | " + " ".join(f"{m[c]:>13.2f}" for c in cols))
    # normalised: fraction of the oracle gain, relative to random
    print("\n% of oracle (greedy-full) information gain, above the random baseline:")
    for d, sub in df.groupby("d"):
        m = sub[cols].mean()
        base, orac = m["random"], m["greedy(full)"]
        def freac(x): return 100*(x-base)/(orac-base+1e-12)
        print(f"  d={int(d):>5}  top-k(diag)={freac(m['top-k(diag)']):5.0f}%  "
              f"greedy(diag)={freac(m['greedy(diag)']):5.0f}%  greedy(lcm)={freac(m['greedy(lcm)']):5.0f}%")
    print("\nsaved exp9_active_learning.csv")


if __name__ == "__main__":
    main()
