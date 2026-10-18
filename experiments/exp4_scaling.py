"""Experiment 4: cost / scaling of the log-determinant, and ablations of the LCM.

Part A -- scaling: covariance-storage cost and posterior log-determinant wall-clock
for Diag / LCM / Full as the parameter count P grows, including a regime where the
dense P x P matrix is infeasible.

Part B -- ablations on a *real* last-layer GGN:
  (a) learned vs fixed 1-D ordering `a`,
  (b) with vs without the independent diagonal ('nugget') term,
  (c) sensitivity of the structured fit to the number of optimisation steps.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as C
from laplace import Laplace
from laplace.lcm import LCMCovariance

warnings.filterwarnings("ignore")
os.makedirs("experiments/results", exist_ok=True)
torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------- Part A: scaling
def bench_scaling():
    Ps = [200, 500, 1000, 2000, 5000, 10000, 20000]
    rows = []
    for P in Ps:
        torch.manual_seed(0)
        lcm = LCMCovariance(P)
        with torch.no_grad():
            lcm.a.copy_(torch.sort(torch.randn(P))[0])
            lcm.w.copy_(torch.randn(P) * 0.5)

        # LCM Kalman log-det
        t0 = time.perf_counter(); _ = lcm.logdet(); t_lcm = (time.perf_counter() - t0) * 1e3
        # Diag log-det (reference cheap baseline)
        diag = lcm.d + lcm.w**2
        t0 = time.perf_counter(); _ = diag.log().sum(); t_diag = (time.perf_counter() - t0) * 1e3
        # Full dense log-det where affordable
        if P <= 5000:
            A = lcm.dense()
            t0 = time.perf_counter(); _ = torch.logdet(A); t_full = (time.perf_counter() - t0) * 1e3
            del A
        else:
            t_full = float("nan")

        rows.append(dict(P=P, mem_diag=P, mem_lcm=3 * P, mem_full=P * P,
                         t_diag_ms=t_diag, t_lcm_ms=t_lcm, t_full_ms=t_full))
        print(f"  P={P:6d}  logdet ms: diag={t_diag:7.2f} lcm={t_lcm:8.2f} "
              f"full={t_full if t_full==t_full else float('nan'):8.2f}  "
              f"cov-params: lcm=3P full=P^2")
    df = pd.DataFrame(rows)
    df.to_csv("experiments/results/exp4_scaling.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax1.loglog(df.P, df.t_lcm_ms, "s-", label="LCM (Kalman)", color="#268")
    ax1.loglog(df.P, df.t_full_ms, "o-", label="Full (dense)", color="#c44")
    ax1.loglog(df.P, df.t_diag_ms, "^-", label="Diag", color="#888")
    ax1.set_xlabel("parameters P"); ax1.set_ylabel("log-det time (ms)")
    ax1.set_title("Posterior log-det cost"); ax1.legend(); ax1.grid(alpha=0.3, which="both")
    ax2.loglog(df.P, df.mem_lcm, "s-", label="LCM (3P)", color="#268")
    ax2.loglog(df.P, df.mem_full, "o-", label="Full (P^2)", color="#c44")
    ax2.loglog(df.P, df.mem_diag, "^-", label="Diag (P)", color="#888")
    ax2.set_xlabel("parameters P"); ax2.set_ylabel("covariance storage (floats)")
    ax2.set_title("Covariance memory"); ax2.legend(); ax2.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig("experiments/results/exp4_scaling.png", dpi=140)
    return df


# --------------------------------------------------------- a real last-layer GGN
def real_lastlayer_ggn():
    C.set_seed(0)
    rng = np.random.default_rng(0)
    n, d, k = 800, 10, 4
    B = rng.standard_normal((d, d))
    L = np.linalg.cholesky(B @ B.T / d + 0.3 * np.eye(d))
    X = (rng.standard_normal((n, d)) @ L.T).astype("float32")
    (Xs,), _ = C.standardize(X)
    y = (Xs @ rng.standard_normal((d, k)) + 0.5 * rng.standard_normal((n, k))).argmax(1)
    net = nn.Sequential(nn.Linear(d, 40), nn.Tanh(), nn.Linear(40, 30), nn.Tanh(),
                        nn.Linear(30, k)).to(C.DEVICE).float()
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    ldr = C.loader(Xs, y.astype("int64"), 128, True)
    for _ in range(150):
        for xb, yb in ldr:
            xb, yb = xb.to(C.DEVICE), yb.to(C.DEVICE)
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()
    la = Laplace(net, "classification", subset_of_weights="last_layer",
                 hessian_structure="full")
    la.fit(C.loader(Xs, y.astype("int64"), 256))
    return la.H.detach().double().cpu()


def fit_lcm(target, learn_a=True, nugget=True, steps=800, lr=3e-2):
    P = target.shape[0]
    lcm = LCMCovariance(P)
    if not nugget:  # freeze diagonal near the numerical floor
        with torch.no_grad():
            lcm.u_raw.copy_(torch.full((P,), -20.0))
    params = [lcm.w]
    if learn_a:
        params.append(lcm.a)
    if nugget:
        params.append(lcm.u_raw)
    opt = torch.optim.Adam(params, lr=lr)
    tnorm = target.norm().clamp(min=1e-12)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((lcm.dense() - target) ** 2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return float((lcm.dense() - target).norm() / tnorm)


def ablations():
    H = real_lastlayer_ggn()
    print(f"\nreal last-layer GGN: P={H.shape[0]}")
    # (a) learned vs fixed ordering, (b) nugget on/off
    rows = []
    for learn_a in [True, False]:
        for nugget in [True, False]:
            err = fit_lcm(H, learn_a=learn_a, nugget=nugget, steps=800)
            rows.append(dict(learn_a=learn_a, nugget=nugget, rel_frob=err))
            print(f"  learn_a={learn_a!s:5} nugget={nugget!s:5} -> rel Frob err={err:.4f}")
    # diagonal-only reference
    diag_err = float((torch.diag(torch.diag(H)) - H).norm() / H.norm())
    print(f"  diagonal-only reference          -> rel Frob err={diag_err:.4f}")
    pd.DataFrame(rows + [dict(learn_a="diag", nugget="-", rel_frob=diag_err)]
                 ).to_csv("experiments/results/exp4_ablation_structure.csv", index=False)

    # (c) steps sensitivity
    steprows = []
    for steps in [50, 100, 200, 400, 800, 1600]:
        err = fit_lcm(H, learn_a=True, nugget=True, steps=steps)
        steprows.append(dict(steps=steps, rel_frob=err))
        print(f"  steps={steps:5d} -> rel Frob err={err:.4f}")
    pd.DataFrame(steprows).to_csv("experiments/results/exp4_ablation_steps.csv", index=False)


if __name__ == "__main__":
    print("=== Part A: scaling of the posterior log-determinant ===")
    bench_scaling()
    print("\n=== Part B: ablations on a real last-layer GGN ===")
    ablations()
    print("\nsaved exp4_scaling.csv/png, exp4_ablation_structure.csv, exp4_ablation_steps.csv")
