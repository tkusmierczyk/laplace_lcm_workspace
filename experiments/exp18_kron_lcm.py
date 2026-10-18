"""Exp 18 — Kron-LCM: the first *empirical* test of KFAC x LCM.

Until now KFAC x LCM existed only as algebra (the one-factor eigen reduction, verified
to machine precision) plus a fit-only measurement of how chain-like real KFAC factors
are. No posterior had ever been built from it and no predictive number had ever been
computed. The reduction now lives in the library as

    Laplace(model, ..., hessian_structure="kron_lcm")        # laplace.KronLCMLaplace

so this experiment is a plain comparison of library posteriors -- no bespoke linear
algebra here, which is the point: every number below comes from the same `Laplace`
interface as the baselines.

Testbed: C-way softmax classification on ordered (spectrum-like) AR(1) features. The
input Kronecker factor A = E[phi phi^T] is then an AR(1) covariance, i.e. *exactly*
inside the LCM family -- the chain's home regime. If Kron-LCM cannot win here it cannot
win anywhere. Softmax output is deliberate: it makes the per-example output Hessian vary,
so KFAC is itself an approximation and Kron-LCM is allowed to beat it.

Structures compared (all share one MAP mean, so only the covariance differs, and each
gets its prior tuned by the library's own `optimize_prior_precision(method='marglik')`):

    full            exact GGN over the layer                  (C i + C)^2
    kron            KFAC                                      i^2 + C^2
    kron (damped)   KFAC with factored damping                i^2 + C^2
    kron_lcm        LCM input factor, natural order           3i + C^2
    kron_lcm (ser)  LCM input factor, *seriated* order        3i + C^2, no side information
    diag            diagonal                                  C i + C
    lowrank         low-rank Hessian eigendecomposition        (k+1)(C i + C)

Probes follow Exp 17, which established that in-distribution marginal metrics are
trace-pinned and therefore blind: in-distribution NLL, covariate-shifted NLL, and a
joint NLL over batches of correlated query points.

Writes results/exp18_kron_lcm.csv.
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
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from laplace import Laplace  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

I, C, N, NTE, RHO, SEEDS = 256, 5, 2000, 1500, 0.9, 3
T_JOINT, N_BATCH, S_MC = 8, 120, 512

# (label, hessian_structure, extra Laplace kwargs)
STRUCTURES = [
    ("full", "full", {}),
    ("kron", "kron", {}),
    ("kron_damp", "kron", {"damping": True}),
    ("kron_lcm", "kron_lcm", {"lcm_order": "natural"}),
    ("kron_lcm_ser", "kron_lcm", {"lcm_order": "seriate"}),
    ("diag", "diag", {}),
]


def ar1(n, i, rho, g):
    e = torch.randn(n, i, generator=g)
    X = torch.empty(n, i)
    X[:, 0] = e[:, 0]
    for j in range(1, i):
        X[:, j] = rho * X[:, j - 1] + math.sqrt(1 - rho**2) * e[:, j]
    return X


def make_data(seed):
    g = torch.Generator().manual_seed(seed)
    c = torch.linspace(0, 1, I)
    W = torch.stack([
        sum(torch.sin(2 * math.pi * f * c + k + p) for f, p in [(2, 0.0), (5, 1.0)])
        for k in range(C)
    ]) * (3.0 / math.sqrt(I))
    out = {}
    for tag, X in (("tr", ar1(N, I, RHO, g)), ("in", ar1(NTE, I, RHO, g)),
                   ("sh", torch.randn(NTE, I, generator=g))):
        p = torch.softmax(X @ W.T, 1)
        out[tag] = (X, torch.multinomial(p, 1, generator=g).squeeze(1))
    # correlated query batches: T points clustered around a shared base point
    base = ar1(N_BATCH, I, RHO, g)
    Xc = base.repeat_interleave(T_JOINT, 0) + 0.3 * ar1(N_BATCH * T_JOINT, I, RHO, g)
    out["co"] = (Xc, torch.multinomial(torch.softmax(Xc @ W.T, 1), 1, generator=g).squeeze(1))
    return out


def train_map(X, y, wd=1e-2, steps=400):
    model = nn.Sequential(nn.Linear(I, C))
    opt = torch.optim.LBFGS(model.parameters(), max_iter=steps,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(X), y, reduction="sum") + 0.5 * wd * sum(
            (p**2).sum() for p in model.parameters())
        loss.backward()
        return loss
    opt.step(closure)
    return model


def nll_ece(probs, y):
    p = probs.detach().numpy()
    yi = y.numpy()
    nll = -np.log(np.maximum(p[np.arange(len(yi)), yi], 1e-300)).mean()
    conf, acc = p.max(1), p.argmax(1) == yi
    b = np.clip((conf * 10).astype(int), 0, 9)
    ece = sum(abs(conf[b == k].mean() - acc[b == k].mean()) * (b == k).mean()
              for k in range(10) if (b == k).any())
    return float(nll), float(ece), float(acc.mean())


@torch.no_grad()
def joint_nll(la, Xt, yt, rng):
    """NLL per point of the *joint* label likelihood over each correlated batch,
    using the library's own `functional_covariance` for the cross-point covariance."""
    out = []
    for b in range(N_BATCH):
        sl = slice(b * T_JOINT, (b + 1) * T_JOINT)
        Xb, yb = Xt[sl], yt[sl].numpy()
        Js = la.backend.jacobians(Xb)[0]
        mu = la.model(Xb).reshape(-1).numpy()
        Sg = la.functional_covariance(Js).numpy()
        ev, V = np.linalg.eigh(Sg)
        f = (mu[None] + rng.standard_normal((S_MC, len(mu)))
             @ (V * np.sqrt(np.clip(ev, 0, None))).T).reshape(S_MC, T_JOINT, C)
        f -= f.max(-1, keepdims=True)
        pr = np.exp(f)
        pr /= pr.sum(-1, keepdims=True)
        lp = np.log(np.maximum(pr[:, np.arange(T_JOINT), yb], 1e-300)).sum(1)
        out.append(-(np.log(np.mean(np.exp(lp - lp.max()))) + lp.max()) / T_JOINT)
    return float(np.mean(out))


@torch.no_grad()
def tempered_nll(la, X, y, grid=np.logspace(-1, 2.5, 29), S=S_MC):
    """Best NLL achievable by scaling the predictive covariance by one constant.

    The control Exp 20 showed to be essential: if inflating another method's variance
    reproduces the structured method's advantage, that advantage is calibration slack
    rather than covariance shape. Oracle -- the scale is chosen on the evaluation set
    itself -- so it upper-bounds what inflation alone can do. Covariance comes from the
    library (`_glm_predictive_distribution`); only the link sampling is done here, and
    identically for every structure.
    """
    f_mu, f_var = la._glm_predictive_distribution(X)
    f_mu = f_mu.numpy()
    ev, V = np.linalg.eigh(f_var.numpy())
    ev = np.clip(ev, 0, None)
    yi = y.numpy()
    best = (float("inf"), None)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((S, len(f_mu), f_mu.shape[1]))
    for t in grid:
        L = V * np.sqrt(t * ev)[:, None, :]
        f = f_mu[None] + np.einsum("tcd,std->stc", L, z)
        f -= f.max(-1, keepdims=True)
        pr = np.exp(f)
        pr /= pr.sum(-1, keepdims=True)
        pm = pr.mean(0)
        v = float(-np.log(np.maximum(pm[np.arange(len(yi)), yi], 1e-300)).mean())
        if v < best[0]:
            best = (v, float(t))
    return best


def n_stored(la, label):
    """Numbers held by the posterior precision -- the honest storage comparison."""
    P = la.posterior_precision
    if label.startswith("kron_lcm"):
        return sum(3 * b["n_large"] + b["n_small"] ** 2 if b["kind"] == "lcm"
                   else sum(len(l) ** 2 for l in b["ls"]) for b in P.blocks)
    if label.startswith("kron"):
        return sum(len(l) ** 2 for ls in P.eigenvalues for l in ls)
    if label == "diag":
        return la.n_params
    return la.n_params**2


def run(seed):
    rng = np.random.default_rng(1000 + seed)
    D = make_data(seed)
    model = train_map(*D["tr"])
    dl = DataLoader(TensorDataset(*D["tr"]), batch_size=256)
    out = {"seed": seed}

    for label, structure, kw in STRUCTURES:
        t0 = time.perf_counter()
        la = Laplace(model, "classification", subset_of_weights="all",
                     hessian_structure=structure, **kw)
        la.fit(dl)
        t_fit = time.perf_counter() - t0
        la.optimize_prior_precision(method="marglik")
        out[f"build_s_{label}"] = t_fit
        out[f"nparam_{label}"] = n_stored(la, label)
        out[f"prior_{label}"] = float(la.prior_precision.mean())
        out[f"marglik_{label}"] = float(la.log_marginal_likelihood())
        for tag in ("in", "sh"):
            probs = la(D[tag][0], pred_type="glm", link_approx="mc", n_samples=S_MC)
            n, e, a = nll_ece(probs, D[tag][1])
            out[f"nll_{tag}_{label}"], out[f"ece_{tag}_{label}"] = n, e
            out[f"acc_{tag}"] = a
        out[f"nll_joint_{label}"] = joint_nll(la, *D["co"], rng)
        nl, t = tempered_nll(la, *D["sh"])
        out[f"nll_sh_temp_{label}"], out[f"temp_sh_{label}"] = nl, t
    return out


def main():
    rows = []
    for s in range(SEEDS):
        t0 = time.perf_counter()
        rows.append(run(s))
        print(f"  seed {s} done in {time.perf_counter()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/exp18_kron_lcm.csv", index=False)
    m = df.mean(numeric_only=True)
    labels = [s[0] for s in STRUCTURES]

    print(f"\n  softmax model on AR(1)-ordered features: i={I}, C={C}, N={N}, rho={RHO},"
          f" {SEEDS} seeds")
    print(f"  MAP accuracy: in-dist {m['acc_in']:.3f}   shifted {m['acc_sh']:.3f}")
    print(f"\n  {'structure':15}{'stored':>9}{'fit s':>8}{'prior':>9}{'marglik':>11}"
          f"{'NLL in':>9}{'NLL shift':>11}{'ECE shift':>11}{'NLL joint':>11}")
    for k in labels:
        print(f"  {k:15}{int(m[f'nparam_{k}']):9d}{m[f'build_s_{k}']:8.2f}"
              f"{m[f'prior_{k}']:9.3f}{m[f'marglik_{k}']:11.1f}{m[f'nll_in_{k}']:9.4f}"
              f"{m[f'nll_sh_{k}']:11.4f}{m[f'ece_sh_{k}']:11.4f}{m[f'nll_joint_{k}']:11.4f}")

    # A "gap closed" figure is only meaningful when diagonal and full actually differ.
    # Exp 17 showed in-distribution queries are trace-pinned, so the diag-full gap can
    # collapse into noise -- in which case the ratio is noise over noise and is suppressed.
    print("\n  diag -> full gap, and gap closed (suppressed when the gap is within noise):")
    for probe, lbl in (("nll_in", "in-dist marginal"), ("nll_sh", "shifted marginal"),
                       ("nll_joint", "joint, correlated")):
        dg, fl = m[f"{probe}_diag"], m[f"{probe}_full"]
        sem = (df[f"{probe}_diag"] - df[f"{probe}_full"]).std() / np.sqrt(len(df))
        sep = (dg - fl) > 3 * sem
        head = f"    {lbl:20} gap {dg-fl:+7.4f} +- {sem:.4f}  "
        print(head + ("".join(f"{k}={100*(dg-m[f'{probe}_{k}'])/(dg-fl):6.1f}%  "
                              for k in labels if k not in ("diag", "full"))
                      if sep else "DEGENERATE: no separation to close, ratios suppressed"))
    print("\n  control -- best shifted NLL under an ORACLE scalar variance scale:")
    print(f"    {'structure':15}{'NLL shift':>11}{'oracle-scaled':>15}{'best scale':>12}")
    for k in labels:
        print(f"    {k:15}{m[f'nll_sh_{k}']:11.4f}{m[f'nll_sh_temp_{k}']:15.4f}"
              f"{m[f'temp_sh_{k}']:12.2f}")
    print(f"\nsaved {RESULTS}/exp18_kron_lcm.csv")


if __name__ == "__main__":
    main()
