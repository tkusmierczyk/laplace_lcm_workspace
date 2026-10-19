"""Exp 25 — the paper's claim at the paper's scale, against a baseline that can fight back.

GAP THIS CLOSES. Exp 22 reproduced arXiv:2605.11804's Figure 4: fitted by their
Frobenius objective, LCM beats a diagonal Gaussian on held-out feature log-likelihood
at every CNN layer. That is the project's one positive result, and it inherits the
paper's *only* baseline -- a diagonal. A diagonal losing shows that some correlation
helps; it does not show that a **chain** is the right correlation. Exp 14 already found
low-rank overtakes LCM on network curvature once it has comparable parameters, but that
was inside this project's precision transplant, never in the paper's own covariance
setting. Exp 22 also ran one seed at D <= 16384 because the laptop could not go further.

HYPOTHESIS. The chain is genuinely the right structure for conv feature covariance, so
LCM beats *any* family of comparable parameter count, not just a diagonal, and the
advantage persists at the paper's scale (D = C x H x W up to ~2e5).

THE CLAIM A PAPER WOULD MAKE. "A 3D-parameter Laplace chain captures feature covariance
better than alternative 3D-parameter structures, at dimensions where dense covariance is
impossible."

CHEAPEST FALSIFIER, AND WHAT WOULD MAKE US DROP IT. Fit, at matched parameter count, a
rank-2 factor analyser (2D loadings + D noise = 3D, exactly LCM's budget) by maximum
likelihood on the same features and compare held-out NLL. **If factor analysis matches or
beats LCM at equal budget, the chain carries no structural advantage** and the Figure 4
result reduces to "correlation beats no correlation". Two further arms decide how to read
whatever happens:

  lcm_fixed_a   coordinates frozen at equal spacing (2D parameters): a stationary AR(1)
                chain. If this matches full LCM, the *learned* coordinates buy nothing --
                and Exps 10/16 already showed they cannot be learned from data.
  lcm_ml        the same chain fitted by exact maximum likelihood instead of Frobenius.
                The comparison is scored by held-out likelihood, so the Frobenius arm is
                being judged on an objective it never optimised; this arm is the fair
                version of LCM, and if it wins, it is a constructive improvement to the
                paper's own fitting rule.

Needs a machine the laptop is not: at D = 196608 the feature matrix alone is several GiB
in float64 and the fits are O(N D) per step. Runs on the GMUM `cpu` partition.

    python exp25_paper_scale_baselines.py --dataset cifar100 --data-root /shared/sets/datasets \
        --channels 16,64,192 --n 3000 --seeds 3

Writes results/exp25_paper_scale.csv (+ .meta.json with both repo hashes).
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common  # noqa: E402
from laplace.lcm import LCMCovariance  # noqa: E402


# ----------------------------------------------------------------------- data
def load_cifar(root: str, dataset: str, n_images: int, seed: int):
    """CIFAR-10/100 from the standard python pickles; no download, no torchvision."""
    if dataset == "cifar10":
        d = os.path.join(root, "cifar-10-batches-py")
        files, key = [f"data_batch_{b}" for b in range(1, 6)], b"labels"
    else:
        d = os.path.join(root, "cifar-100-python")
        files, key = ["train"], b"fine_labels"
    Xs, ys = [], []
    for fn in files:
        with open(os.path.join(d, fn), "rb") as f:
            raw = pickle.load(f, encoding="bytes")
        Xs.append(raw[b"data"])
        ys.append(np.asarray(raw[key]))
    X, y = np.concatenate(Xs), np.concatenate(ys)
    g = np.random.default_rng(seed)
    keep = g.permutation(len(X))[:n_images]
    X = torch.tensor(X[keep], dtype=torch.float64).reshape(-1, 3, 32, 32) / 255.0
    X = (X - X.mean((0, 2, 3), keepdim=True)) / X.std((0, 2, 3), keepdim=True).clamp(min=1e-6)
    return X, torch.tensor(y[keep], dtype=torch.long)


class WideCNN(nn.Module):
    """Stride-1 stack, so the flattened maps reach the paper's D = C x H x W ~ 2e5."""

    def __init__(self, channels, n_cls):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = 3
        for c in channels:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()))
            c_in = c
        self.head = nn.Linear(c_in, n_cls)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x.mean((2, 3)))

    @torch.no_grad()
    def layer_features(self, x, layer, batch=250):
        """Flattened feature map of one layer, written into one preallocated buffer.

        A list of chunks plus `torch.cat` would briefly hold the whole matrix twice --
        4.7 GB each at D=196608 with 3000 images, which is what OOM-killed the first
        cluster attempt. Preallocating removes that spike.
        """
        out = None
        for i in range(0, len(x), batch):
            h = x[i:i + batch]
            for b in self.blocks[:layer + 1]:
                h = b(h)
            h = h.reshape(len(h), -1)
            if out is None:
                out = torch.empty(len(x), h.shape[1], dtype=h.dtype, device=h.device)
            out[i:i + len(h)] = h
        return out


def train(model, X, y, epochs, lr=2e-3, batch=64, seed=0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.cross_entropy(model(X[idx]), y[idx]).backward()
            opt.step()
    return model


# ------------------------------------------------------------------- families
def diag_nll(tr, te, floor):
    """Held-out Gaussian NLL under a diagonal fit -- the paper's baseline. D params."""
    var = tr.var(0).clamp(min=floor)
    return float((0.5 * (math.log(2 * math.pi) + var.log() + te**2 / var)).sum(1).mean())


def fa_fit(X, rank, n_iter=40, seed=0, floor=1e-10):
    """Factor analysis ``Sigma = W W^T + diag(psi)`` by EM. (rank+1)*D parameters.

    Matched against LCM's 3D at rank 2. Everything is O(N D rank); the D x D
    covariance is never formed, exactly as for the chain.
    """
    N, D = X.shape
    g = torch.Generator().manual_seed(seed)
    U, S, V = torch.pca_lowrank(X, q=rank, niter=4)        # randomised SVD init
    W = V * (S / math.sqrt(N)).clamp(min=1e-8)
    psi = (X**2).mean(0).clamp(min=floor)
    eye = torch.eye(rank, dtype=X.dtype)
    for _ in range(n_iter):
        G = W / psi[:, None]                               # Psi^-1 W      (D, r)
        M = eye + W.T @ G                                  # I + W' Psi^-1 W
        Minv = torch.linalg.inv(M)
        Ez = (X @ G) @ Minv                                # (N, r)
        Szz = N * Minv + Ez.T @ Ez
        W = torch.linalg.solve(Szz.T, (X.T @ Ez).T).T      # (X'Ez) Szz^-1
        psi = ((X**2).mean(0) - (X * (Ez @ W.T)).mean(0)).clamp(min=floor)
    del G, Ez
    return W, psi


def fa_nll(W, psi, te):
    """Held-out NLL under ``W W^T + diag(psi)``, by the determinant lemma + Woodbury."""
    D, rank = W.shape
    G = W / psi[:, None]
    M = torch.eye(rank, dtype=W.dtype) + W.T @ G
    logdet = psi.log().sum() + torch.logdet(M)
    P = te @ G                                             # (n, r)
    quad = (te**2 / psi).sum(1) - (P * torch.linalg.solve(M, P.T).T).sum(1)
    return float((0.5 * (D * math.log(2 * math.pi) + logdet + quad)).mean())


def chain_diagnostics(tr):
    """Is the flattened map chain-like? O(N D) -- never forms the D x D covariance."""
    v = tr.var(0).clamp(min=1e-30)
    s = v.sqrt()
    nb = ((tr[:, :-1] * tr[:, 1:]).mean(0) / (s[:-1] * s[1:])).abs().mean()
    j = torch.randperm(tr.shape[1])                        # random pairs stand in for
    far = ((tr * tr[:, j]).mean(0) / (s * s[j])).abs().mean()   # the mean |corr|
    return {"nb_corr": float(nb), "rand_corr": float(far),
            "eff_rank": float(v.sum() ** 2 / (tr.T @ tr / len(tr)).pow(2).sum())
            if tr.shape[1] <= 8192 else float("nan")}


def fit_lcm(tr, objective, steps, lr, batch_size, learn_a=True, floor=1e-6):
    lcm = LCMCovariance(tr.shape[1], eps=floor)
    lcm.a.requires_grad_(learn_a)
    if objective == "moments":
        lcm.fit_moments(tr)                                 # closed form, one pass
        err = float("nan")
    elif objective == "frobenius":
        err = lcm.fit_frobenius(tr, n_steps=steps, lr=lr, batch_size=batch_size)
    else:                                                   # exact maximum likelihood
        lcm._auto_init_scale((tr**2).mean(0))
        params = [p for p in (lcm.a, lcm.w, lcm.u_raw) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        err = float("nan")
        for _ in range(steps):
            sel = torch.randint(0, len(tr), (min(batch_size, len(tr)),))
            opt.zero_grad()
            loss = lcm.nll(tr[sel])
            loss.backward()
            opt.step()
            err = float(loss)
    return lcm, err


# ------------------------------------------------------------------------ run
def run_layer(tr, te, args, seed):
    # A ReLU map has near-dead units (at D=65536, 7 of them have train variance below
    # 1e-6 x the median and one is exactly zero). Their held-out NLL is set purely by
    # whatever floor a family happens to clamp its diagonal at -- 1e-6 for LCM's eps
    # against 1e-10 for a plain diagonal -- which at large D swamps the comparison
    # entirely (the diagonal scored +35.8 nats/dim before this was fixed). So: drop
    # exactly-degenerate dimensions, and give every family the *same* explicit floor.
    v = tr.var(0)
    med = v.median()
    keep = v > args.dead_rel * med
    n_dropped = int((~keep).sum())
    if n_dropped:
        tr, te = tr[:, keep], te[:, keep]
    floor = float(args.var_floor * med)
    D = tr.shape[1]
    rows, diag = [], chain_diagnostics(tr)
    print(f"    (dropped {n_dropped} degenerate dims, shared variance floor {floor:.3e})",
          flush=True)

    def add(name, n_par, nll, secs, extra=""):
        rows.append({"family": name, "D": D, "params": n_par, "params_per_dim": n_par / D,
                     "nll": nll, "nll_per_dim": nll / D, "fit_s": secs, "note": extra,
                     "n_dropped": n_dropped, "var_floor": floor, **diag})
        print(f"    {name:16} {n_par/D:5.1f}D  NLL {nll:14.1f}  "
              f"({nll/D:+8.4f}/dim)  {secs:7.1f}s", flush=True)

    t0 = time.perf_counter()
    add("diag", D, diag_nll(tr, te, floor), time.perf_counter() - t0)

    for name, obj, learn_a, npar in (("lcm_moments", "moments", True, 3 * D),
                                     ("lcm_frob", "frobenius", True, 3 * D),
                                     ("lcm_ml", "likelihood", True, 3 * D),
                                     ("lcm_frob_fixed_a", "frobenius", False, 2 * D)):
        # Exact-likelihood fitting costs ~6 s/step at D=8192 (the Kalman scan runs
        # under autograd), so it is capped by dimension rather than silently dominating.
        if obj == "likelihood" and D > args.ml_max_d:
            print(f"    {'lcm_ml':16} skipped: D > --ml-max-d ({args.ml_max_d})", flush=True)
            continue
        steps = args.ml_steps if obj == "likelihood" else args.steps
        t0 = time.perf_counter()
        lcm, err = fit_lcm(tr, obj, steps, args.lr, args.batch_size, learn_a, floor)
        with torch.no_grad():
            add(name, npar, float(lcm.nll(te)), time.perf_counter() - t0, f"fit={err:.4g}")
        del lcm

    for rank in args.ranks:
        t0 = time.perf_counter()
        W, psi = fa_fit(tr, rank, n_iter=args.em_iters, seed=seed, floor=floor)
        add(f"fa_r{rank}", (rank + 1) * D, fa_nll(W, psi, te), time.perf_counter() - t0)
        del W, psi
    return rows


def run_seed(args, seed):
    torch.manual_seed(seed)
    X, y = load_cifar(args.data_root, args.dataset, args.n, seed)
    n_cls = int(y.max()) + 1
    model = train(WideCNN(args.channels, n_cls), X, y, args.epochs, seed=seed)
    rows = []
    for layer, c in enumerate(args.channels):
        F = model.layer_features(X, layer)
        n_tr = int(0.8 * len(F))
        mu = F[:n_tr].mean(0)
        tr, te = F[:n_tr] - mu, F[n_tr:] - mu
        del F
        print(f"  seed {seed}  layer{layer + 1}  D={tr.shape[1]}  "
              f"({c} channels x 32 x 32), n_train={n_tr}", flush=True)
        for r in run_layer(tr, te, args, seed):
            rows.append({"seed": seed, "layer": f"layer{layer + 1}", "channels": c,
                         "n_train": n_tr, **r})
        del tr, te
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="/home/tom/data")
    p.add_argument("--channels", default="16,64,192",
                   help="stride-1 channel widths; D = channels x 32 x 32 per layer")
    p.add_argument("--n", type=int, default=3000, help="images used")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=300, help="optimiser steps per LCM fit")
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--em-iters", type=int, default=40)
    p.add_argument("--var-floor", type=float, default=1e-4,
                   help="shared diagonal floor, relative to the median feature variance")
    p.add_argument("--dead-rel", type=float, default=1e-6,
                   help="drop dimensions whose train variance is below this x the median")
    p.add_argument("--ml-steps", type=int, default=100, help="steps for the exact-likelihood arm")
    p.add_argument("--ml-max-d", type=int, default=65536, help="skip that arm above this D")
    p.add_argument("--ranks", default="2,3", help="factor-analysis ranks; r=2 matches LCM's 3D")
    p.add_argument("--threads", type=int, default=0, help="0 leaves torch's default")
    p.add_argument("--tag", default="exp25_paper_scale")
    args = p.parse_args()
    args.channels = [int(c) for c in args.channels.split(",")]
    args.ranks = [int(r) for r in args.ranks.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    rows = [r for s in range(args.seeds) for r in run_seed(args, s)]
    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag, config={**vars(args), "elapsed_s": time.perf_counter() - t0})

    print(f"\n  {args.dataset}, {args.n} images, {args.seeds} seeds, "
          f"stride-1 widths {args.channels} -> D = {[c * 1024 for c in args.channels]}")
    print("  held-out Gaussian NLL of the centred features (lower is better), mean over seeds\n")
    chains = ["lcm_moments", "lcm_frob", "lcm_ml", "lcm_frob_fixed_a"]
    # Group by layer alone: D varies by a few dimensions between seeds, because the number
    # of degenerate ReLU units dropped is seed-dependent, so (layer, D) would split seeds.
    piv = df.pivot_table(index="layer", columns="family", values="nll", aggfunc="mean")
    sem = df.pivot_table(index="layer", columns="family", values="nll", aggfunc="sem")
    dims = df.groupby("layer").D.mean()
    fam = [f for f in ["diag", *chains, *(f"fa_r{r}" for r in args.ranks)] if f in piv.columns]
    print(f"  {'layer':8}{'D':>8}" + "".join(f"{f:>19}" for f in fam))
    for key in piv.index:
        row = piv.loc[key]
        best = row[fam].min()
        print(f"  {key:8}{int(dims[key]):8d}" + "".join(
            f"{row[f]:>17.1f}{'*' if row[f] == best else ' '} " for f in fam))
    if len(df.seed.unique()) > 1:
        print("\n  standard error over seeds (same layout):")
        for key in sem.index:
            print(f"  {key:8}{int(dims[key]):8d}" + "".join(
                f"{sem.loc[key, f]:>18.2f} " for f in fam))

    print("\n  VERDICT per layer (does a chain beat a matched-budget factor analyser?)")
    for key in piv.index:
        row = piv.loc[key]
        present = [f for f in chains if f in piv.columns]
        who = min(present, key=lambda f: row[f])
        fa2 = row["fa_r2"] if "fa_r2" in piv.columns else float("nan")
        s2 = sem.loc[key, "fa_r2"] if "fa_r2" in sem.columns else float("nan")
        margin = f" ({abs(row[who] - fa2) / s2:.1f} sem)" if s2 == s2 and s2 > 0 else ""
        verdict = "CHAIN WINS" if row[who] < fa2 else "FACTOR ANALYSIS WINS at equal budget"
        print(f"    {key:8} D={int(dims[key]):7d}  best chain {row[who]:12.1f} ({who})   "
              f"fa_r2 (3D, matched) {fa2:12.1f}   -> {verdict}{margin}")
    print(f"\n  chain-likeness of the flattened maps (neighbour |corr| vs random-pair |corr|):")
    for key, grp in df.groupby("layer"):
        print(f"    {key:8} D={int(dims[key]):7d}  neighbour {grp.nb_corr.mean():.4f}  "
              f"random pair {grp.rand_corr.mean():.4f}  "
              f"ratio {grp.nb_corr.mean()/max(grp.rand_corr.mean(), 1e-12):6.2f}x")
    print(f"\n  total {time.perf_counter() - t0:.1f}s")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
