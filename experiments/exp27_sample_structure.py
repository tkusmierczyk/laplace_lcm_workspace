"""Exp 27 — what the paper actually uses the covariance for: *generating* feature maps.

REMIX (arXiv:2605.11804) does not use its covariance to predict. It **samples** from it, to
invert a network and manufacture replay data for data-free continual learning. That makes
sampling the paper's real use case, and it is a pure structure probe: correlations are
normalised, so **no temperature can give a diagonal the right spatial structure** -- a
diagonal generates white noise at every scale.

So: fit each family to real CNN feature maps, draw samples, and compare their
autocorrelation profile against the held-out features'. In a flattened `C x H x W` map the
lags have meaning -- lag 1 is the neighbouring pixel in a row, lag `W` the pixel above, lag
`H*W` the same pixel in the next channel -- so the profile says *which* structure a family
reproduces, not merely how much.

HYPOTHESIS. The chain reproduces short-lag spatial correlation that a diagonal cannot, and
that a low-rank model reproduces only in a smeared, long-range form.

WHAT WOULD MAKE US DROP IT. If a rank-2 factor analyser at matched parameter count matches
the chain's lag profile, LCM has no advantage even in the paper's own application.

Writes results/exp27_sample_structure.csv.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common  # noqa: E402
import exp25_paper_scale_baselines as E  # loaders, CNN, FA and LCM fits  # noqa: E402


@torch.no_grad()
def lag_profile(X, lags):
    """Mean correlation at each lag along the flattened index. Scale-free by definition."""
    Z = X - X.mean(0)
    sd = Z.std(0).clamp(min=1e-12)
    Z = Z / sd
    return {f"lag{k}": float((Z[:, :-k] * Z[:, k:]).mean()) for k in lags}


def hilbert_perm(channels, side):
    """Flattened-map indices in Hilbert-curve order, per channel.

    The chain's one failure in this experiment is dimensional: on a row-major `C x H x W`
    map it captures lag 1 (the next pixel in a row) and misses lag W (the pixel above),
    because a 1-D chain can follow only one adjacency direction. A space-filling curve is
    the standard fix -- it keeps 2-D neighbours close in 1-D -- and costs nothing but a
    permutation, so the chain can be given 2-D locality for free.
    """
    order = int(round(math.log2(side)))
    assert 2**order == side, "Hilbert ordering needs a power-of-two side"
    seq = []
    for d in range(side * side):
        x = y = 0
        t, s = d, 1
        while s < side:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:
                if rx == 1:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x, y = x + s * rx, y + s * ry
            t, s = t // 4, s * 2
        seq.append(y * side + x)
    base = torch.tensor(seq)
    return torch.cat([c * side * side + base for c in range(channels)])


@torch.no_grad()
def sample_diag(tr, n, floor, gen):
    var = tr.var(0).clamp(min=floor)
    return torch.randn(n, tr.shape[1], generator=gen) * var.sqrt()


@torch.no_grad()
def sample_fa(W, psi, n, gen):
    z = torch.randn(n, W.shape[1], generator=gen)
    return z @ W.T + torch.randn(n, W.shape[0], generator=gen) * psi.sqrt()


def run_layer(tr, te, args, seed, layer, C, HW):
    v = tr.var(0)
    med = v.median()
    keep = v > args.dead_rel * med
    tr, te = tr[:, keep], te[:, keep]
    floor = float(args.var_floor * med)
    D = tr.shape[1]
    W_img = int(round(HW**0.5))
    lags = [k for k in (1, 2, 4, W_img, HW) if k < D]
    gen = torch.Generator().manual_seed(seed)
    truth = lag_profile(te, lags)
    rows = [{"family": "held-out features", "D": D, "params": 0, **truth, "err": 0.0,
             "fit_s": 0.0}]

    def add(name, n_par, S, secs):
        prof = lag_profile(S, lags)
        err = float(np.mean([abs(prof[k] - truth[k]) for k in truth]))
        rows.append({"family": name, "D": D, "params": n_par, **prof, "err": err,
                     "fit_s": secs})
        print(f"    {name:18}{n_par / D:5.1f}D  " + "  ".join(
            f"{k}={prof[k]:+.3f}" for k in truth) + f"   prof.err {err:.4f}", flush=True)

    print(f"    {'held-out features':18}{'  -':5}   " + "  ".join(
        f"{k}={truth[k]:+.3f}" for k in truth), flush=True)
    t0 = time.perf_counter()
    add("diag", D, sample_diag(tr, args.n_samples, floor, gen), time.perf_counter() - t0)
    for name, obj, learn_a, npar in (("lcm_moments", "moments", True, 3 * D),
                                     ("lcm_frob", "frobenius", True, 3 * D),
                                     ("lcm_frob_fixed_a", "frobenius", False, 2 * D)):
        t0 = time.perf_counter()
        lcm, _ = E.fit_lcm(tr, obj, args.steps, args.lr, args.batch_size, learn_a, floor)
        add(name, npar, lcm.sample(args.n_samples, generator=gen), time.perf_counter() - t0)
        del lcm
    # the chain again, but walking a Hilbert curve so 2-D neighbours are 1-D neighbours
    full_perm = hilbert_perm(C, W_img)
    pos = -torch.ones(len(keep), dtype=torch.long)
    pos[torch.nonzero(keep).squeeze(1)] = torch.arange(D)
    perm = pos[full_perm]
    perm = perm[perm >= 0]
    t0 = time.perf_counter()
    lcm, _ = E.fit_lcm(tr[:, perm], "moments", args.steps, args.lr, args.batch_size,
                       True, floor)
    S = lcm.sample(args.n_samples, generator=gen)
    S_nat = torch.empty_like(S)
    S_nat[:, perm] = S
    add("lcm_moments_hilbert", 3 * D, S_nat, time.perf_counter() - t0)
    del lcm, S, S_nat

    for rank in args.ranks:
        t0 = time.perf_counter()
        Wf, psi = E.fa_fit(tr, rank, n_iter=args.em_iters, seed=seed, floor=floor)
        add(f"fa_r{rank}", (rank + 1) * D,
            sample_fa(Wf, psi, args.n_samples, gen), time.perf_counter() - t0)
    for r in rows:
        r.update(seed=seed, layer=layer, channels=C, HW=HW)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", default="/home/tom/data")
    p.add_argument("--channels", default="4,16")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-samples", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--em-iters", type=int, default=40)
    p.add_argument("--ranks", default="2,3")
    p.add_argument("--var-floor", type=float, default=1e-4)
    p.add_argument("--dead-rel", type=float, default=1e-6)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--tag", default="exp27_sample_structure")
    args = p.parse_args()
    args.channels = [int(c) for c in args.channels.split(",")]
    args.ranks = [int(r) for r in args.ranks.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)

    t0, rows = time.perf_counter(), []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        X, y = E.load_cifar(args.data_root, args.dataset, args.n, seed)
        model = E.train(E.WideCNN(args.channels, int(y.max()) + 1), X, y, args.epochs, seed=seed)
        for layer, c in enumerate(args.channels):
            F = model.layer_features(X, layer)
            n_tr = int(0.8 * len(F))
            mu = F[:n_tr].mean(0)
            tr, te = F[:n_tr] - mu, F[n_tr:] - mu
            del F
            print(f"  seed {seed}  layer{layer + 1}  D={tr.shape[1]}", flush=True)
            rows += run_layer(tr, te, args, seed, f"layer{layer + 1}", c, 1024)
            del tr, te
    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag,
                               config={**vars(args), "elapsed_s": time.perf_counter() - t0})

    lagcols = [c for c in df.columns if c.startswith("lag")]
    print(f"\n  Autocorrelation of SAMPLES vs held-out features "
          f"({args.n_samples} samples, {args.seeds} seeds)")
    print("  lag 1/2/4 = neighbouring pixels, lag 32 = the row above, lag 1024 = next channel\n")
    for layer, grp in df.groupby("layer"):
        print(f"  {layer}  (D={int(grp.D.iloc[0])})")
        print(f"    {'family':18}{'params':>8}" + "".join(f"{c:>10}" for c in lagcols)
              + f"{'prof.err':>10}")
        for fam in grp.family.unique():
            g = grp[grp.family == fam]
            print(f"    {fam:18}{g.params.iloc[0] / g.D.iloc[0]:7.1f}D"
                  + "".join(f"{g[c].mean():10.3f}" for c in lagcols)
                  + f"{g.err.mean():10.4f}")
        best = grp[grp.family != "held-out features"].groupby("family").err.mean().idxmin()
        print(f"    closest to the real features: {best}")
    print(f"\n  total {time.perf_counter() - t0:.1f}s\nsaved {path}")


if __name__ == "__main__":
    main()
