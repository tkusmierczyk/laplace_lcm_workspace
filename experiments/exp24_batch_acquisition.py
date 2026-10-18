"""Exp 24 — off-diagonals as a *decision*: batch active learning.

Exp 23 found this project's first scale-invariant separation, by conditioning on revealed
labels. This asks the operational version of the same question: does posterior *structure*
change which points you choose to label, and does that change downstream error? The metric
is test RMSE, not a likelihood, so no calibration slack can rescue a bad choice.

The criterion is the predictive (V-optimal) one, because that is what matches the metric:
labelling candidate ``x`` reduces the predictive variance at a reference point ``z`` by
``Sigma_zx^2 / (Sigma_xx + sigma^2)``, so the greedy score is the total reduction over a
reference sample, and after each pick the joint covariance gets the exact rank-1 Bayesian
downdate. (The purely scale-free alternative -- noise-free D-optimality -- maximises
parameter-space volume instead, which buys atypical high-norm inputs and loses to random
selection on predictive error. Measured, not assumed: that was this experiment's first
design.)

Four questions, in increasing subtlety:

  1. variance_only vs the rest -- does using off-diagonals *at all* help? This arm takes
     the top-k marginal variances and is the only arm that ignores them.
  2. diag vs full -- note a diagonal *weight* posterior still induces a dense
     *function-space* covariance, so this is not "no correlations" versus "correlations",
     it is wrong ones versus right ones.
  3. lcm vs diag -- does the chain recover the right ones?
  4. diag_tempered -- selection is invariant to the posterior scale in both the
     high-signal and low-signal limits but not in between, so the oracle temperature that
     rescued the diagonal in Exps 17-21 is given its chance here too.

rho = 0 is the negative control. Pools can be built with repeated directions
(``--duplicate``), the canonical setting where naive top-k variance buys the same point
five times.

WHAT WOULD MAKE US DROP THE IDEA. If a tempered diagonal selects as well as the exact
posterior at high rho, off-diagonal structure has no operational value here either, and
the case for LCM rests entirely on Exp 23.

Writes results/exp24_acquisition.csv.
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
from exp23_conditional_prediction import ar1_cov  # noqa: E402

# (label, hessian_structure, posterior scales to try, use off-diagonals)
ARMS = [
    ("full", "full", (1.0,), True),
    ("lcm", "lcm", (1.0,), True),
    ("diag", "diag", (1.0,), True),
    ("diag_tempered", "diag", (0.01, 0.1, 1.0, 10.0, 100.0), True),
    ("variance_only", "full", (1.0,), False),
    ("random", None, (1.0,), False),
]


def make_pool(d, rho, n_init, n_pool, n_ref, n_eval, sigma, prior, duplicate, seed):
    g = torch.Generator().manual_seed(seed)
    L = torch.linalg.cholesky(ar1_cov(d, rho) + 1e-10 * torch.eye(d))
    draw = lambda n: torch.randn(n, d, generator=g) @ L.T          # noqa: E731
    w = torch.randn(d, generator=g) / math.sqrt(prior)
    label = lambda X: X @ w + sigma * torch.randn(len(X), generator=g)  # noqa: E731
    out = {}
    for tag, n in (("init", n_init), ("ref", n_ref), ("eval", n_eval)):
        X = draw(n)
        out[tag] = (X, label(X))
    if duplicate > 1:                                   # repeated directions, jittered
        base = draw(max(1, n_pool // duplicate))
        Xp = base.repeat(duplicate, 1)[:n_pool]
        Xp = Xp + 0.01 * torch.randn(Xp.shape, generator=g)
    else:
        Xp = draw(n_pool)
    out["pool"] = (Xp, label(Xp))
    return out, w


def greedy_voptimal(Sig, n_pool, k, taken, sigma, use_offdiag):
    """Greedy predictive-variance reduction over the reference block.

    ``Sig`` is the joint epistemic covariance over ``[pool; ref]``. Each pick applies the
    exact rank-1 Bayesian downdate, so later picks see the covariance *given* the earlier
    ones -- which is where off-diagonals enter.
    """
    if not use_offdiag:
        # The genuine no-off-diagonals arm: rank by marginal variance once and take the
        # top k. No downdate, so nothing tells it that two picks are the same direction.
        score = torch.diagonal(Sig)[:n_pool].clone()
        score[taken] = -float("inf")
        return torch.topk(score, k).indices.tolist()
    S = Sig.clone()
    chosen: list[int] = []
    for _ in range(k):
        var = torch.diagonal(S)[:n_pool]
        cross = S[:n_pool, n_pool:]                     # Sigma_xz for every candidate
        score = ((cross**2).sum(1) / (var + sigma**2)).clone()
        score[taken + chosen] = -float("inf")
        j = int(torch.argmax(score))
        chosen.append(j)
        piv = S[:, j].clone()
        S = S - torch.outer(piv, piv) / (piv[j] + sigma**2)
    return chosen


def run_seed(args, rho, seed):
    D, _ = make_pool(args.d, rho, args.n_init, args.n_pool, args.n_ref, args.n_eval,
                     args.sigma, args.prior, args.duplicate, seed)
    Xev, yev = D["eval"]
    Xp, yp = D["pool"]
    Xjoint = torch.cat([Xp, D["ref"][0]])
    rows = []
    for label, structure, scales, use_off in ARMS:
        gen = torch.Generator().manual_seed(7000 + seed)
        X, y = D["init"][0].clone(), D["init"][1].clone()
        taken: list[int] = []
        for rnd in range(args.rounds + 1):
            la = common.fit_glm(X, y, structure or "diag", args.prior, args.sigma)
            mu, _ = common.glm_marginal(la, Xev, args.sigma)
            rows.append({"rho": rho, "seed": seed, "selector": label, "round": rnd,
                         "n_train": len(X), "rmse": float(((mu - yev) ** 2).mean().sqrt()),
                         "redundancy": batch_redundancy(Xp[taken]) if taken else float("nan")})
            if rnd == args.rounds:
                break
            if structure is None:                       # random floor
                free = [i for i in range(len(Xp)) if i not in taken]
                pick = [free[int(i)] for i in
                        torch.randperm(len(free), generator=gen)[: args.k]]
            else:
                _, S_tot = common.glm_joint(la, Xjoint, args.sigma)
                Sig = S_tot - args.sigma**2 * torch.eye(len(S_tot), dtype=S_tot.dtype)
                # several scales: an oracle temperature for the selection itself, resolved
                # at the end of the run by keeping the best final RMSE
                pick = greedy_voptimal(scales[0] * Sig, len(Xp), args.k, taken,
                                       args.sigma, use_off)
                if len(scales) > 1:
                    best, pick = None, None
                    for t in scales:
                        cand = greedy_voptimal(t * Sig, len(Xp), args.k, taken,
                                               args.sigma, use_off)
                        Xc = torch.cat([X, Xp[cand]])
                        yc = torch.cat([y, yp[cand]])
                        lc = common.fit_glm(Xc, yc, structure, args.prior, args.sigma)
                        m2, _ = common.glm_marginal(lc, Xev, args.sigma)
                        r = float(((m2 - yev) ** 2).mean().sqrt())
                        if best is None or r < best:
                            best, pick = r, cand
            taken = taken + pick
            X = torch.cat([X, Xp[pick]])
            y = torch.cat([y, yp[pick]])
    return rows


def batch_redundancy(Xsel):
    """Mean |cosine| between the acquired inputs -- how much the batch repeats itself."""
    if len(Xsel) < 2:
        return float("nan")
    Z = Xsel / Xsel.norm(dim=1, keepdim=True).clamp(min=1e-12)
    G = (Z @ Z.T).abs()
    n = len(Z)
    return float((G.sum() - n) / (n * (n - 1)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--n-init", type=int, default=8)
    p.add_argument("--n-pool", type=int, default=300)
    p.add_argument("--n-ref", type=int, default=100, help="reference sample scored against")
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--k", type=int, default=4, help="batch size per round")
    p.add_argument("--duplicate", type=int, default=5, help="repeat each pool direction")
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--prior", type=float, default=1.0)
    p.add_argument("--rhos", default="0.0,0.9,0.99")
    p.add_argument("--seeds", type=int, default=15)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--tag", default="exp24_acquisition")
    args = p.parse_args()
    args.rhos = [float(r) for r in args.rhos.split(",")]
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    rows = [r for rho in args.rhos for s in range(args.seeds) for r in run_seed(args, rho, s)]
    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag,
                               config={**vars(args), "elapsed_s": time.perf_counter() - t0})

    names = [a[0] for a in ARMS]
    print(f"\n  Batch active learning, d={args.d}, start {args.n_init}, "
          f"{args.rounds} rounds x {args.k}, pool {args.n_pool} "
          f"({args.duplicate} copies per direction), {args.seeds} seeds")
    for rho, grp in df.groupby("rho"):
        print(f"\n  rho = {rho}" + ("   (negative control)" if rho == 0 else ""))
        piv = grp.pivot_table(index="selector", columns="round", values="rmse", aggfunc="mean")
        print(f"    {'selector':15}" + "".join(f"{f'n={args.n_init + r * args.k}':>10}"
                                               for r in piv.columns))
        for k in names:
            print(f"    {k:15}" + "".join(f"{piv.loc[k, r]:10.4f}" for r in piv.columns))
        fin = grp[grp["round"] == args.rounds].pivot_table(
            index="seed", columns="selector", values="rmse")
        print(f"    paired vs selecting with the exact posterior at "
              f"n={args.n_init + args.rounds * args.k} (positive = worse):")
        for k in names:
            if k == "full":
                continue
            dd = fin[k] - fin["full"]
            sem = dd.std() / math.sqrt(len(dd))
            sig = abs(dd.mean()) / sem if sem > 0 else float("inf")
            print(f"      {k:15}{dd.mean():+9.4f} +- {sem:.4f}  ({sig:4.1f} sigma)"
                  f"{'  <- separated' if sig > 3 else '  <- within noise'}")
        red = grp[grp["round"] == args.rounds].groupby("selector").redundancy.mean()
        print("    batch redundancy (mean |cosine| between acquired inputs, lower is better): "
              + "  ".join(f"{k} {red[k]:.3f}" for k in names))
    print(f"\n  total {time.perf_counter() - t0:.1f}s\nsaved {path}")


if __name__ == "__main__":
    main()
