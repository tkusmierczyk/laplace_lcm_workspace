"""Exp 25b — verify the moment fit's large-D numbers, since they are extreme.

`lcm_moments` reports held-out feature NLLs far below every other family (and negative at
D=65536, which is legitimate for a differential entropy but is exactly the kind of number
that has turned out to be a bug twice in this project). Two independent checks:

  D=16384   form the dense 16384 x 16384 covariance (2.1 GiB) and recompute the log-det and
            held-out quadratic form by Cholesky. Positive-definiteness, log-det and quad all
            have to agree with the O(D) Kalman path.
  D=65536   dense is impossible (32 GiB), so check what can be checked in O(D): all Kalman
            innovation variances positive (equivalent to positive-definiteness), the
            log-det from two independent implementations, and the residual of the O(D)
            solve against the O(D) matvec.

Writes results/exp25_verify.csv.
"""

from __future__ import annotations

import argparse
import math
import time
import warnings

import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

import common  # noqa: E402
import exp25_paper_scale_baselines as E  # noqa: E402
from laplace.lcm import kalman_stats, logdet_from_params, logdet_parallel  # noqa: E402


def features(args, layer, seed=0):
    torch.manual_seed(seed)
    X, y = E.load_cifar(args.data_root, args.dataset, args.n, seed)
    model = E.train(E.WideCNN(args.channels, int(y.max()) + 1), X, y, args.epochs, seed=seed)
    F = model.layer_features(X, layer)
    n_tr = int(0.8 * len(F))
    mu = F[:n_tr].mean(0)
    return F[:n_tr] - mu, F[n_tr:] - mu


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--data-root", default="/home/tom/data")
    p.add_argument("--channels", default="4,16,64")
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--dense-max-d", type=int, default=16384)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tag", default="exp25_verify")
    args = p.parse_args()
    args.channels = [int(c) for c in args.channels.split(",")]
    torch.set_num_threads(args.threads)
    rows = []

    for layer in range(len(args.channels)):
        tr, te = features(args, layer)
        v = tr.var(0)
        med = v.median()
        keep = v > 1e-6 * med
        tr, te = tr[:, keep], te[:, keep]
        floor = float(1e-4 * med)
        D = tr.shape[1]
        lcm, _ = E.fit_lcm(tr, "moments", 0, 0.0, 256, True, floor)
        with torch.no_grad():
            S, vv = kalman_stats(lcm.a, lcm.w, lcm.d, te)
            ld_fast = float(torch.log(S).sum())
            quad_fast = float((vv**2 / S).sum(-1).mean())
            row = {
                "layer": layer + 1, "D": D, "n_test": len(te),
                "min_innovation_var": float(S.min()),      # > 0  <=>  positive definite
                "logdet_kalman": ld_fast,
                "logdet_parallel": float(logdet_parallel(lcm.a, lcm.w, lcm.d)),
                "logdet_sequential": float(logdet_from_params(lcm.a, lcm.w, lcm.d)),
                "quad_kalman": quad_fast,
                "quad_per_dim": quad_fast / D,
                "nll": 0.5 * (D * math.log(2 * math.pi) + ld_fast + quad_fast),
                "coord_span": float(lcm.a.max() - lcm.a.min()),
                "d_min": float(lcm.d.min()), "d_median": float(lcm.d.median()),
            }
            # O(D) solve check: does A (A^-1 y) return y?
            y = te[:4]
            xs = lcm.solve(y)
            back = torch.stack([lcm.matvec(xs[i]) for i in range(len(xs))])
            row["solve_rel_residual"] = float((back - y).norm() / y.norm())

            if D <= args.dense_max_d:
                t0 = time.perf_counter()
                A = lcm.dense()
                L, info = torch.linalg.cholesky_ex(A)
                row["dense_pd"] = int(info) == 0
                if int(info) == 0:
                    row["logdet_dense"] = float(2 * torch.log(torch.diagonal(L)).sum())
                    sol = torch.cholesky_solve(te.T, L)
                    row["quad_dense"] = float((te.T * sol).sum(0).mean())
                    row["logdet_err"] = row["logdet_kalman"] - row["logdet_dense"]
                    row["quad_err"] = row["quad_kalman"] - row["quad_dense"]
                row["dense_s"] = time.perf_counter() - t0
                del A, L
            rows.append(row)
        print(f"  layer{layer + 1} D={D:6d}  min innov var {row['min_innovation_var']:.3e}  "
              f"logdet {row['logdet_kalman']:+14.3f}  quad/dim {row['quad_per_dim']:.4f}  "
              f"NLL {row['nll']:+12.1f}  solve resid {row['solve_rel_residual']:.2e}"
              + (f"  | dense logdet err {row.get('logdet_err', float('nan')):+.3e}"
                 f"  quad err {row.get('quad_err', float('nan')):+.3e}"
                 if "logdet_dense" in row else "  | dense skipped"), flush=True)
        del tr, te, lcm

    df = pd.DataFrame(rows)
    path = common.save_results(df, args.tag, config=vars(args))
    print("\n  All innovation variances positive (positive definite):",
          bool((df.min_innovation_var > 0).all()))
    print("  Log-det agrees across the two O(D) implementations to:",
          f"{float((df.logdet_kalman - df.logdet_parallel).abs().max()):.2e}")
    print("  Largest relative solve residual:", f"{float(df.solve_rel_residual.max()):.2e}")
    if "logdet_err" in df:
        print("  Against dense Cholesky where feasible: log-det to",
              f"{float(df.logdet_err.abs().max()):.2e}, quadratic form to",
              f"{float(df.quad_err.abs().max()):.2e}")
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
