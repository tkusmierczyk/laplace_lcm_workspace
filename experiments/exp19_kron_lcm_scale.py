"""Exp 19 — where Kron-LCM can be built and KFAC cannot.

Exp 18 showed Kron-LCM reaches 90.3% of KFAC's shifted-marginal quality on 80x less
storage at i=256. The storage ratio grows as i/3, so the interesting question is the
feasibility frontier: at what layer width does the *dense* Kronecker factor stop fitting,
and does the LCM route still work there?

This matters because of an honest caveat in the library: `Laplace(...,
hessian_structure="kron_lcm")` goes through the `Kron` pipeline, which accumulates the
dense `i x i` factor before the LCM is fitted to it -- so that entry point is limited by
`i^2` exactly like KFAC, and only the *storage* of the fitted posterior shrinks. The
closed-form estimator itself does not need the matrix: `LCMCovariance.fit_moments(X)`
reads three bands straight off the activations in one pass, `O(N i)` time and `O(i)`
memory. This experiment measures both paths and records where each one dies.

Quality at scale is checked without ever forming `i x i`: the data are AR(1) with known
rho, so the fitted chain must recover `phi = rho` on every edge, which is an `O(i)` check.

Writes results/exp19_scale.csv.
"""

from __future__ import annotations

import gc
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

from laplace.lcm import LCMCovariance, kalman_solve, logdet_from_params  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)

WIDTHS = [512, 2048, 8192, 32768, 65536]
N, RHO, C = 1000, 0.9, 10
GRAM_BUDGET_GIB = 2.0          # refuse to form i x i above this
MACHINE_RAM_GIB = 31.0


def ar1_rows(n, i, rho, g):
    """AR(1) along the feature axis, generated in blocks to keep the loop count down."""
    e = torch.randn(n, i, generator=g) * math.sqrt(1 - rho**2)
    X = torch.empty(n, i)
    X[:, 0] = torch.randn(n, generator=g)
    for j in range(1, i):
        X[:, j] = rho * X[:, j - 1] + e[:, j]
    return X


def run(i):
    g = torch.Generator().manual_seed(0)
    row = {"i": i, "N": N}
    t0 = time.perf_counter()
    X = ar1_rows(N, i, RHO, g)
    row["t_data_s"] = time.perf_counter() - t0

    # ---- streaming path: closed-form fit straight off the activations
    lcm = LCMCovariance(i)
    t0 = time.perf_counter()
    lcm.fit_moments(X)
    row["t_fit_moments_s"] = time.perf_counter() - t0

    # recovered chain decay must equal rho on every edge -- an O(i) quality check
    a_sorted = torch.sort(lcm.a)[0]
    phi_hat = torch.exp(-(a_sorted[1:] - a_sorted[:-1]))
    row["phi_mean"] = float(phi_hat.mean())
    row["phi_err_max"] = float((phi_hat - RHO).abs().max())
    row["w2_over_s0"] = float((lcm.w**2 / (X * X).mean(0)).median())

    # ---- the O(i) posterior primitives at this width (one reduced block)
    prior = torch.full((i,), 1.0)
    a, w, d = lcm.scaled_params(float(N), prior)
    t0 = time.perf_counter()
    row["logdet"] = float(logdet_from_params(a, w, d))
    row["t_logdet_s"] = time.perf_counter() - t0
    B = torch.randn(C, i, generator=g)
    t0 = time.perf_counter()
    sol = kalman_solve(a, w, d, B)
    row["t_solve_s"] = time.perf_counter() - t0
    row["solve_finite"] = bool(torch.isfinite(sol).all())

    # ---- dense factor: what KFAC (and the library's kron_lcm entry point) needs
    gib = i * i * 8 / 2**30
    row["dense_factor_gib"] = gib
    row["lcm_store_gib"] = 3 * i * 8 / 2**30
    row["full_posterior_gib"] = (C * i) ** 2 * 8 / 2**30
    if gib <= GRAM_BUDGET_GIB:
        t0 = time.perf_counter()
        A = X.T @ X / N
        row["t_gram_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        torch.linalg.eigh(A)
        row["t_eigh_s"] = time.perf_counter() - t0
        del A
        gc.collect()
    else:
        row["t_gram_s"] = row["t_eigh_s"] = float("nan")
    del X
    gc.collect()
    return row


def main():
    rows = []
    for i in WIDTHS:
        r = run(i)
        rows.append(r)
        print(f"  i={i:6d}  fit_moments {r['t_fit_moments_s']:6.3f}s  "
              f"logdet {r['t_logdet_s']:6.3f}s  solve {r['t_solve_s']:6.3f}s  "
              f"dense factor {r['dense_factor_gib']:8.3f} GiB  "
              f"gram {r['t_gram_s'] if r['t_gram_s'] == r['t_gram_s'] else float('nan'):.3f}s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/exp19_scale.csv", index=False)

    print(f"\n  AR(1) activations, N={N}, rho={RHO}, C={C} outputs. "
          f"Machine: {MACHINE_RAM_GIB:.0f} GiB RAM, 7.6 GiB VRAM.")
    print(f"\n  {'i':>7}{'fit_moments':>13}{'logdet':>9}{'solve':>9}"
          f"{'LCM store':>12}{'dense factor':>14}{'exact full':>13}{'feasible?':>26}")
    for _, r in df.iterrows():
        dense_ok = r["dense_factor_gib"] < MACHINE_RAM_GIB / 4
        full_ok = r["full_posterior_gib"] < MACHINE_RAM_GIB / 4
        verdict = ("all three" if full_ok else
                   "kron_lcm + KFAC" if dense_ok else "kron_lcm only")
        print(f"  {int(r['i']):7d}{r['t_fit_moments_s']:12.3f}s{r['t_logdet_s']:8.3f}s"
              f"{r['t_solve_s']:8.3f}s{r['lcm_store_gib']*1024:10.2f} MiB"
              f"{r['dense_factor_gib']:11.2f} GiB{r['full_posterior_gib']:10.1f} GiB"
              f"{verdict:>26}")
    print(f"\n  chain recovery (truth rho={RHO}): " + "  ".join(
        f"i={int(r['i'])}: phi={r['phi_mean']:.4f} (max err {r['phi_err_max']:.1e})"
        for _, r in df.iterrows()))
    print(f"  all solves finite: {bool(df['solve_finite'].all())}")
    print(f"\nsaved {RESULTS}/exp19_scale.csv")


if __name__ == "__main__":
    main()
