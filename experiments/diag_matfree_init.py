"""Diagnose the matrix-free fit gap: exp8 initialised w for the UNNORMALISED GGN,
but fit_frobenius targets Sigma_hat = GGN/N. Test with correctly scaled init,
across a couple of dimensions so we get signal quickly."""
import time
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
from laplace.lcm import LCMCovariance


def ar1(n, d, rho, g):
    eps = torch.randn(n, d, generator=g)
    X = torch.empty(n, d)
    X[:, 0] = eps[:, 0]
    s = (1 - rho**2) ** 0.5
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + s * eps[:, j]
    return X


def make(d, tgt_diag, rho):
    lcm = LCMCovariance(d)
    with torch.no_grad():
        lcm.a.copy_(torch.arange(d) * -np.log(rho))
        lcm.w.copy_(torch.full((d,), float(tgt_diag.sqrt())))
        lcm.u_raw.copy_(torch.full((d,), -2.0))
    return lcm


def run(d, n, rho=0.9, noise=0.5, steps=700, lr=5e-2, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = ar1(n, d, rho, g)
    V = X / noise
    GGN = X.T @ X / noise**2
    Shat = GGN / n

    t0 = time.perf_counter()
    lcm1 = make(d, GGN.diagonal().mean(), rho)
    r_dense = lcm1.fit_dense(GGN, n_steps=steps, lr=lr)
    t_dense = time.perf_counter() - t0

    t0 = time.perf_counter()
    lcm2 = make(d, GGN.diagonal().mean(), rho)  # exp8's (wrong) init scale
    lcm2.fit_frobenius(V, n_steps=steps, lr=lr)
    r_wrong = float((lcm2.dense() * n - GGN).norm() / GGN.norm())
    t_wrong = time.perf_counter() - t0

    t0 = time.perf_counter()
    lcm3 = make(d, Shat.diagonal().mean(), rho)  # correct init scale
    lcm3.fit_frobenius(V, n_steps=steps, lr=lr)
    r_fixed = float((lcm3.dense() * n - GGN).norm() / GGN.norm())
    t_fixed = time.perf_counter() - t0

    print(f"d={d:5d} n={n:5d} steps={steps:5d} | dense={r_dense:.4f} ({t_dense:.1f}s) | "
          f"matfree(wrong-init)={r_wrong:.4f} ({t_wrong:.1f}s) | "
          f"matfree(fixed-init)={r_fixed:.4f} ({t_fixed:.1f}s)", flush=True)
    return dict(d=d, n=n, steps=steps, r_dense=r_dense, r_wrong=r_wrong, r_fixed=r_fixed)


if __name__ == "__main__":
    print("=== warm-up: small d, fast signal ===", flush=True)
    run(d=200, n=2000, steps=500)
    run(d=500, n=3000, steps=700)
    print("\n=== target regime (matches exp8) ===", flush=True)
    run(d=1500, n=3000, steps=700)
    print("\n=== more steps, does matfree catch up? ===", flush=True)
    run(d=500, n=3000, steps=2000)
