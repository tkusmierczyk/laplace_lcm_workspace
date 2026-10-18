"""Exp 12 — Cost and complexity: LCM vs KFAC (and Diag / Full).

Every earlier experiment compared LCM and KFAC on *quality* only. This one asks
what each costs, and whether they are even applicable to the same problems.

Three parts:

  (a) STORAGE — floats actually retained by each fitted posterior, across layer
      aspect ratios. KFAC stores ``in^2 + out^2`` per layer, LCM stores ``3*in*out``,
      so which is cheaper depends on the layer's shape, not its size.

  (b) TIME — wall-clock to *construct* the posterior and to evaluate one
      log-determinant. KFAC is a closed-form single pass; LCM needs an iterative
      (Adam) fit, which is the cost asymmetry the project never measured.

  (c) APPLICABILITY — for a scalar-output GLM (the setting of Exp 7-11, where LCM
      wins), KFAC's Kronecker factorisation degenerates: with ``out = 1`` the
      factor ``G`` is 1x1, so ``A (x) G`` is just the dense ``d x d`` input
      covariance. We check numerically that KFAC *equals* full Laplace there, i.e.
      offers no compression and is not an available cheap baseline.

Writes results/exp12_cost_storage.csv, exp12_cost_time.csv, exp12_kfac_glm.csv.
"""

from __future__ import annotations

import os
import time
import warnings

import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from laplace import Laplace  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS, exist_ok=True)
torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def kron_storage(la) -> int:
    """Floats retained by a fitted KFAC posterior (eigenvectors + eigenvalues)."""
    tot = 0
    for evecs in la.H.eigenvectors:
        tot += sum(f.numel() for f in evecs)
    for evals in la.H.eigenvalues:
        tot += sum(f.numel() for f in evals)
    return tot


def _loader(X, y, bs=128):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=bs)


def time_logdet(la, reps=20) -> float:
    la.log_det_posterior_precision  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        la.log_det_posterior_precision
    return (time.perf_counter() - t0) / reps


# --------------------------------------------------------------------------- #
# (a) storage vs layer aspect ratio
# --------------------------------------------------------------------------- #
def part_a() -> pd.DataFrame:
    print("\n=== (a) storage vs layer aspect ratio ===")
    rows = []
    shapes = [(64, 64), (128, 64), (168, 64), (256, 64), (512, 64), (84, 10), (2048, 10)]
    for nin, nout in shapes:
        model = nn.Sequential(nn.Linear(nin, nout))
        X, y = torch.randn(256, nin), torch.randint(0, nout, (256,))
        la = Laplace(model, "classification", hessian_structure="kron", subset_of_weights="all")
        la.fit(_loader(X, y))
        P = la.n_params
        kf = kron_storage(la)
        rows.append(
            dict(
                n_in=nin,
                n_out=nout,
                aspect_ratio=nin / nout,
                P=P,
                store_diag=P,
                store_lcm=3 * P,
                store_kfac=kf,
                store_full=P * P,
                kfac_over_P=kf / P,
                cheaper="LCM" if kf > 3 * P else "KFAC",
            )
        )
        print(
            f"  {nin:>5}x{nout:<5} r={nin/nout:6.2f}  P={P:>8,}  "
            f"KFAC={kf:>12,} ({kf/P:7.2f}xP)  LCM={3*P:>10,} (3.00xP)  -> {rows[-1]['cheaper']}"
        )
    print("  theory: LCM cheaper iff in^2+out^2 > 3*in*out, i.e. aspect ratio > (3+sqrt5)/2 = 2.618")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# (b) construction and log-det time, on a TRAINED model
# --------------------------------------------------------------------------- #
def part_b() -> pd.DataFrame:
    print("\n=== (b) construction / log-det time (trained model, last layer) ===")
    N, n_in, H, C = 3000, 40, 84, 10
    X = torch.randn(N, n_in)
    y = (X @ torch.randn(n_in, C) + 0.5 * torch.randn(N, C)).argmax(1)
    model = nn.Sequential(nn.Linear(n_in, H), nn.Tanh(), nn.Linear(H, C))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        nn.functional.cross_entropy(model(X), y).backward()
        opt.step()
    acc = (model(X).argmax(1) == y).float().mean().item()
    print(f"  trained MAP accuracy {acc:.3f}; last layer {H}->{C}")

    dl = _loader(X, y)
    rows = []
    for struct in ["diag", "kron", "lcm", "full"]:
        t0 = time.perf_counter()
        la = Laplace(model, "classification", hessian_structure=struct, subset_of_weights="last_layer")
        la.fit(dl)
        t_fit = time.perf_counter() - t0
        t_ld = time_logdet(la)
        rows.append(
            dict(
                method=struct,
                P=la.n_params,
                fit_ms=t_fit * 1e3,
                logdet_ms=t_ld * 1e3,
                closed_form=struct != "lcm",
                fit_residual=getattr(la, "lcm_fit_error_", float("nan")),
            )
        )
        print(f"  {struct:5}  fit {t_fit*1e3:9.1f} ms   log-det {t_ld*1e3:8.3f} ms")
    df = pd.DataFrame(rows).set_index("method")
    print(f"  --> LCM fit / KFAC fit         = {df.loc['lcm','fit_ms']/df.loc['kron','fit_ms']:.1f}x")
    print(f"  --> LCM log-det / KFAC log-det = {df.loc['lcm','logdet_ms']/df.loc['kron','logdet_ms']:.1f}x")
    return df.reset_index()


# --------------------------------------------------------------------------- #
# (c) does KFAC apply at all in LCM's niche?
# --------------------------------------------------------------------------- #
def part_c() -> pd.DataFrame:
    print("\n=== (c) KFAC on a scalar-output GLM (the Exp 7-11 setting) ===")
    rows = []
    for d, N in [(100, 400), (200, 600), (400, 900)]:
        model = nn.Linear(d, 1, bias=False)
        X = torch.randn(N, d)
        y = X @ torch.randn(d, 1) * 0.1 + 0.1 * torch.randn(N, 1)
        dl = _loader(X, y, bs=100)
        la_f = Laplace(model, "regression", hessian_structure="full", subset_of_weights="all")
        la_f.fit(dl)
        la_k = Laplace(model, "regression", hessian_structure="kron", subset_of_weights="all")
        la_k.fit(dl)
        Pf = la_f.posterior_precision.double()
        Pk = la_k.posterior_precision.to_matrix().double()
        rel = ((Pf - Pk).norm() / Pf.norm()).item()
        kf = kron_storage(la_k)
        rows.append(
            dict(
                d=d,
                n=N,
                rel_frob_kfac_vs_full=rel,
                store_kfac=kf,
                store_full=d * d,
                store_lcm=3 * d,
                kfac_over_full=kf / (d * d),
                logdet_full=la_f.log_det_posterior_precision.item(),
                logdet_kfac=la_k.log_det_posterior_precision.item(),
            )
        )
        print(
            f"  d={d:<4} KFAC vs Full rel-Frobenius = {rel:.2e}   "
            f"KFAC stores {kf:,} vs Full {d*d:,} ({kf/(d*d):.3f}x)   LCM {3*d:,}"
        )
    print("  --> for out=1 the Kronecker factorisation is an identity, not an approximation:")
    print("      KFAC IS full Laplace, so it offers no compression and is not a cheap baseline here.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    a, b, c = part_a(), part_b(), part_c()
    a.to_csv(os.path.join(RESULTS, "exp12_cost_storage.csv"), index=False)
    b.to_csv(os.path.join(RESULTS, "exp12_cost_time.csv"), index=False)
    c.to_csv(os.path.join(RESULTS, "exp12_kfac_glm.csv"), index=False)
    print(f"\nsaved {RESULTS}/exp12_cost_storage.csv, exp12_cost_time.csv, exp12_kfac_glm.csv")
