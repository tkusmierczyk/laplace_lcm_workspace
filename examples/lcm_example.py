"""Structured-covariance Laplace with the Laplace Covariance Model (LCM).

Runnable demo of the `hessian_structure="lcm"` addition. It fits Full, Diagonal
and LCM last-layer Laplace approximations to the same trained network and shows
that the LCM posterior is much closer to the exact (full) posterior than the
diagonal one, at a fraction of the parameters, while its log-determinant is
computed in near-linear time.

Run:  python examples/lcm_example.py
"""
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from laplace import Laplace

warnings.filterwarnings("ignore")
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def kl_to_full(cov_approx: torch.Tensor, cov_full: torch.Tensor) -> float:
    d = cov_full.shape[0]
    kl = 0.5 * (
        torch.trace(torch.linalg.inv(cov_full) @ cov_approx)
        - d
        + torch.logdet(cov_full)
        - torch.logdet(cov_approx)
    )
    return kl.item()


# ---- data with correlated features -> dense last-layer curvature -------------
N, D, K = 500, 8, 3
B = torch.randn(D, D)
L = torch.linalg.cholesky(B @ B.T + 0.1 * torch.eye(D))
X = torch.randn(N, D) @ L.T
y = (X @ torch.randn(D, K) + 0.5 * torch.randn(N, K)).argmax(1)
loader = DataLoader(TensorDataset(X, y), batch_size=64)

net = nn.Sequential(
    nn.Linear(D, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, K)
)
opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-3)
loss_fn = nn.CrossEntropyLoss()
for _ in range(300):
    for xb, yb in loader:
        opt.zero_grad()
        loss_fn(net(xb), yb).backward()
        opt.step()


def fit(structure, **kw):
    la = Laplace(
        net, "classification", subset_of_weights="last_layer",
        hessian_structure=structure, **kw,
    )
    la.fit(loader)
    return la


full = fit("full")
diag = fit("diag")
lcm = fit("lcm", lcm_steps=800, lcm_lr=3e-2)

cov_full = full.posterior_covariance
kl_lcm = kl_to_full(lcm.posterior_covariance, cov_full)
kl_diag = kl_to_full(torch.diag(diag.posterior_variance), cov_full)

P = lcm.n_params
print(f"last-layer parameters P = {P}")
print(f"{'structure':<10}{'#cov params':>14}{'KL to full posterior':>24}")
print("-" * 48)
print(f"{'full':<10}{P * (P + 1) // 2:>14}{0.0:>24.4f}")
print(f"{'lcm':<10}{3 * P:>14}{kl_lcm:>24.4f}")
print(f"{'diag':<10}{P:>14}{kl_diag:>24.4f}")
print(f"\nLCM structured fit rel. error to the full GGN: {lcm.lcm_fit_error_:.3f}")
print(f"LCM is {kl_diag / max(kl_lcm, 1e-9):.1f}x closer to the full posterior "
      f"than diagonal (KL).")
print(f"fast Kalman log-det == dense log-det: "
      f"{torch.allclose(lcm.log_det_posterior_precision, lcm.posterior_precision.logdet(), atol=1e-4)}")
