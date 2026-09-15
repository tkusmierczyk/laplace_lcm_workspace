"""Library-backed posterior for the regime-B GLM experiments.

Experiments 7–11 and 13–16 were written against the closed-form Gaussian-GLM
posterior (`GGN = X^T X / sigma^2`, `Cov = inv(GGN + lambda I)`) rather than the
`laplace-torch` `Laplace(...)` object. That is the *exact* posterior for a Gaussian
GLM, and it was verified to match the library to 1e-8 (full) / 1e-16 (diag) — but
it means those results describe a parallel implementation rather than the shipped
`LCMLaplace` class.

This adapter removes that gap. It builds the same posterior through the library and
exposes the three quantities the experiments actually consume, so converting an
experiment is a two-line edit:

    la  = fit_glm(X, y, "lcm", prior, sigma, lcm_steps=700, lcm_lr=5e-2)
    mu, var   = marginal(la, Xt, sigma)      # per-point predictive
    mu, Cov   = joint(la, Xt, sigma)         # joint predictive over the batch

Everything here routes through library calls — `Laplace(...)`, `.fit()`,
`backend.jacobians()`, `functional_variance()`, `functional_covariance()` — so the
numbers describe the shipped code path.

**What cannot be converted, and why** (stated so the boundary is explicit):

* The Exp 13/14/15 competitor families (`chol_band*`, `diag_rank*`, the
  LCM+low-rank hybrid) have no counterpart in `laplace-torch`. They are new
  structured families defined for those experiments, so their arms must stay
  hand-rolled. Their `diag`/`lcm`/`full` arms can use this adapter.
* Exp 7 at `d >= 5000` relies on the `n x n` dual/Woodbury form for an exact `full`
  reference; the library's `FullLaplace` is `O(d^2)` and would OOM.
* Exp 9's greedy acquisition loop re-solves the posterior inside an inner loop,
  where re-running `.fit()` per candidate is prohibitively slow.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from laplace import Laplace


def map_weights(X: torch.Tensor, y: torch.Tensor, prior: float, sigma: float) -> torch.Tensor:
    """Ridge MAP — the exact Laplace expansion point for a Gaussian GLM."""
    d = X.shape[1]
    GGN = X.T @ X / sigma**2
    return torch.linalg.solve(GGN + prior * torch.eye(d, dtype=X.dtype), X.T @ y / sigma**2)


def fit_glm(
    X: torch.Tensor,
    y: torch.Tensor,
    structure: str,
    prior: float,
    sigma: float,
    batch_size: int = 256,
    w_map: torch.Tensor | None = None,
    **kw,
):
    """Fit a `laplace-torch` posterior for the linear model `f(x) = w^T x`.

    `structure` is any `hessian_structure` the library accepts — "diag", "full",
    "kron", "lcm". Extra kwargs (e.g. `lcm_steps`, `lcm_lr`) pass through.
    """
    d = X.shape[1]
    model = nn.Linear(d, 1, bias=False).to(X.dtype)
    with torch.no_grad():
        model.weight.copy_((map_weights(X, y, prior, sigma) if w_map is None else w_map).unsqueeze(0))
    ds = torch.utils.data.TensorDataset(X, y.unsqueeze(-1) if y.ndim == 1 else y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    la = Laplace(model, "regression", subset_of_weights="all",
                 hessian_structure=structure, **kw)
    la.fit(dl)
    la.prior_precision = torch.as_tensor(prior, dtype=X.dtype)
    la.sigma_noise = torch.as_tensor(sigma, dtype=X.dtype)
    return la


def _jacobians(la, Xt: torch.Tensor) -> torch.Tensor:
    """Library Jacobians `(batch, outputs, params)` at the MAP."""
    Js, _ = la.backend.jacobians(Xt)
    return Js


@torch.no_grad()
def marginal(la, Xt: torch.Tensor, sigma: float):
    """Per-point predictive mean and total variance, via the library's GLM predictive."""
    f_mu, f_var = la(Xt, pred_type="glm")
    f_mu = f_mu.squeeze(-1)
    f_var = f_var.reshape(f_mu.shape[0], -1)[:, 0]
    return f_mu, f_var + sigma**2


@torch.no_grad()
def joint(la, Xt: torch.Tensor, sigma: float):
    """Joint predictive mean and covariance over a batch of test points.

    Uses `functional_covariance`, the library's own joint GLM predictive
    (`Js P^-1 Js^T`) — the quantity intended for batched/joint prediction.
    """
    f_mu, _ = la(Xt, pred_type="glm")
    f_mu = f_mu.squeeze(-1)
    Cov = la.functional_covariance(_jacobians(la, Xt))
    return f_mu, Cov + sigma**2 * torch.eye(Cov.shape[0], dtype=Cov.dtype, device=Cov.device)
