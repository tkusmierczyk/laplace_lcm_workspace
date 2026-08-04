"""Tests for the Laplace-Covariance-Model (LCM) structured covariance and the
`LCMLaplace` / `LCMLLLaplace` classes that expose it in the Laplace library."""
from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

from laplace import (
    DiagLaplace,
    DiagLLLaplace,
    FullLaplace,
    FullLLLaplace,
    Laplace,
    LCMCovariance,
    LCMLaplace,
    LCMLLLaplace,
)
from laplace.lcm import kernel_matvec, logdet_from_params, logdet_parallel

DT = torch.float64


@pytest.fixture(autouse=True)
def _seed_and_dtype():
    torch.manual_seed(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(DT)
    yield
    torch.set_default_dtype(old)


def _random_lcm(C=40):
    lcm = LCMCovariance(C, dtype=DT)
    with torch.no_grad():
        lcm.a.copy_(torch.sort(torch.randn(C) * 1.5)[0])
        lcm.w.copy_(torch.randn(C) * 0.7)
        lcm.u_raw.copy_(torch.randn(C) * 0.5)
    return lcm


# --------------------------------------------------------------- LCM core module
def test_lcm_matrix_is_spd():
    A = _random_lcm().dense()
    assert torch.allclose(A, A.T, atol=1e-10)
    assert (torch.linalg.eigvalsh(A) > 0).all()


def test_kernel_matvec_matches_dense():
    lcm = _random_lcm()
    x = torch.randn(5, lcm.dim)
    assert torch.allclose(kernel_matvec(lcm.a, x), x @ lcm.kernel().T, atol=1e-9)
    assert torch.allclose(lcm.matvec(x), x @ lcm.dense().T, atol=1e-9)


def test_kalman_logdet_and_quad_match_dense():
    lcm = _random_lcm()
    A = lcm.dense()
    assert torch.allclose(lcm.logdet(), torch.logdet(A), atol=1e-7)
    y = torch.randn(7, lcm.dim)
    quad = torch.einsum("bi,ij,bj->b", y, torch.linalg.inv(A), y)
    assert torch.allclose(lcm.quad_inv(y), quad, atol=1e-6)


def test_nll_matches_torch_mvn():
    lcm = _random_lcm()
    mu = torch.randn(lcm.dim)
    x = torch.randn(11, lcm.dim)
    mvn = MultivariateNormal(mu, covariance_matrix=lcm.dense())
    assert torch.allclose(lcm.nll(x, mu), -mvn.log_prob(x).mean(), atol=1e-6)


def test_sample_covariance_matches_A():
    lcm = _random_lcm(30)
    cov = torch.cov(lcm.sample(200000).T)
    assert (cov - lcm.dense()).norm() / lcm.dense().norm() < 0.05


def test_scaled_params_logdet_matches_dense():
    """Log-det of the posterior precision h*A + diag(prior) via the fast path."""
    lcm = _random_lcm()
    A = lcm.dense()
    h, prior = 3.7, torch.rand(lcm.dim) + 0.2
    a, w, d = lcm.scaled_params(h, prior)
    ref = torch.logdet(h * A + torch.diag(prior))
    assert torch.allclose(logdet_from_params(a, w, d), ref, atol=1e-6)


def test_fit_dense_recovers_exponential_target():
    C = 60
    coord = torch.linspace(-2, 2, C)
    K = torch.exp(-torch.abs(coord[:, None] - coord[None, :]))
    w = 0.5 + torch.rand(C)
    d = 0.1 + 0.2 * torch.rand(C)
    H = torch.diag(d) + w[:, None] * K * w[None, :]
    lcm = LCMCovariance(C, dtype=DT)
    rel = lcm.fit_dense(H, n_steps=1500, lr=5e-2)
    assert rel < 0.02
    # ... and beats a diagonal fit on the same correlated target
    err_diag = (torch.diag(torch.diag(H)) - H).norm() / H.norm()
    assert (lcm.dense() - H).norm() / H.norm() < err_diag


def test_auto_init_prevents_divergence_on_mis_scaled_target():
    """A target whose scale is far from the fixed default init (init_w=0.1) must
    still fit well with auto_init (the default) -- without it, Adam can diverge
    rather than merely converge slowly, silently producing a much worse fit."""
    C = 80
    coord = torch.linspace(-2, 2, C)
    K = torch.exp(-torch.abs(coord[:, None] - coord[None, :]))
    big = 5000.0  # far from the fixed default init scale
    H = big * (torch.diag(0.1 + 0.2 * torch.rand(C)) + K)

    lcm_auto = LCMCovariance(C, dtype=DT)
    rel_auto = lcm_auto.fit_dense(H, n_steps=300, lr=5e-2, auto_init=True)
    lcm_off = LCMCovariance(C, dtype=DT)
    rel_off = lcm_off.fit_dense(H, n_steps=300, lr=5e-2, auto_init=False)

    assert rel_auto < 0.05
    assert rel_auto < rel_off  # auto_init must not be worse than the naive default


def test_matrix_free_fit_matches_dense_fit():
    C, N = 50, 4000
    coord = torch.linspace(-2, 2, C)
    K = torch.exp(-torch.abs(coord[:, None] - coord[None, :]))
    w = 0.5 + torch.rand(C)
    H = torch.diag(0.1 + 0.2 * torch.rand(C)) + w[:, None] * K * w[None, :]
    V = torch.randn(N, C) @ torch.linalg.cholesky(H).T  # rows ~ N(0, H)
    a = LCMCovariance(C, dtype=DT)
    a.fit_dense((V.T @ V) / N, n_steps=1200, lr=5e-2)
    b = LCMCovariance(C, dtype=DT)
    b.fit_frobenius(V, n_steps=1200, lr=5e-2)
    assert (a.dense() - b.dense()).norm() / a.dense().norm() < 0.06


def test_parallel_logdet_matches_sequential_and_dense():
    """Associative-scan log-det == sequential Kalman == dense, and stays
    differentiable (for marginal-likelihood tuning)."""
    for C in [17, 128, 1000]:
        lcm = _random_lcm(C)
        a, w, d = lcm.a, lcm.w, lcm.d
        par = logdet_parallel(a, w, d)
        assert torch.allclose(par, logdet_from_params(a, w, d), atol=1e-6)
        if C <= 128:
            assert torch.allclose(par, torch.logdet(lcm.dense()), atol=1e-6)
    prior = torch.tensor(0.7, requires_grad=True)
    lcm = _random_lcm(20)
    logdet_parallel(lcm.a.detach(), lcm.w.detach(),
                    lcm.d.detach() + prior).backward()
    assert prior.grad is not None and torch.isfinite(prior.grad)


def test_logdet_is_differentiable():
    lcm = _random_lcm(20)
    prior = torch.tensor(0.7, requires_grad=True)
    a, w, d = lcm.scaled_params(torch.tensor(1.0), prior * torch.ones(lcm.dim))
    logdet_from_params(a, w, d).backward()
    assert prior.grad is not None and torch.isfinite(prior.grad)


# ------------------------------------------------------------- library integration
def test_factory_registers_lcm():
    assert isinstance(
        Laplace(nn.Linear(3, 1), "regression", subset_of_weights="all",
                hessian_structure="lcm"), LCMLaplace)
    assert isinstance(
        Laplace(nn.Linear(3, 1), "regression", subset_of_weights="last_layer",
                hessian_structure="lcm"), LCMLLLaplace)


def _reg_problem():
    N, D = 200, 4
    X = torch.randn(N, D)
    y = torch.tanh(X @ torch.randn(D, 1)) + 0.1 * torch.randn(N, 1)
    loader = DataLoader(TensorDataset(X, y), batch_size=50)
    model = nn.Sequential(nn.Linear(D, 12), nn.Tanh(), nn.Linear(12, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(300):
        for xb, yb in loader:
            opt.zero_grad(); ((model(xb) - yb) ** 2).mean().backward(); opt.step()
    return model, loader, D


def _kl_to_full(cov_a, cov_f):
    d = cov_f.shape[0]
    return 0.5 * (torch.trace(torch.linalg.inv(cov_f) @ cov_a) - d
                  + torch.logdet(cov_f) - torch.logdet(cov_a))


def test_lcmlaplace_all_weights_pipeline():
    model, loader, D = _reg_problem()

    def make(hs, **kw):
        la = Laplace(model, "regression", subset_of_weights="all",
                     hessian_structure=hs, **kw)
        la.fit(loader)
        return la

    full, diag = make("full"), make("diag")
    lcm = make("lcm", lcm_steps=600, lcm_lr=3e-2)

    # fast Kalman log-det matches the dense log-det of the structured precision
    assert torch.allclose(lcm.log_det_posterior_precision,
                          lcm.posterior_precision.logdet(), atol=1e-5)
    # marginal likelihood is finite
    assert torch.isfinite(lcm.log_marginal_likelihood())
    # structured curvature approximates the full posterior better than diagonal
    kl_lcm = _kl_to_full(lcm.posterior_covariance, full.posterior_covariance)
    kl_diag = _kl_to_full(torch.diag(diag.posterior_variance),
                          full.posterior_covariance)
    assert kl_lcm <= kl_diag + 1e-6
    # predictive + sampling run end-to-end
    f_mu, f_var = lcm(torch.randn(8, D), pred_type="glm")
    assert torch.isfinite(f_mu).all() and torch.isfinite(f_var).all()
    assert lcm.sample(16).shape == (16, lcm.n_params)


def test_lcmlllaplace_classification_pipeline():
    Nc, Dc, K = 400, 8, 3
    B = torch.randn(Dc, Dc)
    L = torch.linalg.cholesky(B @ B.T + 0.1 * torch.eye(Dc))
    X = torch.randn(Nc, Dc) @ L.T
    y = (X @ torch.randn(Dc, K) + 0.5 * torch.randn(Nc, K)).argmax(1)
    loader = DataLoader(TensorDataset(X, y), batch_size=64)
    net = nn.Sequential(nn.Linear(Dc, 24), nn.Tanh(), nn.Linear(24, 16),
                        nn.Tanh(), nn.Linear(16, K))
    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(250):
        for xb, yb in loader:
            opt.zero_grad(); lossf(net(xb), yb).backward(); opt.step()

    def make(hs, **kw):
        la = Laplace(net, "classification", subset_of_weights="last_layer",
                     hessian_structure=hs, **kw)
        la.fit(loader)
        return la

    full, diag = make("full"), make("diag")
    lcm = make("lcm", lcm_steps=600, lcm_lr=3e-2)

    assert torch.allclose(lcm.log_det_posterior_precision,
                          lcm.posterior_precision.logdet(), atol=1e-4)
    kl_lcm = _kl_to_full(lcm.posterior_covariance, full.posterior_covariance)
    kl_diag = _kl_to_full(torch.diag(diag.posterior_variance),
                          full.posterior_covariance)
    assert kl_lcm < kl_diag
    # valid probabilistic predictive
    probs = lcm(X[:16], pred_type="glm", link_approx="probit")
    assert torch.allclose(probs.sum(1), torch.ones(16), atol=1e-4)
    # marginal-likelihood based prior optimisation does not decrease the marglik
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        before = lcm.log_marginal_likelihood().item()
        lcm.optimize_prior_precision(method="marglik", n_steps=50)
        after = lcm.log_marginal_likelihood().item()
    assert after >= before - 1e-3
    # sampler produces the correct covariance spectrum (convention-robust)
    ev_emp = torch.linalg.eigvalsh(torch.cov(lcm.sample(200000).T)).sort()[0]
    ev_post = torch.linalg.eigvalsh(lcm.posterior_covariance).sort()[0]
    assert (ev_emp - ev_post).norm() / ev_post.norm() < 0.03
