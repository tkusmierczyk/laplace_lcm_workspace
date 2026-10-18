"""Tests for the Laplace-Covariance-Model (LCM) structured covariance and the
`LCMLaplace` / `LCMLLLaplace` classes that expose it in the Laplace library."""
from __future__ import annotations

import warnings

import pytest
import math

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
from laplace.lcm import (
    kalman_stats,
    kalman_stats_sequential,
    kernel_matvec,
    logdet_from_params,
    logdet_parallel,
)

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


def test_scan_filter_matches_sequential_reference():
    """The ported two-scan Kalman filter (mirroring the authors' `log_likelihood`)
    must agree with the independent sequential reference implementation and with
    the dense ground truth -- innovation variances AND innovations."""
    import torch.nn.functional as Fn

    for C in [1, 2, 17, 128, 600]:
        a = torch.sort(torch.randn(C, dtype=DT))[0]
        w = torch.randn(C, dtype=DT) * 0.7
        d = Fn.softplus(torch.randn(C, dtype=DT) * 0.5) + 1e-6
        y = torch.randn(3, C, dtype=DT)

        S_par, v_par = kalman_stats(a, w, d, y)
        S_seq, v_seq = kalman_stats_sequential(a, w, d, y)
        assert torch.allclose(S_par, S_seq, atol=1e-10)
        assert torch.allclose(v_par, v_seq, atol=1e-9)

        # against the dense reference
        K = torch.exp(-torch.abs(a[:, None] - a[None, :]))
        A = torch.diag(d) + w[:, None] * K * w[None, :]
        assert torch.allclose(torch.log(S_par).sum(), torch.logdet(A), atol=1e-8)
        quad = torch.einsum("bi,ij,bj->b", y, torch.linalg.inv(A), y)
        assert torch.allclose((v_par**2 / S_par).sum(-1), quad, atol=1e-7)

    # gradients flow through both scans
    a = torch.sort(torch.randn(40, dtype=DT))[0].requires_grad_(True)
    w = (torch.randn(40, dtype=DT) * 0.5).requires_grad_(True)
    u = torch.zeros(40, dtype=DT, requires_grad=True)
    S, v = kalman_stats(a, w, Fn.softplus(u) + 1e-6, torch.randn(2, 40, dtype=DT))
    (torch.log(S).sum() + (v**2 / S).sum()).backward()
    assert all(torch.isfinite(t.grad).all() for t in (a, w, u))


def test_kernel_matvec_stable_for_long_chains():
    """Regression: the prefix-sum kernel matvec must not overflow for large
    coordinate ranges. The previous offset-based factorisation produced inf/NaN
    once the range exceeded ~709 (float64) even though every kernel entry is <=1."""
    # exact vs dense at a range that would overflow the naive factorisation
    d = 400
    a = torch.arange(d, dtype=DT) * 3.0  # range 1197 >> 709
    x = torch.randn(2, d, dtype=DT)
    K = torch.exp(-torch.abs(a[:, None] - a[None, :]))
    assert torch.allclose(kernel_matvec(a, x), x @ K.T, atol=1e-9)
    # stays finite for long chains
    for dim, spacing in [(5000, 0.105), (20000, 0.105)]:
        a_big = torch.arange(dim, dtype=DT) * spacing
        y = kernel_matvec(a_big, torch.randn(dim, dtype=DT))
        assert torch.isfinite(y).all()
    # and remains differentiable
    a_g = (torch.arange(500, dtype=DT) * 2.0).requires_grad_(True)
    kernel_matvec(a_g, torch.randn(500, dtype=DT)).sum().backward()
    assert torch.isfinite(a_g.grad).all()


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


def test_fit_is_scale_equivariant_and_normalisation_helps():
    """The LCM family is positively homogeneous: A*(c T) = c A*(T). So fitting the
    same target at two very different scales must give the same *relative* error,
    and the fitted matrix must carry the target's scale.

    This is a regression test for a real defect: the fit objective is an
    unnormalised sum of squares while Adam takes steps of absolute size ``lr``, so
    on a GGN-scale target ``auto_init`` put ``w`` at ~300 and each step moved it by
    ~2e-4 relative -- the fit barely left its initialisation. Normalising the target
    internally (and rescaling after) fixes it exactly. Unnormalised fitting is what
    made experiment results differ from the library's own LCMLaplace.
    """
    torch.manual_seed(0)
    d, n, rho = 120, 600, 0.9
    e = torch.randn(n, d, dtype=torch.float64)
    X = torch.empty(n, d, dtype=torch.float64)
    X[:, 0] = e[:, 0]
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + (1 - rho**2) ** 0.5 * e[:, j]
    base = X.T @ X / n
    a0 = torch.arange(d, dtype=torch.float64) * -math.log(rho)

    def fit(target, normalize):
        torch.manual_seed(1)
        m = LCMCovariance(d, dtype=torch.float64)
        with torch.no_grad():
            m.a.copy_(a0)
        err = m.fit_dense(target, n_steps=400, lr=5e-2, normalize=normalize)
        return m, err

    # equivariance: same relative error at 1x and 10000x, and the scale is carried
    m1, e1 = fit(base, True)
    m2, e2 = fit(base * 1e4, True)
    assert abs(e1 - e2) < 1e-6, f"not scale-equivariant: {e1} vs {e2}"
    ratio = float(m2.dense().diagonal().mean() / m1.dense().diagonal().mean())
    assert abs(ratio - 1e4) / 1e4 < 1e-6, f"scale not carried: {ratio}"

    # and normalising is what buys it: unnormalised degrades at large scale
    _, e_un = fit(base * 1e4, False)
    assert e2 < e_un, f"normalised fit ({e2}) should beat unnormalised ({e_un})"

    # fit_frobenius agrees with fit_dense on the same implied target
    torch.manual_seed(1)
    mf = LCMCovariance(d, dtype=torch.float64)
    with torch.no_grad():
        mf.a.copy_(a0)
    mf.fit_frobenius(X, n_steps=400, lr=5e-2)
    rel_f = float((mf.dense() - base).norm() / base.norm())
    assert rel_f < e1 + 0.05, f"matrix-free fit {rel_f} far from dense fit {e1}"


def test_lcm_laplace_accepts_known_and_frozen_ordering():
    """`lcm_init_a` / `lcm_learn_a` let LCMLaplace express the project's own
    recommended recipe -- supply an ordering (known or seriated) and refine it.
    Without them the class can only learn `a` from a linspace init, which Exp 10
    showed cannot discover a latent order.
    """
    torch.manual_seed(0)
    d, n, rho = 100, 600, 0.9
    e = torch.randn(n, d, dtype=torch.float64)
    X = torch.empty(n, d, dtype=torch.float64)
    X[:, 0] = e[:, 0]
    for j in range(1, d):
        X[:, j] = rho * X[:, j - 1] + (1 - rho**2) ** 0.5 * e[:, j]
    y = (X @ (torch.randn(d, dtype=torch.float64) * 0.2)
         + 0.5 * torch.randn(n, dtype=torch.float64)).unsqueeze(1)
    model = torch.nn.Linear(d, 1, bias=False).double()
    dl = DataLoader(TensorDataset(X, y), batch_size=200)
    true_a = torch.arange(d, dtype=torch.float64) * -math.log(rho)

    def run(**kw):
        la = Laplace(model, "regression", subset_of_weights="all",
                     hessian_structure="lcm", lcm_steps=400, lcm_lr=5e-2, **kw)
        la.fit(dl)
        return la

    frozen = run(lcm_init_a=true_a, lcm_learn_a=False)
    assert torch.allclose(frozen.lcm.a.detach(), true_a), "frozen `a` must not move"
    assert not frozen.lcm.a.requires_grad

    refined = run(lcm_init_a=true_a, lcm_learn_a=True)
    assert not torch.allclose(refined.lcm.a.detach(), true_a), "`a` should be refined"
    assert refined.lcm_fit_error_ <= frozen.lcm_fit_error_ + 1e-6

    # A wrong order is much worse than the right one.
    wrong = run(lcm_init_a=true_a[torch.randperm(d)], lcm_learn_a=False)
    assert wrong.lcm_fit_error_ > 3 * frozen.lcm_fit_error_

    # Supplying the order only *helps* when the data is not already in it. The
    # library's default init is `linspace(-1, 1, d) * scale`, which is monotone --
    # i.e. it already encodes the natural order, and being learnable it can fix the
    # spacing too. So the default is competitive on natural-order data and the
    # comparison is only meaningful once the features are permuted.
    perm = torch.randperm(d)
    Xp = X[:, perm]
    model_p = torch.nn.Linear(d, 1, bias=False).double()
    dl_p = DataLoader(TensorDataset(Xp, y), batch_size=200)

    def run_p(**kw):
        la = Laplace(model_p, "regression", subset_of_weights="all",
                     hessian_structure="lcm", lcm_steps=400, lcm_lr=5e-2, **kw)
        la.fit(dl_p)
        return la

    # column j of Xp is original column perm[j], so its true coordinate is perm[j]
    coord_p = true_a[perm]
    assert run_p(lcm_init_a=coord_p, lcm_learn_a=False).lcm_fit_error_ < \
        run_p().lcm_fit_error_, "a supplied ordering must beat the default on permuted data"

    with pytest.raises(ValueError, match="lcm_init_a has shape"):
        run(lcm_init_a=torch.zeros(d + 1, dtype=torch.float64))


# ---------------------------------------------------------------------------
# Kron x LCM: the one-factor eigen reduction
# ---------------------------------------------------------------------------
def _lcm_target(dim, seed=0, span=6.0):
    g = torch.Generator().manual_seed(seed)
    a = torch.sort(torch.rand(dim, generator=g, dtype=torch.float64) * span)[0]
    w = torch.rand(dim, generator=g, dtype=torch.float64) + 0.5
    d = torch.rand(dim, generator=g, dtype=torch.float64) + 0.1
    A = torch.diag(d) + w[:, None] * torch.exp(-(a[:, None] - a[None, :]).abs()) * w[None, :]
    return a, w, d, A


@pytest.mark.parametrize("dim", [1, 2, 7, 64, 300])
def test_kalman_solve_matches_dense(dim):
    """The O(dim) solve must agree with a dense solve, for any coordinate order."""
    from laplace.lcm import kalman_solve

    a, w, d, A = _lcm_target(dim)
    Y = torch.randn(3, 4, dim, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    ref = torch.linalg.solve(A, Y.reshape(-1, dim).T).T.reshape(Y.shape)
    assert torch.allclose(kalman_solve(a, w, d, Y), ref, atol=1e-10)

    perm = torch.randperm(dim, generator=torch.Generator().manual_seed(2))
    got = kalman_solve(a[perm], w[perm], d[perm], Y[..., perm])
    assert torch.allclose(got, ref[..., perm], atol=1e-10)


def test_kalman_solve_preserves_legacy_contract():
    """`kalman_stats` must keep returning a 2-tuple unless asked for the internals."""
    a, w, d, _ = _lcm_target(32)
    y = torch.randn(5, 32, dtype=torch.float64)
    assert len(kalman_stats(a, w, d, y)) == 2
    assert len(kalman_stats(a, w, d, y, return_filter=True)) == 6
    assert len(kalman_stats(a, w, d, None, return_filter=True)) == 6


def test_fit_moments_is_exact_on_an_lcm_target():
    """The closed-form three-band estimator recovers an LCM exactly on the interior.

    The two endpoint amplitudes are genuinely not identified by three bands, so error
    is expected there and nowhere else.
    """
    dim = 60
    _, _, _, A = _lcm_target(dim)
    ev, V = torch.linalg.eigh(A)
    X = (V * ev.sqrt()) @ V.T  # X^T X == A, with dim rows, so the target is A / dim
    m = LCMCovariance(dim).fit_moments(X)
    E = m.dense() * dim - A
    interior = slice(1, -1)
    assert E[interior, interior].norm() / A[interior, interior].norm() < 1e-10
    # and the moments-only entry point must agree exactly
    s0, s1, s2 = (X * X).mean(0), (X[:, :-1] * X[:, 1:]).mean(0), (X[:, :-2] * X[:, 2:]).mean(0)
    assert torch.allclose(LCMCovariance(dim).fit_moments(moments=(s0, s1, s2)).dense(), m.dense())


def test_fit_moments_is_order_equivariant():
    """Permuting the columns and declaring the order must give the same matrix back."""
    dim = 40
    _, _, _, A = _lcm_target(dim, seed=3)
    ev, V = torch.linalg.eigh(A)
    X = (V * ev.sqrt()) @ V.T
    perm = torch.randperm(dim, generator=torch.Generator().manual_seed(4))
    m = LCMCovariance(dim).fit_moments(X)
    mp = LCMCovariance(dim).fit_moments(X[:, perm], order=torch.argsort(perm))
    assert torch.allclose(mp.dense(), m.dense()[perm][:, perm], atol=1e-12)


def _kron_setup(dtype=torch.float64):
    torch.manual_seed(0)
    N, I, H, C = 200, 24, 16, 4
    X, y = torch.randn(N, I, dtype=dtype), torch.randint(0, C, (N,))
    model = nn.Sequential(nn.Linear(I, H), nn.Tanh(), nn.Linear(H, C)).to(dtype)
    return DataLoader(TensorDataset(X, y), batch_size=50), model, X


def test_kron_lcm_registers_in_the_factory():
    _, model, _ = _kron_setup()
    for subset in ("all", "last_layer"):
        la = Laplace(model, "classification", subset_of_weights=subset,
                     hessian_structure="kron_lcm")
        assert la._key == (subset, "kron_lcm")


def test_kron_lcm_dense_fallback_reproduces_kron():
    """With no layer structured, the new precision object must be exactly KronLaplace.

    This pins the whole plumbing -- scale, deltas, the rotation, the parameter
    layout -- independently of the LCM approximation itself.
    """
    dl, model, X = _kron_setup()

    def fit(structure, **kw):
        la = Laplace(model, "classification", subset_of_weights="last_layer",
                     hessian_structure=structure, **kw)
        la.fit(dl)
        return la

    kron, lcm = fit("kron"), fit("kron_lcm", lcm_layers=[])
    Js = kron.backend.last_layer_jacobians(X[:5])[0]
    assert Js.abs().max() > 0  # guard: zero Jacobians would make this vacuous
    assert torch.allclose(kron.posterior_precision.logdet(), lcm.posterior_precision.logdet())
    assert torch.allclose(kron.log_marginal_likelihood(), lcm.log_marginal_likelihood())
    assert torch.allclose(kron.functional_variance(Js), lcm.functional_variance(Js))
    assert torch.allclose(kron.functional_covariance(Js), lcm.functional_covariance(Js))
    assert torch.allclose(kron(X[:20]), lcm(X[:20]))


@pytest.mark.parametrize("order", ["natural", "seriate"])
def test_kron_lcm_reduction_matches_its_dense_form(order):
    """The reduction must be exact: every quantity has to agree with the dense
    materialisation of the same structured precision."""
    dl, model, X = _kron_setup()
    la = Laplace(model, "classification", subset_of_weights="last_layer",
                 hessian_structure="kron_lcm", lcm_order=order)
    la.fit(dl)
    P = la.posterior_precision
    M = P.to_matrix()
    assert torch.allclose(M, M.T, atol=1e-12)
    S = torch.linalg.inv(M)
    assert torch.allclose(P.logdet(), torch.logdet(M))

    Js = la.backend.last_layer_jacobians(X[:5])[0]
    assert Js.abs().max() > 0
    assert torch.allclose(la.functional_variance(Js),
                          torch.einsum("ncp,pq,ndq->ncd", Js, S, Js), atol=1e-10)
    Jf = Js.reshape(-1, Js.shape[-1])
    assert torch.allclose(la.functional_covariance(Js), Jf @ S @ Jf.T, atol=1e-10)
    # one structured block and one dense (bias) block, as expected for a Linear layer
    kinds = [b["kind"] for b in P.blocks]
    assert kinds == ["lcm", "dense"]


def test_kron_lcm_sampling_is_perturb_and_solve():
    """No closed-form square root exists, so samples come from perturb-and-solve; their
    covariance must still converge to the posterior covariance."""
    dl, model, _ = _kron_setup()
    la = Laplace(model, "classification", subset_of_weights="last_layer",
                 hessian_structure="kron_lcm")
    la.fit(dl)
    P = la.posterior_precision
    S = torch.linalg.inv(P.to_matrix())
    n = 200_000
    samples = la.sample(n, generator=torch.Generator().manual_seed(0))
    emp = torch.cov((samples - la.mean).T)
    assert emp.shape == S.shape
    # tolerance set by Monte-Carlo error, ~sqrt(2 p / n)
    assert (emp - S).norm() / S.norm() < 5 * (2 * S.shape[0] / n) ** 0.5
    with pytest.raises(NotImplementedError, match="exponent 1 and -1"):
        P.bmm(torch.randn(2, S.shape[0], dtype=torch.float64), exponent=-0.5)


def test_kron_lcm_rejects_damping():
    """Factored damping approximates the prior; the reduction adds it exactly, so
    asking for both is a contradiction and must fail loudly."""
    dl, model, _ = _kron_setup()
    la = Laplace(model, "classification", subset_of_weights="last_layer",
                 hessian_structure="kron_lcm", damping=True)
    with pytest.raises(ValueError, match="damping is not applicable"):
        la.fit(dl)


def test_kron_last_layer_damping_survives_init():
    """Regression test: `KronLaplace.__init__` runs later in the MRO and used to reset
    `damping` to its default, silently discarding `damping=True` passed to the
    last-layer class (whose docstring documents it as supported)."""
    _, model, _ = _kron_setup()
    assert Laplace(model, "classification", subset_of_weights="last_layer",
                   hessian_structure="kron", damping=True).damping is True
