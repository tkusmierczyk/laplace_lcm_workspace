"""Laplace Covariance Model (LCM): a structured, near-linear-cost Gaussian covariance.

This module implements the structured covariance representation of

    Krukowski et al., "Stop Marginalizing My Dreams: Model Inversion via Laplace
    Kernel for Continual Learning" (arXiv:2605.11804),

re-purposed here as a *posterior-precision* / *covariance* model for the Laplace
approximation of Bayesian neural networks.

For a ``dim``-dimensional variable, each coordinate ``i`` gets a scalar latent
position ``a_i``.  Pairwise correlations follow a Laplace (exponential /
Matern-1/2 / Ornstein-Uhlenbeck) kernel ``K(a)_ij = exp(-|a_i - a_j|)``, scaled
by ``w`` and with an independent diagonal ``d``::

    A = diag(d) + diag(w) K(a) diag(w),      d = softplus(u) + eps.

Only ``3 * dim`` parameters ``(a, w, u)`` are stored.  The 1-D ordering makes the
kernel a Gauss-Markov (AR(1)) process, so ``A`` is the marginal covariance of a
1-D linear-Gaussian state-space model.  This yields exact, near-linear
primitives:

* ``kernel_matvec`` / ``matvec`` -- ``K(a) x`` and ``A x`` in ``O(dim)`` via
  prefix/suffix sums (no dense matrix), the workhorse of the Frobenius fit.
* ``logdet`` -- ``log det A`` in ``O(dim)`` via the Kalman-filter innovation
  variances (the only fast primitive the Laplace marginal likelihood needs).
* ``sample`` -- draw ``x ~ N(0, A)`` in ``O(dim)`` via the AR(1) recursion.
* ``fit_dense`` / ``fit_frobenius`` -- fit ``(a, w, u)`` to a dense target matrix
  or, matrix-free, to a stream of vectors ``v_i`` (their empirical second moment
  ``(1/N) sum_i v_i v_i^T``), the latter being the exact objective of Appendix A.

The dense helpers (``dense``, dense log-det, etc.) are exact and used to validate
the fast paths in the tests; they are ``O(dim^2)``/``O(dim^3)`` and only meant for
the moderate-dimensional regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LCMCovariance", "kernel_matvec", "kalman_stats", "logdet_from_params"]


def kalman_stats(
    a: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    y: torch.Tensor | None = None,
):
    """Sequential Kalman filter for ``A = diag(d) + diag(w) K(a) diag(w)``.

    Treats ``A`` as the covariance of ``x = diag(w) z + eta`` with a latent AR(1)
    chain ``z`` (stationary unit variance) and independent noise ``eta ~ N(0, diag(d))``.

    Returns the innovation variances ``S`` (shape ``(dim,)``, in ascending-``a``
    order) and, if ``y`` is given (already mean-subtracted, shape ``(..., dim)``),
    the whitened innovations ``v`` with ``v_k ~ N(0, S_k)`` independent. Then
    ``log det A = sum_k log S_k`` and ``y^T A^{-1} y = sum_k v_k^2 / S_k``.
    """
    dim = a.shape[-1]
    a_s, idx = torch.sort(a)
    w_s, d_s = w[idx], d[idx]
    dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-12)
    phi = torch.exp(-dt)  # (dim-1,)
    q = (1.0 - phi**2).clamp(min=0.0)

    want_v = y is not None
    if want_v:
        y_s = y.index_select(-1, idx)
        m_filt = torch.zeros(y_s.shape[:-1], dtype=w_s.dtype, device=w_s.device)
        v_list = []

    # Accumulate S in a Python list + stack (autograd-friendly; the log-det must be
    # differentiable w.r.t. w/d so the marginal likelihood can be optimised).
    S_list = []
    P_pred = torch.ones((), dtype=w_s.dtype, device=w_s.device)  # z_0 stationary var
    P_filt = None
    for k in range(dim):
        if k > 0:
            P_pred = phi[k - 1] ** 2 * P_filt + q[k - 1]
        S_k = w_s[k] ** 2 * P_pred + d_s[k]
        S_list.append(S_k)
        gain = P_pred * w_s[k] / S_k  # Kalman gain
        if want_v:
            m_pred = phi[k - 1] * m_filt if k > 0 else torch.zeros_like(m_filt)
            v_k = y_s[..., k] - w_s[k] * m_pred
            v_list.append(v_k)
            m_filt = m_pred + gain * v_k
        P_filt = (1.0 - gain * w_s[k]) * P_pred

    S = torch.stack(S_list, dim=-1)
    if want_v:
        return S, torch.stack(v_list, dim=-1)
    return S, None


def logdet_from_params(a: torch.Tensor, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``log det (diag(d) + diag(w) K(a) diag(w))`` in ``O(dim)`` (sequential Kalman)."""
    S, _ = kalman_stats(a, w, d, None)
    return torch.log(S).sum()


def _parallel_prefix_2x2(M: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix products of a sequence of 2x2 matrices via a Hillis-Steele
    associative scan: returns ``C`` with ``C[k] = M[k] @ M[k-1] @ ... @ M[0]`` in
    ``O(log n)`` sequential steps (vs. ``O(n)`` for a Python loop). Each partial
    product is rescaled to avoid overflow; the rescaling cancels in the Mobius
    ratio it is used for."""
    n = M.shape[-3]
    res = M
    step = 1
    while step < n:
        left = res[..., :-step, :, :]
        right = res[..., step:, :, :]
        comb = right @ left
        norm = comb.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-30)
        comb = comb / norm
        res = torch.cat([res[..., :step, :, :], comb], dim=-3)
        step *= 2
    return res


def logdet_parallel(a: torch.Tensor, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``log det (diag(d) + diag(w) K(a) diag(w))`` via an associative 2x2-matrix
    scan of the Kalman variance (Riccati) recursion -- ``O(dim log dim)`` *depth*,
    fully vectorised (no Python loop over dimensions). Numerically matches
    :func:`logdet_from_params`; preferable for large ``dim``."""
    a_s, idx = torch.sort(a)
    w_s, d_s = w[idx], d[idx]
    C = a_s.shape[-1]
    dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-12)
    phi2 = torch.exp(-2.0 * dt)
    q = (1.0 - phi2).clamp(min=1e-12)

    M = torch.zeros(C, 2, 2, dtype=w_s.dtype, device=w_s.device)
    M[0, 0, 1] = d_s[0]
    M[0, 1, 1] = w_s[0] ** 2 + d_s[0]
    w2 = w_s[1:] ** 2
    M[1:, 0, 0] = d_s[1:] * phi2
    M[1:, 0, 1] = d_s[1:] * q
    M[1:, 1, 0] = w2 * phi2
    M[1:, 1, 1] = w2 * q + d_s[1:]

    Ccum = _parallel_prefix_2x2(M)
    denom = (Ccum[:, 1, 0] + Ccum[:, 1, 1]).clamp(min=1e-30)
    P_filt = ((Ccum[:, 0, 0] + Ccum[:, 0, 1]) / denom).clamp(min=0.0)

    P_prev = torch.ones(C, dtype=w_s.dtype, device=w_s.device)
    P_prev[1:] = P_filt[:-1]
    phi2v = torch.ones(C, dtype=w_s.dtype, device=w_s.device)
    phi2v[1:] = phi2
    qv = torch.zeros(C, dtype=w_s.dtype, device=w_s.device)
    qv[1:] = q
    P_pred = (phi2v * P_prev + qv).clamp(min=1e-12)
    S = (w_s**2 * P_pred + d_s).clamp(min=1e-12)
    return torch.log(S).sum()


def kernel_matvec(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute ``y = K(a) x`` with ``K(a)_ij = exp(-|a_i - a_j|)`` in ``O(dim)``.

    Uses the prefix/suffix-sum decomposition (Appendix B of the paper): after
    sorting the coordinates, ``y_i = L_i + x_i + R_i`` where ``L_i`` (resp.
    ``R_i``) accumulates the exponentially-decaying contributions from
    coordinates below (resp. above) ``i``.  Never materialises the dense kernel.

    Parameters
    ----------
    a : torch.Tensor
        Latent coordinates, shape ``(dim,)``.
    x : torch.Tensor
        Vector(s) to multiply, shape ``(..., dim)``.

    Returns
    -------
    y : torch.Tensor
        ``K(a) x``, same shape as ``x``.
    """
    dim = a.shape[-1]
    a_sorted, idx = torch.sort(a)
    inv_idx = torch.argsort(idx)
    x_sorted = x.index_select(-1, idx)

    # Stable exclusive prefix: L_i = exp(-(a_i - a_0)) * sum_{j<i} exp(a_j - a_0) x_j.
    # Offsetting by the boundary coordinate keeps every exponent <= 0 for the
    # decaying factor; the growing factor is bounded by exp(coordinate range).
    a0 = a_sorted[..., :1]
    aL = a_sorted[..., -1:]
    ex_lo = torch.exp(a_sorted - a0)  # >= 1 but <= exp(range)
    em_lo = torch.exp(-(a_sorted - a0))  # <= 1
    prefix_incl = torch.cumsum(ex_lo * x_sorted, dim=-1)
    prefix_excl = prefix_incl - ex_lo * x_sorted
    left = em_lo * prefix_excl

    # Stable suffix: R_i = exp(a_i - aL) * sum_{j>i} exp(aL - a_j) x_j.
    ex_hi = torch.exp(a_sorted - aL)  # <= 1
    em_hi = torch.exp(aL - a_sorted)  # >= 1 but <= exp(range)
    # Exclusive suffix sum via reversed cumsum.
    rev = torch.flip(em_hi * x_sorted, dims=(-1,))
    suffix_incl = torch.flip(torch.cumsum(rev, dim=-1), dims=(-1,))
    suffix_excl = suffix_incl - em_hi * x_sorted
    right = ex_hi * suffix_excl

    y_sorted = left + x_sorted + right
    return y_sorted.index_select(-1, inv_idx)


class LCMCovariance(nn.Module):
    """Structured PSD matrix ``A = diag(d) + diag(w) K(a) diag(w)``.

    Parameters
    ----------
    dim : int
        Dimensionality ``C`` of the modelled variable.
    eps : float, default=1e-6
        Floor added to the softplus diagonal for numerical stability.
    init_a_scale : float, default=1.0
        Latent coordinates are initialised to ``linspace(-1, 1, dim) * init_a_scale``.
    dtype : torch.dtype, default=torch.float64
        Working precision; float64 is recommended for the log-det / Kalman path.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        init_a_scale: float = 1.0,
        init_w: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.u_raw = nn.Parameter(torch.zeros(dim, dtype=dtype))
        self.w = nn.Parameter(torch.full((dim,), float(init_w), dtype=dtype))
        if dim > 1:
            a0 = torch.linspace(-1.0, 1.0, dim, dtype=dtype) * init_a_scale
        else:
            a0 = torch.zeros(dim, dtype=dtype)
        self.a = nn.Parameter(a0)

    # ------------------------------------------------------------------ params
    @property
    def d(self) -> torch.Tensor:
        """Diagonal (independent-noise) term ``d = softplus(u) + eps``."""
        return F.softplus(self.u_raw) + self.eps

    def _sorted(self):
        """Return quantities in ascending-coordinate order used by the SSM."""
        a_s, idx = torch.sort(self.a)
        return a_s, idx, self.w[idx], self.d[idx]

    # ------------------------------------------------------------------- dense
    def kernel(self) -> torch.Tensor:
        """Dense Laplace kernel ``K(a)`` (``O(dim^2)``; validation only)."""
        return torch.exp(-torch.abs(self.a[:, None] - self.a[None, :]))

    def dense(self) -> torch.Tensor:
        """Dense covariance ``A`` (``O(dim^2)``; validation / moderate dim)."""
        w = self.w
        return torch.diag(self.d) + w[:, None] * self.kernel() * w[None, :]

    # ---------------------------------------------------------------- fast ops
    def kernel_matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``K(a) x`` in ``O(dim)`` (see module-level :func:`kernel_matvec`)."""
        return kernel_matvec(self.a, x)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """``A x = d * x + w * (K(a) (w * x))`` in ``O(dim)``."""
        return self.d * x + self.w * self.kernel_matvec(self.w * x)

    def _kalman(self, y: torch.Tensor | None = None):
        """Kalman filter over the 1-D state-space form of ``A`` (see :func:`kalman_stats`)."""
        return kalman_stats(self.a, self.w, self.d, y)

    def logdet(self) -> torch.Tensor:
        """``log det A`` in ``O(dim)`` via Kalman innovation variances."""
        S, _ = self._kalman(None)
        return torch.log(S).sum()

    def logdet_dense(self) -> torch.Tensor:
        """``log det A`` via dense Cholesky (validation only)."""
        return torch.logdet(self.dense())

    def quad_inv(self, y: torch.Tensor) -> torch.Tensor:
        """``y^T A^{-1} y`` in ``O(dim)`` (``y`` already mean-subtracted)."""
        S, v = self._kalman(y)
        return (v**2 / S).sum(dim=-1)

    def nll(self, x: torch.Tensor, mu: torch.Tensor | None = None) -> torch.Tensor:
        """Mean exact Gaussian negative log-likelihood of rows of ``x``."""
        y = x if mu is None else x - mu
        S, v = self._kalman(y)
        ll = -0.5 * (
            self.dim * torch.log(torch.tensor(2 * torch.pi, dtype=S.dtype))
            + torch.log(S).sum()
            + (v**2 / S).sum(dim=-1)
        )
        return -ll.mean()

    # ------------------------------------------------------------------ sample
    @torch.no_grad()
    def sample(
        self, n_samples: int, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Draw ``x ~ N(0, A)`` (shape ``(n_samples, dim)``) via the AR(1) SSM."""
        a_s, idx, w_s, d_s = self._sorted()
        dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-12)
        phi = torch.exp(-dt)
        q = (1.0 - phi**2).clamp(min=0.0)
        eps_z = torch.randn(
            n_samples, self.dim, dtype=w_s.dtype, device=w_s.device, generator=generator
        )
        z = torch.empty_like(eps_z)
        z[:, 0] = eps_z[:, 0]  # stationary unit variance
        for k in range(1, self.dim):
            z[:, k] = phi[k - 1] * z[:, k - 1] + torch.sqrt(q[k - 1]) * eps_z[:, k]
        eta = torch.randn(
            n_samples, self.dim, dtype=w_s.dtype, device=w_s.device, generator=generator
        )
        x_sorted = w_s * z + torch.sqrt(d_s) * eta
        x = torch.empty_like(x_sorted)
        x.index_copy_(-1, idx, x_sorted)
        return x

    # --------------------------------------------------------------- fitting
    def fit_dense(
        self,
        target: torch.Tensor,
        n_steps: int = 500,
        lr: float = 1e-2,
        tol: float = 1e-9,
        verbose: bool = False,
    ) -> float:
        """Fit ``(a, w, u)`` to a dense PSD target by minimising ``||A - target||_F^2``.

        Returns the final relative Frobenius error ``||A - target||_F / ||target||_F``.
        """
        target = target.to(self.a.dtype)
        opt = torch.optim.Adam([self.a, self.w, self.u_raw], lr=lr)
        target_norm = target.norm().clamp(min=1e-12)
        prev = None
        for step in range(n_steps):
            opt.zero_grad()
            loss = ((self.dense() - target) ** 2).sum()
            loss.backward()
            opt.step()
            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"  [fit_dense] step {step:4d}  loss {loss.item():.6e}")
            if prev is not None and abs(prev - loss.item()) < tol * (prev + 1e-12):
                break
            prev = loss.item()
        with torch.no_grad():
            rel = (self.dense() - target).norm() / target_norm
        return rel.item()

    def fit_frobenius(
        self,
        vectors: torch.Tensor,
        n_steps: int = 500,
        lr: float = 1e-2,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> float:
        """Matrix-free fit to the empirical second moment of ``vectors``.

        Minimises ``||A - S_hat||_F^2`` with ``S_hat = (1/N) sum_i v_i v_i^T`` using
        the Appendix-A decomposition -- only quadratic forms ``v^T A v`` are ever
        evaluated, so the ``N x C`` matrix of vectors is touched but the ``C x C``
        second-moment matrix is *never formed*.

        Parameters
        ----------
        vectors : torch.Tensor
            ``(N, dim)`` matrix of (already centred) vectors ``v_i``.
        batch_size : int, optional
            Mini-batch of vectors per step (stochastic fit); defaults to all.

        Returns
        -------
        loss : float
            Final value of the (constant-dropped) objective.
        """
        vectors = vectors.to(self.a.dtype)
        N = vectors.shape[0]
        opt = torch.optim.Adam([self.a, self.w, self.u_raw], lr=lr)
        last = float("nan")
        for step in range(n_steps):
            if batch_size is not None and batch_size < N:
                sel = torch.randint(0, N, (batch_size,), device=vectors.device)
                V = vectors[sel]
            else:
                V = vectors
            opt.zero_grad()
            w, d = self.w, self.d
            # model term  <A, A> = ||d||^2 + 2 d.w^2 + (w^2)^T K(2a) (w^2)
            w2 = w**2
            model_term = (
                (d**2).sum()
                + 2.0 * (d * w2).sum()
                + (w2 * kernel_matvec(2.0 * self.a, w2)).sum()
            )
            # data term  (2/N) sum_i [ d.v_i^2 + (w*v_i)^T K(a) (w*v_i) ]
            wv = w * V  # (b, dim)
            data_term = 2.0 * (
                (d * (V**2).mean(dim=0)).sum()
                + (wv * kernel_matvec(self.a, wv)).sum(dim=-1).mean()
            )
            loss = model_term - data_term
            loss.backward()
            opt.step()
            last = loss.item()
            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"  [fit_frobenius] step {step:4d}  loss {last:.6e}")
        return last

    # ------------------------------------------------------------------ export
    def scaled_params(self, h_factor, prior_diag: torch.Tensor):
        """Return LCM params of ``h_factor * A + diag(prior_diag)``.

        Scaling by ``h_factor`` and adding a diagonal keep the matrix in the LCM
        family: same ``a``, ``w -> sqrt(h_factor) w``, ``d -> h_factor d + prior``.
        Used to evaluate the posterior-precision log-det without re-fitting. The
        fitted ``(a, w, d)`` are detached (they are not being optimised at this
        point) but ``prior_diag`` and ``h_factor`` keep their gradients so the
        marginal likelihood stays differentiable w.r.t. the prior / noise.
        """
        h = h_factor if torch.is_tensor(h_factor) else torch.as_tensor(
            h_factor, dtype=self.d.dtype, device=self.d.device
        )
        a = self.a.detach()
        w = self.w.detach() * torch.sqrt(h)
        d = h * self.d.detach() + prior_diag.to(self.d.dtype)
        return a, w, d
