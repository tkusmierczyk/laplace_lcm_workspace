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

PROVENANCE
----------
The core filter is a port of the authors' reference implementation
(``efficient_inversion/resnet34/lcm.py`` in REMIX-Model-Inversion-via-Laplace-Kernel):
both associative scans, the 2x2 prefix product, the parameterisation, and the NLL.
Everything beyond that -- standalone log-determinant, quadratic forms, sampling,
prior handling, the Appendix-A/B fitting and prefix-sum kernels, and the numerical
hardening -- is added here because a *posterior* needs more than a likelihood term.
``LCM_PROVENANCE.md`` (next to this file) maps every function to its origin, lists
the deliberate deviations, and records the equivalence evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LCMCovariance",
    "kernel_matvec",
    "kalman_stats",
    "kalman_stats_sequential",
    "logdet_from_params",
    "logdet_parallel",
]


def _inverse_softplus(x: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
    """Numerically stable inverse of ``softplus``: returns ``y`` with
    ``softplus(y) == x`` for ``x > 0``. Mirrors PyTorch's own stable ``softplus``
    threshold: for large ``x``, ``softplus(y) = log1p(exp(y)) ~= y``, so inverting
    via ``log(expm1(x))`` directly would overflow (``expm1`` of a large number is
    itself already ``inf`` in float64 well before 1000) -- used past the threshold
    the identity is exact to float precision, so we switch to it instead.
    """
    return torch.where(x > threshold, x, torch.log(torch.expm1(x.clamp(min=1e-12))))


def kalman_stats(
    a: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    y: torch.Tensor | None = None,
    return_filter: bool = False,
):
    """Kalman filter for ``A = diag(d) + diag(w) K(a) diag(w)`` via two associative
    scans -- a direct port of the reference implementation.

    PROVENANCE: mirrors ``LCM.log_likelihood`` of the authors' ``lcm.py``
    (REMIX-Model-Inversion-via-Laplace-Kernel, ``efficient_inversion/resnet34/lcm.py``),
    including both of its ``_parallel_matrix_prefix_prod`` scans:
      * scan 1 -- the variance (Riccati) recursion  -> innovation variances ``S``
      * scan 2 -- the mean recursion                -> innovations ``v``
    Deviations from the reference are limited to: float64 throughout (the reference
    casts to float64 only inside the scan), a configurable nugget ``eps`` (the
    reference hardcodes ``1e-5`` in this function while its ``__init__`` default is
    ``1e-6``), and returning ``(S, v)`` instead of a scalar likelihood so the pieces
    can be reused for log-determinants and quadratic forms.

    Treats ``A`` as the covariance of ``x = diag(w) z + eta`` with a latent AR(1)
    chain ``z`` (stationary unit variance) and independent noise ``eta ~ N(0, diag(d))``.

    Returns the innovation variances ``S`` (shape ``(dim,)``, ascending-``a`` order)
    and, if ``y`` is given (already mean-subtracted, shape ``(..., dim)``), the
    innovations ``v`` with ``v_k ~ N(0, S_k)`` independent. Then
    ``log det A = sum_k log S_k`` and ``y^T A^{-1} y = sum_k v_k^2 / S_k``.

    With ``return_filter=True`` the filter internals needed to apply ``L^T`` are also
    returned -- ``(S, v, idx, K_gain, alpha, c)`` -- which is what :func:`kalman_solve`
    uses to turn the quadratic form into an actual solve.
    """
    dim = a.shape[-1]
    a_s, idx = torch.sort(a)
    w_s, d_s = w[idx], d[idx]
    if dim == 1:  # degenerate chain: no transitions
        S = (w_s**2 + d_s).clamp(min=1e-12)
        v = None if y is None else y.index_select(-1, idx)
        if return_filter:
            z = torch.zeros(1, dtype=w_s.dtype, device=w_s.device)
            return S, v, idx, z, z + 1.0, z
        return S, v

    dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-12)
    phi = torch.exp(-dt)
    phi2 = phi**2
    q = (1.0 - phi2).clamp(min=1e-12)

    # --- scan 1: variance / Riccati recursion (reference lines 55-82) ---------
    M = torch.zeros(dim, 2, 2, dtype=w_s.dtype, device=w_s.device)
    M[0, 0, 1] = d_s[0]
    M[0, 1, 1] = w_s[0] ** 2 + d_s[0]
    w2 = w_s[1:] ** 2
    M[1:, 0, 0] = d_s[1:] * phi2
    M[1:, 0, 1] = d_s[1:] * q
    M[1:, 1, 0] = w2 * phi2
    M[1:, 1, 1] = w2 * q + d_s[1:]

    Mc = _parallel_prefix_2x2(M)
    denom = (Mc[:, 1, 0] + Mc[:, 1, 1]).clamp(min=1e-30)
    P_filt = ((Mc[:, 0, 0] + Mc[:, 0, 1]) / denom).clamp(min=0.0)

    P_prev = torch.ones(dim, dtype=w_s.dtype, device=w_s.device)
    P_prev = torch.cat([P_prev[:1], P_filt[:-1]])
    phi2v = torch.cat([torch.ones(1, dtype=w_s.dtype, device=w_s.device), phi2])
    qv = torch.cat([torch.zeros(1, dtype=w_s.dtype, device=w_s.device), q])
    P_pred = (phi2v * P_prev + qv).clamp(min=1e-12)
    S = (w_s**2 * P_pred + d_s).clamp(min=1e-12)
    if y is None and not return_filter:
        return S, None

    # --- scan 2: mean recursion (reference lines 84-107) ----------------------
    # m_k = alpha_k m_{k-1} + beta_k  is affine, so it composes as the 2x2 matrices
    # [[alpha, beta], [0, 1]]; the prefix product is normalised inside the scan, so
    # the affine part is recovered as entry (0,1) divided by entry (1,1).
    K_gain = (P_pred * w_s) / S
    alpha = torch.cat(
        [torch.ones(1, dtype=w_s.dtype, device=w_s.device), phi * (1.0 - K_gain[1:] * w_s[1:])]
    )
    phiv = torch.cat([torch.ones(1, dtype=w_s.dtype, device=w_s.device), phi])
    if y is None:  # return_filter only: no mean recursion to run
        return S, None, idx, K_gain, alpha, w_s * phiv

    y_s = y.index_select(-1, idx)
    beta = K_gain * y_s  # (..., dim)

    Mm = torch.zeros(*beta.shape, 2, 2, dtype=w_s.dtype, device=w_s.device)
    Mm[..., 0, 0] = alpha.expand_as(beta)
    Mm[..., 0, 1] = beta
    Mm[..., 1, 1] = 1.0

    Mmc = _parallel_prefix_2x2(Mm)
    m = Mmc[..., 0, 1] / Mmc[..., 1, 1].clamp(min=1e-30)

    m_prev = torch.cat([torch.zeros_like(m[..., :1]), m[..., :-1]], dim=-1)
    c = w_s * phiv
    v = y_s - c * m_prev
    if return_filter:
        return S, v, idx, K_gain, alpha, c
    return S, v


def kalman_stats_sequential(
    a: torch.Tensor,
    w: torch.Tensor,
    d: torch.Tensor,
    y: torch.Tensor | None = None,
):
    """Reference (textbook) sequential Kalman filter -- same contract as
    :func:`kalman_stats`.

    Kept deliberately as an independent, easy-to-audit implementation: it shares no
    code with the scan-based version, so the two agreeing is a genuine cross-check
    of the ported scans. ``O(dim)`` sequential steps, so it is slow for large
    ``dim``; prefer :func:`kalman_stats`.
    """
    dim = a.shape[-1]
    a_s, idx = torch.sort(a)
    w_s, d_s = w[idx], d[idx]
    dt = (a_s[1:] - a_s[:-1]).clamp(min=1e-12)
    phi = torch.exp(-dt)
    q = (1.0 - phi**2).clamp(min=0.0)

    want_v = y is not None
    if want_v:
        y_s = y.index_select(-1, idx)
        m_filt = torch.zeros(y_s.shape[:-1], dtype=w_s.dtype, device=w_s.device)
        v_list = []

    S_list = []
    P_pred = torch.ones((), dtype=w_s.dtype, device=w_s.device)  # z_0 stationary var
    P_filt = None
    for k in range(dim):
        if k > 0:
            P_pred = phi[k - 1] ** 2 * P_filt + q[k - 1]
        S_k = w_s[k] ** 2 * P_pred + d_s[k]
        S_list.append(S_k)
        gain = P_pred * w_s[k] / S_k
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


def kalman_solve(
    a: torch.Tensor, w: torch.Tensor, d: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """``A^{-1} y`` in ``O(dim)`` for ``A = diag(d) + diag(w) K(a) diag(w)``.

    The innovations form of the filter in :func:`kalman_stats` is ``v = L y`` with
    ``L`` unit lower-triangular and ``Cov(v) = diag(S)``, hence ``L A L^T = diag(S)``
    and

        ``A^{-1} = L^T diag(S)^{-1} L``   so   ``A^{-1} y = L^T (v / S)``.

    Applying ``L^T`` is one *backward* first-order linear recursion, run with the same
    associative scan the forward filter uses, so this costs the same ``O(dim)`` work and
    ``O(log dim)`` depth as the log-determinant -- no sequential sweep. Supports
    arbitrary batch dimensions on ``y`` (shape ``(..., dim)``).
    """
    S, v, idx, K_gain, alpha, c = kalman_stats(a, w, d, y, return_filter=True)
    u = v / S
    dim = a.shape[-1]
    if dim == 1:
        out = torch.empty_like(u)
        return out.index_copy_(-1, idx, u)

    # r_j = sum_{k>j} u_k c_k prod_{l=j+1}^{k-1} alpha_l, i.e. the backward recursion
    #   r_j = alpha_{j+1} r_{j+1} + (u c)_{j+1},  r_{dim-1} = 0,
    # which in reversed index m = dim-1-j is the same affine scan as the mean recursion.
    uc = u * c
    one = torch.ones(1, dtype=u.dtype, device=u.device)
    ap = torch.cat([one, alpha[1:].flip(-1)])
    bp = torch.cat([torch.zeros_like(uc[..., :1]), uc[..., 1:].flip(-1)], dim=-1)

    Mm = torch.zeros(*bp.shape, 2, 2, dtype=u.dtype, device=u.device)
    Mm[..., 0, 0] = ap.expand_as(bp)
    Mm[..., 0, 1] = bp
    Mm[..., 1, 1] = 1.0
    Mmc = _parallel_prefix_2x2(Mm)
    r = (Mmc[..., 0, 1] / Mmc[..., 1, 1].clamp(min=1e-30)).flip(-1)

    x_s = u - K_gain * r
    out = torch.empty_like(x_s)
    return out.index_copy_(-1, idx, x_s)


def seriation_order(A: torch.Tensor) -> torch.Tensor:
    """Recover a 1-D ordering of the coordinates of ``A`` by spectral seriation.

    Builds the absolute-correlation graph of ``A`` and sorts the coordinates by the
    Fiedler vector (the second-smallest eigenvector of its graph Laplacian). Uses no
    side information about the true order, so it is the honest choice whenever the
    coordinates are not already physically ordered; costs one ``dim x dim``
    eigendecomposition.
    """
    g = torch.diagonal(A).clamp(min=1e-30).sqrt()
    R = (A / g[:, None] / g[None, :]).abs()
    L = torch.diag(R.sum(1)) - R
    fiedler = torch.linalg.eigh(L)[1][:, 1]
    return torch.argsort(fiedler)


def logdet_from_params(a: torch.Tensor, w: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """``log det (diag(d) + diag(w) K(a) diag(w))`` via the scan-based filter."""
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
    """``log det (diag(d) + diag(w) K(a) diag(w))``.

    Thin alias for :func:`kalman_stats` (which now performs the variance scan in
    parallel, as the reference implementation does). Kept as a named entry point
    because callers -- e.g. ``LCMLaplace.log_det_posterior_precision`` -- refer to it.
    """
    S, _ = kalman_stats(a, w, d, None)
    return torch.log(S).sum()


# Max coordinate span handled inside a single cumsum block. exp() of this must stay
# well inside the dtype's range (float64 overflows near 709, float32 near 88).
_BLOCK_RANGE = 60.0


def _decaying_prefix(a_sorted: torch.Tensor, x_sorted: torch.Tensor) -> torch.Tensor:
    """Exclusive decaying prefix ``L_i = sum_{j<i} exp(a_j - a_i) x_j`` for
    ascending ``a_sorted``, exact for **any** coordinate range.

    Every individual term has a non-positive exponent (``a_j <= a_i``), so the
    result is well-scaled -- but the obvious factorisation through a common
    offset, ``exp(-(a_i - a_0)) * sum_j exp(a_j - a_0) x_j``, evaluates each
    factor separately and **overflows to inf/NaN once the coordinate range
    exceeds ~709 in float64** (~88 in float32), even though their product is
    bounded by 1.

    (A ``logcumsumexp`` formulation is also exact but measured 13-20x slower than
    ``cumsum`` here, so we instead rescale per block -- same asymptotics, and the
    fast path is preserved.)
    """
    dim = a_sorted.shape[-1]
    # Split the (ascending) coordinates into contiguous blocks whose internal range
    # is <= _BLOCK_RANGE, so the fast offset trick is safe *within* a block, and
    # carry the running accumulator across blocks rescaled to each block's origin
    # (the rescaling factor exp(-(off_next - off)) is always <= 1). Cost is one
    # cumsum per block with only ~range/_BLOCK_RANGE blocks, so this keeps the
    # O(dim) cumsum speed while removing the overflow.
    starts = [0]
    while True:
        s = starts[-1]
        nxt = int(torch.searchsorted(a_sorted, a_sorted[s] + _BLOCK_RANGE, right=True))
        nxt = max(nxt, s + 1)
        if nxt >= dim:
            break
        starts.append(nxt)

    carry = torch.zeros(x_sorted.shape[:-1], dtype=x_sorted.dtype, device=x_sorted.device)
    pieces = []
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < len(starts) else dim
        a_blk, x_blk = a_sorted[s:e], x_sorted[..., s:e]
        off = a_sorted[s]
        ex = torch.exp(a_blk - off)  # <= exp(_BLOCK_RANGE), safe
        em = torch.exp(-(a_blk - off))  # <= 1
        incl = torch.cumsum(ex * x_blk, dim=-1)
        excl = incl - ex * x_blk
        pieces.append(em * (excl + carry.unsqueeze(-1)))
        if e < dim:
            # re-express the accumulator relative to the next block's origin
            carry = torch.exp(-(a_sorted[e] - off)) * (carry + incl[..., -1])
    return torch.cat(pieces, dim=-1)


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

    left = _decaying_prefix(a_sorted, x_sorted)
    # R_i = sum_{j>i} exp(a_i - a_j) x_j is the same prefix problem on the
    # reversed, negated coordinates (which are again increasing).
    right = torch.flip(
        _decaying_prefix(torch.flip(-a_sorted, dims=(-1,)),
                         torch.flip(x_sorted, dims=(-1,))),
        dims=(-1,),
    )

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

    def solve(self, y: torch.Tensor) -> torch.Tensor:
        """``A^{-1} y`` in ``O(dim)`` per right-hand side (see :func:`kalman_solve`)."""
        return kalman_solve(self.a, self.w, self.d, y)

    @torch.no_grad()
    def fit_moments(
        self,
        X: torch.Tensor | None = None,
        *,
        moments: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        order: torch.Tensor | None = None,
        clip: float = 0.999,
    ) -> LCMCovariance:
        """Closed-form fit from the first three bands of the target second moment.

        The LCM is a stationary-latent AR(1) plus independent noise, so for consecutive
        coordinates ``p, p+1, p+2`` along the chain the target entries satisfy

            ``A[p,p+1]   = w_p w_{p+1} phi_{p+1}``
            ``A[p+1,p+2] = w_{p+1} w_{p+2} phi_{p+2}``
            ``A[p,p+2]   = w_p w_{p+2} phi_{p+1} phi_{p+2}``

        whose ratio identifies the amplitudes without any optimisation:

            ``w_{p+1}^2 = A[p,p+1] A[p+1,p+2] / A[p,p+2]`` ,

        then ``phi`` from the first band and ``d = diag(A) - w^2``. One pass over the
        data, ``O(N dim)`` time and ``O(dim)`` memory -- the ``dim x dim`` matrix is
        never formed, and there is no optimiser to tune or diverge. Exact (to floating
        point) when the target is itself an LCM, except at the two endpoint amplitudes,
        which these three bands do not identify; they are copied inward.

        Pass either the activations ``X`` of shape ``(N, dim)``, or precomputed
        ``moments = (s0, s1, s2)`` -- the diagonal and the first two off-diagonal bands,
        already in chain order. ``order`` gives the 1-D ordering of the coordinates when
        it is not the natural one (e.g. from seriation); parameters are stored in the
        original indexing either way.
        """
        dev, dt = self.w.device, self.w.dtype
        if (X is None) == (moments is None):
            raise ValueError("fit_moments needs exactly one of `X` or `moments`.")
        perm = (
            torch.arange(self.dim, device=dev)
            if order is None
            else torch.as_tensor(order, device=dev, dtype=torch.long)
        )
        if X is not None:
            P = torch.as_tensor(X, device=dev, dtype=dt)[:, perm]
            s0 = (P * P).mean(0)
            s1 = (P[:, :-1] * P[:, 1:]).mean(0)
            s2 = (P[:, :-2] * P[:, 2:]).mean(0)
        else:
            s0, s1, s2 = (torch.as_tensor(m, device=dev, dtype=dt) for m in moments)

        tol = self.eps * s0.mean().clamp(min=self.eps)
        w2 = torch.zeros(self.dim, device=dev, dtype=dt)
        if self.dim > 2:
            w2[1:-1] = torch.where(s2.abs() > tol, s1[:-1] * s1[1:] / s2, torch.zeros_like(s2))
            w2[0], w2[-1] = w2[1], w2[-2]  # endpoints are not identified by 3 bands
        elif self.dim == 2:
            w2[:] = s1.abs()
        w2 = w2.clamp(min=0.0).minimum(clip * s0)
        w = w2.sqrt()

        if self.dim > 1:
            denom = (w[:-1] * w[1:]).clamp(min=tol)
            phi = torch.where(denom > tol, s1 / denom, torch.full_like(denom, tol))
            phi = phi.clamp(min=tol, max=1.0 - 1e-12)
            a_chain = torch.cat([torch.zeros(1, device=dev, dtype=dt), (-phi.log()).cumsum(0)])
        else:
            a_chain = torch.zeros(1, device=dev, dtype=dt)

        d = (s0 - w2).clamp(min=tol) - self.eps
        self.a.index_copy_(0, perm, a_chain)
        self.w.index_copy_(0, perm, w)
        self.u_raw.index_copy_(0, perm, _inverse_softplus(d.clamp(min=1e-12)))
        return self

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
    @torch.no_grad()
    def _auto_init_scale(self, diag_estimate: torch.Tensor, nugget_frac: float = 0.15) -> None:
        """Rescale ``w`` (and the nugget ``d``) from a per-dimension diagonal
        estimate of the fitting target, keeping the (data-independent) latent
        ordering ``a`` set at construction time.

        Without this, ``w`` keeps its fixed default (``init_w=0.1``) regardless of
        the target's actual magnitude. For gradient-based fitting that mismatch is
        not just slow to correct -- a target diagonal many orders of magnitude
        above ``w^2`` produces a huge initial loss/gradient that can make Adam
        diverge (rather than merely converge slowly), silently producing a much
        worse fit than a correctly-scaled initialization. Both :meth:`fit_dense`
        and :meth:`fit_frobenius` call this by default (``auto_init=True``).
        """
        diag_estimate = diag_estimate.clamp(min=1e-12).to(self.w.dtype)
        w0 = torch.sqrt((1.0 - nugget_frac) * diag_estimate)
        d0 = (nugget_frac * diag_estimate).clamp(min=2 * self.eps) - self.eps
        self.w.copy_(w0)
        self.u_raw.copy_(_inverse_softplus(d0))

    def fit_dense(
        self,
        target: torch.Tensor,
        n_steps: int = 500,
        lr: float = 1e-2,
        tol: float = 1e-9,
        verbose: bool = False,
        auto_init: bool = True,
        normalize: bool = True,
    ) -> float:
        """Fit ``(a, w, u)`` to a dense PSD target by minimising ``||A - target||_F^2``.

        Parameters
        ----------
        auto_init : bool, default=True
            Rescale ``w``/``d`` from ``target``'s diagonal before optimizing (see
            :meth:`_auto_init_scale`). Strongly recommended; disable only to
            reproduce a specific hand-set initialization.
        normalize : bool, default=True
            Fit to ``target / s`` (with ``s`` the mean diagonal) and rescale the
            parameters afterwards. This is **exact** -- the family is positively
            homogeneous, so ``A*(c T) = c A*(T)`` -- and it matters a great deal.
            Adam takes steps of absolute size ``lr``; on a GGN-scale target
            (~1e5) ``auto_init`` puts ``w`` at ~300, so one step moves it by
            ~2e-4 relative and the fit barely leaves its initialization. Fitting
            at unit scale puts every parameter near 1, where ``lr`` is matched.
            Disable only to reproduce pre-fix behaviour.

        Returns
        -------
        rel_error : float
            Final relative Frobenius error ``||A - target||_F / ||target||_F``.
        """
        target = target.to(self.a.dtype)
        scale = (
            target.diagonal().mean().clamp(min=1e-30)
            if normalize
            else torch.ones((), dtype=target.dtype, device=target.device)
        )
        tgt = target / scale
        if auto_init:
            self._auto_init_scale(tgt.diagonal())
        # only optimise what is trainable -- callers may freeze `a` to pin a known
        # ordering and fit `w`/`d` alone (see LCMLaplace's `lcm_learn_a`).
        params = [p for p in (self.a, self.w, self.u_raw) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
        prev = None
        for step in range(n_steps):
            opt.zero_grad()
            loss = ((self.dense() - tgt) ** 2).sum()
            loss.backward()
            opt.step()
            if verbose and step % max(1, n_steps // 10) == 0:
                print(f"  [fit_dense] step {step:4d}  loss {loss.item():.6e}")
            if prev is not None and abs(prev - loss.item()) < tol * (prev + 1e-12):
                break
            prev = loss.item()
        self._rescale_(scale)
        with torch.no_grad():
            rel = (self.dense() - target).norm() / target.norm().clamp(min=1e-12)
        return rel.item()

    @torch.no_grad()
    def _rescale_(self, scale: torch.Tensor) -> None:
        """In-place ``A <- scale * A``, exploiting positive homogeneity.

        ``A = diag(d) + diag(w) K(a) diag(w)``, so scaling ``A`` by ``s`` means
        ``w <- sqrt(s) w`` and ``d <- s d``; the coordinates ``a`` are unchanged
        because the kernel is scale-free. Used to undo the fit-time normalization.
        """
        if float(scale) == 1.0:
            return
        self.w.mul_(scale.sqrt())
        d_new = scale * (F.softplus(self.u_raw) + self.eps) - self.eps
        self.u_raw.copy_(_inverse_softplus(d_new.clamp(min=1e-12)))

    def fit_frobenius(
        self,
        vectors: torch.Tensor,
        n_steps: int = 500,
        lr: float = 1e-2,
        batch_size: int | None = None,
        verbose: bool = False,
        auto_init: bool = True,
        normalize: bool = True,
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
        auto_init : bool, default=True
            Rescale ``w``/``d`` from the empirical diagonal of ``S_hat`` (i.e.
            ``mean(vectors**2, dim=0)``) before optimizing -- see
            :meth:`_auto_init_scale`. Without this, a target whose scale differs
            from the fixed default (``init_w=0.1``) can make Adam diverge rather
            than converge, silently producing a much worse fit.

        Returns
        -------
        loss : float
            Final value of the (constant-dropped) objective.
        """
        vectors = vectors.to(self.a.dtype)
        N = vectors.shape[0]
        # Scale-normalise exactly as fit_dense does. S_hat = (1/N) sum v v^T, so
        # dividing the vectors by sqrt(s) divides S_hat by s; _rescale_ undoes it
        # afterwards. See fit_dense's `normalize` docstring for why it matters.
        scale = (
            (vectors**2).mean().clamp(min=1e-30)
            if normalize
            else torch.ones((), dtype=vectors.dtype, device=vectors.device)
        )
        vectors = vectors / scale.sqrt()
        if auto_init:
            self._auto_init_scale((vectors**2).mean(dim=0))
        # only optimise what is trainable -- callers may freeze `a` to pin a known
        # ordering and fit `w`/`d` alone (see LCMLaplace's `lcm_learn_a`).
        params = [p for p in (self.a, self.w, self.u_raw) if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr)
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
        self._rescale_(scale)
        # the objective was evaluated on the rescaled vectors, so report it on the
        # caller's scale: ||A - S||_F^2 is homogeneous of degree 2 in the scale.
        return last * float(scale) ** 2

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
