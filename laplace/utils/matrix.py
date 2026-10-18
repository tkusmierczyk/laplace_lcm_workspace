from __future__ import annotations

from math import pow
from typing import Iterable

import numpy as np
import opt_einsum as oe
import torch
from torch import nn

from laplace.lcm import (
    LCMCovariance,
    kalman_solve,
    kernel_matvec,
    logdet_from_params,
    seriation_order,
)
from laplace.utils.utils import _is_valid_scalar, block_diag, kron, symeig

__all__ = ["Kron", "KronDecomposed", "KronLCMDecomposed"]


class Kron:
    """Kronecker factored approximate curvature representation for a corresponding
    neural network.
    Each element in `kfacs` is either a tuple or single matrix.
    A tuple represents two Kronecker factors \\(Q\\), and \\(H\\) and a single element
    is just a full block Hessian approximation.

    Parameters
    ----------
    kfacs : list[Iterable[torch.Tensor] | torch.Tensor]
        each element in the list is a tuple of two Kronecker factors Q, H
        or a single matrix approximating the Hessian (in case of bias, for example)
    """

    def __init__(self, kfacs: list[tuple[torch.Tensor] | torch.Tensor]) -> None:
        self.kfacs: list[tuple[torch.Tensor] | torch.Tensor] = kfacs

    @classmethod
    def init_from_model(
        cls,
        model: nn.Module | Iterable[nn.Parameter],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Kron:
        """Initialize Kronecker factors based on a models architecture.

        Parameters
        ----------
        model: `nn.Module` or iterable of parameters, e.g. `model.parameters()`
        device: The device where each of the Kronecker factor lives in.
        dtype: The data type of each Kronecker factor.

        Returns
        -------
        kron : Kron
        """
        if isinstance(model, torch.nn.Module):
            params = model.parameters()
        else:
            params = model

        kfacs = list()
        for p in params:
            if p.ndim == 1:  # bias
                P = p.size(0)
                kfacs.append([torch.zeros(P, P, device=device, dtype=dtype)])
            elif 4 >= p.ndim >= 2:  # fully connected or conv
                if p.ndim == 2:  # fully connected
                    P_in, P_out = p.size()
                else:
                    P_in, P_out = p.shape[0], np.prod(p.shape[1:])

                kfacs.append(
                    [
                        torch.zeros(P_in, P_in, device=device, dtype=dtype),
                        torch.zeros(P_out, P_out, device=device, dtype=dtype),
                    ]
                )
            else:
                raise ValueError("Invalid parameter shape in network.")

        return cls(kfacs)

    def __add__(self, other: Kron) -> Kron:
        """Add up Kronecker factors `self` and `other`.

        Parameters
        ----------
        other : Kron

        Returns
        -------
        kron : Kron
        """
        if not isinstance(other, Kron):
            raise ValueError("Can only add Kron to Kron.")

        kfacs = [
            [Hi.add(Hj) for Hi, Hj in zip(Fi, Fj)]
            for Fi, Fj in zip(self.kfacs, other.kfacs)
        ]

        return Kron(kfacs)

    def __mul__(self, scalar: float | torch.Tensor) -> Kron:
        """Multiply all Kronecker factors by scalar.
        The multiplication is distributed across the number of factors
        using `pow(scalar, 1 / len(F))`. `len(F)` is either `1` or `2`.

        Parameters
        ----------
        scalar : float, torch.Tensor

        Returns
        -------
        kron : Kron
        """
        if not _is_valid_scalar(scalar):
            raise ValueError("Input not valid python or torch scalar.")

        # distribute factors evenly so that each group is multiplied by factor
        kfacs = [[pow(scalar, 1 / len(F)) * Hi for Hi in F] for F in self.kfacs]
        return Kron(kfacs)

    def __len__(self) -> int:
        return len(self.kfacs)

    def decompose(self, damping: bool = False) -> KronDecomposed:
        """Eigendecompose Kronecker factors and turn into `KronDecomposed`.
        Parameters
        ----------
        damping : bool
            use damping

        Returns
        -------
        kron_decomposed : KronDecomposed
        """
        eigvecs, eigvals = list(), list()
        for F in self.kfacs:
            Qs, ls = list(), list()
            for Hi in F:
                if Hi.ndim > 1:
                    # Dense Kronecker factor.
                    eigval, Q = symeig(Hi)
                else:
                    # Diagonal Kronecker factor.
                    eigval = Hi
                    # This might be too memory intensive since len(Hi) can be large.
                    Q = torch.eye(len(Hi), dtype=Hi.dtype, device=Hi.device)
                Qs.append(Q)
                ls.append(eigval)
            eigvecs.append(Qs)
            eigvals.append(ls)
        return KronDecomposed(eigvecs, eigvals, damping=damping)

    def _bmm(self, W: torch.Tensor) -> torch.Tensor:
        """Implementation of `bmm` which casts the parameters to the right shape.

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        # self @ W[batch, k, params]
        assert len(W.size()) == 3
        B, K, P = W.size()
        W = W.reshape(B * K, P)
        cur_p = 0
        SW = list()
        for Fs in self.kfacs:
            if len(Fs) == 1:
                Q = Fs[0]
                p = len(Q)
                W_p = W[:, cur_p : cur_p + p].T
                SW.append((Q @ W_p).T if Q.ndim > 1 else (Q.view(-1, 1) * W_p).T)
                cur_p += p
            elif len(Fs) == 2:
                Q, H = Fs
                p_in, p_out = len(Q), len(H)
                p = p_in * p_out
                W_p = W[:, cur_p : cur_p + p].reshape(B * K, p_in, p_out)
                QW_p = Q @ W_p if Q.ndim > 1 else Q.view(-1, 1) * W_p
                QW_pHt = QW_p @ H.T if H.ndim > 1 else QW_p * H.view(1, -1)
                SW.append(QW_pHt.reshape(B * K, p_in * p_out))
                cur_p += p
            else:
                raise AttributeError("Shape mismatch")
        SW = torch.cat(SW, dim=1).reshape(B, K, P)
        return SW

    def bmm(self, W: torch.Tensor, exponent: float = 1) -> torch.Tensor:
        """Batched matrix multiplication with the Kronecker factors.
        If Kron is `H`, we compute `H @ W`.
        This is useful for computing the predictive or a regularization
        based on Kronecker factors as in continual learning.

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`
        exponent: float, default=1
            only can be `1` for Kron, requires `KronDecomposed` for other
            exponent values of the Kronecker factors.

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        if exponent != 1:
            raise ValueError("Only supported after decomposition.")
        if W.ndim == 1:
            return self._bmm(W.unsqueeze(0).unsqueeze(0)).squeeze()
        elif W.ndim == 2:
            return self._bmm(W.unsqueeze(1)).squeeze()
        elif W.ndim == 3:
            return self._bmm(W)
        else:
            raise ValueError("Invalid shape for W")

    def logdet(self) -> torch.Tensor:
        """Compute log determinant of the Kronecker factors and sums them up.
        This corresponds to the log determinant of the entire Hessian approximation.

        Returns
        -------
        logdet : torch.Tensor
        """
        logdet = 0
        for F in self.kfacs:
            if len(F) == 1:
                logdet += F[0].logdet() if F[0].ndim > 1 else F[0].log().sum()
            else:  # len(F) == 2
                Hi, Hj = F
                p_in, p_out = len(Hi), len(Hj)
                logdet += p_out * Hi.logdet() if Hi.ndim > 1 else p_out * Hi.log().sum()
                logdet += p_in * Hj.logdet() if Hj.ndim > 1 else p_in * Hj.log().sum()
        return logdet

    def diag(self) -> torch.Tensor:
        """Extract diagonal of the entire Kronecker factorization.

        Returns
        -------
        diag : torch.Tensor
        """
        diags = list()
        for F in self.kfacs:
            F0 = F[0].diag() if F[0].ndim > 1 else F[0]
            if len(F) == 1:
                diags.append(F0)
            else:
                F1 = F[1].diag() if F[1].ndim > 1 else F[1]
                diags.append(torch.outer(F0, F1).flatten())
        return torch.cat(diags)

    def to_matrix(self) -> torch.Tensor:
        """Make the Kronecker factorization dense by computing the kronecker product.
        Warning: this should only be used for testing purposes as it will allocate
        large amounts of memory for big architectures.

        Returns
        -------
        block_diag : torch.Tensor
        """
        blocks = list()
        for F in self.kfacs:
            F0 = F[0] if F[0].ndim > 1 else F[0].diag()
            if len(F) == 1:
                blocks.append(F0)
            else:
                F1 = F[1] if F[1].ndim > 1 else F[1].diag()
                blocks.append(kron(F0, F1))
        return block_diag(blocks)

    # for commutative operations
    __radd__ = __add__
    __rmul__ = __mul__


class KronDecomposed:
    """Decomposed Kronecker factored approximate curvature representation
    for a corresponding neural network.
    Each matrix in `Kron` is decomposed to obtain `KronDecomposed`.
    Front-loading decomposition allows cheap repeated computation
    of inverses and log determinants.
    In contrast to `Kron`, we can add scalar or layerwise scalars but
    we cannot add other `Kron` or `KronDecomposed` anymore.

    Parameters
    ----------
    eigenvectors : list[Tuple[torch.Tensor]]
        eigenvectors corresponding to matrices in a corresponding `Kron`
    eigenvalues : list[Tuple[torch.Tensor]]
        eigenvalues corresponding to matrices in a corresponding `Kron`
    deltas : torch.Tensor
        addend for each group of Kronecker factors representing, for example,
        a prior precision
    dampen : bool, default=False
        use dampen approximation mixing prior and Kron partially multiplicatively
    """

    def __init__(
        self,
        eigenvectors: list[tuple[torch.Tensor]],
        eigenvalues: list[tuple[torch.Tensor]],
        deltas: torch.Tensor | None = None,
        damping: bool = False,
    ):
        self.eigenvectors: list[tuple[torch.Tensor]] = eigenvectors
        self.eigenvalues: list[tuple[torch.Tensor]] = eigenvalues

        device: torch.device = eigenvectors[0][0].device
        dtype: torch.dtype = eigenvectors[0][0].dtype

        if deltas is None:
            self.deltas: torch.Tensor = torch.zeros(
                len(self), device=device, dtype=dtype
            )
        else:
            self._check_deltas(deltas)
            self.deltas: torch.Tensor = deltas
        self.damping: bool = damping

    def detach(self) -> KronDecomposed:
        self.deltas = self.deltas.detach()
        return self

    def _check_deltas(self, deltas: torch.Tensor) -> None:
        if not isinstance(deltas, torch.Tensor):
            raise ValueError("Can only add torch.Tensor to KronDecomposed.")

        if deltas.ndim == 0 or (  # scalar
            deltas.ndim == 1  # vector of length 1 or len(self)
            and (len(deltas) == 1 or len(deltas) == len(self))
        ):
            return
        else:
            raise ValueError("Invalid shape of delta added to KronDecomposed.")

    def __add__(self, deltas: torch.Tensor) -> KronDecomposed:
        """Add scalar per layer or only scalar to Kronecker factors.

        Parameters
        ----------
        deltas : torch.Tensor
            either same length as `eigenvalues` or scalar.

        Returns
        -------
        kron : KronDecomposed
        """
        self._check_deltas(deltas)
        return KronDecomposed(self.eigenvectors, self.eigenvalues, self.deltas + deltas)

    def __mul__(self, scalar: torch.Tensor | float) -> KronDecomposed:
        """Multiply by a scalar by changing the eigenvalues.
        Same as for the case of `Kron`.

        Parameters
        ----------
        scalar : torch.Tensor or float

        Returns
        -------
        kron : KronDecomposed
        """
        if not _is_valid_scalar(scalar):
            raise ValueError("Invalid argument, can only multiply Kron with scalar.")

        eigenvalues = [
            [pow(scalar, 1 / len(ls)) * eigval for eigval in ls]
            for ls in self.eigenvalues
        ]
        return KronDecomposed(self.eigenvectors, eigenvalues, self.deltas)

    def __len__(self) -> int:
        return len(self.eigenvalues)

    def logdet(self) -> torch.Tensor:
        """Compute log determinant of the Kronecker factors and sums them up.
        This corresponds to the log determinant of the entire Hessian approximation.
        In contrast to `Kron.logdet()`, additive `deltas` corresponding to prior
        precisions are added.

        Returns
        -------
        logdet : torch.Tensor
        """
        logdet = 0
        for ls, delta in zip(self.eigenvalues, self.deltas):
            if len(ls) == 1:  # not KFAC just full
                logdet += torch.log(ls[0] + delta).sum()
            elif len(ls) == 2:
                l1, l2 = ls
                if self.damping:
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    logdet += torch.log(torch.outer(l1d, l2d)).sum()
                else:
                    logdet += torch.log(torch.outer(l1, l2) + delta).sum()
            else:
                raise ValueError("Too many Kronecker factors. Something went wrong.")
        return logdet

    def _bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        """Implementation of `bmm`, i.e., `self ** exponent @ W`.

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`
        exponent : float
            exponent on `self`

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        # self @ W[batch, k, params]
        assert len(W.size()) == 3
        B, K, P = W.size()
        W = W.reshape(B * K, P)
        cur_p = 0
        SW = list()
        for i, (ls, Qs, delta) in enumerate(
            zip(self.eigenvalues, self.eigenvectors, self.deltas)
        ):
            if len(ls) == 1:
                Q, eigval, p = Qs[0], ls[0], len(ls[0])
                ldelta_exp = torch.pow(eigval + delta, exponent).reshape(-1, 1)
                W_p = W[:, cur_p : cur_p + p].T
                SW.append((Q @ (ldelta_exp * (Q.T @ W_p))).T)
                cur_p += p
            elif len(ls) == 2:
                Q1, Q2 = Qs
                l1, l2 = ls
                p = len(l1) * len(l2)
                if self.damping:
                    l1d, l2d = l1 + torch.sqrt(delta), l2 + torch.sqrt(delta)
                    ldelta_exp = torch.pow(torch.outer(l1d, l2d), exponent).unsqueeze(0)
                else:
                    ldelta_exp = torch.pow(
                        torch.outer(l1, l2) + delta, exponent
                    ).unsqueeze(0)
                p_in, p_out = len(l1), len(l2)
                W_p = W[:, cur_p : cur_p + p].reshape(B * K, p_in, p_out)
                W_p = (Q1.T @ W_p @ Q2) * ldelta_exp
                W_p = Q1 @ W_p @ Q2.T
                SW.append(W_p.reshape(B * K, p_in * p_out))
                cur_p += p
            else:
                raise AttributeError("Shape mismatch")
        SW = torch.cat(SW, dim=1).reshape(B, K, P)
        return SW

    def inv_square_form(self, W: torch.Tensor) -> torch.Tensor:
        # W either Batch x K x params or Batch x params
        SW = self._bmm(W, exponent=-1)
        return torch.bmm(W, SW.transpose(1, 2))

    def bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        """Batched matrix multiplication with the decomposed Kronecker factors.
        This is useful for computing the predictive or a regularization loss.
        Compared to `Kron.bmm`, a prior can be added here in form of `deltas`
        and the exponent can be other than just 1.
        Computes \\(H^{exponent} W\\).

        Parameters
        ----------
        W : torch.Tensor
            matrix `(batch, classes, params)`
        exponent: float, default=1

        Returns
        -------
        SW : torch.Tensor
            result `(batch, classes, params)`
        """
        if W.ndim == 1:
            return self._bmm(W.unsqueeze(0).unsqueeze(0), exponent).squeeze()
        elif W.ndim == 2:
            return self._bmm(W.unsqueeze(1), exponent).squeeze()
        elif W.ndim == 3:
            return self._bmm(W, exponent)
        else:
            raise ValueError("Invalid shape for W")

    def diag(self, exponent: float = 1) -> torch.Tensor:
        """Extract diagonal of the entire decomposed Kronecker factorization.

        Parameters
        ----------
        exponent: float, default=1
            exponent of the Kronecker factorization

        Returns
        -------
        diag : torch.Tensor
        """
        diags = list()
        for Qs, ls, delta in zip(self.eigenvectors, self.eigenvalues, self.deltas):
            if len(ls) == 1:
                Ql = Qs[0] * torch.pow(ls[0] + delta, exponent).reshape(1, -1)
                d = torch.einsum(
                    "mp,mp->m", Ql, Qs[0]
                )  # only compute inner products for diag
                diags.append(d)
            else:
                Q1, Q2 = Qs
                l1, l2 = ls
                if self.damping:
                    delta_sqrt = torch.sqrt(delta)
                    eigval = torch.pow(
                        torch.outer(l1 + delta_sqrt, l2 + delta_sqrt), exponent
                    )
                else:
                    eigval = torch.pow(torch.outer(l1, l2) + delta, exponent)
                d = oe.contract("mp,nq,pq,mp,nq->mn", Q1, Q2, eigval, Q1, Q2).flatten()
                diags.append(d)
        return torch.cat(diags)

    def to_matrix(self, exponent: float = 1) -> torch.Tensor:
        """Make the Kronecker factorization dense by computing the kronecker product.
        Warning: this should only be used for testing purposes as it will allocate
        large amounts of memory for big architectures.

        Parameters
        ----------
        exponent: float, default=1
            exponent of the Kronecker factorization

        Returns
        -------
        block_diag : torch.Tensor
        """
        blocks = list()
        for Qs, ls, delta in zip(self.eigenvectors, self.eigenvalues, self.deltas):
            if len(ls) == 1:
                Q, eigval = Qs[0], ls[0]
                blocks.append(Q @ torch.diag(torch.pow(eigval + delta, exponent)) @ Q.T)
            else:
                Q1, Q2 = Qs
                l1, l2 = ls
                Q = kron(Q1, Q2)
                if self.damping:
                    delta_sqrt = torch.sqrt(delta)
                    eigval = torch.pow(
                        torch.outer(l1 + delta_sqrt, l2 + delta_sqrt), exponent
                    )
                else:
                    eigval = torch.pow(torch.outer(l1, l2) + delta, exponent)
                L = torch.diag(eigval.flatten())
                blocks.append(Q @ L @ Q.T)
        return block_diag(blocks)

    # for commutative operations
    __radd__ = __add__
    __rmul__ = __mul__


class KronLCMDecomposed:
    """Kronecker-factored precision whose *large* factor is kept in the LCM family.

    For a layer with Kronecker factors \\(H_0 \\otimes H_1\\) (sizes `out` and `in`,
    acting on the row-major `(out, in)` parameter layout) and prior \\(\\delta\\), the
    usual KFAC route eigendecomposes *both* factors. Here only the **small** factor is
    eigendecomposed, \\(H_s = Q \\mathrm{diag}(\\gamma) Q^\\top\\), and the rotation
    \\(Q \\otimes I\\) block-diagonalises the posterior precision into `n_small`
    independent problems over the large axis:

        \\((Q \\otimes I)^\\top (H_s \\otimes H_l + \\delta I)(Q \\otimes I)
           = \\bigoplus_j (\\gamma_j H_l + \\delta I)\\).

    The LCM family is closed under \\(A \\mapsto sA + \\mathrm{diag}(c)\\), so every
    reduced block is *again* an LCM and inherits its \\(O(\\mathrm{in})\\) solve and
    log-determinant (`laplace.lcm.kalman_solve`, `laplace.lcm.kalman_stats`). The prior
    is therefore exact -- unlike the factored-damping approximation
    \\((H_0 + \\pi\\sqrt{\\delta}I) \\otimes (H_1 + \\sqrt{\\delta}/\\pi\\,I)\\) that
    `KronDecomposed(damping=True)` uses -- and storage drops from
    \\(\\mathrm{in}^2 + \\mathrm{out}^2\\) to \\(3\\,\\mathrm{in} + \\mathrm{out}^2\\).

    Exposes the same interface as `KronDecomposed`, except that `bmm` supports only
    `exponent` in {1, -1}: an LCM has no closed-form square root, so samplers use
    perturb-and-solve instead (see `KronLCMLaplace.sample`).

    Parameters
    ----------
    blocks : list[dict]
        per-layer representation, built by `from_kron`
    deltas : torch.Tensor
        prior precision per layer
    scale : float
        multiplier on the curvature (set by `__mul__`, e.g. the Hessian factor)
    """

    def __init__(self, blocks: list[dict], deltas: torch.Tensor | None = None,
                 scale: float | torch.Tensor = 1.0):
        self.blocks = blocks
        self.scale = scale
        ref = blocks[0]["Q"] if blocks[0]["kind"] == "lcm" else blocks[0]["Qs"][0]
        if deltas is None:
            self.deltas = torch.zeros(len(self), device=ref.device, dtype=ref.dtype)
        else:
            self._check_deltas(deltas)
            self.deltas = deltas

    # ------------------------------------------------------------------ building
    @classmethod
    def from_kron(
        cls,
        kron: Kron,
        lcm_layers: list[int] | None = None,
        lcm_side: str = "auto",
        order: str | list | None = "natural",
        method: str = "moments",
        fit_kwargs: dict | None = None,
    ) -> KronLCMDecomposed:
        """Fit an LCM to the large factor of each selected layer.

        Parameters
        ----------
        lcm_layers : list[int] or None
            indices into `kron.kfacs` to structure; `None` means every 2-factor layer.
            Layers left out (and all 1-factor / bias blocks) fall back to the exact
            dense eigendecomposition, so a model can mix structured and dense layers.
        lcm_side : {'auto', 'in', 'out'}
            which Kronecker factor carries the chain; 'auto' picks the larger one.
        order : {'natural', 'seriate'} or list
            1-D ordering of the chain axis. 'natural' trusts the coordinate order of the
            inputs (right when the features are physically ordered); 'seriate' recovers
            an order from the factor itself by the Fiedler vector, using no side
            information. A list gives one explicit permutation per structured layer.
        method : {'moments', 'dense', 'frobenius'}
            how to fit the factor. 'moments' is the closed-form three-band estimator
            (`LCMCovariance.fit_moments`) -- no optimiser, and the only one that *can*
            run without ever forming the factor, though this entry point reads the bands
            off the factor `Kron` has already accumulated. The other two run Adam on the
            dense objective (`fit_dense` / `fit_frobenius`).
        """
        fit_kwargs = fit_kwargs or {}
        blocks, fit_errors = [], []
        n_struct = 0
        for li, F in enumerate(kron.kfacs):
            structured = len(F) == 2 and (lcm_layers is None or li in lcm_layers)
            if not structured:
                Qs, ls = [], []
                for Hi in F:
                    if Hi.ndim > 1:
                        eigval, Q = symeig(Hi)
                    else:
                        eigval = Hi
                        Q = torch.eye(len(Hi), dtype=Hi.dtype, device=Hi.device)
                    Qs.append(Q)
                    ls.append(eigval)
                blocks.append({"kind": "dense", "Qs": Qs, "ls": ls})
                continue

            n0, n1 = F[0].shape[0], F[1].shape[0]
            side = lcm_side if lcm_side != "auto" else ("out" if n0 >= n1 else "in")
            transpose = side == "out"            # LCM on factor 0 instead of factor 1
            H_small, H_large = (F[1], F[0]) if transpose else (F[0], F[1])
            gamma, Q = symeig(H_small)

            dim = H_large.shape[0]
            if isinstance(order, (list, tuple)):
                perm = order[n_struct]
            elif order == "seriate":
                perm = seriation_order(H_large)
            elif order in (None, "natural"):
                perm = None
            else:
                raise ValueError(f"unknown order {order!r}")
            n_struct += 1

            lcm = LCMCovariance(dim, dtype=H_large.dtype).to(H_large.device)
            if method == "moments":
                idx = torch.arange(dim, device=H_large.device) if perm is None else \
                    torch.as_tensor(perm, device=H_large.device, dtype=torch.long)
                Hp = H_large[idx][:, idx]
                lcm.fit_moments(
                    moments=(torch.diagonal(Hp), torch.diagonal(Hp, 1), torch.diagonal(Hp, 2)),
                    order=idx, **fit_kwargs)
                err = None
            elif method in ("dense", "frobenius"):
                if perm is not None:
                    with torch.no_grad():
                        lcm.a.copy_(torch.argsort(torch.as_tensor(perm)).to(H_large.dtype))
                fit = lcm.fit_dense if method == "dense" else lcm.fit_frobenius
                err = fit(H_large, **fit_kwargs)
            else:
                raise ValueError(f"unknown method {method!r}")
            fit_errors.append(err)
            blocks.append({"kind": "lcm", "Q": Q, "gamma": gamma.clamp(min=0.0),
                           "lcm": lcm, "transpose": transpose,
                           "n_small": H_small.shape[0], "n_large": dim})
        obj = cls(blocks)
        obj.fit_errors = fit_errors
        return obj

    # ------------------------------------------------------------------- algebra
    def detach(self) -> KronLCMDecomposed:
        self.deltas = self.deltas.detach()
        return self

    def _check_deltas(self, deltas: torch.Tensor) -> None:
        if not isinstance(deltas, torch.Tensor):
            raise ValueError("Can only add torch.Tensor to KronLCMDecomposed.")
        if deltas.ndim == 0 or (
            deltas.ndim == 1 and (len(deltas) == 1 or len(deltas) == len(self))
        ):
            return
        raise ValueError("Invalid shape of delta added to KronLCMDecomposed.")

    def __add__(self, deltas: torch.Tensor) -> KronLCMDecomposed:
        self._check_deltas(deltas)
        return KronLCMDecomposed(self.blocks, self.deltas + deltas, self.scale)

    def __mul__(self, scalar: torch.Tensor | float) -> KronLCMDecomposed:
        """Scale the curvature. Because
        \\((s^{1/2}H_s) \\otimes (s^{1/2}H_l) = s (H_s \\otimes H_l)\\), a single
        multiplier suffices: the reduced blocks become \\(s\\gamma_j H_l + \\delta I\\)."""
        if not _is_valid_scalar(scalar):
            raise ValueError("Invalid argument, can only multiply Kron with scalar.")
        return KronLCMDecomposed(self.blocks, self.deltas, self.scale * scalar)

    def __len__(self) -> int:
        return len(self.blocks)

    def _deltas_list(self):
        if self.deltas.ndim == 0 or len(self.deltas) == 1:
            return [self.deltas.reshape(()) for _ in self.blocks]
        return list(self.deltas)

    def _gammas(self, b):
        return self.scale * b["gamma"]

    def _block_params(self, b, j, delta):
        """LCM parameters of the j-th reduced block \\(s\\gamma_j H_l + \\delta I\\)."""
        g = self._gammas(b)[j]
        prior = delta.expand(b["n_large"]) if delta.ndim else \
            delta.repeat(b["n_large"])
        return b["lcm"].scaled_params(g, prior)

    # ---------------------------------------------------------------- quantities
    def logdet(self) -> torch.Tensor:
        logdet = 0
        for b, delta in zip(self.blocks, self._deltas_list()):
            if b["kind"] == "dense":
                ls = b["ls"]
                if len(ls) == 1:
                    logdet = logdet + torch.log(ls[0] + delta).sum()
                else:
                    l1, l2 = ls
                    logdet = logdet + torch.log(
                        pow(self.scale, 1 / 2) ** 2 * torch.outer(l1, l2) + delta).sum()
            else:
                for j in range(b["n_small"]):
                    logdet = logdet + logdet_from_params(*self._block_params(b, j, delta))
        return logdet

    def _bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        assert len(W.size()) == 3
        if exponent not in (1, -1):
            raise NotImplementedError(
                "KronLCMDecomposed supports exponent 1 and -1 only; an LCM has no "
                "closed-form square root. Use KronLCMLaplace.sample (perturb-and-solve)."
            )
        B, K, P = W.size()
        W = W.reshape(B * K, P)
        cur_p, SW = 0, []
        for b, delta in zip(self.blocks, self._deltas_list()):
            if b["kind"] == "dense":
                Qs, ls = b["Qs"], b["ls"]
                if len(ls) == 1:
                    Q, eigval, p = Qs[0], ls[0], len(ls[0])
                    e = torch.pow(eigval + delta, exponent).reshape(-1, 1)
                    W_p = W[:, cur_p : cur_p + p].T
                    SW.append((Q @ (e * (Q.T @ W_p))).T)
                    cur_p += p
                else:
                    Q1, Q2 = Qs
                    l1, l2 = ls
                    p_in, p_out = len(l1), len(l2)
                    e = torch.pow(self.scale * torch.outer(l1, l2) + delta,
                                  exponent).unsqueeze(0)
                    W_p = W[:, cur_p : cur_p + p_in * p_out].reshape(B * K, p_in, p_out)
                    W_p = Q1 @ ((Q1.T @ W_p @ Q2) * e) @ Q2.T
                    SW.append(W_p.reshape(B * K, p_in * p_out))
                    cur_p += p_in * p_out
            else:
                ns, nl = b["n_small"], b["n_large"]
                W_p = W[:, cur_p : cur_p + ns * nl]
                # row-major (out, in); put the chain axis last and the small axis at -2
                W_p = W_p.reshape(B * K, nl, ns) if b["transpose"] else \
                    W_p.reshape(B * K, ns, nl)
                if b["transpose"]:
                    W_p = W_p.transpose(-1, -2)
                R = torch.einsum("js,bjl->bsl", b["Q"], W_p)     # rotate: Q^T W
                out = torch.empty_like(R)
                for j in range(ns):
                    a, w, d = self._block_params(b, j, delta)
                    if exponent == -1:
                        out[:, j, :] = kalman_solve(a, w, d, R[:, j, :])
                    else:
                        out[:, j, :] = (
                            d * R[:, j, :] + w * kernel_matvec(a, w * R[:, j, :])
                        )
                out = torch.einsum("js,bsl->bjl", b["Q"], out)   # rotate back: Q (.)
                if b["transpose"]:
                    out = out.transpose(-1, -2)
                SW.append(out.reshape(B * K, ns * nl))
                cur_p += ns * nl
        return torch.cat(SW, dim=1).reshape(B, K, P)

    def inv_square_form(self, W: torch.Tensor) -> torch.Tensor:
        SW = self._bmm(W, exponent=-1)
        return torch.bmm(W, SW.transpose(1, 2))

    def bmm(self, W: torch.Tensor, exponent: float = -1) -> torch.Tensor:
        if W.ndim == 1:
            return self._bmm(W.unsqueeze(0).unsqueeze(0), exponent).squeeze()
        elif W.ndim == 2:
            return self._bmm(W.unsqueeze(1), exponent).squeeze()
        elif W.ndim == 3:
            return self._bmm(W, exponent)
        raise ValueError("Invalid shape for W")

    def diag(self, exponent: float = 1) -> torch.Tensor:
        """Diagonal of the (inverse) precision. For `exponent=-1` this costs
        `n_large` solves per layer, i.e. \\(O(\\mathrm{in}^2)\\) -- diagnostics only."""
        n = sum(b["n_small"] * b["n_large"] if b["kind"] == "lcm"
                else (len(b["ls"][0]) * len(b["ls"][1]) if len(b["ls"]) == 2
                      else len(b["ls"][0])) for b in self.blocks)
        E = torch.eye(n, dtype=self.deltas.dtype, device=self.deltas.device)
        return torch.diagonal(self._bmm(E.unsqueeze(1), exponent).squeeze(1))

    def to_matrix(self, exponent: float = 1) -> torch.Tensor:
        """Dense form (validation only)."""
        n = sum(b["n_small"] * b["n_large"] if b["kind"] == "lcm"
                else (len(b["ls"][0]) * len(b["ls"][1]) if len(b["ls"]) == 2
                      else len(b["ls"][0])) for b in self.blocks)
        E = torch.eye(n, dtype=self.deltas.dtype, device=self.deltas.device)
        return self._bmm(E.unsqueeze(1), exponent).squeeze(1).T

    __radd__ = __add__
