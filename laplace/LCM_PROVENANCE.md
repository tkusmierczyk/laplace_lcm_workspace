# `laplace/lcm.py` — provenance and diff against the reference implementation

**Reference:** `efficient_inversion/resnet34/lcm.py` from
[REMIX-Model-Inversion-via-Laplace-Kernel](https://github.com/pkrukowski1/REMIX-Model-Inversion-via-Laplace-Kernel)
(Krukowski et al., arXiv:2605.11804) — 113 lines, 81 of them executable.

**This file:** ~600 lines, ~300 executable. This document explains every difference,
so the port stays auditable.

**Re-verified 2026-10-19** against a fresh clone of that repository (kept outside git at
`reference/REMIX` in the experiments repo, since it is third-party code). Confirmed by
inspection: the parameterisation is identical (`u_raw` zeros, `w = 0.1`, `a = linspace(-1, 1, dim)`),
so is the `exp(-|a_i - a_j|)` kernel, and so is the innovations filter — a 2x2 matrix
parallel-prefix product with per-step max-abs renormalisation, `S = w^2 P_pred + u`,
`K = P_pred w / S`, `alpha = exp(-dt)(1 - K w)`, `v = diff - w phi m_prev`, summed over
dimensions and averaged over the batch. The three claims this document already made about
the reference also hold: `lcm.py` contains no fitting, no sampling and no standalone
log-determinant, and its `laplace_kernel()` is never called.

**The repository ships four copies of `lcm.py`**, and they are not the same model:
`efficient_inversion/{resnet34,vit}/lcm.py` are the `O(d)` filter above (differing from
each other by 10 lines), while `clip/inversion/lcm.py` and `resnet32/inversion/lcm.py` are
36-line **dense** variants exposing `covariance()`, `correlation()` and `forward()`, with
no filter at all. So the paper's continual-learning results (Tables 1-2) are produced by an
`O(d^2)` dense covariance, and only the feature-likelihood results (Figure 4) and the
inversion loss use the scalable path this file ports.

**The two uses model different variables, which decides what any comparison is about.**
Traced 2026-10-19 through `resnet32/continual_runner/task_contrastive_cl_runner.py`
(the entry point behind the continual-learning tables) and `resnet32/utils.py`:

| | Feature-likelihood and inversion loss | Continual-learning tables |
|---|---|---|
| File | `efficient_inversion/{resnet34,vit}/lcm.py` | `resnet32/inversion/lcm.py` (dense) |
| Variables | the **flattened feature map**, e.g. `64 x 56 x 56 = 200 704` for one ResNet-34 layer | **channels only**: `LCM(m.num_features)` over each `BatchNorm2d`, so `16`-`512` |
| Dimension | `~10^5`, which is why the `O(d)` filter exists | `~10^2`, where a dense covariance is free |
| Target fitted | the features' own likelihood, through the filter | the empirical **correlation** matrix `R`, accumulated channel-wise by a streaming Welford update over the training set |
| Fitting rule | gradient descent on `frobenius_dist_reduced` | `fit_lcm_correlation`: mean-squared error between `correlation()` and `R`, Adam, `lr = 0.01`, 200 epochs |
| Across tasks | n/a | `R` is blended as `ratio_old * R_old + ratio_new * R_curr` by class counts, then the fit is re-run and the result frozen |
| Switch | always on | skipped entirely when `alpha_frob = 0` |

Two consequences for how our results should be claimed. First, the comparison between
fitting rules at `d` up to `~2 x 10^5` belongs to the **left** column, which is the setting
the scalable filter was built for and the only one where fitting cost matters. Second, the
published **accuracy** gains come from the **right** column, where the covariance is over
channels: there is no spatial neighbour relation there, so the learned coordinate `a` is
ordering channels, which is exactly the case where this project found order to be
recoverable by seriation but not learnable by gradient descent. A claim about the accuracy
tables has to be tested in that configuration, not this one.

---

## 1. Why the sizes differ

The reference file computes **one quantity**: the Gaussian log-likelihood of features
under the LCM covariance, to be used as a loss term inside a model-inversion loop.
Verified by inspection, it contains **no** fitting code, **no** sampling, **no**
prefix/suffix sums (Appendix B), and **no** standalone log-determinant; its
`laplace_kernel()` is defined but never called.

This file implements a **posterior** for the Laplace approximation, which needs the
log-determinant, quadratic forms, and sampling as *separate* callables, plus the
ability to fit the model to a given curvature matrix. Roughly a third of the file is
docstrings (the reference has one comment line).

## 2. Direct ports — semantically equivalent to the reference

| here | reference | notes |
|---|---|---|
| `_parallel_prefix_2x2` | `LCM._parallel_matrix_prefix_prod` (L22–41) | Hillis–Steele scan over 2×2 matrices with per-step max-abs rescaling. Identical algorithm. |
| `kalman_stats`, scan 1 | `LCM.log_likelihood` L55–82 | variance / Riccati recursion → innovation variances `S`. Matrix entries built exactly as the reference. |
| `kalman_stats`, scan 2 | `LCM.log_likelihood` L84–107 | mean recursion → innovations `v`, via the affine-composition matrices `[[α, β], [0, 1]]`. |
| `LCMCovariance.nll` | `LCM.nll` / `log_likelihood` L108–113 | same per-element Gaussian NLL, `−½(log 2π + log S + v²/S)`. |
| `LCMCovariance.kernel` | `LCM.laplace_kernel` (L19–20) | dense `exp(−|aᵢ−aⱼ|)`. Unused in the reference; here it backs `dense()` and the validation paths. |
| parameterisation `(a, w, u_raw)` | `LCM.__init__` | identical; `d = softplus(u_raw) + eps`. |

**Deliberate deviations inside the ports:**

1. **dtype.** Reference is float32 and casts to float64 only inside the scan; this
   file is float64 throughout (a posterior log-determinant feeds a marginal
   likelihood, so precision matters more than inversion-loop speed).
2. **Nugget `eps`.** The reference's `__init__` defaults to `1e-6` but
   `log_likelihood` hardcodes `+ 1e-5` — an internal inconsistency. Here `eps` is a
   single constructor argument used everywhere (default `1e-6`). Set `eps=1e-5` to
   reproduce the reference's likelihood bit-for-bit.
3. **Return type.** The reference returns a scalar mean likelihood. `kalman_stats`
   returns `(S, v)` so the same filter can serve `logdet`, `quad_inv`, and `nll`
   without recomputation.
4. **Clamps.** Reference clamps `P_filt ≥ 0`, `S ≥ 1e-6`, denominators `≥ 1e-12`.
   Kept, with the `S` floor at `1e-12` (float64 headroom) and denominators at
   `1e-30`.
5. **Degenerate case.** `dim == 1` is handled explicitly (the reference would build
   an empty `dt`).

## 3. Additions the reference does not contain

### 3a. Because a posterior needs more than a likelihood
| function | why it exists |
|---|---|
| `logdet_from_params`, `logdet_parallel`, `LCMCovariance.logdet` | the Laplace **marginal likelihood** needs `log|Σ|` on its own; in the reference it only ever appears inside the NLL |
| `quad_inv` | **predictive variance** needs `xᵀΣ⁻¹x` on its own |
| `sample` | **posterior sampling** for the MC predictive, via the AR(1) recursion. The reference never samples |
| `scaled_params` | the posterior precision `h·A + diag(prior)` stays in the LCM family (same `a`, scaled `w`, shifted `d`). The reference has no prior |
| `matvec` | `A x` products |

### 3b. Paper appendices absent from the reference file
| function | implements |
|---|---|
| `fit_frobenius` | **Appendix A** — the matrix-free Frobenius objective. **Correction (2026-10-19):** this row previously said the reference "fits implicitly by backpropagating its inversion loss". That is wrong. The reference *does* fit explicitly, just not in `lcm.py`: `efficient_inversion/resnet34/utils.py:frobenius_dist_reduced(u, a, w, V)`, driven by Adam in `synthetic_samples.py:138-143` as `frob / D**2`. Our objective is **term-for-term identical** — `sum(u**2)` ↔ `(d**2).sum()`, `xAx_sum(2a, w**2)` ↔ `(w2 * kernel_matvec(2a, w2)).sum()`, `2*sum(u*w**2)` ↔ `2*(d*w2).sum()`, `-(2/B)*sum(u*sum(V**2))` ↔ `-2*(d*(V**2).mean(0)).sum()`, `-(2/B)*xAx_sum(a, V*w)` ↔ `-2*(wv*kernel_matvec(a, wv)).sum(-1).mean()`. Their `frobenius_dist` adds the constant `sum(((V.T@V)/B)**2) = ||S_hat||_F^2`, which we drop (see the `fit_frobenius` docstring). Two deviations: their `xAx_sum` wraps the quadratic form in `torch.abs()`, guarding the sign numerically; and they evaluate `K(a)x` with a complex-arithmetic prefix scan where we use blocked rescaling (§3c) |
| `kernel_matvec`, `_decaying_prefix` | **Appendix B** — the `O(d)` prefix/suffix-sum evaluation of `K(a)x`. Not present in the reference, which only has the dense kernel |
| `fit_dense` | dense counterpart of the above, used for validation and the moderate-`d` regime |

### 3c. Numerical hardening (fixes for silent failures)
| function | issue fixed |
|---|---|
| `_inverse_softplus` | naive `log(expm1(x))` overflows to inf/NaN for moderately large diagonal scales, making Adam diverge silently |
| `_decaying_prefix` blocked rescaling | the offset-based prefix sum overflows once the **coordinate range** exceeds ~709 (float64), returning NaN — this affects any implementation of Appendix B, not just this one |
| `_auto_init_scale` | fit initialisation must be scaled to the target, or the optimiser diverges rather than converging slowly |

### 3d. Validation-only
`dense()`, `logdet_dense()`, `kalman_stats_sequential` exist purely so the fast paths
can be checked against ground truth. `kalman_stats_sequential` is a deliberately
independent, textbook implementation sharing **no code** with the scans — so the two
agreeing is a genuine cross-check rather than a tautology. The reference ships no
tests.

## 4. Equivalence evidence

Reproduce with `pytest tests/test_lcm.py` (16 tests) and the audit scripts in
`experiments/audit/`.

| check | result |
|---|---|
| our NLL vs the reference's `log_likelihood`, identical params, `eps=1e-5` | agree; ours matches the dense `MultivariateNormal` to **0.00e+00**, the reference to 4.4e-07 (it is float32) |
| scan filter vs sequential reference (`d` = 1, 2, 17, 128, 600) | `S` to 1e-16, `v` to 7e-16 |
| scan filter vs dense truth | log-det to 7e-15, quadratic form to 2e-13 |
| `kernel_matvec` vs dense kernel | ~1e-14, finite to coordinate range 1e4 |
| LCM with `w = 0` vs `DiagLaplace` | identical log-det and marginal likelihood; predictions to 8.9e-08 |
| gradients through both scans | finite |

## 5. Performance note

Porting the reference's *second* scan (the mean recursion) replaced a sequential
Python loop and sped up `quad_inv` / `nll` by **16× at d=2000 and 48× at d=10000**.
The reference was better-optimised here than this file previously was; the loop
survives only as `kalman_stats_sequential`, for validation.
