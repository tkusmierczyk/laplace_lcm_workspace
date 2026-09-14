# LCM structured-covariance Laplace approximation (`hessian_structure="lcm"`)

This branch (`lcm-covariance`) adds a **Laplace-Covariance-Model** posterior to the
Laplace library: a structured Hessian/precision approximation based on the Laplace
(exponential / Matérn-½) kernel from Krukowski et al., *"Stop Marginalizing My
Dreams: Model Inversion via Laplace Kernel for Continual Learning"*
(arXiv:2605.11804), re-purposed for Bayesian posterior inference.

## What it is

For `P` parameters, each gets a learned 1-D coordinate `a_i`; the curvature is
approximated by

```
A = diag(d) + diag(w) K(a) diag(w),     K(a)_ij = exp(-|a_i - a_j|),   d = softplus(u) + eps
```

— only `3P` parameters `(a, w, u)`. The 1-D ordering makes this a Gauss-Markov
(AR(1)) covariance whose inverse is tridiagonal, so it captures off-diagonal
curvature that a **diagonal** Laplace discards while keeping a **near-linear**
log-determinant for the marginal likelihood. Diagonal Laplace is the special case
`w = 0`; full Laplace is the dense GGN the LCM is fit to.

The fit minimises `||A - H||_F^2` to the accumulated GGN — exactly the Frobenius
objective of Appendix A of the paper.

## Usage

```python
from laplace import Laplace

# last-layer (recommended: modest P, cheap dense fit)
la = Laplace(model, "classification",
             subset_of_weights="last_layer", hessian_structure="lcm",
             lcm_steps=800, lcm_lr=3e-2)
la.fit(train_loader)
la.optimize_prior_precision(method="marglik")
probs = la(X_test)                 # GLM predictive
print(la.log_marginal_likelihood())  # uses the O(P) Kalman log-det
print(la.lcm_fit_error_)             # rel. Frobenius error of the structured fit

# all-weights variant (small models only, dense fit is O(P^2))
la = Laplace(model, "regression", subset_of_weights="all", hessian_structure="lcm")
```

Extra constructor arguments: `lcm_steps` (Adam steps for the fit), `lcm_lr`,
`lcm_eps` (diagonal floor), `lcm_init_a_scale`, `lcm_fast_logdet` (use the
near-linear Kalman log-det; validated to match the dense log-det).

The structured covariance itself is exposed as `laplace.LCMCovariance` with
standalone primitives: `dense`, `matvec`/`kernel_matvec` (O(P) via prefix sums),
`logdet` (O(P) Kalman), `sample` (O(P) AR(1)), `nll`/`quad_inv`, and both a dense
(`fit_dense`) and a matrix-free (`fit_frobenius`, the App-A objective on raw
vectors) fitting routine.

## What was added

| File | Change |
|------|--------|
| `laplace/lcm.py` | **new** — `LCMCovariance` + Kalman/prefix-sum primitives |
| `laplace/utils/enums.py` | `HessianStructure.LCM = "lcm"` |
| `laplace/baselaplace.py` | **new** `LCMLaplace(FullLaplace)` with fast Kalman log-det |
| `laplace/lllaplace.py` | **new** `LCMLLLaplace(LLLaplace, LCMLaplace)` |
| `laplace/__init__.py` | export `LCMLaplace`, `LCMLLLaplace`, `LCMCovariance` |
| `tests/test_lcm.py` | **new** — 16 unit + integration tests |
| `examples/lcm_example.py` | **new** — runnable demo |
| `laplace/LCM_PROVENANCE.md` | **new** — line-by-line map to the authors' reference `lcm.py` |

## Tests & demo

```
pytest tests/test_lcm.py -q        # 16 passed
python examples/lcm_example.py     # LCM ~4x closer to full posterior than diagonal
```

The tests validate, at machine precision, that the fast primitives match dense
references (kernel matvec, log-det, quadratic form, NLL), that the fit recovers an
exponential-correlation target and beats diagonal, that the matrix-free fit matches
the dense fit, and that `LCMLaplace`/`LCMLLLaplace` run the full pipeline
(marginal likelihood, GLM predictive, prior optimisation, sampling) with the fast
Kalman log-det matching the dense one and a lower KL to the full posterior than
diagonal.

## Scope / limitations of this version

* The `LCMLaplace` fit currently materialises the dense GGN, so it targets the
  moderate-`P` regime (last-layer or small networks). The `LCMCovariance` module
  provides the near-linear primitives and a matrix-free `fit_frobenius`; wiring
  per-sample gradients into `LCMLaplace` for a fully matrix-free large-`P` fit is
  the natural next step.
* A single 1-D ordering of all parameters is the model's main assumption; for real
  network GGNs it captures only what a single Gauss-Markov chain can (see
  `lcm_fit_error_`, which plateaus at ~0.66 rel-Frobenius). A per-layer chain
  (KFAC-style blocks) was evaluated and helps by only 12-22% in KL — still far
  short of KFAC — because the residual gap is *within*-layer Kronecker structure
  that blocking does not address. See `../experiments/EXPERIMENTS.md` Exp 6.
* The **learned** coordinate cannot discover an unknown ordering (recovery
  |Spearman| ~ 0.03) and the kernel needs |Spearman| >= 0.999 to beat diagonal.
  Learning it from an uninformed init on an unknown order is *worse* than diagonal
  (it overfits into a confidently wrong chain), whereas a fixed wrong order
  degrades gracefully to diagonal. See `../experiments/EXPERIMENTS.md` Exp 10/10b.

  There is no `learn_a` flag: `a` is an `nn.Parameter` and is always optimised by
  `fit_dense` / `fit_frobenius`. To pin a **known** order, set it and freeze it
  before fitting:

  ```python
  with torch.no_grad():
      lcm.a.copy_(known_order_coordinate)   # e.g. wavelength index, time lag
  lcm.a.requires_grad_(False)
  lcm.fit_dense(H)                          # now fits only w and d
  ```
* ~~The near-linear Kalman log-det here is a sequential O(P) scan~~ — **done.**
  The paper's associative 2x2-matrix scan is now the default (`logdet_parallel`),
  15-64x faster than the sequential version; the mean recursion was subsequently
  ported to a second scan as well (16x at d=2000, 48x at d=10000 on `quad_inv` /
  `nll`). The sequential filter survives only as `kalman_stats_sequential`, an
  independent reference the tests validate against.
* **Known weakness:** wall-clock for the *fit* scales ~`O(d^1.44)`, not
  `O(d log d)`, because the blocked `kernel_matvec` runs `range/60` sequential
  blocks and the coordinate range grows with `d`. The asymptotic work is `O(d)`;
  an associative-scan block carry would recover the intended asymptotics.
