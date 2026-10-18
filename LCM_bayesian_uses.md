# Uses of the LCM "effective Gaussian representation" in Bayesian learning

Analysis of the structured-covariance representation from *"Stop Marginalizing My Dreams: Model
Inversion via Laplace Kernel for Continual Learning"* (arXiv:2605.11804v1, Krukowski et al.),
targeting Bayesian problems with **≥10k correlated variables**.

---

## 1. What the representation actually is (precise characterization)

The paper's Laplace Covariance Model (LCM) parameterizes a full-covariance Gaussian over `C`
variables with only `3C` numbers `(a, w, d)`:

```
Σ = diag(d) + diag(w) · K(a) · diag(w),      K(a)_ij = exp(−|a_i − a_j|)
```

- `a ∈ ℝ^C` — a **learned 1-D coordinate** for each variable (this is the crux).
- `w ∈ ℝ^C` — per-variable scale (heteroscedastic std of the correlated part).
- `d = softplus(u)+ε > 0` — an independent diagonal "nugget"/noise term.

**The load-bearing fact (Appendix C).** On the *sorted* coordinates, the Laplace kernel is
exactly an AR(1) / Ornstein–Uhlenbeck process:

```
x_1 = ε_1,                  x_i = ρ_{i−1} x_{i−1} + ε_i,   ρ_{i−1} = exp(−(a_i − a_{i−1}))
```

so the correlation matrix factors as `K = L⁻¹ D L⁻ᵀ` with `L` lower-**bidiagonal**, and the
**precision matrix `Q = Σ⁻¹` is strictly tridiagonal** — a bandwidth-1 Gaussian Markov Random
Field (GMRF). Each variable is conditionally independent of all others given its two neighbours
in the learned ordering.

**This is a known object, which is exactly why it is trustworthy.** `exp(−|·|)` is the
exponential / **Matérn-½** covariance = the OU kernel. A 1-D GP with a Matérn-half-integer kernel
is Markovian and admits an exact linear-time state-space (Kalman) solution
(Hartikainen & Särkkä 2010; Särkkä & Solin 2019). The Matérn↔sparse-precision link is the SPDE /
GMRF result of Lindgren, Rue & Lindström (2011), the computational backbone of INLA (Rue & Held
2005). So:

> **LCM = a Matérn-½ Gaussian process placed on a *learned* 1-D latent embedding of the variables,
> with per-variable scaling and a diagonal nugget.**

The genuinely new twist for Bayesian ML is not the Markov trick — it is that the 1-D coordinates
`a` are **learned** so that *arbitrary, non-temporal* correlated variables (network weights,
feature channels, coefficients) get embedded into a chain where correlation decays exponentially
with distance. That turns "scalable GP inference on a time axis" into a general-purpose,
`O(N log N)` structured-covariance *primitive*.

---

## 2. The affordances — the operations it makes cheap

Knowing it is a state-space Gaussian tells us its full "API", each in `O(N)` memory and
`O(N log N)` time (`log N` only from the one-time sort):

| Operation | Cost | Method |
|---|---|---|
| Draw a sample `x ~ N(μ,Σ)` (reparameterizable) | `O(N)` | AR(1) forward recursion `z_k = φ_k z_{k−1} + ε_k` |
| Log-density / exact NLL `log N(x;μ,Σ)` | `O(N log N)` | Kalman filter → sum of 1-D innovations `v_k ~ N(0,S_k)` |
| Log-determinant `log|Σ|` (⇒ Gaussian entropy) | `O(N)` | `Σ log S_k` from the filter / `D` factors |
| Quadratic form `xᵀΣ⁻¹x`, `xᵀK x` | `O(N log N)` | prefix/suffix sums (App. B) |
| Precision- and covariance-vector products | `O(N)` | bidiagonal `L`, or forward/backward sweeps |
| Condition / marginalize / smooth | `O(N)` | Kalman/RTS smoother |
| Fit to an empirical `Σ̂` | `O(N log N)` | Frobenius objective, no matrix ever formed |

Every quantity a Gaussian Bayesian method needs — sample, density, entropy/log-det, quadratic
form, conditionals — is available at near-linear cost, and all are differentiable in `(a,w,d)`.
That is precisely the bottleneck set that blocks full-covariance Gaussians at `N ≥ 10k`
(dense = `O(N²)` memory, `O(N³)` inverse/log-det). The paper's own headline number: a `376,320`-dim
feature covariance drops from **199 GB → 5.7 MB** (35,541×), while keeping *exact* likelihoods.

---

## 3. Ranked candidate uses in Bayesian learning

### (1) Structured-covariance Laplace approximation for BNN posteriors — best fit
The Laplace approximation fits `N(θ_MAP, H⁻¹)` over network weights; the `P×P` Hessian/GGN/Fisher
is the wall. The field lives between two poles: **diagonal** (throws away all weight correlations)
and **KFAC** (Kronecker block structure, heavier). LCM offers a new, genuinely cheap point on that
axis: a **tridiagonal-precision (GMRF) Laplace posterior** with `3P` parameters and `O(P log P)`
predictive sampling and log-det. Concretely, fit `a,w,d` so `diag(d)+diag(w)K diag(w) ≈ H⁻¹`
(or fit the precision directly to `H`, which is *already* sparse for many layers), then draw
posterior weight samples by the AR(1) recursion for MC predictives.
- **Why it fits:** captures *some* off-diagonal weight coupling that diagonal Laplace ignores —
  the paper's whole thesis (diagonal marginalizes real correlations) is the same critique
  Bayesian-deep-learning makes of mean-field.
- **Payoff:** better-calibrated uncertainty / marginal likelihood at diagonal-like cost; a natural
  ablation against diag and KFAC on NLL / ECE / OOD.
- **This is the connection your folder name (`laplace_bayes`) points at:** Laplace *kernel* →
  Laplace *approximation*. Worth making explicit as the lead experiment.

### (2) Structured Gaussian variational posteriors (sparse-precision VI family)
Mean-field Gaussian VI (`q = N(m, diag)`) is the workhorse and ignores posterior correlations;
full-rank is `O(P²)`. LCM is a ready-made **structured variational family** with `O(P)` params:
- ELBO needs exactly `E_q[log p]`, entropy `½log|Σ|`, and reparameterized samples — all cheap here.
- The AR(1) generative form *is* a differentiable reparameterization, so gradients flow to `(a,w,d)`.
- This is a concrete, PD-by-construction instance of **sparse-precision Gaussian VI**
  (cf. Tan & Nott 2018), but with `3P` params and a learned chain instead of a fixed sparsity graph.
- **Payoff:** correlation-aware VI posteriors at mean-field cost for `≥10k`-parameter models
  (last-layer BNNs, Bayesian LoRA adapters, GLMs, deep-kernel heads).

### (3) Scalable GP regression / latent Gaussian models via a *learned-coordinate* kernel
This is the representation's native habitat. For a 1-D input, LCM *is* an exact `O(n)` GP. The new
lever is learning `a`: embed high-dim or non-Euclidean inputs (or the `n` data points themselves)
into a 1-D chain so a Matérn-½ prior becomes Markov, then do exact Kalman-filter inference at
`n ≥ 10k`–`10⁶`. Use as the latent-Gaussian layer in an INLA-style latent Gaussian model, or as a
cheap GMRF prior term.
- **Caveat:** you are committing to a Matérn-½ (rough, single-length-scale) prior and to whatever
  1-D embedding you can learn — see limitations.

### (4) GMRF priors for high-dimensional Bayesian regression
A tridiagonal-precision prior over **ordered** coefficients is a strong, cheap prior wherever an
ordering with exponentially-decaying dependence exists:
- **Genomics / GWAS (≥10k–10⁶ SNPs):** linkage disequilibrium decays roughly with genomic
  distance — order SNPs along the genome and the Laplace-kernel prior is well-motivated and gives a
  banded precision (near-linear posterior).
- **Time series / functional regression / spectra:** neighbouring coefficients correlated;
  bandwidth-1 GMRF = discretized OU smoothness prior.
- **Payoff:** correlated shrinkage that mean-field/ridge cannot express, at linear cost, PD by
  construction.

### (5) Correlated joint predictive / output covariance for UQ
Many tasks need a *joint* Gaussian over `≥10k` outputs, not `10k` independent marginals: joint
predictive over a test set, batch/look-ahead **Bayesian optimization** and **active learning**
(need `Cov` between candidate points), calibrated **conformal/UQ** with dependent scores. LCM gives
a cheap correlated joint (log-det for information gain, `O(N)` joint sampling, conditionals for
sequential selection) where a diagonal predictive would badly under/over-count joint uncertainty.

### (6) Preconditioning & correlated proposals for SG-MCMC / Langevin
A tridiagonal precision is a cheap, PD, correlation-aware **preconditioner** (mass matrix) for
SGLD/HMC over `≥10k` parameters: `Σ⁻¹`- and `Σ^{½}`-vector products are `O(N)`, so momentum
refresh and preconditioned drift stay linear while injecting non-diagonal geometry that diagonal
preconditioners miss.

### (7) Learned-ordering Bayesian filtering / structured SSMs
Inverse direction: because LCM *is* a state-space model, it slots into Bayesian state-space /
deep-SSM inference directly — but with the ordering over "channels" **learned** rather than given
by a time axis, useful for exchangeable-but-correlated latent sets.

---

## 4. The limitations — read before over-committing

The representation is powerful *and* narrow. Be honest about the family it can express:

1. **Bandwidth-1, single-chain conditional independence.** The precision is tridiagonal: every
   variable talks to only two neighbours in **one** learned ordering. Genuinely high-dimensional
   correlation graphs (a 2-D feature map's neighbour structure, a dense NN Hessian, a multi-output
   GP) *cannot* be embedded into a single 1-D Markov chain without distortion — this is the graph-
   bandwidth/embedding problem. Expect the fit to be good only when a good 1-D ordering exists.
2. **Nearly-positive, rank-1-sign correlations, single length-scale.** `K` itself is
   elementwise-positive; the only route to negative correlations is the sign pattern of `w`
   (`sign(Corr_ij) = sign(w_i w_j)`, a rank-1 structure). Correlation decays at one exponential
   rate. Anti-correlation-rich or multi-scale structure is out of family.
3. **Matérn-½ roughness.** `exp(−|·|)` gives non-differentiable (very rough) sample paths. If the
   true dependence is smooth, this under-fits.
4. **Learning `a` is the whole game and is non-convex.** The sort makes the objective piecewise;
   quality hinges entirely on finding a good embedding. The paper's own ViT result (diagonal *beats*
   LCM in early, weakly-correlated layers) is the honest reminder: **structured covariance only pays
   off when the variables are actually correlated in a 1-D-embeddable way.**

---

## 5. Natural extensions that keep the near-linear scaling

- **Higher-order Matérn (`ν = 3/2, 5/2`) → banded (penta-…) precision.** Still `O(N)`, adds
  smoothness and a second length scale — cheap expressivity win.
- **Low-rank + LCM:** `Σ = UUᵀ + LCM` captures a few global factors *plus* local chain structure
  (mirrors low-rank+diagonal Laplace/SWAG, strictly more expressive at `O(Nk)`).
- **Multiple orderings / block-LCM / tree GMRFs:** sum or block several chains to escape the single-
  ordering bottleneck while staying in sparse-precision linear algebra.
- **Amortization:** predict `(a,w,d)` from context for fast structured posteriors across tasks
  (natural in the continual-learning setting the paper already targets).

---

## 6. Suggested first experiment

Lead with the pun that is also the strongest fit: a **tridiagonal-precision Laplace approximation**.
On a mid-size BNN (or just the last layer / a LoRA adapter, `P` from `10⁴`–`10⁶`), fit `(a,w,d)` to
the GGN/Fisher, draw predictive samples via the AR(1) recursion, and benchmark **NLL, ECE, and OOD
detection against diagonal and KFAC Laplace**. If off-diagonal weight structure helps calibration
at diagonal cost, that is a clean, self-contained result — and it directly instantiates "moving
beyond diagonal assumptions" (the paper's thesis) in the Bayesian-inference setting rather than the
model-inversion one.
