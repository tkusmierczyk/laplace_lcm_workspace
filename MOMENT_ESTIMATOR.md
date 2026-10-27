# A closed-form fit for the Laplace-kernel covariance

The covariance family of arXiv:2605.11804 can be fitted **without an optimiser**, from three
bands of the empirical second moment, in one pass over the data. On flattened feature maps at
the paper's own scale this beats their gradient fit by 0.47 to 1.13 nats per dimension while
fitting 17.5–20× faster.

This document is the method and its boundaries. The criticism of the published method is
separate, in [`METHOD_CRITIQUE.md`](METHOD_CRITIQUE.md).

---

## Status: whose idea is this

**Not the paper's.** The paper fits by minimising a squared Frobenius distance, derived in its
appendix ("Derivation of the Frobenius-Norm Covariance Objective") and optimised by Adam. Their
repository contains no moment estimator, no band-based fitting and no closed form of any kind.

**Its ingredients are classical.** The family `Σ = diag(d) + diag(w) K(a) diag(w)` with
`K(a)ᵢⱼ = exp(-|aᵢ-aⱼ|)` is an Ornstein–Uhlenbeck / Matérn-½ process in state-space form, whose
precision is tridiagonal. Method-of-moments and Yule-Walker estimation for AR(1) is textbook,
and moment estimators for OU processes are a developed literature.

**What appears to be new is the non-stationary case.** Textbook Yule-Walker assumes one decay
and one variance. Here every coordinate has its own amplitude `wₚ`, every gap its own decay
`φₚ`, and there is a per-coordinate nugget `dₚ`. The identity below handles that, and two
targeted literature searches did not find it written down (see Sources). **Treat this as
"unverified novelty", not as established.** The AR(1), GMRF and state-space literatures are
large and this is the kind of identity that may well be folklore; a proper search is owed before
any claim is published. What *is* verified is that it is absent from the paper we are comparing
against.

---

## The method

Order the coordinates along the chain and write `φₚ = exp(-(aₚ - aₚ₋₁))` for the decay between
neighbours. For three consecutive coordinates the target entries are

```
A[p,   p+1] = w_p     w_{p+1} φ_{p+1}
A[p+1, p+2] = w_{p+1} w_{p+2} φ_{p+2}
A[p,   p+2] = w_p     w_{p+2} φ_{p+1} φ_{p+2}
```

Multiply the first two, divide by the third, and every factor cancels but one amplitude:

```
w²_{p+1} = A[p,p+1] · A[p+1,p+2] / A[p,p+2]
```

The rest follows in order, with no search:

| Quantity | From |
|---|---|
| amplitudes `w²` | the ratio above, from bands 1 and 2 |
| decays `φ` | `φₚ = A[p,p+1] / (wₚ w_{p+1})`, from band 1 |
| coordinates `a` | `a = cumsum(-log φ)` |
| diagonal `d` | `d = diag(A) - w²` |

Only the diagonal and the first two off-diagonal bands are ever needed: `s0 = E[xₚ²]`,
`s1 = E[xₚxₚ₊₁]`, `s2 = E[xₚxₚ₊₂]`. One pass over the data, `O(N·dim)` time, `O(dim)` memory.
The `dim × dim` matrix is never formed and there is nothing to tune or diverge.

Implementation: `LCMCovariance.fit_moments` in
[`Laplace/laplace/lcm.py:501`](Laplace/laplace/lcm.py). It takes either activations `X` or
precomputed `(s0, s1, s2)`, plus an `order` when the chain order is not the natural index, and
stores parameters in the original indexing.

Two details the three bands force:

- **Endpoint amplitudes are not identified** by three bands, so `w²[0]` and `w²[-1]` are copied
  inward from their neighbours.
- `w²` is clamped to `[0, 0.999·s0]` so the implied diagonal stays positive.

---

## Why it works where gradient fitting does not

The decisive quantity is the **coordinate span**. Expressing decay over a few neighbours at
`D` coordinates needs a span of order `D`. The paper initialises `a = linspace(-1, 1, dim)` — a
span of 2 whatever the dimension — which at `D` = 200 704 puts adjacent coordinates at
correlation 0.99999 and the most distant pair at `exp(-2)` = 0.135. The model starts almost
structureless, and the structured solution is roughly 400 000× further out.

The closed form does not travel there; it **computes** the span. `cumsum(-log φ)` returns
whatever width the bands imply — 817 775 at the first layer of a ResNet-34. Gradient descent,
starting at 2, instead collapses the span to 0 and lets the diagonal absorb the variance.

This is also why fixing the paper's objective does not help: fitting the same family by exact
maximum likelihood, which optimises precisely the scored quantity at 180× the cost, stalls at
0.6811 against the Frobenius rule's 0.6000. Both gradient routes stall in the same region. The
weakness is the optimisation landscape of this parameterisation, not the loss.

---

## Evidence

Held-out Gaussian cost of centred CIFAR-100 features, **nats per dimension, lower is better**,
mean ± standard error over 3 seeds, `results/exp25_paper_scale.csv`:

| Fitting rule | par/dim | 16 384 | 65 536 | 196 608 |
|---|---|---|---|---|
| Diagonal — the paper's own baseline | 1 | 0.8372 ± 0.0099 | 0.7566 ± 0.0074 | 1.0030 ± 0.0093 |
| Factor analysis, 3 factors (larger budget) | 4 | 0.6669 ± 0.0107 | 0.6155 ± 0.0163 | 0.8425 ± 0.0148 |
| Chain, exact maximum likelihood | 3 | 0.6811 ± 0.0085 | — | — |
| Chain, the authors' Frobenius rule | 3 | 0.6000 ± 0.0104 | 0.6165 ± 0.0247 | 0.8472 ± 0.0156 |
| **Chain, this closed form** | 3 | **0.1325 ± 0.0611** | **−0.0537 ± 0.0233** | **−0.2873 ± 0.0261** |

Paired against their rule on the same seeds: **+0.4674 ± 0.0508** nats/dim at 16 384 (9
standard errors), **+0.6702 ± 0.0095** at 65 536 (71), **+1.1345 ± 0.0229** at 196 608 (50).
Fitting time 98 s against 1723 s at the largest width.

Two things in that table matter beyond the headline. Their fitted chain is merely **level with
a plain factor analyser given a larger budget** at both large widths, so the chain's advantage
over low-rank alternatives appears only once it is fitted well. And **freezing the coordinates
at equal spacing** uses one fewer parameter per dimension and beats learning them (0.5969
against 0.6165; 0.8396 against 0.8472) — the learned coordinate does not earn its keep.

### Correctness checks

- **Against dense Cholesky**: log-determinant agreed to 3.1e-10, quadratic form to 3.1e-7, all
  eigenvalues positive ([`experiments/exp25_verify_moments.py`](experiments/exp25_verify_moments.py)).
- **Exact when the target is in the family**, to floating point, except the two endpoint
  amplitudes.
- **Numerically safe at the spans it produces**: the fitted spans reach ~3630 in one setting and
  817 775 in another, and an earlier bug made the scalable kernel product overflow above ~709.
  Checked directly against a dense kernel at spans 2 / 20 / 200 / 3630: max error 8.9e-16,
  nothing non-finite.

---

## Where it does not work

**It produces only positive couplings.** `w = +√(w²)` discards the sign, so every coupling it
emits is positive. That is harmless on flattened feature maps, where neighbouring positions are
almost always positively correlated. It is fatal over **channels**, where 47–49% of off-diagonal
entries are negative: the estimator collapses towards the identity, scoring 0.986–0.995 relative
Frobenius error on the off-diagonal where 1.0 means predicting no correlation at all.

The sign is in principle recoverable from the same bands at no parameter cost, since the decay
is positive and so `sign(A[p,p+1]) = sign(wₚw_{p+1})` propagates along the chain. **It does not
survive testing**: an early small-scale check improved the seriated fit from 0.98 to 0.88, but
across 3 seeds × 4 layers it reached 0.9897, no better than the unsigned version. It is kept in
[`experiments/exp29_channel_correlation_fit.py`](experiments/exp29_channel_correlation_fit.py)
as `fit_moments_signed` and **deliberately not promoted into the library**.

**It needs an ordering.** The natural flattened index works for feature maps. Where there is no
intrinsic order, spectral seriation of the absolute-correlation graph supplies one — but over
channels that is not enough to rescue it, because the problem there is the family, not the order.

**It uses three bands only**, so it cannot exploit longer-range structure, and it has no recourse
when the target is poorly described by a chain.

---

## The scoping that matters

"Fit this family by moments rather than by the published objective" is **right as density
estimation and wrong as advice for the paper's own method.**

Dropping this better-fitting covariance into the authors' inversion loop made the output
**worse**: correlation-profile error 0.2184 against 0.0250 for their no-covariance ablation and
0.0623 for their own fit, and Fréchet distance 368.7 against 310.7. The reason is structural, not
a bug — minimising a Gaussian negative log-likelihood *with respect to the data* is mode-seeking,
so a correctly correlated precision penalises differences between neighbouring coordinates and
smooths the feature map. A better density fit is a stronger smoothing prior. Details in
[`METHOD_CRITIQUE.md`](METHOD_CRITIQUE.md) and Experiment 28.

So any claim built on this estimator has to name its setting: density estimation on flattened,
genuinely chain-like features. It does not transfer to the inversion loop, and it does not
transfer to channels.

---

## Reproduce it

```bash
# fitting comparison at the paper's scale (cluster; ~5 h on one CPU node, job 541479)
sbatch cluster/exp25.sbatch

# correctness against dense Cholesky (laptop, seconds)
python -u experiments/exp25_verify_moments.py
```

| What | Where |
|---|---|
| Estimator | [`Laplace/laplace/lcm.py:501`](Laplace/laplace/lcm.py) — `LCMCovariance.fit_moments` |
| Fitting comparison | [`experiments/exp25_paper_scale_baselines.py`](experiments/exp25_paper_scale_baselines.py), [`cluster/exp25.sbatch`](cluster/exp25.sbatch) → `results/exp25_paper_scale.csv` |
| Correctness check | [`experiments/exp25_verify_moments.py`](experiments/exp25_verify_moments.py) |
| The signed variant that failed | [`experiments/exp29_channel_correlation_fit.py`](experiments/exp29_channel_correlation_fit.py) — `fit_moments_signed` |
| Full write-up and caveats | [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md) — Experiment 25 at line 1950 |
| Boundary against the authors' code | [`Laplace/laplace/LCM_PROVENANCE.md`](Laplace/laplace/LCM_PROVENANCE.md) |

## Sources consulted on prior art

Neither search returned the non-stationary, per-coordinate case; both are general
method-of-moments work on OU processes. This is two searches, not a literature review.

- [Method of Moments Estimation of Ornstein-Uhlenbeck Processes Driven by General Lévy Process](https://arxiv.org/pdf/0807.2832)
- [Estimation of Ornstein-Uhlenbeck Process Using Ultra-High-Frequency Data](https://arxiv.org/pdf/1811.09312)
- [Moment estimators for parameters of Lévy-driven Ornstein–Uhlenbeck processes](https://onlinelibrary.wiley.com/doi/10.1111/jtsa.12630)
- [The multivariate fractional Ornstein–Uhlenbeck process](https://www.sciencedirect.com/science/article/pii/S0304414925002583)
- [Adaptive estimation of autoregressive models with time-varying variances](https://www.sciencedirect.com/science/article/abs/pii/S0304407607001303)
- [Heterogeneous autoregressions in short T panel data models](https://onlinelibrary.wiley.com/doi/full/10.1002/jae.3085)
- [Non-parametric estimation of time varying AR(1) processes with local stationarity and periodicity](https://www.researchgate.net/publication/317230552_Non-parametric_estimation_of_time_varying_AR1-processes_with_local_stationarity_and_periodicity)
