# Moment matching for the Laplace-kernel covariance

A closed-form way to fit the covariance family of *"Stop Marginalizing My Dreams: Model
Inversion via Laplace Kernel for Continual Learning"* (arXiv:2605.11804, Krukowski et al.),
which that paper fits by gradient descent. Three bands of the empirical second moment identify
the parameters outright, in one pass, with no optimiser.

On flattened feature maps at the paper's own scale it beats their gradient fit by 0.47 to 1.13
nats per dimension while fitting 17.5–20× faster. **It does not replace their whole method:**
it needs the coordinate ordering supplied, which their gradient fit is meant to discover. The
sections below give the derivation, the exact relationship to their procedure, and a proposal
for combining the two.

Self-contained; no other document in this repository is needed to read it.

---

## 1. The family

For a covariance over `N` variables the paper proposes

```
Σ = diag(d) + diag(w) K(a) diag(w),        K(a)ᵢⱼ = exp(-|aᵢ - aⱼ|)
```

Each variable `i` gets a coordinate `aᵢ` on a line, an amplitude `wᵢ` and a diagonal (nugget)
term `dᵢ` — three parameters per variable rather than the `O(N²)` of a dense covariance.
Correlation falls off exponentially with coordinate distance, so variables close on the line
are strongly dependent and distant ones nearly independent.

The coordinates are **learned**, which is the paper's distinctive move: sorting by `a` induces
an ordering over the variables, so the model is meant to discover for itself which variables
belong near each other.

**Why it is cheap.** Sort the coordinates and the model is a Gauss–Markov chain: an AR(1)
latent process observed with independent noise, equivalently an Ornstein–Uhlenbeck /
Matérn-½ process in state-space form. Its precision is tridiagonal, so quadratic forms and
log-determinants cost `O(N log N)` by prefix scan, or `O(N)` by a Kalman recursion, instead of
`O(N³)`.

That chain structure is also what makes the fit below possible.

---

## 2. The estimator

Write `φₚ = exp(-(aₚ - aₚ₋₁))` for the decay between neighbours along the sorted order, so
`φ ∈ (0, 1)`. For three consecutive coordinates the model's entries are

```
Σ[p,   p+1] = w_p     w_{p+1} φ_{p+1}
Σ[p+1, p+2] = w_{p+1} w_{p+2} φ_{p+2}
Σ[p,   p+2] = w_p     w_{p+2} φ_{p+1} φ_{p+2}
```

Multiply the first two and divide by the third. Both outer amplitudes cancel, and so do both
decays, leaving one amplitude alone:

```
w²_{p+1} = Σ[p,p+1] · Σ[p+1,p+2] / Σ[p,p+2]
```

Substituting the empirical moments for the model entries identifies `w²` directly. Everything
else then follows in sequence, each step closed-form:

| Step | Formula | From |
|---|---|---|
| amplitudes | `w²_{p+1} = s1_p · s1_{p+1} / s2_p` | bands 1 and 2 |
| decays | `φₚ = s1_p / (wₚ w_{p+1})` | band 1 |
| coordinates | `a = cumsum(-log φ)` | the decays |
| nugget | `d = s0 - w²` | the diagonal |

where `s0 = E[xₚ²]`, `s1 = E[xₚxₚ₊₁]`, `s2 = E[xₚxₚ₊₂]` are the diagonal and first two
off-diagonal bands, taken **along the given order**.

**Cost.** One pass over the data for the three bands, then `O(N)` arithmetic. `O(N)` memory —
the `N × N` matrix is never formed. Nothing to tune, no learning rate, no convergence
criterion, no divergence.

**Two details the three bands force.** The endpoint amplitudes `w²[0]` and `w²[N-1]` are not
identified by three bands, so they are copied inward from their neighbours. And `w²` is clamped
to `[0, 0.999·s0]` so the implied nugget stays positive.

**Correctness.** Exact to floating point when the target genuinely lies in the family, except
at those two endpoints. Checked against a dense Cholesky factorisation at `N` = 16 384:
log-determinant agreeing to 3.1e-10, quadratic form to 3.1e-7, all eigenvalues positive.

### The whole estimator

This is the entire method, transcribed from `LCMCovariance.fit_moments` with the clamping
kept, since the degenerate cases are where a naive transcription breaks:

```python
def fit_moments(X, order=None, eps=1e-6, clip=0.999):
    """Closed-form LCM fit. X is (n_samples, dim), centred. Returns (a, w, d)
    in the ORIGINAL indexing, so the caller's variable order is preserved."""
    n, dim = X.shape
    perm = torch.arange(dim) if order is None else torch.as_tensor(order)
    P = X[:, perm]                                   # into chain order

    # the three bands — the only statistics the fit ever sees
    s0 = (P * P).mean(0)                             # diagonal
    s1 = (P[:, :-1] * P[:, 1:]).mean(0)              # band 1
    s2 = (P[:, :-2] * P[:, 2:]).mean(0)              # band 2

    tol = eps * s0.mean().clamp(min=eps)

    # amplitudes: w^2_{p+1} = s1_p * s1_{p+1} / s2_p
    w2 = torch.zeros(dim, dtype=X.dtype)
    w2[1:-1] = torch.where(s2.abs() > tol, s1[:-1] * s1[1:] / s2, torch.zeros_like(s2))
    w2[0], w2[-1] = w2[1], w2[-2]                    # endpoints unidentified; copy inward
    w2 = w2.clamp(min=0.0).minimum(clip * s0)        # keep the nugget positive
    w = w2.sqrt()                                    # <-- sign is discarded here (see §5)

    # decays from band 1, then coordinates as a cumulative sum of -log phi
    denom = (w[:-1] * w[1:]).clamp(min=tol)
    phi = (s1 / denom).clamp(min=tol, max=1.0 - 1e-12)
    a_chain = torch.cat([torch.zeros(1, dtype=X.dtype), (-phi.log()).cumsum(0)])

    d = (s0 - w2).clamp(min=tol)                     # nugget

    a, wv, dv = (torch.empty(dim, dtype=X.dtype) for _ in range(3))
    a[perm], wv[perm], dv[perm] = a_chain, w, d
    return a, wv, dv
```

That listing is runnable, not pseudo-code: executed against the library on data drawn from a
known chain it reproduces `LCMCovariance.fit_moments` exactly, to 0.0 difference on both the
coordinates and the amplitudes.

Three lines carry the whole idea: the `w2[1:-1]` ratio, the `phi` division, and the `cumsum`.
Everything else is bookkeeping and guards. Note that `a` is never initialised and never
updated — it is *computed*, which is the difference from the gradient fit discussed in §3.

In the library it is a method, used like this:

```python
from laplace.lcm import LCMCovariance

est = LCMCovariance(dim=X.shape[1], dtype=torch.float64)
est.fit_moments(X)                      # natural index order
est.fit_moments(X, order=my_order)      # or supply an ordering
nll = est.nll(X_heldout)                # O(dim) Kalman recursion, no dense matrix
```

**Provenance.** The ingredients are classical — this is a Yule-Walker-style moment estimator
for an OU/AR(1) process. What is not textbook is the *non-stationary* case: a separate
amplitude per coordinate, a separate decay per gap, and a per-coordinate nugget, where
standard treatments assume one variance and one decay. Two targeted literature searches did
not turn it up, but two searches are not a review, so **treat novelty as unverified.** What is
verified is that it is absent from the paper: that paper fits by minimising a squared Frobenius
distance, derived in its own appendix, and its released code contains no moment estimator, no
band-based fitting and no closed form of any kind.

---

## 3. How this relates to the paper's method

The paper minimises `‖Σ − Σ̂‖²_F` over `(d, w, a)` jointly with Adam — learning rate 0.01, 200
epochs in the channel configuration, 50 in the inversion configuration.

The two procedures do **not** solve the same problem:

| | The paper's gradient fit | Moment matching |
|---|---|---|
| Ordering of the variables | **learned** — `a` is free, sorting by it induces an order | **given** — an input, defaulting to the natural index |
| Coordinate spacing | learned | closed form |
| Amplitudes and nugget | learned | closed form |
| Objective | squared Frobenius distance | moment identity, no objective |
| Cost | iterative, tuned | one pass, `O(N)` |

So moment matching addresses a **strictly smaller problem**: the parameters, conditional on an
ordering. The paper's method is more ambitious and attempts both at once.

### Why comparing them on flattened feature maps is still fair

The paper initialises `a = linspace(-1, 1, dim)`, which assigns coordinate *i* in proportion to
*i* — so it already encodes the natural index order. On a flattened feature map that order is
meaningful, since neighbouring indices are adjacent pixels. Both procedures therefore start
from the same ordering information, and the difference between them is the **spacing**.

Held-out Gaussian cost of centred CIFAR-100 features, nats per dimension, lower is better, 3
seeds, dimensions up to 196 608:

| Fitting rule | par/dim | 16 384 | 65 536 | 196 608 |
|---|---|---|---|---|
| Diagonal — the paper's own baseline | 1 | 0.8372 | 0.7566 | 1.0030 |
| Factor analysis, 3 factors (larger budget) | 4 | 0.6669 | 0.6155 | 0.8425 |
| Chain, exact maximum likelihood | 3 | 0.6811 | — | — |
| Chain, the paper's Frobenius rule | 3 | 0.6000 | 0.6165 | 0.8472 |
| Chain, Frobenius with coordinates **frozen** at equal spacing | 2 | 0.6349 | 0.5969 | 0.8396 |
| **Chain, moment matching** | 3 | **0.1325** | **−0.0537** | **−0.2873** |

Paired on the same seeds, the advantage over their rule is +0.4674 ± 0.0508 at 16 384 (9
standard errors), +0.6702 ± 0.0095 at 65 536 (71), +1.1345 ± 0.0229 at 196 608 (50). Fitting
time 98 s against 1723 s at the largest width.

### What that comparison shows, and what it does not

**It shows their fitting is the weak link, not their loss.** Fitting the same family by exact
maximum likelihood — optimising precisely the quantity being scored, at 180× the cost — reaches
0.6811, slightly *worse* than their Frobenius rule. Both gradient routes stall in the same
region, so the difficulty is the optimisation landscape of this parameterisation.

**The mechanism is the coordinate span.** Expressing decay over a few neighbours at `N`
coordinates needs a span of order `N`. The published initialisation gives a span of 2 whatever
the dimension, which at `N` = 200 704 puts adjacent coordinates at correlation 0.99999 and the
most distant pair at `exp(-2)` = 0.135 — almost structureless, and roughly 400 000× away from
the solution. Moment matching does not travel there: `cumsum(-log φ)` *computes* the span.

Run both fits on a ResNet-34's four layers and inspect what each produces (`fit_report` in
`experiments/exp28_model_inversion.py`, cluster job 542802):

| Layer | Their fit: span | neighbour corr. | fitted ÷ real var. | Moments: span | neighbour corr. | fitted ÷ real var. |
|---|---|---|---|---|---|---|
| layer1 | **0.0** | 0.0002 | 3.06× | 817 775 | 0.569 | 1.000 |
| layer2 | **0.0** | 0.0000 | 6.81× | 340 766 | 0.443 | 1.000 |
| layer3 | **0.0** | 0.0000 | 16.03× | 141 889 | 0.477 | 1.000 |
| layer4 | 1.2 | 0.0002 | 0.63× | 16 476 | 0.849 | 1.000 |

With every coordinate coincident `K(a)` is all-ones, so `Σ` can only be
diagonal-plus-rank-one — and the rank-one weights vanish too, which is what the 0.0002
measures. What is left is a diagonal with variances 3 to 16× the real feature variances. The
moment fit matches those variances exactly, by construction, and imposes real neighbour
correlation of 0.44 to 0.85.

**It does not show anything about their ordering claim.** Both arms used the natural index.
Note also the frozen-coordinate row above: fixing the coordinates at equal spacing uses one
parameter less and beats *learning* them at both large widths. On this evidence the learned
ordering is not earning its keep — but that evidence is itself from gradient fits starting at
span 2, so it bounds the published procedure rather than the idea of learning an order.

---

## 4. The ordering requirement, measured

Because the bands are the chain's neighbours only when the coordinates are already sorted, a
wrong order is not a small perturbation. On data drawn from a known chain, held-out negative
log-likelihood, lower is better, 3 seeds. "Order recovery" is |correlation| between a seriated
order and the true one — 1.0 perfect, 0 chance. The diagonal column is the no-structure floor:

| dim | adjacent correlation | order recovery | true order | seriated | random | diagonal |
|---|---|---|---|---|---|---|
| 256 | 0.575 | **0.999** | 361.7 | 370.4 | 434.3 | 437.0 |
| 256 | 0.271 | 0.270 | 426.8 | 436.6 | 436.9 | 437.0 |
| 1024 | 0.575 | 0.690 | 1446.3 | 1674.5 | 1748.4 | 1750.2 |
| 2048 | 0.575 | 0.531 | 2892.3 | 3441.7 | 3497.6 | 3498.8 |
| 2048 | 0.271 | 0.019 | 3418.5 | 3499.7 | 3499.8 | 3499.8 |

- **A random order yields a diagonal**, landing within 1.3 nats of the floor everywhere. The
  implied amplitudes go to zero and the fit collapses. No graceful degradation.
- **Spectral seriation substitutes for the true order only at small dimension and strong
  coupling.** Essentially exact at 256 dimensions with adjacent correlation 0.575; recovery
  falls to 0.531 at 2048 dimensions and to 0.019 once adjacent correlation drops to 0.27, and
  the fit degrades with it.

So where the data supplies a meaningful order — a flattened feature map, a spectrum, a time
series — moment matching is immediately usable. Where the order must be discovered, as over
channels, there is no demonstrated way to supply it at the dimensions this method targets.

## 5. Where it does not work

**It emits only positive couplings.** `w = +√(w²)` discards the sign, so every off-diagonal it
produces has the sign of `wₚw_q` — a positive pattern. Harmless where neighbours are positively
correlated. Fatal over **channels**, where 47–49% of off-diagonal entries are negative: the
estimator collapses towards the identity, scoring 0.986–0.995 relative off-diagonal Frobenius
error where 1.0 means predicting no correlation at all.

The sign is in principle recoverable from the same bands at no parameter cost, since the decay
is positive and so `sign(Σ[p,p+1]) = sign(wₚw_{p+1})` propagates along the chain. **It does not
survive testing:** an early small-scale check improved a seriated fit from 0.98 to 0.88, but
across 3 seeds × 4 layers it reached 0.9897, no better than unsigned. It is kept in the
experiment code and deliberately not promoted into the library.

**It uses three bands only**, so it cannot exploit longer-range structure and has no recourse
when the target is poorly described by a chain at all.

**A better fit is not automatically better downstream.** Dropping this covariance into the
paper's own inversion loop made the generated images *worse*: correlation-profile error 0.2184
against 0.0623 for their own fit and 0.0250 for their no-covariance ablation, and Fréchet
distance 368.7 against 310.7. The reason is structural rather than a bug — minimising a
Gaussian negative log-likelihood *with respect to the data* is mode-seeking, so a correctly
correlated precision penalises differences between neighbouring coordinates and smooths the
feature map. Better density estimation, stronger smoothing prior.

---

## 6. How this could improve the published method

All five proposals below are **untested**. They are ordered cheapest first, and each says what
would falsify it. They apply to the flattened-feature configuration; see the caution at the end
about the channel configuration.

### 6a. Scale the coordinate initialisation with dimension — one line

The span of 2 is dimension-independent, and the required span grows with `N`. Replace
`linspace(-1, 1, dim)` with a span set from the data: the mean absolute band-1 correlation
`ρ₁` implies a gap of `-log ρ₁`, so initialise `a = arange(dim) · (-log ρ₁)`. This needs only
`s1`, costs one pass, and keeps their objective, optimiser and code path untouched.

*Falsified if* the fit still collapses the span, or ends no better than from `linspace(-1, 1)`.
A caution from our own data: over **channels** we tried a span of `dim` and the fit got
substantially worse, because channel covariance is broad rather than local. The right span is
whatever the data implies, not simply a large one.

### 6b. Warm-start their gradient fit from the closed form

Run moment matching, then run their Frobenius descent from the result. This keeps their
objective and their ability to refine beyond what three bands identify — longer-range
structure, the two unidentified endpoints — while starting in the region the optimiser cannot
reach on its own.

*Falsified if* their descent moves away from the moment solution and ends worse than it. That
outcome would be informative in itself: it would mean the Frobenius objective's optimum is
genuinely elsewhere, and the gap in the table above is a disagreement between objectives rather
than an optimisation failure.

### 6c. Alternate closed-form blocks with gradient steps on the ordering only

This is the proposal the structure of the problem most directly supports. The parameters split
by what each method is good at:

- Given the ordering and the decays, **amplitudes and nugget have closed-form solutions.**
- Given the ordering and the amplitudes, **the decays have a closed-form solution** from band 1.
- The **ordering** has no closed form at all — and it is precisely what their gradient fit is
  for.

So alternate:

```python
order = natural_index            # or a seriation of |corr|, or their linspace ordering
for sweep in range(max_sweeps):
    # 1. closed form: reset scale parameters to their optimum for THIS ordering.  O(N)
    a, w, d = fit_moments(X, order=order)
    model.a.data, model.w.data, model.d.data = a, w, d

    # 2. gradient steps on the coordinates ONLY, with the scales frozen.
    #    Their objective, their optimiser -- just a much smaller parameter set.
    opt = torch.optim.Adam([model.a], lr=lr)
    for _ in range(inner_steps):
        opt.zero_grad()
        frobenius_loss(model.covariance(), S_hat).backward()   # their Eq. (5)
        opt.step()

    # 3. the ordering is the discrete state; stop when it stops moving.
    new_order = torch.argsort(model.a)
    if torch.equal(new_order, order):
        break
    order = new_order
```

Step 1 keeps the scale parameters at their moment-matched optimum for the current ordering, so
the gradient in step 2 never has to fix the span — it only has to decide the permutation,
which is the one thing moments cannot give. Step 3 makes the alternation a discrete descent on
the ordering, with a natural stopping rule.

Two properties make this attractive. The expensive part of their fit — iterating on all `3N`
parameters — shrinks to iterating on `N`. And the silent failure mode disappears: the span can
no longer collapse, because step 1 resets it from the data each round.

*Falsified if* the alternation does not improve on moment matching with a fixed natural order,
or if it cycles without converging. The honest prior is mixed: Experiment 30 shows the ordering
is worth a great deal when you have it, and our gradient fits have never recovered an ordering
from an uninformed start — but all of those fits also began at the wrong span, which this
scheme removes. That confound is exactly what makes the experiment worth running.

### 6d. Use the closed-form span as a free diagnostic

Their fit's collapse is invisible in the loss curve, which falls smoothly and looks converged
while the model degenerates to a diagonal. Computing the moment-matched span costs one pass
and gives a reference: if the fitted span is orders of magnitude smaller, the fit has
collapsed. This catches the failure without changing their method at all, and is the one
proposal here with essentially no downside.

### 6e. For channels, address the sign pattern rather than the fitting

Over channels the limitation is representational, not numerical: the achievable sign pattern is
`sign(wᵢwⱼ)`, which is rank-1 — a two-camp split — while almost half the real off-diagonal
entries are negative in an overlapping pattern. A principled attempt would estimate the camp
assignment separately, for example from the sign of the leading eigenvector of the empirical
correlation, and then moment-match within camps.

*This is the most speculative proposal here*, and the evidence is against it: the simpler sign
restoration already failed at scale, and over channels a rank-2 factor model at one extra
parameter per channel beats the chain's best achievable fit by a wide margin. The honest
conclusion over channels is that fitting is not where the problem is.

### The caution that governs all of these

**Improving the fit may not improve the paper's results.** Section 5 records that a
better-fitting covariance made their inversion output measurably worse, because the fitted
covariance enters their loss as a penalty on the generated features, and a correctly
correlated penalty over-smooths. Anyone pursuing 6a–6c should therefore measure the downstream
metric, not only the fit quality, and should expect the covariance weight to need re-tuning:
their loop multiplies this term by a single fixed weight, and a better-fitted covariance hands
it a term of quite different scale and curvature.

There is a reading of our own results under which the paper's gradient fit is *accidentally*
doing something useful — collapsing to a loose, over-wide diagonal that acts as a mild
regulariser. If that is what drives their reported gains, then fixing the fit removes the
mechanism. Establishing which of those is true needs the one experiment nobody has run:
turning their covariance term off and reading the accuracy.

---

## 7. Reproducing what is quoted here

```bash
# fitting comparison at the paper's scale (cluster, ~5 h on one CPU node)
sbatch cluster/exp25.sbatch

# correctness against a dense Cholesky (laptop, seconds)
python -u experiments/exp25_verify_moments.py

# ordering dependence and seriation's reach (laptop, ~3 min on CPU)
python -u experiments/exp30_ordering_dependence.py
```

| What | Where |
|---|---|
| The estimator | `Laplace/laplace/lcm.py` — `LCMCovariance.fit_moments` |
| Fitting comparison at scale | `experiments/exp25_paper_scale_baselines.py` → `results/exp25_paper_scale.csv` |
| Correctness check | `experiments/exp25_verify_moments.py` |
| Ordering dependence | `experiments/exp30_ordering_dependence.py` → `results/exp30_ordering_dependence.csv` |
| Span collapse in the paper's own loop | `experiments/exp28_model_inversion.py` → `results/exp28_fit_diag_table.txt` |
| The signed variant that failed | `experiments/exp29_channel_correlation_fit.py` — `fit_moments_signed` |
| The paper's code, cloned for comparison | `reference/REMIX/` |

Every result file has a `.meta.json` beside it with the configuration, seeds, both git commits,
host and, for cluster runs, the job id.
