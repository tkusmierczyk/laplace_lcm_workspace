# LCM for Bayesian learning — end-to-end project report

**Scope.** This report documents, end to end, an investigation into whether the
Laplace Covariance Model (LCM) — the structured covariance introduced for model
inversion in *"Stop Marginalizing My Dreams"* (arXiv:2605.11804, Krukowski et al.)
— is useful for **Bayesian learning with large numbers of correlated variables**,
and whether that use is strong enough to publish. It records what was done, why
each choice was made, what the results were, the design decisions and their
impact, and the limitations/issues encountered (including bugs found and fixed).

**Bottom line** *(revised after Exp 16 — see the correction below)*. LCM is a
correct, cheap, near-linear structured Gaussian. Its one genuine advantage over
diagonal Laplace is in **joint** (not marginal) predictive uncertainty, in a
regime where the variables have a 1-D ordering with decaying correlation and the
problem is well-enough determined that a joint-covariance gap exists (`n ≳ d`).

Its headline novelty — the **learned** 1-D coordinate — still does not survive:
gradient descent from an uninformed init cannot discover an unknown ordering
(recovery 0.028). **But the stronger negative originally reported here — that the
method is unusable without a *known* order — was a fitting artifact and is
withdrawn.** With a scale-normalised fit and a spectral-seriation initialisation
that is then refined, LCM closes **65%** of the diagonal→full joint gap on data
whose ordering is entirely hidden, against an oracle-order ceiling of 90%.

Two further results sharpen the case. Against its own `O(d)` complexity class
(banded Cholesky, low-rank) LCM wins decisively *in its niche* — and for the right
reason: a low-rank model can match its Frobenius fit to the precision and still
close only a third as much predictive gap, because the posterior covariance here is
near-banded (Exp 14b). On **neural-network** weight curvature the verdict reverses:
low-rank beats LCM per parameter and carries no ordering fragility, which is an
independent argument against weight-space Laplace.

On real known-order data (NIR spectroscopy) LCM recovers essentially all of full
Laplace's joint-UQ advantage at `3d` parameters, though there it is equivalent to a
classical banded / state-space-GP precision. Recommended outcome: a focused
**workshop → TMLR** paper. The framing should now be revisited upward — the
"requires a known order" limitation that motivated the characterisation-only
framing no longer holds. Full verdict in `PUBLICATION_ASSESSMENT.md`, which also
predates the correction.

---

> ## ⚠ CORRECTION (Exp 16) — read before the ordering sections
>
> The **unnormalised Frobenius fit**, not the model, produced this project's central
> ordering negatives. Scale-normalising the fit target (`T/s`, rescale after — exact,
> since the LCM family is positively homogeneous) changes the verdicts on *identical
> data and seeds*:
>
> | claim | as reported here | corrected (Exp 16) |
> |---|---|---|
> | learned coordinate, unknown order | −84% (worse than diagonal) | **+10.2% ± 6.9** |
> | spectral seriation, unknown order | −79% | **+65.4% ± 19.7** (as *init + refine*, not frozen) |
> | headline joint gap, d=1000 | 52% | **87%** (identical seeds) |
>
> Order *recovery* numbers are unchanged (0.028 learned, ~0.79 seriation) — only the
> fit was broken. **"LCM requires a known 1-D order" is therefore withdrawn**; what
> stands is the narrower "gradient descent alone cannot *discover* one". The
> \|Spearman\| ≥ 0.999 threshold is also unreliable: it saturates at 4 d.p. across a
> 33-point swing in outcome. See `experiments/EXPERIMENTS.md` Exp 16.
>
> Everything below produced by an unnormalised fit is **conservative by an unknown
> margin** and has not yet been re-run.

## 1. High-level overview (the arc)

The work proceeded in six phases, each driven by an explicit goal:

1. **Scoping** — survey Bayesian-learning uses of the LCM representation for
   ≥10k correlated variables; select the most promising (`LCM_bayesian_uses.md`).
2. **Focus + theory** — work out the math linking LCM to the Laplace approximation
   (`LCM_laplace_approximation.md`, `LCM_laplace_math_stepbystep.md`).
3. **Development** — fork the `laplace-torch` library and add LCM as a first-class
   covariance backend (`hessian_structure="lcm"`), with tests.
4. **Evaluation** — benchmark LCM against diagonal / KFAC / full Laplace on
   fidelity, regression UQ, image UQ/OOD, and cost (`EXPERIMENTS.md`).
5. **Search for a compelling use** — after the initial evaluation was
   underwhelming, systematically hunt for a regime where LCM wins
   (`COMPELLING_USE.md`), following three leads: per-layer structure, epistemic
   metrics, and the "correct habitat" (1-D-correlated GLMs).
6. **Publication hardening** — close the reviewer-critical gaps: matrix-free fit,
   the ordering baselines, real data, and statistical CIs; write the honest
   assessment (`PUBLICATION_ASSESSMENT.md`, this report).

Selected artifacts: a fork of `aleximmer/Laplace` on branch `lcm-covariance`
(commits `0225c71 → 32a0131 → 9493eee → db6a554`); 14 passing unit/integration
tests; ~13 experiment scripts + result CSVs/figures; and the documents above.

---

## 2. What LCM is (the object under study)

LCM parameterizes a covariance over `d` variables with only `3d` numbers `(a,w,u)`:

```
A = diag(d) + diag(w) K(a) diag(w),   K(a)_ij = exp(-|a_i - a_j|),   d = softplus(u)+eps
```

Each variable `i` gets a scalar **learned coordinate** `a_i`; correlations decay
exponentially with coordinate distance. The key mathematical identity (derived in
`LCM_laplace_math_stepbystep.md`) is that this is a **Matérn-½ / Ornstein–Uhlenbeck
Gaussian process on a 1-D line**, equivalently a **Gauss–Markov (AR(1)) chain**:
its inverse (precision) is tridiagonal, and every operation — sampling, log-density,
log-determinant, quadratic forms — is exact in `O(d)`–`O(d log d)` via the
associated state-space model (Kalman filter). It generalizes diagonal covariance
(the `w=0` case) and is strictly cheaper than dense (`d²`). It is **not**
uniformly cheaper than KFAC: KFAC stores `Σ_l (in_l² + out_l²)`, which beats LCM's
`3P` on square-ish layers and loses badly on rectangular ones (Exp 12 §7). And on
*construction* time LCM is far more expensive than KFAC, because it needs an
iterative fit where KFAC is closed-form.

**Why this looked promising for Bayesian learning:** a Bayesian posterior over many
correlated variables needs exactly those operations (sample, log-det for evidence,
quadratic forms for predictive variance), and dense is intractable at ≥10k. LCM
offered full-covariance-style structure at diagonal-like cost.

---

## 3. Development: the library integration

We added LCM to `laplace-torch` rather than building standalone, so it inherits a
maintained, tested predictive/marginal-likelihood pipeline and directly competes
with the standard baselines (diag/KFAC/full) under one interface.

- `laplace/lcm.py` — `LCMCovariance` module: dense build (validation), `O(d)`
  prefix-sum kernel matvec, sequential + **associative-scan** Kalman log-det, AR(1)
  sampler, exact NLL, and two fitters (`fit_dense`, matrix-free `fit_frobenius`).
- `LCMLaplace(FullLaplace)` and `LCMLLLaplace(LLLaplace, LCMLaplace)` classes,
  registered as `hessian_structure="lcm"`.
- 14 tests: every fast primitive is checked against a dense reference at machine
  precision; the full pipeline (fit → marginal likelihood → GLM predictive →
  prior optimization → sampling) is exercised for all-weights and last-layer.

Design decisions and the bugs found here are in §6.

---

## 4. Evaluation (Experiments 1–4)

The initial benchmark against MAP / Diag / KFAC / Full Laplace.

| Exp | Question | Result |
|---|---|---|
| 1 | Posterior fidelity (KL to full) | LCM **1.6–2.9× lower KL than diagonal**; KFAC best. Gap over diagonal grows with curvature correlation. |
| 2 | UCI regression test-LL (6 sets) | Avg rank Full < KFAC < **LCM ≈ Diag** < MAP. LCM ties diagonal on average; rescues diagonal's collapse on `yacht`. |
| 3/3b | MNIST/CIFAR last-layer NLL/ECE/OOD | LCM **ties or loses** to diagonal on calibration; KFAC/Full best. |
| 4 | Cost / scaling / ablation | 3d vs d² storage; exact log-det where dense OOMs. Ablation: LCM recovers a *controlled* chain to 1e-4 but a *real* NN GGN only to ~0.66 rel-Frobenius — generic NN curvature is not chain-embeddable. |

**Conclusion of Phase 4:** LCM improves *fidelity* over diagonal but this did not
translate into better *predictive* uncertainty for neural-network weights, because
NN-layer curvature is Kronecker-structured (KFAC's home), not chain-structured.

---

## 5. Search for a compelling use (Experiments 5–11)

| Exp | Lead | Result |
|---|---|---|
| 5 | Epistemic OOD (mutual information / BALD instead of max-confidence) | Correctly exposes epistemic uncertainty (MAP→0.50 AUROC), KFAC/Full 0.72, but **LCM 0.62 < Diag 0.64**. Does not rescue LCM. |
| 6 | Per-layer (block-diagonal) LCM | KL to full improves 12–22% vs a single chain; still short of KFAC. |
| 7 | 1-D-correlated GLM (LCM's "home") | The reframing: LCM's structure is only *correct* for genuinely 1-D-correlated, non-Kronecker curvature (GLMs with ordered features). |
| 8/8b/8c | **Marginal vs joint predictive LL** | **The key positive, but modest.** Marginal LL: LCM ties (slightly worse than) diagonal — insensitive to off-diagonal covariance. **Joint LL: LCM closes a d-dependent fraction of the diagonal→full gap — 10-seed CI 62%±49% (d=500), 52%±16% (d=1000), 23%±12% (d=1500)** (§7). |
| 10/10b + controls | **Does the *learned* ordering matter?** | Given the correct order LCM works (+58% joint-gap closed, learned-but-correctly-initialised +59%). With an unknown/permuted order, gradient learning recovers \|Spearman\| = **0.028** (≈ random) and lands at **−84%** (*worse than diagonal*); seriation recovery is highly seed-dependent (**0.31 / 0.70 / 0.99**, mean 0.67) and its variant scores −79%. Fragility curve: the chain beats diagonal at \|Spearman\| ≥ **0.999** but **fails at 0.998**. The model is permutation-**equivariant** — the failure is purely that fitting `a` cannot discover a latent order. Nuance: with a wrong order the *fixed*-order variant degrades **gracefully to diagonal** (0%), while the *learned* variant overfits into a confidently-wrong chain (−84%). |
| 11 | **Real ordered data** (Tecator NIR, CO₂ lags) | On NIR (known wavelength order, n≈d): LCM **closes ~100%** of the diagonal→full joint-LL gap and beats diagonal on marginal too. On CO₂ (n≫d): gap negligible. In both, `LCM-learned ≈ LCM-fixed` — the **learned coordinate is inactive** because the order is already known. |

**Conclusion of Phase 5:** the genuine, real-data-confirmed advantage of LCM is
**joint predictive uncertainty for known-1-D-ordered features**; its distinctive
learned coordinate does no useful work (can't recover unknown orders; unnecessary
when the order is known).

---

## 6. Key design decisions — rationale, impact, alternatives

This section explains *why* each substantive choice was made, what it *cost* us,
and what could be done differently.

### 6.1 Inference method: Laplace approximation
- **Why.** LCM is a covariance model; the Laplace approximation is the cheapest,
  most direct way to turn a trained network into a Gaussian posterior whose
  covariance we can swap. Crucially, the paper's Appendix-A Frobenius objective
  (fit `A` to a sum of outer products) maps *exactly* onto fitting the Fisher —
  see §6.2 — so Laplace is the natural host.
- **Impact.** Confined the study to a Gaussian posterior centred at the MAP with a
  curvature-derived covariance; predictive quality is only as good as (curvature
  model + linearized/MC predictive).
- **Alternative.** Use LCM as a **structured variational family** (fit by ELBO
  rather than post-hoc curvature). This would test the same covariance structure
  under a different, arguably fairer inference frame, and would not depend on the
  GGN. Not pursued; a reasonable follow-up.

### 6.2 Curvature: Generalized Gauss–Newton (GGN) / Fisher, not the Hessian
- **Why GGN.** (i) The exact Hessian of a neural loss is generally **indefinite**,
  so `H⁻¹` is not a valid covariance; the GGN/Fisher is PSD by construction.
  (ii) It is the standard curvature in Laplace-for-NN (the `laplace-torch` default)
  so comparisons to diag/KFAC/full are apples-to-apples. (iii) The GGN is a **sum
  of rank-1 outer products** of per-example Jacobians/gradients, `G = Σ_n g_n g_nᵀ`
  — precisely the form the LCM Frobenius objective consumes, so LCM can be fit to
  it matrix-free.
- **Impact.** Near a well-fit optimum GGN ≈ Hessian, so this is mild for the
  conclusions. But GGN discards the loss-curvature residual term, and the
  **empirical** Fisher (using observed labels) differs from the **true** Fisher
  (expectation over model labels); we used the library's GGN/EF, which is a known,
  accepted approximation. For the **linear/GLM experiments (7–11)** this is not an
  approximation at all: the GGN is exactly `Xᵀ Λ X`, so those results are clean.
- **What could be done differently.** Use the exact Hessian where it is PSD
  (e.g., convex last layers); use MC-sampled labels for the true Fisher; or a
  positive-curvature-corrected Hessian. Unlikely to change the qualitative
  findings (which are about covariance *structure*, not curvature magnitude), but
  worth a robustness check for a camera-ready.

### 6.3 Materializing the dense GGN in `LCMLaplace` (then fixing it)
- **Why initially.** Correctness-first: subclassing `FullLaplace` and fitting LCM
  to the already-formed dense GGN let us reuse its exact, tested sampling /
  predictive / marginal-likelihood code and validate LCM against the exact full
  posterior.
- **Impact.** Capped the *class* at moderate `P` (dense GGN is `O(P²)`), even
  though the LCM *representation* is `O(P)`. This was flagged as a blocker.
- **What was done differently.** The matrix-free `fit_frobenius` path (never forms
  a `d×d` matrix) was shown to match the dense fit almost exactly once a real bug
  was fixed (§6.6). So the scalability is real; only the class wiring is
  conservative. A fully matrix-free `LCMLaplace` (per-sample gradients in, Kalman
  predictive out) is the clean next step.

### 6.4 Evaluation metric: joint vs marginal predictive likelihood
- **Why it mattered.** Every early negative used a **marginal** metric (per-point
  NLL, ECE, single-point OOD). Marginal predictive uncertainty is almost
  insensitive to the *off-diagonal* covariance — the very thing a structured model
  adds (a point made independently by Osband et al. on joint predictions). Once we
  switched to **joint** predictive likelihood over a batch of points, the LCM
  advantage appeared. This single reframing turned a flat negative into the
  project's main positive.
- **Impact / alternative.** Joint LL is the right target for batch decision-making
  (active learning, batch BO, experimental design), so the metric is
  application-relevant; but the downstream *application* (active learning) failed
  for a separate regime reason (§6.5).

### 6.5 The learned coordinate `a` and fitting it by gradient descent
- **Why.** It is the paper's parameterization and is differentiable, so it drops
  into gradient-based fitting.
- **Impact / limitation (central finding).** Fitting `a` is a continuous
  relaxation of a **discrete ordering (permutation) search** — highly non-convex.
  The internal `sort()` handles monotonicity but does not let gradient descent
  cross permutation basins. Consequently: (i) from an uninformed init the optimizer
  **cannot discover** a latent order (recovery |Spearman| ≈ 0.03); (ii) the kernel
  is **fragile** — even a 97.7%-correct order does worse than diagonal, needing
  ≳0.999; (iii) so the learned coordinate is only usable when initialized at (≈)
  the true order, i.e., when the order is already known — in which case it is
  inactive/unnecessary (real-data exp11: learned ≈ fixed).
- **What could be done differently.** Warm-start from spectral **seriation** then
  refine (tested: seriation recovers only the coarse order, insufficient given the
  fragility); discrete/combinatorial ordering optimization; or **restrict the
  method to known-order applications** and drop the "learned" claim. The last is
  the honest scoping.

### 6.6 Numerical: associative-scan log-det, inverse-softplus, auto-init
- **Associative-scan log-det.** The sequential Kalman log-det has Python-loop
  overhead (slower than dense until d≈5k). The paper's 2×2-matrix (Möbius)
  **associative scan** gives `O(d log d)` depth; we implemented it, verified
  machine-precision agreement and differentiability, and measured **15–64×**
  speedups — removing the wall-clock weakness. Now the default.
- **Second scan — the mean recursion (ported later).** A line-by-line comparison
  against the authors' reference `lcm.py` revealed that their `log_likelihood`
  runs the prefix-product scan **twice**: once for the variance/Riccati recursion
  and once for the **mean** recursion. We had parallelised only the first, leaving
  `quad_inv` / `nll` — the quadratic-form path — on a sequential Python loop. The
  mean recursion `m_k = α_k m_{k-1} + β_k` is affine, so it composes as the 2×2
  matrices `[[α, β], [0, 1]]` through the same scan. Porting it gave a further
  **16× at d=2000 and 48× at d=10000**, with values identical and `exp8c`
  reproducing bit-for-bit (max |Δ| = 0.000e+00 over 30 rows × 7 columns).
  **Honest note:** on this specific path the reference implementation was
  better-optimised than ours for most of the project, and the experiments that
  used `quad_inv` ran slower than they needed to. No result changed — only the
  wall-clock. `laplace/LCM_PROVENANCE.md` now maps every function to the reference
  so this class of divergence is visible rather than latent, and the old loop
  survives as `kalman_stats_sequential`, an independent validation reference.
- **Bug 1 — fit initialization (found & fixed).** The initial "matrix-free fit
  is weak" conclusion was **our own bug**: the fit initialization was scaled for
  the unnormalized GGN while `fit_frobenius` targets `GGN/N`. Diagnosing it
  revealed a *second, genuine* library bug — the naive inverse-softplus
  `log(expm1(x))` used to set the nugget **overflows to inf/NaN** for any moderately
  large diagonal scale (Adam then diverges silently). Fixed with a stable
  threshold-based `_inverse_softplus`, plus data-driven `auto_init` on both fitters.
  After the fix, matrix-free ≈ dense fit (0.2197 vs 0.2198 rel-Frobenius at d=1500).
- **Bug 2 — `kernel_matvec` overflow for long chains (found & fixed).** The O(d)
  prefix/suffix-sum factored `exp(a_j − a_i)` through a single global offset:
  `exp(−(a_i−a_0)) · Σ_j exp(a_j−a_0) x_j`. Every kernel entry is ≤ 1 and every
  exponent is non-positive, but the two factors were evaluated *separately*, so the
  growing factor overflows to `inf` once the **coordinate range** exceeds ~709 in
  float64 — yielding silent `NaN`. This depends on the *range*, not on `d`
  (d=20000 at spacing 0.02 → range 400 worked; d=10000 at spacing 0.105 → range
  1050 did not). **Impact:** `fit_frobenius` — precisely the *scalable* path —
  silently returned NaN parameters for long chains, invalidating every `d ≥ 5000`
  row of the first Exp 7 run. The Kalman/state-space primitives were unaffected
  (they use local ratios `exp(−Δa) ≤ 1`). **Fix:** compute the decaying prefix in
  log-space via `logcumsumexp` (`_decaying_prefix`), keeping the accumulator on the
  `a_i` scale (commit `b4d2323`), then — after benchmarking showed that cost 13–20×
  — replaced by **blocked rescaling** (commit `91b7d36`): blocks of coordinate range
  ≤ 60 use the fast offset+`cumsum` trick internally, with the accumulator carried
  across blocks rescaled by a factor ≤ 1. Verified exact vs the dense kernel
  (~1e-15) at ranges up to 3000, finite for d=20000 at range 1e4, differentiable,
  and only ~2× the unsafe original; regression test added.
  **Lesson:** "each term is bounded" does not imply "the factorization is stable" —
  the earlier code even documented the growing factor as "bounded by exp(range)"
  without treating that as a failure mode.
- **Bug 3 — wrong reference in the ordering diagnostic (found & fixed).** Exp 10/10b
  measured order recovery against `argsort(perm)` (the *inverse* permutation)
  instead of `perm`. Verified empirically: freezing `a = perm` fits the GGN to 0.14
  rel-Frobenius, while `a = argsort(perm)` gives 0.98 (no structure captured). This
  mis-stated the **seriation** recovery (reported 0.03; true value 0.31–0.99). The
  substantive conclusions were unaffected — the joint-LL numbers never depended on
  this diagnostic column, and the learned coordinate scores ≈0 against either
  reference — but the corrected numbers make the finding *sharper* (seriation can
  reach 0.99 and *still* be insufficient against the 0.999 threshold).

### 6.7 Baselines and protocol
- Baselines: **Diag, KFAC, Full** Laplace, plus MAP; LowRank was dropped (its
  backend needs an unavailable legacy dependency). KFAC is the strong structured
  competitor; Full is the exact reference (via the `n×n` dual/Woodbury form where
  `d` is large, so it is exact at every `d`).
- Hyperparameters: prior precision (and regression noise) tuned per method by the
  **marginal likelihood** for the predictive benchmarks (fair, per-method best
  case); a **common fixed prior** for the fidelity experiments (to isolate the
  curvature structure); a heuristic noise for the real-data first pass (relative
  diag→full gap is the robust signal there).
- Data regimes: **synthetic AR(1)** features to create a GGN that is *exactly*
  Laplace-kernel-structured (the ideal case — does the method help *at all*?), then
  **real** ordered-feature data (does it help in practice?).

---

## 7. Results summary (headline numbers)

**Fidelity (Exp 1, KL to full, lower better):** LCM 17.5–406 vs Diag 51–658 across
settings (1.6–2.9× better); KFAC 1.5–104 (best).

**Joint vs marginal predictive LL (Exp 8c, synthetic 1-D GLM, 10 seeds ± 95% CI):**

| d | marginal-LL (diag / LCM / full) | joint-LL (diag / LCM / full) | joint gap-to-full closed by LCM |
|---|---|---|---|
| 500 | −0.823 / −0.826 / −0.823 | −0.830 / −0.823 / −0.820 | **62% ± 49%** |
| 1000 | −0.922 / −0.933 / −0.919 | −0.946 / −0.927 / −0.907 | **52% ± 16%** |
| 1500 | −1.076 / −1.110 / −1.057 | −1.116 / −1.097 / −1.033 | **23% ± 12%** |

Reading (rigorous, 10 seeds): the joint-UQ advantage is **real but modest and
shrinks with d** (and is not significant at d=500). On **marginal** LL, LCM is
*slightly worse* than diagonal — the structured fit's imperfection costs a little
per-point calibration, confirming the marginal metric does not reward the
structure. (An earlier 3-seed run with a near-zero nugget reported 70–81%; the
10-seed number with the library's default fit is the trustworthy one, and the gap
is sensitive to the nugget fraction — a tuning knob, not a robust large effect.)

**Ordering (Exp 10, d=800, n=3000, 3 seeds; joint-LL gap-to-full closed):**

| condition | LCM-fixed(natural) | LCM-fixed(seriation) | LCM-learned | learned recovery |
|---|---|---|---|---|
| order **known** (natural) | **+58%** | −79% | **+59%** | 1.000 |
| order **latent** (permuted) | 0% (→ diagonal) | −79% | **−84%** | 0.028 |

**Ordering fragility (Exp 10b, d=400, 3 seeds)** — how accurate must the order be?
Order accuracy was degraded by adding position noise to the true order:

| \|Spearman\| | 1.000 | 0.9997 | 0.9999 | 0.9994 | **0.998** | 0.991 | 0.967 |
|---|---|---|---|---|---|---|---|
| joint-LL | −0.853 | −0.852 | −0.852 | −0.858 | −0.870 | −0.885 | −0.892 |
| beats diagonal (−0.865)? | ✔ | ✔ | ✔ | ✔ | **✘** | ✘ | ✘ |

The threshold is sharp and extreme: **≥0.999 works, 0.998 already fails.**
Recovery attempts fall far short — seriation 0.31/0.70/0.99 (mean 0.67), learned
coordinate 0.05/0.07/0.05 (mean 0.058).

*Caveat (seed variance):* in 1 of 3 fragility seeds diagonal actually beat full
(−0.830 vs −0.836), i.e. the premise "full > diagonal" is itself noisy at this
size; the threshold conclusion holds on the mean but individual seeds vary.

**Real data (Exp 11, joint-LL gap-to-full closed by LCM):**
- Tecator NIR (n=144, d=100, order known): **~100%** (also beats diagonal on
  marginal LL). `learned ≈ fixed`.
- CO₂-AR(64) (n≫d): gap negligible; LCM ≈ diagonal ≈ full.

**Cost:** `3d` covariance storage vs `d²` (full); exact `O(d log d)` log-det, 15–64×
faster than the sequential version and feasible where dense OOMs. The quadratic-form
path (`quad_inv` / `nll`) was later moved onto the reference's second scan: a further
**16× at d=2000, 48× at d=10000** (§6.6).

**Scaling of the matrix-free fit (Exp 7, clean single-process run, n=3000, 800
Adam steps/fit).** After the `kernel_matvec` fix, `fit_frobenius` runs to large `d`
with **no NaN anywhere** (previously NaN for every `d ≥ 5000`):

| d | test-LL diag / **LCM** / full | pred-var rel-err diag / LCM | LCM fit |
|---|---|---|---|
| 500 | −0.814 / **−0.814** / −0.814 | 0.033 / 0.039 | 4.3 s |
| 1 000 | −0.943 / **−0.944** / −0.938 | 0.111 / 0.128 | 11.6 s |
| 2 000 | −1.335 / **−1.348** / −1.250 | 0.441 / 0.460 | 27.6 s |
| 5 000 | −1.303 / **−1.308** / −3.736 | 0.998 / 0.998 | 86.9 s |
| 10 000 | −1.227 / **−1.212** / −4.592 | 0.999 / 0.999 | 244.4 s |
| 20 000 | −1.508 / **−1.491** / −5.283 | 1.000 / 1.000 | 887.8 s |

**Implementation scaling is worse than the theory.** Wall-clock grows ~`O(d^1.44)`
(40× in `d` → 206× in time), not `O(d log d)`: the blocked `kernel_matvec` uses
`range/60` blocks and, at fixed spacing, the coordinate range grows ∝ `d`, so the
number of *sequential* block iterations grows linearly and kernel-launch/Python
overhead dominates at large `d`. The asymptotic *work* is still `O(d)`; the
implementation is not. A fused kernel, or an associative-scan formulation of the
block carry (as already done for the log-determinant), would recover it.

Two caveats on how to read those rows:
- they demonstrate **computational feasibility, not statistical benefit**: with
  `n = 3000` fixed, every `d ≥ 5000` row is in the `d > n` (rank-deficient) regime
  where the exact `full` posterior itself scores *worse* than diagonal, so the
  comparison is degenerate there. The statistically meaningful rows are `d ≤ 2000`
  (`n > d`), consistent with the `n ≳ d` requirement found throughout;
- the blocked-rescaling fix costs ~2× per matvec at fixed range, and progressively
  more as the range (hence block count) grows — see the `O(d^1.44)` note above;
  a fused/segmented kernel would remove this.

*Process note:* the first attempt at this run was contaminated — a previous
instance was not killed before relaunching, so two processes shared the log file
and the GPU until one hit a CUDA OOM. The predictive values were identical to the
clean run, but the timings were inflated ~4×; only the clean run is cited. Stale
`pgrep`-based waiter loops that matched their own command line (and so could never
exit) were also cleaned up.

---

## 8. Limitations & issues (consolidated)

1. **Narrow niche.** The only demonstrated advantage is joint predictive UQ for
   variables with a **known** 1-D order and decaying correlation, in the `n ≳ d`
   regime. It does not help NN weight-space Laplace (KFAC dominates), marginal
   calibration/OOD, or low-data tasks.
2. **The learned coordinate is not useful.** It can't recover unknown orders and is
   unnecessary when the order is known — undercutting the method's novelty over
   classical banded-GMRF / state-space-GP models.
3. **Kernel fragility.** Bandwidth-1 exp-decay is extremely sensitive to
   fine-ordering error (needs ≳0.999); a mis-ordered chain is *confidently wrong*,
   worse than a safe diagonal.
4. **Regime dependence.** When `n ≫ d`, diagonal is already near-full on joint LL,
   so there is little to gain; when curvature is not chain-embeddable (NN layers),
   the fit plateaus (~0.66 residual).
5. **Class scalability.** `LCMLaplace` currently materializes the dense GGN
   (moderate-P only); the matrix-free path exists and works but is not yet wired
   through predictive/sampling end-to-end.
6. **Approximations.** GGN/empirical-Fisher (not exact Hessian/true Fisher) for the
   NN experiments (exact for the GLM ones); heuristic noise in the real-data first
   pass; modest dataset count (2 real datasets × 5 splits).
7. **Bugs encountered (all fixed, all regression-tested).** Three, and notably
   *all three failed silently* rather than raising:
   (a) an init-scale mistake in the experiment scripts;
   (b) an **inverse-softplus overflow** in the library (`log(expm1(x))` → inf/NaN
       for large diagonal scales, making Adam diverge);
   (c) a **`kernel_matvec` overflow** for coordinate ranges > ~709, which returned
       NaN from the *matrix-free* fit and invalidated every `d ≥ 5000` row of the
       first Exp 7 run;
   plus (d) a wrong reference (`argsort(perm)` vs `perm`) in the ordering-recovery
   *diagnostic*, which mis-stated seriation's recovery. The recurring theme —
   silent numerical failure in exponential-family primitives — is the main
   engineering risk in this method family, and argues for finite-value assertions
   in the fit loop.
8. **Cost of the stability fix (measured, then reduced).** The first stable
   rewrite used `logcumsumexp`: exact, but **13–20× slower** than `cumsum`
   (56 ms vs 2.9 ms at n=3000, d=500) — enough to make the matrix-free fit
   impractical. It was replaced by **blocked rescaling** (blocks of coordinate
   range ≤ 60, carrying a running accumulator rescaled to each block's origin by a
   factor ≤ 1), which is exact to ~1e-15, finite at ranges up to 1e4, and only
   **~2×** the cost of the unsafe original. Lesson: a numerically-safe rewrite
   should be benchmarked, not assumed cheap.
9. **Reproducibility gap now closed.** `run_all.py` originally listed only
   experiments 1–4, so the documented "reproduce the campaign" command silently
   skipped everything after; it now enumerates all 15 experiments in phase order.

---

## 9. Verdict & recommendations

- **Publishable, honestly, as a focused contribution** — workshop (AABI /
  NeurIPS-BDL) → TMLR — combining (a) the systems artifact (near-linear structured
  Laplace with an associative-scan log-det, in a maintained library), (b) the
  characterization (fidelity ≠ predictive; marginal-insensitive / joint-sensitive;
  ordering fragility; Kronecker-vs-chain), and (c) the real-data joint-UQ win for
  known-ordered features. See `PUBLICATION_ASSESSMENT.md` for venue analysis,
  novelty check, and reviewer rebuttals.
- **Do not** claim SOTA Bayesian-DL UQ, an active-learning/BO win, or a working
  *learned* ordering.
- **Highest-value next steps:** (i) a fully matrix-free `LCMLaplace` to make the
  scalability claim end-to-end; (ii) more real known-order datasets (spectroscopy,
  genomics-LD, climate transects) with marglik-tuned noise and more seeds;
  (iii) if a broader claim is wanted, solve the ordering problem (seriation+refine
  or discrete search) — but the fragility result suggests limited upside.

---

## 10. Appendix — file index & reproduction

**Analysis docs:** `LCM_bayesian_uses.md`, `LCM_laplace_approximation.md`,
`LCM_laplace_math_stepbystep.md`, `COMPELLING_USE.md`, `PUBLICATION_ASSESSMENT.md`,
`AUDIT.md`, `LCM_REPORT.html`, this report.

**Experimental section:** `experiments/EXPERIMENTS.md` — per-experiment detail
(setup, metrics, result tables, how to read) for all 15 experiments. This report
covers the arc and the *why*; that document covers the *what*.

**Library fork:** `Laplace/` (branch `lcm-covariance`) — `laplace/lcm.py`,
`LCMLaplace`/`LCMLLLaplace` in `laplace/baselaplace.py` & `laplace/lllaplace.py`,
tests in `tests/test_lcm.py` (16). `Laplace/LCM_ADDITION.md` documents the API;
`Laplace/laplace/LCM_PROVENANCE.md` maps every function to the authors' reference
implementation and records the deliberate deviations. Reproduce tests:
`cd Laplace && pytest tests/test_lcm.py`.

**Experiments:** `experiments/exp1…exp11`, `exp8c`, `exp10b`, `ctrl_ordering.py`;
shared harness `experiments/common.py`; results in `experiments/results/`.
Reproduce the campaign: `python experiments/run_all.py` (+ the standalone `exp*`).

**Datasets:** UCI regression (OpenML/sklearn), MNIST/FashionMNIST, CIFAR-10/SVHN
(cached), Tecator NIR (OpenML), Mauna Loa CO₂ (statsmodels).
