# Experiments: Structured-Covariance (LCM) Laplace Approximation

We evaluate the Laplace Covariance Model (LCM) as a structured posterior for the
Laplace approximation of neural-network weights, implemented as
`hessian_structure="lcm"` in the Laplace library (branch `lcm-covariance`). The
LCM approximates the log-likelihood Hessian / generalized Gauss–Newton (GGN) by

$$A = \mathrm{diag}(d) + \mathrm{diag}(w)\,K(a)\,\mathrm{diag}(w),\qquad K(a)_{ij}=e^{-|a_i-a_j|},$$

a Gauss–Markov (Matérn-½) covariance on a *learned* 1-D ordering of the
parameters, using only `3P` numbers. It captures off-diagonal curvature that a
**diagonal** Laplace ignores, while its 1-D state-space structure yields an
`O(P)` log-determinant for the marginal likelihood. Diagonal Laplace is the
special case `w=0`.

---

## TL;DR of findings

The project ran in two phases, and the headline changed between them.

**Phase 1 (Exp 1–4) — fidelity yes, prediction no.** LCM is a strict,
near-diagonal-cost improvement over diagonal Laplace in *posterior fidelity*
(1.6–2.9× lower KL to the exact posterior, Exp 1) and recovers chain-structured
curvature essentially exactly (Exp 4b, 1e-4). But this did **not** transfer to
predictive uncertainty: LCM ties a marginal-likelihood-tuned diagonal on UCI
regression (Exp 2), saturates with it on MNIST and is *worse* on CIFAR-10
(Exp 3). Ablations pinpointed the cause — a single 1-D Markov chain cannot embed
generic NN-weight curvature (Exp 4a plateau at 0.66 rel-Frobenius), because a
layer's GGN is **Kronecker**-structured, which is KFAC's home.

**Phase 2 (Exp 5–11) — the reframing.** Every Phase-1 negative used a **marginal**
metric (per-point NLL, ECE, single-point OOD), and marginal predictive
uncertainty is nearly insensitive to off-diagonal covariance — precisely what LCM
adds. Switching to **joint** predictive likelihood over a batch exposed a real
advantage (Exp 8c: 62%/52%/23% of the diagonal→full gap closed at d=500/1000/1500,
10 seeds), confirmed on real ordered data (Exp 11, Tecator NIR: ~100%). The
boundary conditions are sharp and largely negative: it needs `n ≳ d` (so active
learning fails, Exp 9), it needs the 1-D order to be **known** to within
\|Spearman\| ≥ 0.999 (Exp 10b), and the paper's distinctive *learned* coordinate
cannot discover a latent order (recovery 0.028) and is redundant when the order is
already known (Exp 11: learned ≈ fixed).

**Cost (Exp 12) — the comparison Phase 1 never priced.** LCM and KFAC are not
substitutes. LCM's `3P` storage beats KFAC only on layers with aspect ratio
≳ 2.6:1 (KFAC wins 1.5x on square layers), and LCM is **~57x more expensive to
construct** because it needs an iterative fit where KFAC is closed-form. Crucially,
on the scalar-output GLMs where LCM wins, KFAC *is* full Laplace exactly (measured
rel-Frobenius 1e-06) and offers no compression — so it was never a cheap baseline
in that setting.

**Complexity class (Exp 13) — the peer comparison, split verdict.** Against
banded-Cholesky (`AAᵀ` → tridiagonal precision) and low-rank peers at equal budget:
**in the niche LCM wins decisively** — `3d` parameters to low-rank's `17d` for the
same fit, and +87% of the joint gap closed where every peer closes ≈0% or is
actively harmful. **On NN weight curvature it loses** — `rank4` at `5d` beats LCM
at `3d` and keeps improving, with no ordering fragility. Exp 13 also found that
scale-normalising the fit raises the headline gap from 54% to 87% at `d=1000` on
identical seeds, so the reported numbers are *conservative*.

**Net.** LCM is the strongest diagonal-cost, architecture-agnostic structured
covariance, with one demonstrated real advantage — **joint** predictive
uncertainty for **known-1-D-ordered** features. It is not a replacement for KFAC
on neural-network weights, and not a competitor to it either: the two encode
different curvature geometries. Where it does win, it wins against its whole
complexity class, not just against diagonal.

## Baselines

All methods share the same trained MAP network and differ only in the structure
of the Hessian/posterior-precision approximation:

* **MAP** — point estimate (homoscedastic Gaussian noise for regression; softmax
  for classification); no weight uncertainty.
* **Diag** — diagonal Laplace (the structure LCM strictly generalizes).
* **KFAC** — Kronecker-factored Laplace (a stronger, layer-structured baseline).
* **Full** — exact dense Laplace (gold-standard reference; in the GLM experiments
  it uses the `n×n` dual/Woodbury form, so it is exact at every `d`).
* **LCM** — *ours*, the structured Gauss–Markov covariance (`3P` parameters).

(The library's Low-rank Laplace is omitted: it requires an unavailable legacy
backend, `asdfghjkl`. Exp 13 implements `D + V Vᵀ` directly instead, which turns
out to matter — it is LCM's strongest same-class competitor.)

## Common protocol

Every Laplace variant is fit to the same MAP network with the same GGN backend.
Unless stated otherwise, each method's prior precision — and, for regression, its
observation noise — is tuned *per method* by maximizing that method's Laplace
**marginal likelihood** (gradient ascent on the log-hyperparameters), giving
every baseline its own best-case hyper-parameters. Predictives use the linearized
(GLM) Laplace predictive; classification uses the probit approximation. The LCM
is fit to the GGN by minimizing `‖A − H‖_F²` (Adam, 500–800 steps), the Frobenius
objective of Appendix A of the paper.

**The "gap closed" metric**, used throughout Phase 2, is

```
gap closed = (LL_method − LL_diag) / (LL_full − LL_diag) × 100%
```

i.e. what fraction of the distance from diagonal Laplace to the *exact* posterior
a method recovers. 0% = no better than diagonal; 100% = matches full; negative =
worse than diagonal.

## Experiment map

| # | script | question | verdict |
|---|---|---|---|
| 1 | `exp1_fidelity.py` | Is the posterior more faithful than diagonal? | **positive** |
| 2 | `exp2_uci.py` | Better UCI regression test-LL? | neutral (tie) |
| 3 / 3b | `exp3_image.py`, `exp3b_cifar.py` | Better calibration / OOD? | negative |
| 4 / 4b | `exp4_scaling.py`, `exp4b_ablation_controlled.py` | Cost, and *why* the fit plateaus | **explanatory** |
| 5 | `exp5_epistemic.py` | Do distributional (epistemic) OOD scores rescue it? | negative |
| 6 | `exp6_perlayer.py` | Does a per-layer (block) chain help? | partial |
| 7 | `exp7_correlated_glm.py` | Does it work in its *home* (1-D-correlated GLM)? | **positive + scaling** |
| 8 / 8b / 8c | `exp8_joint.py`, `exp8b_joint_robust.py`, `exp8c_joint_ci.py` | Marginal vs **joint** predictive LL | **the key positive** |
| 9 | `exp9_active_learning.py` | Does joint UQ buy a downstream win? | negative |
| 10 / 10b | `exp10_ordering.py`, `exp10b_fragility.py` | Does the *learned* ordering work? How exact must it be? | **negative, central** |
| 11 | `exp11_realdata.py` | Does it hold on real ordered data? | **positive (1 dataset)** |
| 12 | `exp12_cost.py` | What do LCM and KFAC *cost*, and are they comparable? | **corrective** |
| 13 | `exp13_sameclass.py` | Is LCM the best model in its own O(d) complexity class? | **split: yes in niche, no on NN** |

Supporting diagnostics: `ctrl_ordering.py` (frozen-order control),
`diag_matfree_init.py` (matrix-free fit initialization).

---
---

# Phase 1 — Core evaluation (Experiments 1–4)

## Experiment 1 — Posterior fidelity

**Why.** Before asking whether LCM predicts better, establish whether it *is* a
better posterior. This isolates the covariance model from every downstream
confound (predictive approximation, hyper-parameter tuning, metric choice).

**Setup.** Where the exact full-Laplace posterior is computable, we fix a *common*
prior precision across all methods (isolating the curvature approximation) and
measure how close each structured posterior is to the full one, via
`KL( N(θ*,Σ_approx) ‖ N(θ*,Σ_full) )` and the relative Frobenius error of the
posterior covariance. Four settings (regression/classification × last-layer/
all-weights), AR(1)-correlated features, 3 seeds.

**KL to the full posterior (lower = better):**

| setting | P | Diag | KFAC | **LCM** |
| --- | --- | --- | --- | --- |
| reg / last-layer | 33 | 51.3 | 1.5 | **17.5** |
| reg / all-weights | 793 | 249.1 | 28.1 | **95.2** |
| cls / last-layer | 99 | 73.0 | 5.5 | **30.3** |
| cls / all-weights | 843 | 657.7 | 104.2 | **406.2** |

**Relative Frobenius error of the posterior covariance:**

| setting | Diag | KFAC | **LCM** |
| --- | --- | --- | --- |
| reg / last-layer | 0.942 | 0.162 | **0.758** |
| reg / all-weights | 0.458 | 0.148 | **0.288** |
| cls / last-layer | 0.827 | 0.130 | **0.575** |
| cls / all-weights | 0.748 | 0.264 | **0.573** |

**Outcome.** LCM reduces the KL to the exact posterior by **1.6–2.9×** relative
to diagonal Laplace in every setting, confirming that the Gauss–Markov chain
captures real off-diagonal curvature the diagonal discards. KFAC — which exploits
the network's layer structure — is more faithful still. The ordering
`KFAC < LCM < Diag` is consistent across all settings and both metrics.

**Ablation — when does structure help?** Varying the feature-correlation level
`ρ` (which controls how off-diagonal the last-layer curvature is; figure
`results/exp1_whenithelps.png`), the diagonal's KL to the full posterior *grows*
as correlation increases (AR(1): 57.9→76.7 as `ρ`: 0→0.95), while LCM's stays low
and roughly flat (~20–30). LCM is thus increasingly preferable to diagonal as the
curvature becomes more correlated, and never worse across the range.

**Conclusion.** The fidelity claim is established and never later overturned. It
is the premise for everything that follows — and the fact that it does *not*
imply a predictive win is the project's central tension.

---

## Experiment 2 — UCI regression uncertainty

**Why.** The standard benchmark for Bayesian-NN uncertainty; the first test of
whether Exp 1's fidelity gain reaches a downstream predictive metric.

**Setup.** Single-hidden-layer (50 tanh) BNN regression on six UCI datasets;
inputs/targets standardized; **all-weights** Laplace; prior precision and noise
tuned by marginal likelihood; mean **test log-likelihood** (original scale;
higher = better) over 8 random 90/10 splits. RMSE is identical across Laplace
variants (shared MAP mean), so predictive quality is entirely in the LL.

| dataset | MAP | Diag | KFAC | Full | **LCM** |
| --- | --- | --- | --- | --- | --- |
| california | −0.910 | −0.874 | −0.883 | −0.868 | **−0.889** |
| concrete | −3.232 | −3.182 | −3.132 | −3.095 | **−3.196** |
| diabetes | −5.882 | −5.672 | −5.652 | −5.589 | **−5.638** |
| energy | −2.300 | −2.518 | −2.143 | −2.184 | **−2.461** |
| wine | −1.129 | −0.990 | −1.024 | −0.979 | **−1.012** |
| yacht | −2.133 | −2.907 | −1.974 | −1.954 | **−2.396** |

**Average test-LL rank (1 = best):** Full 1.79 · KFAC 2.35 · **LCM 3.52** ·
Diag 3.54 · MAP 3.79.

**Outcome.** LCM and diagonal are **statistically tied on average** (rank 3.52 vs
3.54), with Full and KFAC clearly ahead. LCM's behaviour relative to diagonal is
*variance-reducing*: on `yacht`, where diagonal collapses to a badly
over-/under-confident LL of −2.91 (worse than MAP), LCM recovers to −2.40; on
`energy` and `diabetes` LCM also beats diagonal. On the other three datasets LCM
is marginally behind a well-tuned diagonal.

**Conclusion.** For all-weights regression, LCM's fidelity edge yields robustness
(it never collapses like diagonal) but not a systematic predictive-LL
improvement. In hindsight (Exp 8) this is exactly what a *marginal* metric should
be expected to show.

---

## Experiment 3 — Image classification calibration & OOD (last-layer)

**Why.** Calibration and OOD detection are the headline applications of Laplace
for deep learning; if structured covariance helps anywhere in deep nets, it
should help here.

**Setup.** Small CNNs; **last-layer** Laplace. In-distribution accuracy / NLL /
expected calibration error (ECE), and OOD-detection AUROC by predictive
confidence. 3 seeds (MNIST) / 2 seeds (CIFAR-10).

**MNIST (OOD = FashionMNIST):**

| method | acc | NLL | ECE | OOD-AUROC |
| --- | --- | --- | --- | --- |
| MAP | 0.9814 | **0.0593** | **0.0053** | **0.9698** |
| Diag | 0.9798 | 0.0743 | 0.0183 | 0.9617 |
| KFAC | 0.9819 | 0.0950 | 0.0456 | 0.9613 |
| Full | 0.9818 | 0.1048 | 0.0548 | 0.9619 |
| **LCM** | 0.9814 | 0.1045 | 0.0536 | **0.9682** |

**CIFAR-10 (OOD = SVHN):**

| method | acc | NLL | ECE | OOD-AUROC |
| --- | --- | --- | --- | --- |
| MAP | 0.8309 | 0.5760 | 0.0678 | **0.8346** |
| Diag | 0.8298 | 0.5432 | 0.0540 | 0.8193 |
| KFAC | 0.8309 | **0.5036** | **0.0180** | 0.7949 |
| Full | 0.8311 | 0.5050 | 0.0251 | 0.7925 |
| **LCM** | 0.8307 | 0.5503 | 0.0893 | 0.7992 |

**Outcome.** MNIST is *saturated*: the MAP network is already well-calibrated
(ECE 0.005), so every Laplace variant slightly *worsens* in-distribution NLL/ECE;
among them LCM attains the best OOD-AUROC (0.968). On the harder CIFAR-10, Laplace
helps in-distribution calibration overall — KFAC/Full cut NLL from 0.576 to ≈0.50
and ECE from 0.068 to ≤0.025 — but **LCM under-performs even diagonal** here
(NLL 0.550 vs 0.543, ECE 0.089 vs 0.054): its imperfect fit to the last-layer GGN
produces a miscalibrated predictive.

**Conclusion.** The fidelity advantage over diagonal does not translate into a
predictive-calibration advantage, and KFAC/Full are clearly preferable for
last-layer image models. This is the sharpest Phase-1 negative and motivated both
Exp 5 (is the *metric* wrong?) and Exp 4a (is the *structure* wrong?).

---

## Experiment 4 — Cost, scaling, and ablations

**Why.** Two separate questions: (a) is the `3P` representation actually cheap in
practice, and (b) *why* does the fit to a real GGN plateau — is it the model, the
optimizer, or the target?

**Scaling of the posterior log-determinant** (figure `results/exp4_scaling.png`).
Covariance storage is `P` (Diag), `3P` (LCM), `P²` (Full):

| P | mem Full (floats) | mem **LCM** | log-det Full (ms) | log-det **LCM** (ms) |
| --- | --- | --- | --- | --- |
| 1 000 | 1.0e6 | **3 000** | 3.0 | 28.9 |
| 5 000 | 2.5e7 | **15 000** | 166.1 | **108.5** |
| 10 000 | 1.0e8 | **30 000** | OOM / — | **346.3** |
| 20 000 | 4.0e8 | **60 000** | OOM / — | **588.6** |

> **⚠ These timings are historical.** They were measured with the original
> *sequential* Kalman log-determinant. The associative-scan implementation
> (`logdet_parallel`, now the default) is **15–64× faster** — 4.8 ms vs 305 ms at
> d=2000 — so the LCM column is now far better than shown, and LCM overtakes the
> dense log-det well before `P≈5k`. See "Implementation notes" below. The
> **memory** column is unaffected and unconditional.

**Outcome (cost).** LCM stores the covariance in `3P` floats vs `P²` for the dense
posterior (a >6 500× reduction at `P=20k`) and computes an *exact* log-determinant
where the dense matrix is infeasible (`P≥10k`).

**Ablation (a) — on a real last-layer GGN (`P=124`):**

| learn `a` | nugget | rel. Frobenius to GGN |
| --- | --- | --- |
| ✓ | ✓ | 0.659 |
| ✓ | ✗ | 0.662 |
| ✗ | ✓ | 0.642 |
| ✗ | ✗ | 0.657 |
| diagonal-only | | 0.973 |

**Ablation (b) — controlled scrambled-AR(1) target (`exp4b`, exactly
LCM-representable in some order):**

| learn `a` | nugget | rel. Frobenius to target |
| --- | --- | --- |
| ✓ | ✓ | **0.0001** |
| ✓ | ✗ | 0.153 |
| ✗ | ✓ | 0.560 |
| ✗ | ✗ | 0.580 |
| diagonal-only | | 0.956 |

**Outcome (ablation).** On the *real* GGN, LCM roughly halves diagonal's Frobenius
error (0.66 vs 0.97) but plateaus at 0.66 — and learning the ordering / adding the
nugget barely move it. The *controlled* experiment isolates the design choices:
when the target genuinely is a (scrambled, nugget-carrying) chain, LCM recovers it
to 1e-4 — but only with **both** a learned ordering (removing it: 0.56) and the
diagonal nugget (removing it: 0.15). The fit converges by ≈200 Adam steps
(0.6585 at 800 steps → 0.6582 at 1600; `results/exp4_ablation_steps.csv`).

**Conclusion.** Both LCM components are essential, and the real-GGN plateau
reflects **the target, not the model, the optimizer, or the budget**. This is the
single most important diagnostic in Phase 1: it says more compute cannot fix the
Phase-1 negatives, and it is independently confirmed by the audit's
Kronecker-vs-chain measurement (see `AUDIT.md` §4).

---
---

# Phase 2 — Search for a compelling use (Experiments 5–11)

Phase 1 left two escape hatches: maybe the *metric* was wrong (Exp 5, 8), or maybe
the *structure* was applied at the wrong granularity (Exp 6) or to the wrong
*problem* (Exp 7, 11). Phase 2 tests all of them.

## Experiment 5 — Epistemic (distributional) OOD scores

**Why.** Exp 3 scored OOD by max predictive confidence — a quantity a MAP network
already provides, which barely uses the posterior at all. Mutual information
(BALD) isolates the *epistemic* component, which is exactly what a covariance
model should improve. This directly tests the "wrong metric" hypothesis.

**Setup.** Same last-layer image setting as Exp 3; OOD ranked by max-confidence,
by predictive entropy, and by **mutual information** (`MI = H[E p] − E H[p]`,
epistemic-only).

| method | AUROC (conf) | AUROC (entropy) | **AUROC (MI)** | NLL | ECE |
| --- | --- | --- | --- | --- | --- |
| MAP | 0.840 | 0.860 | **0.500** | 0.562 | 0.068 |
| Diag | 0.822 | 0.844 | **0.640** | 0.533 | 0.045 |
| KFAC | 0.835 | 0.856 | **0.724** | 0.548 | 0.057 |
| Full | 0.833 | 0.854 | **0.722** | 0.544 | 0.056 |
| **LCM** | 0.815 | 0.830 | **0.624** | 0.589 | 0.122 |

**Outcome.** The MI metric behaves exactly as designed — MAP scores 0.500
(chance), since a point estimate has no epistemic uncertainty, validating the
measurement. KFAC and Full climb to ≈0.72. But **LCM (0.624) lands below diagonal
(0.640)**.

**Conclusion.** The "wrong metric" hypothesis fails *in this form*. Switching from
confidence to a properly epistemic score does not rescue LCM, because the problem
is not which scalar summary is used — it is that the chain does not capture the
right off-diagonal structure for NN weights (Exp 4a). The metric reframing that
*does* work is a different one: single-point → **joint** (Exp 8).

---

## Experiment 6 — Per-layer (block-diagonal) LCM

**Why.** A single global chain must order *all* `P` parameters, mixing weights
from different layers that have no natural adjacency. KFAC's advantage is
per-layer structure; giving LCM one chain per layer is the natural way to borrow
it while keeping `3P` cost.

**Setup.** Fit an independent LCM chain per parameter block, compare KL to the
full posterior against the single global chain, diagonal, and KFAC.

| setting | Diag | KFAC | LCM-global | **LCM-block** | block vs global |
| --- | --- | --- | --- | --- | --- |
| reg / all-weights | 249.1 | **28.1** | 95.2 | **73.9** | −22% |
| cls / all-weights | 657.7 | **104.2** | 406.2 | **357.1** | −12% |
| cls / last-layer | 73.0 | **5.5** | 30.3 | 31.0 | +3% |

**Outcome.** Per-layer blocking gives a real but modest improvement on
all-weights settings (12–22% lower KL) and none at all on a single last layer —
where, trivially, there is only one block, so the +3% is fit noise.

**Conclusion.** Blocking helps in the direction predicted, but does not close the
gap to KFAC (357 vs 104). It confirms the diagnosis rather than fixing it: the
residual gap is *within-layer* Kronecker structure, which blocking does not
address, since a chain within a layer is still bandwidth-1.

---

## Experiment 7 — 1-D-correlated GLM (LCM's home), and scaling

**Why.** Every prior experiment applied LCM where its assumption is known to be
false. The fair test is a problem whose curvature genuinely *is* 1-D-correlated
and non-Kronecker: a GLM on ordered (AR-correlated) features, where the GGN is
exactly `XᵀΛX` — no GGN-vs-Hessian approximation at all. This is also the
scalability testbed, since the matrix-free `fit_frobenius` path is used.

**Setup.** Gaussian GLM, `n=3000` train, AR(1)-correlated ordered features, 800
Adam steps per fit, single clean process. Full uses the exact `n×n` dual form.

| d | test-LL diag / **LCM** / full | pred-var rel-err diag / LCM | LCM fit |
|---|---|---|---|
| 500 | −0.814 / **−0.814** / −0.814 | 0.033 / 0.039 | 4.3 s |
| 1 000 | −0.943 / **−0.944** / −0.938 | 0.111 / 0.128 | 11.6 s |
| 2 000 | −1.335 / **−1.348** / −1.250 | 0.441 / 0.460 | 27.6 s |
| 5 000 | −1.303 / **−1.308** / −3.736 | 0.998 / 0.998 | 86.9 s |
| 10 000 | −1.227 / **−1.212** / −4.592 | 0.999 / 0.999 | 244.4 s |
| 20 000 | −1.508 / **−1.491** / −5.283 | 1.000 / 1.000 | 887.8 s |

**Outcome.** After the `kernel_matvec` overflow fix the matrix-free path runs to
`d = 20 000` with **no NaN anywhere** (previously every `d ≥ 5000` row was NaN —
see "Implementation notes"). But on the *marginal* test-LL, LCM still only ties
diagonal.

**Two caveats on how to read the table.**
- It demonstrates **computational feasibility, not statistical benefit.** With
  `n = 3000` fixed, every `d ≥ 5000` row is in the `d > n` rank-deficient regime
  where the exact `full` posterior itself scores *worse* than diagonal — the
  comparison is degenerate there. The meaningful rows are `d ≤ 2000` (`n > d`).
- **Implementation scaling is worse than the theory:** wall-clock grows ~`O(d^1.44)`
  (40× in `d` → 206× in time), not `O(d log d)`. The blocked `kernel_matvec` uses
  `range/60` blocks and, at fixed spacing, the coordinate range grows ∝ `d`, so
  the number of *sequential* block iterations grows linearly. The asymptotic
  *work* is still `O(d)`; the implementation is not. **This remains open** — see
  `AUDIT.md` recommendation #5.

**Conclusion.** Even in LCM's home territory the marginal metric shows a tie. That
is what forced the decisive question of Exp 8: is the metric itself blind to what
LCM adds?

---

## Experiment 8 / 8b / 8c — Marginal vs **joint** predictive likelihood

**Why.** *The pivotal experiment.* Marginal (per-point) predictive likelihood
integrates out all cross-point correlation, so it is almost insensitive to the
off-diagonal posterior covariance — the only thing LCM adds over diagonal. Joint
predictive likelihood over a batch of test points is the metric that can see it
(a point argued independently by Osband et al. for the Neural Testbed). Exp 8 was
the first look, 8b a robustness pass, and **8c the definitive 10-seed run with
confidence intervals** — cite 8c.

**Setup (8c).** Synthetic 1-D-correlated GLM (LCM's home, as Exp 7), `d` ∈ {500,
1000, 1500}, **10 seeds**, ±95% CI, library-default fit and nugget. Joint LL is
evaluated over batches of test points under the full predictive covariance.

| d | marginal-LL (diag / LCM / full) | joint-LL (diag / LCM / full) | **joint gap closed by LCM** |
|---|---|---|---|
| 500 | −0.823 / −0.826 / −0.823 | −0.830 / −0.823 / −0.820 | **62% ± 49%** |
| 1000 | −0.922 / −0.933 / −0.919 | −0.946 / −0.927 / −0.907 | **52% ± 16%** |
| 1500 | −1.076 / −1.110 / −1.057 | −1.116 / −1.097 / −1.033 | **23% ± 12%** |

**Outcome.** The reframing works, and the contrast within each row is the point:
on **marginal** LL, LCM is *slightly worse* than diagonal (the structured fit's
imperfection costs a little per-point calibration); on **joint** LL over the same
posteriors, LCM is clearly better. This is the project's main positive result,
and it turned a flat negative into a finding.

**Read it honestly, though.** The advantage is **real but modest and shrinks with
`d`** (62% → 52% → 23%), and it is **not statistically significant at d=500**
(±49%). An earlier 3-seed run with a near-zero nugget reported 70–81%; the
10-seed number with the library's default fit is the trustworthy one, and the gap
is sensitive to the nugget fraction — a tuning knob, not a robust large effect.

**Conclusion.** Off-diagonal structure pays off only where the metric rewards
*correlated* predictions. This defines the niche, and everything after Exp 8 is an
attempt to find out how wide that niche actually is.

---

## Experiment 9 — Active learning (does joint UQ buy a downstream win?)

**Why.** Joint predictive uncertainty is exactly what batch decision-making needs:
selecting a *set* of points requires knowing their predictions are correlated, so
you do not pick ten near-duplicates. If Exp 8's advantage is real, greedy batch
selection under the LCM posterior should beat selection under the diagonal one.
This is the natural application-level payoff.

**Setup.** Greedy batch acquisition on the correlated GLM; compare random,
top-k (diagonal), greedy (diagonal), greedy (LCM), greedy (full). Higher =
better. 3 seeds.

| d | random | top-k (diag) | greedy (diag) | **greedy (LCM)** | greedy (full) |
|---|---|---|---|---|---|
| 500 | 132.36 | 138.95 | 140.26 | **137.62** | **141.91** |
| 1000 | 160.76 | 166.21 | 166.99 | **163.83** | **167.92** |
| 1500 | 174.52 | 180.20 | 180.42 | **177.51** | **181.07** |

**Outcome.** A clean negative, and consistently so: **greedy(LCM) is worse than
greedy(diag) at every `d`** — indeed worse than plain top-k diagonal — while
greedy(full) is best, confirming that better joint UQ *does* help here in
principle. So the failure is LCM's fit, not the premise.

**Why it fails — the boundary condition.** Active learning starts in the
low-data regime, `n ≪ d`. LCM's fit residual at `n = 40` is 0.74: with far fewer
samples than dimensions the empirical second-moment matrix is rank-deficient and
there is no reliable correlation structure to fit, so LCM's extra parameters model
noise. Diagonal, having nothing to overfit, degrades gracefully.

**Conclusion.** The joint-UQ advantage requires `n ≳ d` and therefore does **not**
reach the application that most wanted it. This is the single most damaging result
for a practical pitch, and it is a regime limit, not a bug.

---

## Experiment 10 / 10b — Does the *learned* ordering work? How exact must it be?

**Why.** *The most consequential negative.* The learned coordinate `a` is the
paper's distinctive contribution — without it, LCM is a known object (a
Matérn-½/OU state-space GP, i.e. an AR(1) GMRF). So: can gradient descent
**discover** a latent ordering, and how accurate must an ordering be to help?

**A clarification worth stating**, since it is easy to misread this experiment:
the model is permutation-**equivariant** and sorts `a` internally, so it does not
depend on the input order. What is being tested is whether fitting `a` by gradient
descent can *discover* an ordering, as opposed to merely *refining* one it was
already initialized near.

**Setup (Exp 10).** `d=800`, `n=3000`, 3 seeds. Features carry a true 1-D order.
In the `natural` condition that order is the input order; in the `permuted`
condition it is hidden by a random permutation. Variants: LCM with the order
*fixed* to the natural one, *fixed* to a spectral (Fiedler) **seriation** estimate,
and *learned* by gradient descent. `recovery` = \|Spearman\| between the recovered
and the true order.

| condition | LCM-fixed (natural) | LCM-fixed (seriation) | LCM-learned | learned recovery |
|---|---|---|---|---|
| order **known** (natural) | **+58%** | −79% | **+59%** | 1.000 |
| order **latent** (permuted) | 0% (→ diagonal) | −79% | **−84%** | **0.028** |

**Setup (Exp 10b — fragility).** `d=400`, 3 seeds. Degrade a known-correct order
by adding position noise, and find the accuracy at which LCM stops beating
diagonal.

| \|Spearman\| | 1.000 | 0.9999 | 0.9997 | 0.9994 | **0.998** | 0.991 | 0.967 |
|---|---|---|---|---|---|---|---|
| joint-LL | −0.853 | −0.852 | −0.852 | −0.858 | **−0.870** | −0.885 | −0.892 |
| beats diagonal (−0.865)? | ✔ | ✔ | ✔ | ✔ | **✘** | ✘ | ✘ |

**Outcome.** Three findings, each sharper than the last:

1. **Given the correct order, LCM works** (+58% fixed, +59% learned-but-correctly-
   initialized). The method is sound when its assumption is satisfied.
2. **Gradient descent cannot discover a latent order.** Recovery is
   \|Spearman\| = **0.028** — indistinguishable from random. Fitting `a` is a
   continuous relaxation of a discrete permutation search; the internal `sort()`
   enforces monotonicity but gives gradients no way to cross permutation basins.
   Spectral seriation does better but is wildly seed-dependent (0.31 / 0.70 /
   0.99, mean 0.67).
3. **The tolerance is brutally tight.** LCM beats diagonal at \|Spearman\| ≥
   **0.999** but already **fails at 0.998**. So even seriation's best seed (0.99)
   is not good enough.

**A failure mode worth naming.** With a wrong order, the *fixed* variant degrades
**gracefully to diagonal** (0%) — it simply finds no structure. The *learned*
variant lands at **−84%**, *worse than diagonal*: it overfits into a confidently
wrong chain. Learning the coordinate is not a free option; it is actively harmful
when the order is unknown.

*Caveat (seed variance):* in 1 of 3 fragility seeds diagonal actually beat full
(−0.830 vs −0.836) — the premise "full > diagonal" is itself noisy at this size.
The threshold conclusion holds on the mean, but individual seeds vary.

**Conclusion.** The learned coordinate — the paper's novel element — does no
useful work in this setting: it cannot recover unknown orders, and it is
unnecessary when the order is known. The honest scoping is to **restrict the
method to known-order applications and drop the "learned" claim.** Combined with
the fragility threshold, pursuing order recovery has poor expected value: a
method would essentially have to solve the permutation exactly.

---

## Experiment 11 — Real ordered data

**Why.** Exp 8's positive was synthetic, on data generated to match LCM's
assumption. The claim only means something if it survives on real data with a
genuine, externally-given 1-D order.

**Setup.** Two datasets with a natural ordering, 5 seeds each.
**Tecator NIR** — near-infrared spectroscopy predicting fat content; features are
absorbances at ordered wavelengths (`n=144`, `d=100`, so `n ≳ d`).
**CO₂-AR(64)** — Mauna Loa CO₂ with 64 autoregressive lags (`n=1332 ≫ d=64`).

| dataset | metric | Diag | **LCM-fixed** | **LCM-learned** | Full | gap closed (fixed) |
|---|---|---|---|---|---|---|
| Tecator NIR | **joint** | −3.5395 | **−3.4707** | −3.4681 | −3.4717 | **102.6% ± 13.0** |
| Tecator NIR | marginal | −3.6160 | −3.5779 | −3.5779 | −3.5786 | beats diag in 5/5 seeds |
| CO₂-AR(64) | **joint** | −0.6179 | −0.6138 | −0.6147 | −0.6122 | 94.5% ± 81.3 |
| CO₂-AR(64) | marginal | −0.6136 | −0.5781 | −0.5885 | −0.6124 | beats diag in 5/5 seeds |

**Outcome.** On **Tecator** the result is strong and tight: LCM closes ~**100%**
of the diagonal→full joint gap (CI ±13%), i.e. it matches the exact posterior, and
it also beats diagonal on *marginal* LL in every seed — the one place the marginal
metric rewards it, because here the 1-D assumption is genuinely true rather than
approximately so. This is the project's best real-data evidence.

On **CO₂** the point estimate is similar but the CI is ±81% and meaningless: with
`n ≫ d` the diagonal→full gap is itself tiny (0.0057 nats), so the ratio divides
by near-zero noise. Read it as "no meaningful gap to close", not as a second
confirmation.

**The decisive detail.** In both datasets `LCM-learned ≈ LCM-fixed` — they differ
by 0.0090 nats on Tecator against a 0.0678-nat gap (13%), and by 0.0010 against
0.0057 on CO₂. The **learned coordinate is inactive** where the method works,
because the order is already known. This independently corroborates Exp 10 on real
data.

**Conclusion.** The joint-UQ advantage is real and reproduces outside synthetic
data — on **one** dataset, in the `n ≳ d`, known-order regime. That is a genuine
but narrow result, and the honest statement of the project's scope.

---

## Experiment 12 — Cost, complexity, and whether LCM and KFAC are even comparable

**Why.** Experiments 1–6 compared LCM and KFAC on *quality* only, and concluded
KFAC dominates. That conclusion is incomplete in two ways: it never priced the two
methods, and it never asked whether they apply to the same problems. Both turn out
to matter, and one of them reverses part of the framing.

### (a) Storage depends on layer *shape*, not size

KFAC stores `in² + out²` per layer; LCM stores `3·in·out`. So LCM is cheaper
exactly when `in² + out² > 3·in·out`, i.e. when the **aspect ratio**
`r = in/out` exceeds `(3+√5)/2 ≈ 2.618` — independent of how big the layer is.

| layer | `r` | P | KFAC storage | LCM storage | cheaper |
|---|---|---|---|---|---|
| 64×64 | 1.0 | 4 160 | 12 480 (3.00×P) | 12 480 (3.00×P) | **KFAC** (tie) |
| 128×64 | 2.0 | 8 256 | 24 832 (3.01×P) | 24 768 (3.00×P) | LCM (marginal) |
| 168×64 | 2.6 | 10 816 | 36 712 (3.39×P) | 32 448 (3.00×P) | **LCM** |
| 256×64 | 4.0 | 16 448 | 74 112 (4.51×P) | 49 344 (3.00×P) | **LCM** (1.5×) |
| 512×64 | 8.0 | 32 832 | 270 976 (8.25×P) | 98 496 (3.00×P) | **LCM** (2.8×) |
| 84×10 | 8.4 | 850 | 7 360 (8.66×P) | 2 550 (3.00×P) | **LCM** (2.9×) |
| 2048×10 | 204.8 | 20 490 | 4 196 572 (204.8×P) | 61 470 (3.00×P) | **LCM** (68×) |

**This corrects a claim made earlier in the project.** `PROJECT_REPORT.md` stated
LCM was "strictly cheaper than … KFAC". It is not: on a **square** layer KFAC
stores `2m²` against LCM's `3m²`, so KFAC wins by 1.5×. LCM's storage advantage is
real but conditional on rectangular layers — which, admittedly, is the common case
for classifier heads and feature-extractor outputs.

### (b) Time: KFAC is cheaper on every axis

Measured on a trained model, last layer 84→10 (`exp12_cost_time.csv`):

| method | construct posterior | one log-det | closed form? |
|---|---|---|---|
| Diag | 296 ms | 0.061 ms | ✔ |
| **KFAC** | **128 ms** | **0.052 ms** | ✔ single pass |
| **LCM** | **7 340 ms** | **0.680 ms** | ✘ **500 Adam steps** |
| Full | 6 722 ms | 1.876 ms | ✔ |

**LCM is ~57× more expensive to construct than KFAC** — and roughly as expensive
as *full* Laplace — because KFAC accumulates its factors in one closed-form pass
while LCM must solve a non-convex optimization. Its log-det is also 13× slower
than KFAC's.

Asymptotically there is **no crossover**. KFAC's log-det is a single fused
elementwise reduction `Σ log(λ_A λ_G + δ)`; LCM's associative scan carries 4× the
data and `O(log P)` kernel launches:

| P | KFAC log-det | LCM log-det | ratio |
|---|---|---|---|
| 1 000 | 0.006 ms | 0.416 ms | 67× |
| 20 000 | 0.033 ms | 3.98 ms | 122× |
| 100 000 | 0.082 ms | 13.2 ms | 162× |
| 1 000 000 | 0.258 ms | 145 ms | 562× |

**This qualifies the "systems contribution" framing.** The associative-scan log-det
is a genuine 15–64× win over the *sequential LCM* implementation, and it is what
makes LCM feasible where dense Laplace OOMs — but it is **not** a speed win over
KFAC. The honest statement of LCM's cost advantage is *memory on rectangular
layers*, not speed.

### (c) They are not baselines for the same problem

For a **scalar-output** model — the GLM setting of Exp 7–11, where LCM's one real
advantage lives — the Kronecker factorisation `A ⊗ G` has `G ∈ R^{1×1}`, so it
collapses to the dense `d×d` input covariance. This is an *identity*, not an
approximation:

| d | KFAC vs Full, rel-Frobenius | KFAC storage | Full storage | LCM storage |
|---|---|---|---|---|
| 100 | 1.16e-06 | 10 102 | 10 000 (1.010×) | **300** |
| 200 | 1.30e-06 | 40 202 | 40 000 (1.005×) | **600** |
| 400 | 1.53e-06 | 160 402 | 160 000 (1.003×) | **1 200** |

The residual is float32 noise. **KFAC *is* full Laplace here** — `O(d²)` memory,
`O(d³)` compute, zero compression. That is why KFAC does not appear in the Exp
7–11 tables: it is not a cheap baseline in that setting, it is the expensive
reference already reported as `full`. (The experiment scripts noted this in their
docstrings — "KFAC does not apply (no layers)" — but the reasoning had never
reached this document, so its absence looked like an unexplained omission.)

**Outcome.** The two methods encode different curvature geometries and are not
substitutes:

| | KFAC | LCM |
|---|---|---|
| **assumes** | curvature factorises as (input cov) ⊗ (output cov) | curvature decays exponentially along a 1-D order |
| **requires** | layered architecture | a known/learnable 1-D ordering |
| **storage** | `Σ(in²+out²)` — wins on square layers | `3P` — wins at aspect ratio ≳ 2.6 |
| **construction** | closed form, one pass | **iterative, ~57× costlier** |
| **degenerates to** | full Laplace when `out=1` | diagonal Laplace when `w=0` |
| **home** | neural-network weight space | ordered features (spectra, lags, position) |

Neither is a special case of the other. In the overlap — NN weight space — KFAC's
assumption holds and LCM's does not, which is exactly what Exp 1–6 measured. In
LCM's home, KFAC is unavailable as a cheap method.

**Conclusion.** "LCM loses to KFAC" is true, but only inside KFAC's home, and it
was a quality-only verdict. Priced properly the picture is sharper and *more*
favourable to a correct scoping: LCM buys memory on rectangular layers and pays
for it in construction time, and the setting where it genuinely wins on quality is
one where KFAC provides no compression at all. The right claim is not "LCM is a
cheaper KFAC" but **"LCM covers a curvature geometry KFAC cannot express, at `3P`
memory"** — and the honest caveat is that it costs an optimization to build.

---

## Experiment 13 — Is LCM the right model in its own complexity class?

**Why.** Exp 12 established that LCM and KFAC are not substitutes. That leaves a
sharper question unasked for the whole project: LCM was only ever compared against
something *weaker* (diagonal, `d` params) or something *richer* (KFAC, Full). It
was never compared against its **peers** — the other structured PSD families that
also cost `O(d)` parameters. A reviewer goes straight here: *why this 3d
parameterisation and not a banded Cholesky, or a low-rank correction?*

**The competitors**, all fit to the same target with the same objective, the same
Adam budget, and scale-normalisation (below):

| family | form | params |
|---|---|---|
| `diag` | `D` | `d` |
| `chol_band1` | `L Lᵀ`, `L` bidiagonal → **tridiagonal** precision | `2d−1` |
| `chol_band2` | `L Lᵀ`, `L` 2 subdiagonals → pentadiagonal | `3d−3` |
| `diag_rank1` | `D + v vᵀ` | `2d` |
| `diag_rank2` | `D + V Vᵀ`, `V ∈ R^{d×2}` | `3d` |
| **`lcm`** | `D + W K(a) W` | `3d` |

`chol_band1` is the direct "`AAᵀ` that gives a tridiagonal matrix" comparison.

**The distinction that makes this non-obvious.** A banded Cholesky produces a
*sparse* precision. LCM produces a **dense** one: `K(a)` is the exponential kernel,
which is the **inverse** of a tridiagonal matrix (semiseparable), not a tridiagonal
matrix. The two families are close to opposites — sparse-with-support versus
dense-with-decay — despite both being `O(d)` and both being describable as
"chain-like". Which one fits a given curvature is therefore an empirical question,
and the targets below are chosen so that each family has a home.

### (a) Fit quality — every family has a home, and they are different homes

Relative Frobenius error to the target (lower better), `d = 200`:

| target | `diag` (1d) | `band1` (2d) | `band2` (3d) | `rank1` (2d) | `rank2` (3d) | **`lcm` (3d)** |
|---|---|---|---|---|---|---|
| `glm_ar1` — AR(1) GLM GGN (dense semiseparable) | 0.946 | 0.864 | 0.788 | 0.846 | 0.742 | **0.140** |
| `glm_ar1_perm` — same, order hidden | 0.946 | 0.942 | 0.938 | 0.846 | 0.742 | **0.460** |
| `tridiag` — genuinely sparse Markov precision | 0.480 | **0.001** | 0.008 | 0.475 | 0.470 | 0.465 |
| `nn_ggn` — trained last-layer GGN (Kronecker) | 0.916 | 0.903 | 0.885 | 0.831 | 0.760 | **0.754** |

Beyond LCM's budget, to price its structural advantage in parameters:

| target | `rank4` (5d) | `rank8` (9d) | `rank16` (17d) | `band4` (5d) | **`lcm` (3d)** |
|---|---|---|---|---|---|
| `glm_ar1` | 0.571 | 0.343 | 0.147 | 0.651 | **0.140** |
| `glm_ar1_perm` | 0.571 | 0.343 | 0.147 | 0.930 | 0.460 |
| `tridiag` | 0.459 | 0.439 | 0.400 | 0.020 | 0.465 |
| `nn_ggn` | **0.633** | **0.417** | **0.251** | 0.858 | 0.754 |

**Three findings, and they do not all favour LCM.**

1. **In its niche, LCM is dramatically parameter-efficient.** On `glm_ar1` it
   reaches 0.140 with `3d` parameters; low-rank needs **`17d`** (rank-16, 0.147) to
   match it. That is a ~5.7× parameter-efficiency advantage, and it is exactly what
   the structure buys: the GGN of ordered correlated features is `XᵀX ≈ n·K(a)`,
   which *is* LCM's form.
2. **The banded Cholesky has its own home, and LCM cannot enter it.** On a
   genuinely sparse tridiagonal precision `chol_band1` is exact (0.001, KL 0.000)
   while LCM manages only 0.465 — barely better than diagonal. The two families are
   complementary, not competing.
3. **On real NN curvature, low-rank beats LCM per parameter.** LCM is narrowly best
   *within* `3d` (0.754 vs 0.760), but `rank4` at `5d` already beats it (0.633) and
   it keeps improving (`rank8` 0.418, `rank16` 0.251) while LCM has no comparable
   knob. Low-rank is also **order-independent**, so it avoids the fragility of
   Exp 10 entirely. For weight-space Laplace, low-rank is the better structured
   family.

### (b) Joint predictive LL — no same-class competitor closes the gap

The metric that actually mattered (Exp 8c protocol, `d=1000`, 5 seeds, order known):

| model | params | joint gap-to-full closed |
|---|---|---|
| `chol_band1` | 2d | **−65.3% ± 22.3** |
| `chol_band2` | 3d | **−67.3% ± 56.0** |
| `diag_rank1` | 2d | −4.0% ± 4.0 |
| `diag_rank2` | 3d | −2.4% ± 5.1 |
| **`lcm`** | 3d | **+86.9% ± 17.8** |

This is the strongest result in the project's favour. In LCM's niche, on the
decisive metric, **every same-class competitor closes essentially none of the
diagonal→full gap, and the banded ones are actively harmful** — a banded Cholesky
fit to a dense-decaying GGN produces a confidently wrong posterior, much as a
wrongly-ordered LCM does (Exp 10). Only LCM closes the gap.

### (c) A fitting deficiency, found in passing — the headline number was understated

All Exp 13 fits **scale-normalise the target** (fit to `T/s`, multiply the result
by `s`; every family here is positively homogeneous, so this is exact). It was
added because low-rank diverged at GGN scale (~1e5) with Adam at `lr=5e-2` — the
same class of bug as the original fit-init scaling error.

It also changes LCM's headline result. Re-running the Exp 8c protocol with
normalisation, on **identical data and identical seeds** (`joint_diag` and
`joint_full` match to the last digit, so the LCM fit is the only variable):

| seed | Exp 8c (`auto_init`) | Exp 13 (scale-normalised) |
|---|---|---|
| 0 | 42.6% | **75.2%** |
| 1 | 22.0% | **69.2%** |
| 2 | 46.8% | **75.5%** |
| 3 | 101.2% | **118.3%** |
| 4 | 57.6% | **96.4%** |
| **mean** | **54.1%** | **86.9%** |

Better on **every seed**. The library's `auto_init` (which rescales `w`/`d` from
the target diagonal) does not fully solve the scale problem; normalising the target
itself does. So **the reported 62%/52%/23% headline numbers are conservative** —
they understate LCM for a fitting reason, not a modelling one. Exp 8c has not been
re-run under normalisation; doing so is the single highest-value follow-up, and
`LCMCovariance.fit_dense` should probably normalise internally.

**Conclusion.** LCM survives the comparison against its own complexity class, but
the verdict is split and worth stating precisely. *In its niche* it is not merely
better than its peers, it is better by a margin no amount of low-rank or banding
reaches at comparable cost — the `3d ≈ 17d` result and the +87%/−65% joint-LL
split are the clearest evidence in the project that the structure is doing real
work. *Outside* its niche, on neural-network weight curvature, low-rank dominates
it per parameter and carries no ordering fragility, which is a further argument —
independent of Exp 4a and the audit — against pursuing weight-space Laplace.

---
---

## Implementation notes affecting these results

Three bugs were found and fixed during the campaign; two changed conclusions.
Full detail in `../PROJECT_REPORT.md` §6.6.

1. **Fit-initialization scaling (our bug, in the experiment scripts).** The init
   was scaled for the unnormalized GGN while `fit_frobenius` targets `GGN/N`. It
   produced a false conclusion that the matrix-free fit was weak (0.53 vs 0.22).
   After the fix the two agree (0.2197 vs 0.2198 at d=1500).
2. **Inverse-softplus overflow (genuine library bug).** `log(expm1(x))` overflows
   to inf/NaN for moderately large diagonal scales, and Adam then diverged
   *silently*. Fixed with a threshold-based `_inverse_softplus` plus data-driven
   `auto_init`.
3. **`kernel_matvec` coordinate-range overflow (genuine, serious).** The `O(d)`
   prefix-sum factorization overflowed once the *coordinate range* exceeded ~709
   in float64, returning silent `NaN` — and it broke precisely the *scalable*
   path, invalidating every `d ≥ 5000` row of the first Exp 7 run. Fixed by
   blocked rescaling (~2× cost). The Kalman primitives were unaffected.
4. **Ordering diagnostic reference (our bug).** Exp 10/10b measured recovery
   against `argsort(perm)` instead of `perm`, mis-stating the *seriation* number
   (reported 0.03; true 0.31–0.99). Substantive conclusions were unaffected — the
   joint-LL columns never depended on it — but the corrected numbers make the
   finding sharper.

**Performance.** The log-determinant uses the paper's associative 2×2-matrix
(Möbius) scan, **15–64× faster** than the sequential version; the mean recursion
was later ported to a second scan, giving a further **16× at d=2000 / 48× at
d=10000** on `quad_inv`/`nll`. See `../Laplace/laplace/LCM_PROVENANCE.md` for the
line-by-line correspondence with the authors' reference implementation. The Exp 4
scaling table above predates both and is marked accordingly.

---

## Summary & limitations

**What the experiments establish.**

1. **Fidelity (positive).** LCM is a strict, near-diagonal-cost improvement over
   diagonal Laplace: 1.6–2.9× lower KL to the exact posterior (Exp 1), and it
   recovers chain-structured curvature essentially perfectly (Exp 4b, 1e-4).
2. **Cost (mixed).** `3P` covariance storage and an exact, associative-scan
   log-determinant that remains computable where the dense posterior OOMs (Exp 4).
   But priced against KFAC (Exp 12): LCM wins storage only above ~2.6:1 aspect
   ratio, and *loses* on construction time (57x) and log-det time (13-562x). The
   cost advantage is memory on rectangular layers, not speed.
3. **Marginal predictive UQ (negative).** The fidelity advantage does not carry
   over to *marginal* predictive uncertainty: LCM ties a marginal-likelihood-tuned
   diagonal on UCI (Exp 2), saturates on MNIST and loses on CIFAR-10 (Exp 3), and
   is not rescued by epistemic OOD scores (Exp 5). KFAC and Full dominate on
   fully-connected/convolutional models.
4. **Joint predictive UQ (positive — the main result).** Over *batches*, LCM
   closes 62%/52%/23% of the diagonal→full gap at d=500/1000/1500 (Exp 8c,
   10 seeds) and ~100% on Tecator NIR (Exp 11).
5. **Mechanism (explanatory).** A single 1-D Markov chain cannot embed generic
   NN-weight curvature (Exp 4a plateau at 0.66) because layer curvature is
   Kronecker, not chain. Per-layer blocking recovers only 12–22% of that (Exp 6).
6. **LCM beats its own complexity class — in its niche only (Exp 13).** Against
   banded-Cholesky (`AAᵀ`, tridiagonal precision) and low-rank peers at equal
   budget: in the ordered-feature niche LCM needs `3d` where low-rank needs `17d`,
   and closes +87% of the joint gap where every peer closes ≈0% or worse. On real
   NN curvature the verdict reverses — `rank4` at `5d` beats LCM at `3d`, with no
   ordering fragility.
7. **The learned coordinate does no useful work (negative, central).** It cannot
   discover a latent order (recovery 0.028, Exp 10), the tolerance is
   \|Spearman\| ≥ 0.999 (Exp 10b), and it is inactive when the order is known
   (Exp 11). With an unknown order it is *worse than doing nothing* (−84% vs 0%).

**Practical guidance.** Use **LCM over diagonal** when predictions are consumed
*jointly* (batch decision-making, experimental design), the variables have a
**known** 1-D order (spectra, time lags, genomic position), and `n ≳ d`. Prefer
**KFAC/Full** for neural-network weights with exploitable layer structure — and
note KFAC is also cheaper to build there, so the choice is not a cost trade-off.
Reach for LCM's `3P` storage only where the layer is strongly rectangular
(Exp 12a) or where there are no layers at all and dense is infeasible. Do not
expect the learned ordering to find structure you cannot already name.

**Limitations of this evaluation.**

1. **Narrow niche.** The only demonstrated advantage is joint predictive UQ for
   known-1-D-ordered features with `n ≳ d` — and it shrinks with `d` and is not
   significant at d=500.
2. **One real dataset.** Tecator is the only real positive; CO₂'s CI (±81%) makes
   it uninformative. More ordered-feature datasets (NIR/Raman beyond Tecator,
   genomics with LD structure, climate transects) are the highest-value next step.
3. **Seed counts vary.** The headline joint result was re-run at 10 seeds, but
   several earlier experiments use 2–3 and should not be over-read.
4. **`LCMLaplace` still materializes the dense GGN**, capping the *class* at
   moderate `P`, even though the matrix-free `fit_frobenius` path is validated to
   `d = 20 000` (Exp 7). Wiring per-sample gradients into `LCMLaplace` is the
   clean fix.
5. **Implementation scaling is `O(d^1.44)`, not `O(d log d)`**, because the
   blocked kernel's sequential block count grows with the coordinate range. An
   associative-scan block carry — the trick already used for the log-determinant —
   should recover the intended asymptotics. Still open.
6. **Formulation B tested at one setting.** Fitting LCM to the *covariance*
   instead of the precision was checked on the GLM only (and lost decisively —
   `AUDIT.md` §3), not across the campaign.

---

## Open directions (proposed, not yet run)

Four directions raised after Exp 13. **Nothing below has been measured** — each is
a hypothesis with a stated rationale and a pre-registered prediction, so that
running it can confirm *or* refute rather than rationalise. Ordered by expected
value.

### D1. Re-run the headline under scale-normalised fitting — **do this first**
Exp 13(c) showed the reported gap-closed numbers are understated by a fitting
deficiency: 54% → 87% at `d=1000` on identical seeds. Exp 8c (10 seeds, 3 values of
`d`) should be re-run with the normalisation, and `LCMCovariance.fit_dense` should
normalise internally so every downstream user gets it.
*What it gives:* the project's central claim, correctly stated. It may also change
the `d`-dependence — the 62%/52%/23% decay with `d` could partly be the fit
degrading rather than the method degrading, which would materially alter the
"shrinks with `d`" caveat that currently limits the paper.

### D2. Are we beating low-rank? (partly answered — and the answer is *conditional*)
Exp 13(a) already settles this, and the split matters:
- **In the ordered-feature niche: yes, decisively.** LCM needs `3d` where low-rank
  needs `17d`, and on joint LL it is +87% against low-rank's −2%.
- **On NN weight curvature: no.** `rank4` at `5d` beats LCM at `3d`, and rank
  8/16 pull further ahead, with no ordering fragility.

*Consequence, and it is the one the question anticipated:* since LCM does **not**
beat low-rank on weight curvature, scaling that setting up is not worth it. This
now agrees with `AUDIT.md` §4 (the Kronecker-chain gap widens with size) from a
second, independent direction. What remains open is a rank *sweep in the niche* —
where does low-rank cross LCM as `ρ → 1`, and does a **hybrid** `D + W K(a) W +
V Vᵀ` beat either? The hybrid is the interesting object: it could let the chain
carry the local decay while a few dense directions absorb what the chain cannot,
which is precisely the `nn_ggn` failure mode.

### D3. Could learning the rank help LCM's ordering problem?
*Rationale:* the ordering failure (Exp 10, recovery 0.028) is a discrete search
that gradients cannot perform. A low-rank term is order-**independent**, so a
hybrid model could absorb the long-range structure that a mis-ordered chain
currently tries — and fails — to represent, potentially removing the catastrophic
−84% collapse and leaving graceful degradation.
*Prediction:* it will not *fix* ordering recovery (nothing in the parameterisation
makes the permutation search easier) but it should convert the failure mode from
"confidently wrong" to "falls back to low-rank", which is the difference between
−84% and ≈0%. That alone would make LCM safe to use without knowing the order.
*What it gives:* if confirmed, the method becomes deployable without the
`|Spearman| ≥ 0.999` precondition that currently blocks every real application.

### D4. Annealing the nugget to make the ordering learnable
*Rationale:* the loss surface over `a` is non-convex because a small diagonal
nugget `d` makes the kernel term dominate, so the fit is sensitive to every
pairwise distance at once. Starting with a **large** diagonal (the model is then
nearly diagonal, and the objective in `a` is nearly flat but well-conditioned) and
annealing it **down** is graduated non-convexity / a homotopy continuation — the
standard remedy for exactly this class of problem, and the same idea as
deterministic annealing in clustering.
*Prediction:* worth testing, but temper expectations. Exp 10b's fragility threshold
(`|Spearman| ≥ 0.999`, with 0.998 already failing) means annealing must deliver a
*near-exact* permutation, not merely a better one — and the coarse-to-fine schedule
is most natural at recovering coarse structure first. Spectral seriation already
reaches 0.99 on a good seed and is still insufficient. So the realistic upside is
"better than 0.028 but still under threshold", which would be scientifically
interesting and practically useless.
*What it gives:* cheap to run (a schedule on one existing hyper-parameter, no new
model) and it closes out the ordering question either way. If it *did* clear 0.999
it would be the single most valuable result in the project, since it would unlock
the unknown-order case that every negative result traces back to.

### D5. Very wide layers (10⁴+ units) — is there a regime where LCM is the only option?
*Rationale:* Exp 12 measured KFAC storage at `in² + out²`. At `in = 10⁴` that is
`10⁸` floats (0.4–0.8 GB per layer) while LCM needs `3P`. There is therefore a
width at which KFAC becomes infeasible and LCM does not — and modern architectures
(MLP blocks in transformers, wide classifier heads, embedding layers) live there.
*Prediction, stated honestly:* this is a **memory-feasibility** argument, not a
quality argument, and Exp 13 has weakened it — at those widths low-rank is also
`O(kd)`, order-independent, and fits real NN curvature *better* than LCM. So the
likely finding is "LCM is feasible where KFAC is not, but low-rank is feasible
there too and better". The experiment worth running is therefore three-way
(LCM vs low-rank vs KFAC-where-it-fits) at increasing width, and the question it
answers is whether LCM has *any* uncontested regime in weight space. Based on
Exp 13(a) the honest expectation is that it does not, and that the niche remains
ordered *features*, not wide *layers*.

---

## Reproduction

```bash
python experiments/run_all.py          # all 15 experiments, writes results/*.csv
python experiments/exp8c_joint_ci.py   # the headline joint-UQ result (10 seeds)
python experiments/exp12_cost.py       # cost/complexity vs KFAC
python experiments/exp13_sameclass.py  # vs banded-Cholesky / low-rank peers
pytest Laplace/tests/test_lcm.py -q    # 16 tests: fast paths vs dense references
```

Results are written to `experiments/results/`. Related documents:

| document | contents |
|---|---|
| `../PROJECT_REPORT.md` | end-to-end narrative, **design rationale** (§6), consolidated limitations |
| `../AUDIT.md` | adversarial audit — is the port faithful, would a bigger model help? |
| `../PUBLICATION_ASSESSMENT.md` | publication verdict and venue path |
| `COMPELLING_USE.md` | the Phase-2 lead-by-lead assessment |
| `../Laplace/laplace/LCM_PROVENANCE.md` | line-by-line diff against the authors' reference implementation |
| `../LCM_REPORT.html` | synthesized report for a newcomer |
