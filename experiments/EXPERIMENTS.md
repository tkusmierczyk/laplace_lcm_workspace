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

**TL;DR of findings.** LCM is a strict, near-diagonal-cost improvement over
diagonal Laplace in *posterior fidelity* (2–3× lower KL to the exact posterior,
Exp 1), recovers chain-structured curvature exactly (Exp 4b), and is memory-cheap
and feasible where a dense posterior is not (Exp 4). However, this fidelity gain
does **not** reliably transfer to *predictive* uncertainty for weight-space
Laplace: LCM ties a marginal-likelihood-tuned diagonal on regression (Exp 2),
saturates with it on MNIST, and is slightly worse on CIFAR-10 (Exp 3), while
KFAC and full Laplace dominate on these fully-connected/convolutional models.
The ablations pinpoint the cause: a single 1-D Markov chain cannot embed generic
weight-space curvature (Exp 4a residual ≈0.66), even though it captures curvature
perfectly when the correlation structure *is* chain-like (Exp 4b residual
≈1e-4). LCM is therefore best understood as **the strongest diagonal-cost,
architecture-agnostic structured covariance**, not a replacement for KFAC.

## Baselines

All methods share the same trained MAP network and differ only in the structure
of the Hessian/posterior-precision approximation:

* **MAP** — point estimate (homoscedastic Gaussian noise for regression; softmax
  for classification); no weight uncertainty.
* **Diag** — diagonal Laplace (the structure LCM strictly generalizes).
* **KFAC** — Kronecker-factored Laplace (a stronger, layer-structured baseline).
* **Full** — exact dense Laplace (gold-standard reference; tractable only in the
  moderate-`P` / last-layer regime).
* **LCM** — *ours*, the structured Gauss–Markov covariance (`3P` parameters).

(Low-rank Laplace is omitted: it requires an unavailable legacy backend.)

## Common protocol

Every Laplace variant is fit to the same MAP network with the same GGN backend.
Unless stated otherwise, each method's prior precision — and, for regression, its
observation noise — is tuned *per method* by maximizing that method's Laplace
**marginal likelihood** (gradient ascent on the log-hyperparameters), giving
every baseline its own best-case hyper-parameters. Predictives use the linearized
(GLM) Laplace predictive; classification uses the probit approximation. The LCM
is fit to the GGN by minimizing `‖A − H‖_F²` (Adam, 500–600 steps), the Frobenius
objective of Appendix A of the paper. Code is in `experiments/`; results in
`experiments/results/`. Reproduce with `python experiments/run_all.py`
(+ `exp3b_cifar.py`, `exp4b_ablation_controlled.py`).

---

## Experiment 1 — Posterior fidelity

Where the exact full-Laplace posterior is computable, we fix a *common* prior
precision across all methods (isolating the curvature approximation) and measure
how close each structured posterior is to the full one, via
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

**Finding.** LCM reduces the KL to the exact posterior by **1.6–2.9×** relative
to diagonal Laplace in every setting, confirming that the Gauss–Markov chain
captures real off-diagonal curvature the diagonal discards. KFAC — which exploits
the network's layer structure — is more faithful still. The ordering
`KFAC < LCM < Diag` is consistent across all settings and both metrics.

**Ablation — when does structure help?** Varying the feature-correlation level
`ρ` (which controls how off-diagonal the last-layer curvature is; Figure
`results/exp1_whenithelps.png`), the diagonal's KL to the full posterior *grows*
as correlation increases (AR(1): 57.9→76.7 as `ρ`: 0→0.95), while LCM's stays low
and roughly flat (~20–30). LCM is thus increasingly preferable to diagonal as the
curvature becomes more correlated, and never worse across the range.

---

## Experiment 2 — UCI regression uncertainty

Standard single-hidden-layer (50 tanh) BNN regression on six UCI datasets;
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

**Finding.** LCM and diagonal are **statistically tied on average** (rank 3.52 vs
3.54), with Full and KFAC clearly ahead. LCM's behaviour relative to diagonal is
*variance-reducing*: on `yacht`, where diagonal collapses to a badly
over-/under-confident LL of −2.91 (worse than MAP), LCM recovers to −2.40; on
`energy` and `diabetes` LCM also beats diagonal. On the other three datasets LCM
is marginally behind a well-tuned diagonal. Net: for all-weights regression,
LCM's fidelity edge yields robustness (it never collapses like diagonal) but not
a systematic predictive-LL improvement.

---

## Experiment 3 — Image classification calibration & OOD (last-layer)

Small CNNs; **last-layer** Laplace. In-distribution accuracy / NLL / expected
calibration error (ECE), and OOD-detection AUROC by predictive confidence. 3
seeds (MNIST) / 2 seeds (CIFAR-10).

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

**Finding.** MNIST is *saturated*: the MAP network is already well-calibrated
(ECE 0.005), so every Laplace variant slightly *worsens* in-distribution NLL/ECE;
among them LCM attains the best OOD-AUROC (0.968). On the harder CIFAR-10, Laplace
helps in-distribution calibration overall — KFAC/Full cut NLL from 0.576 to ≈0.50
and ECE from 0.068 to ≤0.025 — but **LCM under-performs even diagonal** here
(NLL 0.550 vs 0.543, ECE 0.089 vs 0.054): its imperfect fit to the last-layer GGN
produces a miscalibrated predictive. As on regression, the fidelity advantage
over diagonal does not translate into a predictive-calibration advantage, and
KFAC/Full are clearly preferable for last-layer image models.

---

## Experiment 4 — Cost, scaling, and ablations

**Scaling of the posterior log-determinant** (Figure `results/exp4_scaling.png`).
Covariance storage is `P` (Diag), `3P` (LCM), `P²` (Full):

| P | mem Full (floats) | mem **LCM** | log-det Full (ms) | log-det **LCM** (ms) |
| --- | --- | --- | --- | --- |
| 1 000 | 1.0e6 | **3 000** | 3.0 | 28.9 |
| 5 000 | 2.5e7 | **15 000** | 166.1 | **108.5** |
| 10 000 | 1.0e8 | **30 000** | OOM / — | **346.3** |
| 20 000 | 4.0e8 | **60 000** | OOM / — | **588.6** |

**Finding.** LCM stores the covariance in `3P` floats vs `P²` for the dense
posterior (a >6 500× reduction at `P=20k`) and computes an *exact* log-determinant
where the dense matrix is infeasible (`P≥10k`). The current sequential-Kalman
log-det carries Python-loop overhead, so it only overtakes the dense log-det past
`P≈5k`; the paper's associative 2×2-matrix scan would remove this constant at
`O(P log P)` depth. The memory and feasibility advantages are unconditional.

**Ablation (a) — on a real last-layer GGN (`P=124`):**

| learn `a` | nugget | rel. Frobenius to GGN |
| --- | --- | --- |
| ✓ | ✓ | 0.659 |
| ✓ | ✗ | 0.662 |
| ✗ | ✓ | 0.642 |
| ✗ | ✗ | 0.657 |
| diagonal-only | | 0.973 |

**Ablation (b) — controlled scrambled-AR(1) target (exactly LCM-representable in
some order):**

| learn `a` | nugget | rel. Frobenius to target |
| --- | --- | --- |
| ✓ | ✓ | **0.0001** |
| ✓ | ✗ | 0.153 |
| ✗ | ✓ | 0.560 |
| ✗ | ✗ | 0.580 |
| diagonal-only | | 0.956 |

**Finding.** On the *real* GGN, LCM roughly halves diagonal's Frobenius error
(0.66 vs 0.97) but plateaus at 0.66 — and learning the ordering / adding the
nugget barely move it, because that matrix is simply not 1-D-embeddable (its
intrinsic ceiling). The *controlled* experiment isolates the design choices: when
the target genuinely is a (scrambled, nugget-carrying) chain, LCM recovers it to
1e-4 — but only with **both** a learned ordering (removing it: 0.56) and the
diagonal nugget (removing it: 0.15). Both components are therefore essential, and
the real-GGN plateau reflects the data, not the parameterization. The fit
converges by ≈200 Adam steps (`results/exp4_ablation_steps.csv`).

---

## Summary & limitations

**What the experiments establish.**

1. **Fidelity (positive).** LCM is a strict, near-diagonal-cost improvement over
   diagonal Laplace: 1.6–2.9× lower KL to the exact posterior (Exp 1), and it
   recovers chain-structured curvature essentially perfectly (Exp 4b).
2. **Cost (positive).** `3P` covariance storage and an exact log-determinant that
   remains computable where the dense posterior OOMs (Exp 4).
3. **Predictive UQ (neutral-to-negative).** The fidelity advantage does not carry
   over to predictive uncertainty for weight-space Laplace: LCM ties a
   marginal-likelihood-tuned diagonal on UCI regression (Exp 2), saturates with it
   on MNIST, and is slightly worse on CIFAR-10 (Exp 3). KFAC and full Laplace
   dominate on the fully-connected / convolutional models tested.
4. **Mechanism (explanatory).** A single 1-D Markov chain cannot embed generic
   NN-weight curvature (Exp 4a plateau at 0.66), which is exactly the a-priori
   limitation of a bandwidth-1 structure. LCM helps precisely when the curvature
   has local / 1-D-decaying correlation (Exp 1 ablation, Exp 4b), and reduces to
   diagonal when it does not.

**Practical guidance.** Prefer **LCM over diagonal** when a cheap, architecture-
agnostic structured covariance is wanted, when memory forbids denser structures,
or when the curvature is plausibly locally correlated (it never does worse than
diagonal in fidelity and is far more robust, e.g. `yacht`). Prefer **KFAC/Full**
when the model has exploitable layer structure and predictive calibration is the
target.

**Limitations of this evaluation.** (i) The `LCMLaplace` fit materializes the
dense GGN, so experiments are last-layer / small-`P`; the matrix-free
`fit_frobenius` path (tested in `tests/test_lcm.py`) would extend to large `P`.
(ii) A single global ordering is the model's core assumption; a **per-layer LCM
chain** (KFAC-style block-diagonal) is the natural next step and is directly
motivated by KFAC's superiority here. (iii) The log-det wall-clock uses a
sequential Kalman scan; the associative-scan variant would make it genuinely
`O(P log P)`. (iv) OOD-AUROC uses max-confidence; distributional scores
(predictive entropy / mutual information) may rank methods differently.
