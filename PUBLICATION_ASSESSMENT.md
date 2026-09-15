# Publication assessment: is there a paper in the LCM work?


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


Review of everything executed in this project (library fork, 13 unit/integration
tests, experiments 1–9 + controlled ablations + scaling benchmarks), checked
against the literature. Written 2026-07-16.

## Verdict

**Yes — there is a publishable paper here, but only under an honest framing.**
The evidence supports a *method + characterization* paper ("a near-linear
structured covariance family for Laplace approximations, with a precise account
of when it helps — joint predictive uncertainty in well-determined,
1-D-correlated problems — and when it does not"), targeted at **TMLR** or an
approximate-inference **workshop (AABI / NeurIPS-BDL)** first. The evidence does
**not** support a main-conference "new SOTA Bayesian-DL method" paper, and two
gaps (real data; a fixed-ordering baseline) must be closed before *any*
submission. Details below.

---

## UPDATE (post-hardening, 2026-08-09)

The four blocking gaps were then addressed. Two results **materially revise** the
verdict — read this before §1–§7.

**Gaps resolved:**
- **Matrix-free fit (was "the main technical debt")** — resolved. The weak
  matrix-free fit was an experiment-script init bug, *not* a real weakness; with
  correct scaling `fit_frobenius` matches the dense fit (0.2197 vs 0.2198 @ d=1500).
  Diagnosing it surfaced a genuine library bug (inverse-softplus overflow → NaN),
  now fixed with a stable helper + data-driven `auto_init` + regression test
  (commit `db6a554`).
- **A second, more serious silent-NaN bug was then found in `kernel_matvec`** — the
  very primitive that makes the matrix-free fit `O(d)`. It factored `exp(a_j−a_i)`
  through a global offset, overflowing whenever the *coordinate range* exceeded
  ~709 (float64), so `fit_frobenius` returned NaN for long chains — i.e. **the
  scalable path was broken for exactly the large-`d` regime the scalability claim
  rests on**, and the first Exp 7 run's `d ≥ 5000` rows were NaN. Fixed by blocked
  rescaling (exact to ~1e-15, finite to range 1e4, ~2× cost; commits `b4d2323`,
  `91b7d36`). **Lesson for the paper: any scalability claim must be demonstrated at
  the target `d`, not extrapolated from small-`d` correctness tests** — two of the
  three bugs in this project were silent numerical failures in exponential-family
  primitives that small-scale tests passed.
- **Statistical CIs** — Exp 8c: the joint-vs-marginal headline over **10 seeds
  with 95% CIs** (see `PROJECT_REPORT.md` §7).

**Findings that revise the verdict:**
- **The "learned coordinate" — LCM's main novelty over banded-GMRF / state-space-GP
  models — does NOT survive (Exp 10/10b + controls).** The model is
  permutation-equivariant (given the right coordinates it works: **+58%** joint-gap
  closed vs **+59%** for the learned-but-correctly-initialised variant), but
  *fitting* `a` cannot **discover** a latent order. On permuted features (3 seeds,
  d=800): learned recovery |Spearman| = **0.028** (≈ random) and joint-LL **−84%**
  (i.e. *worse than diagonal*); spectral seriation recovery is highly variable —
  **0.31 / 0.70 / 0.99 across seeds (mean 0.67)** — and its fitted variant scores
  **−79%**. The fragility curve (Exp 10b, 3 seeds) locates the threshold sharply:
  the chain beats diagonal at |Spearman| ≥ **0.999** and *fails* at **0.998**.
  Since no recovery method reliably reaches 0.999, the learned ordering only works
  when initialised at ≈ the true order — i.e. when the order is already known,
  where it is **inactive/unnecessary** (real-data Exp 11: learned ≈ fixed).
  *Mechanism nuance:* with a wrong order the **fixed**-order variant degrades
  *gracefully* to diagonal (0%, it can decorrelate via the global spacing), whereas
  the **learned** variant overfits into a confidently-wrong chain (−84%).
- **Real data (Exp 11) confirms the niche but with the learned coordinate inactive.**
  On Tecator NIR (known wavelength order, n≈d) LCM closes **~100%** of the
  diagonal→full joint-LL gap and beats diagonal on marginal LL too — a genuine
  real-data positive. But `LCM-learned ≈ LCM-fixed` (order already known), and on
  CO₂-AR (n≫d) the gap is negligible.

**Revised verdict.** Still publishable, but the framing must drop any "learned
1-D embedding of arbitrary correlated variables" claim. What remains is: *a cheap,
near-linear structured Gaussian that recovers full-Laplace joint predictive
uncertainty for **known-1-D-ordered** features, plus a careful characterization of
when structured covariance helps (joint not marginal; chain not Kronecker; needs
n≳d and a near-exact order).* This is close to classical **banded-GMRF /
state-space-GP** modeling, so the paper now lives or dies on (i) the systems
contribution (associative-scan log-det in a maintained library) and (ii) crisp
differentiation from that literature — **not** on novelty of the covariance family.
Workshop → TMLR remains right; a main-conference bar is now clearly out of reach
without new methodology (a working ordering-recovery step, which the fragility
result suggests is hard). The reviewer objection "just band it in the known order"
is now **conceded**, not rebutted: LCM ≈ that, with a faster log-det.

---

## 1. Evidence inventory (everything run, graded)

| # | Experiment | Finding (verified numbers) | Strength | Use in a paper |
|---|---|---|---|---|
| 0 | Core module + 13 tests | All primitives match dense refs at machine precision (kernel matvec 4e-15; Kalman log-det; NLL vs torch MVN; sampler; matrix-free ≈ dense fit) | **Strong** | Methods / reproducibility |
| 1 | Posterior fidelity (4 settings × 3 seeds) | KL to full: LCM **1.6–2.9× lower than Diag** everywhere (e.g. 17.5 vs 51.3 last-layer reg); KFAC always best (1.5) | **Strong** | Core positive result |
| 1b | Correlation sweep (ρ: 0→0.95) | Diag KL grows 58→77; LCM stays ~20–30 | **Strong** | "When structure helps" |
| 2 | UCI regression, 6 datasets × 8 splits | Avg test-LL rank: Full 1.79 < KFAC 2.35 < **LCM 3.52 ≈ Diag 3.54** < MAP 3.79; LCM rescues Diag's collapse on yacht (−2.40 vs −2.91) | Solid | Honest boundary (marginal metrics) |
| 3/3b | MNIST + CIFAR-10 last-layer | MNIST saturated; CIFAR: LCM **worse than Diag** on ECE (.089 vs .054); KFAC/Full best | Solid negative | Honest boundary |
| 4 | Scaling | 3P vs P² storage (469 KB vs 3 GB @ P=20k); exact log-det where dense OOMs | **Strong** | Systems contribution |
| 4b | Ablations: real GGN vs controlled chain | Real last-layer GGN plateaus at **0.66** rel-Frob (vs Diag 0.97); controlled scrambled-AR(1) recovered to **1e-4**, needing *both* learned ordering (else 0.56) and nugget (else 0.15) | **Strong** | The mechanistic core: the model is sound; generic NN curvature is not chain-embeddable |
| 5 | Epistemic OOD (MI/BALD), CIFAR | MAP 0.50 (as theory demands), KFAC/Full 0.72, Diag 0.64, **LCM 0.62** | Solid negative | Honest boundary |
| 6 | Per-layer (block-diag) LCM | KL improves 12–22% vs global chain; still ≠ KFAC | Mild | Appendix |
| 8/8b | **Joint vs marginal predictive LL** (1-D-correlated GLM, n=3000, 3 seeds) | Marginal: LCM ties Diag. **Joint: LCM closes 70–81% of the Diag→Full gap** — *NOTE: this 3-seed number is optimistic; superseded by the 10-seed Exp 8c (62±49% / 52±16% / 23±12% at d=500/1000/1500) — see UPDATE* | key positive, **modest** | Central claim; now on real data too (Exp 11) |
| 9 | Batch active learning | **Fails**: greedy(LCM) 43–55% of oracle info-gain vs greedy(Diag) 83–90%; mechanism isolated — GGN fit residual 0.74 @ n=40 (rank-40) → 0.13 @ n=3000 | Strong negative *with mechanism* | Valuable boundary result |
| — | `logdet_parallel` (associative scan) | Machine-precision match to sequential, differentiable, **15–64× faster** (4.8 ms vs 305 ms @ d=2000) | **Strong** | Systems contribution |
| — | Matrix-free fit quality | `fit_frobenius` residual 0.53 vs dense-fit 0.22 on the ideal GGN | **Weakness** | The main technical debt |

The coherent scientific story across all of it: *structured-covariance gains are
invisible to marginal metrics, real in joint metrics, and bounded by whether the
curvature is chain-embeddable (not Kronecker) and well-determined (n ≳ d).*

## 2. Novelty check against the literature

* **Joint-vs-marginal is established, not novel.** Osband et al.'s line —
  [Epistemic Neural Networks](https://arxiv.org/abs/2107.08924), [The Neural
  Testbed](https://arxiv.org/abs/2110.04629), [Evaluating High-Order Predictive
  Distributions](https://proceedings.mlr.press/v180/osband22a/osband22a.pdf),
  [Evaluating Predictive Distributions](https://openreview.net/forum?id=S7vWxSkqv_M) —
  states exactly our observed pattern ("agents similar in marginals, clearly
  separated in joints"). This *reduces* the novelty of the insight but *raises*
  the credibility of our finding and hands us an accepted evaluation framework
  (and recent momentum, e.g. [joint predictions for batch
  BO](https://arxiv.org/abs/2511.10590)). Frame as: *a structure family that
  cheaply improves joint predictions*, citing Osband as motivation.
* **No banded/tridiagonal/Markov-chain structure appears in the Laplace-for-NN
  literature** (diag, [KFAC](https://openreview.net/pdf?id=Skdvd2xAZ), low-rank,
  last-layer, subnetwork are the known families). The structure-family claim is
  plausibly novel — a thorough related-work pass is still required.
* **"Learn an ordering, then band" exists in statistics**: seriation + banding
  (Isoband/Best-Permutation two-step methods), [permutation-based Bayesian
  inverse covariance](https://arxiv.org/abs/1902.09353), [sparse-Cholesky latent
  structure in ordered data](https://www.researchgate.net/publication/343806366),
  [SPICE](https://www.stat.berkeley.edu/users/bickel/Rothman%20et%20al%202007-spice.pdf),
  [active seriation](https://arxiv.org/pdf/2603.15336). These are mostly
  discrete/two-step. What remains distinctive in LCM: **continuous, differentiable
  latent coordinates** + **exact O(d log d) state-space inference** (Kalman /
  associative scan) + integration as a **Laplace posterior** in a maintained
  library. Cite and differentiate carefully.
* **The parameterization itself is REMIX's** (arXiv 2605.11804, May 2026). Ours is
  the transfer to Bayesian posterior approximation, the fit-to-Fisher connection,
  the scan log-det, the library integration, and the empirical characterization.
  The paper is ~2 months old, so follow-up collision risk is low — verify its
  citations once before submission (today's search quota cut that check short).

## 3. The three claims that survive scrutiny

1. **Method/systems.** A structured covariance family for Laplace approximations
   that strictly generalizes diagonal at `3P` parameters, with exact near-linear
   primitives (sampling, NLL, quadratic forms, and a differentiable
   associative-scan log-det, 15–64× faster than sequential), shipped as
   `hessian_structure="lcm"` in a laplace-torch fork with tests.
2. **Characterization (the science).** Fidelity gains over diagonal are real and
   grow with curvature correlation, are invisible to marginal predictive metrics,
   and materialize in *joint* predictive quality — 70–81% of the diagonal→full gap
   closed in the method's home regime.
3. **Boundaries with mechanism (the honesty).** Kronecker-structured NN curvature
   defeats a single chain (KFAC wins); low-data (n ≲ d) low-rank curvature defeats
   the fit (active learning fails, with the fit-vs-n curve as proof of mechanism).

## 4. Blocking gaps (must fix before any submission)

1. **Real data.** The joint-UQ win (exp8b) is on synthetic AR(1) features where
   the GGN is chain-structured *by construction*. Reviewers will not accept this
   alone. Needed: at least one real dataset with 1-D-decaying correlation —
   NIR/Raman **spectroscopy** regression, **genomics with LD** structure, or
   ordered **time-series/climate transect** features — showing the same
   marginal-tie / joint-win pattern.
2. **The fixed-ordering baseline (most dangerous objection).** In the demo the
   true ordering is *known* (feature index), so a plain banded/AR(1) precision fit
   with fixed order — or seriation-then-band — might match LCM without learning
   anything. The learned ordering only earns its place on a task where the order
   is unknown or latent (permuted features as a control; latent-order data such as
   single-cell pseudotime or unordered sensor layouts as the showcase). exp4b
   (scrambled-chain recovery to 1e-4 vs 0.56 without learning `a`) is the seed of
   this argument — it must be extended from fit-quality to predictive quality.
3. **Matrix-free fit quality.** The headline joint-UQ result used the dense
   `O(d²)` fit; the matrix-free path (the scalability story) fits worse
   (0.53 vs 0.22). Either improve it (full-batch/damped/rank-aware fitting) or
   scope the paper's claims to d where the dense fit is fine (d ≲ few·10³) and
   present the primitives as capability + the fit as an open problem. For TMLR the
   second is acceptable; for a main conference it is not.
4. **Statistical hygiene.** 3 seeds → 5–10 seeds with CIs on the headline tables.

## 5. Publication paths

**Path A — workshop paper (AABI, NeurIPS-BDL): ~1–2 weeks, high confidence.**
"When does structured covariance help the Laplace approximation?" — the full
honest study: fidelity, marginal-vs-joint, mechanism, boundaries, scan log-det.
Material is ~80% written (EXPERIMENTS.md + COMPELLING_USE.md); needs the
fixed-order baseline and ideally one real dataset. Negative-results-with-mechanism
studies are valued at these venues.

**Path B — TMLR full paper: ~4–8 weeks, realistic target.**
Method + systematic study. TMLR's acceptance bar is *supported claims*, not
impact — our honest scoping is a strength there, and the library artifact counts.
Requires all four gap-fixes in §4, including one unknown-order/latent-order
demonstration to justify the learned coordinates against seriation baselines.

**Path C — main conference (AISTATS/UAI more than NeurIPS/ICML): 3–6 months, uncertain.**
Requires new research, not packaging: either (i) solve the low-rank/low-data fit
(would unlock batch AL/BO — a genuine application win), or (ii) a compelling
latent-1-D real application where learned ordering is essential, ideally plus a
recovery/identifiability result for the ordering (exp4b shows empirical recovery;
a proposition would elevate it). Do not attempt with current evidence.

**Recommendation: A → B.** Submit the workshop version to get community feedback
and a timestamp, then extend to TMLR. Treat C as conditional on solving the
low-rank fit, which is the single highest-leverage open problem in the project.

## 6. Anticipated reviews and our answers

| Objection | Answer |
|---|---|
| "Synthetic only" | Fix via §4.1 (real spectroscopy/genomics data) — non-negotiable |
| "Known order ⇒ just band it" | Fix via §4.2 (permuted/latent-order tasks + seriation baselines); exp4b already shows 1e-4 vs 0.56 recovery |
| "Joint metrics are Osband's point" | Cited as motivation; our contribution is the *structure family* and its characterization, evaluated within that framework |
| "KFAC beats you on NNs" | In the paper as a scoped boundary — the target regime (GLMs/ordered features) has no layer structure for KFAC to exploit |
| "Your scalable fit is weak" | Scope claims to dense-fit regime (TMLR) or fix the fitter (conference); primitives' scalability is proven independently |
| "REMIX already did this" | Their contribution is the parameterization for model inversion; ours is Bayesian-posterior transfer + scan log-det + integration + characterization — stated explicitly |

## 7. What must NOT be claimed

SOTA Bayesian-DL uncertainty; active-learning/BO gains (demonstrated failure);
scalability of the *fit* (only the primitives scale today); novelty of the
joint-vs-marginal insight; superiority to KFAC on neural networks.

## 8. Assets already in hand

Forked library (branch `lcm-covariance`, commits 0225c71 → 32a0131 → 9493eee)
with 13 passing tests; 9 reproducible experiments + results CSVs/figures;
EXPERIMENTS.md (paper-style write-up); COMPELLING_USE.md (niche analysis);
this assessment. Estimated completeness: Path A ≈ 80%, Path B ≈ 60%.
