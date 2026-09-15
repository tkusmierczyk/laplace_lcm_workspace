# Publication assessment: is there a paper in the LCM work?

Review of everything executed in this project (library fork, 16 unit/integration
tests, Experiments 1–16 plus controlled ablations and scaling benchmarks), checked
against the literature.

*Originally written 2026-07-16; materially revised 2026-08-09 (hardening) and
**2026-09-15 (Exp 12–16)**. The 2026-09-15 revision is not cosmetic — Exp 16 showed
that the finding which scoped the entire paper was a fitting artifact, and the
verdict below has been rewritten accordingly.*

---

## Verdict (current)

**Yes — and the case is now stronger than at any earlier point, though the new
evidence is thin.**

The paper was previously scoped as *characterization + systems*, because LCM
appeared to need a **known** 1-D ordering, which made it a re-packaging of
classical banded-GMRF modelling. **That limitation has been withdrawn (Exp 16).**
With a scale-normalised fit and a spectral-seriation initialisation that is then
refined, LCM closes **65.4% ± 19.7** of the diagonal→full joint-predictive gap on
data whose ordering is *entirely hidden*, against an oracle-order ceiling of
**89.6% ± 20.8**. That is a working method for the unknown-order case, which is
exactly what the previous assessment said was missing and judged hard to obtain.

Two further results from Exp 13/14 strengthen the scientific core:

* **LCM beats its whole `O(d)` complexity class in its niche** — `3d` parameters
  against low-rank's `17d` for the same fit, and **+93.4%** joint-gap closed where
  every same-budget peer closes ≈0% or is actively harmful. This pre-empts the
  "why not a banded Cholesky / low-rank correction?" objection with data.
* **It wins for the right reason, and we can show the mechanism.** A rank-32 model
  can match LCM's Frobenius fit to the precision (0.191 vs 0.137) and still close
  only **29.3%** of the joint gap, because the posterior covariance here is
  near-banded (`K⁻¹` is tridiagonal) and diagonal-plus-low-rank cannot be. This is
  a genuinely interesting general point — *Frobenius fit is a poor proxy for
  predictive quality* — and it is the kind of result that carries a paper.

**Target: workshop (AABI / NeurIPS-BDL) → TMLR, unchanged. But the blocking gap is
now replication, not invention.** The corrected results rest on **3 seeds at one
`(d, ρ)` setting**, and the rest of the campaign still runs on the deficient fit.
Until Exp 8c/10/10b/11 are re-run under normalisation, the headline numbers in this
project are conservative by an unknown margin and the new ones are under-powered.
Path C (main conference) moves from "do not attempt" to "conditional on that
replication succeeding" — see §5.

---

## 0. The correction that changed the verdict

Exp 13(c) found that scale-normalising the Frobenius fit target (fit to `T/s`,
multiply the result by `s` — **exact**, since the LCM family is positively
homogeneous, not a tuning choice) raised the headline joint-UQ result from 54% to
87% on identical seeds. Exp 10/10b — the project's central *negatives* — used the
same unnormalised fit. Exp 16 re-ran Exp 10's protocol at its exact settings,
changing one variable at a time:

| variant | joint gap closed | order recovery |
|---|---|---|
| **A** as published (uninformed init, unnormalised) | **−94.5% ± 48.8** | 0.028 |
| **B** the same, **scale-normalised** | **+10.2% ± 6.9** | 0.029 |
| **C** seriation **frozen**, unnormalised (as published) | −65.5% ± 41.8 | 0.793 |
| **D** seriation as **init, then refined**, normalised | **+65.4% ± 19.7** | 0.792 |
| **E** seriation **frozen**, normalised | +11.2% ± 7.9 | 0.793 |
| **F** oracle order, refined, normalised | +89.6% ± 20.8 | 1.000 |

A reproduces the published number, so the original run was sound *as executed*.
But **normalisation alone turns −94.5% into +10.2% with the recovered order
unchanged** — the chain was never the problem, the fit was. And seriation works
(+65.4%) only when used as an initialisation that is **then refined**; Exp 10 froze
it, which is worth +11.2%.

**Consequences for the paper:**

1. **"Requires a known order" is withdrawn.** The scoping that forced the
   characterization-only framing is gone.
2. **"With a wrong order LCM is worse than doing nothing" is withdrawn.** It
   degrades gracefully (+10.2%).
3. **The \|Spearman\| ≥ 0.999 fragility threshold is unreliable.** Fragility is
   real — the curve still collapses — but it shifts up ~35 points under
   normalisation, and \|Spearman\| *saturates at 4 decimal places across a 33-point
   swing in outcome*, so it is the wrong ruler. Do not put the 0.999 figure in a
   paper.
4. **Every unnormalised number in this project is conservative by an unknown
   margin.** That includes the 62/52/23% headline.

**What still stands:** gradient descent *alone* cannot discover a latent order
(recovery 0.028, unchanged under normalisation). The method needs seriation to
supply the ordering; the learned coordinate refines rather than discovers. That is
a smaller claim than the paper originally wanted, but an honest and defensible one.

---

## 1. Evidence inventory (everything run, graded)

| # | Experiment | Finding (verified numbers) | Strength | Use in a paper |
|---|---|---|---|---|
| 0 | Core module + **16** tests | All primitives match dense refs at machine precision; both associative scans cross-checked against an independent sequential implementation | **Strong** | Methods / reproducibility |
| 1 | Posterior fidelity (4 settings × 3 seeds) | KL to full: LCM **1.6–2.9× lower than Diag** everywhere; KFAC always best | **Strong** | Core positive |
| 1b | Correlation sweep (ρ: 0→0.95) | Diag KL grows 58→77; LCM stays ~20–30 | **Strong** | "When structure helps" |
| 2 | UCI regression, 6 sets × 8 splits | Avg rank: Full 1.79 < KFAC 2.35 < **LCM 3.52 ≈ Diag 3.54** < MAP 3.79 | Solid | Honest boundary (marginal metrics) |
| 3/3b | MNIST + CIFAR-10 last-layer | CIFAR: LCM **worse than Diag** on ECE (.089 vs .054) | Solid negative | Honest boundary |
| 4 | Scaling | 3P vs P² storage; exact log-det where dense OOMs | **Strong** | Systems |
| 4b | Ablations: real GGN vs controlled chain | Real GGN plateaus at **0.66** rel-Frob; controlled scrambled-AR(1) recovered to **1e-4**, needing *both* learned ordering and nugget | **Strong** | Mechanistic core |
| 5 | Epistemic OOD (MI/BALD) | KFAC/Full 0.72, Diag 0.64, **LCM 0.62** | Solid negative | Honest boundary |
| 6 | Per-layer (block-diag) LCM | KL improves 12–22%; still ≠ KFAC | Mild | Appendix |
| 7 | 1-D-correlated GLM, matrix-free to d=20k | No NaN anywhere after the `kernel_matvec` fix; marginal-LL ties Diag | Solid | Scalability capability |
| 8c | **Joint vs marginal** (10 seeds, CIs) | Joint gap closed **62±49 / 52±16 / 23±12%** at d=500/1000/1500 — *unnormalised; Exp 16 implies these are low* | key positive | Central claim |
| 9 | Batch active learning | **Fails**: greedy(LCM) 43–55% of oracle info-gain vs greedy(Diag) 83–90%; mechanism isolated (fit residual 0.74 @ n=40) | Strong negative *with mechanism* | Valuable boundary |
| 10/10b | Ordering | ⚠ **superseded by Exp 16.** What survives: learned recovery **0.028** | — | Cite only the recovery figure |
| 11 | **Real data** (Tecator NIR) | Closes **~100%** of the joint gap, beats Diag on marginal too, 5 seeds | **Strong** | The real-data anchor |
| 12 | **Cost vs KFAC** | LCM's `3P` beats KFAC only above aspect ratio ≈2.6; KFAC is **57× cheaper to construct** and 13–562× cheaper per log-det. For scalar-output GLMs **KFAC *is* full Laplace** (1.3e-06) | **Strong** | Kills the "just use KFAC" objection in the target regime; forces honesty elsewhere |
| 13 | **Own complexity class** | In niche: LCM `3d` = low-rank `17d`; joint gap **+87%** vs peers ≈0% / −65%. Banded Cholesky has its own home LCM cannot enter | **Strong** | Reviewer-proofing |
| 14 | **Low-rank in depth; large d** | **Matched fit ≠ matched prediction** (rank32: 0.191 fit, +29.3% gap vs LCM 0.137, +93.4%). At d=1e5 LCM 2 MB vs 74.5 GB dense — but low-rank equally feasible and ~70× faster | **Strong** | The mechanism result; also the honest scale story |
| 15 | Ordering fixes | Hybrid promising but CIs overlap. **Nugget annealing fails** (recovery 0.035–0.039 vs 0.100 baseline) | Negative | Appendix / "what we tried" |
| 16 | **Normalisation correction** | See §0 | **Strong** | Reframes the whole paper |

The coherent scientific story: *structured-covariance gains are invisible to
marginal metrics, real in joint metrics, bounded by whether the curvature is
chain-embeddable (not Kronecker) and well-determined (n ≳ d) — and the structure
matters more than the capacity, since a model with 10× the parameters and a better
Frobenius fit predicts far worse.*

## 2. Novelty check against the literature

* **Joint-vs-marginal is established, not novel.** Osband et al. —
  [Epistemic Neural Networks](https://arxiv.org/abs/2107.08924), [The Neural
  Testbed](https://arxiv.org/abs/2110.04629), [Evaluating High-Order Predictive
  Distributions](https://proceedings.mlr.press/v180/osband22a/osband22a.pdf),
  [Evaluating Predictive Distributions](https://openreview.net/forum?id=S7vWxSkqv_M)
  — state exactly our observed pattern. This *reduces* the novelty of the insight
  but *raises* credibility and hands us an accepted evaluation framework. Frame as:
  *a structure family that cheaply improves joint predictions*, citing Osband as
  motivation.
* **No banded/tridiagonal/Markov-chain structure appears in the Laplace-for-NN
  literature** (diag, [KFAC](https://openreview.net/pdf?id=Skdvd2xAZ), low-rank,
  last-layer, subnetwork are the known families). Plausibly novel; a thorough
  related-work pass is still required.
* **"Learn an ordering, then band" exists in statistics**: seriation + banding
  (Isoband / Best-Permutation two-step methods), [permutation-based Bayesian
  inverse covariance](https://arxiv.org/abs/1902.09353), [sparse-Cholesky latent
  structure in ordered data](https://www.researchgate.net/publication/343806366),
  [SPICE](https://www.stat.berkeley.edu/users/bickel/Rothman%20et%20al%202007-spice.pdf),
  [active seriation](https://arxiv.org/pdf/2603.15336).
  **This comparison must be handled more carefully after Exp 16, because our
  working recipe — seriate, then refine — is itself a two-step method, which is
  closer to that literature than the original "learned coordinate" framing was.**
  What remains distinctive, and must be demonstrated rather than asserted:
  (i) the refinement is **continuous and differentiable**, and Exp 16 shows it is
  worth **+54 points** over freezing the seriation (+65.4% vs +11.2%) — that gap
  *is* the contribution; (ii) exact `O(d log d)` state-space inference; (iii) it is
  a **Laplace posterior** in a maintained library, not a covariance-estimation
  method. **Action: a direct head-to-head against seriation-then-band (a fixed
  banded Cholesky in the seriated order) is now the single most important missing
  baseline.** Exp 13 shows a banded Cholesky loses badly *in the natural order*;
  it has not been tested in the *seriated* order, and that is precisely what a
  statistics reviewer will ask for.
* **The parameterization itself is REMIX's** (arXiv 2605.11804). Ours is the
  transfer to Bayesian posterior approximation, the fit-to-Fisher connection, the
  scan log-det, the library integration, and the empirical characterization. See
  `Laplace/laplace/LCM_PROVENANCE.md` for a line-by-line account of what was
  ported versus added — useful for an honest contribution statement.

## 3. The claims that survive scrutiny

1. **Method/systems.** A structured covariance family for Laplace approximations
   that strictly generalizes diagonal at `3P` parameters, with exact near-linear
   primitives and a differentiable associative-scan log-det (15–64× faster than
   sequential; a second scan later gave a further 16–48× on quadratic forms),
   shipped as `hessian_structure="lcm"` in a laplace-torch fork with 16 tests.
2. **A working recipe for unknown orders** *(new, and the strongest claim)*.
   **Seriate → refine → normalise** closes 65% of the diagonal→full joint gap with
   a completely hidden ordering, ~3/4 of the oracle. Currently 3 seeds at one
   setting; replication is the gating requirement.
3. **Characterization (the science).** Fidelity gains over diagonal are real and
   grow with curvature correlation, are invisible to marginal metrics, and
   materialize in *joint* predictive quality — confirmed on real NIR spectroscopy.
4. **Structure beats capacity** *(new)*. A low-rank model with 10× the parameters
   and a better Frobenius fit closes a third as much joint gap, because it cannot
   represent a near-banded posterior covariance. Generalises beyond LCM.
5. **Boundaries with mechanism (the honesty).** Kronecker NN curvature defeats a
   single chain (KFAC wins, and is also far cheaper to build); low-data `n ≲ d`
   defeats the fit (active learning fails, with the fit-vs-`n` curve as proof);
   low-rank beats LCM on NN weight curvature per parameter.

## 4. Blocking gaps (must fix before any submission)

The 2026-08 list is largely resolved; the new list is shorter but sharper.

**Resolved:**
- ~~Real data~~ — Exp 11 (Tecator NIR, 5 seeds, ~100% of the gap closed).
- ~~Matrix-free fit quality~~ — was an experiment-script init bug, not a real
  weakness; `fit_frobenius` matches the dense fit (0.2197 vs 0.2198 @ d=1500), and
  runs to `d = 20 000` with no NaN after the `kernel_matvec` fix.
- ~~Statistical hygiene on the headline~~ — Exp 8c, 10 seeds with CIs.
- ~~The fixed-ordering baseline~~ — *partly*. Exp 16 supplies the permuted-order
  control the old §4.2 demanded, and the learned refinement now earns its place
  (+54 points over frozen seriation). But see the new gap 2.

**Open, in priority order:**

1. **Re-run the campaign under normalised fitting.** Exp 8c, 10, 10b and 11 all use
   the deficient fit. This is not optional bookkeeping: the `d`-dependence of the
   headline (62→52→23%) may be partly *fit degradation rather than method
   degradation*, and that caveat currently limits the paper's central claim. Also
   fix `LCMCovariance.fit_dense`/`fit_frobenius` to normalise internally.
2. **Seriation-then-band head-to-head** (see §2). A fixed banded Cholesky in the
   *seriated* order is the strongest remaining "you invented nothing" objection,
   and it is cheap to run.
3. **Power for the new results.** Exp 16 is 3 seeds at one `(d, ρ)`; Exp 13(b)/14(b)
   are 5. The seriate-then-refine result needs a multi-seed study across `d` and
   `ρ`, and on the Exp 11 real datasets, before it can headline a paper.
4. **A better ordering-accuracy metric.** \|Spearman\| saturates at 4 d.p. across a
   33-point swing (Exp 16b). Kendall tau-distance on adjacent pairs, or the achieved
   kernel error, would discriminate where Spearman cannot.
5. **Implementation scaling.** Wall-clock is `O(d^1.44)`, not `O(d log d)`, because
   the blocked kernel's sequential block count grows with the coordinate range. An
   associative-scan block carry should fix it. Weakest part of the systems story.

## 5. Publication paths

**Path A — workshop (AABI, NeurIPS-BDL): ~2 weeks, high confidence.**
"When does structured covariance help the Laplace approximation?" — fidelity,
marginal-vs-joint, mechanism, boundaries, scan log-det, and now the normalisation
correction as a cautionary methods result in its own right. Material is ~85%
written. Needs gap 1 (re-run) and gap 2 (seriated-band baseline).

**Path B — TMLR full paper: ~4–8 weeks, realistic and now better-supported.**
Method + systematic study. TMLR's bar is *supported claims*, not impact — and the
project now has a method (seriate→refine→normalise), a mechanism result
(structure beats capacity), a complexity-class comparison, and a real-data anchor.
Requires gaps 1–3.

**Path C — main conference (AISTATS/UAI): 3–6 months, no longer ruled out.**
The previous assessment said "do not attempt", because it required "a compelling
latent-1-D application where learned ordering is essential" and judged the ordering
problem intractable. **Exp 16 substantially removes that blocker**: the unknown-order
case works. What would make it conference-grade is (i) gap 1–3 replicating cleanly,
(ii) a latent-order real application — single-cell pseudotime, unordered sensor
arrays — where seriation+refinement beats both diagonal and seriation+banding, and
(iii) ideally an identifiability or recovery statement for the refinement step.
This is now a *research plan*, not a wish.

**Recommendation: A → B, with C as a live option** if the replication in gap 1
holds up. Do not skip gap 1 — publishing the conservative numbers would understate
the method and, worse, publishing the withdrawn ordering claims would be wrong.

## 6. Anticipated reviews and our answers

| Objection | Answer |
|---|---|
| "Synthetic only" | Exp 11: Tecator NIR, real wavelength-ordered spectroscopy, 5 seeds, ~100% of the joint gap closed |
| **"Known order ⇒ just band it"** | **No longer conceded.** Exp 16: with a *hidden* order, seriate→refine→normalise closes 65.4% vs 11.2% for frozen seriation — the differentiable refinement is worth +54 points. Still owed: the same comparison against a banded Cholesky in the seriated order (gap 2) |
| "Why not low-rank / a banded Cholesky?" | Exp 13/14: at matched budget LCM closes +93.4% where low-rank closes −29.8% and banded −155.7%; low-rank needs `17d` to match LCM's `3d` fit and *still* closes only 29.3% of the gap, because it cannot represent a near-banded posterior covariance |
| "Joint metrics are Osband's point" | Cited as motivation; our contribution is the *structure family*, its mechanism, and its characterization within that framework |
| "KFAC beats you on NNs" | Conceded and scoped — KFAC wins on NN weights *and* is 57× cheaper to build. But in the target regime (scalar-output GLMs / ordered features) **KFAC is exactly full Laplace** (Exp 12c, 1.3e-06), so it offers no compression and is not an available baseline there |
| "Your scalable fit is weak" | Withdrawn: it was an init bug. `fit_frobenius` matches the dense fit and runs to d=20 000. Remaining honest caveat: wall-clock is `O(d^1.44)`, not `O(d log d)` |
| "REMIX already did this" | Their contribution is the parameterization for model inversion; ours is the Bayesian-posterior transfer, scan log-det, library integration and characterization. `LCM_PROVENANCE.md` documents the boundary line by line |
| "Your earlier draft said the opposite" | If any version circulated: Exp 16 is a clean, controlled retraction with the mechanism identified. Better to lead with it as a methods contribution than to have a reviewer find it |

## 7. What must NOT be claimed

SOTA Bayesian-DL uncertainty; active-learning/BO gains (demonstrated failure);
novelty of the joint-vs-marginal insight; superiority to KFAC on neural networks;
**that LCM is cheaper than KFAC** (it is not — only lower-memory on rectangular
layers, and 57× more expensive to construct); **that gradient descent discovers
orderings** (it does not — seriation supplies them, recovery from an uninformed
init is 0.028); and the **\|Spearman\| ≥ 0.999** fragility threshold, which is an
artifact of a saturating metric and a deficient fit.

## 8. Assets in hand

Forked library (branch `lcm-covariance`) with **16** passing tests and a provenance
document; **20** reproducible experiment scripts + result CSVs/figures;
`experiments/EXPERIMENTS.md` (paper-style write-up, now including a full
experimental-setup section); `PROJECT_REPORT.md` (end-to-end narrative);
`AUDIT.md` (adversarial audit); `COMPELLING_USE.md`; this assessment.

Estimated completeness: **Path A ≈ 85%, Path B ≈ 65%, Path C ≈ 25%** — the Path C
figure being non-zero only because Exp 16 changed what is reachable.
