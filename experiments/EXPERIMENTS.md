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
metric evaluated on **in-distribution** test points, where a trace identity pins
the mean predictive variance regardless of covariance structure (Exp 17) — so no
per-point metric could separate the methods there. Switching to **joint** predictive likelihood over a batch exposed a real
advantage (Exp 8c: **87%/84%/64%** of the diagonal→full gap closed at
d=500/1000/1500, 10 seeds, through the library), confirmed on real ordered data
(Exp 11, Tecator NIR: ~100%). The
boundary conditions are sharp: it needs `n ≳ d` (so active learning fails,
Exp 9), and the paper's distinctive *learned* coordinate cannot discover a latent
order (recovery 0.028) and is redundant when the order is already known (Exp 11:
learned ≈ fixed). **The ordering conclusions have since been corrected — see
below and Exp 16.**

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

**Correction (Exp 16) — the ordering catastrophe was a fitting artifact.** The
unnormalised Frobenius fit, not the model, produced Exp 10/10b's central negatives.
On identical data, scale-normalising moves the learned-coordinate result from
**−94.5% to +10.2%** with the recovered order *unchanged*, and seriation from
**−65.5% to +65.4%** when used as an initialisation that is then refined rather
than frozen. So **LCM does not require a known order**: seriate, refine, normalise,
and it recovers ~3/4 of what the oracle ordering buys. What survives is the
narrower claim that gradient descent *alone* cannot discover a latent order.

**Net.** LCM is the strongest diagonal-cost, architecture-agnostic structured
covariance, with one demonstrated real advantage — **joint** predictive
uncertainty for 1-D-ordered features, whose ordering may be *discovered by
seriation* rather than known a priori. It is not a replacement for KFAC on
neural-network weights, and not a competitor to it either: the two encode
different curvature geometries. Where it does win, it wins against its whole
complexity class — and for the right reason: Exp 14(b) shows a low-rank model can
match its *Frobenius* fit and still close only a third as much of the predictive
gap, because it cannot represent the near-banded posterior covariance.

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
| 14 | `exp14_lowrank.py` | Low-rank in depth; behaviour at very large `d` | **matched fit ≠ matched prediction** |
| 15 | `exp15_ordering_fixes.py` | Hybrid LCM+low-rank; nugget annealing | hybrid weak +, annealing **fails** |
| 16 | `exp16_normalisation.py` | Were the ordering negatives a fitting artifact? | **yes — overturns Exp 10/10b** |
| 17 | `exp17_where_offdiagonals_matter.py` | Which queries make off-diagonal covariance matter? | **none found — shift measures variance scale, not structure** |
| 18 | `exp18_kron_lcm.py` | Does KFAC x LCM work as a posterior? | **yes — ties KFAC on 80x less storage; but see the scale control** |
| 19 | `exp19_kron_lcm_scale.py` | Where can Kron-LCM be built and KFAC not? | **i>=32k: 1.5 MiB vs 32 GiB at i=65k** |
| 20 | `exp20_kron_lcm_real.py` | Kron-LCM on real spectra, C=3 | **shift win is calibration, not structure; seriation exact** |
| 21 | `exp21_kron_lcm_cnn.py` | Are *learned* feature maps chain-structured? | **no — effective rank 1.75 of 368; chain closes nothing** |
| 22 | `exp22_reproduce_paper_claim.py` | Does our code reproduce the paper's own claim? | **yes — LCM beats diagonal at every layer** |
| 23 | `exp23_conditional_prediction.py` | Is there *any* query where off-diagonal covariance beats a diagonal that has been given the best possible single variance scale? Predicting held-out points from other revealed points is scale-free, so it can only reflect shape | **yes — the first such query found; chain closes 47% of the gap at correlation 0.9, 71% at 0.99, and ties the diagonal at correlation 0 as it must** |
| 24 | `exp24_batch_acquisition.py` | Same question as a *decision*: does structure change which points you choose to label, measured by error after labelling them? | **yes — chain matches the exact posterior, while a best-scaled diagonal stays 6.5 standard errors worse and loses even to random choice** |
| 25 | `exp25_paper_scale_baselines.py`, `exp25_verify_moments.py` | The paper only compares against a diagonal. Does the chain still win against a low-rank model given the *same* number of stored numbers — and does the paper's fitting rule hold up? | **structure vindicated, fitting rule not: a closed-form fit of the same model beats the paper's by 3–25x and runs 30–1200x faster**; verified against exact dense algebra |
| 26 | `exp26_conditional_real.py` | Does experiment 23's win survive on real data and on a trained network? | **only with little training data: 83% of the gap closed at 25 training points (4.6 standard errors), nothing once the posterior narrows; never on a learned feature map** |
| 27 | `exp27_sample_structure.py` | The paper *samples* from its covariance to invent training data. Do samples carry the real spatial correlation? | **chain is the only cheap model that does (8 standard errors ahead at 65 536 dimensions); its one blind spot, vertical neighbours, is fixed by ordering along a space-filling curve** |
| 28 | `exp28_model_inversion.py` | The decisive downstream test: run the authors' *own* image-inversion loop unchanged and swap only the covariance fit. Does off-diagonal structure improve the invented images, and does our closed-form fit improve them further? | **in flight** — their `W_LCM = 0` ablation is itself a diagonal mean/variance match, so this isolates off-diagonal structure exactly |

Supporting diagnostics: `ctrl_ordering.py` (frozen-order control),
`diag_matfree_init.py` (matrix-free fit initialization),
`kron_lcm_checks.py` (the numbers quoted in the KFAC x LCM theory note).

---
---

## Experimental setup (how everything was actually run)

### Two regimes — and they differ in a way that matters

The campaign has **two distinct experimental protocols**, and which one an
experiment uses determines what its numbers mean.

| regime | experiments | posterior obtained by |
|---|---|---|
| **A — library** | 1, 2, 3, 3b, 4, 5, 6, 12 | `laplace-torch` `Laplace(...)` object: `.fit()` accumulates the GGN through the library's backend, hyper-parameters tuned by the library's `log_marginal_likelihood`, predictions through `la(X, pred_type="glm")` |
| **B — analytic GLM** | 7, 8/8b/8c, 9, 10/10b, 11, 13, 14, 15, 16 | **no library posterior.** The GGN is formed in closed form as `XᵀΛX` (Gaussian: `XᵀX/σ²`), the posterior covariance is `(GGN + λI)⁻¹` by direct inversion, and only the standalone `LCMCovariance` module is used — to fit `A` to that GGN |

Regime B exists because a Gaussian GLM has an **exact** Laplace posterior: the GGN
is the true Hessian, so there is no GGN-vs-Hessian approximation, no backend, and
the `full` reference is exact rather than merely dense. That makes it the clean
setting for the structural questions (ordering, complexity class, joint UQ). The
cost is that regime-B results say nothing directly about neural networks — which is
why Exp 14(c) re-runs the same protocol on a trained network's last-layer features,
where a last-layer Laplace *is* a GLM.

### Regime A — library experiments

**Software.** Python 3.13.5, PyTorch 2.10.0+cu128, `laplace-torch` fork branch
`lcm-covariance`, scikit-learn 1.6.1, statsmodels 0.14.4. Single NVIDIA RTX 4060
Laptop GPU (8 GB); everything also runs on CPU.

**MAP training** (`common.train_map`) — Adam throughout, full-batch loss
`MSELoss` (regression) or `CrossEntropyLoss` (classification), model in `.eval()`
afterwards:

| experiment | architecture | data | optimiser | epochs | batch |
|---|---|---|---|---|---|
| 1 | `Linear(d,32)–Tanh–Linear(32,32)–Tanh–Linear(32,k)` | synthetic AR(1) features, `ρ=0.7`; reg `n=500`, `d=8` (last-layer) / `d=6` (all-weights); cls `n=600`, `k=3` | Adam, lr `5e-3`, wd `1e-3` | 120 | 128 |
| 6 | same, `hidden = 24` (all-weights) or `32` (last-layer) | as Exp 1 | Adam, lr `5e-3`, wd `1e-3` | 120 | 128 |
| 2 | `Linear(d,50)–Tanh–Linear(50,1)` | 6 UCI sets (california, concrete, diabetes, energy, wine, yacht) via sklearn/OpenML; inputs **and** targets standardised on train; 8 random 90/10 splits | Adam, lr `1e-2`, wd `1e-4` | 120 | 128 |
| 3 | `Conv(1,16)–ReLU–MaxPool–Conv(16,32)–ReLU–MaxPool–Flatten–Linear(1568,84)–ReLU–Linear(84,10)` | MNIST (in) vs FashionMNIST (OOD), 4 000 test / 4 000 OOD, GGN over the first 8 000 train points | Adam, lr `1e-3`, wd `5e-4` | 5 | 256 |
| 3b, 5 | 3-block Conv+BatchNorm+ReLU CNN → `Flatten–Linear(2048,128)–ReLU–Linear(128,10)` | CIFAR-10 (in) vs SVHN (OOD), GGN over the first 10 000 train points | Adam, lr `1e-3`, wd `5e-4`, **cosine-annealed** | 15 | 256 |
| 4 / 4b | `Linear(d,40)–Tanh–Linear(40,30)–Tanh–Linear(30,k)` | synthetic AR(1); 4b instead fits a *controlled* scrambled-AR(1) target matrix and trains no network | Adam, lr `5e-3`, wd `1e-3` | — | — |
| 12b | `Linear(40,84)–Tanh–Linear(84,10)` | synthetic, `N = 3 000`, labels `argmax(XW + 0.5ε)`; MAP reaches 100% train accuracy | Adam, lr `1e-2`, **full-batch**, 400 steps | — | full |
| 12a / 12c | single `Linear(in,out)` (12a) or `Linear(d,1,bias=False)` (12c) | random `N=256` (12a) / `N ∈ {400,600,900}` for `d ∈ {100,200,400}` (12c) | **not trained** — 12a measures storage, which depends only on shapes; 12c measures the KFAC-vs-Full identity, which holds at any parameter value | — | 64 / 100 |

**Posterior construction** (`common.fit_laplace`). One code path for every
structure, so the comparison is like-for-like:

```python
la = Laplace(model, likelihood,
             subset_of_weights = "all" | "last_layer",
             hessian_structure = "diag" | "kron" | "full" | "lcm",
             lcm_steps=600, lcm_lr=3e-2)     # LCM only
la.fit(train_loader)
```

* **Curvature:** the library default, `CurvlinopsGGN` — the GGN/Fisher, not the
  exact Hessian, which is indefinite for a neural loss and so would not give a
  valid covariance (rationale in `../PROJECT_REPORT.md` §6.2).
* **Hyper-parameters:** prior precision — and, for regression, observation noise —
  are tuned **per method** by maximising that method's own Laplace marginal
  likelihood: Adam on `log θ`, lr `1e-1`, 100–200 steps (`tune_steps` 200 default;
  150 for Exp 2, 100 for Exp 3). Every baseline therefore gets its own best case.
  *Exception:* Exp 1 and Exp 6 skip tuning and pass `common_prior = 1.0`, fixing
  one prior precision across all structures, so the KL measures **only** the
  curvature approximation and not a hyper-parameter search.
* **Predictive:** linearised GLM predictive, `la(X, pred_type="glm")`; for
  classification the **probit** link approximation. Regression test-LL is reported
  in the original target scale (the `y`-standardisation Jacobian is undone).
* **Fidelity metrics** (Exp 1, 6): the dense posterior covariance is extracted
  uniformly for every structure as `functional_covariance` with identity Jacobians,
  then compared by `KL(N(θ*,Σ_approx) ‖ N(θ*,Σ_full))` and relative Frobenius.

### Regime B — analytic GLM experiments

**Data generator** (`exp10_ordering.simulate`, shared by Exp 8–11, 13–16). Features
are an explicit AR(1) chain across the `d` dimensions,
`x_j = ρ x_{j-1} + √(1−ρ²) ε_j`, so the feature order *is* the latent 1-D
coordinate. Targets are `y = Xβ + σε` with a fixed smooth
`β = Σ_f sin(2πf·coord + φ_f)/√d` over `f ∈ {2,5,9}`. Defaults: `ρ = 0.9`,
`σ = 0.5`, `n = 2 000`–`3 000`, `n_test = 300`–`400`, prior precision `λ = 1.0`.

**Posterior.** `GGN = XᵀX/σ²`; `Σ = (GGN + λI)⁻¹`; posterior mean
`Σ Xᵀy/σ²`. For Exp 7's large-`d` rows the exact `full` reference uses the `n×n`
dual/Woodbury form, so it stays exact at every `d`. Everything is float64.

**Two regime-B experiments train a network** (to obtain a realistic curvature
target, then treat its last layer as a GLM):

* **Exp 13, `nn_ggn` target.** `Linear(20,H)–Tanh–Linear(H,C)` with `C` the first of
  `(5,4,6,8,10,2)` dividing `d` and `H = d/C`, so the GGN is exactly `d×d` and
  comparable to the other three targets. `N = 600` synthetic samples, Adam lr `1e-2`,
  300 steps, cross-entropy. The exact last-layer GGN is then assembled as
  `mean_i Λ_i ⊗ φ_i φ_iᵀ` with `Λ_i = diag(p_i) − p_i p_iᵀ` (vectorised; verified
  against the explicit Kronecker sum to 5.5e-17).
* **Exp 14(c).** `Linear(20,60)–Tanh–Linear(60,1)` regression on
  `y = 2·sin(Xw) + 0.3ε`, `N = 2 500` train / 300 test, Adam lr `1e-2`, 500 steps,
  MSE. Observation noise is set to the **empirical residual sd** of the trained
  network, not tuned. The last-layer features `Φ = tanh(W₁x)` then play the role of
  `X`, so `GGN = ΦᵀΦ/σ²` and a last-layer Laplace is exactly the regime-B GLM.

**Per-experiment regime-B settings.**

| exp | `d` | `n` train | `n` test | seeds | notes |
|---|---|---|---|---|---|
| 7 | 500–20 000 | 3 000 | 2 000 | 3 (`d ≤ 5000`), else 1 | matrix-free `fit_frobenius`; `full` via `n×n` dual |
| 8c | 500 / 1000 / 1500 | 3 000 | 400 | 10 | `a` initialised at the true order |
| 9 | 500 / 1000 / 1500 | 40 labelled + 800 pool | 1 000 | 3 | greedy batch acquisition, `B = 25` |
| 10 | 800 | 3 000 | 400 | 3 | natural vs permuted conditions |
| 10b | 400 | 2 000 | 300 | 3 | order degraded by position noise, `a` frozen |
| 11 | 100 (Tecator) / 64 (CO₂) | 144 / 1 332 | 40% holdout | 5 | real data, 60/40 random splits |
| 13 | 200 (fit) / 1000 (joint) | — / 3 000 | — / 400 | 1 / 5 | same-class competitors |
| 14 | 300 / 500 / 60 | 1 200 / 2 500 / 2 500 | — / 300 / 300 | 3 / 5 / 3 | (a) rank sweep, (b) decisive, (c) NN last layer |
| 15 | 400 | 2 500 | 300 | 4 | permuted order; hybrid and annealing |
| 16 | 800 (A) / 400 (B) | 3 000 | 400 | 3 | reproduces Exp 10's settings exactly |

**Real data (Exp 11).** *Tecator* — OpenML `tecator` v1, 100 wavelength-ordered
absorbance channels predicting fat content, `n = 240`, 5 random 60/40 splits
(`n_train = 144`). *CO₂* — statsmodels Mauna Loa weekly series, interpolated,
turned into a `d = 64` lag-regression (lag index = the 1-D order), `n_train = 1 332`.
Features are standardised on train; observation noise is set to `0.5·sd(y_train)`
rather than tuned.

**Metrics.** Marginal predictive LL uses per-point variance
`xᵀΣx + σ²`; **joint** predictive LL uses the full test covariance
`X_test Σ X_testᵀ + σ²I` via Cholesky, normalised by the number of test points.
"Gap closed" is defined in the section above.

**Permutation convention** (Exp 10, 15, 16). `X[:, perm]` puts original column
`perm[j]` at position `j`, and original column `i` carries latent coordinate `i`,
so the true coordinate of permuted column `j` is **`perm[j]`**, not
`argsort(perm)`. This was once a live bug (see Implementation notes §4) and is now
pinned empirically: freezing `a = perm` fits to 0.13 rel-Frobenius while
`a = argsort(perm)` gives 0.92, indistinguishable from random (0.92).

### LCM fit settings

`LCMCovariance(dim, eps=1e-6, init_a_scale=1.0, init_w=0.1, dtype=float64)`.
`d = softplus(u)+eps`; `a` initialised on `linspace(−1,1,dim)·init_a_scale` unless
overridden. Fitting minimises `‖A − T‖_F²/‖T‖_F²` with Adam and `auto_init=True`
(rescales `w`/`d` from the target diagonal).

| experiment | steps | lr | `a` initialisation | target normalised? |
|---|---|---|---|---|
| 1, 2, 3, 3b, 6 (via library) | 600 | 3e-2 | library default | ✘ |
| 5 | 500 | 3e-2 | library default | ✘ |
| 4 | 800 (50–1600 sweep) | 3e-2 | default / frozen | ✘ |
| 7 | 800 | 5e-2 | `ρ`-matched spacing | ✘ (matrix-free `fit_frobenius`) |
| 8c | 700 | 5e-2 | `arange(d)·(−log ρ)` = true order | ✘ |
| 9 | 700 | 5e-2 | default | ✘ |
| 10, 10b, 11 | 700–800 | 5e-2 | uninformed / seriation / natural | ✘ |
| **13a** | 800 | 5e-2 | true order on `glm_ar1`, else library default | **✔** |
| **13b** | 700 | 5e-2 | `arange(d)·(−log ρ)` = true order | **✔** |
| **14a / 14b** | 600 | 5e-2 | `ρ`-matched true order | **✔** |
| **14c** | 600 | 5e-2 | library default (no meaningful order) | **✔** |
| **15 (D2)** | 600 | 5e-2 | uninformed (order is hidden) | **✔** |
| **15 (D4)** | 1 020 total | 5e-2 | uninformed / seriation / oracle, per arm | **✔** |
| **16a / 16b** | 800 / 700 | 5e-2 | per arm (see the Exp 16 table) | **✔ and ✘ side by side** |

**The normalisation column is the single most important row-to-row difference in
this table.** Fitting `T/s` and multiplying the result by `s` is exact (the family
is positively homogeneous), but the unnormalised fit misbehaves at GGN scale
(~1e5). Exp 13(c) and Exp 16 quantify it: the same configuration moves from
54% → 87% on the headline, and from −94.5% → +10.2% on the ordering result. **Every
experiment with ✘ in that column is conservative by an unknown margin.**

*Root cause, and the fix.* The fit objective is an **unnormalised** sum of squares
while Adam takes steps of absolute size `lr`. On a GGN-scale target `auto_init`
sets `w ≈ 300`, so one step of `lr = 5e-2` moves it by ~2e-4 relative and the fit
barely leaves its initialisation. `LCMCovariance.fit_dense` and `fit_frobenius` now
**normalise internally** (`normalize=True` by default) and rescale the parameters
afterwards via positive homogeneity — so the library path and the regime-B path now
compute the same thing. Pass `normalize=False` to reproduce pre-fix behaviour.
`tests/test_lcm.py::test_fit_is_scale_equivariant_and_normalisation_helps` pins it.
**Consequence:** the ✘ rows above describe how those experiments *were run*; re-run
today they would use the corrected fit.

### Baseline implementations in regime B

`diag` is `diag(1/(diag(GGN)+λ))` — the exact diagonal Laplace, not a fitted
approximation. `full` is the exact inverse (or the `n×n` dual form at large `d`).

**Competitor families (Exp 13–15)** are all fitted by the *same* routine as LCM —
Adam on `‖M − T‖_F²/‖T‖_F²`, same step count, same lr, same scale-normalisation —
and all are initialised from the target diagonal, so no family is advantaged by its
initialisation:

* `chol_band(b)`: `M = L Lᵀ`, `L` lower-banded with `b` subdiagonals, diagonal via
  `softplus`; `d + b·d − b(b+1)/2` parameters (`b=1` → **tridiagonal** precision).
* `diag_rank(k)`: `M = softplus(u) + V Vᵀ`, `V ∈ R^{d×k}`; `(k+1)·d` parameters,
  `V` initialised at `√(0.1·mean(diag T)/k)·N(0,1)`.
* `hybrid_k` (Exp 15): `M = D + W K(a) W + V Vᵀ`, `(3+k)·d` parameters, with the LCM
  part initialised at a 0.6/0.3 `w`/nugget split of the target diagonal.

**Annealing schedules (Exp 15 D4)**, all given the same 1 020-step total budget as
the `plain` baseline:

| variant | schedule |
|---|---|
| `anneal-hard` | nugget fraction `φ ∈ (0.9, 0.7, 0.5, 0.3, 0.15, 0.05)`, 120 steps each with `w` and `u` **frozen** (only `a` trained), then 300 steps with everything unfrozen |
| `anneal-soft` | `φ ∈ (0.9, 0.6, 0.3, 0.1)`, 170 steps each, schedule used only to **re-initialise** the split — all parameters trainable throughout |
| `restarts` | 3 random restarts × 340 steps, best kept (the equal-compute control) |
| `seriation` | Fiedler-vector seriation of the `|correlation|` graph Laplacian as `a`-init, then a normal fit |

### Reported statistics

Means over seeds with **95% confidence intervals** (`1.96·sd/√n_seeds`). Seed
counts vary and are stated per experiment — 10 (Exp 8c), 8 splits (Exp 2), 5
(Exp 11, 13b, 14b), 3–4 elsewhere. Exp 3b/5 use 2. Small-seed results should not
be over-read; see Limitations.

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
integrates out all cross-point correlation, so on in-distribution queries it is
nearly insensitive to the
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
| 500 | −0.823 / −0.824 / −0.823 | −0.830 / −0.821 / −0.820 | **86.7% ± 24** |
| 1000 | −0.922 / −0.923 / −0.919 | −0.946 / −0.914 / −0.907 | **84.3% ± 11** |
| 1500 | −1.076 / −1.079 / −1.057 | −1.116 / −1.064 / −1.033 | **64.2% ± 10** |

**Outcome.** The reframing works, and the contrast within each row is the point:
on **marginal** LL LCM now *ties* diagonal (−0.923 vs −0.922 at d=1000), while on
**joint** LL over the same posteriors it is clearly and significantly better. This
is the project's main positive result, and it turned a flat negative into a finding.

**These numbers were revised upward.** This experiment now runs through
`laplace-torch` (via `glm_laplace.py`) with the normalisation fix in
`LCMCovariance`, so it describes the shipped `LCMLaplace` class. With `diag` and
`full` unchanged to 1e-14 — the LCM path being the only variable — the result moved:

| d | hand-rolled, unnormalised fit | **library + normalisation** |
|---|---|---|
| 500 | 61.8% ± 49 | **86.7% ± 24** |
| 1000 | 52.1% ± 16 | **84.3% ± 11** |
| 1500 | 23.2% ± 12 | **64.2% ± 10** |

Confidence intervals roughly halve, the result is significant at every `d`, and
**the `d`-dependence flattens** (87/84/64 against 62/52/23). That last point
matters: the apparent decay with `d` was substantially *fit* degradation rather
than *method* degradation, which had been one of the main caveats limiting this
claim. On the marginal metric LCM also stops losing (−0.9325 → −0.9229).

*Still to note:* an earlier 3-seed run with a near-zero nugget reported 70–81%; the
10-seed library number above is the trustworthy one.

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

> ### ⚠ SUPERSEDED — the headline negatives here were a fitting artifact
>
> **Exp 16 shows the two central claims of this section do not survive a corrected
> fit.** Exp 10/10b used the unnormalised Frobenius fit; scale-normalising it (exact,
> not a tuning change — see Exp 13c) moves the *same* configurations from
> "catastrophically worse than diagonal" to "useful":
>
> | claim below | as published | corrected (Exp 16) |
> |---|---|---|
> | learned coordinate, unknown order | **−84%** | **+10.2% ± 6.9** |
> | spectral seriation, unknown order | **−79%** | **+65.4% ± 19.7** |
>
> The order *recovery* numbers (0.028 learned, ~0.79 seriation) are **unchanged** —
> only the fit quality was broken. So the finding "gradient descent cannot discover a
> latent order" **stands**; the finding "LCM is therefore unusable without a known
> order" **does not**. Read this section for the ordering-recovery results and the
> mechanism; take the joint-LL verdicts from Exp 16.

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
itself does. So the reported 62%/52%/23% headline numbers were **conservative** —
they understated LCM for a fitting reason, not a modelling one.

**Both follow-ups this section recommended have since been done.**
`LCMCovariance.fit_dense`/`fit_frobenius` now normalise internally, and Exp 8c has
been re-run through `laplace-torch`: the headline is **86.7/84.3/64.2%**, up from
62/52/23%, with roughly half the CI width. See Exp 8c above.

**Conclusion.** LCM survives the comparison against its own complexity class, but
the verdict is split and worth stating precisely. *In its niche* it is not merely
better than its peers, it is better by a margin no amount of low-rank or banding
reaches at comparable cost — the `3d ≈ 17d` result and the +87%/−65% joint-LL
split are the clearest evidence in the project that the structure is doing real
work. *Outside* its niche, on neural-network weight curvature, low-rank dominates
it per parameter and carries no ordering fragility, which is a further argument —
independent of Exp 4a and the audit — against pursuing weight-space Laplace.

---

## Experiment 14 — Low-rank in depth, and behaviour at very large `d`

**Why.** Exp 13 left the gating question half-answered: LCM beats low-rank at
matched budget in its niche, but low-rank wins on NN curvature given more
parameters. If low-rank simply dominates, there is no reason to scale LCM up.

### (a) Where does low-rank catch up? It depends on correlation

Relative Frobenius to the AR(1) GLM GGN (`d=300`, 3 seeds):

| `ρ` | **`lcm` 3d** | `rank2` 3d | `rank8` 9d | `rank16` 17d | `rank32` 33d | `band1` 2d |
|---|---|---|---|---|---|---|
| 0.5 | **0.358** | 0.667 | 0.605 | 0.532 | 0.410 | 0.465 |
| 0.9 | **0.148** | 0.813 | 0.475 | 0.242 | **0.091** | 0.866 |
| 0.99 | 0.023 | 0.241 | 0.030 | **0.010** | **0.005** | 0.984 |

**At matched `3d` budget LCM wins at every correlation level.** Low-rank catches
up only with far more parameters, and *the more correlated the data, the faster it
catches up* — at `ρ=0.99` rank-16 already beats LCM, because a near-perfectly
correlated matrix is genuinely low-rank. Conversely at `ρ=0.5` even **rank-32
(33d)** fails to match LCM's `3d`. LCM's edge is largest at *moderate*
correlation, which is also the regime where the structure is most informative.

### (b) The decisive test — matched *fit* does **not** give matched *prediction*

`d=500`, 5 seeds, order known. Both columns come from the same fits:

| model | rel-Frobenius (precision) | joint gap closed |
|---|---|---|
| **`lcm`** 3d | **0.137** | **+93.4% ± 19.6** |
| `rank2` 3d | 0.864 | −29.8% ± 48.0 |
| `rank8` 9d | 0.638 | −19.1% ± 34.0 |
| `rank16` 17d | 0.413 | −32.0% ± 104.5 |
| `rank32` 33d | 0.191 | +29.3% ± 41.0 |
| `band1` 2d | 0.866 | −155.7% ± 149.9 |

**`rank32` gets within striking distance of LCM on the precision fit (0.191 vs
0.137) and still closes only 29% of the gap against LCM's 93%** — with a CI four
times wider. There is a concrete reason. We fit the *precision*; the posterior
covariance is `(A + λI)⁻¹`. For this problem the GGN ≈ `n·K(a)`, and `K⁻¹` is
**tridiagonal**, so the true posterior covariance is near-banded. A low-rank
precision inverts (Woodbury) to *diagonal-plus-low-rank*, which cannot be banded —
so it misses the structure that matters for prediction no matter how well it
matches the precision in Frobenius norm.

This is the strongest evidence in the project that LCM captures the **right**
structure rather than merely more of it, and it also shows **Frobenius fit is a
poor proxy for predictive quality** — a caveat that applies to Exp 1, 4 and 13.

### (c) On NN curvature, low-rank wins — predictively too

A last-layer Laplace *is* a GLM in the last-layer features, so the same protocol
applies to a trained network (`d=60` features, 3 seeds):

| model | rel-Frobenius | joint gap closed |
|---|---|---|
| `lcm` 3d | 0.690 | 6.4% ± 2.4 |
| `rank2` 3d | 0.648 | 3.7% ± 2.0 |
| `rank8` 9d | **0.347** | **12.7% ± 2.5** |
| `band1` 2d | 0.783 | 2.0% ± 3.2 |

At matched `3d` LCM still edges it (6.4% vs 3.7%), but `rank8` doubles LCM's
result and LCM has no comparable knob. Note also that **every** method closes only
2–13% here — NN weight curvature is a poor habitat for all of them, which is the
same conclusion Exp 4a and the audit reached by other routes.

### (d) Very large `d` — is there a regime only LCM can reach?

Measured, matrix-free paths only (float64):

| `d` | LCM storage | LCM matvec / log-det | `rank8` storage / matvec | dense (or wide-layer KFAC) |
|---|---|---|---|---|
| 1 000 | 0.00002 GB | 0.87 / 2.24 ms | 0.0001 GB / 0.19 ms | 0.01 GB |
| 20 000 | 0.0004 GB | 3.17 / 8.76 ms | 0.0013 GB / 0.11 ms | 2.98 GB |
| 100 000 | 0.0022 GB | 10.3 / 21.3 ms | 0.0067 GB / 0.14 ms | **74.5 GB** |

**Answer to "are very large matrices an interesting case for LCM?" — yes for
feasibility, no for exclusivity.** At `d = 10⁵` LCM needs 2 MB and ~10 ms per
matvec while a dense posterior needs 74.5 GB and a wide-layer KFAC (`in² `floats)
needs the same. But **low-rank is equally feasible there and ~70× faster per
matvec**, so LCM has no uncontested regime on *cost* alone. Its case at scale
rests entirely on (b): in the ordered-feature niche low-rank is cheap but
predictively poor. The applications that fit are wide *ordered* problems —
spectroscopy at full resolution, genomics along the genome, long-lag time series —
not wide neural-network layers.

---

## Experiment 15 — Two proposed fixes for the ordering problem

**Why.** Both target what was then believed to be the central negative: that with
an unknown order LCM collapses to −84%. (Exp 16 shows that premise was itself an
artifact — but the two interventions are still worth reporting, and the annealing
result is a clean negative.)

### (a) Hybrid `D + W K(a) W + V Vᵀ` — adding an order-independent term

`d=400`, unknown (permuted) order, 4 seeds, budgets matched at `5d`:

| model | params | rel-Frobenius | joint gap closed | order recovery |
|---|---|---|---|---|
| `lcm` | 3d | 0.750 | 28.4% ± 34.4 | 0.170 |
| **`hybrid_k2`** | 5d | **0.606** | **38.4% ± 34.4** | 0.114 |
| `rank4` | 5d | 0.748 | 12.6% ± 19.7 | — |

The hybrid fits better and predicts somewhat better than either component alone,
and beats the equal-budget low-rank model by 3×. But the confidence intervals
overlap heavily and order recovery does **not** improve (0.114 vs 0.170), matching
the pre-registered prediction that a low-rank term cannot make the discrete search
easier. **Verdict: promising but not established** — it needs more seeds, and
Exp 16 changes the baseline it was meant to rescue.

### (b) Nugget annealing — **the proposal does not work**

Start with a large diagonal and anneal down (graduated non-convexity). Two
implementations, because the first constrains the model and that is a confound:
*hard* freezes `w`/`d` at each stage and trains only `a`; *soft* uses the schedule
only to re-initialise the split and keeps everything trainable. Equal compute
throughout. `d=400`:

| variant | order recovery | joint gap closed |
|---|---|---|
| `plain` (baseline) | 0.100 | −21.1% |
| `anneal-hard` | 0.035 | −9.6% |
| `anneal-soft` | 0.039 | −4.1% |
| **`seriation`** | **0.624** | **+42.0%** |
| `anneal-soft+seriation` | 0.622 | +40.0% |

**Annealing makes order recovery *worse*, not better** (0.035–0.039 vs 0.100), in
both implementations — so the failure is the idea, not the confound. The small
improvement in joint gap comes with a worse fit and is best read as incidental
regularisation. Layering annealing on top of seriation adds nothing (+40.0 vs
+42.0).

*Why it likely fails:* graduated non-convexity smooths a landscape whose optimum
*moves continuously* as the smoothing is withdrawn. Here the object being searched
is a **permutation**; a large nugget does not make the correct permutation easier
to find, it makes the objective **less sensitive to `a` at all**, so the coarse
stages supply almost no ordering signal to warm-start the fine ones.

**What does work is seriation** — a one-shot spectral computation, no schedule.
That is the practical recommendation, and Exp 16 strengthens it considerably.

---

## Experiment 16 — The ordering catastrophe was a fitting artifact

**Why.** Exp 13(c) found that scale-normalising the Frobenius target (fit to
`T/s`, multiply by `s` — exact, since the family is positively homogeneous) raised
the headline joint-UQ number from 54% to 87%. Exp 10/10b — the project's central
*negative* results — used that same unnormalised fit. This tests how far the
deficiency reaches.

**It reaches all the way.** Exp 10's protocol at its exact settings (`d=800`,
`n=3000`, permuted, seeds 0–2), changing one variable at a time:

| variant | joint gap closed | order recovery |
|---|---|---|
| **A** exp10 as published (uninformed init, unnormalised) | **−94.5% ± 48.8** | 0.028 |
| **B** the same, **scale-normalised** | **+10.2% ± 6.9** | 0.029 |
| **C** seriation **frozen**, unnormalised (as published) | −65.5% ± 41.8 | 0.793 |
| **D** seriation as **init, then refined**, normalised | **+65.4% ± 19.7** | 0.792 |
| **E** seriation **frozen**, normalised | +11.2% ± 7.9 | 0.793 |
| **F** oracle (true order), refined, normalised | +89.6% ± 20.8 | 1.000 |

**A reproduces Exp 10** (−94.5% vs the published −84%; recovery 0.028 either way),
so the original run was sound as executed. But:

1. **A → B: normalisation alone turns −94.5% into +10.2%, with the recovered order
   unchanged (0.028 → 0.029).** The "learned coordinate overfits into a
   confidently wrong chain" conclusion is therefore **false** — the chain was never
   the problem; the fit was. With a corrected fit, an unknown order degrades
   *gracefully*, which is exactly the safe-by-default behaviour Exp 15(a) was
   designed to engineer and which turns out to be there already.
2. **C → D: seriation goes from −65.5% to +65.4%**, via two independent changes,
   each of which matters (E isolates them: frozen-but-normalised gives only
   +11.2%). So seriation needs to be used as an **initialisation that is then
   refined**, not frozen — which Exp 10 never tried.
3. **D vs F: with an unknown order, seriation+refinement reaches +65.4% against the
   oracle's +89.6%** — roughly three-quarters of what knowing the true order buys.

### The fragility threshold: real, but shifted, and measured with a bad ruler

Exp 10b's curve re-run with and without normalisation (`d=400`, order frozen):

| position noise | \|Spearman\| | gap (unnormalised) | gap (normalised) |
|---|---|---|---|
| 0.0 | 1.0000 | 21.7% | **58.2%** |
| 0.3 | 1.0000 | 21.2% | **59.0%** |
| 1.0 | 1.0000 | −3.6% | **26.3%** |
| 3.0 | 0.9997 | −30.2% | −3.4% |
| 10.0 | 0.9965 | −27.7% | −52.2% |

Fragility is **real** — accuracy still collapses the result — but the whole curve
shifts up by ~35 points, so the break-even point moves. And note rows 1–3: all
three read `|Spearman| = 1.0000` while the gap moves by 33 points. **\|Spearman\|
saturates and is a poor diagnostic near 1**, so the specific "≥ 0.999 required,
0.998 fails" claim should not be relied on; it is an artifact of a coarse ruler as
much as of a sharp threshold.

**Conclusion — this is the most consequential correction in the project.** The
claim that LCM requires a *known* order, which scoped the entire method and drove
the publication assessment, does not survive. With a corrected fit and a spectral
initialisation, LCM closes **65%** of the diagonal→full gap on data whose ordering
is entirely hidden. What survives from Exp 10 is the narrower and still-true
finding that **gradient descent alone cannot discover a latent order** (recovery
0.028, unchanged) — you need seriation to supply it. The practical recipe is:
*seriate, then refine, and normalise the target.*

---

## Experiment 17 — Where off-diagonal covariance actually shows up

**The project's "marginal metrics are blind to off-diagonal covariance" framing was
imprecise, and the imprecision hid a better setting.** The marginal predictive
variance is `xᵀΣx`, which plainly depends on off-diagonals. What actually made our
marginal metrics tie is a property of the **test distribution**, not of the metric.

**The mechanism.** For in-distribution inputs `x ~ N(0, Σ_f)` and a posterior
`Σ ≈ (n Σ_f/σ² )⁻¹`, the mean predictive variance is pinned by a trace identity,
independently of the covariance *structure*:

```latex
\mathbb{E}\big[x^\top \Sigma x\big] = \operatorname{tr}(\Sigma\,\Sigma_f) = \frac{\sigma^2 d}{n}
\qquad\Longrightarrow\qquad
\mathbb{E}[\text{predictive variance}] = \sigma^2\Big(1 + \frac{d}{n}\Big)
```

Any approximation that gets the trace about right matches on average. Measured at
`d=300`, `n=1500`, `σ=0.5`: the three posteriors give mean predictive variance
0.3003 (diagonal), 0.2948 (LCM) and 0.3124 (full) against the identity's 0.3000.

**Setup.** One AR(1) GLM, `d=300`, `n=1500`, `ρ=0.9`, prior fixed at 1, 5 seeds,
all posteriors from `laplace-torch`. Four probes on the *same* three posteriors:
in-distribution inputs; **shifted** inputs with identical unit marginal variances
but no correlation; the sum of 20 independent predictions; and the sum of 20
predictions at *clustered* (mutually correlated) query points.

| Probe | diag | LCM | full | gap closed by LCM |
|---|---|---|---|---|
| (a) marginal LL, in-distribution | −0.8473 | −0.8472 | −0.8459 | **0.6%** |
| (b) marginal LL, **shifted inputs** | −1.7806 | −1.3802 | −1.3541 | **93.9%** |
| (c) LL of a sum, independent queries | −2.3163 | −2.3173 | −2.3172 | 108.6% *(on 0.001 nats — noise)* |
| (d) LL of a sum, correlated queries | −3.1855 | −3.1759 | −3.1464 | 24.5% |

Calibration, `E[(y−μ)²/var]`, ideal 1:

| Probe | diag | LCM | full |
|---|---|---|---|
| in-distribution | 1.061 | 1.078 | 1.017 |
| **shifted inputs** | **2.926** | **1.359** | 1.045 |
| sum, independent queries | 1.004 | 1.022 | 0.964 |
| sum, correlated queries | 1.320 | 1.381 | 1.051 |

**Outcome — and the control that reinterprets it.** Under covariate shift a plain
per-point marginal metric closes **93.9%** of the gap, and diagonal Laplace is **three
times overconfident** (z² = 2.93) where full Laplace is calibrated (1.05). But giving
each method **one oracle scalar scale** on its epistemic variance collapses the
difference entirely:

| rescaled probe | diag | LCM | full | best scale (d/L/f) |
|---|---|---|---|---|
| (a) in-distribution | −0.8460 | −0.8452 | −0.8452 | 1.3 / 1.6 / 1.1 |
| (b) **shifted** | **−1.3525** | **−1.3521** | **−1.3515** | **12.2 / 1.6 / 1.1** |

All three land within **0.001 nats**. So this probe measures whether a method's overall
variance *scale* is right, **not whether its off-diagonal structure is right** — a
diagonal posterior plus one temperature is as good as the full posterior under shift.
What the structured posteriors genuinely provide is that they arrive near the correct
scale by themselves (1.1–1.6) where diagonal needs **12.2x**. That is useful — under real
shift you have no labels with which to fit a temperature — but it is an
automatic-calibration property, not an off-diagonal one. The same control gives the same
verdict in Exp 18 (classification) and Exp 20 (real spectra), so the reading is consistent
across all three settings. **This retracts the earlier claim that covariate shift is "the
right setting" for off-diagonal covariance.**

**The aggregates were a null.** Summing independent predictions changes nothing
(0.001 nats): independent queries have no cross-covariance to get wrong. Summing
*correlated* queries does expose a gap, but only 24.5% of it, and LCM is no
better calibrated than diagonal there (1.38 vs 1.32) — because clustering points
around an in-distribution base keeps the query inside the trace-pinned regime.

**Conclusion — one mechanism explains all four probes.** What matters is whether the
query direction lies in the **small-eigenvalue subspace** of the data covariance.
In-distribution queries concentrate on large-eigenvalue directions, where every
approximation agrees. Shifted queries put weight on the small-eigenvalue directions,
where the approximations do differ. But per the control above, what that difference
amounts to is the overall variance **scale**, which one scalar repairs: **no probe tried
in this project separates off-diagonal structure from a correctly-scaled diagonal.**
Extrapolation is where the methods visibly diverge; it is not evidence that the
off-diagonals are what matter. That remains the central open problem.

---

## Experiment 18 — Kron-LCM: the first empirical test of KFAC x LCM

KFAC x LCM had never been run. What existed was the algebra (the one-factor eigen
reduction, verified to machine precision) and a fit-only measurement of how chain-like
real KFAC factors are. No posterior had ever been built from it. It now lives in the
library, so it is reached the same way as every baseline:

```python
Laplace(model, "classification", hessian_structure="kron_lcm", lcm_order="seriate")
```

### What was added to the library (not to the experiment)

| Addition | Where | What it is |
|---|---|---|
| `kalman_solve` | `laplace/lcm.py` | `A^-1 y` in `O(dim)`. The filter gives `A^-1 = L^T S^-1 L`, so a solve is one *backward* affine scan reusing the existing associative scan — no sequential sweep |
| `LCMCovariance.fit_moments` | `laplace/lcm.py` | the closed-form three-band estimator; no optimiser, one pass, never forms `dim x dim` |
| `seriation_order` | `laplace/lcm.py` | Fiedler-vector ordering, so the chain axis can be recovered with no side information |
| `KronLCMDecomposed` | `laplace/utils/matrix.py` | the reduction as a precision object with `KronDecomposed`'s interface |
| `KronLCMLaplace`, `KronLCMLLLaplace` | `baselaplace.py`, `lllaplace.py` | `hessian_structure="kron_lcm"`, all-weights and last-layer |

Correctness (15 new tests in `Laplace/tests/test_lcm.py`, 33 total; the library's own 491
tests still pass):

* `kalman_solve` vs dense solves: **<= 2.6e-15** relative, for `dim` 1...500, any coordinate order.
* `fit_moments` on an exact LCM target: **3.0e-15** on the interior. The two endpoint
  amplitudes are genuinely unidentified by three bands (they carry the whole 3e-2
  full-matrix error); order-equivariant to 1e-12.
* **Dense fallback (`lcm_layers=[]`) reproduces `KronLaplace` bit-for-bit** — log-det,
  marginal likelihood, `functional_variance`, `functional_covariance` and the predictive
  all `0.00e+00`. This pins the plumbing independently of the LCM approximation.
* Structured path vs the dense form of the same precision: log-det exact, functional
  variance/covariance **<= 1e-15**, sampling (perturb-and-solve, since an LCM has no
  closed-form square root) within Monte-Carlo error.

Two pre-existing bugs surfaced and were fixed: `KronLLLaplace(damping=True)` silently
lost the setting (`KronLaplace.__init__` runs later in the MRO and reset it, although the
docstring documents initialising as supported), and `backend.jacobians` returns **all
zeros** for a last-layer Laplace, where `last_layer_jacobians` is the right call. The
latter was present in `common.glm_joint`; it is harmless there because only the
`subset_of_weights='all'` GLM experiments call it, but it made the first validation
attempt vacuous until caught. Both now have regression tests.

### Result — softmax on AR(1)-ordered features, i=256, C=5, N=2000, rho=0.9, 3 seeds

Each structure has its prior tuned by the library's own
`optimize_prior_precision(method="marglik")`, so none of them loses on a scalar.

**Only one of the three probes separates diagonal from full at all**, which is Exp 17's
finding reappearing:

| Probe | diag -> full gap | status |
|---|---|---|
| in-distribution marginal | **-0.0055 +- 0.0045** | degenerate — diagonal scores *better* than full |
| **covariate-shifted marginal** | **+1.1223 +- 0.1257** | separates (8.9 sem) |
| joint, correlated queries | **+0.0035 +- 0.0097** | degenerate — gap is 0.4 sem |

So only the shifted column below carries information; "gap closed" on the other two is a
ratio of noise and the script now suppresses it.

| structure | stored | NLL shifted | gap closed | ECE shifted | marglik |
|---|---|---|---|---|---|
| full | 1 651 225 | 1.4599 | 100.0% | 0.095 | -653 |
| kron (KFAC) | 65 586 | 1.5517 | 91.8% | 0.144 | -765 |
| kron, damped | 65 586 | 1.5394 | 92.9% | 0.146 | -765 |
| **kron_lcm** | **818** | **1.5482** | **92.1%** | 0.148 | -772 |
| kron_lcm, seriated | 818 | 1.8333 | 66.7% | 0.211 | -959 |
| diag | 1 285 | 2.5822 | 0.0% | 0.310 | -1325 |

**Outcome.** On the one probe with any power, Kron-LCM closes **92.1%** of the
diagonal-to-full gap against KFAC's 91.8% — a tie — while storing **818 numbers against
KFAC's 65 586 (80x less) and full's 1.65M (2019x less)**. That is the first evidence the
construction works as a posterior and not only as algebra. Given the true feature order it
matches KFAC at 1/80th the memory; asked to *recover* the order by seriation it falls to
66.7%, so ordering is again where the method pays, consistent with Exp 10/16.

**The same scale control as Exp 20 applies here too.** Under one oracle scalar scale on
the predictive covariance, every structure converges:

| structure | NLL shifted | oracle-scaled | best scale |
|---|---|---|---|
| full | 1.4599 | 1.4154 | 1.98 |
| kron | 1.5517 | 1.4176 | 3.51 |
| kron_lcm | 1.5482 | 1.4181 | 3.51 |
| kron_lcm, seriated | 1.8333 | 1.4216 | 6.25 |
| **diag** | 2.5822 | **1.4107** | **12.22** |

Tempered diagonal is the **best** of all six (1.4107), and the spread across structures is
~0.01 nats against an untempered gap of 1.12. So the 92.1% describes how well each
method's marginal-likelihood-tuned scale happens to suit the shifted set, not the quality
of its covariance shape. Kron-LCM's standing relative to KFAC is unchanged — they tie
before and after — but neither beats a correctly-scaled diagonal.

**Correction to the preliminary (1-seed) reading.** A first single-seed run appeared to
show Kron-LCM *ahead* of KFAC on the joint probe (45.2% vs 2.7%). With 3 seeds the
diag-to-full gap on that probe is 0.0035 nats, so those figures were noise over noise and
are withdrawn. Clustering query points around an in-distribution base keeps them inside
the trace-pinned regime — exactly what Exp 17(d) measured, where the same probe recovered
only 24.5% in a setting where it did have power.

---
---

## Experiment 19 — where Kron-LCM can be built and KFAC cannot

Exp 18 established parity-minus-2.2-points on 80x less storage at `i=256`. The storage
ratio grows as `i/3`, so the question that decides whether any of this matters is the
feasibility frontier. Measured on this machine (31 GiB RAM, 7.6 GiB VRAM), AR(1)
activations, `N=1000`, `rho=0.9`, `C=10` outputs:

| i | `fit_moments` | log-det | solve | LCM storage | dense factor (KFAC) | exact full | feasible here |
|---|---|---|---|---|---|---|---|
| 512 | 0.002 s | 0.001 s | 0.003 s | 0.01 MiB | 0.002 GiB | 0.2 GiB | all three |
| 2 048 | 0.003 s | 0.001 s | 0.007 s | 0.05 MiB | 0.03 GiB | 3.1 GiB | all three |
| 8 192 | 0.022 s | 0.002 s | 0.014 s | 0.19 MiB | 0.50 GiB | **50 GiB** | kron_lcm + KFAC |
| 32 768 | 0.093 s | 0.006 s | 0.054 s | 0.75 MiB | **8.0 GiB** | 800 GiB | **kron_lcm only** |
| 65 536 | **0.188 s** | 0.008 s | 0.195 s | **1.50 MiB** | **32 GiB** | 3 200 GiB | **kron_lcm only** |

**Outcome.** At `i=65 536` the whole posterior factor is fitted in **0.188 s** and stored
in **1.50 MiB**, and its log-determinant (0.008 s) and solves (0.195 s, all finite) run
at the same width. KFAC's dense factor there would need **32 GiB — more than this machine
has** — and the exact full posterior 3 200 GiB. The crossover on this hardware is between
`i=8 192` (KFAC still comfortable at 0.5 GiB) and `i=32 768` (8 GiB: past VRAM and a
quarter of RAM). So the regime where Kron-LCM is the *only* option starts at roughly
30 000 inputs, which is reached by wide MLP heads, long-sequence attention projections and
high-resolution 1-D feature maps.

Quality does not degrade with width: the fitted chain recovers `phi = 0.9031` against the
true `rho = 0.9` at every width from 512 to 65 536 (worst single edge off by 2.9e-2),
checked in `O(i)` without ever forming the matrix.

**The honest caveat.** `Laplace(..., hessian_structure="kron_lcm")` routes through the
library's `Kron` accumulator, which materialises the `i x i` factor *before* the LCM is
fitted to it — so that entry point is bounded by `i^2` exactly like KFAC, and only the
fitted posterior's storage shrinks. The numbers above come from
`LCMCovariance.fit_moments(X)` applied directly to activations, which is `O(N i)` time and
`O(i)` memory and needs no matrix. The library therefore has the primitive but not yet a
streaming Kron accumulator that uses it; adding one is what would make the frontier above
reachable through the normal `Laplace(...)` call. That is the single most valuable
remaining piece of implementation.

---
---

## Experiment 20 — Kron-LCM on real ordered data, with more than one output

Tecator near-infrared spectroscopy: 240 meat samples, 100 wavelength-ordered absorbance
channels, **three** targets (moisture, fat, protein), so `C=3` and the Kronecker
factorisation is not vacuous — this fixes the flaw in the earlier flagship plan, where a
single output would have made KFAC *equal* the full layer Hessian. With an isotropic
Gaussian likelihood the output factor is constant across examples, so **KFAC is exact
here**, which isolates the only question that matters for Kron-LCM: what does replacing
the 100x100 input factor with a 300-number chain cost when everything else is exact?
Internal check passes: `kron` and `full` agree to **1.9e-05** nats.

Shift probe is a real one — train on the low/mid-fat samples, test on the 50 highest-fat
ones (extrapolation along a physically meaningful direction). 5 seeds, prior tuned per
method by `optimize_prior_precision(method="marglik")`.

| structure | stored | marglik | NLL in-dist | NLL shifted | z² in-dist | z² shifted |
|---|---|---|---|---|---|---|
| full | 91 809 | **−163.2** | 0.6192 | 23.7052 | 1.193 | 16.291 |
| kron (= exact here) | 10 018 | −163.3 | 0.6192 | 23.7059 | 1.193 | 16.292 |
| **kron_lcm** | **318** | −530.3 | 1.8500 | **4.5134** | 0.196 | **1.111** |
| kron_lcm, seriated | 318 | −530.3 | 1.8500 | 4.5134 | 0.196 | 1.111 |
| diag | 303 | −1536.2 | **0.5959** | 21.9363 | 0.853 | 14.833 |

Two things jump out, and the **"gap closed vs full" metric is meaningless here**: full is
not the ceiling, it is the *worst* method under shift (NLL 23.7, z² = 16.3 — errors four
times larger than the predicted standard deviation). Kron-LCM looks spectacular: 5x
better NLL and near-perfect calibration (z² = 1.111 against the ideal 1.0).

### The control that overturns the obvious reading

Kron-LCM's variances are simply *bigger*. Giving every method one oracle scalar variance
scale, chosen on the shifted set itself (so it upper-bounds what inflation alone can buy):

| structure | NLL shifted | oracle-scaled | best scale |
|---|---|---|---|
| full | 23.7052 | **4.4780** | 46.0 |
| kron | 23.7059 | **4.4779** | 46.0 |
| kron_lcm | 4.5134 | 4.4980 | **1.1** |
| diag | 21.9363 | 6.1534 | 101.1 |

**So Kron-LCM's entire advantage here is calibration slack, not covariance shape.** Once
full is allowed one temperature it *ties and slightly beats* Kron-LCM (4.478 vs 4.498).
What the chain actually does is land at the right overall scale by itself — it needs
temperature **1.1**, where full and KFAC need **46x** inflation and diagonal **101x**.
That is a real and useful property (no post-hoc temperature set on held-out shifted data,
which you would not have), but it is an automatic-calibration claim, not a
better-structure claim, and it must not be reported as the latter. Consistent with this,
Kron-LCM's marginal likelihood is far *worse* (−530 vs −163): it fits the curvature
considerably less well and still predicts better under shift — the fit-versus-prediction
divergence again, this time on real data and in the method's favour.

The same control has since been applied to Exp 18, with the same verdict: every
structure converges to ~1.41 nats once tempered, and tempered diagonal is best.

### Ordering: seriation is exact on real spectra

Natural and seriated orderings give *bit-identical* numbers above, because spectral
seriation recovers the wavelength order **exactly, reversed** — and reversal leaves
`|t_p − t_q|` unchanged, so the chain is invariant to it. Measuring when this works:

| factor | neighbour \|corr\| | seriation Spearman | recovered? |
|---|---|---|---|
| Tecator NIR, n=140 | 1.0000 | **1.0000** | **exactly, reversed** |
| AR(1) ρ=0.99, N=2000 | 0.9899 | 1.0000 | monotone |
| AR(1) ρ=0.9, N=2000 (Exp 18) | 0.9000 | 0.9060 | mostly, with local errors |
| AR(1) ρ=0.9, N=200 | 0.8949 | **0.4827** | **no** |

Recovery is governed by how smooth the chain is and how well the factor is estimated. On
real smooth spectra it is free; at ρ=0.9 it has local errors that cost Exp 18 twenty-two
points of gap closed (90.3% → 68.0%); with `N` small relative to `i` it collapses.

---
---

## Experiment 21 — is a *learned* feature map chain-structured? (the deep-learning test)

Exps 18 and 20 both put the chain on raw, physically ordered inputs. That left untouched
the question that decides whether any of this concerns deep learning, and which the
KFAC x LCM note's main recommendation rests on: **when a network learns a 1-D feature map,
is the covariance of those learned features still a smooth chain?**

A small 1-D CNN (two conv layers, 8 channels, max-pool) is trained on Tecator spectra and
last-layer Laplace is taken over `Linear(368, 3)`, so the Kronecker input factor is the
covariance of the flattened penultimate feature map. Three outputs keep the factorisation
non-vacuous; KFAC is again exact (internal check: `dNLL = +2.3e-04`). 5 seeds.

### The answer is no

| diagnostic | value | reading |
|---|---|---|
| neighbour \|corr\| along the flattened index | 0.7152 | — |
| mean \|corr\| over *all* pairs | **0.2529** | a chain needs neighbour >> mean; the ratio is only 2.8x |
| seriation agreement with that index | **0.0375** | the flattened order has no special status; seriation finds an unrelated one |
| **effective rank of the factor** | **1.75 of 368** | the factor is essentially **rank 2** |
| LCM fit error to the true factor | 0.5575 natural / 0.9817 seriated | a poor fit either way |

Stable across every seed: effective rank 1.32–2.06, seriation Spearman 0.015–0.082,
neighbour correlation 0.62–0.74. **A learned 1-D feature map's covariance is nearly
rank-2, not a chain** — the same verdict the hidden-layer factor measurements reached
earlier (effective rank 1.3–1.6, low-rank beating the chain 12–29x), now reproduced on the
one construction the theory note actually recommended.

### The predictive numbers, with the control

| structure | marglik | NLL in-dist | NLL shifted | z² shifted | tempered | best scale |
|---|---|---|---|---|---|---|
| full | **−410.7** | **2.1512** | 7.6376 | 4.21 | 4.9450 | 11.7 |
| kron | −415.5 | 2.1514 | 7.6470 | 4.22 | 4.9405 | 11.7 |
| kron_lcm | −475.8 | 2.7046 | **5.5004** | **1.90** | 4.8235 | 2.5 |
| kron_lcm, seriated | −1283.2 | 2.4576 | 6.9392 | 3.28 | 5.3339 | 6.4 |
| diag | −1595.5 | 2.4346 | 7.2980 | 3.56 | 5.5189 | 8.8 |

Paired against full, across seeds:

* in-distribution: kron_lcm is **+0.553 ± 0.098 worse (5.6 sem)** — significantly worse.
* shifted, untempered: **−2.137 ± 0.599 better (3.6 sem)** — the same calibration effect as
  Exps 18/20 (it needs temperature 2.5 where full needs 11.7).
* shifted, **tempered**: **−0.122 ± 0.068 (1.8 sem)** — **not significant.**

So the chain's advantage here is again variance scale, and once that is controlled for
nothing survives; in-distribution it is clearly worse, and its marginal likelihood is far
worse (−476 vs −411), consistent with the 0.56 fit error.

**Outcome — this closes the theory note's main recommendation, negatively.** Putting the
chain on a *learned* feature map does not work, because learned feature maps are not
chains: they are nearly rank-2. The place for a chain remains a *physically* ordered axis
(wavelengths, lags), where Exp 20 showed seriation recovers the order exactly. For learned
representations the measurements have now said the same thing three times — low-rank is
the right structure.

---
---

## Experiment 22 — reproducing the paper's own claim, to test our implementation

Every negative in this project concerns a **transplant**: LCM was taken out of REMIX
(model inversion for data-free continual learning) and used as the structured *precision*
of a Laplace posterior. Before trusting those negatives, the implementation has to clear
the paper's own bar. arXiv:2605.11804 Figure 4 claims LCM attains higher **feature
log-likelihood** than a diagonal Gaussian at every ResNet-34 layer, at `D = C x H x W` up
to ~2e5. Reproduced here in miniature: a CNN on local CIFAR-10 (4 classes), flattened
feature maps, LCM fitted with **the paper's own Frobenius objective** (`fit_frobenius`),
scored by held-out Gaussian NLL against a diagonal fit.

| layer | D = CxHxW | n train | NLL LCM | NLL diagonal | gain per dim | LCM wins? |
|---|---|---|---|---|---|---|
| layer1 | 16384 | 1280 | **13220.0** | 14603.0 | 0.0844 | **yes** |
| layer2 | 8192 | 1280 | **6502.9** | 6570.1 | 0.0082 | **yes** |
| layer3 | 4096 | 1280 | **3336.0** | 3883.7 | 0.1337 | **yes** |

**The paper's claim reproduces: LCM beats the diagonal baseline at every layer.** So the
port is sound *in the setting the paper uses it in*, and this project's negative results
are not attributable to a broken implementation. They are about the transplant.

### What the transplant changed

| | Paper (REMIX) | This project |
|---|---|---|
| What LCM models | the **covariance** of features, Sigma | the posterior **precision** (GGN/Hessian) |
| Fit objective | Frobenius on the empirical covariance (eqs 5-7) | `fit_dense` (`baselaplace.py`) |
| What `a` indexes | the flattened feature map, `D` up to 2e5 | weight-space dimensions |
| Judged by | feature log-likelihood; continual-learning accuracy | predictive LL, calibration |

The first row is the consequential one. The paper puts **Sigma** in the LCM family, so its
inverse is tridiagonal — which is exactly what makes its likelihood cheap. This project
puts the **precision** in the family, so the resulting covariance is *not* LCM. Combined
with the fit-versus-prediction mechanism (Frobenius error on a precision is dominated by
its large eigenvalues, while predictions depend on its small ones), that is a sufficient
explanation for why the transplant underperforms while the original works.

### A claim the paper never made

The project had been testing whether LCM can **recover a known ordering**. The paper
claims no such thing. Its coordinates are described as inducing *"an ordering over
channels [that] allows kernel interactions to be computed via directional accumulations"*
— an enabler for the `O(C log C)` prefix-sum algorithms, not a recovery result. In its
setting the coordinates index conv channels, which have no canonical order to recover.
Exp 10/10b/16's ordering negatives therefore stand as findings about **this project's
extension**, not as refutations of the paper.

**What could not be run here.** The paper's `D ~ 2e5` at full dataset size is out of reach:
the Frobenius fit costs `O(N D)` per step, and a first attempt at `D=16384` with `N=2880`
and 400 steps was killed after 880 s on one layer. The table above uses `N=1280` and 150
steps, which still spans the high-`D` regime the claim concerns.

---
---

## Experiment 23 — the probe a temperature cannot fake (**the central open problem, solved**)

`exp23_conditional_prediction.py` · `results/exp23_conditional.csv`

Exps 17–21 left this project with one central open problem: **no probe separated
off-diagonal structure from a correctly scaled diagonal.** Every metric tried was
dominated by overall variance scale, and given one oracle scalar each, a diagonal matched
or beat the exact posterior (Exp 18: diag 1.4107 vs full 1.4154). This experiment closes
that problem, and the answer to "what would the right setting be?" turns out to be
**conditional (transductive) prediction**.

### Why marginal metrics were always going to fail

A marginal predictive likelihood averages per-point variances, and Exp 17's trace identity
pins their mean at `sigma^2 (1 + d/n)` *regardless of covariance structure*. Conditioning
does not average. Reveal the labels of some test points and predict the rest:

```
mu_{Q|O} = mu_Q + t Sigma_QO (t Sigma_OO + sigma^2 I)^-1 (y_O - mu_O)
```

is driven entirely by the **off-diagonal block** `Sigma_QO`, and as `t -> infinity` it
becomes `mu_Q + Sigma_QO Sigma_OO^-1 (y_O - mu_O)`, which is **exactly invariant to
rescaling the posterior covariance**. A temperature cannot move it by construction, so any
separation on that estimator is attributable to off-diagonal structure and nothing else.

### Result — Bayesian linear model, d=128, n_train=60 (< d), sigma=0.3, 30 seeds

20 test labels revealed, 20 queried. The generative prior precision and noise are handed
to every arm, so `full` **is** the exact posterior — the gold standard, not a competitor.
All numbers are RMSE on the queried block under the scale-invariant limit estimator.

| `rho` | marginal (no conditioning) | full (exact) | LCM | diag | LCM vs diag (paired) | gap closed |
|---|---|---|---|---|---|---|
| 0.0 (control) | 8.1176 | 6.9586 | 7.5490 | 7.5449 | −0.0041 ± 0.0097 (**−0.4σ**) | — |
| 0.5 | 7.1287 | 5.7920 | 6.5821 | 6.6067 | +0.0245 ± 0.0501 (0.5σ) | 3% |
| 0.9 | 3.5223 | 2.6271 | 3.2141 | 3.7412 | +0.5270 ± 0.0879 (**6.0σ**) | **47%** |
| 0.99 | 1.1835 | 0.9325 | 1.0655 | 1.3870 | +0.3215 ± 0.0484 (**6.6σ**) | **71%** |

**The separation is real, scale-free and regime-dependent.** At `rho = 0` the chain and the
diagonal are indistinguishable (−0.4σ) — the negative control passes, confirming the probe
measures structure rather than something else. From `rho = 0.9` the chain captures about
half the diagonal→exact gap, and 71% at `rho = 0.99`.

The qualitative version is sharper still. Conditioning on revealed labels **improves** the
exact posterior by 14–25%, and LCM inherits 7–10% of that, but with a diagonal posterior
conditioning actively **hurts**:

| `rho` | full | LCM | diag |
|---|---|---|---|
| 0.0 | +14.3% | +7.0% | +7.1% |
| 0.5 | +18.8% | +7.7% | +7.3% |
| 0.9 | +25.4% | +8.7% | **−6.2%** |
| 0.99 | +21.2% | +10.0% | **−17.2%** |

A diagonal posterior asked to transfer information between correlated test points transfers
it in the wrong direction. This is the first predictive result in the project that a scalar
temperature cannot reproduce — note the untempered conditional NLLs (26.9 for LCM, 32.6 for
diag at `rho=0.9`, against 2.23 for full) are still catastrophic and still mostly fixed by
the oracle scale (2.51 / 2.63 / 2.21), exactly as Exps 17–21 found. **Scale and structure
are separate defects, and only conditional prediction sees the second one.**

---
---

## Experiment 24 — off-diagonals as a *decision*: batch active learning

`exp24_batch_acquisition.py` · `results/exp24_acquisition.csv`

Exp 23 separated structure from scale on a *metric*. This asks the operational question:
does posterior structure change **which points you choose to label**, and does that change
downstream error? The score is test RMSE, so no calibration slack can rescue a bad choice,
and the arms include the oracle-tempered diagonal that has beaten structure everywhere
else in this project.

The criterion is the predictive (V-optimal) one, because that is what matches the metric:
labelling `x` cuts the predictive variance at a reference point `z` by
`Sigma_zx^2/(Sigma_xx + sigma^2)`, so the greedy score sums that over a reference sample and
each pick applies the exact rank-1 Bayesian downdate. **A first design used noise-free
D-optimality** — which is *exactly* scale-invariant — **and it lost to random selection**,
because maximising parameter-space volume buys atypical high-norm inputs rather than
predictive accuracy. That is recorded rather than hidden: the criterion has to match the
metric, and the scale-invariance then has to be tested rather than assumed.

### Result — d=64, start 8 points, 5 rounds x 4 acquisitions, pool of 300 (5 copies per direction), 20 seeds

Test RMSE at `n=28`, paired against selecting with the exact posterior (positive = worse):

| selector | `rho`=0 (control) | `rho`=0.9 | `rho`=0.99 |
|---|---|---|---|
| full (exact) | — | — | — |
| **LCM** | +0.3263 ± 0.0973 (3.4σ) | **+0.1843 ± 0.0805 (2.3σ, within noise)** | +0.0907 ± 0.0254 (3.6σ) |
| diag | +0.5339 ± 0.0827 (6.5σ) | +1.0730 ± 0.1280 (8.4σ) | +0.4017 ± 0.0706 (5.7σ) |
| **diag, oracle temperature** | +0.2946 ± 0.0828 (3.6σ) | **+0.6591 ± 0.1015 (6.5σ)** | +0.1686 ± 0.0375 (4.5σ) |
| variance only (no off-diagonals) | +1.1085 ± 0.1134 (9.8σ) | +1.0008 ± 0.1183 (8.5σ) | +0.3243 ± 0.0514 (6.3σ) |
| random (floor) | +0.1322 ± 0.1146 (1.2σ) | +0.3558 ± 0.0765 (4.7σ) | +0.1315 ± 0.0369 (3.6σ) |

**The oracle temperature does not rescue the diagonal here.** It closes under 40% of its
deficit and remains **6.5σ** worse than exact at `rho=0.9`, where LCM is statistically
indistinguishable from the exact posterior. Head to head:

| | `rho`=0 | `rho`=0.9 | `rho`=0.99 |
|---|---|---|---|
| tempered diag − LCM | −0.0317 ± 0.1064 (−0.3σ) | **+0.4748 ± 0.0883 (5.4σ)** | +0.0779 ± 0.0304 (2.6σ) |

The control passes again: with `rho = 0` the chain and the tempered diagonal tie, and
random selection matches the exact posterior (1.2σ) because there is little structure to
exploit.

Two further readings, both striking:

- **Selecting with a diagonal posterior is worse than selecting at random**, at every
  correlation level (random − diag: −0.40, −0.72, −0.27; −3.6σ, −5.6σ, −4.0σ). A diagonal
  does not merely lose information, it actively misdirects the acquisition.
- The **mechanism is visible**: mean |cosine| between acquired inputs at `rho=0.9` is
  0.274 (full), 0.279 (LCM), 0.397 (diag), 0.415 (tempered diag), 0.374 (variance only).
  Structure is what stops a batch from repeating itself; a temperature cannot supply it,
  since scaling every variance leaves redundancy untouched.

Note that `variance_only` is the only arm that ignores off-diagonals altogether — a
diagonal *weight* posterior still induces a dense *function-space* covariance, so "diag"
here means **wrong** correlations rather than none. Both failure modes are measurable, and
they are not the same: ignoring them costs +1.00, getting them wrong costs +1.07.

---
---

## What could not be run on this machine

Recorded so the gaps are not mistaken for negative results. The machine is an RTX 4060
Laptop (7.6 GiB) with 31 GiB RAM.

| Wanted | Blocker |
|---|---|
| Library `lowrank` baseline (`LowRankLaplace`) | **Upstream-broken, not a missing install** (diagnosed 2026-10-19). The distribution `asdfghjkl==0.1a4` that `pyproject.toml` pins installs a module named **`asdl`** (`top_level.txt`), and *is* already present in this anaconda base — so the long-standing "not installed" note was wrong. But `LowRankLaplace` needs `AsdfghjklHessian.eig_lowrank` (`laplace/curvature/asdfghjkl.py:212`), whose module-level import requires `COV`, `fisher_for_cross_entropy`, `hessian_for_loss` and `hessian_eigenvalues` — none of which 0.1a4's renamed API provides, while `asdfghjkl` itself is no longer resolvable on PyPI. The modern `AsdlHessian` imports cleanly but has no `eig_lowrank`. So upstream pins a version incompatible with its own import, on any machine. Routes: an archived pre-rename release, or implement `eig_lowrank` ourselves (top-k eigenpairs of the GGN via Lanczos/`torch.lobpcg`, ~40 lines on the existing curvature interface). Exp 13/14 compared low-rank *factors* with their own code, so the gap is in the library-based comparison only |
| Exact `full` Laplace beyond ~i=2048 | measured in Exp 19: 3.1 GiB at i=2048 with C=10, **50 GiB at i=8192**. So above a few thousand inputs Kron-LCM cannot be scored against exact Full at all — only against KFAC |
| Dense KFAC factors at i >= 32 768 | measured: `i^2` float64 is 0.5 GiB at i=8192, **8 GiB at i=32 768**, **32 GiB at i=65 536** (more than this machine's RAM). Kron-LCM runs there in 0.188 s / 1.5 MiB, so above ~30k inputs there is no baseline to compare against on this hardware |
| Real wide architectures (ResNet/ViT last layers, transformer MLPs) | a 2048→1000 head is fine for KFAC but `Full` is out of reach on any machine; large-scale KFAC-vs-Kron-LCM comparisons need more than 7.6 GiB of VRAM |
| Many-seed studies at scale | the headline results rest on 3–5 seeds; widening CIs at i >= 2000 is wall-clock bound rather than impossible |

Since Exp 23 the GMUM SLURM cluster is the escalation path for these rows; `cluster/`
holds the submission scripts. See the gap register below for what is queued there.

---
---

## Experiment 26 — does Exp 23's separation survive on real features?

`exp26_conditional_real.py` · `results/exp26_*.csv`

Exp 23 was a synthetic AR(1) GLM, and it only separated above `rho ~ 0.9`. Two settings in
this project already have opposite structural diagnostics, so the predictions were fixed
before the run: Tecator spectra (neighbour |corr| 1.0000, seriation exact — chain *should*
work) and Exp 21's learned CNN map (effective rank 1.75 of 368, seriation agreement 0.0375
— chain should *not*).

### First: the scale-free limit estimator does not survive contact with real data

Exp 23's estimator conditions with the noise dropped, which is what makes it exactly
scale-invariant. On real features it is unusable, and not because of the chain:

| setting | effective rank | full: no conditioning → scale-free limit |
|---|---|---|
| Tecator spectra | **1.03** | 0.3366 → 0.8553 |
| CNN last-layer map | **1.32** | 0.4579 → 1.0380 |

With an effectively rank-one function-space posterior, `Sigma_OO` is singular and the
noise-free conditional is an ill-posed extrapolation that hurts **every** family, the exact
posterior worst of all (it has the most off-diagonal structure to extrapolate along). This
was checked against the alternative reading — that the failure was a numerical choice of
mine — by replacing the jittered solve with a pseudo-inverse, which is the well-defined
limit for a rank-deficient block and still exactly scale-free. Same outcome.

So on real data each arm gets its own **oracle scale** instead, the control used throughout
Exps 17–21. That is conservative and still structure-controlled: every family is given its
best possible temperature, so what remains is shape.

### Result — Tecator NIR spectra, C=3 targets, 10 seeds

| train size | no conditioning | full | kron | Kron-LCM | diag | conditioning gain (exact) | diag − LCM | gap closed |
|---|---|---|---|---|---|---|---|---|
| **n=25** (< d=100) | 0.4299 | **0.2998** | 0.3005 | 0.3079 | 0.3469 | **+30.3%** | +0.0391 ± 0.0084 (**4.6σ**) | **83%** |
| n=60 | 0.3431 | 0.2870 | 0.2873 | 0.3007 | 0.3134 | +16.4% | +0.0127 ± 0.0076 (1.7σ) | 48% |
| n=140 (all, 3 seeds) | 0.3366 | 0.3364 | 0.3364 | 0.3336 | 0.3374 | +0.1% | — | — |

**Exp 23's separation does transfer to real ordered data — but only while the posterior is
broad.** At `n=25` conditioning is worth 30% of the error and the chain captures 83% of
what the exact posterior gets, significantly. By the full training set the epistemic
covariance has shrunk below the noise, conditioning is worth 0.1% to anybody, and no family
differs from another. That is a scope condition, not a defect: it is the small-data regime
that motivates Bayesian treatment in the first place.

On the CNN arm the chain never separates from the diagonal (1.8σ at full data, 2.0σ at
n=25, and −18% "closed" at n=60 — i.e. slightly worse than the diagonal). Exp 21's
diagnostics predicted this, and they were right.

---
---

## Experiment 25 — the paper's claim against a baseline that can fight back

`exp25_paper_scale_baselines.py`, `exp25_verify_moments.py` · `results/exp25_local.csv`,
`results/exp25_verify.csv`

Exp 22 reproduced Figure 4 but inherited the paper's **only** baseline, a diagonal. A
diagonal losing shows that *some* correlation helps, not that a **chain** is the right
correlation. Five further families, on held-out feature NLL (CIFAR-10, N=1200, 3 seeds,
this laptop; the cluster run extends `D` to the paper's ~2e5):

| D | diag (1D) | **lcm_moments (3D)** | lcm_frob (3D, paper's objective) | lcm_ml (3D) | fixed_a (2D) | fa_r2 (3D, matched) | fa_r3 (4D) |
|---|---|---|---|---|---|---|---|
| 4 096 | 3623.8 ± 44 | **729.8 ± 282** | 2403.7 ± 50 | 2445.3 ± 47 | 2415.4 ± 59 | 2993.1 ± 62 | 2842.5 ± 64 |
| 16 373 | 11995.4 ± 342 | **1636.2 ± 722** | 9064.8 ± 190 | — | 8624.8 ± 274 | 9672.9 ± 358 | 9170.7 ± 294 |
| 65 533 | 59374.4 ± 937 | **−1666.1 ± 2730** | 44166.8 ± 713 | — | 42334.7 ± 658 | 46953.2 ± 839 | 44250.7 ± 718 |

Four readings, in order of importance.

1. **The structure choice is vindicated.** A chain beats factor analysis at **matched
   parameter count** (3D) at every width, and beats rank 3 (4D) too. The paper's family is
   not merely better than a diagonal; it is better than the natural low-rank competitor on
   the same budget. (Exp 14 had found the opposite on *network curvature* in this project's
   precision transplant — so that negative was about the transplant, like the others.)
2. **The paper's fitting rule is leaving a factor of 3–25× on the table.** `lcm_moments` —
   our closed-form three-band estimator — beats the Frobenius fit of the *same family* by
   3.3× at D=4096, 5.5× at 16 373, and by 45 833 nats at 65 533. It is also **30–1200×
   faster**: 0.3 s / 1.9 s / 13.4 s against 10.4 s / 59.3 s / 378.5 s. `||A - S||_F^2`
   weights large eigenvalues and is nearly blind to the small ones that dominate a log-det,
   which is exactly what a likelihood cares about.
3. **Learned coordinates still buy nothing.** Freezing them at equal spacing (2D) matches
   or beats the full 3D Frobenius fit at every width (8624.8 vs 9064.8; 42334.7 vs
   44166.8), consistent with Exps 10/16.
4. **Fitting the family by exact likelihood does not help either** (2445.3 vs 2403.7) and
   costs 8× more (78.1 s vs 10.4 s), so the Frobenius arm was not merely being scored on a
   foreign objective — gradient fitting of *any* objective is the weak part, not the loss.

### The verification, because these numbers are extreme

A negative NLL and a log-det of −202 146 are exactly the kind of result that turned out to
be a bug twice in this project, so `exp25_verify_moments.py` checks them independently:

| D | min innovation variance | log-det, two O(D) implementations | solve residual | **quad / dim** | vs dense Cholesky |
|---|---|---|---|---|---|
| 4 096 | 1.848e-03 | agree to 0.00e+00 | 7.67e-12 | 1.3266 | log-det **3.11e-10**, quad **3.10e-07** |
| 16 384 | 5.044e-04 | agree to 0.00e+00 | 6.65e-14 | 1.0519 | log-det **1.46e-10**, quad **1.42e-10** |
| 65 529 | 4.040e-05 | agree to 0.00e+00 | 1.25e-13 | 1.0541 | dense impossible (32 GiB) |

Positive definite at every width, two independent log-det implementations agreeing exactly,
and — the decisive number — **quad/dim ≈ 1.05**, meaning the fitted covariance explains
*held-out* data at the right scale. The very negative log-dets are legitimately tight
fits, not degeneracy.

One methodological note: the first large-`D` output of this experiment was wrong in the
diagonal's favour by 35.8 nats/dim, because each family clamped its diagonal at a different
floor and a ReLU map has near-dead units (7 of 65 536 below 1e-6 × median variance, one
exactly zero). Degenerate dimensions are now dropped and every family gets one shared,
reported floor. Before the fix, factor analysis appeared to win.

### At the paper's own scale, on the cluster — confirmed, and the advantage grows

The laptop table above runs to 65 536 dimensions on CIFAR-10. This repeats it on CIFAR-100
with 3000 images (2400 for fitting, 600 held out) at widths that reach **196 608
dimensions**, which is the order the paper works at. Three seeds, 5 h 2 m on one CPU node
(`results/exp25_paper_scale.csv`).

Held-out Gaussian cost of the centred features, in **nats per dimension, lower is better**,
mean ± standard error over 3 seeds. "Parameters per dimension" is the budget each family
spends, so families on the same number are directly comparable:

| How the covariance was fitted | par/dim | 16 384 dims | 65 536 dims | 196 608 dims |
|---|---|---|---|---|
| Diagonal only, the paper's own baseline | 1 | 0.8372 ± 0.0099 | 0.7566 ± 0.0074 | 1.0030 ± 0.0093 |
| Factor analysis, 2 factors (matched budget) | 3 | 0.6965 ± 0.0102 | 0.6568 ± 0.0125 | 0.8866 ± 0.0135 |
| Factor analysis, 3 factors (larger budget) | 4 | 0.6669 ± 0.0107 | 0.6155 ± 0.0163 | 0.8425 ± 0.0148 |
| Chain, fitted by exact maximum likelihood | 3 | 0.6811 ± 0.0085 | — | — |
| Chain, the authors' Frobenius rule | 3 | 0.6000 ± 0.0104 | 0.6165 ± 0.0247 | 0.8472 ± 0.0156 |
| Chain, Frobenius with the order frozen | 2 | 0.6349 ± 0.0043 | 0.5969 ± 0.0195 | 0.8396 ± 0.0191 |
| **Chain, our closed-form moment fit** | 3 | **0.1325 ± 0.0611** | **−0.0537 ± 0.0233** | **−0.2873 ± 0.0261** |

The exact-likelihood row stops after the first width because that fit is `O(D)` per step but
needs ~1150 s even at 16 384 dimensions, so the run skips it above a cutoff.

Four readings.

1. **The closed-form fit beats the authors' fitting rule by a margin that widens with
   dimension**, comparing the same seeds pairwise: **+0.4674 ± 0.0508** nats per dimension
   at 16 384 (9 standard errors), **+0.6702 ± 0.0095** at 65 536 (71), **+1.1345 ± 0.0229**
   at 196 608 (50). It is also **18–20× faster to fit** (98 s against 1723 s at the largest
   width). The laptop run found the same thing at smaller sizes, so this is now confirmed
   at the scale the method is sold on.
2. **Their fitted chain does not beat a plain factor analyser at scale.** At 65 536 and
   196 608 dimensions, their Frobenius-fitted chain (0.6165, 0.8472) is level with factor
   analysis using 3 factors (0.6155, 0.8425), which is given a *larger* budget. So the
   chain's advantage over low-rank alternatives is not a property of the family alone — it
   only appears once the family is fitted well. The structural claim survives; the way the
   paper obtains it does not carry it.
3. **Learning the coordinate ordering by gradient descent buys nothing, again.** Freezing
   the order at equal spacing uses one fewer parameter per dimension and is *better* at both
   large widths (0.5969 against 0.6165; 0.8396 against 0.8472). This is the fifth
   independent place in this project where the learned coordinate fails to earn its keep.
4. **The weak point is gradient fitting, not the choice of loss.** Exact maximum likelihood
   on the same family lands at 0.6811, close to the Frobenius rule's 0.6000 and nowhere near
   the closed form's 0.1325, despite optimising precisely the quantity being scored and
   costing 180× more time. Both gradient routes stall in the same region; the closed form
   starts in a much better one. That is a statement about the parameterisation's
   optimisation landscape, and it is the most likely mechanism behind readings 1–3.

The flattened feature maps are strongly chain-like at every width, which is the precondition
for any of this: neighbouring coordinates correlate 0.80 / 0.85 / 0.94 in absolute value
against 0.13 / 0.12 / 0.14 for random pairs, a ratio of 5.9× / 7.0× / 6.9×.

---
---

## Experiment 27 — the paper's actual use case: do the *samples* have the right structure?

`exp27_sample_structure.py` · `results/exp27_sample_structure.csv`

REMIX does not use its covariance to predict. It **samples** from it, to invert a network
and manufacture replay data. That makes sampling the claim worth testing, and it is a pure
structure probe: correlations are normalised, so **no temperature can give a diagonal the
right spatial structure**. In a flattened `C x H x W` map the lags mean something — lag 1 is
the next pixel in a row, lag 32 the pixel above, lag 1024 the next channel.

Mean correlation of 512 samples at each lag, against held-out features (3 seeds):

| family | lag 1 | lag 2 | lag 4 | lag 32 (row above) | lag 1024 (next channel) | profile error |
|---|---|---|---|---|---|---|
| **held-out truth** | 0.854 | 0.753 | 0.655 | 0.841 | 0.199 | — |
| diag (1D) | 0.000 | −0.000 | 0.000 | 0.000 | 0.000 | 0.6604 ± 0.031 |
| **lcm_moments (3D)** | **0.851** | **0.767** | **0.643** | 0.143 | 0.000 | **0.1871 ± 0.021** |
| lcm_moments, Hilbert order (3D) | 0.659 | 0.444 | 0.195 | **0.658** | −0.001 | 0.2694 ± 0.017 |
| lcm_frob (3D) | 0.444 | 0.437 | 0.432 | 0.433 | 0.108 | 0.2896 ± 0.016 |
| fa_r3 (4D) | 0.410 | 0.408 | 0.406 | 0.401 | 0.095 | 0.3164 ± 0.011 |
| fa_r2 (3D) | 0.347 | 0.345 | 0.345 | 0.340 | 0.085 | 0.3679 ± 0.005 |

(D = 16 382; layer 1 at D = 4096 is the same ordering, `lcm_moments` 0.1988 ± 0.019 best.)

### Confirmed on the cluster: CIFAR-100, 5 seeds, D up to 65 536

`results/exp27_cifar100.csv` (job 540753, trzmiel, 1 h 29 m). Profile error, lower is better:

| family | D = 16 384 | D = 65 536 |
|---|---|---|
| **lcm_moments (3D)** | **0.1773 ± 0.0034** | **0.1658 ± 0.0045** |
| lcm_moments, Hilbert order (3D) | 0.2475 ± 0.0022 | 0.2116 ± 0.0043 |
| lcm_frob (3D, paper's objective) | 0.3696 ± 0.0134 | 0.4613 ± 0.0150 |
| fa_r3 (4D) | 0.3772 ± 0.0066 | 0.4260 ± 0.0046 |
| fa_r2 (3D, matched) | 0.4203 ± 0.0086 | 0.4983 ± 0.0027 |
| lcm_frob_fixed_a (2D) | 0.4280 ± 0.0101 | 0.5288 ± 0.0079 |
| diag (1D) | 0.6244 ± 0.0100 | 0.7049 ± 0.0038 |

With five seeds the margins are no longer arguable: at D = 65 536 the moment-fitted chain
is **~8σ** ahead of the next-best arm (its own Hilbert variant) and **~50σ** ahead of
rank-3 factor analysis. It tracks the true decay 0.939 / 0.866 / 0.745 with
0.936 / 0.889 / 0.804, and the gap to every flat-profile family *widens* with D — the
Frobenius fit degrades from 0.3696 to 0.4613 as dimension grows, while the moment fit
improves. Cross-channel correlation is near zero in these features (truth 0.026 / 0.053,
against 0.199 on CIFAR-10), so here the chain's only real miss is still the vertical
neighbour: lag 32 is 0.230 against a true 0.922, which the Hilbert ordering lifts to 0.776.

**The chain is the only cheap family that reproduces *local* structure.** It tracks the
true decay 0.854 → 0.753 → 0.655 almost exactly. Low-rank cannot: factor analysis produces
a **flat** profile (0.347 at every lag), i.e. one global correlation, and the diagonal
produces white noise at any scale, as predicted.

**Why the Frobenius fit loses, mechanically.** `lcm_frob` is also flat (0.444 everywhere) —
it fits the same global component low-rank does. The fitted coordinate span says why:
**3.1 for Frobenius against 3630 for moments**. A narrow span makes `exp(-|a_p - a_q|)`
nearly constant for all pairs, so the chain degenerates into "everything equally
correlated". This is the same defect that costs it 3–25× in Exp 25, seen directly.

**The chain's one failure is dimensional, and fixable.** It gets lag 1 right and lag 32
(equally strong in truth, 0.841) wrong at 0.143, because a 1-D chain on a row-major map can
follow only one adjacency direction. Walking a **Hilbert curve** instead — a free
permutation — lifts lag 32 from 0.143 to **0.658** while costing lag 2 and lag 4
(0.767 → 0.444, 0.643 → 0.195). Average profile error is worse (0.2694 vs 0.1871), so it is
a trade rather than a win on this metric, but it is the only arm that reproduces both
adjacency directions at all, and which one matters depends on the downstream use. No family
reproduces the cross-channel correlation (truth 0.199).

---
---

## Experiment 28 — does the covariance change the images the method actually produces?

`exp28_model_inversion.py`, `exp28_check_chunk_exact.py` · `results/exp28_full_*.csv`

**Why this is the decisive test.** Every other experiment here asked the structured
covariance to *predict*. The method it comes from does not predict with it: it uses the
covariance's log-likelihood as a loss term while optimising images that a frozen network
believes in, so those images can stand in for a stored dataset. Experiment 27 (the sampling
probe above) showed the chain is the only cheap family whose samples carry real spatial
correlation. This asks whether that matters for the images.

It is also the one place where the authors' own ablation is exactly the control this project
spent weeks building by hand. With their covariance weight set to zero the loop **still**
matches every layer's feature mean and variance, so their own baseline is a *diagonal*
feature match, and the covariance term's entire contribution is the off-diagonal structure
on top of it.

Three arms share one loop, one set of weights and one schedule — the authors' own, 1000
steps at 112 pixels, 2000 at 224, 400 polishing steps, batch 50, all four ResNet-34 layers.
Only the covariance differs: their weight set to zero; their Frobenius fit; or the same
family fitted by our closed-form moment estimator and copied into *their* module so the loop
is untouched. Scored three ways, none circular: the inverted network's own confidence in the
target class (higher is better), the gap between the generated and real correlation-vs-
distance curves (Experiment 27's metric, lower is better), and Fréchet distance in a
ResNet-18 that was never inverted and so cannot be gamed by the loop (lower is better).

**Pre-registered falsifier.** If the three arms are indistinguishable on the held-out
network, the covariance term does not matter for the method's own application, and
Experiment 27's sampling result has no downstream consequence. That would be the most
important negative this project could report.

### Status: the baseline arm is in, with a noise calibration; the two covariance arms are running

The baseline arm ran **3 seeds** rather than 1, because it costs two minutes and the
approved single-seed design gave no way to separate a small difference between arms from
ordinary run-to-run variation. The metrics turn out to be very stable, which sets a clear
bar for the arms still running:

| Score | Mean over 3 seeds | Run-to-run s.d. | Range |
|---|---|---|---|
| Confidence of the inverted network (higher better) | 99.93% | 0.01 p.p. | 99.92–99.94% |
| Correlation-profile error (lower better) | 0.0250 | 0.0012 | 0.0237–0.0262 |
| Fréchet distance, held-out network (lower better) | 310.68 | 1.44 | 309.59–312.31 |

So a covariance arm needs to beat roughly 0.004 in profile error or 4.3 in Fréchet distance
to be distinguishable at three standard deviations, even with one seed.

Two findings are already firm, before the covariance arms finish.

1. **The authors' zero-covariance baseline already reproduces the real correlation
   structure almost exactly**, which leaves the covariance term very little room. Generated
   against real correlation at each distance: 0.716 vs 0.722 at distance 1, 0.421 vs 0.455
   at 2, 0.210 vs 0.269 at 4, 0.642 vs 0.640 at 7, 0.040 vs 0.063 at 49. This is a
   confound the paper's own ablation does not isolate: the features are not drawn from the
   covariance, they are produced by a convolutional network from optimised images, and
   convolution plus the matched per-layer means and variances supply most of the spatial
   correlation for free. Experiment 27's result was about *sampling* from the model; this
   pathway never samples from it.
2. **The covariance term costs about 350× the rest of the loop.** The whole 3400-step
   schedule takes 2.0 minutes without it and 12.4 seconds *per step* with it, and peak
   memory goes from 2.1 GB to 14.3 GB. Whatever the term buys has to be weighed against
   that, and against the measured fact that the structure it is meant to impose is largely
   there without it.

---
---

## Gap register

What is settled, what is in flight, and what was identified and deliberately not run.
Kept here so that "not done" is never mistaken for "does not work".

### Closed

| Gap | How | Outcome |
|---|---|---|
| **No probe separated off-diagonal structure from a correctly scaled diagonal** — the project's central open problem after Exps 17–21 | Exps 23 and 24 | **Closed.** Conditional prediction is exactly scale-invariant in the limit, and batch acquisition is a decision a temperature cannot redirect. Both separate LCM from a tempered diagonal at `rho >= 0.9` (6.0σ and 5.4σ), and both negative controls pass at `rho = 0` |
| The Fig-4 reproduction (Exp 22) inherited the paper's **only baseline, a diagonal** | Exp 25 (the fitting-rule comparison) adds factor analysis at matched 3-parameter budget, an equal-spacing chain, and the same chain fitted by exact likelihood | **Closed.** At the paper's own scale the chain beats factor analysis only once it is fitted well; their Frobenius-fitted chain is level with a *larger*-budget factor analyser at 65 536 and 196 608 dimensions |
| Does our closed-form moment fit beat the paper's Frobenius objective on the paper's own metric? | Exp 25's closed-form arm | **Closed, confirmed at the paper's scale, 3 seeds.** Paired advantage +0.4674 ± 0.0508 / +0.6702 ± 0.0095 / +1.1345 ± 0.0229 nats per dimension at 16 384 / 65 536 / 196 608 dimensions (9 / 71 / 50 standard errors), while fitting 18–20× faster. Verified against dense Cholesky: log-det agreed to 3.1e-10, quadratic form to 3.1e-7, all eigenvalues positive |
| Is the Frobenius objective the problem, or is it gradient fitting of this parameterisation? | Exp 25's exact-maximum-likelihood arm, which optimises the scored quantity directly | **Gradient fitting is the problem.** Exact likelihood reaches 0.6811 nats per dimension against the Frobenius rule's 0.6000 and the closed form's 0.1325, at 180× the fitting time. Both gradient routes stall in the same region |
| Is `fit_moments` inside `kernel_matvec`'s safe numerical domain? It produces coordinate spans of ~3630, and a span over ~709 broke that routine earlier in the campaign | direct check against a dense kernel at spans 2 / 20 / 200 / 3630 | **No issue**: max error 8.9e-16 at span 3630, nothing non-finite. The blocked-rescaling fix holds |
| Was the feature-NLL comparison fair across families? | found while reading Exp 25's first large-`D` output | **It was not, and is now.** A ReLU map has near-dead units (7 of 65536 below 1e-6 x median variance, one exactly zero), and each family clamped its diagonal at a different floor (LCM `eps` 1e-6 vs 1e-10 elsewhere), which handed the diagonal +35.8 nats/dim at `D=65536`. Degenerate dimensions are now dropped and every family gets one shared, reported floor |

### In flight

| Gap | State |
|---|---|
| Exp 25 (the fitting-rule comparison) at the paper's scale: `D` up to ~2e5 on CIFAR-100, 3 seeds | **Done** — job 541479 on trzmiel, 5 h 2 m, 10 CPUs, 48G; table and four readings above. Four submission faults were fixed to get there: a 16-CPU request against the `quick` QoS 10-CPU cap; `quota` missing on the compute node, which failed the setup job under `set -e` and so cancelled the `afterok` dependants; a 64G memory request that pended on Resources while 52 CPUs sat idle; and then 32G, which was enough for the laptop's 1200 images but not for the 3000 submitted, so job 541456 was OOM-killed partway through `D = 65536`. Now 48G, with the feature matrix preallocated instead of grown by `torch.cat` |
| Exp 28 (the image-inversion comparison) at the authors' full schedule | **Running**, approved after the budget was measured: jobs 541778 (our closed-form fit), 541779 (their Frobenius fit) and 541780 (their no-covariance ablation), one arm each so every job fits the one-day queue limit, 3400 steps, all four layers, their batch of 50, ~24 GPU-hours in total for one seed |
| Exp 25 locally at `D` up to 65536, 3 seeds | **Done**, 3 seeds — table above; verified against dense Cholesky |
| Exp 27 on the cluster: CIFAR-100, 5 seeds, `D` to 65536 | **Done** (job 540753, 1 h 29 m) — table above; the moment fit leads by ~8σ |
| ~~Library `lowrank` baseline~~ | **Closed, negatively** — see the table above. Not an install problem: upstream pins `asdfghjkl==0.1a4`, which renamed its module to `asdl` and dropped the four functions `AsdfghjklHessian` imports. Unblocking it means writing `eig_lowrank` ourselves |

### Promising, identified, not run

1. ~~**Model inversion — the paper's own application.**~~ **Now Exp 28, built on the
   authors' own loop** (cloned to `reference/REMIX`, imported rather than copied since
   their repository carries no LICENSE). Their `W_LCM = 0` ablation keeps the per-layer
   mean/variance match, so it *is* the diagonal control this project kept constructing by
   hand, and the covariance term's whole contribution is off-diagonal structure. Scored on
   the inverted network's confidence, Exp 27's correlation-profile error, and Frechet
   distance in a held-out ResNet-18 that was never inverted.

   **Their published loop does not fit the 24 GB card its batch size implies.** The
   log-likelihood holds a `(batch, dim, 2, 2)` float64 tensor per prefix-scan step under
   autograd: 321 MB at `layer1`'s 200 704 dimensions and their batch of 50, about 18 steps,
   roughly three such tensors alive per step. Summing the four layers' losses before the
   backward pass keeps all four graphs at once, about 32 GB — an earlier note here said
   11 GB, which was an underestimate, and the first full run was killed by it after 17
   seconds with 22.7 GB allocated (job 541487). The fix recomputes each layer's
   log-likelihood during the backward pass (`torch.utils.checkpoint`), which is exact
   because the scan is deterministic and leaves one layer's graph live at a time. That
   keeps the authors' batch size and all four layers; shrinking the batch instead would
   weaken the feature mean/variance match that *every* arm depends on, including their own
   `W_LCM = 0` ablation, and so would change the baseline rather than just the budget.

   Recomputing alone was still not enough: a *single* layer's scan graph is about 20 GB at
   their batch of 50, and a second run died at 22.9 GB inside the recompute (job 541566).
   The batch is therefore also split *inside* the likelihood. That is exact rather than an
   approximation, because their likelihood averages a per-image total over the batch and
   every other quantity in it depends only on the parameters; `exp28_check_chunk_exact.py`
   verifies it against their unmodified module, where the value agrees to 3e-14 and the
   gradient with respect to the images — the only thing the loop optimises — to 2e-16.
   Peak memory measured afterwards: **14.3 GB** on a 24 GB card. Measured speed: **12.4
   seconds per optimisation step** with all four layers, so the authors' full 3400-step
   schedule costs about 11.7 GPU-hours for each arm that uses the covariance.
2. **The quantitative claim of Tables 1–2** (+0.51 to +1.48 p.p. continual-learning
   accuracy on CIFAR-100 / Tiny-ImageNet, +0.12/+0.31 on ViT) has never been independently
   checked. Both datasets are already on the cluster (`cifar-100-python`,
   `tiny-imagenet-200`). Cost is above the 4-GPU-hour threshold, so it needs an explicit
   go-ahead.

   **Approved as the next experiment after the image-inversion result lands**, and tracing
   the authors' continual-learning entry point changed what it should test. That pipeline
   does *not* apply the covariance to the flattened feature map the way every experiment
   here has. It builds one dense covariance per `BatchNorm2d` over **channels only**
   (`LCM(m.num_features)`, so 16–512 variables, not ~200 000), accumulates the empirical
   **correlation** matrix channel-wise by a streaming Welford update over the training set,
   and fits by mean-squared error between the model's `correlation()` and that matrix —
   Adam, learning rate 0.01, 200 epochs — blending `ratio_old * R_old + ratio_new * R_curr`
   by class counts across tasks and freezing the result. It is skipped entirely when their
   `alpha_frob` weight is zero. See the table in `Laplace/laplace/LCM_PROVENANCE.md`.

   Two things follow. The comparison is cheap, because 16–512 variables is laptop-sized and
   the fitting-cost advantage is irrelevant there; the question is purely whether a
   closed-form fit matches the correlation matrix better than 200 epochs of mean-squared
   error. And the ordering problem is sharper, because channels have no spatial neighbour
   relation, so the learned coordinate is seriating channels — the case where this project
   already found order to be recoverable by seriation but at chance when learned by
   gradient descent.
3. **Conditional prediction on real data and on a network.** Exp 23 is a synthetic GLM.
   The same probe on Tecator (Exp 20's loader) and on Exp 21's CNN last layer is
   laptop-sized and is the obvious next step — it would tell us whether the `rho >= 0.9`
   threshold is ever met by real features. Exp 21 already measured neighbour |corr| 0.7152
   against a mean of 0.2529 in a learned map, so this is not a foregone conclusion.
4. **Why Frobenius loses.** `||A - S||_F^2` weights large eigenvalues and is nearly blind
   to the small ones that dominate a log-det. If the `lcm_moments` result survives seeds
   and scale, "fit the chain by moments, not Frobenius" is a one-line constructive
   improvement to the published method, and the cheapest paper in the vicinity.
5. **Streaming `Kron` accumulator** so `Laplace(..., hessian_structure="kron_lcm")` reaches
   `i >= 32k` without materialising `i^2`. Still the most valuable unwritten piece of
   implementation: Exp 19 showed the estimator itself runs at `i=65536` in 0.188 s, but the
   library entry point cannot feed it.

### Deferred

- Exps 13/14/15 still use hand-rolled posteriors rather than the library.
- Exps 10/10b/11 have not been re-run under normalised fitting (Exp 16's correction).
- `KronLCMDecomposed.bmm` supports exponents 1 and −1 only — the chain has no closed-form
  square root — so `sample()` goes through perturb-and-solve, untested at scale.
- Only Exps 23–25 take CLI arguments and write a provenance stamp; Exps 1–22 predate both
  conventions.

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
   closes **87%/84%/64%** of the diagonal→full gap at d=500/1000/1500 (Exp 8c,
   10 seeds, run through `laplace-torch`) and ~100% on Tecator NIR (Exp 11).
5. **Mechanism (explanatory).** A single 1-D Markov chain cannot embed generic
   NN-weight curvature (Exp 4a plateau at 0.66) because layer curvature is
   Kronecker, not chain. Per-layer blocking recovers only 12–22% of that (Exp 6).
6. **The ordering negatives were a fitting artifact (Exp 16) — major correction.**
   Scale-normalising the fit moves the learned coordinate from −94.5% to **+10.2%**
   with the order unchanged, and seriation from −65.5% to **+65.4%** when refined
   rather than frozen. **LCM does not require a known order.** What survives is
   that gradient descent *alone* cannot discover one (recovery 0.028).
7. **Frobenius fit is a poor proxy for predictive quality (Exp 14b).** A low-rank
   model matched LCM's precision fit (0.191 vs 0.137) yet closed only 29% of the
   joint gap against LCM's 93%, because the posterior covariance here is near-banded
   and diagonal-plus-low-rank cannot be. This caveat applies to Exp 1, 4 and 13.
8. **Annealing the nugget does not help (Exp 15b).** Both variants made order
   recovery worse. Spectral seriation is the intervention that works.
9. **LCM beats its own complexity class — in its niche only (Exp 13).** Against
   banded-Cholesky (`AAᵀ`, tridiagonal precision) and low-rank peers at equal
   budget: in the ordered-feature niche LCM needs `3d` where low-rank needs `17d`,
   and closes +87% of the joint gap where every peer closes ≈0% or worse. On real
   NN curvature the verdict reverses — `rank4` at `5d` beats LCM at `3d`, with no
   ordering fragility.
10. **The learned coordinate cannot *discover* an order (still true).** Gradient
   descent from an uninformed init recovers \|Spearman\| = 0.028 (Exp 10), and it is
   inactive when the order is already known (Exp 11: learned ≈ fixed). But per
   item 6 this no longer makes LCM unusable: a corrected fit degrades gracefully
   (+10.2%), and seriation supplies the order that gradients cannot find (+65.4%).

**Practical guidance — the recipe.** Use **LCM over diagonal** when predictions
are consumed *jointly* (batch decision-making, experimental design), the variables
have a 1-D order — **known or seriable** (spectra, time lags, genomic position) —
and `n ≳ d`. Then:

1. **Scale-normalise the fit target** (`T/s`, rescale after). Not optional: it is
   the difference between −94.5% and +10.2% (Exp 16).
2. **Initialise `a` by spectral seriation, then refine it** — do not freeze it, and
   do not expect gradient descent to find the order unaided (+65.4% vs +11.2%
   frozen vs +10.2% uninformed).
3. Skip annealing; it makes recovery worse (Exp 15b).

Prefer **KFAC/Full** for neural-network weights with exploitable layer structure —
KFAC is also cheaper to build there, so it is not a cost trade-off — and prefer
**low-rank** over LCM on NN curvature specifically (Exp 13a, 14c). Reach for LCM's
`3P` storage where the layer is strongly rectangular (Exp 12a), or where there are
no layers at all and dense is infeasible.

**Limitations of this evaluation.**

1. **Narrow niche.** The only demonstrated advantage is joint predictive UQ for
   1-D-ordered features with `n ≳ d`. The order need not be known a priori
   (Exp 16), but it must be *seriable*. The reported `d`-dependence is measured
   with the deficient fit and should be re-established (see Open directions).
2. **One real dataset.** Tecator is the only real positive; CO₂'s CI (±81%) makes
   it uninformative. More ordered-feature datasets (NIR/Raman beyond Tecator,
   genomics with LD structure, climate transects) are the highest-value next step.
3. **Seed counts vary.** The headline joint result was re-run at 10 seeds, but
   several experiments — including Exp 14–16, which carry the corrections — use
   3–5 and should not be over-read. Exp 16's headline effects are far larger than
   their CIs, but its absolute levels are not precisely pinned.
3b. **Most numbers predate the normalisation fix.** Every result produced by an
   unnormalised fit is conservative by an unknown margin; only Exp 13(b)–16 use
   the corrected fit.
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

## Open directions — status after Exp 14–16

The five directions recorded earlier have now been **run**. Outcomes against their
pre-registered predictions:

| # | direction | prediction | outcome |
|---|---|---|---|
| D1 | beating low-rank? | conditional | **confirmed, and sharpened** — LCM wins at matched `3d` at every `ρ`; low-rank catches up only with far more parameters and faster at high `ρ`. Crucially Exp 14(b): low-rank can match the *Frobenius* fit and still close only 1/3 as much predictive gap |
| D2 | hybrid LCM+low-rank | won't fix recovery; will convert −84% → ≈0% | **half right.** Recovery indeed unchanged (0.114 vs 0.170). The graceful-degradation half was **pre-empted**: Exp 16 shows a corrected fit already degrades gracefully (+10.2%), so there was no catastrophe to fix |
| D3 | very large `d` | feasible but not exclusive | **confirmed.** At `d=10⁵` LCM needs 2 MB vs 74.5 GB dense — but low-rank is equally feasible and ~70× faster. Exclusivity comes from *quality* (D1), not cost |
| D4 | nugget annealing | better than 0.028, still under threshold | **wrong, in the optimistic direction.** Recovery got *worse* (0.035–0.039 vs 0.100) in both hard and soft variants. Seriation dominates |
| D5 | re-run the headline normalised | may change the `d`-dependence | **not yet run for Exp 8c** — but Exp 16 shows the deficiency reaches much further than the headline, so this is now more urgent, not less |

### What is still open

1. **Re-run Exp 8c, 10, 10b, 11 under normalised fitting.** Exp 16 corrects Exp 10
   at 3 seeds; the full campaign has not been re-run. Every number in this document
   produced by an unnormalised fit is suspect *in the conservative direction*.
   This is now the single highest-value task.
2. **Fix `LCMCovariance.fit_dense` / `fit_frobenius` to normalise internally**, so
   the deficiency cannot recur. The change is exact, not a heuristic.
3. **Seriate-then-refine as the default recipe.** Exp 16(D) reaches 65% of the
   diagonal→full gap with a *hidden* order. This deserves a proper multi-seed study
   across `d` and `ρ`, and on the real datasets of Exp 11 — it is the most
   publishable result the project has.
4. **The hybrid, with enough seeds to resolve it** (Exp 15a, CIs currently overlap).
5. **A better ordering-accuracy metric than \|Spearman\|**, which saturates at 4
   decimal places across a 33-point swing in outcome (Exp 16b). Kendall tau-distance
   on adjacent pairs, or the achieved kernel error, would discriminate where
   Spearman cannot.

## Reproduction

```bash
python experiments/run_all.py          # all 15 experiments, writes results/*.csv
python experiments/exp8c_joint_ci.py   # the headline joint-UQ result (10 seeds)
python experiments/exp12_cost.py       # cost/complexity vs KFAC
python experiments/exp13_sameclass.py  # vs banded-Cholesky / low-rank peers
python experiments/exp14_lowrank.py    # low-rank depth + large-d feasibility
python experiments/exp16_normalisation.py  # the correction to Exp 10/10b
pytest Laplace/tests/test_lcm.py -q    # 16 tests: fast paths vs dense references
```

Results are written to `experiments/results/`. Related documents:

Full setup — architectures, datasets, optimisers, learning rates, posterior
construction and hyper-parameter protocol — is in **Experimental setup** above.

| document | contents |
|---|---|
| `../PROJECT_REPORT.md` | end-to-end narrative, **design rationale** (§6), consolidated limitations |
| `../AUDIT.md` | adversarial audit — is the port faithful, would a bigger model help? |
| `../PUBLICATION_ASSESSMENT.md` | publication verdict and venue path |
| `COMPELLING_USE.md` | the Phase-2 lead-by-lead assessment |
| `../Laplace/laplace/LCM_PROVENANCE.md` | line-by-line diff against the authors' reference implementation |
| `../LCM_REPORT.html` | synthesized report for a newcomer |
